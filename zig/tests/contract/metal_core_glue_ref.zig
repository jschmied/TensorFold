//! metal_core_glue.zig's float64 references, bounds, simulations and exact checks, by kernel.
const std = @import("std");
const contract = @import("contract");
const tf = @import("tensorfold");
const Glue = @import("metal_core_glue.zig").Glue;

const c = contract.spec;
const g = contract.gates;
const nref = contract.norms_ref;
const sumIn = contract.rows.sumIn;

fn bf(bits: u16) f64 {
    return c.fromBf16(bits);
}

const Qk = struct { x: []align(1) const u16, qn: []align(1) const u16, kn: []align(1) const u16, cos: []align(1) const f32, sin: []align(1) const f32, pos: []align(1) const u32 };

fn qkInputs(p: *const Glue) Qk {
    const s = p.shape;
    const d = s.k;
    var at: usize = 0;
    const x = std.mem.bytesAsSlice(u16, p.host[0 .. s.rows * s.n * d * 2]);
    at += x.len * 2;
    const norms = std.mem.bytesAsSlice(u16, p.host[at..][0 .. 4 * d]);
    at += 4 * d;
    const cos = std.mem.bytesAsSlice(f32, p.host[at..][0 .. 64 * d * 4]);
    at += 64 * d * 4;
    const sin = std.mem.bytesAsSlice(f32, p.host[at..][0 .. 64 * d * 4]);
    at += 64 * d * 4;
    return .{ .x = x, .qn = norms[0..d], .kn = norms[d..], .cos = cos, .sin = sin, .pos = std.mem.bytesAsSlice(u32, p.host[at..][0 .. s.rows * 4]) };
}

/// q heads, then the K cache, then the V cache: each head's outputs at `out` with its sums' magnitudes in `mag`.
fn qk(p: *const Glue, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
    const s = p.shape;
    const d = s.k;
    const hk: usize = 4;
    const h = s.n - 2 * hk;
    const in = qkInputs(p);
    var t: [256]f64 = undefined;
    var y: [256]f64 = undefined;
    for (0..s.rows) |r| for (0..s.n) |hd| {
        const src = in.x[(r * s.n + hd) * d ..][0..d];
        const slot = in.pos[r] % s.rows;
        const dst = if (hd < h) (r * h + hd) * d else if (hd < h + hk) (s.rows * h + slot * hk + hd - h) * d else (s.rows * h + s.rows * hk + slot * hk + hd - h - hk) * d;
        if (hd >= h + hk) {
            for (0..d) |i| out[dst + i] = bf(src[i]);
            if (mag) |m| @memset(m[dst..][0..d], 0);
            continue;
        }
        for (src, t[0..d]) |v, *o| o.* = bf(v) * bf(v);
        var ss: f64 = 0;
        if (acc) |a| {
            ss = sumIn(a, order, t[0..d]);
        } else for (t[0..d]) |v| {
            ss += v;
        }
        const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(d)) + 1e-6);
        const w = if (hd < h) in.qn else in.kn;
        for (0..d) |i| y[i] = bf(src[i]) * inv * bf(w[i]);
        const p0 = in.pos[r];
        for (0..d) |i| {
            const other = if (i < d / 2) -y[i + d / 2] else y[i - d / 2];
            const cs: f64 = in.cos[p0 * d + i];
            const sn: f64 = in.sin[p0 * d + i];
            out[dst + i] = y[i] * cs + other * sn;
            if (mag) |m| m[dst + i] = @abs(y[i] * cs) + @abs(other * sn);
        }
    };
}

fn addRms(p: *const Glue, acc: ?c.Accum, order: g.Order, out: []f64) !void {
    const rows = p.rows.?;
    const k = p.shape.k;
    const t = try p.a.alloc(f64, 4 * k);
    defer p.a.free(t);
    for (0..p.shape.rows) |r| {
        const x = t[0..k];
        const res = t[k..][0..k];
        const sq = t[2 * k ..][0..k];
        for (x, res, 0..) |*xv, *rv, i| {
            xv.* = bf(rows.x[r * k + i]);
            rv.* = p.aux[r * k + i];
        }
        for (sq, x) |*o, v| o.* = v * v;
        const s1 = if (acc) |a| sumIn(a, order, sq) else blk: {
            var z: f64 = 0;
            for (sq) |v| z += v;
            break :blk z;
        };
        const inv1 = 1 / @sqrt(s1 / @as(f64, @floatFromInt(k)) + 1e-6);
        for (res, x, 0..) |*rv, v, i| rv.* += v * inv1 * bf(rows.w[i]);
        for (sq, res) |*o, v| o.* = v * v;
        const s2 = if (acc) |a| sumIn(a, order, sq) else blk: {
            var z: f64 = 0;
            for (sq) |v| z += v;
            break :blk z;
        };
        const inv2 = 1 / @sqrt(s2 / @as(f64, @floatFromInt(k)) + 1e-6);
        for (0..k) |i| out[r * k + i] = res[i] * inv2 * bf(rows.w[k - 1 - i]);
    }
}

fn combine(p: *const Glue, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
    const s = p.shape;
    const slots = s.k;
    const w = p.aux[0 .. s.rows * slots];
    const y = p.aux[s.rows * slots ..];
    var t: [64]f64 = undefined;
    for (0..s.rows) |r| for (0..s.n) |d| {
        var sum: f64 = 0;
        var abs: f64 = 0;
        for (0..slots) |sl| {
            t[sl] = w[r * slots + sl] * y[(r * slots + sl) * s.n + d];
            sum += t[sl];
            abs += @abs(t[sl]);
        }
        out[r * s.n + d] = if (acc) |a| sumIn(a, order, t[0..slots]) else sum;
        if (mag) |m| m[r * s.n + d] = abs;
    };
}

pub fn reference(p: *Glue, want: []f64, mag: []f64) !void {
    switch (p.v.kind) {
        .embed, .select, .plan => {
            @memset(want, 0);
            @memset(mag, 0);
        },
        .rms => try p.rows.?.reference(want, mag),
        .add_rms => {
            try addRms(p, null, .sequential, want);
            for (mag, want) |*m, w| m.* = @abs(w);
        },
        .qk => qk(p, null, .sequential, want, mag),
        .combine => combine(p, null, .sequential, want, mag),
    }
}

/// add_rms: norms_ref's residual-step bounds (the add's fp32 rounding carried through the second norm), at least.
pub fn bounds(p: *Glue, want: []const f64, _: []const f64, allow: []f64) !void {
    if (p.v.kind != .add_rms) return;
    const k = p.shape.k;
    const rows = p.rows.?;
    const t = try p.a.alloc(f64, 7 * k);
    defer p.a.free(t);
    for (0..p.shape.rows) |r| {
        const x = t[0..k];
        const w1 = t[k..][0..k];
        const w2 = t[2 * k ..][0..k];
        const add = t[3 * k ..][0..k];
        const res = t[4 * k ..][0..k];
        for (0..k) |i| {
            x[i] = bf(rows.x[r * k + i]);
            w1[i] = bf(rows.w[i]);
            w2[i] = bf(rows.w[k - 1 - i]);
            res[i] = p.aux[r * k + i] + 0;
        }
        nref.rms(x, w1, 1e-6, add);
        for (res, add) |*v, a| v.* += a;
        nref.addRmsBounds(p.aux[r * k ..][0..k], add, w2, want[r * k ..][0..k], res, .bf16, t[5 * k ..][0..k], t[6 * k ..][0..k]);
        for (allow[r * k ..][0..k], t[6 * k ..][0..k]) |*l, y| l.* = @max(l.*, y);
    }
}

pub fn simulate(p: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
    switch (p.v.kind) {
        .rms => try p.rows.?.simulate(acc, order, out),
        .add_rms => try addRms(p, acc, order, out),
        .qk => qk(p, acc, order, out, null),
        .combine => combine(p, acc, order, out, null),
        else => return error.NoSums,
    }
}

const Pick = struct { e: u32, key: f64 };

/// The host route: choices in fp32 (logit, or sigmoid in float64, plus bias), kept groups, top-k, lower id on ties.
fn hostRoute(p: *const Glue, row: usize, picks: []u32, wts: []f64, gaps: []f64) void {
    const rt = p.route();
    const ne = rt.experts;
    const all: []align(1) const f32 = std.mem.bytesAsSlice(f32, p.host);
    const lg = all[row * ne ..][0..ne];
    const bias = all[p.shape.rows * ne ..][0..ne];
    var choice: [4096]f64 = undefined;
    for (0..ne) |e| choice[e] = if (rt.score) 1 / (1 + @exp(-@as(f64, lg[e]))) + bias[e] else @as(f32, lg[e] + bias[e]);
    var kept: u32 = ~@as(u32, 0);
    if (rt.groups > 1) {
        const gs = ne / rt.groups;
        var sums: [32]f64 = undefined;
        for (0..rt.groups) |gi| {
            var x: f64 = -std.math.inf(f64);
            var y: f64 = -std.math.inf(f64);
            for (choice[gi * gs ..][0..gs]) |v| if (v > x) {
                y = x;
                x = v;
            } else if (v > y) {
                y = v;
            };
            sums[gi] = x + y;
        }
        kept = 0;
        for (0..rt.keep) |_| {
            var best: usize = 32;
            for (0..rt.groups) |gi| if ((kept >> @intCast(gi)) & 1 == 0 and (best == 32 or sums[gi] > sums[best])) {
                best = gi;
            };
            kept |= @as(u32, 1) << @intCast(best);
        }
    }
    const gs = if (rt.groups > 1) ne / rt.groups else ne;
    var total: f64 = 0;
    for (0..rt.topk) |s| {
        var best: usize = ne;
        var second: f64 = -std.math.inf(f64);
        for (0..ne) |e| {
            if ((kept >> @intCast(e / gs)) & 1 == 0 or std.mem.indexOfScalar(u32, picks[0..s], @intCast(e)) != null) continue;
            if (best == ne or choice[e] > choice[best]) {
                if (best != ne) second = @max(second, choice[best]);
                best = e;
            } else second = @max(second, choice[e]);
        }
        picks[s] = @intCast(best);
        gaps[s] = choice[best] - second;
        wts[s] = 1 / (1 + @exp(-@as(f64, lg[best])));
        total += wts[s];
    }
    for (wts[0..rt.topk]) |*w| w.* = (if (rt.normalize) w.* / total else w.*) * rt.scale;
    picks[rt.topk] = @intCast(ne);
    wts[rt.topk] = 1;
}

pub fn check(p: *Glue, _: g.Gate) !?[]const u8 {
    const s = p.shape;
    const out = try p.rig.outputBytes();
    switch (p.v.kind) {
        .embed => {
            const ids: []align(1) const u32 = std.mem.bytesAsSlice(u32, p.host[0 .. s.rows * 4]);
            const table: []align(1) const u16 = std.mem.bytesAsSlice(u16, p.host[s.rows * 4 ..]);
            const got: []align(1) const u32 = std.mem.bytesAsSlice(u32, out);
            for (0..s.rows) |r| for (0..s.k) |i| {
                const want = @as(u32, table[ids[r] * s.k + i]) << 16;
                if (got[r * s.k + i] != want) return try std.fmt.bufPrint(&p.msg, "row {d} col {d}: 0x{x} where the widen gives 0x{x}", .{ r, i, got[r * s.k + i], want });
            };
            return null;
        },
        .select, .plan => {
            const slots = s.k + 1;
            const pairs = s.rows * slots;
            const picks: []align(1) const u32 = std.mem.bytesAsSlice(u32, out[0 .. pairs * 4]);
            const wts: []align(1) const f32 = std.mem.bytesAsSlice(f32, out[pairs * 4 ..][0 .. pairs * 4]);
            var hp: [64]u32 = undefined;
            var hw: [64]f64 = undefined;
            var gap: [64]f64 = undefined;
            for (0..s.rows) |r| {
                hostRoute(p, r, hp[0..slots], hw[0..slots], gap[0..slots]);
                for (0..slots) |sl| {
                    const got = picks[r * slots + sl];
                    const tie = p.v.score and gap[sl] < 1e-6; // sigmoid's last bits: the GPU's exp may order a near tie
                    if (got != hp[sl] and !tie) return try std.fmt.bufPrint(&p.msg, "row {d} slot {d}: expert {d} where the rule picks {d}", .{ r, sl, got, hp[sl] });
                    if (got == hp[sl] and @abs(wts[r * slots + sl] - hw[sl]) > 8 * 0x1p-24 * @abs(hw[sl]) + 0x1p-140)
                        return try std.fmt.bufPrint(&p.msg, "row {d} slot {d}: weight {e} where float64 gives {e}", .{ r, sl, wts[r * slots + sl], hw[sl] });
                }
            }
            if (p.v.kind == .plan) return planCheck(p, picks, out[2 * pairs * 4 ..]);
            return null;
        },
        else => return null,
    }
}

/// Members: pairs by expert in pair order; items: (expert, first, count <= 16) in that order; the count last.
fn planCheck(p: *Glue, picks: []align(1) const u32, rest: []const u8) !?[]const u8 {
    const pairs = picks.len;
    const groups = p.shape.n + 1;
    const members: []align(1) const u32 = std.mem.bytesAsSlice(u32, rest[0 .. pairs * 4]);
    const most = tf.block_rows.maxItems(pairs, groups);
    const items: []align(1) const u32 = std.mem.bytesAsSlice(u32, rest[pairs * 4 ..][0 .. most * 12]);
    const count = std.mem.readInt(u32, rest[pairs * 4 + most * 12 ..][0..4], .little);
    var at: usize = 0;
    var n: usize = 0;
    for (0..groups) |e| {
        const first = at;
        for (picks, 0..) |x, q| if (x == e) {
            if (members[at] != q) return try std.fmt.bufPrint(&p.msg, "member {d} is pair {d} where expert {d}'s next is {d}", .{ at, members[at], e, q });
            at += 1;
        };
        var i = first;
        while (i < at) : (i += 16) {
            const want = [3]u32{ @intCast(e), @intCast(i), @intCast(@min(16, at - i)) };
            if (!std.mem.eql(u32, &want, &[3]u32{ items[3 * n], items[3 * n + 1], items[3 * n + 2] })) return try std.fmt.bufPrint(&p.msg, "item {d} is ({d}, {d}, {d}) where the rule makes ({d}, {d}, {d})", .{ n, items[3 * n], items[3 * n + 1], items[3 * n + 2], want[0], want[1], want[2] });
            n += 1;
        }
    }
    if (count != n) return try std.fmt.bufPrint(&p.msg, "count {d} where the rule makes {d}", .{ count, n });
    return null;
}
