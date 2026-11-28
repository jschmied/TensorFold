//! tf-kolibri-metal selftest: Kolibri-1's Metal forward on the seeded checkpoint against the float64 reference.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const k1 = @import("kolibri1_metal");

const fwd = k1.forward;

const usage =
    \\usage: tf-kolibri-metal selftest [DIR] [--plant BLOCK:N] [--shape parity|wide]
    \\  DIR    holds the seeded checkpoint (written when DIR has no config.json; default kolibri1-selftest)
    \\  plant  a planted fault: glue:1-3, rows:1-3 or moe:1-3 in the kernels; on the host, swap:1-3 (one swap
    \\         past the tie margin; every near tie, past the cap; a far token winning a near-tie row) or route:1
    \\         (a near-tie flip, then a far-off expert); the run must fail its gate (exit 1)
    \\Exit 0: every gate passed; 1: a gate failed; 2: usage or setup.
    \\
;

const tokens_n = 640; // past one 576-key ring of 16-row calls, so a sliding layer wraps; rows 511..513 see 512..514
const prefix = 320;
const gates = struct {
    const cosine = 0.9999; // every row's logits and each tap row against float64
    const rel = 0.02; // a tap row's relative L2 error
    const tie = 2e-2; // a top-1 swap passes only where float64's own top-2 margin is below this (~3 sigma)
    const tie_cap = 4; // and at most one in this many of those near rows swaps
    const alone_margin = 1e-3; // MoE alone: a pick may differ only where the host's rank keys are this close
    const flip_margin = 2e-2; // the chain: a route may differ only where the reference's own keys are this close
};

const Stats = struct { worst_cos: f64 = 1, worst_row: usize = 0, worst_rel: f64 = 0, swaps: usize = 0, ties: usize = 0, tie_gap: f64 = 0, near: usize = 0, rows: usize = 0, ok: bool = true };

fn compare(st: *Stats, got: []const f32, want: []const f64, width: usize, logits: bool) void {
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var err: f64 = 0;
        for (g, w) |a, b| {
            dot += a * b;
            gg += @as(f64, a) * a;
            ww += b * b;
            err += (a - b) * (a - b);
        }
        const cos = dot / @sqrt(gg * ww);
        const rel = @sqrt(err / ww);
        if (cos < st.worst_cos) st.worst_row = r;
        st.worst_cos = @min(st.worst_cos, cos);
        st.worst_rel = @max(st.worst_rel, rel);
        st.rows += 1;
        if (!(cos >= gates.cosine)) st.ok = false;
        if (!logits and !(rel <= gates.rel)) st.ok = false;
        if (!logits) continue;
        const top = std.mem.indexOfMax(f64, w);
        const mine = std.mem.indexOfMax(f32, g);
        var second: f64 = -std.math.inf(f64);
        for (w, 0..) |v, i| if (i != top) {
            second = @max(second, v);
        };
        if (w[top] - second < gates.tie) st.near += 1; // rows the tie margin could excuse at all
        if (top == mine) continue;
        const margin = w[top] - w[mine]; // float64's top over the device's winner, whichever token that is
        if (margin < gates.tie) {
            std.debug.print("  excused top-1 swap, row {d}: float64 margin {e:.3} over the device's winner\n", .{ r, margin });
            st.ties += 1;
            st.tie_gap = @max(st.tie_gap, margin);
        } else {
            st.swaps += 1;
            st.ok = false;
        }
    }
    if (logits and st.ties * gates.tie_cap > st.near) st.ok = false; // more excused swaps than the cap allows
}

fn report(name: []const u8, st: Stats) void {
    std.debug.print("{s} {s}: {d} rows, worst cosine {d:.7}, worst rel {e:.3}, top-1 swaps {d} (ties {d}, widest {e:.2}; {d} rows have a top-2 margin under {e:.0}, at most 1 in {d} may swap)\n", .{ if (st.ok) "PASS" else "FAIL", name, st.rows, st.worst_cos, st.worst_rel, st.swaps, st.ties, st.tie_gap, st.near, gates.tie, gates.tie_cap });
}

fn sameBytes(name: []const u8, a: []const f32, b: []const f32) bool {
    const ok = std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    std.debug.print("{s} {s}: {d} values byte for byte\n", .{ if (ok) "PASS" else "FAIL", name, a.len });
    return ok;
}

/// `ids` through `m` in calls of `chunk` rows (the last ones one row at a time from `singles` on): every row's logits.
fn run(a: std.mem.Allocator, m: *fwd.Model, ids: []const u32, chunk: usize, singles: usize, res: ?mtl.Buffer, picks: ?mtl.Buffer) ![]f32 {
    const v = m.c.vocab;
    const out = try a.alloc(f32, ids.len * v);
    var at: usize = 0;
    while (at < ids.len) {
        const n = if (at >= singles) 1 else @min(chunk, singles - at);
        const t: fwd.Taps = .{ .buf = if (at == 0 and n == ids.len) res else null, .picks = if (picks) |b| .{ .buf = b, .rows = ids.len, .at = at } else null };
        const lg = try m.forward(ids[at..][0..n], true, t);
        @memcpy(out[at * v ..][0 .. n * v], lg);
        at += n;
    }
    return out;
}

/// The reference on the device's routes; at each layer its own choice (same history) against the device's.
fn forced(a: std.mem.Allocator, map: *const k1.names.Map, c: k1.config.Config, ids: []const u32, got: []const u32, name: []const u8, ok: *bool, route_plant: u8) !k1.ref.Out {
    const slots = c.top_k + 1;
    const n = ids.len;
    const own = try a.alloc([]u32, c.layers);
    const keys = try a.alloc([]f64, c.layers);
    const force = try a.alloc([]const u32, c.layers);
    for (own, keys, force, 0..) |*o, *k, *f, l| {
        o.* = try a.alloc(u32, n * slots);
        k.* = try a.alloc(f64, n * c.experts);
        f.* = got[l * n * slots ..][0 .. n * slots];
    }
    for (force, 0..) |f, l| for (0..n) |r| if (!valid(c, f[r * slots ..][0..slots])) {
        std.debug.print("FAIL {s} layer {d} row {d}: device route {any} is not a route (ids, repeats or shared slot)\n", .{ name, l, r, f[r * slots ..][0..slots] });
        ok.* = false;
        return error.InvalidRoute;
    };
    const out = try k1.ref.runWith(a, map, c, ids, .{ .picks = own, .keys = keys, .force = force });
    for (0..c.layers) |l| {
        const dev = if (route_plant != 0) try badRoute(a, c, force[l], keys[l]) else force[l]; // route:1 checks the gate itself
        ok.* = routes(try std.fmt.allocPrint(a, "{s} layer {d} routes", .{ name, l }), c, own[l], keys[l], dev, gates.flip_margin) and ok.*;
    }
    return out;
}

/// A device route's every routed id below E, unique, the shared slot last; false: nothing may be forced from it.
fn valid(c: k1.config.Config, got: []const u32) bool {
    const k = c.top_k;
    for (got[0..k], 0..) |e, i| if (e >= c.experts or std.mem.indexOfScalar(u32, got[0..i], e) != null) return false;
    return got[k] == c.experts;
}

/// A route's worst key gap: each added expert against each dropped, each out-of-order pair (null: equal; 0: bad tie).
fn rowGap(keys: []const f64, own: []const u32, got: []const u32, k: usize) ?f64 {
    if (std.mem.eql(u32, own[0..k], got[0..k])) return null;
    var worst: f64 = 0;
    for (got[0..k]) |x| if (std.mem.indexOfScalar(u32, own[0..k], x) == null) for (own[0..k]) |d| {
        if (std.mem.indexOfScalar(u32, got[0..k], d) != null) continue;
        const g = keys[d] - keys[x]; // the reference ranked d over x
        if (g <= 0) return 0;
        worst = @max(worst, g);
    };
    for (0..k) |i| for (i + 1..k) |j| {
        const g = keys[got[j]] - keys[got[i]]; // > 0: j ranked below i, out of key order
        if (g == 0 and got[i] > got[j]) return 0;
        worst = @max(worst, g);
    };
    return worst;
}

/// Device routes against the reference's own, layer by layer: how many rows differ, and whether each is a near tie.
fn routes(name: []const u8, c: k1.config.Config, own: []const u32, keys: []const f64, got: []const u32, bound: f64) bool {
    const slots = c.top_k + 1;
    const n = own.len / slots;
    var differ: usize = 0;
    var worst: f64 = 0;
    var bad: usize = 0;
    for (0..n) |r| {
        const row = got[r * slots ..][0..slots];
        const g = if (!valid(c, row)) std.math.inf(f64) else rowGap(keys[r * c.experts ..][0..c.experts], own[r * slots ..][0..slots], row, c.top_k) orelse continue;
        differ += 1;
        worst = @max(worst, g);
        if (!(g > 0 and g < bound)) {
            bad += 1;
            if (bad <= 3) std.debug.print("  row {d}: device {any}, reference {any}, key gap {e:.3}\n", .{ r, got[r * slots ..][0..slots], own[r * slots ..][0..slots], g });
        }
    }
    std.debug.print("{s} {s}: {d} of {d} rows route differently, largest key gap {e:.3} (bound {e:.0}, ties broken wrong or past it: {d})\n", .{ if (bad == 0) "PASS" else "FAIL", name, differ, n, worst, bound, bad });
    return bad == 0;
}

/// Each layer's MoE half on the host from Metal's attention residual, and its picks against Metal's.
fn local(a: std.mem.Allocator, map: *const k1.names.Map, c: k1.config.Config, tapped: []const f32, picks: []const u32) !bool {
    const d = c.hidden;
    const slots = c.top_k + 1;
    var ok = true;
    for (0..c.layers) |l| {
        const res = try a.alloc(f64, tokens_n * d);
        for (res, tapped[l * 2 * tokens_n * d ..][0 .. tokens_n * d]) |*o, v| o.* = v;
        const hp = try a.alloc(u32, tokens_n * slots);
        const keys = try a.alloc(f64, tokens_n * c.experts);
        try k1.ref.moeHalf(a, map, c, l, res, hp, keys);
        var st: Stats = .{};
        compare(&st, tapped[(l * 2 + 1) * tokens_n * d ..][0 .. tokens_n * d], res, d, false);
        report(try std.fmt.allocPrint(a, "A layer {d} MoE alone (from Metal's attention residual)", .{l}), st);
        ok = ok and st.ok;
        ok = routes(try std.fmt.allocPrint(a, "A layer {d} MoE alone, picks", .{l}), c, hp, keys, picks[l * tokens_n * slots ..][0 .. tokens_n * slots], gates.alone_margin) and ok;
    }
    return ok;
}

const Plant = struct { ctl: fwd.Ctl = .{}, swap: u8 = 0, route: u8 = 0 };

fn plant(text: []const u8) !Plant {
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.Usage;
    const n = try std.fmt.parseInt(u8, text[colon + 1 ..], 10);
    if (n == 0 or n > 3) return error.Usage;
    const block = text[0..colon];
    if (std.mem.eql(u8, block, "glue")) return .{ .ctl = .{ .glue = n } };
    if (std.mem.eql(u8, block, "rows")) return .{ .ctl = .{ .rows = n } };
    if (std.mem.eql(u8, block, "moe")) return .{ .ctl = .{ .moe = n } };
    if (std.mem.eql(u8, block, "swap")) return .{ .swap = n };
    if (std.mem.eql(u8, block, "route") and n == 1) return .{ .route = n };
    return error.Usage;
}

/// swap:3: on the first row whose float64 top two are a near tie, the device's winner becomes the lowest token.
fn farWinner(out: []f32, want: []const f64, v: usize) []const f32 {
    for (0..want.len / v) |r| {
        const w = want[r * v ..][0..v];
        const top = std.mem.indexOfMax(f64, w);
        var second: f64 = -std.math.inf(f64);
        for (w, 0..) |x, i| if (i != top) {
            second = @max(second, x);
        };
        if (w[top] - second >= gates.tie) continue;
        const far = std.mem.indexOfMin(f64, w);
        const o = out[r * v ..][0..v];
        o[far] = o[std.mem.indexOfMax(f32, o)] + 1e-3;
        std.debug.print("  planted: row {d} (top-2 margin {e:.3}) won by token {d}, {e:.3} below the top\n", .{ r, w[top] - second, far, w[top] - w[far] });
        return out;
    }
    return out;
}

/// route:1 on a copy of device routes: a near-tie adjacent pair swapped (acceptable), then the last pick far off.
fn badRoute(a: std.mem.Allocator, c: k1.config.Config, got: []const u32, keys: []const f64) ![]u32 {
    const out = try a.dupe(u32, got);
    const slots = c.top_k + 1;
    const k = c.top_k;
    for (0..got.len / slots) |r| {
        const row = out[r * slots ..][0..slots];
        const kr = keys[r * c.experts ..][0..c.experts];
        for (0..k - 2) |s| {
            const g = kr[row[s]] - kr[row[s + 1]];
            if (!(g > 0 and g < gates.flip_margin)) continue;
            std.mem.swap(u32, &row[s], &row[s + 1]);
            var low: ?u32 = null;
            for (kr, 0..) |x, e| if (std.mem.indexOfScalar(u32, row[0..k], @intCast(e)) == null and (low == null or x < kr[low.?])) {
                low = @intCast(e);
            };
            std.debug.print("  planted: row {d}: slots {d}, {d} swapped (gap {e:.3}), slot {d} now expert {d}\n", .{ r, s, s + 1, g, k - 1, low });
            row[k - 1] = low.?;
            return out;
        }
    }
    return out;
}

/// Gate controls on a copy of the logits: 1 swaps the nearest row past the tie bound, 2 every near tie, 3 far.
fn swapped(a: std.mem.Allocator, got: []const f32, want: []const f64, v: usize, mode: u8) ![]const f32 {
    if (mode == 0) return got;
    const out = try a.dupe(f32, got);
    if (mode == 3) return farWinner(out, want, v);
    var pick: ?usize = null;
    var pick_gap: f64 = std.math.inf(f64);
    for (0..want.len / v) |r| {
        const w = want[r * v ..][0..v];
        const top = std.mem.indexOfMax(f64, w);
        var second: usize = if (top == 0) 1 else 0;
        for (w, 0..) |x, i| if (i != top and x > w[second]) {
            second = i;
        };
        const margin = w[top] - w[second];
        const lift = mode == 2 and margin < gates.tie;
        if (lift) out[r * v + second] = out[r * v + top] + @abs(out[r * v + top]) * 1e-6 + 1e-6;
        if (mode == 1 and margin >= gates.tie and margin < pick_gap) {
            pick = r;
            pick_gap = margin;
        }
    }
    if (pick) |r| {
        const w = want[r * v ..][0..v];
        const top = std.mem.indexOfMax(f64, w);
        var second: usize = if (top == 0) 1 else 0;
        for (w, 0..) |x, i| if (i != top and x > w[second]) {
            second = i;
        };
        out[r * v + second] = out[r * v + top] + @as(f32, @floatCast(1e-3));
        std.debug.print("  planted: row {d}'s runner-up lifted past its top, float64 margin {e:.3}\n", .{ r, pick_gap });
    }
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or !std.mem.eql(u8, args[1], "selftest")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var dir_path: []const u8 = "kolibri1-selftest";
    var planted: Plant = .{};
    var shape = k1.fixture.parity;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--plant") and i + 1 < args.len) {
            i += 1;
            planted = plant(args[i]) catch {
                std.debug.print("{s}", .{usage});
                return 2;
            };
        } else if (std.mem.eql(u8, args[i], "--shape") and i + 1 < args.len) {
            i += 1;
            shape = if (std.mem.eql(u8, args[i], "wide")) k1.fixture.wide else k1.fixture.parity;
        } else dir_path = args[i];
    }
    const ctl = planted.ctl;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    if (dir.access(io, "config.json", .{})) |_| {} else |_| {
        std.debug.print("writing the seeded checkpoint to {s}\n", .{dir_path});
        try k1.fixture.write(gpa, io, dir, shape);
    }
    const device = try mtl.Device.init();
    defer device.deinit();
    var w = try k1.weights.load(gpa, io, device, dir_path);
    defer w.deinit();
    const c = w.config;
    var prng = std.Random.DefaultPrng.init(0x4b31);
    const ids = try a.alloc(u32, tokens_n);
    for (ids) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    const other = try a.dupe(u32, ids);
    for (other[prefix..]) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));

    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir_path);
    defer ck.close();
    var map = try k1.names.map(gpa, &ck, c);
    defer map.deinit();
    std.debug.print("float64 reference over {d} rows\n", .{ids.len});
    const slots = c.top_k + 1;

    var ok = true;
    // A: the whole prompt in one call, with taps; twice, byte for byte; gates against the reference on its routes
    var big = try fwd.Model.init(gpa, device, &w, tokens_n, tokens_n, ctl);
    defer big.deinit();
    const taps_buf = try device.buffer(c.layers * 2 * tokens_n * c.hidden * 4, mtl.ResourceOptions.shared);
    defer taps_buf.deinit();
    const picks_buf = try device.buffer(c.layers * tokens_n * slots * 4, mtl.ResourceOptions.shared);
    defer picks_buf.deinit();
    const t0 = std.Io.Clock.awake.now(io);
    const one = try run(a, &big, ids, tokens_n, tokens_n, taps_buf, picks_buf);
    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    std.debug.print("info A forward: {d} rows in one call, {d:.1} ms wall (first call, with taps)\n", .{ tokens_n, ms });
    const got = try a.dupe(u32, picks_buf.slice(u32, c.layers * tokens_n * slots));
    const want = try forced(a, &map, c, ids, got, "A", &ok, planted.route);
    var st: Stats = .{};
    compare(&st, try swapped(a, one, want.logits, c.vocab, planted.swap), want.logits, c.vocab, true);
    report("A one call: logits", st);
    ok = ok and st.ok;
    const tapped = taps_buf.slice(f32, c.layers * 2 * tokens_n * c.hidden);
    for (0..c.layers) |l| for (0..2) |h| {
        var ts: Stats = .{};
        compare(&ts, tapped[(l * 2 + h) * tokens_n * c.hidden ..][0 .. tokens_n * c.hidden], if (h == 0) want.attn[l] else want.moe[l], c.hidden, false);
        report(try std.fmt.allocPrint(a, "A layer {d} {s} residual", .{ l, if (h == 0) "attention" else "MoE" }), ts);
        ok = ok and ts.ok;
    };
    ok = try local(a, &map, c, tapped, got) and ok;
    try big.rewind(0);
    ok = sameBytes("A again", one, try run(a, &big, ids, tokens_n, tokens_n, null, null)) and ok;

    // B: 16-row calls, then one row at a time past the ring's wrap, against the reference on B's own routes
    var small = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer small.deinit();
    const chunked = try run(a, &small, ids, 16, 592, null, picks_buf);
    const want_b = try forced(a, &map, c, ids, picks_buf.slice(u32, c.layers * tokens_n * slots), "B", &ok, planted.route);
    st = .{};
    compare(&st, try swapped(a, chunked, want_b.logits, c.vocab, planted.swap), want_b.logits, c.vocab, true);
    report(try std.fmt.allocPrint(a, "B 16-row calls, 1-row from 592 (ring {d})", .{small.ring}), st);
    ok = ok and st.ok;

    // C: 64 rows past a shared prefix, back to the prefix: the same rows again, then another suffix, == fresh
    var fork = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer fork.deinit();
    const first = try run(a, &fork, ids[0 .. prefix + 64], 16, prefix + 64, null, null);
    try fork.rewind(prefix);
    ok = sameBytes("C same suffix resumed == first", try run(a, &fork, ids[prefix..][0..64], 16, 64, null, null), first[prefix * c.vocab ..]) and ok;
    try fork.rewind(prefix);
    const forked = try run(a, &fork, other[prefix..], 16, tokens_n - prefix, null, null);
    var fresh = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer fresh.deinit();
    const other_fresh = try run(a, &fresh, other, 16, tokens_n, null, null);
    ok = sameBytes("C other suffix resumed == fresh", forked, other_fresh[prefix * c.vocab ..]) and ok;
    const refused = if (fork.rewind(prefix)) |_| false else |e| e == error.RingOverwritten;
    std.debug.print("{s} C a resume whose window the ring overwrote is refused\n", .{if (refused) "PASS" else "FAIL"});
    ok = ok and refused;
    // consecutive rewinds: 640 -> 576 keeps keys 64..575; on to 512 needs keys 0..63, which 576..639 overwrote
    const again = if (fork.rewind(576)) |_| (if (fork.rewind(512)) |_| false else |e| e == error.RingOverwritten) else |_| false;
    std.debug.print("{s} C 640 -> 576 resumes, then 576 -> 512 is refused\n", .{if (again) "PASS" else "FAIL"});
    ok = ok and again;
    // the start of a context, before any wrap: back to 1 keeps key 0, and resumes as a fresh run that stopped at 1
    var start = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer start.deinit();
    _ = try run(a, &start, ids[0..100], 16, 100, null, null);
    try start.rewind(1);
    const resumed = try a.dupe(f32, try start.forward(ids[1..17], true, null));
    var stop = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer stop.deinit();
    _ = try stop.forward(ids[0..1], true, null);
    ok = sameBytes("C 100 -> 1 before a wrap resumes == a run stopped at 1", resumed, try stop.forward(ids[1..17], true, null)) and ok;

    // D: a late failure gives back everything: a buffer error injected at the first, a middle and the last buffer
    const base = fwd.live;
    for ([_]usize{ 1, 2 * c.layers + 1, 2 * c.layers + 23 }) |at| {
        var plain = ctl;
        plain.fail_after = at;
        const refused_init = if (fwd.Model.init(gpa, device, &w, tokens_n, 16, plain)) |m| blk: {
            var mm = m;
            mm.deinit();
            break :blk false;
        } else |e| e == error.Injected;
        const back = fwd.live == base;
        std.debug.print("{s} D init failing at buffer {d}: refused {}, model buffers alive {d} (before {d})\n", .{ if (refused_init and back) "PASS" else "FAIL", at, refused_init, fwd.live, base });
        ok = ok and refused_init and back;
    }
    // and a checkpoint whose expert spans two shards is refused by name after earlier layers made their tables
    const split_path = try std.fmt.allocPrint(a, "{s}-split", .{dir_path});
    try std.Io.Dir.cwd().createDirPath(io, split_path);
    var split_dir = try std.Io.Dir.cwd().openDir(io, split_path, .{});
    defer split_dir.close(io);
    try k1.fixture.write(gpa, io, split_dir, k1.fixture.split);
    const wbase = k1.weights.live;
    const split_refused = if (k1.weights.load(gpa, io, device, split_path)) |ws| blk: {
        var x = ws;
        x.deinit();
        break :blk false;
    } else |e| e == error.ExpertsAcrossShards;
    std.debug.print("{s} D a split expert is refused (ExpertsAcrossShards {}), weight buffers alive {d} (before {d})\n", .{ if (split_refused and k1.weights.live == wbase) "PASS" else "FAIL", split_refused, k1.weights.live, wbase });
    ok = ok and split_refused and k1.weights.live == wbase;

    std.debug.print("{s} kolibri1 metal selftest{s}\n", .{ if (ok) "PASS" else "FAIL", if (ctl.glue + ctl.rows + ctl.moe + planted.swap + planted.route != 0) " (planted)" else "" });
    return if (ok) 0 else 1;
}
