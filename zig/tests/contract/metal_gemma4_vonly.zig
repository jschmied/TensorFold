//! Gemma 4 31B's V-only kernels (TF_GEMMA4_V_ONLY), each against its K+V twin's bits: decode, key rebuild, prep.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const cfg = tf.gemma4.config;
const vonly = tf.gemma4.vonly;

const geo = cfg.dense31b;
const dh = cfg.global_dim;
const nq = geo.query_heads;
const nk = geo.global_kv_heads;
const width = nq * dh + nk * dh; // a q|k row; global values are the keys' raw values
const cd = 256; // gemma_decode_attention's keys a threadgroup
const decode_rows = tf.gemma4.state.max_rows;

/// gemma_decode_attention_v (+ merge); gemma_key_rebuild; gemma_qkv_prep_v (values alone into the cache).
const Kind = enum { decode, rebuild, prep };
const Variant = struct { kind: Kind, control: bool = false };
const variants = [_]Variant{ .{ .kind = .decode }, .{ .kind = .rebuild }, .{ .kind = .prep }, .{ .kind = .decode, .control = true }, .{ .kind = .rebuild, .control = true }, .{ .kind = .prep, .control = true } };

fn nameOf(comptime v: Variant) []const u8 {
    const base = switch (v.kind) {
        .decode => "metal gemma4 31B V-only decode attention global +merge",
        .rebuild => "metal gemma4 31B V-only key rebuild",
        .prep => "metal gemma4 31B V-only qkv prep",
    };
    if (!v.control) return base;
    return "control: " ++ base ++ switch (v.kind) {
        .decode, .rebuild => ", one key product left unrounded",
        .prep => ", the keys written into the cache",
    };
}

const names = blk: {
    var out: [variants.len][]const u8 = undefined;
    for (variants, &out) |v, *n| n.* = nameOf(v);
    break :blk out;
};

const decode_served = [_]c.Shape{ .{ .rows = 1, .n = dh, .k = 0 }, .{ .rows = 1, .n = dh, .k = 3000 }, .{ .rows = 1, .n = dh, .k = 2047 }, .{ .rows = 5, .n = dh, .k = 255 }, .{ .rows = decode_rows, .n = dh, .k = 700 } };
/// Rows at positions k..: the first, a page's edge, 2^16, a 64K context's angles, 262,100 (cos's slow path).
const key_served = [_]c.Shape{ .{ .rows = 64, .n = dh, .k = 0 }, .{ .rows = 64, .n = dh, .k = 65530 }, .{ .rows = 64, .n = dh, .k = 262100 }, .{ .rows = 512, .n = dh, .k = 70000 }, .{ .rows = 9, .n = dh, .k = 250 } };

fn depth(s: c.Shape) usize {
    return s.k + s.rows + dh;
}

/// Gate 5's texts for a record named "gemma4 31B vonly global <function>": the prompt library or the glue's.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !struct { here: []const u8, pre_m5: []const u8 } {
    if (std.mem.endsWith(u8, name, " gemma_qkv_prep_v")) {
        const glue = try tf.gemma4.kernels.glueSource(a, geo, true, false, 4);
        const text = try std.mem.concat(a, u8, &.{ glue, sources.gemma4_vonly_glue });
        return .{ .here = text, .pre_m5 = text };
    }
    const h = try std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define SL {d}\n#define WINDOW 0\n#define RING 0\n", .{ dh, nq, nq / nk, dh / 128 });
    const body = try std.mem.replaceOwned(u8, a, sources.gemma4_prompt_attention, "#include \"../nax.h\"", sources.nax);
    const pre = try std.mem.concat(a, u8, &.{ h, "#define TF_SIMD_FRAGS 1\n", body, sources.gemma4_vonly_prompt });
    return .{ .here = if (device.tensorUnits()) try std.mem.concat(a, u8, &.{ h, body, sources.gemma4_vonly_prompt }) else pre, .pre_m5 = pre };
}

const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    /// The tested kernel; the twins: K+V decode, the clean rebuild, gemma_qkv_prep; the merge.
    pipe: mtl.Pipeline,
    twin: mtl.Pipeline,
    rebuild: mtl.Pipeline,
    prep: mtl.Pipeline,
    merge: mtl.Pipeline,
    rows: usize = 0,
    p0: usize = 0,
    cap: usize = 0,
    stress: bool = false,
    vp: []u16 = &.{}, // decode: values by position [kv heads, cap, dh]
    in: [6]rig_.Ref = undefined,
    work: [6]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Probe {
        const clean = try vonly.library(a, device, geo, false);
        defer clean.deinit();
        const tested = if (v.control and v.kind != .prep) try vonly.library(a, device, geo, true) else try vonly.library(a, device, geo, false);
        defer tested.deinit();
        const glue = try tf.gemma4.kernels.glueSource(a, geo, true, false, 4);
        defer a.free(glue);
        const vglue = if (v.control and v.kind == .prep) blk: {
            const from = "if (cm.y && kind == 2) cache";
            if (std.mem.count(u8, sources.gemma4_vonly_glue, from) != 1) return error.ControlEditMissing;
            break :blk try std.mem.replaceOwned(u8, a, sources.gemma4_vonly_glue, from, "if (cm.y && kind >= 1) cache");
        } else try a.dupe(u8, sources.gemma4_vonly_glue);
        defer a.free(vglue);
        const gtext = try std.mem.concat(a, u8, &.{ glue, vglue });
        defer a.free(gtext);
        const fun = switch (v.kind) {
            .decode => "gemma_decode_attention_v",
            .rebuild => "gemma_key_rebuild",
            .prep => "gemma_qkv_prep_v",
        };
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .v = v,
            .name = name,
            .pipe = if (v.kind == .prep) try rig_.pipeline(device, gtext, fun) else try mtl.Pipeline.init(device, tested, fun, false),
            .twin = try mtl.Pipeline.init(device, clean, "gemma_decode_attention", false),
            .rebuild = try mtl.Pipeline.init(device, clean, "gemma_key_rebuild", false),
            .prep = try rig_.pipeline(device, glue, "gemma_qkv_prep"),
            .merge = try rig_.pipeline(device, glue, "gemma_attention_merge"),
        };
    }

    fn deinit(p: *Probe) void {
        p.a.free(p.vp);
        p.rig.deinit();
        for ([_]mtl.Pipeline{ p.pipe, p.twin, p.rebuild, p.prep, p.merge }) |x| x.deinit();
    }

    fn spec(p: *const Probe) c.Spec {
        return switch (p.v.kind) {
            .decode => .{ .name = p.name, .accum = .f32, .depth = depth, .ops = 4, .out = .bf16, .replaces = "gemma_decode_attention on gemma_key_rebuild's keys", .tiles = .{ .{ .size = 2 }, .{}, .{ .size = cd, .min = 0 } }, .served = &decode_served },
            .rebuild => .{ .name = p.name, .accum = .none, .depth = depth, .out = .bf16, .replaces = "gemma_qkv_prep's keys", .tiles = .{ .{ .size = 1 }, .{}, .{} }, .served = &key_served },
            .prep => .{ .name = p.name, .accum = .none, .depth = depth, .out = .bf16, .replaces = "gemma_qkv_prep's caches, keys left as they were", .tiles = .{ .{ .size = 1 }, .{}, .{} }, .served = key_served[0..2] },
        };
    }

    pub fn count(p: *Probe) usize {
        return switch (p.v.kind) {
            .decode => p.rows * nq * dh,
            .rebuild => nk * p.rows * dh,
            .prep => 2 * nk * p.cap * dh, // the key cache (left poisoned), then the value cache
        };
    }

    fn chunks(p: *const Probe) usize {
        return (p.p0 + p.rows + cd - 1) / cd;
    }

    /// KW and INVF: the 31B's global frequencies; keys' norm weights in [0.25, 1.75), or 0 (gate 4: every key 0).
    fn fixed(p: *Probe, rng: std.Random) !void {
        var kw: [dh]u16 = undefined;
        var qw: [dh]u16 = undefined;
        for (&kw, &qw) |*k, *q| {
            k.* = if (p.stress) 0 else c.toBf16(0.25 + 1.5 * rng.float(f32));
            q.* = c.toBf16(0.25 + 1.5 * rng.float(f32));
        }
        var inv: [dh / 2]f32 = undefined;
        cfg.inverseFrequencies(true, &inv);
        p.in[3] = try p.rig.input(std.mem.sliceAsBytes(&kw));
        p.in[4] = try p.rig.input(std.mem.sliceAsBytes(&inv));
        p.in[5] = try p.rig.input(std.mem.sliceAsBytes(&qw));
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const most: usize = if (p.v.kind == .decode) decode_rows else 512;
        if (shape.rows == 0 or shape.rows > most or shape.n != dh) return false;
        p.rig.reset();
        p.a.free(p.vp);
        p.vp = &.{};
        p.rows = shape.rows;
        p.p0 = shape.k;
        p.stress = fill == .stress;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try p.fixed(rng);
        if (p.v.kind == .decode) return p.decodeInputs(fill, rng);
        // raw q|k rows at positions p0..; rows 0..8 of the keys carry edge values (and in gate 1, NaN, +-inf)
        const raw = try p.a.alloc(u16, p.rows * width);
        defer p.a.free(raw);
        for (raw) |*x| x.* = c.toBf16(@floatCast(rng.floatNorm(f64) * 1.5));
        const edges = [_]struct { u16, usize, u16 }{ .{ 3, 5, 0x0001 }, .{ 3, 300, 0x8001 }, .{ 4, 1, 0x000b }, .{ 8, 64, 0x0080 }, .{ 2, 0, 0x8000 } };
        for (0..nk) |h| {
            if (p.rows > 1) @memset(raw[1 * width + nq * dh + h * dh ..][0..dh], 0x8000);
            for (edges) |e| if (e[0] < p.rows) {
                raw[e[0] * width + nq * dh + h * dh + e[1]] = e[2];
            };
            if (fill == .nonfinite and p.rows > 7) for ([_][2]u16{ .{ 5, 0x7fc0 }, .{ 6, 0x7f80 }, .{ 7, 0xff80 } }) |e| {
                raw[e[0] * width + nq * dh + h * dh + 9] = e[1];
            };
        }
        const pos = try p.a.alloc(i32, p.rows);
        defer p.a.free(pos);
        for (pos, 0..) |*x, j| x.* = @intCast(p.p0 + j);
        p.cap = p.rows;
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(raw));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(pos));
        for (p.work[0..3], [_]usize{ p.rows * nq * dh * 2, nk * p.rows * dh * 2, nk * p.rows * dh * 2 }) |*w, n| w.* = try p.rig.scratch(n);
        if (p.v.kind == .prep) {
            const half = nk * p.cap * dh * 2;
            p.work[3] = try p.rig.output(half); // the key cache, compared too: a V-only prep leaves its poison
            p.out = try p.rig.output(half);
            p.work[4] = try p.rig.scratch(2 * half); // the twin's bits: poison, then gemma_qkv_prep's value cache
            p.work[5] = try p.rig.scratch(half);
            return true;
        }
        // the rebuild reads gemma_qkv_prep's values: made here, copied in as its input
        try p.prepRun(p.prep, p.work[0], p.work[0], 0); // no cache slots: the caches stay unbound
        p.in[2] = try p.rig.input(p.rig.bytes(p.work[2], nk * p.rows * dh * 2));
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// Decode inputs: values random in [-1, 1) (gate 1: NaN and +-inf among them), zeros past the last key.
    fn decodeInputs(p: *Probe, fill: g.Fill, rng: std.Random) !bool {
        const n = p.p0 + p.rows;
        p.cap = std.mem.alignForward(usize, n, cd);
        p.vp = try p.a.alloc(u16, nk * p.cap * dh);
        @memset(p.vp, 0);
        for (0..nk) |h| for (0..n) |j| for (p.vp[(h * p.cap + j) * dh ..][0..dh]) |*x| {
            x.* = c.toBf16(rng.float(f32) * 2 - 1);
        };
        const q = try p.a.alloc(u16, p.rows * nq * dh);
        defer p.a.free(q);
        for (q) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) { // every key 0 (KW is 0), so every weight 1; values cancel but for four ones
            @memset(q, 0);
            @memset(p.vp, 0);
            for (0..nk) |h| for (0..p.p0 + 1) |j| for (0..dh) |e| {
                p.vp[(h * p.cap + j) * dh + e] = c.toBf16(stressValue(j, p.p0 + 1));
            };
        }
        if (fill == .nonfinite) {
            q[(0 * nq + 1) * dh + 3] = c.toBf16(std.math.nan(f32));
            p.vp[p.p0 * dh + 5] = c.toBf16(std.math.inf(f32));
            p.vp[p.p0 * dh + 6] = c.toBf16(-std.math.inf(f32));
            if (p.p0 + 2 < n) p.vp[(p.p0 + 2) * dh] = c.toBf16(std.math.nan(f32)); // its key NaN too
        }
        p.in[0] = try p.rig.input(std.mem.sliceAsBytes(q));
        p.in[1] = try p.rig.input(std.mem.sliceAsBytes(p.vp));
        var meta: [tf.gemma4.state.meta_ints]i32 = @splat(0);
        meta[1] = @intCast(p.chunks());
        meta[3] = @intCast(p.rows);
        p.in[2] = try p.rig.input(std.mem.asBytes(&meta));
        const slots = nq * p.rows * p.chunks();
        for (p.work[0..3], [_]usize{ slots * 4, slots * 4, slots * dh * 4 }) |*w, n_| w.* = try p.rig.scratch(n_);
        p.work[3] = try p.rig.scratch(nk * p.cap * dh * 2); // the twin's keys, rebuilt from the values
        p.work[4] = try p.rig.scratch(p.count() * 2); // the twin's output
        p.work[5] = try p.rig.scratch(4);
        p.out = try p.rig.output(p.count() * 2);
        return true;
    }

    /// gemma_qkv_prep (or the V-only one) over the raw rows: q, k, v rows in work[0..3], caches kc and vc at cap slots.
    fn prepRun(p: *Probe, pipe: mtl.Pipeline, kc: rig_.Ref, vc: rig_.Ref, slots: usize) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(pipe);
        e.setBuffer(p.in[0].buf, p.in[0].off, 0);
        e.setBuffer(p.in[5].buf, p.in[5].off, 1);
        e.setBuffer(p.in[3].buf, p.in[3].off, 2);
        e.setBuffer(p.in[4].buf, p.in[4].off, 3);
        e.setBuffer(p.in[1].buf, p.in[1].off, 4);
        e.setValue(@as(f32, 1e-6), 5);
        for (p.work[0..3], 6..) |w, i| e.setBuffer(w.buf, w.off, i);
        e.setBuffer(kc.buf, kc.off, 9);
        e.setBuffer(vc.buf, vc.off, 10);
        e.setValue([3]i32{ @intCast(slots), @intCast(slots), 0 }, 11); // ring = slots: row j's slot (p0 + j) % slots
        e.dispatchGroups(.{ .width = nq + 2 * nk, .height = p.rows }, .{ .width = 32 });
        try p.rig.finish(pass);
    }

    /// Decode partials then the merge into `out`, keys from `keys` (the twin) or rebuilt (V-only).
    fn decodeRun(p: *Probe, pipe: mtl.Pipeline, keys: rig_.Ref, out: rig_.Ref) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(pipe);
        e.setBuffer(p.in[0].buf, p.in[0].off, 0);
        e.setBuffer(keys.buf, keys.off, 1);
        e.setBuffer(p.in[1].buf, p.in[1].off, 2);
        e.setValue([4]i32{ @intCast(p.rows), @intCast(p.p0), @intCast(p.cap), @intCast(p.chunks()) }, 3);
        for (p.work[0..3], 4..) |w, i| e.setBuffer(w.buf, w.off, i);
        e.setBuffer(p.in[3].buf, p.in[3].off, 7);
        e.setBuffer(p.in[4].buf, p.in[4].off, 8);
        e.dispatchGroups(.{ .width = p.chunks(), .height = nk, .depth = (p.rows * (nq / nk) + 15) / 16 }, .{ .width = 32 * dh / 128 });
        e.setPipeline(p.merge);
        for (p.work[0..3], 0..) |w, i| e.setBuffer(w.buf, w.off, i);
        e.setBuffer(p.in[2].buf, p.in[2].off, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.setBuffer(p.work[5].buf, p.work[5].off, 5);
        e.setValue(@as(i32, 0), 6);
        e.dispatchGroups(.{ .width = 1, .height = nq, .depth = p.rows }, .{ .width = 32 });
        try p.rig.finish(pass);
    }

    /// gemma_key_rebuild of `v` ([nk, n, dh], row j at j or at POS[j]) into `keys`.
    fn rebuildRun(p: *Probe, pipe: mtl.Pipeline, v: rig_.Ref, n: usize, positions: bool, keys: rig_.Ref) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(pipe);
        e.setBuffer(v.buf, v.off, 0);
        e.setBuffer(p.in[3].buf, p.in[3].off, 1);
        e.setBuffer(p.in[4].buf, p.in[4].off, 2);
        e.setBuffer(p.in[1].buf, p.in[1].off, 3);
        e.setValue([2]i32{ @intCast(n), @intFromBool(positions) }, 4);
        e.setBuffer(keys.buf, keys.off, 5);
        e.dispatchGroups(.{ .width = n, .height = nk }, .{ .width = 32 });
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        switch (p.v.kind) {
            .decode => try p.decodeRun(p.pipe, p.in[1], p.out), // V-only: the values bound as keys too, never read so
            .rebuild => try p.rebuildRun(p.pipe, p.in[2], p.rows, true, p.out),
            .prep => try p.prepRun(p.pipe, p.work[3], p.out, p.cap),
        }
    }

    /// The K+V twin's bytes on the same inputs.
    pub fn replaced(p: *Probe) ![]const u8 {
        const n = p.count() * 2;
        switch (p.v.kind) {
            .decode => {
                try p.rebuildRun(p.rebuild, p.in[1], p.cap, false, p.work[3]);
                try p.decodeRun(p.twin, p.work[3], p.work[4]);
                return p.rig.bytes(p.work[4], n);
            },
            .rebuild => {
                try p.prepRun(p.prep, p.work[0], p.work[0], 0);
                return p.rig.bytes(p.work[1], n);
            },
            .prep => {
                @memset(p.rig.bytes(p.work[4], n / 2), 0xff);
                try p.prepRun(p.prep, p.work[5], .{ .buf = p.work[4].buf, .off = p.work[4].off + n / 2 }, p.cap);
                return p.rig.bytes(p.work[4], n);
            },
        }
    }

    /// Gate 1: the twin's bytes, NaN payloads included (the prep's: its key cache still poisoned).
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.a.dupe(u8, try p.rig.outputBytes());
        defer p.a.free(got);
        const want = try p.replaced();
        if (!std.mem.eql(u8, got, want)) return "its bytes differ from its K+V twin's";
        return null;
    }

    /// Gate 4 (decode): every weight 1, so each output is its row's values' exact mean.
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (!p.stress or p.v.kind != .decode) return error.ReferenceIsTheTwin;
        try p.simulate(.f32, .sequential, want);
        @memcpy(mag, want);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!p.stress) return error.StressRequiresEqualScores;
        var terms: std.ArrayList(f64) = .empty;
        defer terms.deinit(p.a);
        for (0..p.rows) |r| for (0..nk) |hk| for (0..dh) |e| {
            terms.clearRetainingCapacity();
            for (0..p.p0 + r + 1) |j| try terms.append(p.a, c.fromBf16(p.vp[(hk * p.cap + j) * dh + e]));
            const v = c.fromBf16(c.toBf16(@floatCast(contract.rows.sumIn(acc, order, terms.items) / @as(f64, @floatFromInt(terms.items.len)))));
            for (hk * (nq / nk)..(hk + 1) * (nq / nk)) |h| out[(r * nq + h) * dh + e] = v;
        };
    }
};

/// Gate 4's value for key `j` of `n`: big halves that cancel, four ones left, so only exact fp32 sums give 4.
fn stressValue(j: usize, n: usize) f32 {
    const half = n / 2;
    if (j >= 2 * half) return 0;
    const at = if (j < half) j else j - half;
    const big: f32 = if (at % 2 == 0) 127 else 16256;
    const one: f32 = if (at < 8 and at % 2 == 0) 1 else 0;
    return if (j < half) big + one else -big;
}

const vt = g.hooks(Probe);

/// The V-only kernels and their controls, on every Mac (bit-equal gates need no device-specific bound).
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for (variants, names) |v, name| {
            if (only) |o| if (std.mem.indexOf(u8, name, o) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, v, name);
            errdefer p.deinit();
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, .{ .spec = p.spec(), .ctx = p, .vt = &vt, .control = if (v.control) .shapes else null });
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
