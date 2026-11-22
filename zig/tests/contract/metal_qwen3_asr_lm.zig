//! qwen3_asr.metal's LM kernels under the contract: add + RMS norm, SwiGLU, and QK norm + RoPE into the caches.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_qwen3_asr.zig");
const math = @import("metal_qwen3_asr_math.zig");

const c = contract.spec;
const g = contract.gates;
const Mutation = k.Mutation;
const Range = k.Range;
const bf = k.bf;
const f32r = k.f32r;
const u = math.u;
const chain = math.chain;
const rowDepth = math.rowDepth;
const Rows_ = math.Rows_;
const QkRope_ = extern struct { h: u32, hk: u32, dim: u32, eps: f32, cap: u32 };

// --- q3a_add_rmsnorm: h = bf16(x + res), y = bf16(w bf16(h rsqrt(mean h^2 + eps))) ---------------------------------

pub const AddRms = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    with_res: bool,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    res: []u16 = &.{},
    w: []u16 = &.{},
    ins: [3]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,

    /// Rows: pass rows; n = k: the LM's hidden width. With the residual, y then h are checked.
    pub fn spec(with_res: bool) c.Spec {
        return .{
            .name = if (with_res) "metal qwen3-asr q3a_add_rmsnorm, residual added (y and h)" else "metal qwen3-asr q3a_add_rmsnorm",
            .accum = .f32,
            .depth = rowDepth,
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            // a prompt pass and a decode row of the 0.6B (1,024) and the 1.7B (2,048)
            .served = &.{ .{ .rows = 150, .n = 1024, .k = 1024 }, .{ .rows = 1, .n = 1024, .k = 1024 }, .{ .rows = 33, .n = 2048, .k = 2048 }, .{ .rows = 1, .n = 2048, .k = 2048 } },
            .sums = .{ .bf16 = 0, .why = "the bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, with_res: bool) !AddRms {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "q3a_add_rmsnorm"), .m = m, .with_res = with_res };
    }
    pub fn deinit(t: *AddRms) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *AddRms) void {
        inline for (.{ "x", "res", "w" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *AddRms) g.Kernel {
        return k.mark(.{ .spec = spec(t.with_res), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(AddRms);

    pub fn prepare(t: *AddRms, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n != shape.k) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const n = shape.rows * shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, n);
        t.res = try t.a.alloc(u16, n);
        t.w = try t.a.alloc(u16, shape.k);
        if (fill == .stress) {
            // integers whose squares sum exactly in fp32 (2,048 x 1,600 at most) and overflow fp16 in any order
            for (t.x) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            @memset(t.res, 0);
            for (t.w) |*v| v.* = c.toBf16(1);
        } else {
            k.fillBf16(rng, t.x, 4);
            k.fillBf16(rng, t.res, 2);
            for (t.w) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        }
        if (fill == .nonfinite) {
            const W = shape.k;
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 4 }, .{ 3, 5 } };
            const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1], k.planted[2] };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                t.x[p[0] * W + p[1]] = c.toBf16(v);
            };
            if (t.with_res and shape.rows > 4) t.res[4 * W + 6] = c.toBf16(k.planted[0]);
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.res));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.outs[0] = try t.rig.output(n * 2);
        t.outs[1] = if (t.with_res) try t.rig.output(n * 2) else try t.rig.scratch(n * 2);
        return true;
    }

    pub fn run(t: *AddRms) !void {
        const s = t.shape;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        // without the residual the binding binds x again as res
        const r = if (t.with_res) t.ins[1] else t.ins[0];
        p.e.setBuffer(r.buf, r.off, 1);
        p.e.setBuffer(t.ins[2].buf, t.ins[2].off, 2);
        p.e.setBuffer(t.outs[1].buf, t.outs[1].off, 3);
        p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 4);
        p.e.setValue(Rows_{ .rows = u(s.rows), .width = u(s.k), .eps = 1e-6, .has_res = @intFromBool(t.with_res) }, 5);
        p.e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *AddRms) usize {
        return t.shape.rows * t.shape.k * @as(usize, if (t.with_res) 2 else 1);
    }

    fn h(t: *AddRms, i: usize) f64 {
        const x: f64 = c.fromBf16(t.x[i]);
        return if (t.with_res) bf(f32r(x + c.fromBf16(t.res[i]))) else x;
    }

    fn squares(t: *AddRms, r: usize, out: []f64) void {
        for (out, 0..) |*o, j| {
            const v = t.h(r * t.shape.k + j);
            o.* = v * v;
        }
    }

    /// y from each row's sum of squares (float64, or an accumulator's), then h.
    fn outputs(t: *AddRms, acc: c.Accum, order: g.Order, out: []f64) !void {
        const W = t.shape.k;
        const sq = try t.a.alloc(f64, W);
        defer t.a.free(sq);
        for (0..t.shape.rows) |r| {
            t.squares(r, sq);
            var ss: f64 = 0;
            for (sq) |v| ss += v;
            if (acc != .none) ss = contract.rows.sumIn(acc, order, sq);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(W)) + 1e-6);
            for (0..W) |j| out[r * W + j] = bf(c.fromBf16(t.w[j]) * bf(t.h(r * W + j) * inv));
        }
        if (t.with_res) for (0..t.shape.rows * W) |i| {
            out[t.shape.rows * W + i] = t.h(i);
        };
    }

    pub fn reference(t: *AddRms, want: []f64, mag: []f64) !void {
        try t.outputs(.none, .sequential, want);
        @memset(mag, 0);
    }

    pub fn simulate(t: *AddRms, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.outputs(acc, order, out);
    }

    /// The squares' fp32 chain (exact at gate 4), mean, eps, sqrt, 1 / (4 roundings), h * inv, bf16, w, bf16.
    pub fn intervals(t: *AddRms, low: []f64, high: []f64) !void {
        const W = t.shape.k;
        const sq = try t.a.alloc(f64, W);
        defer t.a.free(sq);
        const wf: f64 = @floatFromInt(W);
        for (0..t.shape.rows) |r| {
            t.squares(r, sq);
            var ss: f64 = 0;
            for (sq) |v| ss += v;
            const e = if (t.fill == .stress) 0 else chain(rowDepth(t.shape), ss);
            const inv = (Range{ .lo = 1 / @sqrt((ss + e) / wf + 1e-6), .hi = 1 / @sqrt((ss - e) / wf + 1e-6) }).fp32(4);
            for (0..W) |j| {
                const y = Range.point(t.h(r * W + j)).mul(inv).fp32(1).bf16().mul(Range.point(c.fromBf16(t.w[j]))).fp32(1).bf16();
                low[r * W + j] = y.lo;
                high[r * W + j] = y.hi;
            }
        }
        if (t.with_res) for (0..t.shape.rows * W) |i| {
            low[t.shape.rows * W + i] = t.h(i);
            high[t.shape.rows * W + i] = t.h(i);
        };
    }

    pub fn bounds(t: *AddRms, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};

// --- q3a_swiglu: out [rows, i] = bf16(silu(gate) up) of [gate | up] rows ------------------------------------------

pub const Swiglu = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    gu: []u16 = &.{},
    in: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: pass rows; n: the intermediate width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen3-asr q3a_swiglu",
            .accum = .none,
            .depth = k.noSums,
            // precise exp (4 ulps), 1 +, 1 /, * g, * up, with room
            .ops = 10,
            .out = .bf16,
            .served = &.{ .{ .rows = 150, .n = 3072, .k = 1 }, .{ .rows = 1, .n = 3072, .k = 1 }, .{ .rows = 33, .n = 6144, .k = 1 }, .{ .rows = 1, .n = 6144, .k = 1 } },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !Swiglu {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "q3a_swiglu"), .m = m };
    }
    pub fn deinit(t: *Swiglu) void {
        t.a.free(t.gu);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Swiglu) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Swiglu);

    pub fn prepare(t: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.gu);
        t.gu = &.{};
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = shape.n;
        t.gu = try t.a.alloc(u16, shape.rows * 2 * n);
        k.fillBf16(rng, t.gu, 6);
        if (fill == .nonfinite) {
            // gate NaN, +Inf, -Inf (silu(-Inf) = -Inf x 0); an infinite up in row 3
            const at = [_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, n + 4 } };
            const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1] };
            for (at, vals) |p, v| if (p[0] < shape.rows) {
                t.gu[p[0] * 2 * n + p[1]] = c.toBf16(v);
            };
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.gu));
        t.out = try t.rig.output(shape.rows * n * 2);
        return true;
    }

    pub fn run(t: *Swiglu) !void {
        const s = t.shape;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue([2]u32{ u(s.rows), u(s.n) }, 2);
        p.e.dispatchGroups(mtl.Size.of((s.n + 255) / 256, s.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Swiglu) usize {
        return t.shape.rows * t.shape.n;
    }

    /// g (1 / (1 + exp(-g))) up in float64, the kernel's formula (NaN where -Inf meets 0, as the kernel's).
    pub fn reference(t: *Swiglu, want: []f64, mag: []f64) !void {
        const n = t.shape.n;
        for (want, 0..) |*w, i| {
            const r = i / n;
            const gv: f64 = c.fromBf16(t.gu[r * 2 * n + i % n]);
            const up: f64 = c.fromBf16(t.gu[r * 2 * n + n + i % n]);
            w.* = (gv * (1 / (1 + @exp(-gv)))) * up;
        }
        @memset(mag, 0);
    }
};

// --- q3a_qk_rope: q and k RMS-normed per head (bf16 points), rotated in fp32 from the tables; v copied ----------

const heads = 16;
const kv_heads = 8;
const head_dim = 128;
const all_heads = heads + 2 * kv_heads;

pub const QkRope = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    qkv: []u16 = &.{},
    qn: []u16 = &.{},
    kn: []u16 = &.{},
    cs: []f32 = &.{},
    sn: []f32 = &.{},
    pos: []i32 = &.{},
    ins: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    /// Rows: pass rows, each at its own cache position (a permutation of the rows: every cache row written);
    /// n: q, k and v heads (both LMs: 16, 8, 8); k: the head width (128). Checked: q, then the k and v caches.
    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen3-asr q3a_qk_rope",
            .accum = .f32,
            .depth = struct {
                fn f(_: c.Shape) usize {
                    return 2 + 5 + 5; // a thread's square, the simd_sum tree, the 4 simdgroups' partials
                }
            }.f,
            .ops = 6,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 150, .n = all_heads, .k = head_dim }, .{ .rows = 1, .n = all_heads, .k = head_dim }, .{ .rows = 17, .n = all_heads, .k = head_dim } },
            .sums = .{ .bf16 = 0, .why = "the bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !QkRope {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "q3a_qk_rope"), .m = m };
    }
    pub fn deinit(t: *QkRope) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *QkRope) void {
        inline for (.{ "qkv", "qn", "kn", "cs", "sn", "pos" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *QkRope) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(QkRope);

    pub fn prepare(t: *QkRope, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != all_heads or shape.k != head_dim) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const rows = shape.rows;
        const d = head_dim;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.qkv = try t.a.alloc(u16, rows * all_heads * d);
        t.qn = try t.a.alloc(u16, d);
        t.kn = try t.a.alloc(u16, d);
        t.cs = try t.a.alloc(f32, rows * d);
        t.sn = try t.a.alloc(f32, rows * d);
        t.pos = try t.a.alloc(i32, rows);
        if (fill == .stress) {
            for (t.qkv) |*v| v.* = c.toBf16(@floatFromInt(@as(i32, @intCast(20 + rng.uintLessThan(u32, 21))) * (if (rng.boolean()) @as(i32, 1) else -1)));
            for (t.qn) |*v| v.* = c.toBf16(1);
            for (t.kn) |*v| v.* = c.toBf16(1);
        } else {
            k.fillBf16(rng, t.qkv, 3);
            for (t.qn) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
            for (t.kn) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        }
        for (t.cs, t.sn) |*cv, *sv| {
            const th = rng.float(f64) * 2 * std.math.pi;
            cv.* = @floatCast(@cos(th));
            sv.* = @floatCast(@sin(th));
        }
        for (t.pos, 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(i32, t.pos);
        if (fill == .nonfinite) {
            // a q head, a k head and a v head of different rows; both infinities in one q head of row 3
            const at = [_][2]usize{ .{ 0, 5 }, .{ 1, heads * d + 7 }, .{ 2, (heads + kv_heads) * d + 9 }, .{ 3, 2 * d + 1 }, .{ 3, 2 * d + 70 } };
            const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1], k.planted[2] };
            for (at, vals) |p, v| if (p[0] < rows) {
                t.qkv[p[0] * all_heads * d + p[1]] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.qkv));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.qn));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.kn));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.cs));
        t.ins[4] = try t.rig.input(std.mem.sliceAsBytes(t.sn));
        t.ins[5] = try t.rig.input(std.mem.sliceAsBytes(t.pos));
        t.outs[0] = try t.rig.output(rows * heads * d * 2);
        t.outs[1] = try t.rig.output(rows * kv_heads * d * 2);
        t.outs[2] = try t.rig.output(rows * kv_heads * d * 2);
        return true;
    }

    pub fn run(t: *QkRope) !void {
        const rows = t.shape.rows;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |in, i| p.e.setBuffer(in.buf, in.off, i);
        for (t.outs, 6..) |o, i| p.e.setBuffer(o.buf, o.off, i);
        p.e.setValue(QkRope_{ .h = heads, .hk = kv_heads, .dim = head_dim, .eps = 1e-6, .cap = u(rows) }, 9);
        p.e.dispatchGroups(mtl.Size.of(rows, all_heads, 1), mtl.Size.of(head_dim, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *QkRope) usize {
        return t.shape.rows * (heads + 2 * kv_heads) * head_dim;
    }

    fn x(t: *QkRope, r: usize, hd: usize, d: usize) f64 {
        return c.fromBf16(t.qkv[(r * all_heads + hd) * head_dim + d]);
    }

    /// Where row r, head hd's value d lands among the outputs (q [rows, h, dim], then kc and vc [pos, hk, dim]).
    fn slot(t: *QkRope, r: usize, hd: usize, d: usize) usize {
        const rows = t.shape.rows;
        const at: usize = @intCast(t.pos[r]);
        if (hd < heads) return (r * heads + hd) * head_dim + d;
        const base = rows * heads * head_dim;
        if (hd < heads + kv_heads) return base + (at * kv_heads + hd - heads) * head_dim + d;
        return base + rows * kv_heads * head_dim + (at * kv_heads + hd - heads - kv_heads) * head_dim + d;
    }

    /// Every output from its head's sum of squares (float64, or an accumulator's): the float64 formula into `lo`
    /// (`hi` unused), or with `slack` the fp32 steps and bf16 points as ranges (Range drops NaN: finite inputs only).
    fn outputs(t: *QkRope, acc: c.Accum, order: g.Order, slack: bool, lo: []f64, hi: []f64) void {
        const d = head_dim;
        var sq: [head_dim]f64 = undefined;
        var y: [head_dim]Range = undefined;
        for (0..t.shape.rows) |r| for (0..all_heads) |hd| {
            if (hd >= heads + kv_heads) {
                for (0..d) |i| {
                    lo[t.slot(r, hd, i)] = t.x(r, hd, i);
                    if (slack) hi[t.slot(r, hd, i)] = t.x(r, hd, i);
                }
                continue;
            }
            for (&sq, 0..) |*s, i| s.* = t.x(r, hd, i) * t.x(r, hd, i);
            var ss: f64 = 0;
            for (sq) |v| ss += v;
            if (acc != .none) ss = contract.rows.sumIn(acc, order, &sq);
            const w = if (hd < heads) t.qn else t.kn;
            const at: usize = @intCast(t.pos[r]);
            if (!slack) {
                const inv = 1 / @sqrt(ss / d + 1e-6);
                var p: [head_dim]f64 = undefined;
                for (&p, 0..) |*v, i| v.* = bf(c.fromBf16(w[i]) * bf(t.x(r, hd, i) * inv));
                for (0..d) |i| {
                    const sv: f64 = if (i < d / 2) -t.sn[at * d + i] else t.sn[at * d + i];
                    lo[t.slot(r, hd, i)] = bf(p[i] * t.cs[at * d + i] + p[(i + d / 2) % d] * sv);
                }
                continue;
            }
            const e = if (t.fill == .stress) 0 else chain(spec().depth(t.shape), ss);
            const inv = (Range{ .lo = 1 / @sqrt((ss + e) / d + 1e-6), .hi = 1 / @sqrt((ss - e) / d + 1e-6) }).fp32(4);
            for (&y, 0..) |*yv, i| yv.* = Range.point(t.x(r, hd, i)).mul(inv).fp32(1).bf16().mul(Range.point(c.fromBf16(w[i]))).fp32(1).bf16();
            for (0..d) |i| {
                const partner = (i + d / 2) % d;
                const cv: f64 = t.cs[at * d + i];
                const sv: f64 = if (i < d / 2) -t.sn[at * d + i] else t.sn[at * d + i];
                // two products and their sum in fp32 (fused or not)
                const e3 = 3 * c.unit(.f32) * (y[i].most() * @abs(cv) + y[partner].most() * @abs(sv)) + 0x1p-148;
                const o = y[i].mul(Range.point(cv)).add(y[partner].mul(Range.point(sv))).widen(e3).bf16();
                lo[t.slot(r, hd, i)] = o.lo;
                hi[t.slot(r, hd, i)] = o.hi;
            }
        };
    }

    pub fn reference(t: *QkRope, want: []f64, mag: []f64) !void {
        t.outputs(.none, .sequential, false, want, mag);
        @memset(mag, 0);
    }

    pub fn simulate(t: *QkRope, acc: c.Accum, order: g.Order, out: []f64) !void {
        t.outputs(acc, order, false, out, &.{});
    }

    pub fn intervals(t: *QkRope, low: []f64, high: []f64) !void {
        t.outputs(.none, .sequential, true, low, high);
    }

    pub fn bounds(t: *QkRope, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};
