//! qwen3_asr.metal's arithmetic under the contract: biased GEMM and GEMV, LayerNorm (the LM's kernels: _lm.zig).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const k = @import("metal_qwen3_asr.zig");

const c = contract.spec;
const g = contract.gates;
const Mutation = k.Mutation;
const Range = k.Range;
const bf = k.bf;
const f32r = k.f32r;
const lm = @import("metal_qwen3_asr_lm.zig");
const AddRms = lm.AddRms;
const Swiglu = lm.Swiglu;
const QkRope = lm.QkRope;

pub const fn_names = [_][]const u8{ "q3a_gemm", "q3a_gemv", "q3a_layernorm", "q3a_add_rmsnorm", "q3a_swiglu", "q3a_qk_rope" };

const Gemm_ = extern struct { m: u32, n: u32, k: u32, res: u32 };
pub const Rows_ = extern struct { rows: u32, width: u32, eps: f32, has_res: u32 };

pub fn u(x: usize) u32 {
    return @intCast(x);
}

/// gamma(d) at fp32 times `mag`: what a d-deep fp32 chain of sums can carry.
pub fn chain(depth: usize, mag: f64) f64 {
    return c.gamma(@floatFromInt(depth), c.unit(.f32)) * mag;
}

// --- q3a_gemm and q3a_gemv: y [rows, n] = bf16(bf16(x . w^T + b) + res), fp32 sums --------------------------------

/// The tile's simdgroup chain over k (8 x 8 MMAs, k in steps of 8): at most k adds.
fn gemmDepth(s: c.Shape) usize {
    return s.k + 8;
}

/// A lane's fma chain over k / 32 columns, then the simd_sum tree.
fn gemvDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

pub const Gemm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    one_row: bool,
    biased: bool, // a bias and a residual, as the tower's fc2 and o; else the LM's zeros and none
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    x: []u16 = &.{},
    w: []u16 = &.{},
    b: []u16 = &.{},
    res: []u16 = &.{},
    ins: [4]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: x rows (one launch a row for the GEMV); n: outputs a row; k: the input width.
    pub fn spec(one_row: bool, biased: bool) c.Spec {
        const names = [2][2][]const u8{
            .{ "metal qwen3-asr q3a_gemm", "metal qwen3-asr q3a_gemm, bias and residual" },
            .{ "metal qwen3-asr q3a_gemv", "metal qwen3-asr q3a_gemv, bias and residual" },
        };
        // first an odd small call for the tiles' edges; then the 0.6B tower's conv2/3, qkv, fc2 and conv_out
        const tower: []const c.Shape = &.{ .{ .rows = 37, .n = 130, .k = 40 }, .{ .rows = 64, .n = 480, .k = 9 * 480 }, .{ .rows = 104, .n = 3 * 896, .k = 896 }, .{ .rows = 104, .n = 896, .k = 3584 }, .{ .rows = 13, .n = 896, .k = 7680 } };
        // the LMs' decode rows: 0.6B qkv, down and the tied head; 1.7B gate | up
        const decode: []const c.Shape = &.{ .{ .rows = 1, .n = 130, .k = 40 }, .{ .rows = 1, .n = 4096, .k = 1024 }, .{ .rows = 1, .n = 1024, .k = 3072 }, .{ .rows = 1, .n = 151936, .k = 1024 }, .{ .rows = 1, .n = 12288, .k = 2048 } };
        return .{
            .name = names[@intFromBool(one_row)][@intFromBool(biased)],
            .accum = .f32,
            .depth = if (one_row) gemvDepth else gemmDepth,
            .ops = 2,
            .mids = 1,
            .out = .bf16,
            .tiles = if (one_row) .{ .{}, .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 32, .step = 1, .min = 1 } } else .{ .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 32, .step = 1, .min = 1 } },
            .served = if (one_row) decode else tower,
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, one_row: bool, biased: bool) !Gemm {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, if (one_row) "q3a_gemv" else "q3a_gemm"), .m = m, .one_row = one_row, .biased = biased };
    }
    pub fn deinit(t: *Gemm) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Gemm) void {
        inline for (.{ "x", "w", "b", "res" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Gemm) g.Kernel {
        return k.mark(.{ .spec = spec(t.one_row, t.biased), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Gemm);

    pub fn prepare(t: *Gemm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        t.fill = fill;
        const r = shape.rows;
        const n = shape.n;
        const kk = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, r * kk);
        t.w = try t.a.alloc(u16, n * kk);
        t.b = try t.a.alloc(u16, n);
        t.res = try t.a.alloc(u16, r * n);
        @memset(t.b, 0);
        @memset(t.res, 0);
        if (fill == .stress) {
            // integer terms, each half mirroring the other (8 nudged): partials pass 2^17, the dot is small and exact
            const h = kk / 2;
            @memset(t.x, 0);
            @memset(t.w, 0);
            for (0..r) |row| for (0..h) |i| {
                const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 32));
                t.x[row * kk + i] = c.toBf16(v);
                t.x[row * kk + h + i] = c.toBf16(v + @as(f32, if (i < 8) 1 else 0));
            };
            for (0..n) |col| for (0..h) |i| {
                const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 7));
                t.w[col * kk + i] = c.toBf16(q);
                t.w[col * kk + h + i] = c.toBf16(-q);
            };
        } else {
            k.fillBf16(rng, t.x, 2);
            k.fillBf16(rng, t.w, 0.1);
            if (t.biased) {
                k.fillBf16(rng, t.b, 0.5);
                k.fillBf16(rng, t.res, 2);
            }
        }
        if (fill == .nonfinite) {
            // NaN in row 0, +Inf in row 1, -Inf in row 2, both infinities in row 3
            const at = [_][2]usize{ .{ 0, kk / 3 }, .{ 1, kk / 2 }, .{ 2, kk - 1 }, .{ 3, 0 }, .{ 3, kk - 1 } };
            const vals = [_]f32{ k.planted[0], k.planted[1], k.planted[2], k.planted[1], k.planted[2] };
            for (at, vals) |p, v| if (p[0] < r) {
                t.x[p[0] * kk + p[1]] = c.toBf16(v);
            };
        }
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.b));
        t.ins[3] = try t.rig.input(std.mem.sliceAsBytes(t.res));
        t.out = try t.rig.output(r * n * 2);
        return true;
    }

    pub fn run(t: *Gemm) !void {
        const s = t.shape;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setBuffer(t.ins[2].buf, t.ins[2].off, 2);
        const res: u32 = @intFromBool(t.biased);
        if (t.one_row) {
            p.e.setValue(Gemm_{ .m = 1, .n = u(s.n), .k = u(s.k), .res = res }, 5);
            for (0..s.rows) |r| {
                p.e.setBuffer(t.ins[0].buf, t.ins[0].off + r * s.k * 2, 0);
                p.e.setBuffer(t.ins[3].buf, t.ins[3].off + r * s.n * 2, 3);
                p.e.setBuffer(t.out.buf, t.out.off + r * s.n * 2, 4);
                p.e.dispatchGroups(mtl.Size.of((s.n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
        } else {
            p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
            p.e.setBuffer(t.ins[3].buf, t.ins[3].off, 3);
            p.e.setBuffer(t.out.buf, t.out.off, 4);
            p.e.setValue(Gemm_{ .m = u(s.rows), .n = u(s.n), .k = u(s.k), .res = res }, 5);
            p.e.dispatchGroups(mtl.Size.of((s.n + 31) / 32, (s.rows + 31) / 32, 1), mtl.Size.of(128, 1, 1));
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Gemm) usize {
        return t.shape.rows * t.shape.n;
    }

    /// Each output from its sum: bf16(fp32(S + b)), then bf16(fp32(that + res)) when biased.
    fn finish(t: *Gemm, i: usize, sum: f64) f64 {
        var v = bf(f32r(sum + c.fromBf16(t.b[i % t.shape.n])));
        if (t.biased) v = bf(f32r(v + c.fromBf16(t.res[i])));
        return v;
    }

    /// Every output's sum in float64 (acc none) or in an accumulator and order, and its terms' magnitude.
    fn sums(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const kk = t.shape.k;
        const buf = try t.a.alloc(f64, kk);
        defer t.a.free(buf);
        for (0..t.shape.rows) |r| for (0..t.shape.n) |col| {
            var s: f64 = 0;
            var m: f64 = 0;
            for (buf, 0..) |*term, j| {
                term.* = @as(f64, c.fromBf16(t.x[r * kk + j])) * c.fromBf16(t.w[col * kk + j]);
                s += term.*;
                m += @abs(term.*);
            }
            if (acc != .none) s = contract.rows.sumIn(acc, order, buf);
            out[r * t.shape.n + col] = s;
            if (mag) |ms| ms[r * t.shape.n + col] = if (std.math.isFinite(m)) m else 0;
        };
    }

    pub fn reference(t: *Gemm, want: []f64, mag: []f64) !void {
        try t.sums(.none, .sequential, want, mag);
        for (want, 0..) |*w, i| w.* = t.finish(i, w.*);
    }

    pub fn simulate(t: *Gemm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.sums(acc, order, out, null);
        for (out, 0..) |*o, i| o.* = t.finish(i, o.*);
    }

    /// The sum within gamma(depth) of its magnitude (exact on gate 4's integers), then the two rounding points.
    pub fn intervals(t: *Gemm, low: []f64, high: []f64) !void {
        const n = t.count();
        const buf = try t.a.alloc(f64, 2 * n);
        defer t.a.free(buf);
        try t.sums(.none, .sequential, buf[0..n], buf[n..]);
        const depth = spec(t.one_row, t.biased).depth(t.shape);
        for (low, high, buf[0..n], buf[n..], 0..) |*lo, *hi, s, m, i| {
            const e = if (t.fill == .stress) 0 else chain(depth, m);
            var r = Range.point(s).widen(e).add(Range.point(c.fromBf16(t.b[i % t.shape.n]))).fp32(1).bf16();
            if (t.biased) r = r.add(Range.point(c.fromBf16(t.res[i]))).fp32(1).bf16();
            lo.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn bounds(t: *Gemm, want: []const f64, _: []const f64, allow: []f64) !void {
        try k.boundsFromIntervals(t.a, t, want, allow);
    }
};

// --- q3a_layernorm: bf16((x - mean) rsqrt(var + eps) w + b), fp32 sums -------------------------------------------

/// A thread's strided adds, the simd_sum tree, then 8 simdgroups' partials.
pub fn rowDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8;
}

pub const LayerNorm = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    rows: ?contract.rows.Rows = null,
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    /// Rows: tower rows; n = k: the width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal qwen3-asr q3a_layernorm",
            .accum = .f32,
            .depth = rowDepth,
            // the mean's division, x - mean, the variance's division, + eps, rsqrt (2), d * inv, * w, + b
            .ops = 10,
            .out = .bf16,
            // a window of the 0.6B's and the 1.7B's tower, a short one
            .served = &.{ .{ .rows = 104, .n = 896, .k = 896 }, .{ .rows = 104, .n = 1024, .k = 1024 }, .{ .rows = 13, .n = 896, .k = 896 } },
            .sums = .{ .bf16 = 0, .why = "a bf16 variance moves the output by its square root: a few output ulps, not 100" },
        };
    }
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation) !LayerNorm {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try k.pipeline(a, device, m, "q3a_layernorm"), .m = m };
    }
    pub fn deinit(t: *LayerNorm) void {
        if (t.rows) |r| r.deinit();
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *LayerNorm) g.Kernel {
        return k.mark(.{ .spec = spec(), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(LayerNorm);

    pub fn prepare(t: *LayerNorm, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n != shape.k) return false;
        t.rig.reset();
        if (t.rows) |r| r.deinit();
        t.rows = null;
        const r = try contract.rows.Rows.init(t.a, .layer_norm, shape);
        t.rows = r;
        r.fill(fill, seed);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(r.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(r.w));
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(r.b));
        t.out = try t.rig.output(shape.rows * shape.k * 2);
        return true;
    }

    pub fn run(t: *LayerNorm) !void {
        const s = t.rows.?.shape;
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        for (t.ins, 0..) |in, i| p.e.setBuffer(in.buf, in.off, i);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.setValue(Rows_{ .rows = u(s.rows), .width = u(s.k), .eps = 1e-5, .has_res = 0 }, 4);
        p.e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *LayerNorm) usize {
        return t.rows.?.outputs();
    }

    pub fn reference(t: *LayerNorm, want: []f64, mag: []f64) !void {
        try t.rows.?.reference(want, mag);
    }

    pub fn simulate(t: *LayerNorm, acc: c.Accum, order: g.Order, out: []f64) !void {
        try t.rows.?.simulate(acc, order, out);
    }

    /// The generic bound, plus the fp32 steps' error where the bias cancels d inv w (want near 0, its terms not).
    pub fn bounds(t: *LayerNorm, want: []const f64, _: []const f64, allow: []f64) !void {
        const r = t.rows.?;
        for (allow, want, 0..) |*l, w, i| {
            const b: f64 = c.fromBf16(r.b[i % r.shape.k]);
            if (std.math.isFinite(w)) l.* += 0x1p-21 * (@abs(w - b) + @abs(b));
        }
    }
};

/// The arithmetic kernels and one control each.
pub const Set = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    gemms: std.ArrayList(*Gemm) = .empty,
    lns: std.ArrayList(*LayerNorm) = .empty,
    rms: std.ArrayList(*AddRms) = .empty,
    swi: std.ArrayList(*Swiglu) = .empty,
    rope: std.ArrayList(*QkRope) = .empty,

    fn add(s: *Set, comptime T: type, list: *std.ArrayList(*T), out: *std.ArrayList(g.Kernel), only: ?[]const u8, m: Mutation, extra: anytype) !void {
        const x = try s.a.create(T);
        x.* = @call(.auto, T.init, .{ s.a, s.device, s.queue, m } ++ extra) catch |err| {
            s.a.destroy(x);
            return err;
        };
        try list.append(s.a, x);
        const kern = x.kernel();
        if (k.wanted(only, kern.spec.name)) try out.append(s.a, kern);
    }

    pub fn collect(s: *Set, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |one_row| for ([_]bool{ false, true }) |biased| try s.add(Gemm, &s.gemms, out, only, .none, .{ one_row, biased });
        try s.add(Gemm, &s.gemms, out, only, .gemm_no_residual, .{ false, true });
        try s.add(Gemm, &s.gemms, out, only, .gemv_bias_column0, .{ true, true });
        for ([_]Mutation{ .none, .layernorm_no_bias }) |m| try s.add(LayerNorm, &s.lns, out, only, m, .{});
        for ([_]bool{ false, true }) |r| try s.add(AddRms, &s.rms, out, only, .none, .{r});
        try s.add(AddRms, &s.rms, out, only, .rms_mean_over_threads, .{false});
        for ([_]Mutation{ .none, .swiglu_swapped }) |m| try s.add(Swiglu, &s.swi, out, only, m, .{});
        for ([_]Mutation{ .none, .rope_turns_back }) |m| try s.add(QkRope, &s.rope, out, only, m, .{});
    }

    pub fn deinit(s: *Set) void {
        inline for (.{ "gemms", "lns", "rms", "swi", "rope" }) |f| {
            for (@field(s, f).items) |x| {
                x.deinit();
                s.a.destroy(x);
            }
            @field(s, f).deinit(s.a);
        }
    }
};
