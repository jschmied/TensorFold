//! Qwen3-ASR's own Metal kernels (qwen3_asr.metal) under the contract: the moves here, the arithmetic in _math.zig.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const math = @import("metal_qwen3_asr_math.zig");

const c = contract.spec;
const g = contract.gates;

/// The source a test compiles: qwen3_asr.metal, or with one planted fault (each red at the shapes gate).
pub const Mutation = enum {
    none,
    gemm_no_residual,
    gemv_bias_column0,
    conv_taps_transposed,
    flatten_f_major,
    layernorm_no_bias,
    rms_mean_over_threads,
    swiglu_swapped,
    embed_own_row,
    copy_drops_last,
    rope_turns_back,
    split_k_reads_v,

    pub fn name(m: Mutation) []const u8 {
        return switch (m) {
            .none => "",
            .gemm_no_residual => "control: metal qwen3-asr q3a_gemm, the residual is not added",
            .gemv_bias_column0 => "control: metal qwen3-asr q3a_gemv, every column adds column 0's bias",
            .conv_taps_transposed => "control: metal qwen3-asr q3a_conv_cols, kh and kw swapped",
            .flatten_f_major => "control: metal qwen3-asr q3a_flatten, rows written f-major",
            .layernorm_no_bias => "control: metal qwen3-asr q3a_layernorm, the bias is dropped",
            .rms_mean_over_threads => "control: metal qwen3-asr q3a_add_rmsnorm, the mean of squares divides by the threads",
            .swiglu_swapped => "control: metal qwen3-asr q3a_swiglu, gate and up swapped",
            .embed_own_row => "control: metal qwen3-asr q3a_embed, row i reads table row i, not ids[i]",
            .copy_drops_last => "control: metal qwen3-asr q3a_copy, the last element is not copied",
            .rope_turns_back => "control: metal qwen3-asr q3a_qk_rope, the rotation turns the other way",
            .split_k_reads_v => "control: metal qwen3-asr q3a_split_qkv, k is read from the v columns",
        };
    }

    pub const Edit = struct { from: []const u8, to: []const u8 };

    pub fn edit(m: Mutation) Edit {
        return switch (m) {
            .none => unreachable,
            .gemm_no_residual => .{ .from = "if (p.res != 0) v = bf(v + float(res[at]));", .to = "if (p.res != 0) v = bf(v);" },
            .gemv_bias_column0 => .{ .from = "float v = bf(s + float(b[col]));", .to = "float v = bf(s + float(b[0]));" },
            .conv_taps_transposed => .{ .from = "kh = (col % 9) / 3, kw = col % 3;", .to = "kw = (col % 9) / 3, kh = col % 3;" },
            .flatten_f_major => .{ .from = "const uint fi = i % f, ci = (i / f) % c,", .to = "const uint ci = i % c, fi = (i / c) % f," },
            .layernorm_no_bias => .{ .from = "* inv * float(w[i]) + float(b[i]));", .to = "* inv * float(w[i]));" },
            .rms_mean_over_threads => .{ .from = "tg_sum(s, part, n, lane, sg) / float(p.width) + p.eps)", .to = "tg_sum(s, part, n, lane, sg) / float(n) + p.eps)" },
            .swiglu_swapped => .{
                .from = "const float g = float(gu[ulong(r) * 2 * p.y + c]), u = float(gu[ulong(r) * 2 * p.y + p.y + c]);",
                .to = "const float u = float(gu[ulong(r) * 2 * p.y + c]), g = float(gu[ulong(r) * 2 * p.y + p.y + c]);",
            },
            .embed_own_row => .{ .from = "table[ulong(ids[at.y]) * p.y + at.x]", .to = "table[ulong(at.y) * p.y + at.x]" },
            .copy_drops_last => .{ .from = "if (i < n) dst[i] = src[i];", .to = "if (i + 1 < n) dst[i] = src[i];" },
            .rope_turns_back => .{ .from = "(d < half_ ? -yp : yp)", .to = "(d < half_ ? yp : -yp)" },
            .split_k_reads_v => .{ .from = "k[dst] = qkv[src + d];", .to = "k[dst] = qkv[src + 2 * d];" },
        };
    }
};

/// qwen3_asr.metal as the binding's metallib builds it (safe math, precise fp32 functions), with a control's edit.
pub fn pipeline(a: std.mem.Allocator, device: mtl.Device, m: Mutation, name: []const u8) !mtl.Pipeline {
    var text: []const u8 = sources.qwen3_asr;
    var owned: ?[]u8 = null;
    defer if (owned) |t| a.free(t);
    if (m != .none) {
        const e = m.edit();
        if (std.mem.count(u8, text, e.from) != 1) return error.ControlEditMissing;
        owned = try std.mem.replaceOwned(u8, a, text, e.from, e.to);
        text = owned.?;
    }
    const opts: mtl.CompileOptions = .{ .math_mode = .safe, .math_functions = .precise, .language = mtl.CompileOptions.mlx().language };
    const lib = try mtl.Library.fromSource(device, text, opts);
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// A test's kernel under a control's name and gate.
pub fn mark(k: g.Kernel, m: Mutation) g.Kernel {
    var out = k;
    if (m != .none) {
        out.spec.name = m.name();
        out.control = .shapes;
    }
    return out;
}

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value rounded to bf16 (to nearest even) as the kernels' bf() does.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// A value rounded to fp32.
pub fn f32r(v: f64) f64 {
    return @as(f32, @floatCast(v));
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

/// Random bf16 values in (-s, s).
pub fn fillBf16(rng: std.Random, out: []u16, s: f32) void {
    for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * s);
}

/// A value's spread through fp32 steps and bf16 rounding points (each end rounds the same way, so it stays inside).
pub const Range = struct {
    lo: f64,
    hi: f64,
    pub fn point(v: f64) Range {
        return .{ .lo = v, .hi = v };
    }
    /// Each end moved out by `ulps` fp32 roundings of its own size (and fp32's least step).
    pub fn fp32(a: Range, ulps: f64) Range {
        const step = ulps * c.unit(.f32);
        return .{ .lo = a.lo - step * @abs(a.lo) - 0x1p-149, .hi = a.hi + step * @abs(a.hi) + 0x1p-149 };
    }
    pub fn widen(a: Range, e: f64) Range {
        return .{ .lo = a.lo - e, .hi = a.hi + e };
    }
    pub fn bf16(a: Range) Range {
        return .{ .lo = bf(a.lo), .hi = bf(a.hi) };
    }
    pub fn add(a: Range, b: Range) Range {
        return .{ .lo = a.lo + b.lo, .hi = a.hi + b.hi };
    }
    pub fn mul(a: Range, b: Range) Range {
        const p = [_]f64{ a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
        return .{ .lo = @min(@min(p[0], p[1]), @min(p[2], p[3])), .hi = @max(@max(p[0], p[1]), @max(p[2], p[3])) };
    }
    pub fn most(a: Range) f64 {
        return @max(@abs(a.lo), @abs(a.hi));
    }
};

/// Bounds from a test's intervals: each output's allowance is its interval's farther end (none for non-finite ones).
pub fn boundsFromIntervals(a: std.mem.Allocator, t: anytype, want: []const f64, allow: []f64) !void {
    const n = want.len;
    const buf = try a.alloc(f64, 2 * n);
    defer a.free(buf);
    try t.intervals(buf[0..n], buf[n..]);
    for (allow, want, buf[0..n], buf[n..]) |*l, w, lo, hi| l.* = if (std.math.isFinite(w)) @max(@max(@abs(w - lo), @abs(hi - w)), 0x1p-133) else 0;
}

const Conv = extern struct { n: u32, fi: u32, ti: u32, c: u32, fo: u32, to: u32, ld: u32 };

fn u(x: usize) u32 {
    return @intCast(x);
}

fn ceil(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The kernels that only move bf16 values: every output's bits checked against the host's replay.
pub const MoveKind = enum { conv_cols, flatten, embed, copy, split_qkv };

/// rows chunks, n channels, k = f 256 + t (conv_cols, flatten); rows ids, n hidden, k table rows (embed);
/// n values (copy); rows, n = heads x dim (split_qkv).
pub const Move = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    m: Mutation,
    kind: MoveKind,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    ids: []i32 = &.{},
    want: []u16 = &.{},
    ins: [2]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    const fn_names = [_][]const u8{ "q3a_conv_cols", "q3a_flatten", "q3a_embed", "q3a_copy", "q3a_split_qkv" };

    pub fn spec(kind: MoveKind) c.Spec {
        const served: []const c.Shape = switch (kind) {
            // the tower's three convs (two chunks each; c 1 then 480) and odd grids for the padding's edges
            .conv_cols => &.{ .{ .rows = 2, .n = 1, .k = 128 * 256 + 100 }, .{ .rows = 2, .n = 480, .k = 64 * 256 + 50 }, .{ .rows = 2, .n = 480, .k = 32 * 256 + 25 }, .{ .rows = 3, .n = 3, .k = 7 * 256 + 5 }, .{ .rows = 1, .n = 1, .k = 1 * 256 + 1 } },
            .flatten => &.{ .{ .rows = 2, .n = 480, .k = 16 * 256 + 13 }, .{ .rows = 3, .n = 8, .k = 5 * 256 + 3 }, .{ .rows = 1, .n = 1, .k = 1 * 256 + 1 } },
            // a prompt's ids into the 0.6B's and the 1.7B's hidden width, from a table of 4,096 rows
            .embed => &.{ .{ .rows = 150, .n = 1024, .k = 4096 }, .{ .rows = 1, .n = 1024, .k = 4096 }, .{ .rows = 33, .n = 2048, .k = 4096 } },
            // a window's fc1 rows before GELU, an LM row, an odd count
            .copy => &.{ .{ .rows = 1, .n = 104 * 3584, .k = 1 }, .{ .rows = 1, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 1001, .k = 1 } },
            // a window of the 0.6B's and of the 1.7B's tower, a short last window
            .split_qkv => &.{ .{ .rows = 104, .n = 896, .k = 1 }, .{ .rows = 104, .n = 1024, .k = 1 }, .{ .rows = 7, .n = 896, .k = 1 } },
        };
        const names = [_][]const u8{
            "metal qwen3-asr q3a_conv_cols (3x3 stride-2 conv taps, zero padding)",
            "metal qwen3-asr q3a_flatten (conv rows to torch's [c, f] order)",
            "metal qwen3-asr q3a_embed",
            "metal qwen3-asr q3a_copy",
            "metal qwen3-asr q3a_split_qkv",
        };
        return .{ .name = names[@backingInt(kind)], .accum = .none, .depth = noSums, .out = .bf16, .served = served };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, m: Mutation, kind: MoveKind) !Move {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pipeline(a, device, m, fn_names[@backingInt(kind)]), .m = m, .kind = kind };
    }
    pub fn deinit(t: *Move) void {
        t.free();
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Move) void {
        t.a.free(t.x);
        t.a.free(t.ids);
        t.a.free(t.want);
        t.x = &.{};
        t.ids = &.{};
        t.want = &.{};
    }
    pub fn kernel(t: *Move) g.Kernel {
        return mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.m);
    }
    const vt = g.hooks(Move);

    const Geo = struct { n: usize, ch: usize, f: usize, t: usize };
    fn geo(s: c.Shape) Geo {
        return .{ .n = s.rows, .ch = s.n, .f = s.k / 256, .t = s.k % 256 };
    }

    pub fn prepare(t: *Move, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        const gm = geo(shape);
        if ((t.kind == .conv_cols or t.kind == .flatten) and (gm.f == 0 or gm.t == 0)) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const xn: usize = switch (t.kind) {
            .conv_cols, .flatten => gm.n * gm.f * gm.t * gm.ch,
            .embed => shape.k * shape.n,
            .copy => shape.n,
            .split_qkv => shape.rows * 3 * shape.n,
        };
        t.x = try t.a.alloc(u16, xn);
        fillBf16(rng, t.x, 4);
        if (t.kind == .embed) {
            t.ids = try t.a.alloc(i32, shape.rows);
            for (t.ids, 0..) |*id, i| id.* = @intCast((i * 7 + 3 + rng.uintLessThan(usize, shape.k)) % shape.k);
        }
        if (fill == .nonfinite) for (planted, 0..) |v, i| {
            const at = if (t.kind == .embed) @as(usize, @intCast(t.ids[i % shape.rows])) * shape.n + (3 * i + 1) % shape.n else (xn / 3 * i + 1) % xn;
            t.x[at] = c.toBf16(v);
        };
        try t.replay();
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        if (t.kind == .embed) t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.ids));
        if (t.kind == .split_qkv) {
            for (&t.outs) |*o| o.* = try t.rig.output(shape.rows * shape.n * 2);
        } else t.outs[0] = try t.rig.output(t.want.len * 2);
        return true;
    }

    /// The host's copy of what the kernel writes: zero where the conv's padding is.
    fn replay(t: *Move) !void {
        const s = t.shape;
        const gm = geo(s);
        switch (t.kind) {
            .conv_cols => {
                const fo = (gm.f - 1) / 2 + 1;
                const to = (gm.t - 1) / 2 + 1;
                const w = 9 * gm.ch;
                t.want = try t.a.alloc(u16, gm.n * fo * to * w);
                for (0..gm.n) |n| for (0..fo) |f| for (0..to) |tt| for (0..w) |j| {
                    var v: u16 = 0;
                    const sf = @as(isize, @intCast(2 * f + j % 9 / 3)) - 1;
                    const st = @as(isize, @intCast(2 * tt + j % 3)) - 1;
                    if (sf >= 0 and st >= 0 and sf < gm.f and st < gm.t) v = t.x[((n * gm.f + @as(usize, @intCast(sf))) * gm.t + @as(usize, @intCast(st))) * gm.ch + j / 9];
                    t.want[((n * fo + f) * to + tt) * w + j] = v;
                };
            },
            .flatten => {
                t.want = try t.a.alloc(u16, t.x.len);
                for (0..gm.n) |n| for (0..gm.t) |tt| for (0..gm.ch) |ci| for (0..gm.f) |f| {
                    t.want[(n * gm.t + tt) * gm.ch * gm.f + ci * gm.f + f] = t.x[((n * gm.f + f) * gm.t + tt) * gm.ch + ci];
                };
            },
            .embed => {
                t.want = try t.a.alloc(u16, s.rows * s.n);
                for (0..s.rows) |r| @memcpy(t.want[r * s.n ..][0..s.n], t.x[@as(usize, @intCast(t.ids[r])) * s.n ..][0..s.n]);
            },
            .copy => t.want = try t.a.dupe(u16, t.x),
            .split_qkv => {
                const d = s.n;
                t.want = try t.a.alloc(u16, 3 * s.rows * d);
                for (0..3) |part| for (0..s.rows) |r| {
                    @memcpy(t.want[(part * s.rows + r) * d ..][0..d], t.x[r * 3 * d + part * d ..][0..d]);
                };
            },
        }
    }

    pub fn run(t: *Move) !void {
        const s = t.shape;
        const gm = geo(s);
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        const one = mtl.Size.of(256, 1, 1);
        switch (t.kind) {
            .conv_cols => {
                const fo = (gm.f - 1) / 2 + 1;
                const to = (gm.t - 1) / 2 + 1;
                p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 1);
                p.e.setValue(Conv{ .n = u(gm.n), .fi = u(gm.f), .ti = u(gm.t), .c = u(gm.ch), .fo = u(fo), .to = u(to), .ld = u(9 * gm.ch) }, 2);
                p.e.dispatchGroups(mtl.Size.of(ceil(9 * gm.ch, 256), gm.n * fo * to, 1), one);
            },
            .flatten => {
                p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 1);
                p.e.setValue(Conv{ .n = u(gm.n), .fi = u(gm.f), .ti = u(gm.t), .c = u(gm.ch), .fo = 0, .to = 0, .ld = 0 }, 2);
                p.e.dispatchGroups(mtl.Size.of(ceil(t.x.len, 256), 1, 1), one);
            },
            .embed => {
                p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 0);
                p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 1);
                p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 2);
                p.e.setValue([2]u32{ u(s.rows), u(s.n) }, 3);
                p.e.dispatchGroups(mtl.Size.of(ceil(s.n, 256), s.rows, 1), one);
            },
            .copy => {
                p.e.setBuffer(t.outs[0].buf, t.outs[0].off, 1);
                p.e.setValue(u(s.n), 2);
                p.e.dispatchGroups(mtl.Size.of(ceil(s.n, 256), 1, 1), one);
            },
            .split_qkv => {
                for (t.outs, 1..) |o, i| p.e.setBuffer(o.buf, o.off, i);
                p.e.setValue([2]u32{ u(s.rows), u(s.n) }, 4);
                p.e.dispatchGroups(mtl.Size.of(ceil(s.n, 256), s.rows, 1), one);
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Move) usize {
        return t.want.len;
    }

    pub fn reference(t: *Move, want: []f64, mag: []f64) !void {
        for (want, t.want) |*w, b| w.* = c.fromBf16(b);
        @memset(mag, 0);
    }

    /// Every output's bits equal to the host's copy (NaN payloads too).
    pub fn check(t: *Move, _: g.Gate) !?[]const u8 {
        const got = try t.rig.outputBytes();
        for (t.want, 0..) |w, i| {
            const v = std.mem.readInt(u16, got[2 * i ..][0..2], .little);
            if (v != w) return try std.fmt.bufPrint(&t.msg, "output {d} has bits {x:0>4} where the copy has {x:0>4}", .{ i, v, w });
        }
        return null;
    }
};

/// Every registration, on the heap (the gates hold pointers to them); each control after the kernel it breaks.
pub const Suite = struct {
    a: std.mem.Allocator,
    moves: std.ArrayList(*Move) = .empty,
    math: math.Set,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .math = .{ .a = a, .device = device, .queue = queue } };
        errdefer s.destroy();
        const pairs = [_]struct { MoveKind, Mutation }{ .{ .conv_cols, .conv_taps_transposed }, .{ .flatten, .flatten_f_major }, .{ .embed, .embed_own_row }, .{ .copy, .copy_drops_last }, .{ .split_qkv, .split_k_reads_v } };
        for (pairs) |pr| for ([_]Mutation{ .none, pr[1] }) |m| {
            const x = try a.create(Move);
            x.* = Move.init(a, device, queue, m, pr[0]) catch |err| {
                a.destroy(x);
                return err;
            };
            try s.moves.append(a, x);
            const k = x.kernel();
            if (wanted(only, k.spec.name)) try out.append(a, k);
        };
        try s.math.collect(out, only);
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.moves.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.moves.deinit(s.a);
        s.math.deinit();
        s.a.destroy(s);
    }
};

/// A registration the run's --only selects (all without it).
pub fn wanted(only: ?[]const u8, name: []const u8) bool {
    const o = only orelse return true;
    return std.mem.indexOf(u8, name, o) != null;
}

/// Gate 5's text for a `qwen3-asr ` launch: qwen3_asr.metal as is, on every chip (it includes no nax.h).
pub fn limitsText(name: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, name, "qwen3-asr ")) return null;
    const fun = name[std.mem.lastIndexOfScalar(u8, name, ' ').? + 1 ..];
    for (Move.fn_names ++ math.fn_names) |f| if (std.mem.eql(u8, f, fun)) return sources.qwen3_asr;
    return null;
}
