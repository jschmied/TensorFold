//! MLX's affine 4-bit layout at group 64 on the host: 8 codes a u32 word, low nibble first, bf16 scale and bias.
const std = @import("std");
const bf16_round = @import("bf16_round");

pub const group = 64;

pub fn f32of(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The nearest bf16, ties to even; NaN stays NaN.
pub const bf16of = bf16_round.bits;

/// Row-major values from codes, scales and biases: scale * q + bias in f32 (the rows' width a multiple of 64).
pub fn dequantize(words: []const u32, scales: []const u16, biases: []const u16, out: []f32) void {
    for (out, 0..) |*o, i| {
        const q: u32 = (words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
        o.* = f32of(scales[i / group]) * @as(f32, @floatFromInt(q)) + f32of(biases[i / group]);
    }
}

/// Codes, scales and biases for row-major values: each group's minimum is its bias, its range over 15 its scale.
pub fn quantize(values: []const f32, words: []u32, scales: []u16, biases: []u16) void {
    @memset(words, 0);
    for (0..values.len / group) |g| {
        const v = values[g * group ..][0..group];
        const lo = std.mem.min(f32, v);
        const hi = std.mem.max(f32, v);
        scales[g] = bf16of((hi - lo) / 15);
        biases[g] = bf16of(lo);
        const s = f32of(scales[g]);
        const b = f32of(biases[g]);
        for (v, g * group..) |x, i| {
            const q: u32 = if (s == 0) 0 else @intFromFloat(std.math.clamp(@round((x - b) / s), 0, 15));
            words[i / 8] |= q << @intCast(4 * (i % 8));
        }
    }
}

/// MLX's affine 4-bit quantizer, groups of `gs`; the first group with a non-finite range, scale or bias, else null.
pub fn quantizeMlx(comptime gs: usize, values: []const f32, words: []u32, scales: []u16, biases: []u16) ?usize {
    @memset(words, 0);
    for (0..values.len / gs) |g| {
        const v = values[g * gs ..][0..gs];
        for (v) |x| if (!std.math.isFinite(x)) return g;
        const lo = std.mem.min(f32, v);
        const hi = std.mem.max(f32, v);
        const step = @max((hi - lo) / 15, 1e-7);
        if (!std.math.isFinite(step)) return g;
        const low_edge = @abs(lo) > @abs(hi);
        const signed = if (low_edge) step else -step;
        const edge = if (low_edge) lo else hi;
        const q0 = @round(edge / signed);
        const s: f32 = if (q0 == 0) signed else edge / q0;
        const b: f32 = if (q0 == 0) 0 else edge;
        if (!std.math.isFinite(s) or s == 0 or !std.math.isFinite(b)) return g;
        scales[g] = bf16of(s);
        biases[g] = bf16of(b);
        if (!std.math.isFinite(f32of(scales[g])) or !std.math.isFinite(f32of(biases[g]))) return g;
        for (v, g * gs..) |x, i| {
            const q: u32 = @intFromFloat(std.math.clamp(@round((x - b) / s), 0, 15));
            words[i / 8] |= q << @intCast(4 * (i % 8));
        }
    }
    return null;
}

/// The nearest of the 16 codes to `x`, ties to even.
fn nearest(x: f32) u8 {
    const bounded = std.math.clamp(x, 0, 15);
    const lo = @floor(bounded);
    const integer: u8 = @intFromFloat(lo);
    const fraction = bounded - lo;
    return integer + @intFromBool(fraction > 0.5 or (fraction == 0.5 and integer & 1 == 1));
}

fn fitError(values: []const f32, w: []const f32, codes: []const u8, s: f32, b: f32) f64 {
    var e: f64 = 0;
    for (values, w, codes) |v, wi, q| {
        const r = @as(f64, s) * @as(f64, @floatFromInt(q)) + b - v;
        e += @as(f64, wi) * r * r;
    }
    return e;
}

/// Refits one group's codes, bf16 scale and bias by least squares weighted by `w`, kept only where its error drops.
pub fn quantizeFit(values: []const f32, w: []const f32, codes: []u8, scale: *u16, bias: *u16) bool {
    std.debug.assert(values.len == w.len and values.len == codes.len and values.len <= 256);
    var best = fitError(values, w, codes, f32of(scale.*), f32of(bias.*));
    var cur: [256]u8 = undefined;
    var kept: [256]u8 = undefined;
    @memcpy(cur[0..codes.len], codes);
    var bs = scale.*;
    var bb = bias.*;
    var changed = false;
    // up to three alternating steps: the line for the current codes, then the codes for that line
    for (0..3) |_| {
        var sw: f64 = 0;
        var sq: f64 = 0;
        var sqq: f64 = 0;
        var sv: f64 = 0;
        var sqv: f64 = 0;
        for (values, w, cur[0..codes.len]) |v, wi, q| {
            const qi: f64 = @floatFromInt(q);
            sw += wi;
            sq += wi * qi;
            sqq += wi * qi * qi;
            sv += wi * v;
            sqv += wi * qi * v;
        }
        const det = sw * sqq - sq * sq;
        if (!(det > 1e-30) or !(sw > 0)) break;
        const sc = (sw * sqv - sq * sv) / det;
        if (!(sc > 0)) break;
        const s16 = bf16of(@floatCast(sc));
        const b16 = bf16of(@floatCast((sv - sc * sq) / sw));
        const s = f32of(s16);
        const b = f32of(b16);
        if (!(s > 0) or !std.math.isFinite(s) or !std.math.isFinite(b)) break;
        for (values, cur[0..codes.len]) |v, *q| q.* = nearest((v - b) / s);
        const e = fitError(values, w, cur[0..codes.len], s, b);
        if (!(e < best)) break;
        best = e;
        bs = s16;
        bb = b16;
        @memcpy(kept[0..codes.len], cur[0..codes.len]);
        changed = true;
    }
    if (!changed) return false;
    scale.* = bs;
    bias.* = bb;
    @memcpy(codes, kept[0..codes.len]);
    return true;
}

test "quantizeFit never raises a group's weighted error, keeps a flat group, and leaves zero weights' fit alone" {
    var prng = std.Random.DefaultPrng.init(1870);
    const r = prng.random();
    var raised: usize = 0;
    var moved: usize = 0;
    for (0..512) |_| {
        var v: [32]f32 = undefined;
        var w: [32]f32 = undefined;
        for (&v, &w) |*x, *y| {
            x.* = f32of(bf16of(r.floatNorm(f32) * 0.02));
            y.* = @exp(r.floatNorm(f32) * 2);
        }
        const lo = std.mem.min(f32, &v);
        var scale = bf16of(@max((std.mem.max(f32, &v) - lo) / 15, 1e-8));
        var bias = bf16of(lo);
        var q: [32]u8 = undefined;
        for (&q, v) |*c, x| c.* = nearest((x - f32of(bias)) / f32of(scale));
        const e0 = fitError(&v, &w, &q, f32of(scale), f32of(bias));
        if (quantizeFit(&v, &w, &q, &scale, &bias)) moved += 1;
        if (fitError(&v, &w, &q, f32of(scale), f32of(bias)) > e0) raised += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), raised);
    try std.testing.expect(moved > 256);
    var flat: [32]f32 = @splat(0.25);
    var ones: [32]f32 = @splat(1);
    var q: [32]u8 = @splat(0);
    var s = bf16of(1e-8);
    var b = bf16of(0.25);
    try std.testing.expect(!quantizeFit(&flat, &ones, &q, &s, &b));
    var zero: [32]f32 = @splat(0);
    try std.testing.expect(!quantizeFit(&flat, &zero, &q, &s, &b));
}

test "dequantized codes land within half a step, a flat group exactly on its bias; bf16 ties go to even" {
    var values: [2 * group]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.05;
    @memset(values[group..], 0.02);
    var words: [2 * group / 8]u32 = undefined;
    var scales: [2]u16 = undefined;
    var biases: [2]u16 = undefined;
    quantize(&values, &words, &scales, &biases);
    var back: [2 * group]f32 = undefined;
    dequantize(&words, &scales, &biases, &back);
    for (values, back, 0..) |v, b, i| try std.testing.expect(@abs(v - b) <= f32of(scales[i / group]) * 0.51 + 1e-3);
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16of(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16of(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16of(@bitCast(@as(u32, 0x3f818000))));
    try std.testing.expect(std.math.isNan(f32of(bf16of(std.math.nan(f32)))));
}

test "affine4_host bf16of: the old spelling's bits on every zero, subnormal, infinity, NaN and tie" {
    const old = struct {
        /// bf16of as it was spelled before it was core's bits(): the oracle for every input, NaN included.
        fn spelled(x: f32) u16 {
            const u: u32 = @bitCast(x);
            if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
            return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        }

        fn kept(u: u32) !void {
            const v: f32 = @bitCast(u);
            const h = bf16of(v);
            try std.testing.expectEqual(spelled(v), h);
            if (std.math.isNan(v)) try std.testing.expect(std.math.isNan(f32of(h)));
        }
    };
    for (0..1 << 25) |i| { // sign, exponent 0 or 255, every mantissa
        const top: u32 = if (i & 1 << 23 != 0) 0x7f80_0000 else 0;
        try old.kept(@as(u32, @intCast(i >> 24)) << 31 | top | @as(u32, @intCast(i & 0x7f_ffff)));
    }
    for (0..1 << 16) |hi| for ([_]u32{ 0, 1, 0x7fff, 0x8000, 0x8001, 0xffff }) |lo| try old.kept(@as(u32, @intCast(hi)) << 16 | lo);
    var prng = std.Random.DefaultPrng.init(1863);
    for (0..1 << 22) |_| try old.kept(prng.random().int(u32));
}
