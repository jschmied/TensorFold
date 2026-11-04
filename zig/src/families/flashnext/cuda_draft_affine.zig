//! An optional group-32 affine proposal head; the main checkpoint and its verifier remain unchanged.
const std = @import("std");
const host4 = @import("core").affine4_host;

pub fn bf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn value(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

fn code(x: f32) u8 {
    const bounded = std.math.clamp(x, 0, 15);
    const lo = @floor(bounded);
    const integer: u8 = @intFromFloat(lo);
    const fraction = bounded - lo;
    return integer + @intFromBool(fraction > 0.5 or (fraction == 0.5 and integer & 1 == 1));
}

/// Codes use the FP32 range before its BF16 table rounding, matching the Python proposal-head recipe.
pub fn row(input: []const u8, words: []u8, scales: []u8, biases: []u8) !void {
    const n = input.len / 2;
    if (n == 0 or n % 32 != 0 or input.len != n * 2 or words.len != n / 2 or scales.len != n / 16 or biases.len != scales.len)
        return error.UnexpectedTensor;
    for (0..n / 32) |group| {
        const at = group * 32;
        var lo = std.math.inf(f32);
        var hi = -std.math.inf(f32);
        for (0..32) |i| {
            const v = value(input, at + i);
            if (!std.math.isFinite(v)) return error.UnexpectedTensor;
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const scale = @max((hi - lo) / 15.0, 1e-8);
        if (!std.math.isFinite(scale)) return error.UnexpectedTensor;
        std.mem.writeInt(u16, scales[group * 2 ..][0..2], bf16(scale), .little);
        std.mem.writeInt(u16, biases[group * 2 ..][0..2], bf16(lo), .little);
        for (0..16) |j| {
            const low = code((value(input, at + j * 2) - lo) / scale);
            const high = code((value(input, at + j * 2 + 1) - lo) / scale);
            words[group * 16 + j] = low | (high << 4);
        }
    }
}

/// `row`, then each group refit by core's quantizeFit under the input columns' weights `w` (diag(H)).
pub fn rowFit(input: []const u8, words: []u8, scales: []u8, biases: []u8, w: []const f32) !void {
    try row(input, words, scales, biases);
    if (w.len != input.len / 2) return error.UnexpectedTensor;
    for (0..input.len / 64) |group| {
        const at = group * 32;
        var v: [32]f32 = undefined;
        var q: [32]u8 = undefined;
        for (&v, &q, 0..) |*x, *c, i| {
            x.* = value(input, at + i);
            const byte = words[group * 16 + i / 2];
            c.* = if (i & 1 == 0) byte & 15 else byte >> 4;
        }
        var s = std.mem.readInt(u16, scales[group * 2 ..][0..2], .little);
        var b = std.mem.readInt(u16, biases[group * 2 ..][0..2], .little);
        if (!host4.quantizeFit(&v, w[at..][0..32], &q, &s, &b)) continue;
        std.mem.writeInt(u16, scales[group * 2 ..][0..2], s, .little);
        std.mem.writeInt(u16, biases[group * 2 ..][0..2], b, .little);
        for (0..16) |j| words[group * 16 + j] = q[2 * j] | (q[2 * j + 1] << 4);
    }
}

test "proposal codes have known nearest-even ties and a flat row stays on its exact bias" {
    var input: [128]u8 = undefined;
    for (0..64) |i| std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16(if (i < 32) @as(f32, @floatFromInt(@min(i, 30))) / 2 else 7), .little);
    var words: [32]u8 = undefined;
    var scales: [4]u8 = undefined;
    var biases: [4]u8 = undefined;
    try row(&input, &words, &scales, &biases);
    try std.testing.expectEqual(@as(u8, 0x00), words[0]);
    try std.testing.expectEqual(@as(u8, 0x21), words[1]);
    try std.testing.expectEqual(@as(u8, 0x22), words[2]);
    try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, scales[0..2], .little));
    try std.testing.expectEqual(@as(u16, 0x40e0), std.mem.readInt(u16, biases[2..4], .little));
    for (words[16..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    std.mem.writeInt(u16, input[0..2], 0x7f80, .little);
    try std.testing.expectError(error.UnexpectedTensor, row(&input, &words, &scales, &biases));
}

test "proposal approximation is bounded per value against an independent FP64 dequantizer" {
    var input: [128]u8 = undefined;
    for (0..64) |i| std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16(@sin(@as(f32, @floatFromInt(i)) * 0.37) * 0.05), .little);
    var words: [32]u8 = undefined;
    var scales: [4]u8 = undefined;
    var biases: [4]u8 = undefined;
    try row(&input, &words, &scales, &biases);
    for (0..2) |g| {
        var lo = std.math.inf(f64);
        var hi = -std.math.inf(f64);
        for (0..32) |i| {
            const v: f64 = value(&input, g * 32 + i);
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const scale = (hi - lo) / 15;
        const stored_s: f64 = value(&scales, g);
        const stored_b: f64 = value(&biases, g);
        const bound = scale * 0.50001 + 15 * @abs(stored_s - scale) + @abs(stored_b - lo);
        for (0..32) |i| {
            const at = g * 32 + i;
            const q = (words[at / 2] >> @as(u3, @intCast((at % 2) * 4))) & 15;
            const back = @as(f64, @floatFromInt(q)) * stored_s + stored_b;
            try std.testing.expect(@abs(back - value(&input, at)) <= bound);
        }
    }
}

test "the weighted refit lowers a row's weighted output error and keeps its layout" {
    var prng = std.Random.DefaultPrng.init(1870);
    const r = prng.random();
    var input: [256 * 2]u8 = undefined;
    var w: [256]f32 = undefined;
    for (0..256) |i| {
        std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16(r.floatNorm(f32) * 0.02), .little);
        w[i] = @exp(r.floatNorm(f32) * 2);
    }
    var w0: [128]u8 = undefined;
    var s0: [16]u8 = undefined;
    var b0: [16]u8 = undefined;
    try row(&input, &w0, &s0, &b0);
    var w1: [128]u8 = undefined;
    var s1: [16]u8 = undefined;
    var b1: [16]u8 = undefined;
    try rowFit(&input, &w1, &s1, &b1, &w);
    var e: [2]f64 = .{ 0, 0 };
    for (0..256) |i| {
        const g = i / 32;
        const v = value(&input, i);
        const shift: u3 = @intCast((i & 1) * 4);
        const q0: f64 = @floatFromInt((w0[i / 2] >> shift) & 15);
        const q1: f64 = @floatFromInt((w1[i / 2] >> shift) & 15);
        const d0 = q0 * value(&s0, g) + value(&b0, g) - v;
        const d1 = q1 * value(&s1, g) + value(&b1, g) - v;
        e[0] += w[i] * d0 * d0;
        e[1] += w[i] * d1 * d1;
    }
    try std.testing.expect(e[1] < e[0]);
    try std.testing.expectError(error.UnexpectedTensor, rowFit(&input, &w1, &s1, &b1, w[0..32]));
}
