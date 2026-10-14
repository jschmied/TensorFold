//! Kimi Delta Attention in float64, the reference every backend's KDA kernels are checked against.
const std = @import("std");

pub const keep = 3;

/// x rounded to the nearest bf16, ties to even (finite normal values).
pub fn bf16(x: f64) f64 {
    const u: u64 = @bitCast(x);
    const lsb = (u >> 45) & 1;
    return @bitCast((u + ((@as(u64, 1) << 44) - 1) + lsb) & ~((@as(u64, 1) << 45) - 1));
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// A channel's conv over its window (oldest first) and the new raw value, then SiLU, in fp32 as the model's conv runs.
pub fn conv(win: [keep]f64, x: f64, w: []const f64) f64 {
    var acc: f32 = 0;
    for (win, 0..) |p, j| acc = @mulAdd(f32, @floatCast(p), @floatCast(w[j]), acc);
    acc = @mulAdd(f32, @floatCast(x), @floatCast(w[keep]), acc);
    return acc * (1 / (1 + @exp(-acc)));
}

/// A channel's log decay: lower_bound sigmoid(e^A (f + dt_bias)) when lower_bound < 0, else -e^A softplus(f + dt_bias).
pub fn logDecay(f: f64, dt_bias: f64, a_log: f64, lower_bound: f64) f64 {
    const x = f + dt_bias;
    if (lower_bound < 0) return lower_bound * sigmoid(@exp(a_log) * x);
    const sp = if (x > 20) x else std.math.log1p(@exp(x));
    return -@exp(a_log) * sp;
}

/// x / sqrt(sum x^2 + 1e-6) * scale, in place.
pub fn l2norm(x: []f64, scale: f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const f = scale / @sqrt(ss + 1e-6);
    for (x) |*v| v.* *= f;
}

/// One head's delta rule on S [v][k]: decay by e^g, then u = beta (v - S^T k), S += k u^T; o = S^T q.
pub fn step(s: []f64, q: []const f64, k: []const f64, v: []const f64, g: []const f64, beta: f64, o: []f64) void {
    const d = k.len;
    for (0..v.len) |j| {
        const row = s[j * d ..][0..d];
        var mem: f64 = 0;
        for (row, k, g) |*x, kk, gg| {
            x.* *= @exp(gg);
            mem += x.* * kk;
        }
        const u = beta * (v[j] - mem);
        var acc: f64 = 0;
        for (row, k, q) |*x, kk, qq| {
            x.* += kk * u;
            acc += x.* * qq;
        }
        o[j] = acc;
    }
}

/// The gated output norm of a head's row: o / rms * w * sigmoid(gate); tol: o's bound in, y's bound out (first order).
pub fn gatedNorm(o: []const f64, gate: []const f64, w: []const f64, eps: f64, y: []f64, tol: []f64) void {
    var ss: f64 = 0;
    var cross: f64 = 0;
    for (o, tol) |x, t| {
        ss += x * x;
        cross += @abs(x) * t;
    }
    const n: f64 = @floatFromInt(o.len);
    const rstd = 1 / @sqrt(ss / n + eps);
    const moved = cross / n / (ss / n + eps); // rstd's relative change at the most
    for (y, tol, o, gate, w) |*out, *t, x, z, ww| {
        out.* = x * rstd * ww * sigmoid(z);
        t.* = @abs(ww * sigmoid(z)) * rstd * (t.* + @abs(x) * moved) + std.math.pow(f64, 2, -8) * @abs(out.*);
    }
}

/// A head's weights: conv [3 d][4] (its q, k, v channels), A_log, dt_bias [d], the norm [d]; lower_bound 0: softplus.
pub const Weights = struct { conv: []const f64, a_log: f64, dt_bias: []const f64, norm: []const f64, lower_bound: f64, eps: f64 };

/// One (stream, head): S [d v][d k] and each of its 3 d channels' conv window.
pub const Head = struct {
    s: []f64,
    win: [][keep]f64,

    /// A row from raw x [3 d], f [d], beta logit b, gate [d]: y unrounded, tol 2^-7 of o's terms through the norm.
    pub fn row(h: *Head, w: Weights, x: []const f64, f: []const f64, b: f64, gate: []const f64, scratch: []f64, y: []f64, tol: []f64) void {
        const d = f.len;
        const qkv = scratch[0 .. 3 * d];
        for (qkv, x, h.win, 0..) |*c, raw, *win, i| {
            c.* = bf16(conv(win.*, raw, w.conv[4 * i ..][0..4]));
            win.* = .{ win[1], win[2], raw };
        }
        const g = scratch[3 * d .. 4 * d];
        for (g, f, w.dt_bias) |*gg, ff, db| gg.* = logDecay(ff, db, w.a_log, w.lower_bound);
        l2norm(qkv[0..d], 1 / @sqrt(@as(f64, @floatFromInt(d))));
        l2norm(qkv[d .. 2 * d], 1);
        const o = scratch[4 * d .. 5 * d];
        const beta = sigmoid(b);
        step(h.s, qkv[0..d], qkv[d .. 2 * d], qkv[2 * d .. 3 * d], g, beta, o);
        var kq: f64 = 0;
        for (qkv[0..d], qkv[d .. 2 * d]) |q, k| kq += @abs(q * k);
        for (o, tol, 0..) |*e, *t, j| {
            var mag: f64 = beta * @abs(qkv[2 * d + j]) * kq;
            for (h.s[j * d ..][0..d], qkv[0..d]) |sv, q| mag += @abs(sv * q);
            t.* = std.math.pow(f64, 2, -7) * mag;
            e.* = bf16(e.*);
        }
        gatedNorm(o, gate, w.norm, w.eps, y, tol);
    }
};

test "bf16 rounding, the gates and one delta step against hand values" {
    try std.testing.expectEqual(@as(f64, 1.0), bf16(1.0 + 1.0 / 256.0)); // a tie rounds to even
    try std.testing.expectEqual(@as(f64, 1.0 + 1.0 / 64.0), bf16(1.0 + 3.0 / 256.0));
    try std.testing.expectApproxEqRel(@as(f64, -5.0 * sigmoid(@exp(0.5) * 1.25)), logDecay(1, 0.25, 0.5, -5), 1e-15);
    try std.testing.expectApproxEqRel(@as(f64, -@exp(0.5) * 25.0), logDecay(25, 0, 0.5, 0), 1e-15);
    var s = [_]f64{ 1, 0, 0, 1 }; // S [v][k] = I, decay e^0 = 1
    var o: [2]f64 = undefined;
    step(&s, &.{ 1, 0 }, &.{ 0, 1 }, &.{ 3, 5 }, &.{ 0, 0 }, 0.5, &o);
    // u = 0.5 (v - S^T k) = (1.5, 2): S[0] = (1, 1.5), S[1] = (0, 3); o = S q = (1, 0)
    try std.testing.expectEqualSlices(f64, &.{ 1, 1.5, 0, 3 }, &s);
    try std.testing.expectEqualSlices(f64, &.{ 1, 0 }, &o);
}
