//! Kolibri-1's whole forward in float64 over every row of a prompt, rounded where the engine rounds (decode points).
const std = @import("std");
const core = @import("core");
const names = @import("names.zig");
const Config = @import("config.zig").Config;

const Tensor = core.checkpoint.Tensor;
const e4m3 = core.quant.fp8.e4m3;

/// Rows after each layer's attention half and its MoE half (fp32 residuals), then the logits of every row.
pub const Out = struct {
    attn: [][]f64, // [layer][rows * D]
    moe: [][]f64,
    logits: []f64, // [rows * V]
};

pub fn bf16At(t: Tensor, i: usize) f64 {
    return @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16));
}

fn f32At(t: Tensor, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)));
}

/// x rounded to bf16, nearest even (through fp32 first, as a kernel's fp32 value would be stored).
pub fn roundBf16(x: f64) f64 {
    const b: u32 = @bitCast(@as(f32, @floatCast(x)));
    return @as(f32, @bitCast((b +% 0x7fff +% ((b >> 16) & 1)) & 0xffff0000));
}

pub fn roundF32(x: f64) f64 {
    return @as(f32, @floatCast(x));
}

/// out[j] = sum over 128-column blocks of (sum of x * code) * the block's scale: the kernels' order of operations.
fn fp8Rows(lut: *const [256]f64, w: Tensor, s: Tensor, x: []const f64, out: []f64) void {
    const k = w.dim(1);
    const kb = (k + 127) / 128;
    for (out, 0..) |*o, j| {
        const row = w.bytes[j * k ..][0..k];
        var acc: f64 = 0;
        for (0..kb) |b| {
            var part: f64 = 0;
            const end = @min(k, b * 128 + 128);
            for (b * 128..end) |c| part += x[c] * lut[row[c]];
            acc += part * f32At(s, (j / 128) * kb + b);
        }
        o.* = acc;
    }
}

fn rms(x: []const f64, w: Tensor, eps: f64, out: []f64) void {
    var ss: f64 = 0;
    for (x) |v| ss += v * v;
    const r = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    for (out, x, 0..) |*o, v, i| o.* = v * r * bf16At(w, i);
}

const Ctx = struct {
    m: *const names.Map,
    c: Config,
    lut: [256]f64,
    layer: usize,
    n: usize,
    tokens: []const u32,
    res: []f64, // [n, D] fp32 values
    x: []f64, // [n, D] bf16 values
    q: []f64, // [n, H, Dh]
    k: []f64, // [n, HK, Dh]
    v: []f64,
    att: []f64, // [n, H * Dh]
    tmp: []f64, // per worker scratch
    tmp_len: usize,
};

/// Calls f(ctx, row, scratch) for every row over the machine's cores.
fn rows(ctx: *Ctx, comptime f: fn (*Ctx, usize, []f64) void) !void {
    const workers = @max(1, @min(std.Thread.getCpuCount() catch 1, ctx.tmp.len / ctx.tmp_len));
    const Worker = struct {
        fn run(cx: *Ctx, id: usize, count: usize) void {
            var r = id;
            while (r < cx.n) : (r += count) f(cx, r, cx.tmp[id * cx.tmp_len ..][0..cx.tmp_len]);
        }
    };
    var threads: [64]std.Thread = undefined;
    const count = @min(workers, threads.len);
    for (0..count) |i| threads[i] = try std.Thread.spawn(.{}, Worker.run, .{ ctx, i, count });
    for (threads[0..count]) |t| t.join();
}

/// q, k and v of one row: projections, per-head norms, RoPE on sliding layers (fp32 tables), each rounded to bf16.
fn qkvRow(cx: *Ctx, r: usize, t: []f64) void {
    const c = cx.c;
    const L = cx.m.layers[cx.layer];
    const d = c.hidden;
    const dh = c.head_dim;
    const qkv = t[0..c.qkvDim()];
    const x = cx.x[r * d ..][0..d];
    fp8Rows(&cx.lut, L.q.w, L.q.s.?, x, qkv[0 .. c.heads * dh]);
    fp8Rows(&cx.lut, L.k.?.w, L.k.?.s.?, x, qkv[c.heads * dh ..][0 .. c.kv_heads * dh]);
    fp8Rows(&cx.lut, L.v.?.w, L.v.?.s.?, x, qkv[(c.heads + c.kv_heads) * dh ..][0 .. c.kv_heads * dh]);
    for (qkv) |*v| v.* = roundBf16(v.*);
    const full = c.full[cx.layer];
    for (0..c.heads + c.kv_heads) |hd| {
        const dst = if (hd < c.heads) cx.q[(r * c.heads + hd) * dh ..][0..dh] else cx.k[(r * c.kv_heads + hd - c.heads) * dh ..][0..dh];
        const y = t[c.qkvDim()..][0..dh];
        rms(qkv[hd * dh ..][0..dh], if (hd < c.heads) L.q_norm else L.k_norm.?, c.eps, y);
        for (0..dh) |i| {
            if (full) {
                dst[i] = roundBf16(y[i]);
                continue;
            }
            const j = i % (dh / 2);
            const frac = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh));
            const ang = @as(f64, @floatFromInt(r)) / std.math.pow(f64, c.rope_theta, frac);
            const rot = if (i < dh / 2) -y[i + dh / 2] else y[i - dh / 2];
            dst[i] = roundBf16(y[i] * roundF32(@cos(ang)) + rot * roundF32(@sin(ang)));
        }
    }
    @memcpy(cx.v[r * c.kv_heads * dh ..][0 .. c.kv_heads * dh], qkv[(c.heads + c.kv_heads) * dh ..][0 .. c.kv_heads * dh]);
}

/// Row r's attention: keys (r - window, r] on a sliding layer, [0, r] on a full one; softmax in float64.
fn attnRow(cx: *Ctx, r: usize, t: []f64) void {
    const c = cx.c;
    const dh = c.head_dim;
    const lo = if (!c.full[cx.layer] and r + 1 > c.window) r + 1 - c.window else 0;
    const sc = t[0 .. r + 1 - lo];
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(dh)));
    for (0..c.heads) |hd| {
        const g = hd / (c.heads / c.kv_heads);
        const q = cx.q[(r * c.heads + hd) * dh ..][0..dh];
        var most: f64 = -std.math.inf(f64);
        for (sc, lo..) |*s, p| {
            var dot: f64 = 0;
            for (q, cx.k[(p * c.kv_heads + g) * dh ..][0..dh]) |a, b| dot += a * b;
            s.* = dot * scale;
            most = @max(most, s.*);
        }
        var sum: f64 = 0;
        for (sc) |*s| {
            s.* = @exp(s.* - most);
            sum += s.*;
        }
        for (0..dh) |e| {
            var acc: f64 = 0;
            for (sc, lo..) |s, p| acc += s * cx.v[(p * c.kv_heads + g) * dh + e];
            cx.att[(r * c.heads + hd) * dh + e] = roundBf16(acc / sum);
        }
    }
}

/// o, the post-attention norm into the residual, then the MoE half; `x` ends as the next layer's normed input.
fn restRow(cx: *Ctx, r: usize, t: []f64) void {
    const c = cx.c;
    const L = cx.m.layers[cx.layer];
    const d = c.hidden;
    const o = t[0..d];
    const u = t[d..][0..d];
    fp8Rows(&cx.lut, L.o.w, L.o.s.?, cx.att[r * c.heads * c.head_dim ..][0 .. c.heads * c.head_dim], o);
    for (o) |*v| v.* = roundBf16(v.*);
    const res = cx.res[r * d ..][0..d];
    rms(o, L.post_attn_norm, c.eps, u);
    for (res, u) |*v, w| v.* = roundF32(v.* + w);
}

fn moeRow(cx: *Ctx, r: usize, t: []f64) void {
    const c = cx.c;
    const L = cx.m.layers[cx.layer];
    const d = c.hidden;
    const res = cx.res[r * d ..][0..d];
    const x = t[0..d];
    rms(res, L.pre_moe_norm, c.eps, x);
    for (x) |*v| v.* = roundBf16(v.*);
    const out = t[d..][0..d];
    @memset(out, 0);
    moe(cx, L, x, out, t[2 * d ..]);
    for (out) |*v| v.* = roundBf16(v.*);
    rms(out, L.post_moe_norm, c.eps, x);
    for (res, x) |*v, w| v.* = roundF32(v.* + w);
    const next = if (cx.layer + 1 < c.layers) cx.m.layers[cx.layer + 1].input_norm else cx.m.norm;
    rms(res, next, c.eps, cx.x[r * d ..][0..d]);
    for (cx.x[r * d ..][0..d]) |*v| v.* = roundBf16(v.*);
}

/// Top-k by fp32 logit + bias (lower id on ties), weight fp32 sigmoid(logit), unnormalised; the shared expert at 1.
fn moe(cx: *Ctx, L: names.Layer, x: []const f64, out: []f64, t: []f64) void {
    const c = cx.c;
    const d = c.hidden;
    const width = c.moe_width;
    const logit = t[0..c.experts];
    const key = t[c.experts..][0..c.experts];
    const gate = t[2 * c.experts ..][0..width];
    const up = t[2 * c.experts + width ..][0..width];
    const y = t[2 * c.experts + 2 * width ..][0..d];
    for (logit, key, 0..) |*lg, *ky, e| {
        var acc: f64 = 0;
        for (x, 0..) |v, i| acc += v * bf16At(L.router.?, e * d + i);
        lg.* = roundF32(acc);
        ky.* = lg.* + bf16At(L.bias.?, e);
    }
    for (0..c.top_k + 1) |slot| {
        var e: usize = c.experts;
        var w: f64 = 1;
        if (slot < c.top_k) {
            e = std.mem.indexOfMax(f64, key);
            w = roundF32(1.0 / (1.0 + @exp(-logit[e])));
            key[e] = -std.math.inf(f64);
        }
        const ex = L.experts[e];
        fp8Rows(&cx.lut, ex.gate.w, ex.gate.s, x, gate);
        fp8Rows(&cx.lut, ex.up.w, ex.up.s, x, up);
        for (gate, up) |*g, v| g.* = roundBf16(g.* / (1.0 + @exp(-g.*)) * v);
        fp8Rows(&cx.lut, ex.down.w, ex.down.s, gate, y);
        for (out, y) |*o, v| o.* += w * roundF32(v);
    }
}

fn headRow(cx: *Ctx, r: usize, _: []f64) void {
    const c = cx.c;
    const d = c.hidden;
    const x = cx.x[r * d ..][0..d];
    for (cx.att[r * c.vocab ..][0..c.vocab], 0..) |*lg, j| {
        var acc: f64 = 0;
        for (x, 0..) |v, i| acc += v * bf16At(cx.m.head, j * d + i);
        lg.* = acc;
    }
}

/// `tokens` at positions 0.. through every layer of `m` (Kolibri-1, block FP8): taps and every row's logits.
pub fn run(a: std.mem.Allocator, m: *const names.Map, c: Config, tokens: []const u32) !Out {
    if (c.v11) return error.Kolibri11NotCovered;
    const n = tokens.len;
    const d = c.hidden;
    const per = @max(c.qkvDim() + c.head_dim, n, 3 * d + 2 * c.experts + 2 * c.moe_width);
    const workers = @min(std.Thread.getCpuCount() catch 1, 64);
    var cx: Ctx = .{
        .m = m,
        .c = c,
        .lut = undefined,
        .layer = 0,
        .n = n,
        .tokens = tokens,
        .res = try a.alloc(f64, n * d),
        .x = try a.alloc(f64, n * d),
        .q = try a.alloc(f64, n * c.heads * c.head_dim),
        .k = try a.alloc(f64, n * c.kv_heads * c.head_dim),
        .v = try a.alloc(f64, n * c.kv_heads * c.head_dim),
        .att = try a.alloc(f64, n * @max(c.heads * c.head_dim, c.vocab)),
        .tmp = try a.alloc(f64, workers * per),
        .tmp_len = per,
    };
    for (&cx.lut, 0..) |*v, i| v.* = e4m3(@intCast(i));
    for (tokens, 0..) |t, r| {
        if (t >= c.vocab) return error.TokenPastVocab;
        for (0..d) |i| cx.res[r * d + i] = bf16At(m.embed, t * d + i);
        rms(cx.res[r * d ..][0..d], m.layers[0].input_norm, c.eps, cx.x[r * d ..][0..d]);
        for (cx.x[r * d ..][0..d]) |*v| v.* = roundBf16(v.*);
    }
    var out: Out = .{ .attn = try a.alloc([]f64, c.layers), .moe = try a.alloc([]f64, c.layers), .logits = undefined };
    for (0..c.layers) |i| {
        cx.layer = i;
        try rows(&cx, qkvRow);
        try rows(&cx, attnRow);
        try rows(&cx, restRow);
        out.attn[i] = try a.dupe(f64, cx.res);
        try rows(&cx, moeRow);
        out.moe[i] = try a.dupe(f64, cx.res);
    }
    try rows(&cx, headRow);
    out.logits = try a.dupe(f64, cx.att[0 .. n * c.vocab]);
    return out;
}

test "Kolibri reference: a sliding row past the window ignores the keys before it, a full row does not" {
    const fixture = @import("fixture.zig");
    const testing = std.testing;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const s: fixture.Shape = .{ .full = &.{ false, true }, .experts = 3, .top_k = 2, .vocab = 32, .hidden = 128, .heads = 4, .kv_heads = 1, .width = 128, .window = 3, .eos = 31 };
    try fixture.write(testing.allocator, testing.io, tmp.dir, s);
    const cwd = try std.process.currentPathAlloc(testing.io, a);
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    const c = try Config.read(a, testing.io, path);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, testing.io, path);
    defer ck.close();
    var m = try names.map(testing.allocator, &ck, c);
    defer m.deinit();
    const one = try run(a, &m, c, &.{ 5, 7, 9, 11, 13 });
    const two = try run(a, &m, c, &.{ 6, 7, 9, 11, 13 }); // only row 0 differs: outside row 4's window of 3
    const d = c.hidden;
    try testing.expectEqualSlices(f64, one.attn[0][4 * d ..][0..d], two.attn[0][4 * d ..][0..d]);
    try testing.expect(!std.mem.eql(f64, one.attn[1][4 * d ..][0..d], two.attn[1][4 * d ..][0..d]));
    try testing.expect(!std.mem.eql(f64, one.attn[0][2 * d ..][0..d], two.attn[0][2 * d ..][0..d]));
}
