//! One layer in weights.load's device layouts, packed on the host: lane-matmul FP8 tiles, stacked experts, fp32 norms.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const names = @import("names.zig");

const Tensor = core.checkpoint.Tensor;
const fp8 = cuda.fp8;
const fx8 = cuda.fp8_experts;

/// Fp8BlockLinear.from_checkpoint's arrays: codes [npad/64][k/64][8][32][2][8] and fp32 scale tiles.
pub const Linear = struct { w8: []u8, bs: []f32, n: usize, k: usize };

pub const Layer = struct {
    input_norm: []f32,
    qkv: Linear, // q, k and v rows stacked, as attention reads them in one call
    q_norm: []f32,
    k_norm: []f32,
    o: Linear,
    post_attn_norm: []f32,
    pre_moe_norm: []f32,
    router: []u8, // [E, D] bf16 as stored
    bias: []f32,
    up: []u32, // fp8_experts.Layer.up: [E][NI/32][D/32][2][32][2][4]
    down: []u32, // [E][D/32][NI/32][1][32][2][4]
    up_scale: []f32, // [E][2][NI/128][D/128] (gate, up)
    down_scale: []f32, // [E][1][D/128][NI/128]

    pub fn deinit(l: *Layer, gpa: std.mem.Allocator) void {
        inline for (.{ "input_norm", "q_norm", "k_norm", "post_attn_norm", "pre_moe_norm", "bias", "up_scale", "down_scale" }) |f|
            gpa.free(@field(l, f));
        for ([_]Linear{ l.qkv, l.o }) |x| {
            gpa.free(x.w8);
            gpa.free(x.bs);
        }
        gpa.free(l.router);
        gpa.free(l.up);
        gpa.free(l.down);
        l.* = undefined;
    }
};

/// `.float()` of a bf16 tensor: exact.
pub fn widen(gpa: std.mem.Allocator, t: Tensor) ![]f32 {
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    const out = try gpa.alloc(f32, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
    return out;
}

/// fp32 values of a tensor whose bytes may sit at any offset of the mapped file.
fn floats(dst: []f32, t: Tensor) void {
    for (dst, 0..) |*o, i| o.* = @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
}

/// Row-stacked block-FP8 parts as one projection (torch.cat of codes and of scales, then from_checkpoint).
fn linear(gpa: std.mem.Allocator, parts: []const names.Fp8) !Linear {
    const k = parts[0].w.dim(1);
    var n: usize = 0;
    var blocks: usize = 0;
    for (parts) |p| {
        if (p.w.dim(1) != k or p.w.dim(0) % names.block != 0) return error.UnexpectedTensor;
        n += p.w.dim(0);
        blocks += p.s.numel();
    }
    const codes = try gpa.alloc(u8, n * k);
    defer gpa.free(codes);
    const inv = try gpa.alloc(f32, blocks);
    defer gpa.free(inv);
    var at: usize = 0;
    var sat: usize = 0;
    for (parts) |p| {
        @memcpy(codes[at..][0..p.w.bytes.len], p.w.bytes);
        at += p.w.bytes.len;
        floats(inv[sat..][0..p.s.numel()], p.s);
        sat += p.s.numel();
    }
    const npad = fp8.padded(n);
    const w8 = try gpa.alloc(u8, npad * k);
    errdefer gpa.free(w8);
    fp8.packCodes(w8, codes, n, k);
    const bs = try gpa.alloc(f32, npad * (k / fp8.group));
    fp8.packScales(bs, inv, n, k);
    return .{ .w8 = w8, .bs = bs, .n = n, .k = k };
}

/// fp8x.make: gate and up interleaved, down alone, each expert's scales stacked (gate then up).
fn experts(gpa: std.mem.Allocator, l: *Layer, ex: []const names.Expert, c: Config) !void {
    const ni = c.moe_width;
    const d = c.hidden;
    const one = ni * d;
    const sb = (ni / names.block) * (d / names.block);
    l.up = try gpa.alloc(u32, ex.len * one / 2);
    l.down = try gpa.alloc(u32, ex.len * one / 4);
    l.up_scale = try gpa.alloc(f32, ex.len * 2 * sb);
    l.down_scale = try gpa.alloc(f32, ex.len * sb);
    for (ex, 0..) |e, i| {
        try fx8.packGateUp(l.up[i * one / 2 ..][0 .. one / 2], e.gate.w.bytes, e.up.w.bytes, ni, d);
        fx8.packOne(l.down[i * one / 4 ..][0 .. one / 4], e.down.w.bytes, d, ni);
        floats(l.up_scale[2 * i * sb ..][0..sb], e.gate.s);
        floats(l.up_scale[(2 * i + 1) * sb ..][0..sb], e.up.s);
        floats(l.down_scale[i * sb ..][0..sb], e.down.s);
    }
}

/// weights.load's Layer for one mapped layer.
pub fn layer(gpa: std.mem.Allocator, m: names.Layer, c: Config) !Layer {
    var l: Layer = undefined;
    l.qkv = try linear(gpa, &.{ m.q, m.k, m.v });
    l.o = try linear(gpa, &.{m.o});
    l.input_norm = try widen(gpa, m.input_norm);
    l.q_norm = try widen(gpa, m.q_norm);
    l.k_norm = try widen(gpa, m.k_norm);
    l.post_attn_norm = try widen(gpa, m.post_attn_norm);
    l.pre_moe_norm = try widen(gpa, m.pre_moe_norm);
    l.bias = try widen(gpa, m.bias);
    l.router = try gpa.dupe(u8, m.router.bytes);
    try experts(gpa, &l, m.experts, c);
    return l;
}

const testing = std.testing;

/// `<dir>/<name>.bin` from oracle/kolibri_layer.py.
fn fixture(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}.bin", .{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(1 << 32));
}

test "layer 0 packed as weights.load lays it out (TF_KOLIBRI_DIR, TF_KOLIBRI_LAYER0)" {
    const dir = testing.environ.getPosix("TF_KOLIBRI_DIR") orelse return error.SkipZigTest;
    const fx = testing.environ.getPosix("TF_KOLIBRI_LAYER0") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const cfg = try Config.read(a, testing.io, dir);
    var ck = try core.checkpoint.Checkpoint.openModel(a, testing.io, dir);
    defer ck.close();
    var m = try names.map(a, &ck, cfg);
    defer m.deinit();
    var l = try layer(a, m.layers[0], cfg);
    defer l.deinit(a);
    const got = .{
        .{ "qkv_w8", std.mem.sliceAsBytes(l.qkv.w8) },             .{ "qkv_bs", std.mem.sliceAsBytes(l.qkv.bs) },
        .{ "o_w8", std.mem.sliceAsBytes(l.o.w8) },                 .{ "o_bs", std.mem.sliceAsBytes(l.o.bs) },
        .{ "up", std.mem.sliceAsBytes(l.up) },                     .{ "down", std.mem.sliceAsBytes(l.down) },
        .{ "up_scale", std.mem.sliceAsBytes(l.up_scale) },         .{ "down_scale", std.mem.sliceAsBytes(l.down_scale) },
        .{ "input_norm", std.mem.sliceAsBytes(l.input_norm) },     .{ "q_norm", std.mem.sliceAsBytes(l.q_norm) },
        .{ "k_norm", std.mem.sliceAsBytes(l.k_norm) },             .{ "post_attn_norm", std.mem.sliceAsBytes(l.post_attn_norm) },
        .{ "pre_moe_norm", std.mem.sliceAsBytes(l.pre_moe_norm) }, .{ "router", l.router },
        .{ "bias", std.mem.sliceAsBytes(l.bias) },
    };
    inline for (got) |g| {
        const want = try fixture(a, fx, g[0]);
        defer a.free(want);
        if (!std.mem.eql(u8, want, g[1])) {
            const at = std.mem.indexOfDiff(u8, want, g[1]).?;
            std.debug.print("layer 0 {s}: {d} bytes, want {d}; first difference at {d}\n", .{ g[0], g[1].len, want.len, at });
            return error.TestUnexpectedResult;
        }
    }
}
