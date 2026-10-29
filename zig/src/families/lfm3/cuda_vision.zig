//! LFM3-VL's vision tower and projector on CUDA: bf16 weights as stored, one picture's patches to its language rows.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const vi = @import("vision_image.zig");

const vit = cuda.vit;
const H = vi.hidden;
const D = H / vi.heads;
const kvw = vi.kv_heads * D;
const F = vi.intermediate;
/// A patch row: channels, frames, 16 x 16 pixels.
pub const patch_width = 3 * vi.temporal * vi.patch * vi.patch;
/// A projected row: the language model's width.
pub const out_width = 2048;
const merged = H * vi.merge * vi.merge;
const projector_hidden = 4096;
const eps: f32 = 1e-5;
const projector_eps: f32 = 1e-6;
const rope_theta: f64 = 1e6;
/// Which layers attend; the rest are gated depthwise convolutions.
const attends = [vi.layers]bool{ false, false, true, false, true, false, true, false, true, false, true, false, true, false };

pub const Layer = struct {
    operator_norm: u64,
    ffn_norm: u64,
    /// conv: in_proj [3 H, H] (B | C | x); attention: q | k | v [H + 2 kvw, H]
    mix_in: u64,
    out_proj: u64,
    taps: u64 = 0, // conv: [H, 9]
    q_norm: u64 = 0,
    k_norm: u64 = 0,
    gate_up: u64, // w1 | w3 [2 F, H]
    down: u64,
};

pub const Tower = struct {
    gpa: std.mem.Allocator,
    k: *const kern.Kernels,
    vit: vit.Kernels,
    attn: vit.Attention,
    gemms: vit.Gemms,
    patchifier: u64 = 0,
    embedding_norm: u64 = 0,
    layers: [vi.layers]Layer = undefined,
    ln_w: u64 = 0,
    ln_b: u64 = 0,
    l1_w: u64 = 0,
    l1_b: u64 = 0,
    l2_w: u64 = 0,
    l2_b: u64 = 0,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Device bytes the weights hold.
    weight_bytes: u64 = 0,
    most: usize = 0,
    s: Scratch = undefined,

    const Scratch = struct { patches: u64, h: u64, x: u64, r: u64, wide: u64, y: u64, qkv: u64, att: u64, acc: u64, cos: u64, sin: u64, pos: u64 };

    /// The checkpoint's vision_tower.* and multi_modal_projector.* (138 bf16 tensors) and scratch for `most` patches.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, k: *const kern.Kernels, dir: []const u8, most: usize) !Tower {
        var t: Tower = .{ .gpa = gpa, .k = k, .vit = try vit.Kernels.load(d), .attn = undefined, .gemms = undefined, .most = most };
        errdefer t.vit.unload();
        t.attn = try vit.Attention.load(d);
        errdefer t.attn.unload();
        t.gemms = try vit.Gemms.init(gpa, d);
        errdefer t.gemms.deinit();
        errdefer t.buffers.deinit(gpa);
        errdefer for (t.buffers.items) |*b| b.free();
        try t.load(io, d, dir);
        const p = most;
        t.s = .{
            .patches = try t.alloc(d, p * patch_width * 2),
            .h = try t.alloc(d, p * H * 2),
            .x = try t.alloc(d, p * H * 2),
            .r = try t.alloc(d, p * H * 2),
            .wide = try t.alloc(d, p * 2 * F * 2),
            .y = try t.alloc(d, p * F * 2),
            .qkv = try t.alloc(d, p * 3 * H * 2),
            .att = try t.alloc(d, p * H * 2),
            .acc = try t.alloc(d, p / 4 * projector_hidden * 4),
            .cos = try t.alloc(d, p * D * 4),
            .sin = try t.alloc(d, p * D * 4),
            .pos = try t.alloc(d, p * 4),
        };
        return t;
    }

    pub fn deinit(t: *Tower) void {
        for (t.buffers.items) |*b| b.free();
        t.buffers.deinit(t.gpa);
        t.gemms.deinit();
        t.attn.unload();
        t.vit.unload();
    }

    fn alloc(t: *Tower, d: *const cuda.Driver, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(d, @max(bytes, 256));
        errdefer {
            var x = b;
            x.free();
        }
        try t.buffers.append(t.gpa, b);
        return b.ptr;
    }

    fn upload(t: *Tower, d: *const cuda.Driver, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(d, bytes);
        errdefer {
            var x = b;
            x.free();
        }
        try t.buffers.append(t.gpa, b);
        t.weight_bytes += bytes.len;
        return b.ptr;
    }

    /// bf16 tensors stacked along their first dimension into one buffer, each checked by name and shape.
    fn stacked(t: *Tower, d: *const cuda.Driver, ck: *core.checkpoint.Checkpoint, names: []const []const u8, shapes: []const []const usize) !u64 {
        var all: std.ArrayList(u8) = .empty;
        defer all.deinit(t.gpa);
        for (names, shapes) |n, sh| try all.appendSlice(t.gpa, (try ck.expect(n, .bf16, sh)).bytes);
        return t.upload(d, all.items);
    }

    fn load(t: *Tower, io: std.Io, d: *const cuda.Driver, dir: []const u8) !void {
        var ck = try core.checkpoint.Checkpoint.openModel(t.gpa, io, dir);
        defer ck.close();
        var b: [3][160]u8 = undefined;
        const pv = "vision_tower.";
        t.patchifier = try t.stacked(d, &ck, &.{pv ++ "patchifier.weight"}, &.{&.{ H, patch_width }});
        t.embedding_norm = try t.stacked(d, &ck, &.{pv ++ "embedding_norm.weight"}, &.{&.{H}});
        for (&t.layers, attends, 0..) |*l, attention, i| {
            const name = struct {
                fn f(buf: []u8, layer: usize, suffix: []const u8) ![]const u8 {
                    return std.fmt.bufPrint(buf, pv ++ "layers.{d}.{s}", .{ layer, suffix });
                }
            }.f;
            const one = struct {
                fn f(tt: *Tower, dd: *const cuda.Driver, c: *core.checkpoint.Checkpoint, buf: []u8, layer: usize, suffix: []const u8, shape: []const usize) !u64 {
                    return tt.stacked(dd, c, &.{try name(buf, layer, suffix)}, &.{shape});
                }
            }.f;
            l.operator_norm = try one(t, d, &ck, &b[0], i, "operator_norm.weight", &.{H});
            l.ffn_norm = try one(t, d, &ck, &b[0], i, "ffn_norm.weight", &.{H});
            if (attention) {
                l.mix_in = try t.stacked(d, &ck, &.{ try name(&b[0], i, "self_attn.q_proj.weight"), try name(&b[1], i, "self_attn.k_proj.weight"), try name(&b[2], i, "self_attn.v_proj.weight") }, &.{ &.{ H, H }, &.{ kvw, H }, &.{ kvw, H } });
                l.out_proj = try one(t, d, &ck, &b[0], i, "self_attn.out_proj.weight", &.{ H, H });
                l.q_norm = try one(t, d, &ck, &b[0], i, "self_attn.q_layernorm.weight", &.{D});
                l.k_norm = try one(t, d, &ck, &b[0], i, "self_attn.k_layernorm.weight", &.{D});
            } else {
                l.mix_in = try one(t, d, &ck, &b[0], i, "conv.in_proj.weight", &.{ 3 * H, H });
                l.out_proj = try one(t, d, &ck, &b[0], i, "conv.out_proj.weight", &.{ H, H });
                // MLX stores the taps [H, 3, 3, 1], torch [H, 1, 3, 3]: the same [H, 9] values either way
                const taps = try ck.get(try name(&b[0], i, "conv.spatial_conv.weight"));
                if (!taps.is(.bf16, &.{ H, 3, 3, 1 }) and !taps.is(.bf16, &.{ H, 1, 3, 3 })) return error.UnexpectedTensor;
                l.taps = try t.upload(d, taps.bytes);
            }
            l.gate_up = try t.stacked(d, &ck, &.{ try name(&b[0], i, "feed_forward.w1.weight"), try name(&b[1], i, "feed_forward.w3.weight") }, &.{ &.{ F, H }, &.{ F, H } });
            l.down = try one(t, d, &ck, &b[0], i, "feed_forward.w2.weight", &.{ H, F });
        }
        const pp = "multi_modal_projector.";
        t.ln_w = try t.stacked(d, &ck, &.{pp ++ "layer_norm.weight"}, &.{&.{H}});
        t.ln_b = try t.stacked(d, &ck, &.{pp ++ "layer_norm.bias"}, &.{&.{H}});
        t.l1_w = try t.stacked(d, &ck, &.{pp ++ "linear_1.weight"}, &.{&.{ projector_hidden, merged }});
        t.l1_b = try t.stacked(d, &ck, &.{pp ++ "linear_1.bias"}, &.{&.{projector_hidden}});
        t.l2_w = try t.stacked(d, &ck, &.{pp ++ "linear_2.weight"}, &.{&.{ out_width, projector_hidden }});
        t.l2_b = try t.stacked(d, &ck, &.{pp ++ "linear_2.bias"}, &.{&.{out_width}});
    }

    /// One picture's bf16 patch rows (merge-block order, [gh gw, patch_width]) to `out` [gh gw / 4, out_width] bf16.
    pub fn encode(t: *Tower, d: *const cuda.Driver, s: cuda.Stream, pixels: []const u16, gh: usize, gw: usize, out: u64) !void {
        const n = gh * gw;
        if (n == 0 or n > t.most or gh % vi.merge != 0 or gw % vi.merge != 0 or pixels.len != n * patch_width) return error.Lfm3ImageGrid;
        const x = t.s;
        const tab = try t.gpa.alloc(f32, 2 * n * D);
        defer t.gpa.free(tab);
        axial(gh, gw, tab[0 .. n * D], tab[n * D ..]);
        const pos = try t.gpa.alloc(i32, n);
        defer t.gpa.free(pos);
        for (pos, 0..) |*p, i| p.* = @intCast(i);
        const up = struct {
            fn to(dv: *const cuda.Driver, ptr: u64, bytes: []const u8) !void {
                try (cuda.DeviceBuffer{ .d = dv, .ptr = ptr, .len = bytes.len, .borrowed = true }).upload(0, bytes);
            }
        }.to;
        try up(d, x.patches, std.mem.sliceAsBytes(pixels));
        try up(d, x.cos, std.mem.sliceAsBytes(tab[0 .. n * D]));
        try up(d, x.sin, std.mem.sliceAsBytes(tab[n * D ..]));
        try up(d, x.pos, std.mem.sliceAsBytes(pos));
        const k = t.k;
        const round: cuda.norms.Tail = .{ .round = true };
        try t.gemms.run(s, x.patches, t.patchifier, x.r, n, H, patch_width, .bf16);
        try k.norms.addRmsTail(s, x.r, null, t.embedding_norm, x.x, x.h, null, n, H, eps, round); // h = the norm
        for (t.layers, attends) |l, attention| {
            try k.norms.addRmsTail(s, x.h, null, l.operator_norm, x.r, x.x, null, n, H, eps, round);
            if (attention) try t.attend(s, l, n) else try t.conv(s, l, gh, gw);
            // h += the mixer's rows, then the FFN on its norm, h += that
            try k.norms.addRmsTail(s, x.r, x.h, l.ffn_norm, x.h, x.x, null, n, H, eps, round);
            try t.gemms.run(s, x.x, l.gate_up, x.wide, n, 2 * F, H, .bf16);
            try k.siluMul(s, x.wide, x.y, n, F);
            try t.gemms.run(s, x.y, l.down, x.r, n, H, F, .bf16);
            try t.vit.pointwise(s, x.r, x.h, x.h, n * H, .none);
        }
        // the projector: LayerNorm, each merge block's four rows as one, Linear, exact GELU, Linear
        try t.vit.layerNorm(s, x.h, t.ln_w, t.ln_b, x.x, n, H, projector_eps);
        const m = n / (vi.merge * vi.merge);
        try vit.linear(&t.gemms, &t.vit, s, x.acc, x.x, t.l1_w, t.l1_b, x.wide, m, projector_hidden, merged);
        try t.vit.pointwise(s, x.wide, null, x.wide, m * projector_hidden, .gelu);
        try vit.linear(&t.gemms, &t.vit, s, x.acc, x.wide, t.l2_w, t.l2_b, out, m, out_width, projector_hidden);
    }

    /// x's rows through in_proj, the gated 3 x 3 depthwise conv over the grid and out_proj into r.
    fn conv(t: *Tower, s: cuda.Stream, l: Layer, gh: usize, gw: usize) !void {
        const x = t.s;
        const n = gh * gw;
        try t.gemms.run(s, x.x, l.mix_in, x.wide, n, 3 * H, H, .bf16);
        try t.vit.dwconv3(s, x.wide, l.taps, x.y, gh, gw, H, vi.merge);
        try t.gemms.run(s, x.y, l.out_proj, x.r, n, H, H, .bf16);
    }

    /// x through q | k | v, norms and axial RoPE into qkv [n, 3 H] (each KV head twice: 16 heads), attention, out_proj.
    fn attend(t: *Tower, s: cuda.Stream, l: Layer, n: usize) !void {
        const x = t.s;
        const ld: i64 = H + 2 * kvw;
        try t.gemms.run(s, x.x, l.mix_in, x.wide, n, H + 2 * kvw, H, .bf16);
        const src: cuda.qk_rows.Src = .{ .q = x.wide, .q_ld = ld, .q_hs = D, .k = x.wide + H * 2, .k_ld = ld, .k_hs = D, .v = x.wide + (H + kvw) * 2, .v_ld = ld, .v_hs = D };
        const shape: cuda.qk_rows.Shape = .{ .norms = l.q_norm, .norm_k = l.k_norm, .norms_bf16 = true, .cos = x.cos, .sin = x.sin, .rot = D, .eps = eps, .round = true };
        const group = vi.heads / vi.kv_heads;
        for (0..group) |g| {
            const o: cuda.qk_rows.Out = .{ .q = x.qkv, .k = x.qkv + (H + g * D) * 2, .v = x.qkv + (2 * H + g * D) * 2, .q_ld = 3 * H, .q_hs = D, .kv_ld = 3 * H, .kv_hs = group * D };
            try t.k.qk.rowsFrom(s, src, .{ .h = vi.heads, .hk = vi.kv_heads, .dim = D }, shape, x.pos, null, o, n);
        }
        try t.attn.full(s, x.qkv, x.att, n, vi.heads, D, @floatCast(1.0 / @sqrt(@as(f64, D))));
        try t.gemms.run(s, x.att, l.out_proj, x.r, n, H, H, .bf16);
    }
};

/// Axial RoPE fp32 cos/sin rows, merge-block order: quarters [h, w, h, w], inv 1 / theta^(2i / (D / 2)), f64 once.
pub fn axial(gh: usize, gw: usize, cos: []f32, sin: []f32) void {
    const q = D / 4;
    var inv: [q]f64 = undefined;
    for (&inv, 0..) |*v, i| v.* = 1.0 / std.math.pow(f64, rope_theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, D / 2));
    var r: usize = 0;
    for (0..gh / vi.merge) |by| for (0..gw / vi.merge) |bx| for (0..vi.merge) |my| for (0..vi.merge) |mx| {
        const hw = [2]f64{ @floatFromInt(by * vi.merge + my), @floatFromInt(bx * vi.merge + mx) };
        for (0..D) |dd| {
            const i = dd % (D / 2);
            const a = hw[i / q] * inv[i % q];
            cos[r * D + dd] = @floatCast(@cos(a));
            sin[r * D + dd] = @floatCast(@sin(a));
        }
        r += 1;
    };
}

test "axial rows: a merge block's second row is one column right, h angles then w angles, the half repeated" {
    var cos: [16 * D]f32 = undefined;
    var sin: [16 * D]f32 = undefined;
    axial(4, 4, &cos, &sin);
    try std.testing.expectEqual(@as(f32, 0), sin[0]); // row 0: (0, 0)
    try std.testing.expectEqual(@as(f32, 1), cos[D - 1]);
    try std.testing.expectEqual(@as(f32, 0), sin[1 * D]); // row 1: (0, 1), its h angle 0
    try std.testing.expect(sin[1 * D + D / 4] != 0); // and its w angle not
    try std.testing.expectEqual(sin[1 * D + D / 4], sin[1 * D + D / 2 + D / 4]);
    try std.testing.expectEqual(sin[2 * D], sin[3 * D]); // rows 2 (1, 0) and 3 (1, 1) share h
    try std.testing.expectEqual(sin[4 * D + D / 4], sin[6 * D + D / 4]); // rows 4 (0, 2) and 6 (1, 2) share w
}
