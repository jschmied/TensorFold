//! LFM3-VL's 138 dense BF16 vision tensors, borrowed from the owned checkpoint without conversions.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const spec = @import("vision_spec.zig");

const vision = spec.lfm3_vl_2b;
pub const width = vision.hidden;
pub const intermediate = vision.intermediate;
pub const depth = vision.layers;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Conv = struct { in: Ref, out: Ref, spatial: Ref };
pub const Attention = struct { q: Ref, k: Ref, v: Ref, out: Ref, q_norm: Ref, k_norm: Ref };
pub const Layer = struct {
    operator_norm: Ref,
    ffn_norm: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
    mixer: union(enum) { conv: Conv, attention: Attention },
};
pub const Weights = struct {
    patch: Ref,
    norm: Ref,
    blocks: [depth]Layer,
    projector_norm: Ref,
    projector_norm_bias: Ref,
    linear1: Ref,
    bias1: Ref,
    linear2: Ref,
    bias2: Ref,
};

const Loader = struct {
    a: std.mem.Allocator,
    checkpoint: *const ck.Checkpoint,
    seen: std.StringHashMapUnmanaged(void) = .empty,

    fn get(self: *Loader, layer: ?usize, m: spec.Module) !Ref {
        var buf: [192]u8 = undefined;
        const want = try spec.tensor(&buf, vision, layer, m);
        const name = want.name;
        const shape = want.shape.slice();
        const t = try self.checkpoint.get(name);
        if (t.dtype != .bf16 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape))
            return error.UnexpectedLfm3VisionTensor;
        var bytes: usize = 2;
        for (shape) |dim| bytes = std.math.mul(usize, bytes, dim) catch return error.TensorBytes;
        const end = std.math.add(usize, t.offset, bytes) catch return error.TensorBytes;
        if (t.bytes != bytes or end > t.buffer.length() or t.offset % 4 != 0) return error.TensorBytes;
        try self.seen.put(self.a, try self.a.dupe(u8, name), {});
        return .{ .buf = t.buffer, .off = t.offset };
    }
};

pub fn load(checkpoint: *const ck.Checkpoint) !Weights {
    var arena = std.heap.ArenaAllocator.init(checkpoint.allocator);
    defer arena.deinit();
    var l: Loader = .{ .a = arena.allocator(), .checkpoint = checkpoint };
    var out: Weights = .{
        .patch = try l.get(null, .patchifier),
        .norm = try l.get(null, .embedding_norm),
        .blocks = undefined,
        .projector_norm = try l.get(null, .projector_norm),
        .projector_norm_bias = try l.get(null, .projector_norm_bias),
        .linear1 = try l.get(null, .linear_1),
        .bias1 = try l.get(null, .linear_1_bias),
        .linear2 = try l.get(null, .linear_2),
        .bias2 = try l.get(null, .linear_2_bias),
    };
    for (&out.blocks, 0..) |*block, i| {
        block.operator_norm = try l.get(i, .operator_norm);
        block.ffn_norm = try l.get(i, .ffn_norm);
        block.gate = try l.get(i, .w1);
        block.up = try l.get(i, .w3);
        block.down = try l.get(i, .w2);
        block.mixer = if (vision.mixers[i] == .attention) .{ .attention = .{
            .q = try l.get(i, .q_proj),
            .k = try l.get(i, .k_proj),
            .v = try l.get(i, .v_proj),
            .out = try l.get(i, .out_proj),
            .q_norm = try l.get(i, .q_norm),
            .k_norm = try l.get(i, .k_norm),
        } } else .{ .conv = .{
            .in = try l.get(i, .conv_in),
            .out = try l.get(i, .conv_out),
            .spatial = try l.get(i, .spatial_conv),
        } };
    }
    var it = checkpoint.tensors.keyIterator();
    while (it.next()) |name| {
        if (spec.isVision(name.*) and !l.seen.contains(name.*)) return error.UnexpectedLfm3VisionTensor;
    }
    if (l.seen.count() != spec.count(vision)) return error.UnexpectedLfm3VisionTensor;
    return out;
}
