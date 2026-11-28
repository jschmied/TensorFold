//! Kolibri-1's checkpoint on Metal: each shard's mapping as one no-copy buffer, every tensor a view at its offset.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const blocks = @import("metal_blocks");
const Config = @import("config.zig").Config;
const names = @import("names.zig");

const Ref = blocks.block_rows.Ref;
const ExpertAt = blocks.block_rows.ExpertAt;

pub const Layer = struct {
    input_norm: Ref,
    q: [2]Ref, // codes, scales
    k: [2]Ref,
    v: [2]Ref,
    q_norm: Ref,
    k_norm: Ref,
    o: [2]Ref,
    post_attn_norm: Ref,
    pre_moe_norm: Ref,
    router: Ref,
    bias: Ref, // fp32 [E], widened at load (the select reads fp32)
    experts: blocks.moe_sigmoid.Experts, // E routed, then the shared one
    post_moe_norm: Ref,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    config: Config,
    ck: core.checkpoint.Checkpoint,
    shards: []mtl.Buffer, // one a shard file, over its mapping
    owned: std.ArrayList(mtl.Buffer) = .empty, // the experts' offset tables and the widened biases
    embed: Ref = undefined,
    norm: Ref = undefined,
    head: Ref = undefined,
    layers: []Layer = &.{},

    pub fn deinit(w: *Weights) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(w.gpa);
        for (w.shards) |b| b.deinit();
        w.gpa.free(w.shards);
        w.gpa.free(w.layers);
        w.ck.close();
        w.* = undefined;
    }

    /// The shard buffer and offset holding `t`, checked 16-byte aligned (the kernels' word loads).
    pub fn view(w: *const Weights, t: core.checkpoint.Tensor) !Ref {
        const at = @intFromPtr(t.bytes.ptr);
        for (w.ck.files.items, w.shards) |f, b| {
            const base = @intFromPtr(f.map.memory.ptr);
            if (at < base or at + t.bytes.len > base + f.map.memory.len) continue;
            if ((at - base) % 16 != 0) return error.MisalignedTensor;
            return .{ .buf = b, .off = at - base };
        }
        return error.TensorOutsideShards;
    }

    fn proj(w: *const Weights, p: ?names.Proj) ![2]Ref {
        const x = p orelse return error.KvSourceLayersNotOnMetal;
        return .{ try w.view(x.w), try w.view(x.s orelse return error.Bf16AttentionNotOnMetal) };
    }

    fn keep(w: *Weights, bytes: []const u8) !mtl.Buffer {
        const b = try w.device.buffer(@max(bytes.len, 16), mtl.ResourceOptions.shared);
        errdefer b.deinit();
        @memcpy(b.contents()[0..bytes.len], bytes);
        try w.owned.append(w.gpa, b);
        return b;
    }

    /// A layer's experts: every tensor in one shard (the kernels take one weights buffer), offsets into it.
    fn experts(w: *Weights, l: names.Layer, width: usize) !blocks.moe_sigmoid.Experts {
        const at = try w.gpa.alloc(ExpertAt, l.experts.len);
        defer w.gpa.free(at);
        const first = try w.view(l.experts[0].gate.w);
        for (l.experts, at) |e, *o| {
            const v = [_]Ref{ try w.view(e.gate.w), try w.view(e.gate.s), try w.view(e.up.w), try w.view(e.up.s), try w.view(e.down.w), try w.view(e.down.s) };
            for (v) |r| if (r.buf.id != first.buf.id) return error.ExpertsAcrossShards;
            o.* = .{ .gate = v[0].off, .gate_s = v[1].off, .up = v[2].off, .up_s = v[3].off, .down = v[4].off, .down_s = v[5].off };
        }
        return .{ .w = .{ .buf = first.buf }, .at = .{ .buf = try w.keep(std.mem.sliceAsBytes(at)) }, .width = width };
    }
};

/// Every tensor checked by names.zig, each shard mapped once; copied: only expert offset tables and fp32 biases.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    const c = try Config.read(gpa, io, dir);
    if (c.v11) return error.Kolibri11NotOnMetal;
    var w: Weights = .{ .gpa = gpa, .device = device, .config = c, .ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir), .shards = &.{} };
    errdefer w.ck.close();
    var m = try names.map(gpa, &w.ck, c);
    defer m.deinit();
    w.shards = try gpa.alloc(mtl.Buffer, w.ck.files.items.len);
    var made: usize = 0;
    errdefer {
        for (w.shards[0..made]) |b| b.deinit();
        gpa.free(w.shards);
    }
    const page = std.heap.pageSize();
    for (w.ck.files.items, w.shards) |f, *b| {
        const len = (f.map.memory.len + page - 1) / page * page;
        b.* = try device.bufferNoCopy(@constCast(f.map.memory.ptr), len, mtl.ResourceOptions.shared);
        made += 1;
    }
    w.layers = try gpa.alloc(Layer, c.layers);
    errdefer gpa.free(w.layers);
    w.embed = try w.view(m.embed);
    w.norm = try w.view(m.norm);
    w.head = try w.view(m.head);
    const bias = try gpa.alloc(f32, c.experts);
    defer gpa.free(bias);
    for (w.layers, m.layers) |*out, l| {
        for (bias, 0..) |*b, e| b.* = @floatCast(@import("ref.zig").bf16At(l.bias.?, e));
        out.* = .{
            .input_norm = try w.view(l.input_norm),
            .q = try w.proj(l.q),
            .k = try w.proj(l.k),
            .v = try w.proj(l.v),
            .q_norm = try w.view(l.q_norm),
            .k_norm = try w.view(l.k_norm orelse return error.KvSourceLayersNotOnMetal),
            .o = try w.proj(l.o),
            .post_attn_norm = try w.view(l.post_attn_norm),
            .pre_moe_norm = try w.view(l.pre_moe_norm),
            .router = try w.view(l.router orelse return error.DenseLayersNotOnMetal),
            .bias = .{ .buf = try w.keep(std.mem.sliceAsBytes(bias)) },
            .experts = try w.experts(l, c.moe_width),
            .post_moe_norm = try w.view(l.post_moe_norm),
        };
    }
    return w;
}
