//! Metadata fixtures exercise admission without allocating tensor payloads or a GPU device.
const std = @import("std");
const w = @import("weights.zig");
const c = @import("config.zig");
const st = @import("safetensors");

fn config(bits: u8, group: u16) c.Config {
    return .{ .arena = std.heap.ArenaAllocator.init(std.testing.allocator), .kind = .text, .context = 262144, .quant = .{ .global = .{ .bits = bits, .group = group } } };
}

fn header(a: std.mem.Allocator, cfg: *const c.Config, namespace: w.Namespace) !st.Header {
    var h: st.Header = .empty;
    var offset: usize = 0;
    for (try w.schema(a, cfg, namespace)) |e| {
        var bytes: usize = e.dtype.size();
        for (e.shape[0..e.rank]) |d| bytes *= d;
        try h.put(a, e.name, .{ .dtype = e.dtype, .rank = e.rank, .shape = e.shape, .begin = offset, .end = offset + bytes });
        offset += bytes;
    }
    return h;
}

test "official text4 tensor count and bytes; both namespaces and every target bit width" {
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]w.Namespace{ .bare, .wrapped }) |namespace| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(bits, 64);
        defer cfg.deinit();
        const h = try header(arena.allocator(), &cfg, namespace);
        const m = try w.validate(std.testing.allocator, &cfg, &h);
        try std.testing.expectEqual(@as(usize, 594), m.tensors);
        try std.testing.expectEqual(namespace, m.namespace);
        if (bits == 4) try std.testing.expectEqual(@as(usize, 973008384), m.bytes);
    };
}

test "VL text uses mixed8 embedding with group32 without admitting its image tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cfg = config(4, 32);
    defer cfg.deinit();
    cfg.kind = .vision_language;
    try cfg.quant.modules.put(cfg.arena.allocator(), "embed_tokens", .{ .bits = 8, .group = 32 });
    var h = try header(a, &cfg, .wrapped);
    const embed = h.get("language_model.model.embed_tokens.weight").?;
    try std.testing.expectEqual(@as(usize, 512), embed.shape[1]);
    const gate = h.get("language_model.model.layers.2.self_attn.gate_proj.scales").?;
    try std.testing.expectEqualSlices(usize, &.{ 16, 64 }, gate.shape[0..2]);
    try h.put(a, "vision_tower.unqualified.weight", .{ .dtype = .bf16, .rank = 1, .shape = .{ 1, 1, 1, 1, 1 }, .begin = 0, .end = 2 });
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 594), m.tensors);
    try std.testing.expectEqual(@as(usize, 1212106240), m.bytes);
}

test "missing head gate, wrong packed width and malformed convolution are rejected" {
    const keys = [_][]const u8{
        "language_model.model.layers.2.self_attn.gate_proj.weight",
        "language_model.model.layers.0.conv.in_proj.weight",
        "language_model.model.layers.0.conv.conv.weight",
        "language_model.model.layers.0.post_operator_norm.weight",
    };
    for (keys, 0..) |key, i| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        if (i == 0) {
            _ = h.swapRemove(key);
            try std.testing.expectError(error.MissingTensor, w.validate(std.testing.allocator, &cfg, &h));
        } else {
            h.getPtr(key).?.shape[if (i == 3) 0 else 1] += 1;
            try std.testing.expectError(error.TensorShape, w.validate(std.testing.allocator, &cfg, &h));
        }
    }
}

test "bad dtype, byte range, namespace and unexpected text tensors are rejected" {
    for (0..4) |mode| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var cfg = config(4, 64);
        defer cfg.deinit();
        var h = try header(arena.allocator(), &cfg, .wrapped);
        const key = "language_model.model.embed_tokens.weight";
        switch (mode) {
            0 => h.getPtr(key).?.dtype = .bf16,
            1 => h.getPtr(key).?.end -= 1,
            2 => try h.put(arena.allocator(), "model.embed_tokens.weight", h.get(key).?),
            3 => try h.put(arena.allocator(), "language_model.lm_head.weight", h.get(key).?),
            else => unreachable,
        }
        const err = switch (mode) {
            0 => error.TensorDType,
            1 => error.TensorBytes,
            2 => error.MixedNamespaces,
            3 => error.UnexpectedTensor,
            else => unreachable,
        };
        try std.testing.expectError(err, w.validate(std.testing.allocator, &cfg, &h));
    }
}

test "the ModelOpt NVFP4 export: 502 bare tensors, FP4 FFN with its scales, everything else bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cfg = try c.parse(std.testing.allocator, @import("config_test.zig").nvfp4);
    defer cfg.deinit();
    const h = try header(arena.allocator(), &cfg, .bare);
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 502), m.tensors);
    try std.testing.expectEqual(@as(usize, 1829689408), m.bytes);
    const codes = h.get("model.layers.0.feed_forward.w2.weight").?;
    try std.testing.expect(codes.dtype == .u8 and codes.shape[0] == 2048 and codes.shape[1] == 3840);
    try std.testing.expectEqual(@as(u8, 0), h.get("model.layers.0.feed_forward.w2.weight_scale_2").?.rank);
    try std.testing.expect(h.get("model.layers.2.self_attn.gate_proj.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.embed_tokens.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.layers.0.conv.in_proj.scales") == null);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1, 7 }, h.get("model.layers.0.conv.conv.weight").?.shape[0..3]);
}

test "the block FP8 export: 402 bare tensors, FP8 FFN and q/k/v/o with block scales, convs and gates bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cfg = try c.parse(std.testing.allocator, @import("config_fp8_test.zig").fp8);
    defer cfg.deinit();
    const h = try header(arena.allocator(), &cfg, .bare);
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(@as(usize, 402), m.tensors);
    try std.testing.expectEqual(@as(usize, 2209949952), m.bytes);
    const down = h.get("model.layers.0.feed_forward.w2.weight_scale").?;
    try std.testing.expect(down.dtype == .bf16 and down.shape[0] == 16 and down.shape[1] == 60);
    try std.testing.expect(h.get("model.layers.2.self_attn.k_proj.weight").?.dtype == .f8_e4m3);
    try std.testing.expect(h.get("model.layers.2.self_attn.gate_proj.weight").?.dtype == .bf16);
    try std.testing.expect(h.get("model.layers.0.conv.in_proj.weight").?.dtype == .bf16);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1, 7 }, h.get("model.layers.0.conv.conv.weight").?.shape[0..3]);
}

/// LFM3-1.7B-Base's safetensors header (7bca5038): its 286 tensors' names, dtypes, shapes and byte ranges.
const base_header = @embedFile("base_header.json");
const base_data = 3457603072;

/// The fixture as a header: read here, since the CUDA build's `safetensors` re-exports the types without the parser.
fn baseHeader(a: std.mem.Allocator) !st.Header {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, base_header, .{});
    var h: st.Header = .empty;
    var it = root.object.iterator();
    while (it.next()) |e| {
        const t = e.value_ptr.object;
        const dtype = std.meta.stringToEnum(st.DType, try std.ascii.allocLowerString(a, t.get("dtype").?.string)).?;
        const dims = t.get("shape").?.array.items;
        const at = t.get("data_offsets").?.array.items;
        var entry: st.Entry = .{ .dtype = dtype, .rank = @intCast(dims.len), .shape = @splat(1), .begin = @intCast(at[0].integer), .end = @intCast(at[1].integer) };
        for (dims, 0..) |d, i| entry.shape[i] = @intCast(d.integer);
        try std.testing.expect(entry.end <= base_data);
        try h.put(a, e.key_ptr.*, entry);
    }
    return h;
}

test "LFM3-1.7B-Base's real header: 286 bf16 tensors under model., tied head, admitted as plain bf16" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cfg = try c.parse(std.testing.allocator, @import("config_bf16_test.zig").base);
    defer cfg.deinit();
    const h = try baseHeader(arena.allocator());
    const m = try w.validate(std.testing.allocator, &cfg, &h);
    try std.testing.expectEqual(w.Namespace.bare, m.namespace);
    try std.testing.expectEqual(@as(usize, 286), m.tensors);
    try std.testing.expectEqual(@as(usize, base_data), m.bytes);
    try std.testing.expectEqual(h.count(), m.tensors);
    for (h.values()) |t| try std.testing.expect(t.dtype == .bf16);
    try std.testing.expectEqualSlices(usize, &.{ 16, 2048 }, h.get("model.layers.2.self_attn.gate_proj.weight").?.shape[0..2]);
    try std.testing.expectEqualSlices(usize, &.{ 2048, 1, 7 }, h.get("model.layers.0.conv.conv.weight").?.shape[0..3]);
    // the generated schema's header is admitted the same way
    const g = try header(arena.allocator(), &cfg, .bare);
    try std.testing.expectEqual(m, try w.validate(std.testing.allocator, &cfg, &g));
}

test "Base's header with a missing tensor, a wrong dtype, a short range, a quantized companion or an lm_head is refused" {
    for (0..6) |mode| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var cfg = try c.parse(std.testing.allocator, @import("config_bf16_test.zig").base);
        defer cfg.deinit();
        var h = try baseHeader(a);
        const key = "model.layers.4.self_attn.k_proj.weight";
        const err = switch (mode) {
            0 => blk: {
                _ = h.orderedRemove(key);
                break :blk error.MissingTensor;
            },
            1 => blk: {
                h.getPtr(key).?.dtype = .f16;
                break :blk error.TensorDType;
            },
            2 => blk: {
                h.getPtr(key).?.end -= 2;
                break :blk error.TensorBytes;
            },
            3 => blk: {
                try h.put(a, "model.layers.4.self_attn.k_proj.scales", h.get("model.layers.4.self_attn.k_norm.weight").?);
                break :blk error.Lfm3Bf16QuantizedTensor;
            },
            4 => blk: {
                try h.put(a, "lm_head.weight", h.get("model.embed_tokens.weight").?);
                break :blk error.Lfm3Bf16UntiedHead;
            },
            5 => blk: {
                try h.put(a, "model.layers.4.self_attn.extra.weight", h.get("model.layers.4.self_attn.k_norm.weight").?);
                break :blk error.UnexpectedTensor;
            },
            else => unreachable,
        };
        try std.testing.expectError(err, w.validate(std.testing.allocator, &cfg, &h));
    }
}

test "a quantization block on the plain checkpoint's config refuses its bf16 tensors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const base = @import("config_bf16_test.zig").base;
    const tail = "\n  \"vocab_size\": 128000\n}";
    for ([_][]const u8{
        "\"quantization\": {\"bits\": 4, \"group_size\": 64}",
        "\"quantization_config\": {\"bits\": 8, \"group_size\": 64, \"mode\": \"affine\"}",
    }) |block| {
        const bytes = try std.mem.concat(std.testing.allocator, u8, &.{ base[0 .. base.len - tail.len], "\n  ", block, ",", tail });
        defer std.testing.allocator.free(bytes);
        var cfg = try c.parse(std.testing.allocator, bytes);
        defer cfg.deinit();
        try std.testing.expectEqual(c.Scheme.mlx_affine, cfg.scheme);
        const h = try baseHeader(arena.allocator());
        try std.testing.expectError(error.TensorDType, w.validate(std.testing.allocator, &cfg, &h));
    }
}
