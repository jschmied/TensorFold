//! The CUDA path's shape table against the configs it reads: the shipped checkpoint's and the synthetic dense one.
const std = @import("std");
const config = @import("config.zig");
const shape = @import("cuda_shape.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const t = std.testing;

test "shape: the shipped config yields the compiled shape and the constants the path has always had" {
    var c = try config.parse(t.allocator, @embedFile("fixtures/config.json"));
    defer c.deinit();
    try t.expectEqualDeep(shape.flash_next, try shape.Shape.fromConfig(&c));
    try t.expectEqualDeep(shape.flash_next, shape.compiled);
    try t.expectEqual(@as(usize, 48), wt.layers_n);
    try t.expectEqual(@as(usize, 36), wt.linear_n);
    try t.expectEqual(@as(usize, 12), wt.attn_n);
    try t.expectEqual(@as(usize, 10240), wt.wide);
    try t.expectEqual(@as(usize, 11), wt.slots);
    try t.expectEqual(@as(usize, 13952), wt.proj_n);
    try t.expectEqual(@as(usize, 6144), wt.out_k);
    try t.expectEqual(@as(usize, 32), wt.half);
    try t.expectEqual(@as(usize, 2052), wt.index_idw);
    try t.expectEqual(@as(usize, 2 * 16 * 128 + 48 * 128), gdn.conv_dim);
}

test "shape: the synthetic dense config yields the Qwen4-27B shape" {
    var c = try config.parse(t.allocator, @embedFile("fixtures/dense_config_parser_only.json"));
    defer c.deinit();
    const s = try shape.Shape.fromConfig(&c);
    try t.expectEqualDeep(shape.qwen4_dense, s);
    try t.expect(s.dense());
    try t.expectEqual(@as(usize, 640), s.low); // read from the config, not Flash Next's 320
    try t.expectEqual(@as(usize, 20480), s.wide());
    try t.expectEqual(@as(usize, 48), s.linearLayers());
    try t.expectEqual(@as(usize, 16), s.attnLayers());
    try t.expectEqual(@as(usize, 24 * 512 + 2 * 4 * 256 + 5 * 128), s.projN());
    try t.expectEqual(@as(usize, 6144), s.outK());
    try t.expectEqual(@as(usize, 1), s.slots());
    try t.expectEqual(@as(?[]const u8, null), s.limits());
}

test "shape: each kernel limit names the config field a shape breaks" {
    const Case = struct { []const u8, []const u8, usize };
    const cases = [_]Case{
        .{ "hc_count", "streams", 8 },
        .{ "hidden_size", "dims", 5000 },
        .{ "hc_lowrank", "low", 100 },
        .{ "head_dim", "head_dim", 128 },
        .{ "num_key_value_heads", "kv_heads", 5 },
        .{ "num_attention_heads", "q_heads", 48 },
        .{ "indexer_head_dim", "index_dim", 64 },
        .{ "linear_num_value_heads", "gdn_value_heads", 24 },
        .{ "linear_conv_kernel_dim", "taps", 3 },
        .{ "intermediate_size", "ffn", 100 },
    };
    inline for (cases) |case| {
        var s = shape.flash_next;
        @field(s, case[1]) = case[2];
        try t.expectEqualStrings(case[0], s.limits() orelse "accepted");
    }
    var dense = shape.qwen4_dense;
    dense.ffn = 0;
    try t.expectEqualStrings("intermediate_size", dense.limits().?);
    var moe = shape.flash_next;
    moe.top_k = 600;
    try t.expectEqualStrings("num_experts_per_tok", moe.limits().?);
}

test "shape: a layer schedule other than every fourth full attention is refused" {
    var c = try config.parse(t.allocator, @embedFile("fixtures/dense_config_parser_only.json"));
    defer c.deinit();
    c.kinds[3] = .linear_attention;
    try t.expectError(error.UnsupportedLayerSchedule, shape.Shape.fromConfig(&c));
}

test "shape: a config with some dense and some MoE layers is refused" {
    var c = try config.parse(t.allocator, @embedFile("fixtures/config.json"));
    defer c.deinit();
    c.moe_layer[0] = false;
    try t.expectError(error.ProvisionalMixedLayers, shape.Shape.fromConfig(&c));
}

fn schedule(gpa: std.mem.Allocator, base: []const u8, drop_types: bool, interval: ?i64) !config.Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, base, .{});
    defer parsed.deinit();
    const root = &parsed.value.object;
    const text = if (root.getPtr("text_config")) |v| &v.object else root;
    if (drop_types) _ = text.swapRemove("layer_types");
    if (interval) |n| try text.put(gpa, "full_attention_interval", .{ .integer = n }) else _ = text.swapRemove("full_attention_interval");
    const bytes = try std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
    defer gpa.free(bytes);
    return config.parse(gpa, bytes);
}

test "shape: the schedule is read from layer_types or from full_attention_interval, and the two must agree" {
    inline for (.{ @embedFile("fixtures/config.json"), @embedFile("fixtures/dense_config_parser_only.json") }, .{ shape.flash_next, shape.qwen4_dense }) |text, want| {
        var types = try schedule(t.allocator, text, false, null);
        defer types.deinit();
        try t.expectEqualDeep(want, try shape.Shape.fromConfig(&types));
        var interval = try schedule(t.allocator, text, true, 4);
        defer interval.deinit();
        try t.expectEqualSlices(config.Kind, types.kinds[0..types.layers], interval.kinds[0..interval.layers]);
        try t.expectEqualDeep(want, try shape.Shape.fromConfig(&interval));
        var both = try schedule(t.allocator, text, false, 4);
        defer both.deinit();
        try t.expectEqualDeep(want, try shape.Shape.fromConfig(&both));
        var clash = try schedule(t.allocator, text, false, 3);
        defer clash.deinit();
        try t.expectError(error.UnsupportedLayerSchedule, shape.Shape.fromConfig(&clash));
    }
    try t.expectError(error.MissingFlashField, schedule(t.allocator, @embedFile("fixtures/config.json"), true, null));
}

test "shape: another period is a shape of its own, indexed by its period, and only the compiled one is served" {
    var c = try schedule(t.allocator, @embedFile("fixtures/dense_config_parser_only.json"), true, 8);
    defer c.deinit();
    const s = try shape.Shape.fromConfig(&c);
    try t.expectEqual(@as(usize, 8), s.full_every);
    try t.expectEqual(@as(usize, 56), s.linearLayers());
    try t.expectEqual(@as(usize, 8), s.attnLayers());
    try t.expect(s.linearLayer(6) and !s.linearLayer(7));
    try t.expectEqual(@as(usize, 7), s.linearIndex(8));
    try t.expectEqual(@as(usize, 1), s.attnIndex(8));
    try t.expect(!std.meta.eql(s, shape.compiled));
    var flat = shape.flash_next;
    flat.full_every = 1;
    try t.expectEqualStrings("full_attention_interval", flat.limits().?);
    // the 3:1 schedule's indexes are the ones the serving path has always used
    for (0..48) |l| {
        try t.expectEqual(l % 4 != 3, shape.flash_next.linearLayer(l));
        try t.expectEqual(l - l / 4, shape.flash_next.linearIndex(l));
        try t.expectEqual(l / 4, shape.flash_next.attnIndex(l));
    }
}
