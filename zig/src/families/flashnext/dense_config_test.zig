//! A dense Flash Next config (provisional: no released checkpoint) parses beside the MoE one without changing it.
const std = @import("std");
const config = @import("config.zig");
const t = std.testing;

const dense = @embedFile("fixtures/dense_config.json");
const moe = @embedFile("fixtures/config.json");

fn without(gpa: std.mem.Allocator, key: []const u8) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, dense, .{});
    defer parsed.deinit();
    _ = parsed.value.object.swapRemove(key);
    return std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
}

fn with(gpa: std.mem.Allocator, key: []const u8, value: i64) ![]u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, dense, .{});
    defer parsed.deinit();
    try parsed.value.object.put(gpa, key, .{ .integer = value });
    return std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
}

test "dense Flash Next config: the Qwen4-27B shape parses as dense" {
    var c = try config.parse(t.allocator, dense);
    defer c.deinit();
    try t.expect(c.isDense());
    try t.expectEqual(@as(usize, 14336), c.intermediate);
    try t.expectEqual(@as(usize, 0), c.experts + c.topk + c.expert_width + c.shared_width);
    try t.expectEqual(@as(usize, 5120), c.hidden);
    try t.expectEqual(@as(usize, 20480), try c.wide());
    try t.expectEqual(@as(usize, 640), c.hc_lowrank);
    try t.expectEqual(@as(usize, 24), c.heads);
    try t.expectEqual(@as(usize, 4), c.kv_heads);
    try t.expectEqual(@as(usize, 256), c.head_dim);
    try t.expectEqual(@as(usize, 2048), try c.keyWidth());
    try t.expectEqual(@as(usize, 6144), try c.valueWidth());
    try t.expectEqual(@as(usize, 64), c.layers);
    try t.expectEqual(@as(usize, 48), std.mem.count(config.Kind, c.kinds[0..c.layers], &.{.linear_attention}));
}

test "dense Flash Next config: the shipped MoE config is not dense and has no dense width" {
    var c = try config.parse(t.allocator, moe);
    defer c.deinit();
    try t.expect(!c.isDense());
    try t.expectEqual(@as(usize, 0), c.intermediate);
    try t.expectEqual(@as(usize, 512), c.experts);
    try t.expect(c.topk > 0 and c.expert_width > 0 and c.shared_width > 0);
}

test "dense Flash Next config: a dense config needs its width and refuses a half MoE description" {
    const no_width = try without(t.allocator, "intermediate_size");
    defer t.allocator.free(no_width);
    try t.expectError(error.ProvisionalDenseNeedsIntermediateSize, config.parse(t.allocator, no_width));
    const zero = try with(t.allocator, "intermediate_size", 0);
    defer t.allocator.free(zero);
    try t.expectError(error.ProvisionalDenseNeedsIntermediateSize, config.parse(t.allocator, zero));
    for ([_][]const u8{ "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size" }) |key| {
        const half = try with(t.allocator, key, 8);
        defer t.allocator.free(half);
        try t.expectError(error.ProvisionalDenseHasMoeFields, config.parse(t.allocator, half));
    }
}

test "dense Flash Next config: experts without the rest of the MoE fields still fail as before" {
    const bare = try with(t.allocator, "num_experts", 8);
    defer t.allocator.free(bare);
    try t.expectError(error.MissingFlashField, config.parse(t.allocator, bare));
    const none = try with(t.allocator, "num_experts", 0);
    defer t.allocator.free(none);
    try t.expectError(error.InvalidFlashInteger, config.parse(t.allocator, none));
}

test "dense Flash Next config: the HC rank is read, never defaulted to Flash Next's 320" {
    const no_rank = try without(t.allocator, "hc_lowrank");
    defer t.allocator.free(no_rank);
    try t.expectError(error.MissingFlashField, config.parse(t.allocator, no_rank));
    for ([_]i64{ 640, 320 }) |rank| {
        const text = try with(t.allocator, "hc_lowrank", rank);
        defer t.allocator.free(text);
        var c = try config.parse(t.allocator, text);
        defer c.deinit();
        try t.expectEqual(@as(usize, @intCast(rank)), c.hc_lowrank);
    }
}
