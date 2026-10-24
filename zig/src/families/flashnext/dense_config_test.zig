//! A provisional dense config (parser-only: its 6-bit quantization has no CUDA load) parses beside the MoE one.
const std = @import("std");
const config = @import("config.zig");
const t = std.testing;

const dense = @embedFile("fixtures/dense_config_parser_only.json");
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
    try t.expect(!c.isDense() and !c.isMixed());
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

test "dense Flash Next config: num_experts 0 or null is all dense, experts without the rest still fail" {
    const bare = try with(t.allocator, "num_experts", 8);
    defer t.allocator.free(bare);
    try t.expectError(error.MissingFlashField, config.parse(t.allocator, bare));
    const none = try with(t.allocator, "num_experts", 0);
    defer t.allocator.free(none);
    var c = try config.parse(t.allocator, none);
    defer c.deinit();
    try t.expect(c.isDense() and !c.isMixed());
}

/// The shipped MoE config with `key` set to the JSON text `value`, parsed by Intel's per-layer rule.
fn moeWith(gpa: std.mem.Allocator, pairs: []const struct { []const u8, []const u8 }) !config.Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, moe, .{});
    defer parsed.deinit();
    const root = &parsed.value.object;
    const text = if (root.getPtr("text_config")) |v| &v.object else root;
    for (pairs) |p| {
        var v = try std.json.parseFromSlice(std.json.Value, gpa, p[1], .{});
        defer v.deinit();
        try text.put(gpa, p[0], try cloneValue(parsed.arena.allocator(), v.value));
    }
    const bytes = try std.json.Stringify.valueAlloc(gpa, parsed.value, .{});
    defer gpa.free(bytes);
    return config.parse(gpa, bytes);
}

fn cloneValue(a: std.mem.Allocator, v: std.json.Value) !std.json.Value {
    return switch (v) {
        .array => |arr| blk: {
            var out = std.json.Array.init(a);
            for (arr.items) |x| try out.append(try cloneValue(a, x));
            break :blk .{ .array = out };
        },
        .string => |str| .{ .string = try a.dupe(u8, str) },
        else => v,
    };
}

test "dense Flash Next config: mlp_only_layers makes those layers dense and needs intermediate_size" {
    try t.expectError(error.ProvisionalDenseNeedsIntermediateSize, moeWith(t.allocator, &.{.{ "mlp_only_layers", "[0, 1]" }}));
    var c = try moeWith(t.allocator, &.{ .{ "mlp_only_layers", "[0, 1]" }, .{ "intermediate_size", "9216" } });
    defer c.deinit();
    try t.expect(c.isMixed() and !c.isDense());
    try t.expect(!c.moe_layer[0] and !c.moe_layer[1] and c.moe_layer[2] and c.moe_layer[c.layers - 1]);
    try t.expectEqual(@as(usize, 9216), c.intermediate);
    try t.expectEqual(@as(usize, 512), c.experts);
}

test "dense Flash Next config: decoder_sparse_step 2 makes every other layer dense" {
    var c = try moeWith(t.allocator, &.{ .{ "decoder_sparse_step", "2" }, .{ "intermediate_size", "9216" } });
    defer c.deinit();
    for (0..c.layers) |i| try t.expectEqual(i % 2 == 1, c.moe_layer[i]);
    try t.expect(c.isMixed());
    try t.expectError(error.InvalidFlashShape, moeWith(t.allocator, &.{.{ "decoder_sparse_step", "0" }}));
}

test "dense Flash Next config: a dense layer takes silu only, and every layer in mlp_only_layers is all dense" {
    try t.expectError(error.ProvisionalDenseActivation, moeWith(t.allocator, &.{ .{ "mlp_only_layers", "[0]" }, .{ "intermediate_size", "9216" }, .{ "hidden_act", "\"gelu\"" } }));
    var c = try moeWith(t.allocator, &.{ .{ "mlp_only_layers", "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]" }, .{ "intermediate_size", "9216" } });
    defer c.deinit();
    try t.expect(c.isDense() and !c.isMixed());
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

test "dense Flash Next config: an mlp_only_layers entry past the last layer is refused, not dropped" {
    try t.expectError(error.InvalidFlashLayers, moeWith(t.allocator, &.{ .{ "mlp_only_layers", "[48]" }, .{ "intermediate_size", "9216" } }));
    var c = try moeWith(t.allocator, &.{ .{ "mlp_only_layers", "[47]" }, .{ "intermediate_size", "9216" } });
    defer c.deinit();
    try t.expect(!c.moe_layer[47] and c.moe_layer[46]);
}

test "dense Flash Next config: PLE is a capability, absent unless the config declares it" {
    var bare = try config.parse(t.allocator, dense);
    defer bare.deinit();
    try t.expect(!bare.hasPle());
    try t.expectEqual(@as(usize, 0), bare.ple_dim + bare.ple_conv + bare.ngram + bare.ngram_heads + bare.ngram_vocab + bare.ngram_shards);
    const declared = try with(t.allocator, "ple_embed_dim", 5120);
    defer t.allocator.free(declared);
    var c = try config.parse(t.allocator, declared);
    defer c.deinit();
    try t.expect(c.hasPle());
    try t.expectEqual(@as(usize, 5120), c.ple_dim);
    try t.expectEqual(@as(usize, 3), c.ngram);
    const odd = try with(t.allocator, "ple_embed_dim", 5121);
    defer t.allocator.free(odd);
    try t.expectError(error.InvalidFlashNgram, config.parse(t.allocator, odd));
}

test "dense Flash Next config: the shipped MoE config keeps its PLE" {
    var c = try config.parse(t.allocator, moe);
    defer c.deinit();
    try t.expect(c.hasPle());
    try t.expectEqual(@as(usize, 2560), c.ple_dim);
    try t.expectEqual(@as(usize, 3), c.ngram);
    try t.expectEqual(@as(usize, 8), c.ngram_heads);
    try t.expectEqual(@as(usize, 20_000_000), c.ngram_vocab);
    try t.expectEqual(@as(usize, 128), c.ngram_shards);
}
