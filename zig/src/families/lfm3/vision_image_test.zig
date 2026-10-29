//! LFM3-VL's processor settings as its config files give them (the bits themselves: core/picture/patches_test.zig).
const std = @import("std");
const vi = @import("vision_image.zig");

/// LiquidAI/LFM3-VL-2B-MLX-4bit (c823fade)'s processor_config.json image_processor, and its marker ids.
const processor_json =
    \\{"image_processor": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": 2,
    \\ "patch_size": 16, "resample": 3, "rescale_factor": 0.00392156862745098,
    \\ "size": {"shortest_edge": 65536, "longest_edge": 16777216}, "temporal_patch_size": 2}}
;
const config_json =
    \\{"image_token_id": 125030, "vision_start_token_id": 125009, "vision_end_token_id": 125010}
;

test "the published processor reads as torchvision's resize with fused normalization, in its pixel bounds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try vi.processor(arena.allocator(), processor_json);
    try std.testing.expectEqual(@as(u64, 65536), p.min_pixels);
    try std.testing.expectEqual(@as(u64, 16777216), p.max_pixels);
    try std.testing.expect(p.fused and p.resize == .torch);
    try std.testing.expectEqual(@as(f32, 0.5), p.mean[2]);
    const m = try vi.marks(arena.allocator(), config_json);
    try std.testing.expectEqual([3]u32{ 125030, 125009, 125010 }, [3]u32{ m.image, m.start, m.end });
}

test "another patch size, resample or a missing bound is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const edits = [_][2][]const u8{ .{ "\"patch_size\": 16", "\"patch_size\": 14" }, .{ "\"resample\": 3", "\"resample\": 2" }, .{ "\"longest_edge\": 16777216", "\"longest\": 1" } };
    for (edits) |e| {
        const bytes = try std.mem.replaceOwned(u8, arena.allocator(), processor_json, e[0], e[1]);
        try std.testing.expectError(error.VisionConfig, vi.processor(arena.allocator(), bytes));
    }
}
