//! The core Metal blocks' launches (block_rows, moe_sigmoid, row_glue): threads a group and threadgroup bytes.
const std = @import("std");
const M = @import("limits.zig").MetalLaunch;

pub const records = [_]M{
    .{ .name = "core rows tf_block_rows", .threads = 128, .tg_memory = 0, .reserved = 128, .m1 = 128, .m2 = 128 },
    .{ .name = "core rows tf_block_glu", .threads = 128, .tg_memory = 0, .reserved = 128, .m1 = 128, .m2 = 128 },
    .{ .name = "core rows tf_block_down", .threads = 128, .tg_memory = 0, .reserved = 128, .m1 = 128, .m2 = 128 },
    .{ .name = "core glue tf_glue_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "core glue tf_glue_rms", .threads = 256, .tg_memory = 32, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "core glue tf_glue_add_rms", .threads = 256, .tg_memory = 32, .reserved = 256, .m1 = 256, .m2 = 256 },
    .{ .name = "core glue tf_glue_qk", .threads = 32, .tg_memory = 0, .reserved = 32 },
    .{ .name = "core glue tf_moe_select", .threads = 64, .tg_memory = 0, .reserved = 64 },
    .{ .name = "core glue tf_moe_plan", .threads = 1, .tg_memory = 0, .reserved = 32 },
    .{ .name = "core glue tf_moe_combine", .threads = 256, .tg_memory = 0, .reserved = 256 },
};

test "the core Metal blocks' launches fit every Apple family" {
    try std.testing.expectEqual(@as(usize, 0), try @import("registry.zig").checkMetal(std.testing.allocator, &records, true));
}
