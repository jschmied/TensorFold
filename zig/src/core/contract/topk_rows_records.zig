//! topk_rows.cu's launches: every family's exact row top-k, a row a block (128 threads, or 1,024 for few rows).
const std = @import("std");
const registry = @import("registry.zig");
const R = registry.CudaRecord;
const U = @import("limits.zig").Usage;

fn narrow(regs: [5]u32) [5]U {
    var out: [5]U = undefined;
    for (&out, regs) |*u, r| u.* = .{ .regs = r, .static = 1064, .bound = 128 };
    return out;
}

fn wide(regs: [5]u32) [5]U {
    var out: [5]U = undefined;
    for (&out, regs) |*u, r| u.* = .{ .regs = r, .static = 1288, .bound = 1024 };
    return out;
}

const narrow_name = "topk rows tf_topk_rows";
const wide_name = "topk rows tf_topk_rows_wide";

/// nvcc 13.0, sm_80 / 86 / 89 / 120 / 121.
pub const cuda = [_]R{
    .{ .launch = .{ .name = narrow_name, .symbol = "tf_topk_rows", .image = "topk_rows", .threads = 128 }, .usage = narrow(.{ 55, 55, 55, 40, 40 }) },
    .{ .launch = .{ .name = wide_name, .symbol = "tf_topk_rows_wide", .image = "topk_rows", .threads = 1024 }, .usage = wide(.{ 32, 56, 56, 60, 64 }) },
};

/// nvcc 13.2 where it differs.
pub const nvcc132 = [_]registry.Override{
    .{ .name = narrow_name, .usage = narrow(.{ 55, 55, 55, 48, 48 }) },
    .{ .name = wide_name, .usage = wide(.{ 32, 56, 56, 64, 64 }) },
};

test "topk rows records: both launches fit every SM by their measured records" {
    for (cuda) |r| try std.testing.expectEqual(@as(usize, 0), try registry.checkCuda(std.testing.allocator, &.{r}, true));
}
