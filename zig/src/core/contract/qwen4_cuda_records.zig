//! The launches a provisional dense Qwen4 shape (HC 4 x 5120 rank 640) makes with other dynamic memory or threads.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "flashnext_hc fn_hc_down (Qwen4 shape)", .symbol = "fn_hc_down", .image = "flashnext_hc", .threads = 128, .dynamic = 22016 }, .usage = .{ .{ .regs = 56, .static = 64, .bound = 128 }, .{ .regs = 54, .static = 64, .bound = 128 }, .{ .regs = 54, .static = 64, .bound = 128 }, .{ .regs = 60, .static = 64, .bound = 128 }, .{ .regs = 60, .static = 64, .bound = 128 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_upmix (Qwen4 shape)", .symbol = "fn_hc_upmix", .image = "flashnext_hc", .threads = 256, .dynamic = 28928 }, .usage = .{ .{ .regs = 69, .static = 2048, .bound = 256 }, .{ .regs = 64, .static = 2048, .bound = 256 }, .{ .regs = 64, .static = 2048, .bound = 256 }, .{ .regs = 78, .static = 2048, .bound = 256 }, .{ .regs = 78, .static = 2048, .bound = 256 } } },
    .{ .launch = .{ .name = "flashnext_hc fn_hc_reduce_act (Qwen4 shape)", .symbol = "fn_hc_reduce_act", .image = "flashnext_hc", .threads = 672 }, .usage = .{ .{ .regs = 32, .static = 4096, .bound = 1024 }, .{ .regs = 63, .static = 4096, .bound = 1024 }, .{ .regs = 63, .static = 4096, .bound = 1024 }, .{ .regs = 58, .static = 4096, .bound = 1024 }, .{ .regs = 58, .static = 4096, .bound = 1024 } } },
};
