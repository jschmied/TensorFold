//! Kolibri 1 on CUDA in Zig: the Python engine's FP8 kernels and layouts, so tokens match it bit for bit.

pub const Config = @import("config.zig").Config;
pub const names = @import("names.zig");

test {
    _ = @import("config.zig");
    _ = @import("names.zig");
}
