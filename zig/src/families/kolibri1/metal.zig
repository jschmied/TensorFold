//! Kolibri-1 (block FP8) on Metal: config and names shared with CUDA, the forward wired from core Metal blocks.
pub const Config = @import("config.zig").Config;
pub const config = @import("config.zig");
pub const names = @import("names.zig");
pub const fixture = @import("fixture.zig");
pub const ref = @import("ref.zig");
pub const weights = @import("metal_weights.zig");
pub const forward = @import("metal_forward.zig");

test {
    _ = @import("metal_forward.zig");
}
