//! Gemma 4 31B's V-only kernels (metal_gemma4_vonly.zig): the decode twin keeps a block's keys in its 32 KiB tile.
const M = @import("limits.zig").MetalLaunch;
// the decode and the rebuild sit in the prompt library (nax.h's fragments): no offline build, as their K+V twins
pub const records = [_]M{
    .{ .name = "gemma4 31B vonly global gemma_decode_attention_v", .threads = 128, .tg_memory = 32768, .reserved = 128, .offline = false },
    .{ .name = "gemma4 31B vonly global gemma_key_rebuild", .threads = 32, .tg_memory = 0, .offline = false },
    .{ .name = "gemma4 31B vonly global gemma_qkv_prep_v", .threads = 32, .tg_memory = 0, .offline = false },
};
