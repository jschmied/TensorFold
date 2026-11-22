//! Qwen3-ASR's Metal launches (qwen3_asr.metal): each reserves its threads in the source; the bytes are its own.
const M = @import("limits.zig").MetalLaunch;
pub const records = [_]M{
    .{ .name = "qwen3-asr tower gemm q3a_gemm", .threads = 128, .tg_memory = 12288, .reserved = 128 },
    .{ .name = "qwen3-asr row gemv q3a_gemv", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr conv taps q3a_conv_cols", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr conv rows q3a_flatten", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr layer norm q3a_layernorm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen3-asr add rms norm q3a_add_rmsnorm", .threads = 256, .tg_memory = 144, .reserved = 256 },
    .{ .name = "qwen3-asr swiglu q3a_swiglu", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr embed q3a_embed", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr copy q3a_copy", .threads = 256, .tg_memory = 0, .reserved = 256 },
    .{ .name = "qwen3-asr qk norm rope q3a_qk_rope", .threads = 128, .tg_memory = 2192, .reserved = 512 },
    .{ .name = "qwen3-asr split qkv q3a_split_qkv", .threads = 256, .tg_memory = 0, .reserved = 256 },
};
