//! The plain bf16 checkpoint's admission (LFM3-1.7B-Base), against its published config.json.
const std = @import("std");
const c = @import("config.zig");

test "a checkpoint without a quantization block parses as plain bf16, served on CUDA only" {
    var config = try c.parse(std.testing.allocator, base);
    defer config.deinit();
    try std.testing.expectEqual(c.Scheme.bf16, config.scheme);
    try std.testing.expectEqual(c.Kind.text, config.kind);
    try std.testing.expectEqual(c.bf16_width, try config.width("model.layers.2.self_attn.q_proj"));
    try std.testing.expectEqual(@as(usize, 262144), config.context);
    try std.testing.expectError(error.Lfm3Bf16NotOnMetal, c.servedOn(&config, .metal));
    try c.servedOn(&config, .cuda);
}

test "a quantization block on the plain config is that block's scheme; a null or unknown block is refused" {
    const tail = "\n  \"vocab_size\": 128000\n}";
    try std.testing.expect(std.mem.endsWith(u8, base, tail));
    const mlx = try std.mem.concat(std.testing.allocator, u8, &.{ base[0 .. base.len - tail.len], "\n  \"quantization\": {\"bits\": 4, \"group_size\": 64},", tail });
    defer std.testing.allocator.free(mlx);
    var q = try c.parse(std.testing.allocator, mlx);
    defer q.deinit();
    try std.testing.expectEqual(c.Scheme.mlx_affine, q.scheme);
    for ([_][]const u8{ "\"quantization\": null", "\"quantization_config\": null", "\"quantization_config\": {\"quant_method\": \"bitsandbytes\"}" }) |block| {
        const bytes = try std.mem.concat(std.testing.allocator, u8, &.{ base[0 .. base.len - tail.len], "\n  ", block, ",", tail });
        defer std.testing.allocator.free(bytes);
        if (c.parse(std.testing.allocator, bytes)) |parsed| {
            var p = parsed;
            p.deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

test "a plain config of another dtype is refused" {
    const half = try std.mem.replaceOwned(u8, std.testing.allocator, base, "\"dtype\": \"bfloat16\"", "\"dtype\": \"float16\"");
    defer std.testing.allocator.free(half);
    try std.testing.expect(!std.mem.eql(u8, half, base));
    try std.testing.expectError(error.UnsupportedLfm3Config, c.parse(std.testing.allocator, half));
}

/// LiquidAI/LFM3-1.7B-Base (7bca5038)'s config.json as published.
pub const base =
    \\{
    \\  "architectures": [
    \\    "Lfm3ForCausalLM"
    \\  ],
    \\  "auto_map": {
    \\    "AutoConfig": "configuration_lfm3.Lfm3Config",
    \\    "AutoModel": "modeling_lfm3.Lfm3Model",
    \\    "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"
    \\  },
    \\  "bos_token_id": 124894,
    \\  "conv_L_cache": 7,
    \\  "conv_bias": false,
    \\  "dtype": "bfloat16",
    \\  "eos_token_id": 124900,
    \\  "head_dim": 128,
    \\  "hidden_size": 2048,
    \\  "intermediate_size": 7680,
    \\  "layer_types": [
    \\    "conv",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv",
    \\    "sliding_attention",
    \\    "conv",
    \\    "full_attention",
    \\    "conv"
    \\  ],
    \\  "max_position_embeddings": 262144,
    \\  "model_type": "lfm3",
    \\  "norm_eps": 1e-05,
    \\  "num_attention_heads": 16,
    \\  "num_hidden_layers": 24,
    \\  "num_key_value_heads": 4,
    \\  "pad_token_id": 124893,
    \\  "rope_parameters": {
    \\    "full_attention": {
    \\      "partial_rotary_factor": 0.5,
    \\      "rope_theta": 1000000.0,
    \\      "rope_type": "default"
    \\    },
    \\    "sliding_attention": {
    \\      "rope_theta": 10000,
    \\      "rope_type": "default"
    \\    }
    \\  },
    \\  "sliding_window": 256,
    \\  "tie_word_embeddings": true,
    \\  "transformers_version": "5.12.1",
    \\  "use_cache": true,
    \\  "use_sandwich_norm": true,
    \\  "vocab_size": 128000
    \\}
;
