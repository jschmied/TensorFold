//! The model dimensions the CUDA path is built for, from one table, and the kernel limits any shape must meet.

const std = @import("std");
const config = @import("config.zig");

/// One checkpoint's dimensions. A zero `experts` is a dense model, whose feed-forward is `ffn` wide.
pub const Shape = struct {
    layers: usize,
    dims: usize,
    streams: usize,
    low: usize,
    vocab: usize,
    experts: usize,
    top_k: usize,
    moe_w: usize,
    ffn: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    index_heads: usize,
    index_dim: usize,
    index_ratio: usize,
    index_top: usize,
    rotary_pairs: usize,
    gdn_key_heads: usize,
    gdn_value_heads: usize,
    gdn_key_dim: usize,
    gdn_value_dim: usize,
    taps: usize,
    /// Every `full_every`-th layer is full attention (the schedule's period), the others are DeltaNet.
    full_every: usize,

    pub fn dense(s: Shape) bool {
        return s.experts == 0;
    }

    pub fn linearLayer(s: Shape, layer: usize) bool {
        return (layer + 1) % s.full_every != 0;
    }

    pub fn linearLayers(s: Shape) usize {
        return s.layers - s.layers / s.full_every;
    }

    pub fn attnLayers(s: Shape) usize {
        return s.layers / s.full_every;
    }

    /// The layer's place among the DeltaNet layers (its state slot).
    pub fn linearIndex(s: Shape, layer: usize) usize {
        return layer - layer / s.full_every;
    }

    /// The layer's place among the full-attention layers.
    pub fn attnIndex(s: Shape, layer: usize) usize {
        return layer / s.full_every;
    }

    pub fn wide(s: Shape) usize {
        return s.streams * s.dims;
    }

    /// The expert slots a row runs: the routed top-k and the shared expert.
    pub fn slots(s: Shape) usize {
        return s.top_k + 1;
    }

    /// The q|gate, k, v and indexer columns: Flash Next's layout, provisional for Qwen4 (Intel does not confirm it).
    pub fn projN(s: Shape) usize {
        return s.q_heads * 2 * s.head_dim + 2 * s.kv_heads * s.head_dim + (s.index_heads + 1) * s.index_dim;
    }

    pub fn outK(s: Shape) usize {
        return s.q_heads * s.head_dim;
    }

    /// The first config field whose value the kernels cannot run, or null; each limit names its kernel constant.
    pub fn limits(s: Shape) ?[]const u8 {
        if (s.layers == 0 or s.layers > config.max_layers) return "num_hidden_layers";
        if (s.full_every < 2 or s.full_every > s.layers) return "full_attention_interval";
        if (s.streams != 4) return "hc_count"; // cuda_glue's row HC kernels take four streams (the HC kernel holds eight)
        if (s.dims == 0 or s.dims % 256 != 0) return "hidden_size"; // the HC fold and the 256-wide K tile of the router
        if (s.low == 0 or s.low % 32 != 0) return "hc_lowrank"; // one quantization group per scale
        if (s.head_dim != 256) return "head_dim"; // flashnext_attn.cu HD
        if (s.kv_heads == 0 or s.q_heads % s.kv_heads != 0) return "num_key_value_heads";
        if (s.q_heads / s.kv_heads > 16) return "num_attention_heads"; // flashnext_attn.cu QR: a KV head's tile holds 16
        if (s.index_dim != 128) return "indexer_head_dim"; // flashnext_attn.cu DI
        if (s.index_heads == 0 or s.index_ratio != 4 or s.index_top != 512) return "indexer_budget"; // the block indexer
        if (s.rotary_pairs * 2 > s.head_dim) return "partial_rotary_factor";
        if (s.gdn_key_heads != 16) return "linear_num_key_heads"; // flashnext_gdn.cu NK
        if (s.gdn_value_heads != 48) return "linear_num_value_heads"; // NV
        if (s.gdn_key_dim != 128) return "linear_key_head_dim"; // DK
        if (s.gdn_value_dim != 128) return "linear_value_head_dim"; // DV
        if (s.taps != 4) return "linear_conv_kernel_dim"; // TAPS
        if (s.dense()) return if (s.ffn == 0 or s.ffn % 32 != 0) "intermediate_size" else null;
        if (s.ffn != 0) return "intermediate_size";
        if (s.top_k == 0 or s.top_k > s.experts) return "num_experts_per_tok";
        if (s.moe_w == 0 or s.moe_w % 32 != 0) return "moe_intermediate_size";
        return null;
    }

    /// The shape of a parsed config, or the layer or field the CUDA path would not serve.
    pub fn fromConfig(c: *const config.Config) !Shape {
        if (c.isMixed()) return error.ProvisionalMixedLayers; // provisional: the CUDA path runs one feed-forward kind in every layer
        const dense_ffn = c.isDense();
        const every = try scheduleEvery(c);
        const s: Shape = .{
            .layers = c.layers,
            .dims = c.hidden,
            .streams = c.hc_count,
            .low = c.hc_lowrank,
            .vocab = c.vocab,
            .experts = if (dense_ffn) 0 else c.experts,
            .top_k = if (dense_ffn) 0 else c.topk,
            .moe_w = if (dense_ffn) 0 else c.expert_width,
            .ffn = c.intermediate,
            .q_heads = c.heads,
            .kv_heads = c.kv_heads,
            .head_dim = c.head_dim,
            .index_heads = c.index_heads,
            .index_dim = c.index_dim,
            .index_ratio = c.index_ratio,
            .index_top = c.index_budget / c.index_ratio,
            .rotary_pairs = c.rotary_dim / 2,
            .gdn_key_heads = c.key_heads,
            .gdn_value_heads = c.value_heads,
            .gdn_key_dim = c.key_dim,
            .gdn_value_dim = c.value_dim,
            .taps = c.conv,
            .full_every = every,
        };
        for (0..c.layers) |i| {
            if ((c.kinds[i] == .linear_attention) != s.linearLayer(i)) return error.UnsupportedLayerSchedule;
        }
        if (s.limits() != null) return error.UnsupportedShape;
        return s;
    }
};

/// A config's schedule period: the first full-attention layer's place, which full_attention_interval must match.
fn scheduleEvery(c: *const config.Config) !usize {
    var first: usize = 0;
    while (first < c.layers and c.kinds[first] == .linear_attention) first += 1;
    if (first == c.layers) return error.UnsupportedLayerSchedule; // no full-attention layer
    if (c.text.get("full_attention_interval")) |v| if (v != .null and (v != .integer or v.integer != first + 1)) return error.UnsupportedLayerSchedule;
    return first + 1;
}

/// The shipped Flash Next checkpoint: 2560 wide, 48 layers, 512 experts, Q24/KV2.
pub const flash_next: Shape = .{
    .layers = 48,
    .dims = 2560,
    .streams = 4,
    .low = 320,
    .vocab = 248320,
    .experts = 512,
    .top_k = 10,
    .moe_w = 640,
    .ffn = 0,
    .q_heads = 24,
    .kv_heads = 2,
    .head_dim = 256,
    .index_heads = 4,
    .index_dim = 128,
    .index_ratio = 4,
    .index_top = 512,
    .rotary_pairs = 32,
    .gdn_key_heads = 16,
    .gdn_value_heads = 48,
    .gdn_key_dim = 128,
    .gdn_value_dim = 128,
    .taps = 4,
    .full_every = 4,
};

/// A provisional dense shape implied by Intel's kernels, no released checkpoint; the layer count is a placeholder.
pub const qwen4_dense: Shape = blk: {
    var s = flash_next;
    s.layers = 64;
    s.dims = 5120;
    s.low = 640;
    s.experts = 0;
    s.top_k = 0;
    s.moe_w = 0;
    s.ffn = 14336;
    s.kv_heads = 4;
    break :blk s;
};

/// The shape this build serves (cuda_prompt_wt, cuda_gdn); step compile-flashnext-dense sets qwen4_dense to compile it.
pub const compiled: Shape = if (std.mem.eql(u8, @import("flashnext_shape").name, "qwen4_dense")) qwen4_dense else flash_next;

comptime {
    if (compiled.limits()) |field| @compileError("the compiled Flash Next shape is outside the kernels' limits at " ++ field);
}
