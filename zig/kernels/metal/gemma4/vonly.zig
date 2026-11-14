//! Gemma 4's V-only twins (TF_GEMMA4_V_ONLY): main's prep and decode text with the key write dropped, keys rebuilt.
const std = @import("std");
const se = @import("../core/source_edit.zig");
const edit = se.edit;
const between = se.between;

/// gemma_qkv_prep_v: main's prep with the key head's cache write dropped (a V-only global cache holds values alone).
pub const glue = blk: {
    var s: []const u8 = between(@embedFile("glue.metal"), "// gilby's _EMIT:", "// gilby's attend partials");
    s = edit(s, "inline void gemma_emit(", "inline void gemma_emit_v(", 1);
    s = edit(s, "    if (cm.y) cache[at + int(lane) + 32 * i] = bfloat(y[i]);\n", "    if (cm.y && kind == 2) cache[at + int(lane) + 32 * i] = bfloat(y[i]);\n", 1);
    s = edit(s, "kernel void gemma_qkv_prep(", "kernel void gemma_qkv_prep_v(", 1);
    s = edit(s, "  gemma_emit(kind,", "  gemma_emit_v(kind,", 1);
    break :blk s;
};

/// gemma_decode_attention_v: main's decode, each 32-key block's keys rebuilt once from the values into the tile.
const decode = blk: {
    @setEvalBranchQuota(20_000_000);
    const src = @embedFile("prompt_attention.metal");
    var s: []const u8 = src[std.mem.indexOf(u8, src, "[[max_total_threads_per_threadgroup(128)]]").?..];
    s = edit(s, "gemma_decode_attention(", "gemma_decode_attention_v(", 1);
    s = edit(s, "device float* PO [[buffer(6)]],", "device float* PO [[buffer(6)]], const device bfloat* KW [[buffer(7)]],\n    const device float* INVF [[buffer(8)]],", 1);
    s = edit(s, "  const device bfloat* kc = K + long(kv) * cap * DH + int(sg) * 128;\n", "  const device bfloat* vh = V + long(kv) * cap * DH;\n", 1);
    s = edit(s, "  threadgroup float part[SL][32][16];\n", "  threadgroup bfloat kt[32 * DH]; // the block's keys; the scores' partial sums reuse it\n" ++
        "  threadgroup float (*part)[32][16] = reinterpret_cast<threadgroup float (*)[32][16]>(kt);\n", 1);
    s = edit(s, "    const device bfloat* kblk = kc + long(kpos) * DH;\n", "    gemma_key_tile(kt, vh, KW, INVF, kpos, sg * 32 + lane, 32 * SL);\n" ++
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n    const threadgroup bfloat* kblk = kt + int(sg) * 128;\n", 1);
    s = edit(s, "    if (SL > 1) {\n", "    threadgroup_barrier(mem_flags::mem_threadgroup); // every simdgroup's keys read before part reuses them\n    if (SL > 1) {\n", 1);
    break :blk s;
};

/// The rebuild helpers, gemma_key_rebuild (a prompt chunk's keys into a scratch) and gemma_decode_attention_v.
pub const prompt = @embedFile("vonly.metal") ++ decode;
