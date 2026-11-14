//! TF_GEMMA4_V_ONLY on Metal: global layers keep their values alone, and attention rebuilds each key from its value.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const kernels = @import("kernels.zig");

/// TF_GEMMA4_V_ONLY's text: 0 or 1 (unset or blank: 0); any other text is refused by name.
pub fn asked(raw: ?[]const u8) !bool {
    return @import("sys").environment.value(bool, "TF_GEMMA4_V_ONLY", raw, false) catch error.BadGemmaVOnly;
}

/// TF_GEMMA4_V_ONLY as the process has it.
pub fn read() !bool {
    return @import("sys").environment.read(bool, "TF_GEMMA4_V_ONLY", false) catch error.BadGemmaVOnly;
}

/// V-only refuses drafts (each 16-row tile rebuilds a block's keys: +45 % at 4 rows), own values, and one key head.
pub fn refusal(geo: cfg.Geometry, drafts: bool) ?anyerror {
    if (drafts) return error.GemmaVOnlyDrafts;
    if (!geo.keys_are_values) return error.GemmaVOnlyKeysNotValues;
    if (geo.global_kv_heads < 2) return error.GemmaVOnlyOneKeyHead;
    return null;
}

/// A refusal's words, each naming the flag and what to change; null for any other error.
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadGemmaVOnly => "TF_GEMMA4_V_ONLY: 0 or 1 (1: Gemma 4's global layers keep values alone; attention makes each key from its value)",
        error.GemmaVOnlyDrafts => "TF_GEMMA4_V_ONLY=1 decodes one row a round on Metal (a drafted window's rows make each key block again per tile): serve --no-drafts, or unset TF_GEMMA4_V_ONLY",
        error.GemmaVOnlyKeysNotValues => "TF_GEMMA4_V_ONLY=1: this Gemma 4 checkpoint's global layers have their own v_proj (attention_k_eq_v false), so keys are not made from values: unset TF_GEMMA4_V_ONLY",
        error.GemmaVOnlyOneKeyHead => "TF_GEMMA4_V_ONLY=1: this Gemma 4 model's global layers have one key head, where V-only attention is slower than K+V: unset TF_GEMMA4_V_ONLY",
        else => null,
    };
}

/// The global layers' V-only pipelines: the prep that writes values alone, decode and the prompt's key rebuild.
pub const Pipes = struct {
    prep: mtl.Pipeline,
    decode: mtl.Pipeline,
    rebuild: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry, embed_bits: u8) !Pipes {
        const glue = try kernels.glueSource(a, geo, true, false, embed_bits);
        defer a.free(glue);
        const gtext = try std.mem.concat(a, u8, &.{ glue, ks.gemma4_vonly_glue });
        defer a.free(gtext);
        const glib = try mtl.Library.fromSource(device, gtext, mtl.CompileOptions.mlx());
        defer glib.deinit();
        const prep = try mtl.Pipeline.init(device, glib, "gemma_qkv_prep_v", false);
        errdefer prep.deinit();
        const plib = try library(a, device, geo, false);
        defer plib.deinit();
        const decode = try mtl.Pipeline.init(device, plib, "gemma_decode_attention_v", false);
        errdefer decode.deinit();
        return .{ .prep = prep, .decode = decode, .rebuild = try mtl.Pipeline.init(device, plib, "gemma_key_rebuild", false) };
    }

    pub fn deinit(p: Pipes) void {
        p.prep.deinit();
        p.decode.deinit();
        p.rebuild.deinit();
    }
};

/// The global prompt library with the V-only kernels; `plant` (the contract's control): one key product unrounded.
pub fn library(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry, plant: bool) !mtl.Library {
    const base = try kernels.promptSource(device, a, geo, true);
    defer a.free(base);
    const text = try std.mem.concat(a, u8, &.{ if (plant) "#define TF_KEY_PLANT\n" else "", base, ks.gemma4_vonly_prompt });
    defer a.free(text);
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

/// Bytes the prompt's key scratch takes: one global layer's keys at `capacity` (a chunk rebuilds them all).
pub fn scratchBytes(geo: cfg.Geometry, capacity: usize) u64 {
    return @as(u64, geo.global_kv_heads) * capacity * cfg.global_dim * 2;
}

test "TF_GEMMA4_V_ONLY takes exactly 0 or 1 through environment.zig; every other text is refused by name" {
    try std.testing.expect(!try asked(null));
    try std.testing.expect(!try asked(""));
    try std.testing.expect(!try asked("0"));
    try std.testing.expect(try asked("1"));
    for ([_][]const u8{ "off", "false", "no", "on", "2", " 0", "1 ", "yes", "true" }) |v| try std.testing.expectError(error.BadGemmaVOnly, asked(v));
}

test "TF_GEMMA4_V_ONLY decodes one row a round: drafts refused, then values that are not keys (the E4B), the 12B" {
    try std.testing.expectEqual(@as(?anyerror, null), refusal(cfg.dense31b, false));
    try std.testing.expectEqual(@as(?anyerror, error.GemmaVOnlyDrafts), refusal(cfg.dense31b, true));
    var own_values = cfg.dense31b;
    own_values.keys_are_values = false;
    try std.testing.expectEqual(@as(?anyerror, error.GemmaVOnlyKeysNotValues), refusal(own_values, false));
    try std.testing.expectEqual(@as(?anyerror, error.GemmaVOnlyOneKeyHead), refusal(cfg.unified12b, false));
    try std.testing.expectEqual(@as(?anyerror, error.GemmaVOnlyKeysNotValues), refusal(cfg.e4b, false));
    for ([_]anyerror{ error.BadGemmaVOnly, error.GemmaVOnlyDrafts, error.GemmaVOnlyKeysNotValues, error.GemmaVOnlyOneKeyHead }) |e|
        try std.testing.expect(std.mem.indexOf(u8, explain(e).?, "TF_GEMMA4_V_ONLY") != null);
}

test "V-only on the 31B: its 10 global layers keep values alone, 1280 MiB less a stream at 32K, a 128 MiB scratch" {
    const state = @import("state.zig");
    const g = cfg.dense31b;
    var globals: usize = 0;
    for (0..g.layers) |i| globals += @intFromBool(cfg.global(i));
    try std.testing.expectEqual(@as(usize, 10), globals);
    try std.testing.expectEqual(@as(u64, 1280) << 20, state.cacheBytes(g, 32768, false) - state.cacheBytes(g, 32768, true));
    try std.testing.expectEqual(@as(u64, 128) << 20, scratchBytes(g, 32768));
}
