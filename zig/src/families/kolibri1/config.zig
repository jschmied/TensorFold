//! Kolibri 1's dimensions from config.json, refused as the Python engine refuses them (FP8 128 x 128, sigmoid routing).
const std = @import("std");

pub const max_layers = 64;
pub const max_eos = 4;

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

pub const Config = struct {
    layers: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    experts: usize, // routed; the shared expert comes after them
    top_k: usize,
    moe_width: usize,
    shared_width: usize,
    window: usize, // sliding layers attend to this many keys back, the row's own included
    full: [max_layers]bool = @splat(false), // full attention takes no positions; sliding layers take RoPE
    rope_theta: f32,
    eps: f32,
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,

    pub fn qkvDim(self: Config) usize {
        return (self.heads + 2 * self.kv_heads) * self.head_dim;
    }

    pub fn fullCount(self: Config) usize {
        var n: usize = 0;
        for (self.full[0..self.layers]) |f| n += @intFromBool(f);
        return n;
    }

    pub fn isEos(self: Config, token: u32) bool {
        for (self.eos[0..self.eos_count]) |e| if (e == token) return true;
        return false;
    }

    /// End tokens from generation_config.json's text, after config.json's own, as the released files end replies.
    pub fn addEos(self: *Config, allocator: std.mem.Allocator, generation: []const u8) !void {
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, generation, .{});
        defer parsed.deinit();
        if (parsed.value.object.get("eos_token_id")) |v| try eosFrom(self, v);
    }

    /// `dir`/config.json, and `dir`/generation_config.json's end tokens when it is there.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        var why: Why = .{};
        var c = parse(gpa, text, &why) catch |e| {
            if (why.len > 0) std.log.err("{s}: {s}", .{ path, why.text() });
            return e;
        };
        const gen_path = try std.fs.path.join(gpa, &.{ dir, "generation_config.json" });
        defer gpa.free(gen_path);
        const gen = std.Io.Dir.cwd().readFileAlloc(io, gen_path, gpa, .limited(1 << 20)) catch |e| switch (e) {
            error.FileNotFound => return c,
            else => return e,
        };
        defer gpa.free(gen);
        try c.addEos(gpa, gen);
        return c;
    }
};

fn int(obj: std.json.ObjectMap, key: []const u8, why: *Why) !usize {
    const v = obj.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    if (v != .integer or v.integer <= 0) {
        why.set("config.json's {s} is not a positive integer", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

/// One end token, or a list of them; null entries are skipped and a token already listed is kept once.
fn eosFrom(c: *Config, v: std.json.Value) !void {
    const items: []const std.json.Value = switch (v) {
        .integer => &.{v},
        .array => |a| a.items,
        else => return,
    };
    for (items) |x| {
        if (x != .integer) continue;
        const t: u32 = @intCast(x.integer);
        if (c.isEos(t)) continue;
        if (c.eos_count == max_eos) return error.TooManyEos;
        c.eos[c.eos_count] = t;
        c.eos_count += 1;
    }
}

/// config.json's text; a refusal's reason goes to `why`.
pub fn parse(allocator: std.mem.Allocator, text: []const u8, why: *Why) !Config {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const mt = o.get("model_type") orelse return error.BadConfig;
    if (mt != .string or !std.mem.eql(u8, mt.string, "kolibri1")) {
        why.set("model_type {f} is not kolibri1", .{std.json.fmt(mt, .{})});
        return error.NotKolibri1;
    }
    const q = if (o.get("quantization_config")) |v| (if (v == .object) v.object else null) else null;
    const fp8 = if (q) |m| (if (m.get("quant_method")) |x| x == .string and std.mem.eql(u8, x.string, "fp8") else false) else false;
    const blocks = if (q) |m| (if (m.get("weight_block_size")) |x| x == .array and x.array.items.len == 2 and
        x.array.items[0] == .integer and x.array.items[0].integer == 128 and
        x.array.items[1] == .integer and x.array.items[1].integer == 128 else false) else false;
    if (!fp8 or !blocks) {
        why.set("Kolibri 1's CUDA engine reads its FP8 checkpoint (128 x 128 blocks, fp32 scales)", .{});
        return error.UnsupportedQuantization;
    }
    if (o.get("norm_topk_prob")) |v| if (v == .bool and v.bool) {
        why.set("Kolibri 1 routes by unnormalised sigmoid weights; norm_topk_prob is not supported", .{});
        return error.UnsupportedRouting;
    };
    var c = Config{
        .layers = try int(o, "num_hidden_layers", why),
        .hidden = try int(o, "hidden_size", why),
        .heads = try int(o, "num_attention_heads", why),
        .kv_heads = try int(o, "num_key_value_heads", why),
        .head_dim = try int(o, "head_dim", why),
        .vocab = try int(o, "vocab_size", why),
        .experts = try int(o, "num_experts", why),
        .top_k = try int(o, "num_experts_per_tok", why),
        .moe_width = try int(o, "moe_intermediate_size", why),
        .shared_width = try int(o, "shared_expert_intermediate_size", why),
        .window = try int(o, "sliding_window", why),
        .rope_theta = 10000.0,
        .eps = @floatCast(try number(o.get("rms_norm_eps") orelse return error.BadConfig)),
    };
    if (c.layers > max_layers) {
        why.set("{d} layers; this build holds at most {d}", .{ c.layers, max_layers });
        return error.BadConfig;
    }
    // rope_parameters.rope_theta wins over the top-level rope_theta, as the Python loader reads them
    if (o.get("rope_theta")) |v| c.rope_theta = @floatCast(try number(v));
    if (o.get("rope_parameters")) |v| if (v == .object) if (v.object.get("rope_theta")) |t| {
        c.rope_theta = @floatCast(try number(t));
    };
    const types = o.get("layer_types") orelse {
        why.set("config.json has no layer_types", .{});
        return error.BadConfig;
    };
    if (types != .array or types.array.items.len != c.layers) {
        why.set("layer_types does not name each of the {d} layers", .{c.layers});
        return error.BadConfig;
    }
    for (types.array.items, 0..) |t, i| {
        if (t != .string) return error.BadConfig;
        if (std.mem.eql(u8, t.string, "full_attention")) {
            c.full[i] = true;
        } else if (!std.mem.eql(u8, t.string, "sliding_attention")) {
            why.set("layer {d} is {s}; Kolibri 1 has full_attention and sliding_attention layers", .{ i, t.string });
            return error.UnsupportedLayer;
        }
    }
    if (o.get("eos_token_id")) |v| try eosFrom(&c, v);
    return c;
}

/// The shapes the kernels serve (Kolibri 1, FP8 128 x 128): the captured kernels are built for these.
pub fn checkShapes(c: Config, why: *Why) !void {
    const ok = c.layers == 50 and c.hidden == 2560 and c.heads == 48 and c.kv_heads == 4 and c.head_dim == 128 and
        c.vocab == 128000 and c.experts == 384 and c.top_k == 6 and c.moe_width == 512 and c.shared_width == 512 and
        c.window == 513 and c.fullCount() == 10;
    if (!ok) {
        why.set("this Kolibri 1's shapes differ from the kernels built for Kolibri 1 (50 layers, 2560 wide, 384 experts)", .{});
        return error.UnsupportedShapes;
    }
}

/// Kolibri 1's config.json as released (layer_types: every fifth layer full), its long module list cut to two entries.
pub const released =
    \\{"model_type": "kolibri1", "hidden_size": 2560, "num_hidden_layers": 50, "num_attention_heads": 48,
    \\ "num_key_value_heads": 4, "head_dim": 128, "rms_norm_eps": 1e-06, "vocab_size": 128000, "rope_theta": 10000.0,
    \\ "num_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    \\ "norm_topk_prob": false, "sliding_window": 513, "eos_token_id": 127906,
    \\ "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "full_attention"],
    \\ "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
    \\  "modules_to_not_convert": ["model.layers.0.mlp.gate", "model.layers.1.mlp.gate"]}}
;

fn refused(text: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, parse(std.testing.allocator, text, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

fn replaced(from: []const u8, to: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, std.testing.allocator, released, from, to);
}

test "parse the released config, its shapes and end tokens" {
    var why: Why = .{};
    var c = try parse(std.testing.allocator, released, &why);
    try checkShapes(c, &why);
    try std.testing.expectEqual(@as(usize, 50), c.layers);
    try std.testing.expect(c.full[4] and c.full[49] and !c.full[0] and !c.full[48]);
    try std.testing.expectEqual(@as(usize, 7168), c.qkvDim());
    try std.testing.expectEqual(@as(f32, 1e-6), c.eps);
    try std.testing.expect(c.isEos(127906) and !c.isEos(127901));
    try c.addEos(std.testing.allocator, "{\"eos_token_id\": [127906, 127901], \"top_k\": 128}");
    try std.testing.expectEqual(@as(usize, 2), c.eos_count);
    try std.testing.expect(c.isEos(127901));
}

test "the Python engine's refusals" {
    const not = try replaced("\"model_type\": \"kolibri1\"", "\"model_type\": \"qwen3_moe\"");
    defer std.testing.allocator.free(not);
    try refused(not, error.NotKolibri1, "kolibri1");
    const blocks = try replaced("[128, 128]", "[64, 64]");
    defer std.testing.allocator.free(blocks);
    try refused(blocks, error.UnsupportedQuantization, "128 x 128");
    const method = try replaced("\"quant_method\": \"fp8\"", "\"quant_method\": \"awq\"");
    defer std.testing.allocator.free(method);
    try refused(method, error.UnsupportedQuantization, "FP8");
    const norm = try replaced("\"norm_topk_prob\": false", "\"norm_topk_prob\": true");
    defer std.testing.allocator.free(norm);
    try refused(norm, error.UnsupportedRouting, "norm_topk_prob");
    const layer = try replaced("\"full_attention\"]", "\"linear_attention\"]");
    defer std.testing.allocator.free(layer);
    try refused(layer, error.UnsupportedLayer, "linear_attention");
    const short = try replaced("\"num_hidden_layers\": 50", "\"num_hidden_layers\": 49");
    defer std.testing.allocator.free(short);
    try refused(short, error.BadConfig, "49 layers");
}

test "rope_parameters' theta wins, and other shapes are refused before any kernel runs" {
    const rope = try replaced("\"rope_theta\": 10000.0", "\"rope_theta\": 10000.0, \"rope_parameters\": {\"rope_theta\": 5e5}");
    defer std.testing.allocator.free(rope);
    var why: Why = .{};
    const c = try parse(std.testing.allocator, rope, &why);
    try std.testing.expectEqual(@as(f32, 5e5), c.rope_theta);
    var wide = c;
    wide.hidden = 4096;
    try std.testing.expectError(error.UnsupportedShapes, checkShapes(wide, &why));
}
