//! A seeded Kolibri-1 checkpoint in the released layout: real attention geometry, fewer layers, experts and tokens.
const std = @import("std");
const core = @import("core");

const fixture = core.checkpoint_fixture;
const fp8 = core.quant.fp8;

/// What the file shrinks; heads, head dim, hidden, expert width and window stay the released model's.
pub const Shape = struct {
    full: []const bool = &.{ false, false, true }, // a layer's kind: sliding (false) or full attention
    experts: usize = 8,
    top_k: usize = 6,
    vocab: usize = 1024,
    hidden: usize = 2560,
    heads: usize = 48,
    kv_heads: usize = 4,
    head_dim: usize = 128,
    width: usize = 512, // routed and shared experts alike
    window: usize = 513,
    seed: u64 = 0x6b6f6c,
    eos: u32 = 1023,
    split: bool = false, // two shards, layer 1's expert 0 gate in the first and its up and down in the second
};

/// The parity checkpoint: two sliding layers then a full one, 8 experts top-6 and the shared one.
pub const parity: Shape = .{};
/// Two shards splitting one expert (as 30 of 50 released layers do): small, for the loader's refusal.
pub const split: Shape = .{ .full = &.{ false, true }, .experts = 3, .top_k = 2, .vocab = 32, .hidden = 256, .heads = 4, .kv_heads = 1, .width = 128, .window = 3, .eos = 31, .split = true };

/// All 384 experts top-6 in one sliding layer: upper expert ids, uneven groups and the real router width.
pub const wide: Shape = .{ .full = &.{false}, .experts = 384 };

const Gen = struct {
    a: std.mem.Allocator,
    rng: std.Random.DefaultPrng,
    list: std.ArrayList(fixture.Entry) = .empty,

    fn add(g: *Gen, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !void {
        try g.list.append(g.a, .{ .name = name, .dtype = dtype, .shape = try g.a.dupe(usize, shape), .bytes = bytes });
    }

    /// bf16 values mean + spread * N(0, 1), rounded to nearest even.
    fn bf16(g: *Gen, name: []const u8, shape: []const usize, mean: f32, spread: f32) !void {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const out = try g.a.alloc(u16, n);
        const r = g.rng.random();
        for (out) |*o| o.* = toBf16(mean + spread * @as(f32, @floatCast(r.floatNorm(f64))));
        try g.add(name, "BF16", shape, std.mem.sliceAsBytes(out));
    }

    /// The last router and bias entries' expert 1 made expert 0's: equal logits and keys, ties to the lower id.
    fn tie(g: *Gen, d: usize) void {
        const it = g.list.items;
        const router: []u8 = @constCast(it[it.len - 2].bytes);
        const bias: []u8 = @constCast(it[it.len - 1].bytes);
        @memcpy(router[2 * d ..][0 .. 2 * d], router[0 .. 2 * d]);
        @memcpy(bias[2..4], bias[0..2]);
    }

    /// e4m3 codes [n, k] (signed zeros, subnormals, normals to 2^1; no NaN) and fp32 scales unequal block to block.
    fn fp8Proj(g: *Gen, pre: []const u8, n: usize, k: usize) !void {
        const r = g.rng.random();
        const w = try g.a.alloc(u8, n * k);
        for (w) |*c| {
            const sign: u8 = if (r.boolean()) 0x80 else 0;
            const pick = r.uintLessThan(u8, 32);
            const e: u8 = if (pick == 0) 0 else 3 + pick % 6; // 1 in 32 subnormal or zero; else 2^-4 .. 2^1
            c.* = sign | (e << 3) | r.uintLessThan(u8, 8);
        }
        const nb = fp8.blocks(n);
        const kb = fp8.blocks(k);
        const s = try g.a.alloc(f32, nb * kb);
        const base = 0.5 / @sqrt(@as(f32, @floatFromInt(k)));
        for (s) |*v| v.* = base * (0.5 + r.float(f32)); // adjacent blocks differ: a misplaced scale shows
        try g.add(try std.fmt.allocPrint(g.a, "{s}.weight", .{pre}), "F8_E4M3", &.{ n, k }, w);
        try g.add(try std.fmt.allocPrint(g.a, "{s}.weight_scale_inv", .{pre}), "F32", &.{ nb, kb }, std.mem.sliceAsBytes(s));
    }
};

pub fn toBf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}

/// config.json for `s`, the released one's keys and quantization block.
pub fn configText(a: std.mem.Allocator, s: Shape) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.print("{{\"model_type\": \"kolibri1\", \"hidden_size\": {d}, \"num_hidden_layers\": {d}, ", .{ s.hidden, s.full.len });
    try w.print("\"num_attention_heads\": {d}, \"num_key_value_heads\": {d}, \"head_dim\": {d}, ", .{ s.heads, s.kv_heads, s.head_dim });
    try w.print("\"rms_norm_eps\": 1e-06, \"vocab_size\": {d}, \"rope_theta\": 10000.0, \"num_experts\": {d}, ", .{ s.vocab, s.experts });
    try w.print("\"num_experts_per_tok\": {d}, \"moe_intermediate_size\": {d}, ", .{ s.top_k, s.width });
    try w.print("\"shared_expert_intermediate_size\": {d}, \"norm_topk_prob\": false, ", .{s.width});
    try w.print("\"sliding_window\": {d}, \"eos_token_id\": {d}, \"layer_types\": [", .{ s.window, s.eos });
    for (s.full, 0..) |f, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", if (f) "full_attention" else "sliding_attention" });
    try w.writeAll("], \"quantization_config\": {\"quant_method\": \"fp8\", \"activation_scheme\": \"dynamic\", ");
    try w.writeAll("\"weight_block_size\": [128, 128]}}\n");
    return out.toOwnedSlice();
}

/// Every tensor names.zig maps, seeded: embed, then each layer, then the final norm and the head.
pub fn entries(a: std.mem.Allocator, s: Shape) ![]const fixture.Entry {
    var g: Gen = .{ .a = a, .rng = .init(s.seed) };
    const d = s.hidden;
    const qd = s.heads * s.head_dim;
    const kvd = s.kv_heads * s.head_dim;
    try g.bf16("model.embed_tokens.weight", &.{ s.vocab, d }, 0, 1);
    for (0..s.full.len) |i| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        for ([_][]const u8{ "input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm" }) |n|
            try g.bf16(try std.fmt.allocPrint(a, "{s}{s}.weight", .{ p, n }), &.{d}, 1, 0.1);
        try g.bf16(try std.fmt.allocPrint(a, "{s}self_attn.q_norm.weight", .{p}), &.{s.head_dim}, 1, 0.1);
        try g.bf16(try std.fmt.allocPrint(a, "{s}self_attn.k_norm.weight", .{p}), &.{s.head_dim}, 1, 0.1);
        for ([_]struct { []const u8, usize, usize }{ .{ "q", qd, d }, .{ "k", kvd, d }, .{ "v", kvd, d }, .{ "o", d, qd } }) |x|
            try g.fp8Proj(try std.fmt.allocPrint(a, "{s}self_attn.{s}_proj", .{ p, x[0] }), x[1], x[2]);
        try g.bf16(try std.fmt.allocPrint(a, "{s}mlp.gate.weight", .{p}), &.{ s.experts, d }, 0, 1 / @sqrt(@as(f32, @floatFromInt(d))));
        try g.bf16(try std.fmt.allocPrint(a, "{s}moe.router.expert_bias", .{p}), &.{s.experts}, 0, 0.1);
        g.tie(d); // experts 0 and 1 tie exactly in every row: the lower-id rule decides
        for (0..s.experts + 1) |j| {
            const e = if (j < s.experts) try std.fmt.allocPrint(a, "{s}mlp.experts.{d}", .{ p, j }) else try std.fmt.allocPrint(a, "{s}mlp.shared_experts", .{p});
            try g.fp8Proj(try std.fmt.allocPrint(a, "{s}.gate_proj", .{e}), s.width, d);
            try g.fp8Proj(try std.fmt.allocPrint(a, "{s}.up_proj", .{e}), s.width, d);
            try g.fp8Proj(try std.fmt.allocPrint(a, "{s}.down_proj", .{e}), d, s.width);
        }
    }
    try g.bf16("model.norm.weight", &.{d}, 1, 0.1);
    try g.bf16("lm_head.weight", &.{ s.vocab, d }, 0, 1 / @sqrt(@as(f32, @floatFromInt(d))));
    return g.list.items;
}

/// `dir`/config.json and `dir`/model.safetensors for `s`; the same seed writes the same bytes on every box.
pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, s: Shape) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = try configText(a, s) });
    const all = try entries(a, s);
    if (!s.split) return dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = try fixture.aligned(a, all) }); // on 16 bytes
    var cut: usize = 0;
    while (!std.mem.eql(u8, all[cut].name, "model.layers.1.mlp.experts.0.up_proj.weight")) cut += 1;
    const names_ = [2][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" };
    try dir.writeFile(io, .{ .sub_path = names_[0], .data = try fixture.aligned(a, all[0..cut]) });
    try dir.writeFile(io, .{ .sub_path = names_[1], .data = try fixture.aligned(a, all[cut..]) });
    var out: std.Io.Writer.Allocating = .init(a);
    try out.writer.writeAll("{\"weight_map\": {");
    for (all, 0..) |e, i| try out.writer.print("{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", e.name, names_[@intFromBool(i >= cut)] });
    try out.writer.writeAll("}}\n");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = out.written() });
}

const testing = std.testing;

test "Kolibri fixture: its config parses with the released geometry, every tensor maps, none is left over" {
    const config = @import("config.zig");
    const names = @import("names.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const small: Shape = .{ .full = &.{ false, true }, .experts = 3, .top_k = 2, .vocab = 64, .hidden = 256, .heads = 4, .kv_heads = 1, .width = 128, .eos = 63 };
    try write(testing.allocator, testing.io, tmp.dir, small);
    var why: config.Why = .{};
    const text = try configText(testing.allocator, parity);
    defer testing.allocator.free(text);
    const c = try config.parse(testing.allocator, text, &why);
    try config.checkShapes(c, &why);
    try testing.expectEqual(@as(usize, 7168), c.qkvDim());
    try testing.expect(!c.full[1] and c.full[2] and c.window == 513 and c.top_k == 6);
    const cwd = try std.process.currentPathAlloc(testing.io, testing.allocator);
    defer testing.allocator.free(cwd);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer testing.allocator.free(path);
    const sc = try config.Config.read(testing.allocator, testing.io, path);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, testing.io, path);
    defer ck.close();
    var m = try names.map(testing.allocator, &ck, sc);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 4), m.layers[1].experts.len);
}

test "Kolibri fixture: the same seed gives the same bytes, another seed others" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const s: Shape = .{ .full = &.{false}, .experts = 2, .top_k = 1, .vocab = 8, .hidden = 128, .heads = 4, .kv_heads = 1, .width = 128 };
    var other = s;
    other.seed += 1;
    const one = try fixture.image(a, try entries(a, s));
    try testing.expectEqualSlices(u8, one, try fixture.image(a, try entries(a, s)));
    try testing.expect(!std.mem.eql(u8, one, try fixture.image(a, try entries(a, other))));
}

test "Kolibri fixture: the split checkpoint puts layer 1's expert 0 gate and up in different shards" {
    const config = @import("config.zig");
    const names = @import("names.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try write(testing.allocator, testing.io, tmp.dir, split);
    const cwd = try std.process.currentPathAlloc(testing.io, testing.allocator);
    defer testing.allocator.free(cwd);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer testing.allocator.free(path);
    const c = try config.Config.read(testing.allocator, testing.io, path);
    var ck = try core.checkpoint.Checkpoint.openModel(testing.allocator, testing.io, path);
    defer ck.close();
    try testing.expectEqual(@as(usize, 2), ck.files.items.len);
    var m = try names.map(testing.allocator, &ck, c);
    defer m.deinit();
    const e = m.layers[1].experts[0];
    const in = struct {
        fn file(k: *const core.checkpoint.Checkpoint, t: core.checkpoint.Tensor) usize {
            for (k.files.items, 0..) |f, i| if (@intFromPtr(t.bytes.ptr) >= @intFromPtr(f.map.memory.ptr) and
                @intFromPtr(t.bytes.ptr) < @intFromPtr(f.map.memory.ptr) + f.map.memory.len) return i;
            return k.files.items.len;
        }
    };
    try testing.expect(in.file(&ck, e.gate.w) != in.file(&ck, e.up.w));
    try testing.expectEqual(in.file(&ck, m.layers[0].experts[0].gate.w), in.file(&ck, m.layers[0].experts[0].down.w));
}
