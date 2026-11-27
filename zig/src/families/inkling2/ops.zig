//! Inkling 2's glue kernels (inkling2/ops.cu): embedding, residual mix, row dots, router, SwiGLU epilogue, convs.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");

pub const max_blocks = config.max_blocks;
pub const max_queries = 3;

/// ops.cu's MixArgs, field for field.
pub const MixArgs = extern struct {
    blocks: [max_blocks]u64 = @splat(0),
    partial: u64,
    query: u64,
    norm: [max_queries]u64 = @splat(0),
    out: u64,
    n_blocks: i32,
    partial_bf16: i32,
    nq: i32,
    rows: i32,
    c: i32,
    eps: f32,
};

/// ops.cu's ConvArgs, field for field.
pub const ConvArgs = extern struct {
    x: u64,
    state: u64,
    w: u64,
    dyn: u64 = 0,
    base: u64 = 0,
    resid: u64 = 0,
    out: u64 = 0,
    segs: u64 = 0, // int4 per stream: row0, rows, state slot (0: one stream)
    hist: u64 = 0, // fp32 [rows][c]: each row's input as the state keeps it (0: none)
    part_stride: i64 = 0, // elements between the ranks' partials of x (parts > 1)
    state_stride: i64 = 0, // elements between the streams' states
    parts: i32 = 1,
    x_bf16: i32 = 0,
    round_in: i32 = 0,
    heads: i32 = 1,
    c: i32,
    rows: i32,
    mode: i32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    rows_mod: cuda.Module,
    rowdot_tiles: cuda.Function,
    rowdot_tiles_bf16: cuda.Function,
    prompt_tail: cuda.Function,
    embed: cuda.Function,
    mix: cuda.Function,
    mix_rows: cuda.Function,
    mix_roles: cuda.Function,
    rowdot: cuda.Function,
    topk: cuda.Function,
    conv: cuda.Function,
    softcap: cuda.Function,
    top1: cuda.Function,
    tobf: cuda.Function,
    put_words: cuda.Function,
    rewind: cuda.Function,
    race1: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_ops);
        errdefer mod.unload();
        var rows_mod = try cuda.Module.load(d, cuda.kernels.inkling2_rows);
        errdefer rows_mod.unload();
        return .{
            .mod = mod,
            .rows_mod = rows_mod,
            .rowdot_tiles = try rows_mod.function("tf_ink2_rowdot_tiles"),
            .rowdot_tiles_bf16 = try rows_mod.function("tf_ink2_rowdot_tiles_bf16"),
            .prompt_tail = try rows_mod.function("tf_ink2_prompt_tail"),
            .embed = try mod.function("tf_ink2_embed"),
            .mix = try mod.function("tf_ink2_mix"),
            .mix_rows = try mod.function("tf_ink2_mix_rows"),
            .mix_roles = try mod.function("tf_ink2_mix_roles"),
            .rowdot = try mod.function("tf_ink2_rowdot"),
            .topk = try mod.function("tf_ink2_topk"),
            .conv = try mod.function("tf_ink2_conv"),
            .softcap = try mod.function("tf_ink2_softcap"),
            .top1 = try mod.function("tf_ink2_top1"),
            .tobf = try mod.function("tf_ink2_tobf"),
            .put_words = try mod.function("tf_ink2_put_words"),
            .rewind = try mod.function("tf_ink2_conv_rewind"),
            .race1 = try mod.function("tf_ink2_race1"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.rows_mod.unload();
        k.mod.unload();
        k.* = undefined;
    }
};

fn int(x: anytype) i32 {
    return @intCast(x);
}

fn go(f: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, threads: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, s, a);
}

/// Embedding rows of `ids` (int32 on the device) through embed_norm into bf16 [rows, c].
pub fn embed(k: *const Kernels, s: cuda.Stream, ids: u64, rows: usize, table: u64, norm: u64, out: u64, c: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ ids, table, norm, out }) |v| a.add(v);
    a.add(int(c));
    a.add(eps);
    try go(k.embed, s, .{ .x = @intCast(rows) }, 256, &a);
}

/// The residual mix of each row and query, normed: a block a row.
pub fn mix(k: *const Kernels, s: cuda.Stream, m: MixArgs) !void {
    if (m.c > 10 * 512 or @rem(m.c, 128) != 0) return error.Inkling2MixWidth; // ops.cu: kMixCache x 512 columns, 16-byte copies
    var a: cuda.Args = .{};
    a.add(m);
    try go(if (m.rows <= 16) k.mix else k.mix_rows, s, .{ .x = @intCast(m.rows) }, 512, &a); // equal bits either way
}

/// Up to three independent mixes of one row count in one launch, blockIdx.z the role: each role's bits are mix's.
pub fn mixRoles(k: *const Kernels, s: cuda.Stream, roles: []const MixArgs) !void {
    if (roles.len == 0 or roles.len > max_queries) return error.Inkling2MixRoles;
    for (roles) |m| if (m.rows != roles[0].rows or m.c > 10 * 512 or @rem(m.c, 128) != 0) return error.Inkling2MixRoles;
    if (roles[0].rows > 16) {
        for (roles) |m| try mix(k, s, m); // wide calls stay on mix_rows, a launch a role
        return;
    }
    var r: [max_queries]MixArgs = undefined;
    for (&r, 0..) |*d, i| d.* = roles[@min(i, roles.len - 1)];
    var a: cuda.Args = .{};
    a.add(r);
    try go(k.mix_roles, s, .{ .x = @intCast(roles[0].rows), .z = @intCast(roles.len) }, 512, &a);
}

/// Tiles: fp32 outs from 32 rows; bf16 outs past 128 rows (prompt chunks), 32+ outputs, 16-byte aligned bases.
fn tiled(rows: usize, j: usize, c: usize, x: u64, w: u64, out_bf16: bool) bool {
    if (c % 128 != 0) return false;
    if (!out_bf16) return rows >= 32;
    return rows >= 129 and j >= 32 and x % 16 == 0 and w % 16 == 0;
}

/// out[row, j] = x[row] . w[j] over `c` (bf16 in), `j` outputs a row, bf16 or fp32 out, sigmoid when `sigmoid`.
pub fn rowdot(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, j: usize, c: usize, out_bf16: bool, sigmoid: bool) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ x, w, out }) |v| a.add(v);
    if (tiled(rows, j, c, x, w, out_bf16)) { // 32 x 16 tiles, the same sums in the same order
        for ([_]i32{ int(rows), int(j), int(c), @intFromBool(sigmoid) }) |v| a.add(v);
        const f = if (out_bf16) k.rowdot_tiles_bf16 else k.rowdot_tiles;
        return go(f, s, .{ .x = @intCast((rows + 31) / 32), .y = @intCast((j + 15) / 16) }, 256, &a);
    }
    for ([_]i32{ int(j), int(c), @intFromBool(out_bf16), @intFromBool(sigmoid) }) |v| a.add(v);
    try go(k.rowdot, s, .{ .x = @intCast(rows), .y = @intCast((j + 7) / 8) }, 256, &a);
}

/// The router's picks and weights from fp32 sigmoid scores [rows, experts] and the selection bias.
pub fn topk(k: *const Kernels, s: cuda.Stream, scores: u64, bias: u64, pick: u64, wts: u64, rows: usize, experts: usize, top: usize, scale: f32) !void {
    if (experts > 512 * 8 or top > 32) return error.Inkling2RouterShape;
    var a: cuda.Args = .{};
    for ([_]u64{ scores, bias, pick, wts }) |v| a.add(v);
    a.add(int(experts));
    a.add(int(top));
    a.add(scale);
    try go(k.topk, s, .{ .x = @intCast(rows) }, 512, &a);
}

/// A short convolution over each of `streams` streams' rows, their states carried (ops.cu's tf_ink2_conv).
pub fn conv(k: *const Kernels, s: cuda.Stream, c: ConvArgs, streams: usize) !void {
    var a: cuda.Args = .{};
    a.add(c);
    // a stream's rows in blocks of 32 (blocks past a stream's rows return at once)
    const z: u32 = @intCast(std.math.divCeil(i32, c.rows, 32) catch unreachable);
    try go(k.conv, s, .{ .x = @intCast(std.math.divCeil(i32, c.c, 128) catch unreachable), .y = @intCast(streams), .z = z }, 128, &a);
}

/// ops.cu's RewindSite: a conv site's slot-0 state, byte offsets of its rows in a hist and in a slot's backup, its C.
pub const RewindSite = extern struct { state: u64, hist: i64, backup: i64, c: i32, pad: i32 = 0 };

/// A slot's last window keeps `n` rows: every site's state from its backup (`backup`: the slot's) and `hist` rows.
pub fn rewind(k: *const Kernels, s: cuda.Stream, sites: u64, n_sites: usize, widest: usize, backup: u64, hist: u64, slot: usize, row0: usize, n: usize) !void {
    var a: cuda.Args = .{};
    for ([_]u64{ sites, backup, hist }) |v| a.add(v);
    for ([_]i32{ int(slot), int(row0), int(n) }) |v| a.add(v);
    try go(k.rewind, s, .{ .x = @intCast((widest + 127) / 128), .y = @intCast(n_sites) }, 128, &a);
}

/// tanh(x / cap) * cap over `n` fp32 logits in place.
pub fn softcap(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, cap: f32) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(n)));
    a.add(cap);
    try go(k.softcap, s, .{ .x = @intCast((n + 255) / 256) }, 256, &a);
}

/// Each of `rows` fp32 rows (`stride` apart): its largest value among the first `n` (lowest id on ties), id + `base`.
pub fn top1(k: *const Kernels, s: cuda.Stream, x: u64, rows: usize, stride: usize, n: usize, base: usize, val: u64, id: u64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(stride)));
    a.add(int(n));
    a.add(int(base));
    a.add(val);
    a.add(id);
    try go(k.top1, s, .{ .x = @intCast(rows) }, 1024, &a);
}

/// One fp32 row's keyed Gumbel race at position `pos` (sample.cu's keys, ids `base` + i): best score and id.
pub fn race1(k: *const Kernels, s: cuda.Stream, x: u64, n: usize, base: usize, seed: u64, inv_t: f32, pos: usize, val: u64, id: u64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(n));
    a.add(int(base));
    a.add(seed);
    a.add(inv_t);
    a.add(@as(u32, @intCast(pos)));
    a.add(val);
    a.add(id);
    try go(k.race1, s, .{ .x = 1 }, 1024, &a);
}

/// `rows` fp32 rows into bf16 rows of `out` at column `col0`: `n` columns rounded, the rest of `width` set to -inf.
pub fn tobf(k: *const Kernels, s: cuda.Stream, x: u64, rows: usize, stride: usize, n: usize, out: u64, out_stride: usize, col0: usize, width: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(stride)));
    a.add(int(n));
    a.add(out);
    a.add(@as(i64, @intCast(out_stride)));
    a.add(@as(i64, @intCast(col0)));
    a.add(int(width));
    try go(k.tobf, s, .{ .x = @intCast((width + 255) / 256), .y = @intCast(rows) }, 256, &a);
}

/// ops.cu's Ink2Words: a link step's header and ids, passed by value.
pub const Words = extern struct { w: [96]u32 };

/// The first `n` of `words` to `dst` on the stream (no host buffer to keep alive).
pub fn putWords(k: *const Kernels, s: cuda.Stream, dst: u64, n: usize, words: Words) !void {
    var a: cuda.Args = .{};
    a.add(dst);
    a.add(int(n));
    a.add(words);
    try go(k.put_words, s, .{ .x = 1 }, 128, &a);
}

test "ops: the argument structs keep ops.cu's layouts" {
    try std.testing.expectEqual(@as(usize, 8 * 8 + 8 + 8 + 3 * 8 + 8 + 6 * 4), @sizeOf(MixArgs));
    try std.testing.expectEqual(@as(usize, 9 * 8 + 2 * 8 + 7 * 4 + 4), @sizeOf(ConvArgs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(RewindSite));
}

test "ops.rowdot: which calls take the tile kernels" {
    const t = std.testing;
    try t.expect(!tiled(128, 384, 2304, 0, 0, true)); // a decode round keeps tf_ink2_rowdot
    try t.expect(tiled(129, 384, 2304, 0, 0, true));
    try t.expect(!tiled(129, 31, 2304, 0, 0, true)); // narrow outputs (the sinks) keep it too
    try t.expect(tiled(129, 32, 2304, 0, 0, true));
    try t.expect(!tiled(129, 32, 2304, 2, 0, true)); // each base alone off 16-byte alignment falls back
    try t.expect(!tiled(129, 32, 2304, 0, 2, true));
    try t.expect(!tiled(129, 32, 2304 + 1, 0, 0, true));
    try t.expect(!tiled(31, 16, 2304, 0, 0, false)); // fp32 as before: 32 rows, c % 128
    try t.expect(tiled(32, 16, 2304, 2, 2, false));
}
