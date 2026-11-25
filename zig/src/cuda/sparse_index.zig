//! sparse_index.cu for every CUDA family: indexer scores of query rows over keys, and keys' block summaries.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Key width; keys a scores block; query heads a row at most (their bf16 rows fill 16 KiB of shared memory).
pub const dim = 128;
const keys_a_block = 64;
pub const max_heads = 64;

pub const Phi = enum(i32) { identity = 0, relu = 1 };
pub const Reduce = enum(i32) { sum = 0, max = 1 };
pub const Summary = enum(i32) { mean = 0, max = 1, first = 2, last = 3 };

/// rows of h query heads over n keys of hk key heads, `ks` elements apart; scores `ld` apart (ld >= n).
pub const Scores = struct { rows: usize, n: usize, h: usize, hk: usize, ks: usize, ld: usize, scale: f32, phi: Phi = .relu, reduce: Reduce = .sum };

/// n keys `ks` elements apart into ceil(n / block) summaries `os` apart.
pub const Blocks = struct { n: usize, block: usize, ks: usize, os: usize, summary: Summary = .mean, unit: bool = false };

pub fn checkScores(s: Scores) error{IndexShape}!void {
    if (s.h == 0 or s.h > max_heads or s.hk == 0 or s.h % s.hk != 0 or s.ks < s.hk * dim or s.ks % 8 != 0) return error.IndexShape;
    if (s.ld < s.n or s.n > std.math.maxInt(i32) or s.rows > 65535 or s.ld > std.math.maxInt(i64)) return error.IndexShape;
}

pub fn checkBlocks(b: Blocks) error{IndexShape}!void {
    if (b.block == 0 or b.block > std.math.maxInt(i32) or b.ks < dim or b.os < dim) return error.IndexShape;
    if (b.n > std.math.maxInt(i64) or (b.n + b.block - 1) / b.block > std.math.maxInt(i32)) return error.IndexShape;
}

pub const SparseIndex = struct {
    mod: Module,
    scores: Function,
    blocks: Function,

    pub fn load(d: *const Driver) !SparseIndex {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.sparse_index);
        errdefer mod.unload();
        return .{ .mod = mod, .scores = try mod.function("tf_index_scores"), .blocks = try mod.function("tf_index_blocks") };
    }

    pub fn unload(x: *SparseIndex) void {
        x.mod.unload();
    }

    /// q bf16 [rows, h, 128]; w f32 [rows, h] or 0; counts i32 [rows] or 0; out f32 [rows, ld], -inf past a count.
    pub fn runScores(x: *const SparseIndex, s: Stream, q: u64, k: u64, w: u64, counts: u64, out: u64, sh: Scores) !void {
        try checkScores(sh);
        if (sh.rows == 0 or sh.n == 0) return;
        var a: launch_.Args = .{};
        for ([_]u64{ q, k }) |v| a.add(v);
        a.add(@as(i64, @intCast(sh.ks)));
        for ([_]u64{ w, counts, out }) |v| a.add(v);
        a.add(@as(i64, @intCast(sh.ld)));
        for ([_]usize{ sh.n, sh.h, sh.hk }) |v| a.add(@as(i32, @intCast(v)));
        a.add(sh.scale);
        a.add(@backingInt(sh.phi));
        a.add(@backingInt(sh.reduce));
        const grid: launch_.Dim3 = .{ .x = @intCast((sh.n + keys_a_block - 1) / keys_a_block), .y = @intCast(sh.rows), .z = 1 };
        const smem: u32 = @intCast(sh.h * dim * 2 + sh.h * 4);
        try launch_.launch(x.scores, .{ .grid = grid, .block = .{ .x = keys_a_block }, .shared = smem }, s, &a);
    }

    /// k bf16 key rows; out bf16 [ceil(n / block), os] (the last block summarises the keys it has).
    pub fn runBlocks(x: *const SparseIndex, s: Stream, k: u64, out: u64, b: Blocks) !void {
        try checkBlocks(b);
        if (b.n == 0) return;
        var a: launch_.Args = .{};
        a.add(k);
        a.add(@as(i64, @intCast(b.ks)));
        a.add(out);
        a.add(@as(i64, @intCast(b.os)));
        a.add(@as(i64, @intCast(b.n)));
        a.add(@as(i32, @intCast(b.block)));
        a.add(@backingInt(b.summary));
        a.add(@as(i32, @intFromBool(b.unit)));
        const grid: launch_.Dim3 = .{ .x = @intCast((b.n + b.block - 1) / b.block), .y = 1, .z = 1 };
        try launch_.launch(x.blocks, .{ .grid = grid, .block = .{ .x = dim } }, s, &a);
    }
};

test "checks take Kolibri-sized indexers and refuse heads past 64, partial groups and overlapping key heads" {
    const ok: Scores = .{ .rows = 4, .n = 16384, .h = 64, .hk = 1, .ks = dim, .ld = 16384, .scale = 1 };
    try checkScores(ok);
    var x = ok;
    x.h = 65;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.h = 48;
    x.hk = 5;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.hk = 2; // two key heads in rows one key head wide
    try std.testing.expectError(error.IndexShape, checkScores(x));
    x = ok;
    x.ld = 100;
    try std.testing.expectError(error.IndexShape, checkScores(x));
    try checkBlocks(.{ .n = 16384, .block = 64, .ks = dim, .os = dim });
    try std.testing.expectError(error.IndexShape, checkBlocks(.{ .n = 16, .block = 0, .ks = dim, .os = dim }));
}
