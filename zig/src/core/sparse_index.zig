//! A sparse-attention indexer's key selection in float64, for any model's policy: the reference its kernels follow.
const std = @import("std");

/// The activation on each head's dot product before heads are combined.
pub const Phi = enum { relu, identity };
/// How heads combine into one score.
pub const Reduce = enum { sum, max };
/// A block's summary key from its keys (each key first scaled to unit RMS when `norm_first`).
pub const Summary = enum { mean, max, first, last };
/// What a later layer takes from the layer that selected.
pub const Reuse = enum { none, final, pool };

pub const Policy = struct {
    topk: usize, // tokens selected a query
    block: usize = 1, // keys a block summary covers; 1: one stage, no summaries
    pool_blocks: usize = 0, // blocks the first stage keeps; 0: all
    sink: usize = 0, // the first keys, always attended
    local: usize = 0, // the most recent keys (the query's own included), always attended
    mandatory_in_topk: bool = false, // sink and local keys count inside `topk` (else they come on top)
    phi: Phi = .relu,
    reduce: Reduce = .sum,
    summary: Summary = .mean,
    norm_first: bool = false, // summaries average unit-RMS keys
    open_block: bool = true, // the last, incomplete block is summarised from the keys it has (else left out)
};

/// One query position's index heads [heads, dim] and optional per-head weights.
pub const Query = struct { q: []const f64, heads: usize, weights: ?[]const f64 = null };

/// Index keys [n, dim], position i at row i.
pub const Keys = struct { k: []const f64, dim: usize };

/// sum (or max) over heads of w_h * phi(q_h . key).
pub fn score(p: Policy, q: Query, key: []const f64) f64 {
    const dim = key.len;
    var acc: f64 = if (p.reduce == .max) -std.math.inf(f64) else 0;
    for (0..q.heads) |h| {
        var dot: f64 = 0;
        for (q.q[h * dim ..][0..dim], key) |a, b| dot += a * b;
        if (p.phi == .relu) dot = @max(dot, 0);
        const w = if (q.weights) |ws| ws[h] else 1;
        acc = if (p.reduce == .max) @max(acc, w * dot) else acc + w * dot;
    }
    return acc;
}

const Scored = struct { pos: u32, s: f64 };

/// Higher score first; on equal scores the lower position.
fn before(_: void, a: Scored, b: Scored) bool {
    return a.s > b.s or (a.s == b.s and a.pos < b.pos);
}

/// The positions a query attends and the candidates its first stage kept (for pool reuse), both ascending.
pub const Selection = struct {
    keys: []u32,
    pool: []u32,

    pub fn deinit(s: Selection, gpa: std.mem.Allocator) void {
        gpa.free(s.keys);
        gpa.free(s.pool);
    }
};

fn mandatory(p: Policy, t: usize, i: usize) bool {
    return i < p.sink or i + p.local > t;
}

/// Block b's summary over positions [b * block, min((b + 1) * block, t + 1)) into `out`.
fn summarise(p: Policy, keys: Keys, b: usize, t: usize, out: []f64) void {
    const lo = b * p.block;
    const hi = @min(lo + p.block, t + 1);
    @memset(out, if (p.summary == .max) -std.math.inf(f64) else 0);
    for (lo..hi) |i| {
        const k = keys.k[i * keys.dim ..][0..keys.dim];
        var r: f64 = 1;
        if (p.norm_first) {
            var ss: f64 = 0;
            for (k) |v| ss += v * v;
            r = if (ss > 0) 1 / @sqrt(ss / @as(f64, @floatFromInt(keys.dim))) else 0;
        }
        const take = switch (p.summary) {
            .first => i == lo,
            .last => i + 1 == hi,
            else => true,
        };
        if (take) for (out, k) |*o, v| {
            o.* = if (p.summary == .max) @max(o.*, v * r) else o.* + v * r;
        };
    }
    if (p.summary == .mean) for (out) |*o| {
        o.* /= @floatFromInt(hi - lo);
    };
}

/// The query at position t (keys 0..t) under `p`; `pool_from`: rescore that pool instead of searching.
pub fn select(gpa: std.mem.Allocator, p: Policy, q: Query, keys: Keys, t: usize, pool_from: ?[]const u32) !Selection {
    if (p.topk == 0 or p.block == 0 or keys.k.len < (t + 1) * keys.dim) return error.BadPolicy;
    // the candidates: every causal key, the first stage's blocks, or the pool handed down
    var cands: std.ArrayList(u32) = .empty;
    defer cands.deinit(gpa);
    if (pool_from) |pool| {
        for (pool) |i| if (i <= t) try cands.append(gpa, i);
    } else if (p.block == 1) {
        for (0..t + 1) |i| try cands.append(gpa, @intCast(i));
    } else {
        const full = (t + 1) / p.block;
        const blocks = if (p.open_block) (t + p.block) / p.block else full;
        const sum = try gpa.alloc(f64, keys.dim);
        defer gpa.free(sum);
        var scored = try gpa.alloc(Scored, blocks);
        defer gpa.free(scored);
        for (0..blocks) |b| {
            summarise(p, keys, b, t, sum);
            scored[b] = .{ .pos = @intCast(b), .s = score(p, q, sum) };
        }
        std.mem.sort(Scored, scored, {}, before);
        const keep = if (p.pool_blocks == 0) blocks else @min(blocks, p.pool_blocks);
        for (scored[0..keep]) |s| for (s.pos * p.block..@min((s.pos + 1) * p.block, t + 1)) |i| try cands.append(gpa, @intCast(i));
        // keys of a block left out (the open one) are only reachable as mandatory keys
    }
    std.mem.sort(u32, cands.items, {}, std.sort.asc(u32));
    const pool = try gpa.dupe(u32, cands.items);
    errdefer gpa.free(pool);
    // the second stage: tokens by their own score; mandatory keys apart unless they count inside topk
    var scored: std.ArrayList(Scored) = .empty;
    defer scored.deinit(gpa);
    for (cands.items) |i| if (p.mandatory_in_topk or !mandatory(p, t, i)) {
        try scored.append(gpa, .{ .pos = i, .s = score(p, q, keys.k[i * keys.dim ..][0..keys.dim]) });
    };
    std.mem.sort(Scored, scored.items, {}, before);
    const in = try gpa.alloc(bool, t + 1);
    defer gpa.free(in);
    var budget = p.topk;
    var count: usize = 0;
    for (in, 0..) |*x, i| {
        x.* = mandatory(p, t, i);
        if (x.*) count += 1;
        if (x.* and p.mandatory_in_topk) budget -|= 1;
    }
    var taken: usize = 0;
    for (scored.items) |s| {
        if (taken == budget) break;
        if (in[s.pos]) continue;
        in[s.pos] = true;
        taken += 1;
    }
    const out = try gpa.alloc(u32, count + taken);
    var j: usize = 0;
    for (in, 0..) |x, i| if (x) {
        out[j] = @intCast(i);
        j += 1;
    };
    return .{ .keys = out, .pool = pool };
}

/// What a selection must be: ascending, no repeats, never past t, every mandatory key in, at most the budget.
pub fn check(p: Policy, t: usize, keys: []const u32) error{ Unsorted, Future, MissingMandatory, OverBudget }!void {
    for (keys, 0..) |k, j| {
        if (j > 0 and keys[j - 1] >= k) return error.Unsorted;
        if (k > t) return error.Future;
    }
    var mand: usize = 0;
    for (0..t + 1) |i| if (mandatory(p, t, i)) {
        mand += 1;
        if (std.sort.binarySearch(u32, keys, @as(u32, @intCast(i)), struct {
            fn cmp(a: u32, b: u32) std.math.Order {
                return std.math.order(a, b);
            }
        }.cmp) == null) return error.MissingMandatory;
    };
    const most = if (p.mandatory_in_topk) @max(p.topk, mand) else p.topk + mand;
    if (keys.len > most) return error.OverBudget;
}

test {
    _ = @import("sparse_index_test.zig");
}
