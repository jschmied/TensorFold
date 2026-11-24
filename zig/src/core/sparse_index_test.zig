//! sparse_index's properties on random keys, each with a planted fault its check must catch.
const std = @import("std");
const si = @import("sparse_index.zig");
const testing = std.testing;
const gpa = testing.allocator;

const dim = 8;
const heads = 3;

fn randoms(seed: u64, n: usize) ![]f64 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try gpa.alloc(f64, n);
    for (out) |*v| v.* = prng.random().floatNorm(f64);
    return out;
}

const Rig = struct {
    keys: []f64,
    q: []f64,
    fn init(seed: u64, n: usize) !Rig {
        return .{ .keys = try randoms(seed, n * dim), .q = try randoms(seed + 1, heads * dim) };
    }
    fn deinit(r: Rig) void {
        gpa.free(r.keys);
        gpa.free(r.q);
    }
    fn sel(r: Rig, p: si.Policy, t: usize) !si.Selection {
        return si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, null);
    }
};

const flat: si.Policy = .{ .topk = 16, .sink = 2, .local = 4 };
const two: si.Policy = .{ .topk = 16, .block = 4, .pool_blocks = 6, .sink = 2, .local = 4 };

test "sparse index: every selection is causal, ascending, without repeats, keeps its mandatory keys and its budget" {
    const r = try Rig.init(1, 200);
    defer r.deinit();
    for ([_]si.Policy{ flat, two, .{ .topk = 16, .block = 4, .pool_blocks = 6, .sink = 2, .local = 4, .mandatory_in_topk = true } }) |p| {
        for ([_]usize{ 0, 1, 5, 17, 63, 64, 65, 199 }) |t| {
            const s = try r.sel(p, t);
            defer s.deinit(gpa);
            try si.check(p, t, s.keys);
        }
    }
}

test "sparse index: the check catches a future key, a repeat, a missing sink key and one key over budget (planted)" {
    const r = try Rig.init(2, 100);
    defer r.deinit();
    const s = try r.sel(flat, 60);
    defer s.deinit(gpa);
    const bad = try gpa.alloc(u32, s.keys.len + 1);
    defer gpa.free(bad);
    @memcpy(bad[0..s.keys.len], s.keys);
    bad[s.keys.len] = 61; // the next position: the future
    try testing.expectError(error.Future, si.check(flat, 60, bad));
    bad[s.keys.len] = s.keys[s.keys.len - 1]; // the last one again
    try testing.expectError(error.Unsorted, si.check(flat, 60, bad));
    try testing.expectError(error.MissingMandatory, si.check(flat, 60, s.keys[1..])); // key 0, a sink key
    const over = try gpa.alloc(u32, 61);
    defer gpa.free(over);
    for (over, 0..) |*o, i| o.* = @intCast(i);
    try testing.expectError(error.OverBudget, si.check(flat, 60, over));
}

test "sparse index: at most topk + mandatory keys in context: every key is attended, whatever the scores or stages" {
    const r = try Rig.init(3, 40);
    defer r.deinit();
    for ([_]si.Policy{ flat, two, .{ .topk = 22, .block = 4, .pool_blocks = 6 } }) |p| {
        const most = if (p.mandatory_in_topk) p.topk else p.topk + p.sink + p.local;
        const t = @min(most, 22) - 1; // 22 keys: within every policy's reach (two-stage: 6 blocks x 4 >= 22)
        const s = try r.sel(p, t);
        defer s.deinit(gpa);
        try testing.expectEqual(t + 1, s.keys.len);
    }
    // planted: one key past the reach of a flat policy without mandatory keys leaves exactly one out
    const p: si.Policy = .{ .topk = 16 };
    const s = try r.sel(p, 16);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(usize, 16), s.keys.len);
}

test "sparse index: blocks with every block kept, or one key a block, are the flat search" {
    const r = try Rig.init(4, 150);
    defer r.deinit();
    for ([_]usize{ 3, 40, 149 }) |t| {
        const a = try r.sel(flat, t);
        defer a.deinit(gpa);
        var p = flat;
        p.block = 4;
        p.pool_blocks = 0;
        const b = try r.sel(p, t);
        defer b.deinit(gpa);
        try testing.expectEqualSlices(u32, a.keys, b.keys);
        // the flat search rescoring a pool of every key agrees too
        const c = try si.select(gpa, flat, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, b.pool);
        defer c.deinit(gpa);
        try testing.expectEqualSlices(u32, a.keys, c.keys);
    }
}

test "sparse index: two stages: tokens come only from the kept blocks or the mandatory ranges" {
    const r = try Rig.init(5, 300);
    defer r.deinit();
    const t = 250;
    const s = try r.sel(two, t);
    defer s.deinit(gpa);
    try testing.expect(s.pool.len <= two.pool_blocks * two.block);
    for (s.keys) |k| {
        const in_pool = std.mem.indexOfScalar(u32, s.pool, k) != null;
        try testing.expect(in_pool or k < two.sink or k + two.local > t);
    }
}

test "sparse index: the selection for position t ignores keys after t (chunked prompts, rolled-back drafts)" {
    const r = try Rig.init(6, 260);
    defer r.deinit();
    for ([_]si.Policy{ flat, two }) |p| for ([_]usize{ 10, 99, 200 }) |t| {
        const whole = try r.sel(p, t);
        defer whole.deinit(gpa);
        const cut = try si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = r.keys[0 .. (t + 1) * dim], .dim = dim }, t, null);
        defer cut.deinit(gpa);
        try testing.expectEqualSlices(u32, whole.keys, cut.keys);
        // drafts appended, then rejected: new keys past t change nothing for t
        const more = try gpa.dupe(f64, r.keys);
        defer gpa.free(more);
        for (more[(t + 1) * dim ..]) |*v| v.* *= -3;
        const drafted = try si.select(gpa, p, .{ .q = r.q, .heads = heads }, .{ .k = more, .dim = dim }, t, null);
        defer drafted.deinit(gpa);
        try testing.expectEqualSlices(u32, whole.keys, drafted.keys);
    };
}

test "sparse index: reuse: final selection is the owner's positions; a pool rescored with another query can differ" {
    const r = try Rig.init(7, 400);
    defer r.deinit();
    const t = 380;
    const owner = try r.sel(two, t);
    defer owner.deinit(gpa);
    // final reuse takes the owner's keys as they are; pool reuse rescores the owner's candidates with its own query
    const other = try randoms(99, heads * dim);
    defer gpa.free(other);
    const follower = try si.select(gpa, two, .{ .q = other, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, owner.pool);
    defer follower.deinit(gpa);
    try si.check(two, t, follower.keys);
    for (follower.keys) |k| try testing.expect(std.mem.indexOfScalar(u32, owner.pool, k) != null or k < two.sink or k + two.local > t);
    try testing.expect(!std.mem.eql(u32, follower.keys, owner.keys)); // a different query picks other tokens of the pool
    // the owner's own query over its own pool gives back its final selection
    const same = try si.select(gpa, two, .{ .q = r.q, .heads = heads }, .{ .k = r.keys, .dim = dim }, t, owner.pool);
    defer same.deinit(gpa);
    try testing.expectEqualSlices(u32, owner.keys, same.keys);
}

test "sparse index: scores: ReLU drops negative heads, max takes the strongest, ties go to the lower position" {
    const q = [_]f64{ 1, 0, -1, 0 }; // two heads of dim 2
    const key = [_]f64{ 2, 0 };
    try testing.expectEqual(@as(f64, 2), si.score(.{ .topk = 1 }, .{ .q = &q, .heads = 2 }, &key));
    try testing.expectEqual(@as(f64, 0), si.score(.{ .topk = 1, .phi = .identity }, .{ .q = &q, .heads = 2 }, &key));
    try testing.expectEqual(@as(f64, 2), si.score(.{ .topk = 1, .reduce = .max, .phi = .identity }, .{ .q = &q, .heads = 2 }, &key));
    const w = [_]f64{ 0.5, 3 };
    try testing.expectEqual(@as(f64, 1), si.score(.{ .topk = 1 }, .{ .q = &q, .heads = 2, .weights = &w }, &key));
    // equal keys everywhere: the lowest positions win
    const keys: [20]f64 = @splat(1);
    const s = try si.select(gpa, .{ .topk = 3 }, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 9, null);
    defer s.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, s.keys);
}

test "sparse index: summaries: mean of unit keys and mean then compare differ on a block with one loud key" {
    // block 0: one key of norm 10 and three of norm 0.1 along another axis; block 1: middling keys along that axis
    const keys = [_]f64{ 10, 0, 0, 0.1, 0, 0.1, 0, 0.1, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 };
    const q = [_]f64{ 0, 1 };
    const base: si.Policy = .{ .topk = 4, .block = 4, .pool_blocks = 1 };
    var unit = base;
    unit.norm_first = true;
    const a = try si.select(gpa, base, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 7, null);
    defer a.deinit(gpa);
    const b = try si.select(gpa, unit, .{ .q = &q, .heads = 1 }, .{ .k = &keys, .dim = 2 }, 7, null);
    defer b.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7 }, a.pool); // plain mean: block 1 (0.075 against 0.7)
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, b.pool); // unit keys: block 0 (1.06 against 1.0)
}
