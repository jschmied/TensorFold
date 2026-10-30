//! TF_KV_SHARE: a kept prompt state holds its stream's whole pages by reference in the core pool, copies the rest.
const std = @import("std");
const kv = @import("kv_pool.zig");
const Allocator = std.mem.Allocator;

/// TF_KV_SHARE=1: kept states of a paged family hold their whole pages (unset or 0: copies, as before).
pub fn sharing() bool {
    const v = std.c.getenv("TF_KV_SHARE") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Whole pages below position `at`: the ones a kept state holds, the rest it copies.
pub fn whole(at: usize, page_tokens: usize) usize {
    return at / page_tokens;
}

/// A kept state's pages: one more holder of each of the stream's first `n`, freed by `release`.
pub fn retain(gpa: Allocator, pool: *kv.Pool, lease: []const u32, n: usize) ![]u32 {
    if (n > lease.len) return error.PagesPastLease;
    const ids = try gpa.dupe(u32, lease[0..n]);
    for (ids) |id| pool.retain(id);
    return ids;
}

/// The state's holds given back; the last holder's release frees a page.
pub fn release(gpa: Allocator, pool: *kv.Pool, ids: []u32) void {
    for (ids) |id| pool.release(id);
    gpa.free(ids);
}

/// A resumed stream's empty lease starts with the kept pages: each held once more, by the lease.
pub fn adopt(pool: *kv.Pool, lease: *kv.Lease, ids: []const u32) !void {
    if (lease.ids.items.len != 0) return error.LeaseNotEmpty;
    for (ids) |id| pool.retain(id);
    try pool.adopt(lease, ids); // releases them again if it fails
}

/// A kept state's copied rows in store-held pages: byte o in chunk c = o / chunk, region c % regions, page c / regions.
pub const Blob = struct {
    ids: []const u32,
    chunk: u64, // the bytes one page holds in one region (a layer's K or V)
    regions: usize,

    /// Pages that hold `bytes` this way.
    pub fn pagesFor(bytes: u64, chunk: u64, regions: usize) usize {
        const page = chunk * regions;
        return @intCast((bytes + page - 1) / page);
    }

    /// Bytes [off, off + len) as pieces inside one chunk: f(ctx, region, page id, in-chunk offset, from `off`, len).
    pub fn each(b: Blob, off: u64, len: u64, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, u32, u64, u64, u64) anyerror!void) !void {
        var o = off;
        while (o < off + len) {
            const c = o / b.chunk;
            const within = o % b.chunk;
            const n = @min(off + len - o, b.chunk - within);
            const k: usize = @intCast(c / b.regions);
            if (k >= b.ids.len) return error.PastBlob;
            try f(ctx, @intCast(c % b.regions), b.ids[k], within, o - off, n);
            o += n;
        }
    }
};

/// The pool's reclaim over prompt_cache's `Store`: kept states leave, least valuable first, until `want` pages free.
pub fn Reclaim(comptime Store: type) type {
    return struct {
        const Self = @This();
        store: *Store,
        pool: *kv.Pool,

        pub fn wire(r: *Self) void {
            r.pool.reclaim = .{ .ctx = r, .call = call };
        }

        pub fn unwire(r: *Self) void {
            r.pool.reclaim = null;
        }

        fn call(ctx: *anyopaque, want: usize) bool {
            const r: *Self = @ptrCast(@alignCast(ctx));
            r.store.collectExpired();
            while (r.pool.available() < want) {
                const i = r.store.victimBut(null, &.{}, 0) orelse break; // borrowed states stay
                r.store.remove(i);
                r.store.counts.evicted += 1;
            }
            return r.pool.available() >= want;
        }
    };
}

const testing = std.testing;

test "a kept state holds its whole pages: the lease's end leaves them kept, its release frees them" {
    var pool = try kv.Pool.init(testing.allocator, 8, 1);
    defer pool.deinit();
    var lease: kv.Lease = .{};
    try pool.promise(&lease, 3);
    try pool.cover(&lease, 3, null);
    const ids = try retain(testing.allocator, &pool, lease.ids.items, whole(2 * 16 + 5, 16));
    try testing.expectEqual(@as(usize, 2), ids.len);
    pool.finish(&lease);
    try testing.expectEqual(@as(usize, 6), pool.available()); // the third page went back, two are kept
    try testing.expectEqual(@as(usize, 2), pool.census().kept);
    release(testing.allocator, &pool, ids);
    try testing.expect(pool.idle());
    try testing.expectEqual(@as(usize, 8), pool.available());
}

test "a resume adopts the kept pages at its lease's head; both holders release before the page is free" {
    var pool = try kv.Pool.init(testing.allocator, 8, 1);
    defer pool.deinit();
    var first: kv.Lease = .{};
    try pool.promise(&first, 2);
    try pool.cover(&first, 2, null);
    const ids = try retain(testing.allocator, &pool, first.ids.items, 2);
    pool.finish(&first);
    var next: kv.Lease = .{};
    try pool.promise(&next, 1);
    try adopt(&pool, &next, ids);
    try pool.cover(&next, 3, null);
    try testing.expectEqualSlices(u32, ids, next.ids.items[0..2]);
    try testing.expectEqual(@as(u32, 2), pool.holders(ids[0]));
    release(testing.allocator, &pool, ids); // the store lets go first: the lease still maps them
    try testing.expectEqual(@as(u32, 1), pool.holders(next.ids.items[0]));
    pool.finish(&next);
    try testing.expect(pool.idle());
}

test "adopting into a lease that already maps pages is refused" {
    var pool = try kv.Pool.init(testing.allocator, 4, 1);
    defer pool.deinit();
    var lease: kv.Lease = .{};
    try pool.promise(&lease, 1);
    try pool.cover(&lease, 1, null);
    try testing.expectError(error.LeaseNotEmpty, adopt(&pool, &lease, &.{2}));
    pool.finish(&lease);
}

/// A store of kept states, each holding pages: the oldest unborrowed leaves first, as prompt_cache's victimBut picks.
const Fake = struct {
    pool: *kv.Pool,
    held: std.ArrayList([]u32) = .empty,
    borrowed: usize = 0, // the first this many are borrowed
    counts: struct { evicted: usize = 0 } = .{},

    pub fn collectExpired(_: *Fake) void {}

    pub fn victimBut(f: *const Fake, _: ?*anyopaque, _: []const u32, _: u32) ?usize {
        return if (f.held.items.len > f.borrowed) f.borrowed else null;
    }

    pub fn remove(f: *Fake, i: usize) void {
        release(testing.allocator, f.pool, f.held.orderedRemove(i));
    }
};

test "a promise past the free pages sheds kept states, oldest first, borrowed ones never" {
    var pool = try kv.Pool.init(testing.allocator, 6, 1);
    defer pool.deinit();
    var fake: Fake = .{ .pool = &pool, .borrowed = 1 };
    defer fake.held.deinit(testing.allocator);
    for (0..3) |_| { // three finished streams, two pages kept each
        var l: kv.Lease = .{};
        try pool.promise(&l, 2);
        try pool.cover(&l, 2, null);
        try fake.held.append(testing.allocator, try retain(testing.allocator, &pool, l.ids.items, 2));
        pool.finish(&l);
    }
    var r: Reclaim(Fake) = .{ .store = &fake, .pool = &pool };
    r.wire();
    defer r.unwire();
    var l: kv.Lease = .{};
    try pool.promise(&l, 3); // two states leave (four pages free); the borrowed first stays
    try testing.expectEqual(@as(usize, 1), fake.held.items.len);
    try testing.expectEqual(@as(usize, 2), fake.counts.evicted);
    try testing.expectError(error.LaneMemory, pool.promise(&l, 2)); // only the borrowed state's pages are left
    pool.finish(&l);
    fake.borrowed = 0;
    fake.remove(0);
    try testing.expect(pool.idle());
}

const Pieces = struct {
    got: *std.ArrayList([5]u64),
    fn add(p: Pieces, region: usize, id: u32, within: u64, from: u64, n: u64) anyerror!void {
        try p.got.append(testing.allocator, .{ region, id, within, from, n });
    }
};

test "a blob's bytes walk its regions a chunk at a time, then its next page; a run splits at chunk edges" {
    const b: Blob = .{ .ids = &.{ 7, 3 }, .chunk = 10, .regions = 2 };
    try testing.expectEqual(@as(usize, 2), Blob.pagesFor(21, 10, 2));
    var got: std.ArrayList([5]u64) = .empty;
    defer got.deinit(testing.allocator);
    try b.each(5, 30, Pieces{ .got = &got }, Pieces.add); // bytes 5..35: chunks 0 (page 7, K), 1 (page 7, V), 2 (page 3, K), 3 (page 3, V)
    try testing.expectEqualSlices([5]u64, &.{ .{ 0, 7, 5, 0, 5 }, .{ 1, 7, 0, 5, 10 }, .{ 0, 3, 0, 15, 10 }, .{ 1, 3, 0, 25, 5 } }, got.items);
    try testing.expectError(error.PastBlob, b.each(39, 2, Pieces{ .got = &got }, Pieces.add));
}
