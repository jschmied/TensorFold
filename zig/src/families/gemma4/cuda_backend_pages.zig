//! Gemma 4's backend over pool pages: a lane's pages as its context grows, and TF_KV_SHARE's kept states.
const std = @import("std");
const fwd = @import("cuda_forward.zig");
const snap = @import("cuda_snap.zig");
const kept = @import("core").kv_kept;
const backend = @import("cuda_backend.zig");
const Gemma = backend.Gemma;
const Lane = backend.Lane;

/// Positions a state kept at `at` holds as whole pages (0: it copies them all).
pub fn heldBelow(x: *const Gemma, at: usize) usize {
    return if (x.share) kept.whole(at, fwd.page_tokens) * fwd.page_tokens else 0;
}

/// The most pool pages a state kept at `at` holds for its copied rows (none unless sharing); asked before attach.
pub fn keptPages(x: *const Gemma, at: usize) usize {
    return if (kept.sharing()) snap.blobPages(x.m, at, 0) else 0;
}

/// The pool's free pages in a seeded random order, leased in that order (a check's hook).
pub fn shufflePages(x: *Gemma, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    if (x.pool) |*p| prng.random().shuffle(u32, p.free[0..p.nfree]);
}

/// `n` free pool pages the store holds for a state's copied rows (none unless sharing); NoCachePages: all leased.
pub fn holdBlob(x: *Gemma, n: usize) ![]u32 {
    if (!x.share or n == 0) return &.{};
    const ids = try x.gpa.alloc(u32, n);
    if (!x.pool.?.hold(ids)) {
        x.gpa.free(ids);
        return error.NoCachePages;
    }
    return ids;
}

/// A dropped state's held pages and copied-row pages back to the pool, after any restore's queued copies out of them.
pub fn releaseKept(x: *Gemma, k: *snap.Saved) void {
    if (k.pages.len > 0 or k.blob.len > 0) x.s.synchronize() catch {};
    if (k.pages.len > 0) {
        kept.release(x.gpa, &x.pool.?, k.pages);
        k.pages = &.{};
    }
    if (k.blob.len > 0) {
        kept.release(x.gpa, &x.pool.?, k.blob);
        k.blob = &.{};
    }
}

/// A state's held pages into a lane that did not adopt them: their rows copied into the lane's own pages.
pub fn putHeld(x: *Gemma, l: *Lane, k: *const snap.Saved) !void {
    if (k.pages.len == 0) return;
    const ids = l.lease.ids.items;
    if (ids.len >= k.pages.len and std.mem.eql(u32, ids[0..k.pages.len], k.pages)) return; // adopted: nothing to copy
    try reach(x, l, k.from);
    try snap.copyPages(x.m, x.s, l.slot, k.pages);
}

/// A new lane's table starts with the kept state's pages, held once more by its lease (rows below `from`, unwritten).
pub fn adoptHeld(x: *Gemma, l: *Lane, k: *const snap.Saved) !void {
    try kept.adopt(&x.pool.?, &l.lease, k.pages);
    try x.m.mapPages(x.s, l.slot, k.pages, 0);
}

const Map = struct {
    x: *Gemma,
    slot: usize,
    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const m: *const Map = @ptrCast(@alignCast(ctx));
        try m.x.m.mapPages(m.x.s, m.slot, ids[from..], from);
    }
};

/// The lane's pages cover positions [0, end): pages taken from its promise as its context grows.
pub fn reach(x: *Gemma, l: *Lane, end: usize) !void {
    const p = if (x.pool) |*p| p else return;
    const n = backend.pagesOf(end);
    if (n <= l.lease.ids.items.len) return;
    var map: Map = .{ .x = x, .slot = l.slot };
    try p.cover(&l.lease, n, .{ .ctx = &map, .write = Map.write });
}
