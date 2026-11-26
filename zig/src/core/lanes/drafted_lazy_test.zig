//! Lazy absorb: depth-0 rows owed, absorbed before the next hold, leave the eager drafter's state (ring wrap, limits).
const std = @import("std");
const be = @import("backend.zig");
const dr = @import("drafter.zig");
const Stream = @import("stream.zig").Stream;
const Drafted = @import("drafted.zig").Drafted;
const FakeDrafter = @import("drafted_test.zig").FakeDrafter;

const gpa = std.testing.allocator;

/// A target holding each position's token as its tapped row, in a ring of `keep` positions, `view` rows a read.
const Ring = struct {
    const keep = 256;
    const view = 64;
    tokens: [1024]u32 = undefined,
    written: u64 = 0, // positions [written - keep, written) readable
    refused: u32 = 0,

    fn cast(ptr: *anyopaque) *Ring {
        return @ptrCast(@alignCast(ptr));
    }
    fn features(ptr: *anyopaque, _: *Stream, _: []const u32, start: u64, n: u32) anyerror!be.Features {
        const r = cast(ptr);
        if (n > view or start + n > r.written or start < r.written -| keep) {
            r.refused += 1;
            return error.RowsNotHeld;
        }
        return .{ .buffer = @intFromPtr(&r.tokens), .offset = start * 4, .rows = n, .row_bytes = 4, .space = .host, .dtype = .u32 };
    }
    fn prefill(_: *anyopaque, _: *Stream) anyerror!void {}
    fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return 0;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return 0;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return 0;
    }
    fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {}
    fn keepRows(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    fn release(_: *anyopaque, _: *Stream) void {}
    fn backend(r: *Ring) be.Backend {
        return .{ .ptr = r, .vtable = &.{ .prefill = prefill, .features = features, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keepRows, .draft = draft, .release = release } };
    }
};

/// What one run leaves: the drafter's rows by position, its held drafts at every hold, its absorb calls.
const Run = struct { history: [1024]u32 = undefined, len: usize = 0, held: [128][3]u32 = undefined, holds: usize = 0, absorbs: u32 = 0, refused: u32 = 0 };

/// Rounds of 1-3 kept rows from position 8 to past 600: `holds_every`-th round holds 3 drafts, the rest depth 0.
fn run(lazy: u32, holds_every: usize) !Run {
    var ring: Ring = .{};
    for (&ring.tokens, 0..) |*t, p| t.* = @intCast((p * 7 + 3) % 97);
    var fd: FakeDrafter = .{};
    defer fd.deinit();
    fd.facts_.lazy_rows = lazy;
    var x = try Drafted.init(gpa, ring.backend(), fd.drafter());
    defer x.deinit();
    var s = try Stream.init(gpa, .{ .id = "lazy", .prompt = ring.tokens[0..8], .max_new = 900 });
    defer s.deinit(gpa);
    try x.adopt(&s);
    defer x.backend().release(&s);
    var out: Run = .{};
    var start: u64 = 8;
    var round: usize = 0;
    while (start < 600) : (round += 1) {
        const kept: usize = 1 + (round * 5 + 1) % 3;
        ring.written = start + kept; // the verify's kept rows are on the target (a keep rolled the rest back)
        const depth: u32 = if (round % holds_every == holds_every - 1) 3 else 0;
        const rows = [_]u32{ 0, 1, 2 };
        try x.backend().draft(&.{.{ .stream = &s, .follow = ring.tokens[start + 1 ..][0..kept], .rows = rows[0..kept], .start = start, .position = start + kept + 1, .depth = depth }});
        if (depth > 0) {
            const l = fd.lanes.getPtr(&s).?;
            @memcpy(&out.held[out.holds], l.held.items[0..3]);
            out.holds += 1;
        }
        start += kept;
    }
    // the stream's last rows: a final hold pays what is owed, so both runs end on the same state
    ring.written = start + 1;
    try x.backend().draft(&.{.{ .stream = &s, .follow = ring.tokens[start + 1 ..][0..1], .rows = &.{0}, .start = start, .position = start + 2, .depth = 1 }});
    const l = fd.lanes.getPtr(&s).?;
    out.len = l.history.items.len;
    @memcpy(out.history[0..out.len], l.history.items);
    out.absorbs = fd.absorbs;
    out.refused = ring.refused;
    return out;
}

fn expectSame(eager: Run, lazy: Run) !void {
    try std.testing.expectEqual(eager.len, lazy.len);
    try std.testing.expectEqualSlices(u32, eager.history[0..eager.len], lazy.history[0..lazy.len]);
    try std.testing.expectEqual(eager.holds, lazy.holds);
    for (eager.held[0..eager.holds], lazy.held[0..lazy.holds]) |a, b| try std.testing.expectEqualSlices(u32, &a, &b);
    try std.testing.expectEqual(@as(u32, 0), lazy.refused);
}

test "lazy absorb: owed depth-0 rows absorbed before the next hold give the eager drafter's rows and drafts" {
    const eager = try run(0, 4);
    const lazy = try run(dr.owed_max, 4);
    try expectSame(eager, lazy);
    try std.testing.expect(lazy.absorbs < eager.absorbs); // depth-0 rounds made no absorb call of their own
}

test "lazy absorb: a stream that never holds flushes at the owed limit, past the target ring's wrap" {
    const eager = try run(0, 1000);
    for ([_]u32{ 47, 48, 49, dr.owed_max }) |limit| {
        const lazy = try run(limit, 1000);
        try expectSame(eager, lazy);
        // about one flush a limit's rows: 600 positions, never a view past the target's ring
        try std.testing.expect(lazy.absorbs <= 600 / (limit - 2) + 2);
    }
}

test "lazy absorb: through the engine, a lazy drafter's prompt rows at the first draft and owed rows give the same rounds" {
    const dt = @import("drafted_test.zig");
    const cases = [_]dt.Case{ .{ .prompt = &dt.p1, .max_new = 60 }, .{ .prompt = &dt.long, .sampling = dt.sampled, .max_new = 50 } };
    const eager = try dt.run(&cases, .wrapped, true);
    defer dt.freeOut(eager);
    dt.lazy_rows = dr.owed_max;
    defer dt.lazy_rows = 0;
    const lazy = try dt.run(&cases, .wrapped, true);
    defer dt.freeOut(lazy);
    for (eager.tokens, lazy.tokens) |a, b| try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expectEqual(eager.drafted, lazy.drafted);
    try std.testing.expectEqual(eager.accepted, lazy.accepted);
    try std.testing.expect(lazy.head.prompt_rows > 0); // the prompts' rows reached the drafter, at the first draft
}
