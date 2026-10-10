//! The tree attention's host plan (kernels/attention.py plan_host): rows, streams, items and parents in one int32 list.
const std = @import("std");

pub const chunk = 512; // keys a partial slot covers
pub const group = 4; // chunks folded into one fp32 partial
pub const span = chunk * group;
pub const min_grouped = 64; // groups times window rows needed to keep enough programs busy
pub const max_nodes = 128;
pub const query_tile = 16;

/// Whole committed groups fold in their programs only when groups times rows fills the GPU.
pub fn groups(p: usize, w: usize) usize {
    const g = p / span;
    return if (g * w >= min_grouped) g else 0;
}

/// One slot per folded group, then one per chunk through the window's last key.
pub fn slots(p: usize, w: usize) usize {
    const g = groups(p, w);
    return g + (p + w + chunk - 1) / chunk - g * group;
}

/// A stream's window: its local parents (-1 at a root) and the keys it has committed.
pub const Window = struct { parents: []const i32, committed: usize };

pub const Plan = struct { flat: []i32, width: usize, streams: usize, items: usize, most: usize };

/// plan_host: `heads_a_kv` query heads a key head; the caller frees `flat`.
pub fn build(gpa: std.mem.Allocator, windows: []const Window, heads_a_kv: usize) !Plan {
    var rows: std.ArrayList(i32) = .empty;
    defer rows.deinit(gpa);
    var table: std.ArrayList(i32) = .empty;
    defer table.deinit(gpa);
    var items: std.ArrayList(i32) = .empty;
    defer items.deinit(gpa);
    var glob: std.ArrayList(i32) = .empty;
    defer glob.deinit(gpa);
    var start: usize = 0;
    var most: usize = 0;
    for (windows, 0..) |win, s| {
        const w = win.parents.len;
        if (w < 1 or w > max_nodes) return error.WindowWidth;
        const p = win.committed;
        const n = slots(p, w);
        most = @max(most, n);
        for (0..w) |_| try rows.append(gpa, @intCast(s));
        for ([_]usize{ start, w, p, n }) |v| try table.append(gpa, @intCast(v));
        for (win.parents) |x| try glob.append(gpa, if (x < 0) -1 else x + @as(i32, @intCast(start)));
        const g = groups(p, w);
        const chunks = p / chunk - g * group;
        var code: i64 = 0;
        while (code < g + chunks) : (code += 1) {
            const c: i32 = if (code < g) @intCast(code) else -1 - @as(i32, @intCast(code - @as(i64, @intCast(g))));
            var first: usize = 0;
            while (first < w * heads_a_kv) : (first += query_tile) {
                for ([_]i32{ @intCast(s), @intCast(first), c }) |v| try items.append(gpa, v);
            }
        }
        start += w;
    }
    const flat = try gpa.alloc(i32, rows.items.len + table.items.len + items.items.len + glob.items.len);
    var at: usize = 0;
    for ([_][]const i32{ rows.items, table.items, items.items, glob.items }) |part| {
        @memcpy(flat[at..][0..part.len], part);
        at += part.len;
    }
    return .{ .flat = flat, .width = start, .streams = windows.len, .items = items.items.len / 3, .most = most };
}

test "a plan lists rows, streams, items and parents as plan_host does" {
    const gpa = std.testing.allocator;
    // two serial chains: 2 rows after 600 keys, 1 row after 40 (attention.py's [[-1, 0], [-1]], [600, 40], group 12)
    const p = try build(gpa, &.{ .{ .parents = &.{ -1, 0 }, .committed = 600 }, .{ .parents = &.{-1}, .committed = 40 } }, 12);
    defer gpa.free(p.flat);
    // rows 0,0,1; streams (0,2,600,2) (2,1,40,1); stream 0's chunk -1 for first 0 and 16; parents -1,0,-1
    const want = [_]i32{ 0, 0, 1, 0, 2, 600, 2, 2, 1, 40, 1, 0, 0, -1, 0, 16, -1, -1, 0, -1 };
    try std.testing.expectEqualSlices(i32, &want, p.flat);
    try std.testing.expectEqual(@as(usize, 2), p.items);
    try std.testing.expectEqual(@as(usize, 2), p.most);
}

test "groups fold only when they fill the GPU" {
    try std.testing.expectEqual(@as(usize, 0), groups(4096, 1)); // 2 groups x 1 row < 64
    try std.testing.expectEqual(@as(usize, 2), groups(4096, 32));
    try std.testing.expectEqual(@as(usize, 2 + 1), slots(4096, 32)); // 2 groups, then chunk 8 holds keys 4096..4127
}
