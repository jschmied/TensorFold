//! The draft head's input statistics (core.draft_stats): captured on a prompt set, read back for the weighted fit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("cuda_prompt_wt.zig");
const stats = core.draft_stats;

/// TF_FLASHNEXT_DRAFT_STATS_CAPTURE=path: each scored head input's squares, the file rewritten every 64 rows.
pub const Capture = struct {
    path: [*:0]const u8,
    sum: u64,
    sums: [wt.dims]f64 = @splat(0),
    acc: stats.Accumulator = .{ .sum = &.{} },

    pub fn fromEnv(a: std.mem.Allocator, sum: u64) !?*Capture {
        const path = std.c.getenv("TF_FLASHNEXT_DRAFT_STATS_CAPTURE") orelse return null;
        const c = try a.create(Capture);
        c.* = .{ .path = path, .sum = sum };
        c.acc.sum = &c.sums;
        std.log.info("flashnext draft head: capturing its input statistics to {s}", .{std.mem.span(path)});
        return c;
    }

    /// One bf16 row of `wt.dims` at `x` on the device (a host sync: capture runs are not timed).
    pub fn add(c: *Capture, stream: cuda.Stream, x: u64) !void {
        var row: [wt.dims]u16 = undefined;
        try stream.synchronize();
        const d = stream.d;
        try d.check(d.api.cuMemcpyDtoH_v2(&row, x, wt.dims * 2), "cuMemcpyDtoH");
        var f: [wt.dims]f32 = undefined;
        for (&f, row) |*o, v| o.* = @bitCast(@as(u32, v) << 16);
        c.acc.add(&f);
        if (c.acc.rows % 64 == 0) c.write();
    }

    pub fn write(c: *const Capture) void {
        if (c.acc.rows == 0) return;
        var mean: [wt.dims]f32 = undefined;
        c.acc.mean(&mean);
        var file: [stats.size(wt.dims)]u8 = undefined;
        stats.encode(&file, c.sum, c.acc.rows, &mean);
        const flags: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
        const fd = std.c.open(c.path, flags, @as(std.c.mode_t, 0o644));
        if (fd < 0) return std.log.warn("flashnext draft head: cannot write {s}", .{std.mem.span(c.path)});
        defer _ = std.c.close(fd);
        const name = std.mem.span(c.path);
        if (std.c.write(fd, &file, file.len) == file.len) return;
        std.log.warn("flashnext draft head: short write to {s}", .{name});
    }
};

/// TF_FLASHNEXT_DRAFT_STATS's means into `out` when the file is this head's; else null, and the log says why.
pub fn load(sum: u64, out: []f32) ?u64 {
    const path = std.c.getenv("TF_FLASHNEXT_DRAFT_STATS") orelse {
        std.log.warn("flashnext draft head: weighted fit asked, TF_FLASHNEXT_DRAFT_STATS unset; min/max fit", .{});
        return null;
    };
    const name = std.mem.span(path);
    var file: [stats.size(wt.dims) + 1]u8 = undefined;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) {
        std.log.warn("flashnext draft head: no statistics at {s}; min/max fit", .{name});
        return null;
    }
    defer _ = std.c.close(fd);
    var len: usize = 0;
    while (len < file.len) {
        const got = std.c.read(fd, file[len..].ptr, file.len - len);
        if (got <= 0) break;
        len += @intCast(got);
    }
    const rows = stats.decode(file[0..len], sum, out) catch |err| {
        std.log.warn("flashnext draft head: {s} at {s} (head {x:0>16}); min/max fit", .{ @errorName(err), name, sum });
        return null;
    };
    std.log.info("flashnext draft head: weighted fit from {s}, {d} rows", .{ name, rows });
    return rows;
}
