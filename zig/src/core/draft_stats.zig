//! A draft head's input statistics, diag(H) = E[x^2] per input column, in a file keyed by the head's checksum.
const std = @import("std");

pub const magic = "TFDHSTAT".*;
pub const version: u32 = 1;
pub const header = 32;

/// The head's checksum: Wyhash of the exact bytes its fit quantizes.
pub fn checksum(head: []const u8) u64 {
    return std.hash.Wyhash.hash(1870, head);
}

pub fn size(width: usize) usize {
    return header + width * 4;
}

/// magic, version, width, checksum, rows (little endian), then `width` f32 means of x^2.
pub fn encode(out: []u8, sum: u64, rows: u64, mean: []const f32) void {
    std.debug.assert(out.len == size(mean.len));
    @memcpy(out[0..8], &magic);
    std.mem.writeInt(u32, out[8..12], version, .little);
    std.mem.writeInt(u32, out[12..16], @intCast(mean.len), .little);
    std.mem.writeInt(u64, out[16..24], sum, .little);
    std.mem.writeInt(u64, out[24..32], rows, .little);
    for (mean, 0..) |m, i| std.mem.writeInt(u32, out[header + i * 4 ..][0..4], @bitCast(m), .little);
}

pub const Error = error{ StatsFormat, StatsWidth, StatsChecksum, StatsEmpty, StatsValue };

/// The means into `out` and the rows they came from, when `bytes` is this head's file; else why not.
pub fn decode(bytes: []const u8, sum: u64, out: []f32) Error!u64 {
    if (bytes.len < header or !std.mem.eql(u8, bytes[0..8], &magic)) return error.StatsFormat;
    if (std.mem.readInt(u32, bytes[8..12], .little) != version) return error.StatsFormat;
    if (std.mem.readInt(u32, bytes[12..16], .little) != out.len or bytes.len != size(out.len)) return error.StatsWidth;
    if (std.mem.readInt(u64, bytes[16..24], .little) != sum) return error.StatsChecksum;
    const rows = std.mem.readInt(u64, bytes[24..32], .little);
    if (rows == 0) return error.StatsEmpty;
    for (out, 0..) |*o, i| {
        o.* = @bitCast(std.mem.readInt(u32, bytes[header + i * 4 ..][0..4], .little));
        if (!std.math.isFinite(o.*) or o.* < 0) return error.StatsValue;
    }
    return rows;
}

/// Running sums of x^2 per column; `mean` writes them out as the file's f32 means.
pub const Accumulator = struct {
    sum: []f64,
    rows: u64 = 0,

    pub fn add(acc: *Accumulator, row: []const f32) void {
        for (acc.sum, row) |*s, x| s.* += @as(f64, x) * @as(f64, x);
        acc.rows += 1;
    }

    pub fn mean(acc: *const Accumulator, out: []f32) void {
        const n: f64 = @floatFromInt(@max(acc.rows, 1));
        for (out, acc.sum) |*o, s| o.* = @floatCast(s / n);
    }
};

test "draft stats round-trip, and a file for another head, width, format or with no rows is refused by name" {
    const head = [_]u8{ 1, 2, 3, 4 };
    const sum = checksum(&head);
    var sums: [4]f64 = @splat(0);
    var acc: Accumulator = .{ .sum = &sums };
    acc.add(&.{ 1, -2, 0, 0.5 });
    acc.add(&.{ 3, 2, 0, 0.5 });
    var mean: [4]f32 = undefined;
    acc.mean(&mean);
    try std.testing.expectEqualSlices(f32, &.{ 5, 4, 0, 0.25 }, &mean);
    var file: [size(4)]u8 = undefined;
    encode(&file, sum, acc.rows, &mean);
    var back: [4]f32 = undefined;
    try std.testing.expectEqual(@as(u64, 2), try decode(&file, sum, &back));
    try std.testing.expectEqualSlices(f32, &mean, &back);
    try std.testing.expectError(error.StatsChecksum, decode(&file, checksum(&.{ 1, 2, 3, 5 }), &back));
    var three: [3]f32 = undefined;
    try std.testing.expectError(error.StatsWidth, decode(&file, sum, &three));
    try std.testing.expectError(error.StatsFormat, decode(file[0..16], sum, &back));
    encode(&file, sum, 0, &mean);
    try std.testing.expectError(error.StatsEmpty, decode(&file, sum, &back));
    mean[1] = std.math.nan(f32);
    encode(&file, sum, 2, &mean);
    try std.testing.expectError(error.StatsValue, decode(&file, sum, &back));
}
