//! LFM3-VL's vision tower and projector on one picture's patch rows; tools/lfm3/vision_check.py compares the features.
const std = @import("std");
const cuda = @import("cuda");
const lfm3 = @import("lfm3_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// MODEL PIXELS GH GW OUT: PIXELS bf16 [GH GW, 1536] in merge-block order, OUT bf16 [GH GW / 4, 2048].
pub fn features(gpu: Gpu, rest: []const []const u8) !void {
    if (rest.len != 5) return error.MissingArgument;
    const gpa = gpu.gpa;
    const io = gpu.io;
    const gh = try std.fmt.parseInt(usize, rest[2], 10);
    const gw = try std.fmt.parseInt(usize, rest[3], 10);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, rest[1], gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    const pixels = try gpa.alloc(u16, bytes.len / 2);
    defer gpa.free(pixels);
    @memcpy(std.mem.sliceAsBytes(pixels), bytes[0 .. pixels.len * 2]);
    var k = try lfm3.kernels.Kernels.load(gpu.ctx);
    defer k.deinit();
    var t = try lfm3.vision.Tower.init(gpa, io, gpu.d, &k, rest[0], gh * gw);
    defer t.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const rows = gh * gw / 4;
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * lfm3.vision.out_width * 2);
    defer out.free();
    try t.encode(gpu.d, s, pixels, gh, gw, out.ptr);
    try s.synchronize();
    const t0 = check.now(io);
    try t.encode(gpu.d, s, pixels, gh, gw, out.ptr);
    try s.synchronize();
    const ms = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6;
    const host = try gpa.alloc(u8, rows * lfm3.vision.out_width * 2);
    defer gpa.free(host);
    try out.download(0, host);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = rest[4], .data = host });
    std.debug.print("lfm3-vision: {d}x{d} patches -> {d} rows, {d:.2} ms the second run, weights {d} MiB\n", .{ gh, gw, rows, ms, t.weight_bytes >> 20 });
}
