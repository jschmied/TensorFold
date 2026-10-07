//! Kolibri 1's device weights: layer 0 read back against weights.load's bytes (oracle/kolibri_layer.py).
const std = @import("std");
const cuda = @import("cuda");
const kolibri1 = @import("kolibri1");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn back(gpu: Gpu, ptr: u64, len: usize) ![]u8 {
    const out = try gpu.gpa.alloc(u8, len);
    errdefer gpu.gpa.free(out);
    const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = len };
    try b.download(0, out);
    return out;
}

fn same(gpu: Gpu, dir: []const u8, name: []const u8, ptr: u64) !void {
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/{s}.bin", .{ dir, name });
    defer gpu.gpa.free(path);
    const want = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 32));
    defer gpu.gpa.free(want);
    const got = try back(gpu, ptr, want.len);
    defer gpu.gpa.free(got);
    try check.sameBytes(name, got, want);
}

pub fn layer0(gpu: Gpu, model: []const u8, fixture: []const u8) !void {
    var w = try kolibri1.weights.load(gpu.gpa, gpu.io, gpu.d, model, 1);
    defer w.deinit();
    const l = w.layers[0];
    const e = l.experts;
    const pairs = .{
        .{ "qkv_w8", l.qkv.codes },          .{ "qkv_bs", l.qkv.scales }, .{ "o_w8", l.o.codes },      .{ "o_bs", l.o.scales },
        .{ "up", e.up },                     .{ "down", e.down },         .{ "up_scale", e.up_scale }, .{ "down_scale", e.down_scale },
        .{ "input_norm", l.input_norm },     .{ "q_norm", l.q_norm },     .{ "k_norm", l.k_norm },     .{ "post_attn_norm", l.post_attn_norm },
        .{ "pre_moe_norm", l.pre_moe_norm }, .{ "router", l.router },     .{ "bias", l.bias },
    };
    inline for (pairs) |p| try same(gpu, fixture, p[0], p[1]);
    check.pass("kolibri1 weights: layer 0's 15 device arrays equal weights.load's ({d} experts, qkv [{d}, {d}])", .{ e.experts, l.qkv.n, l.qkv.k });
}
