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

const CallJson = struct {
    name: ?[]const u8 = null,
    prompt: bool,
    rows: ?[]const usize = null,
    chains: []const struct { slot: usize, p0: usize, tokens: []const u32 },
    file: ?[]const u8 = null,
    shape: ?[]const usize = null,
};
const CallsJson = struct { context: usize, slots: usize, calls: []const CallJson };

/// The Python engine's forwards replayed in order (capture_kolibri.py); `set` is that run's captured Triton set.
pub fn forward(gpu: Gpu, model: []const u8, capture: []const u8, set: []const u8) !void {
    const gpa = gpu.gpa;
    const path = try std.fmt.allocPrint(gpa, "{s}/calls.json", .{capture});
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(CallsJson, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const calls = parsed.value;
    var w = try kolibri1.weights.load(gpa, gpu.io, gpu.d, model, null);
    defer w.deinit();
    var k = try kolibri1.kernels.Kernels.load(gpa, gpu.io, gpu.ctx, set);
    defer k.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var widest: usize = 1;
    for (calls.calls) |c| {
        var n: usize = 0;
        for (c.chains) |ch| n += ch.tokens.len;
        widest = @max(widest, n);
    }
    var m = try kolibri1.forward.Model.init(gpa, gpu.d, stream, &w, calls.context, calls.slots, widest);
    defer m.deinit();
    const o: kolibri1.kernels.Ops = .{ .k = &k, .s = stream };
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, widest * w.config.vocab * 4);
    defer logits.free();
    var bad: usize = 0;
    for (calls.calls) |c| {
        const chains = try gpa.alloc(kolibri1.forward.Chain, c.chains.len);
        defer gpa.free(chains);
        for (c.chains, chains) |src, *dst| dst.* = .{ .slot = src.slot, .p0 = src.p0, .tokens = src.tokens };
        try logits.fill8(0xff, stream.handle);
        try m.forward(o, chains, c.prompt, c.rows, logits.ptr);
        try stream.synchronize();
        const name = c.name orelse continue;
        const file = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ capture, c.file.? });
        defer gpa.free(file);
        const want = try std.Io.Dir.cwd().readFileAlloc(gpu.io, file, gpa, .limited(1 << 32));
        defer gpa.free(want);
        const got = try back(gpu, logits.ptr, want.len);
        defer gpa.free(got);
        const wf = std.mem.bytesAsSlice(f32, @as([]align(1) u8, want));
        const gf = std.mem.bytesAsSlice(f32, @as([]align(1) u8, got));
        var diff: usize = 0;
        var most: f32 = 0;
        for (wf, gf) |a, b| {
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) diff += 1;
            most = @max(most, @abs(a - b));
        }
        std.debug.print("RESULT kolibri1 forward {s}: {d} of {d} logits differ from Python's bytes (largest {e})\n", .{ name, diff, wf.len, most });
        if (diff != 0) bad += 1;
    }
    try check.expect(bad == 0, "kolibri1 forward: every captured call equals Python's logits", .{});
    check.pass("kolibri1 forward: {d} calls replayed, every named one's logits equal Python's bytes", .{calls.calls.len});
}
