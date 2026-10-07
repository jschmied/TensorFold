//! Grouped NVFP4 experts against the Python bytes (oracle/nvfp4_experts.py): packing, the plan, gate-up and down.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const fx4 = cuda.nvfp4_experts;

fn arr(fx: Fixture, comptime fmt: []const u8, args: anytype) ![]u8 {
    var name: [32]u8 = undefined;
    return fx.bytes(try std.fmt.bufPrint(&name, fmt, args));
}

pub fn experts(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const e: usize = @intCast(try fx.int("experts"));
    const d: usize = @intCast(try fx.int("dims"));
    const ni: usize = @intCast(try fx.int("width"));
    const slots: usize = @intCast(try fx.int("slots"));

    var raw: [3][2][]u8 = undefined; // gate, up, down: words, scales
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |m, i| {
        raw[i][0] = try arr(fx, "{s}_words", .{m});
        raw[i][1] = try arr(fx, "{s}_scales", .{m});
    }
    defer for (raw) |r| for (r) |b| a.free(b);
    const wb = ni * d / 2; // one expert's words, any matrix
    const sbytes = ni * d / 16;
    const blk = ni / fx4.cols * (d / 32) * fx4.words;
    const pu = try a.alloc(u32, e * 2 * blk);
    defer a.free(pu);
    const pd = try a.alloc(u32, e * blk);
    defer a.free(pd);
    for (0..e) |x| {
        const gate = [2][]const u8{ raw[0][0][x * wb ..][0..wb], raw[0][1][x * sbytes ..][0..sbytes] };
        const up = [2][]const u8{ raw[1][0][x * wb ..][0..wb], raw[1][1][x * sbytes ..][0..sbytes] };
        try fx4.packGateUp(pu[x * 2 * blk ..][0 .. 2 * blk], gate, up, ni, d);
        fx4.packOne(pd[x * blk ..][0..blk], raw[2][0][x * wb ..][0..wb], raw[2][1][x * sbytes ..][0..sbytes], d, ni);
    }
    inline for (.{ .{ "packed_up", pu }, .{ "packed_down", pd } }) |c| {
        const want = try fx.bytes(c[0]);
        defer a.free(want);
        try check.sameBytes(c[0], std.mem.sliceAsBytes(c[1]), want);
    }
    // make(): each expert's (gate, up) global scales side by side, down's alone
    const gg = try fx.bytes("gate_global");
    defer a.free(gg);
    const ug = try fx.bytes("up_global");
    defer a.free(ug);
    const us = try a.alloc(u8, 8 * e);
    defer a.free(us);
    for (0..e) |x| {
        @memcpy(us[8 * x ..][0..4], gg[4 * x ..][0..4]);
        @memcpy(us[8 * x + 4 ..][0..4], ug[4 * x ..][0..4]);
    }
    const want_us = try fx.bytes("up_scale");
    defer a.free(want_us);
    try check.sameBytes("up_scale", us, want_us);
    const ds = try fx.bytes("down_global");
    defer a.free(ds);
    check.pass("nvfp4 experts: {d} x [{d}, {d}] gate-up, down and scales packed as make() packs them", .{ e, ni, d });

    var dup = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pu));
    defer dup.free();
    var ddown = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pd));
    defer ddown.free();
    var dus = try cuda.DeviceBuffer.fromHost(gpu.d, us);
    defer dus.free();
    var dds = try cuda.DeviceBuffer.fromHost(gpu.d, ds);
    defer dds.free();
    const layer: fx4.Layer = .{ .up = dup.ptr, .down = ddown.ptr, .up_scale = dus.ptr, .down_scale = dds.ptr, .width = ni, .dims = d, .experts = e };
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var ex = try fx4.Experts.load(gpu.d, sms);
    defer ex.unload();
    var router = try grouped.Router.load(gpu.d);
    defer router.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    inline for (.{ .{ "decode", false }, .{ "prompt", true } }) |set| {
        var it = std.mem.tokenizeScalar(u8, try fx.string(set[0]), ',');
        while (it.next()) |tok| {
            const rows = try std.fmt.parseInt(usize, tok, 10);
            try oneCase(gpu, fx, &router, ex, stream, layer, rows, slots, set[1]);
        }
    }
}

fn oneCase(gpu: Gpu, fx: Fixture, router: *const grouped.Router, ex: fx4.Experts, s: cuda.Stream, l: fx4.Layer, rows: usize, slots: usize, prompt: bool) !void {
    const a = gpu.gpa;
    const pairs = rows * slots;
    const picks = try arr(fx, "picks{d}", .{rows});
    defer a.free(picks);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, picks);
    defer dpicks.free();
    var plan = try grouped.Plan.init(gpu.d, rows, slots, l.experts);
    defer plan.deinit();
    try router.route(s, dpicks.ptr, pairs, &plan, fx4.tile);
    try s.synchronize();

    const counts = try check.download(gpu, plan.counts);
    defer a.free(counts);
    const want_counts = try arr(fx, "counts{d}", .{rows});
    defer a.free(want_counts);
    try check.sameBytes("plan counts", counts, want_counts);
    const n_items: usize = @intCast(std.mem.bytesToValue(i32, counts[0..4]));
    const items = try check.download(gpu, plan.items);
    defer a.free(items);
    const want_items = try arr(fx, "items{d}", .{rows});
    defer a.free(want_items);
    try check.sameBytes("plan items", items[0 .. n_items * 12], want_items[0 .. n_items * 12]);
    const members = try check.download(gpu, plan.members);
    defer a.free(members);
    const want_members = try arr(fx, "members{d}", .{rows});
    defer a.free(want_members);
    try check.sameBytes("plan members", members, want_members);

    const x = try arr(fx, "x{d}", .{rows});
    defer a.free(x);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
    defer dx.free();
    var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.width * 2);
    defer act.free();
    try ex.run(s, .gate_up, dx.ptr, l.dims, rows, slots, l, &plan, act.ptr, -1);
    try s.synchronize();
    const got_act = try check.download(gpu, act);
    defer a.free(got_act);
    const want_act = try arr(fx, "act{d}", .{rows});
    defer a.free(want_act);
    try check.sameBytes("gate-up", got_act, want_act);

    const width: usize = if (prompt) 2 else 4;
    var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * width);
    defer y.free();
    try ex.run(s, if (prompt) .down_bf16 else .down_f32, act.ptr, l.width, rows, slots, l, &plan, y.ptr, -1);
    try s.synchronize();
    const got_y = try check.download(gpu, y);
    defer a.free(got_y);
    const want_y = try arr(fx, "y{d}", .{rows});
    defer a.free(want_y);
    try check.sameBytes("down", got_y, want_y);
    check.pass("nvfp4 experts: {d} rows x {d} slots ({s}, {d} items) plan, gate-up and down equal Python's bytes", .{ rows, slots, if (prompt) "prompt" else "decode", n_items });
}
