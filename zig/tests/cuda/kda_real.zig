//! kda.cu and kda_prompt.cu on one real layer's inputs (a fixture of fla's run): against fla and against float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("kda_rig.zig");

const D = rig.D;

fn take(comptime T: type, a: std.mem.Allocator, bytes: []const u8, at: *usize, n: usize) ![]T {
    const out = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[at.*..][0 .. n * @sizeOf(T)]);
    at.* += n * @sizeOf(T);
    return out;
}

/// Relative L2 distance and the largest difference over the rms of `want`.
fn distance(got: []const f64, want: []const f64) [2]f64 {
    var dd: f64 = 0;
    var ww: f64 = 0;
    var most: f64 = 0;
    for (got, want) |g, w| {
        dd += (g - w) * (g - w);
        ww += w * w;
        most = @max(most, @abs(g - w));
    }
    return .{ @sqrt(dd / ww), most / @sqrt(ww / @as(f64, @floatFromInt(want.len))) };
}

fn widen(a: std.mem.Allocator, x: anytype) ![]f64 {
    const out = try a.alloc(f64, x.len);
    for (out, x) |*o, v| o.* = if (@TypeOf(v) == u16) rig.bf(v) else v;
    return out;
}

/// Each head's float64 run over all rows: the worst fraction of kda_ref's bound for `y` [rows][H D].
fn bound(m: *const rig.Model, a: std.mem.Allocator, y: []const u16, rows: usize) !f64 {
    const H = m.cfg.heads;
    const seq = try a.alloc(usize, rows);
    for (seq, 0..) |*x, i| x.* = i;
    const want = try a.alloc(f64, rows * D);
    const tol = try a.alloc(f64, rows * D);
    const ws = try a.alloc(f64, D * D);
    const got = try a.alloc(u16, rows * D);
    var worst: f64 = 0;
    for (0..H) |h| {
        try m.reference(0, h, seq, want, tol, ws);
        for (0..rows) |r| @memcpy(got[r * D ..][0..D], y[(r * H + h) * D ..][0..D]);
        worst = @max(worst, rig.yFraction(got, want, tol));
    }
    return worst;
}

pub fn run(gpu: check.Gpu, path: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => return std.debug.print("SKIP kda-real: no fixture at {s}\n", .{path}),
        else => return e,
    };
    if (bytes.len < 32 or !std.mem.eql(u8, bytes[0..8], "TFKDA001")) return error.BadFixture;
    const P: usize = std.mem.readInt(u32, bytes[8..12], .little);
    const N: usize = std.mem.readInt(u32, bytes[12..16], .little);
    const H: usize = std.mem.readInt(u32, bytes[16..20], .little);
    if (std.mem.readInt(u32, bytes[20..24], .little) != D) return error.BadFixture;
    const lb: f32 = @bitCast(std.mem.readInt(u32, bytes[24..28], .little));
    const eps: f32 = @bitCast(std.mem.readInt(u32, bytes[28..32], .little));
    const T = P + N;
    const hd = H * D;
    var at: usize = 32;
    var m: rig.Model = .{ .cfg = .{ .heads = H, .slots = 1, .lower_bound = lb, .eps = eps, .pool = T, .most = T }, .a = a, .qkv = try take(u16, a, bytes, &at, T * 3 * hd), .f = try take(u16, a, bytes, &at, T * hd), .b = try take(u16, a, bytes, &at, T * H), .gate = try take(u16, a, bytes, &at, T * hd), .conv = try take(f32, a, bytes, &at, 3 * hd * 4), .a_log = try take(f32, a, bytes, &at, H), .dt_bias = try take(f32, a, bytes, &at, hd), .norm = try take(f32, a, bytes, &at, D), .s0 = undefined, .conv0 = undefined };
    const fla_y = try take(u16, a, bytes, &at, T * hd);
    const fla_sp = try take(f32, a, bytes, &at, hd * D);
    const fla_st = try take(f32, a, bytes, &at, hd * D);
    if (at != bytes.len) return error.BadFixture;
    m.s0 = try a.alloc(f32, hd * D);
    m.conv0 = try a.alloc(u16, 3 * 3 * hd);
    @memset(m.s0, 0);
    @memset(m.conv0, 0);
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try cuda.kda.Kda.load(gpu.d);
    defer k.unload();
    var dv = try rig.Dev.init(&m, gpu, s, &k);
    defer dv.deinit();
    const seq = try a.alloc(usize, T);
    for (seq, 0..) |*x, i| x.* = i;
    const y = try a.alloc(u16, T * hd); // the prompt kernel over P rows, then N decode rows
    @memcpy(y[0 .. P * hd], try dv.call(a, .prompt, &.{.{ .slot = 0, .rows = seq[0..P] }}));
    const sp = (try dv.state(a, 0)).s;
    for (P..T) |r| @memcpy(y[r * hd ..][0..hd], try dv.call(a, .rows, &.{.{ .slot = 0, .rows = seq[r..][0..1] }}));
    const st = (try dv.state(a, 0)).s;
    try dv.reset(); // every row as a decode row
    const yd = try a.alloc(u16, T * hd);
    for (0..T) |r| @memcpy(yd[r * hd ..][0..hd], try dv.call(a, .rows, &.{.{ .slot = 0, .rows = seq[r..][0..1] }}));
    const fy = try widen(a, fla_y);
    const vy = distance(try widen(a, y), fy);
    const vd = distance(try widen(a, yd), fy);
    const vsp = distance(try widen(a, sp), try widen(a, fla_sp));
    const vst = distance(try widen(a, st), try widen(a, fla_st));
    const by = try bound(&m, a, y, T);
    const bd = try bound(&m, a, yd, T);
    const bf = try bound(&m, a, fla_y, T);
    std.debug.print("RESULT kda-real {d} heads, {d} prompt + {d} decode rows against fla: outputs rel L2 {e:.2} (all rows as decode rows {e:.2}), " ++
        "most |diff| / rms {d:.4} ({d:.4}); states rel L2 {e:.2} after the prompt, {e:.2} at the end\n", .{ H, P, N, vy[0], vd[0], vy[1], vd[1], vsp[0], vst[0] });
    std.debug.print("RESULT kda-real against float64: worst fraction of the bound {d:.3} (all rows as decode rows {d:.3}; fla's own outputs {d:.3})\n", .{ by, bd, bf });
    try check.expect(by <= 1 and bd <= 1, "kda-real: within float64 tolerance ({d:.3}, {d:.3})", .{ by, bd });
    try check.expect(vy[0] < 1e-2 and vd[0] < 1e-2 and vst[0] < 1e-2, "kda-real: within 1 % of fla (rel L2 {e:.2}, {e:.2}, state {e:.2})", .{ vy[0], vd[0], vst[0] });
    check.pass("kda-real: Ling-3.1 layer 0 within float64 tolerance (worst {d:.3}) and within 1 % of fla (rel L2 {e:.2})", .{ @max(by, bd), @max(vy[0], vd[0]) });
}
