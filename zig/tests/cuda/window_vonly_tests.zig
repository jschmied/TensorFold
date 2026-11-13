//! TF_GEMMA4_V_ONLY at 512: keys from values equal qk_round's; V-only pages give the K+V bits (replays too).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rows = @import("window_rows_tests.zig");
const wc = @import("window_rows_check.zig");
const pages512 = @import("window_pages512_tests.zig");

const wa = cuda.window_attention;
const qk = cuda.qk_rows;

const d = 512;
const nan: u16 = 0x7fc0;

const Case = struct { name: []const u8, h: usize, hk: usize, top: usize, rows: usize, p0: usize, n: usize };

const cases = [_]Case{
    .{ .name = "31B global (32/4 heads of 512), 3000 positions", .h = 32, .hk = 4, .top = 3000, .rows = 17, .p0 = 2904, .n = 96 },
    .{ .name = "12B global shape (16/1 heads of 512), 5000 positions", .h = 16, .hk = 1, .top = 5000, .rows = 17, .p0 = 4700, .n = 300 },
};

/// Positions past 2^16 at the 31B's frequencies: the angles' reduction at a 64K context.
const far = 70000;

/// k and v rows [n, hk, 512] as the served writer makes them (tf_qk_rows_round, v from k), positions p0 + i.
fn written(b: *rows.Bufs, a: std.mem.Allocator, s: cuda.Stream, k: *const qk.QkRows, rng: std.Random, c: Case, p0: usize, n: usize, kn: u64, qn: u64, inv: u64) !struct { k: u64, v: u64, pos: u64 } {
    const raw_q = try a.alloc(u16, n * c.h * d);
    const raw_k = try a.alloc(u16, n * c.hk * d);
    var i: usize = 0;
    while (i < raw_q.len) : (i += d) wc.row(rng, raw_q[i..][0..d], 1.0 + rng.float(f32));
    i = 0;
    while (i < raw_k.len) : (i += d) wc.row(rng, raw_k[i..][0..d], 0.5 + 4 * rng.float(f32));
    const pos = try a.alloc(i32, n);
    for (pos, 0..) |*p, j| p.* = @intCast(p0 + j);
    const dk = try b.up(std.mem.sliceAsBytes(raw_k));
    const qw: i64 = @intCast(c.h * d);
    const kw: i64 = @intCast(c.hk * d);
    const src: qk.Src = .{ .q = try b.up(std.mem.sliceAsBytes(raw_q)), .q_ld = qw, .q_hs = d, .k = dk, .k_ld = kw, .k_hs = d, .v = dk, .v_ld = kw, .v_hs = d };
    const shape: qk.Shape = .{ .norms = qn, .norm_k = kn, .norms_bf16 = true, .inv = inv, .rot = d, .eps = 1e-6, .round = true, .v_norm = true };
    const out: qk.Out = .{ .q = try b.out(s, n * c.h * d * 2), .k = try b.out(s, n * c.hk * d * 2), .v = try b.out(s, n * c.hk * d * 2) };
    const dp = try b.up(std.mem.sliceAsBytes(pos));
    try k.rowsFrom(s, src, .{ .h = c.h, .hk = c.hk, .dim = d }, shape, dp, null, out, n);
    return .{ .k = out.k, .v = out.v, .pos = dp };
}

/// The recomputed keys against qk_round's (red with the planted flip); returns the values compared.
fn keysEqual(b: *rows.Bufs, s: cuda.Stream, g: *const wa.WindowAttention, c: Case, kv: anytype, n: usize, vk: wa.Vk, what: []const u8) !u64 {
    var counts: [2]u64 = undefined;
    for ([_]bool{ true, false }) |plant| {
        const dc = try b.up(std.mem.sliceAsBytes(&[2]u64{ 0, 0 }));
        try g.checkVo(s, kv.k, kv.v, kv.pos, c.hk, vk, plant, dc, n);
        try s.synchronize();
        const x: cuda.DeviceBuffer = .{ .d = b.gpu.d, .ptr = dc, .len = 16 };
        try x.download(0, std.mem.sliceAsBytes(&counts));
        try check.expect(counts[0] == n * c.hk * d, "window vonly {s} {s}: every key value compared ({d})", .{ c.name, what, counts[0] });
        if (plant) {
            try check.expect(counts[1] > 0, "window vonly {s} {s}: the planted flip is caught (red control)", .{ c.name, what });
        } else try check.expect(counts[1] == 0, "window vonly {s} {s}: {d} of {d} recomputed key values differ from qk_round's", .{ c.name, what, counts[1], counts[0] });
    }
    return counts[0];
}

fn case(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, g: *const wa.WindowAttention, k: *const qk.QkRows, rng: std.Random, c: Case, inv: u64) !u64 {
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    var kn_h: [d]u16 = undefined;
    var qn_h: [d]u16 = undefined;
    for (&kn_h, &qn_h) |*x, *y| {
        x.* = core.bf16_round.bits(0.25 + 1.5 * rng.float(f32));
        y.* = core.bf16_round.bits(0.25 + 1.5 * rng.float(f32));
    }
    const kn = try b.up(std.mem.sliceAsBytes(&kn_h));
    const qn = try b.up(std.mem.sliceAsBytes(&qn_h));
    const vk: wa.Vk = .{ .w = kn, .inv = inv };
    var compared: u64 = 0;
    // far positions: the bits at a 64K context's angles
    const far_kv = try written(&b, a, s, k, rng, c, far, 512, kn, qn, inv);
    compared += try keysEqual(&b, s, g, c, far_kv, 512, vk, "at positions 70000..70511");
    const kv = try written(&b, a, s, k, rng, c, 0, c.top, kn, qn, inv);
    compared += try keysEqual(&b, s, g, c, kv, c.top, vk, "at positions 0..");
    try s.synchronize();
    const kc = try a.alloc(u16, c.top * c.hk * d);
    const vc = try a.alloc(u16, kc.len);
    try b.get(kv.k, kc);
    try b.get(kv.v, vc);
    const poison = try a.alloc(u16, 1 << 20); // the V-only kernels' kc: NaN, so a read would show
    @memset(poison, nan);
    const no_keys = try b.up(std.mem.sliceAsBytes(poison));
    // decode rows over K+V pages vs V-only pages; row 1's q holds +inf: its heads run the merge's float64 replay
    const q = try a.alloc(u16, c.rows * c.h * d);
    var i: usize = 0;
    while (i < q.len) : (i += d) wc.row(rng, q[i..][0..d], 1.0);
    q[1 * c.h * d + 7] = 0x7f80;
    const pos = try a.alloc(i32, c.rows);
    const slot = try a.alloc(i32, c.rows);
    for (pos, slot, 0..) |*p, *sl, j| {
        p.* = @intCast(if (j == 0) 0 else if (j == c.rows - 1) c.top - 1 else (j * 7919) % c.top);
        sl.* = 0;
    }
    var w: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = 0, .ring = c.top, .scale = 1 };
    const dq = try b.up(std.mem.sliceAsBytes(q));
    const dp = try b.up(std.mem.sliceAsBytes(pos));
    const ds = try b.up(std.mem.sliceAsBytes(slot));
    const scratch = try b.out(s, wa.WindowAttention.scratchBytes(w, c.rows));
    const n = c.rows * c.h * d;
    for ([_]usize{ 37, 256 }) |tok| {
        const pool = try pages512.paged(&b, a, rng, kc, vc, 1, c.top, c.hk, tok);
        w.pg = pool.pages;
        w.vk = null;
        const want_d = try b.out(s, n * 2);
        try g.decode(s, dq, pool.kc, pool.vc, want_d, dp, ds, w, scratch, c.rows);
        w.vk = vk;
        const got_d = try b.out(s, n * 2);
        try g.decode(s, dq, no_keys, pool.vc, got_d, dp, ds, w, scratch, c.rows);
        try s.synchronize();
        const want = try a.alloc(u16, n);
        const got = try a.alloc(u16, n);
        try b.get(want_d, want);
        try b.get(got_d, got);
        var odd: usize = 0;
        for (want) |x| odd += @intFromBool(x & 0x7f80 == 0x7f80);
        try check.expect(odd > 0, "window vonly {s}: the planted +inf reaches the merge's replay", .{c.name});
        var what: [200]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&what, "window vonly {s}: decode over V-only pages of {d} gives the K+V bits", .{ c.name, tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    }
    // prompt rows p0 .. p0 + n over the same keys; the last row's q holds +inf (the float64 replay)
    const pq = try a.alloc(u16, c.n * c.h * d);
    i = 0;
    while (i < pq.len) : (i += d) wc.row(rng, pq[i..][0..d], 1.0);
    pq[(c.n - 1) * c.h * d + 7] = 0x7f80;
    const dpq = try b.up(std.mem.sliceAsBytes(pq));
    const top = c.p0 + c.n;
    for ([_]bool{ false, true }) |p32| for ([_]usize{ 32, 256 }) |tok| {
        const pool = try pages512.paged(&b, a, rng, kc[0 .. top * c.hk * d], vc[0 .. top * c.hk * d], 1, top, c.hk, tok);
        var pw: wa.Window = .{ .h = c.h, .hk = c.hk, .d = d, .win = 0, .ring = 0, .scale = 1, .p32 = p32, .pg = pool.pages };
        const m = c.n * c.h * d;
        const want_d = try b.out(s, m * 2);
        try g.promptRows(s, dpq, pool.kc, pool.vc, want_d, c.p0, c.n, pw, null);
        pw.vk = vk;
        const got_d = try b.out(s, m * 2);
        try g.promptRows(s, dpq, no_keys, pool.vc, got_d, c.p0, c.n, pw, null);
        try s.synchronize();
        const want = try a.alloc(u16, m);
        const got = try a.alloc(u16, m);
        try b.get(want_d, want);
        try b.get(got_d, got);
        var odd: usize = 0;
        for (want) |x| odd += @intFromBool(x & 0x7f80 == 0x7f80);
        try check.expect(odd > 0, "window vonly {s}: the planted +inf reaches the prompt's replay", .{c.name});
        var what: [200]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&what, "window vonly {s}: prompt rows ({s}) over V-only pages of {d} give the K+V bits", .{ c.name, if (p32) "P32" else "P as bf16", tok }), std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
    };
    return compared;
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try wa.WindowAttention.load(gpu.d);
    defer g.unload();
    try g.loadPages(gpu.d);
    var k = try qk.QkRows.load(gpu.d);
    defer k.unload();
    var inv_h: [d / 2]f32 = undefined;
    core.rope_freq.proportional(1000000.0, d, 64, &inv_h); // Gemma 4's global table (cuda_forward.inverseFrequencies)
    var b: rows.Bufs = .{ .gpu = gpu };
    defer b.deinit();
    const inv = try b.up(std.mem.sliceAsBytes(&inv_h));
    var prng = std.Random.DefaultPrng.init(0x7e0f);
    var compared: u64 = 0;
    for (cases) |c| {
        _ = arena.reset(.retain_capacity);
        compared += try case(gpu, arena.allocator(), s, &g, &k, prng.random(), c, inv);
    }
    check.pass("window vonly 512: {d} keys made from values equal qk_round's bit for bit (positions to 70511; the planted flip red), and decode (pages of 37, 256) and prompt rows (P as bf16 and P32, pages of 32, 256) over V-only pages with NaN keys give the K+V kernels' bits, float64 replays included", .{compared});
}
