//! kda.cu and kda_prompt.cu against core/kda_ref.zig: decode rows, verify windows and prompt chunks, and their bits.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("kda_rig.zig");

const D = rig.D;
const Part = rig.Part;

const Worst = struct { decode: f64 = 0, prompt: f64 = 0, state: f64 = 0 };

fn range(a: std.mem.Allocator, start: usize, n: usize) ![]usize {
    const v = try a.alloc(usize, n);
    for (v, 0..) |*x, i| x.* = start + i;
    return v;
}

/// Each head's float64 run of `seq` from `slot`'s initial state: rows [rows][H D] and the final S, as fractions.
fn against(m: *const rig.Model, a: std.mem.Allocator, slot: usize, seq: []const usize, y: []const u16, s: ?[]const f32, worst: *f64, sworst: *f64) !void {
    const H = m.cfg.heads;
    const want = try a.alloc(f64, seq.len * D);
    const tol = try a.alloc(f64, seq.len * D);
    const ws = try a.alloc(f64, D * D);
    const got = try a.alloc(u16, seq.len * D);
    for (0..H) |h| {
        try m.reference(slot, h, seq, want, tol, ws);
        for (0..seq.len) |r| @memcpy(got[r * D ..][0..D], y[(r * H + h) * D ..][0..D]);
        worst.* = @max(worst.*, rig.yFraction(got, want, tol));
        if (s) |st| sworst.* = @max(sworst.*, rig.sFraction(st[h * D * D ..][0 .. D * D], ws));
    }
}

fn rowsOf(y: []const u16, H: usize, first: usize, n: usize) []const u16 {
    return y[first * H * D ..][0 .. n * H * D];
}

/// Serial decode, the same rows in verify windows (rejected drafts after the accepted ones), and a stream alone.
fn decode(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, k: *const cuda.kda.Kda, rng: std.Random, cfg: rig.Cfg, n: usize, worst: *Worst) !void {
    var m = try rig.Model.init(a, rng, cfg);
    var dv = try rig.Dev.init(&m, gpu, s, k);
    defer dv.deinit();
    const S = cfg.slots;
    const H = cfg.heads;
    const hd = H * D;
    const seqs = try a.alloc([]usize, S);
    for (seqs, 0..) |*q, st| q.* = try range(a, st * n, n);
    const serial = try a.alloc(u16, S * n * hd); // [stream][row][H D]
    var parts: [8]Part = undefined;
    for (0..n) |r| {
        for (0..S) |st| parts[st] = .{ .slot = st, .rows = seqs[st][r..][0..1] };
        const y = try dv.call(a, .rows, parts[0..S]);
        for (0..S) |st| @memcpy(serial[(st * n + r) * hd ..][0..hd], y[st * hd ..][0..hd]);
    }
    const final = try a.alloc(@TypeOf(try dv.state(a, 0)), S);
    for (final, 0..) |*f, st| {
        f.* = try dv.state(a, st);
        try against(&m, a, st, seqs[st], serial[st * n * hd ..][0 .. n * hd], f.s, &worst.decode, &worst.state);
    }
    try dv.reset(); // stream 1 alone
    for (0..n) |r| {
        const y = try dv.call(a, .rows, &.{.{ .slot = 1, .rows = seqs[1][r..][0..1] }});
        try check.sameBytes("kda decode: a stream alone vs beside others", std.mem.sliceAsBytes(y), std.mem.sliceAsBytes(serial[(n + r) * hd ..][0..hd]));
    }
    const alone = try dv.state(a, 1);
    try check.sameBytes("kda decode: state alone", std.mem.sliceAsBytes(alone.s), std.mem.sliceAsBytes(final[1].s));
    try dv.reset(); // windows: pending row + drafts, the first `take` right, the rest from elsewhere in the pool
    const garbage = S * n;
    const pos = try a.alloc(usize, S);
    const kept = try a.alloc(usize, S);
    @memset(pos, 0);
    @memset(kept, 0);
    var rounds: usize = 0;
    while (true) : (rounds += 1) {
        var np: usize = 0;
        var takes: [8]usize = undefined;
        var slots: [8]usize = undefined;
        for (0..S) |st| {
            if (pos[st] == n and kept[st] == 0) continue;
            const left = n - pos[st];
            const width = if (left == 0) 0 else 1 + rng.uintLessThan(usize, 16);
            const take = if (left == 0) 0 else @min(left, 1 + rng.uintLessThan(usize, width));
            const rows = try a.alloc(usize, width);
            for (rows, 0..) |*x, i| x.* = if (i < take) seqs[st][pos[st] + i] else garbage + rng.uintLessThan(usize, 64);
            parts[np] = .{ .slot = st, .rows = rows, .kept = kept[st], .commit = left == 0 };
            takes[np] = take;
            slots[np] = st;
            np += 1;
        }
        if (np == 0) break;
        const y = try dv.call(a, .rows, parts[0..np]);
        var first: usize = 0;
        for (parts[0..np], takes[0..np], slots[0..np]) |p, take, st| {
            try check.sameBytes("kda window rows vs serial", std.mem.sliceAsBytes(rowsOf(y, H, first, take)), std.mem.sliceAsBytes(serial[(st * n + pos[st]) * hd ..][0 .. take * hd]));
            first += p.rows.len;
            pos[st] += take;
            kept[st] = take;
        }
    }
    for (0..S) |st| {
        const got = try dv.state(a, st);
        try check.sameBytes("kda windows: state vs serial", std.mem.sliceAsBytes(got.s), std.mem.sliceAsBytes(final[st].s));
        try check.sameBytes("kda windows: conv window vs serial", std.mem.sliceAsBytes(got.conv), std.mem.sliceAsBytes(final[st].conv));
    }
    if (rounds < n / 8) return error.TestFailed; // windows of more than one row ran
}

/// Prompt chunks of several lengths in one call, each alone, cut at multiples of 16, then decode rows after them.
fn prompt(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, k: *const cuda.kda.Kda, rng: std.Random, cfg: rig.Cfg, lens: []const usize, worst: *Worst) !void {
    var m = try rig.Model.init(a, rng, cfg);
    var dv = try rig.Dev.init(&m, gpu, s, k);
    defer dv.deinit();
    const H = cfg.heads;
    const hd = H * D;
    const S = lens.len;
    var parts: [8]Part = undefined;
    const seqs = try a.alloc([]usize, S);
    var start: usize = 0;
    for (seqs, lens, 0..) |*q, len, st| {
        q.* = try range(a, start, len + 6); // 6 decode rows after the prompt
        start += len + 6;
        parts[st] = .{ .slot = st, .rows = q.*[0..len] };
    }
    const y = try dv.call(a, .prompt, parts[0..S]);
    const finals = try a.alloc(@TypeOf(try dv.state(a, 0)), S);
    var first: usize = 0;
    for (lens, finals, 0..) |len, *f, st| {
        f.* = try dv.state(a, st);
        try against(&m, a, st, seqs[st][0..len], rowsOf(y, H, first, len), f.s, &worst.prompt, &worst.state);
        first += len;
    }
    try dv.reset();
    first = 0;
    for (lens, 0..) |len, st| { // each alone
        const got = try dv.call(a, .prompt, &.{.{ .slot = st, .rows = seqs[st][0..len] }});
        try check.sameBytes("kda prompt: a stream alone vs beside others", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(rowsOf(y, H, first, len)));
        try check.sameBytes("kda prompt: state alone", std.mem.sliceAsBytes((try dv.state(a, st)).s), std.mem.sliceAsBytes(finals[st].s));
        first += len;
    }
    try dv.reset();
    first = 0;
    for (lens, 0..) |len, st| { // cut at a multiple of 16
        const cut = len / 32 * 16;
        const y0 = try dv.call(a, .prompt, &.{.{ .slot = st, .rows = seqs[st][0..cut] }});
        const y1 = try dv.call(a, .prompt, &.{.{ .slot = st, .rows = seqs[st][cut..len] }});
        try check.sameBytes("kda prompt: cut at 16 rows, first part", std.mem.sliceAsBytes(y0), std.mem.sliceAsBytes(rowsOf(y, H, first, cut)));
        try check.sameBytes("kda prompt: cut at 16 rows, second part", std.mem.sliceAsBytes(y1), std.mem.sliceAsBytes(rowsOf(y, H, first + cut, len - cut)));
        const got = try dv.state(a, st);
        try check.sameBytes("kda prompt: cut state", std.mem.sliceAsBytes(got.s), std.mem.sliceAsBytes(finals[st].s));
        try check.sameBytes("kda prompt: cut conv window", std.mem.sliceAsBytes(got.conv), std.mem.sliceAsBytes(finals[st].conv));
        first += len;
    }
    const dec = try a.alloc(u16, S * 6 * hd); // decode after the prompts, against float64 over the whole sequence
    for (0..6) |r| {
        for (0..S) |st| parts[st] = .{ .slot = st, .rows = seqs[st][lens[st] + r ..][0..1] };
        const yd = try dv.call(a, .rows, parts[0..S]);
        for (0..S) |st| @memcpy(dec[(st * 6 + r) * hd ..][0..hd], yd[st * hd ..][0..hd]);
    }
    for (lens, seqs, 0..) |len, q, st| {
        const all = try a.alloc(u16, (len + 6) * hd);
        @memset(all[0 .. len * hd], 0);
        @memcpy(all[len * hd ..], dec[st * 6 * hd ..][0 .. 6 * hd]);
        const want = try a.alloc(f64, (len + 6) * D);
        const tol = try a.alloc(f64, (len + 6) * D);
        const ws = try a.alloc(f64, D * D);
        const got = try a.alloc(u16, 6 * D);
        for (0..H) |h| {
            try m.reference(st, h, q, want, tol, ws);
            for (0..6) |r| @memcpy(got[r * D ..][0..D], all[((len + r) * H + h) * D ..][0..D]);
            worst.decode = @max(worst.decode, rig.yFraction(got, want[len * D ..], tol[len * D ..]));
        }
    }
    // a window's kept rows replayed by the prompt kernel == replayed by a rows call, then the prompt
    const st0 = try dv.state(a, 0);
    const win = seqs[0][0..5];
    const tail = seqs[1][0..lens[1]];
    var ys: [2][]u16 = undefined;
    var ss: [2][]f32 = undefined;
    for (0..2) |arm| {
        try dv.sb.upload(0, std.mem.sliceAsBytes(st0.s));
        try dv.cb.upload(0, std.mem.sliceAsBytes(st0.conv));
        _ = try dv.call(a, .rows, &.{.{ .slot = 0, .rows = win, .commit = false }});
        if (arm == 1) _ = try dv.call(a, .rows, &.{.{ .slot = 0, .rows = &.{}, .kept = 3 }});
        ys[arm] = try dv.call(a, .prompt, &.{.{ .slot = 0, .rows = tail, .kept = if (arm == 0) 3 else 0 }});
        ss[arm] = (try dv.state(a, 0)).s;
    }
    try check.sameBytes("kda prompt: kept rows replayed in the prompt call", std.mem.sliceAsBytes(ys[0]), std.mem.sliceAsBytes(ys[1]));
    try check.sameBytes("kda prompt: state after kept rows", std.mem.sliceAsBytes(ss[0]), std.mem.sliceAsBytes(ss[1]));
}

/// The fastest of `reps` single calls, each between its own events: another process's work on the GPU only adds.
fn timeit(dv: *rig.Dev, kind: rig.Kind, n: usize, reps: usize) !f64 {
    var e0 = try cuda.Event.init(dv.gpu.d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(dv.gpu.d, true);
    defer e1.deinit();
    var best: f64 = std.math.inf(f64);
    for (0..reps) |_| {
        try e0.record(dv.s);
        try dv.again(kind, n);
        try e1.record(dv.s);
        try e1.synchronize();
        best = @min(best, @as(f64, try cuda.Event.elapsedMs(e0, e1)) * 1000);
    }
    return best;
}

/// Microseconds a call at Ling-3.1's 64 heads: a decode row, a 16-row window, a 2,048-row prompt; 1 and 8 streams.
fn timings(gpu: check.Gpu, a: std.mem.Allocator, s: cuda.Stream, k: *const cuda.kda.Kda, rng: std.Random) !void {
    var m = try rig.Model.init(a, rng, .{ .heads = 64, .slots = 8, .lower_bound = -5, .pool = 2048, .most = 2048 });
    var dv = try rig.Dev.init(&m, gpu, s, k);
    defer dv.deinit();
    const rows = try range(a, 0, 2048);
    var us: [5]f64 = undefined;
    var parts: [8]Part = undefined;
    _ = try dv.call(a, .rows, &.{.{ .slot = 0, .rows = rows[0..1] }});
    us[0] = try timeit(&dv, .rows, 1, 50);
    for (0..8) |st| parts[st] = .{ .slot = st, .rows = rows[st..][0..1] };
    _ = try dv.call(a, .rows, &parts);
    us[1] = try timeit(&dv, .rows, 8, 50);
    _ = try dv.call(a, .rows, &.{.{ .slot = 0, .rows = rows[0..16], .commit = false }});
    us[2] = try timeit(&dv, .rows, 1, 50);
    for (0..8) |st| parts[st] = .{ .slot = st, .rows = rows[16 * st ..][0..16], .commit = false };
    _ = try dv.call(a, .rows, &parts);
    us[3] = try timeit(&dv, .rows, 8, 20);
    _ = try dv.call(a, .prompt, &.{.{ .slot = 0, .rows = rows }});
    us[4] = try timeit(&dv, .prompt, 1, 10);
    std.debug.print("RESULT kda 64x128 us/call: decode row {d:.1} (8 streams {d:.1}), 16-row window {d:.1} (8 streams {d:.1}), 2048-row prompt {d:.1}\n", .{ us[0], us[1], us[2], us[3], us[4] });
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var k = try cuda.kda.Kda.load(gpu.d);
    defer k.unload();
    var prng = std.Random.DefaultPrng.init(850);
    const rng = prng.random();
    var w: Worst = .{};
    const lens = [_]usize{ 1, 37, 64, 300, 1024 };
    try decode(gpu, a, s, &k, rng, .{ .heads = 4, .slots = 3, .lower_bound = -5, .pool = 3 * 48 + 64, .most = 64 }, 48, &w);
    try decode(gpu, a, s, &k, rng, .{ .heads = 3, .slots = 2, .lower_bound = 0, .pool = 2 * 40 + 64, .most = 64 }, 40, &w);
    try decode(gpu, a, s, &k, rng, .{ .heads = 64, .slots = 2, .lower_bound = -5, .pool = 2 * 12 + 64, .most = 64 }, 12, &w);
    try prompt(gpu, a, s, &k, rng, .{ .heads = 2, .slots = 5, .lower_bound = -5, .pool = 1500, .most = 1500 }, &lens, &w);
    try prompt(gpu, a, s, &k, rng, .{ .heads = 2, .slots = 5, .lower_bound = 0, .pool = 1500, .most = 1500 }, &lens, &w);
    try prompt(gpu, a, s, &k, rng, .{ .heads = 64, .slots = 2, .lower_bound = -5, .pool = 160, .most = 160 }, &.{ 37, 100 }, &w);
    const ok = w.decode <= 1 and w.prompt <= 1 and w.state <= 1;
    try check.expect(ok, "kda: within float64 tolerance (decode {d:.3}, prompt {d:.3}, state {d:.3})", .{ w.decode, w.prompt, w.state });
    check.pass("kda: decode, windows and prompts within float64 tolerance (worst decode {d:.3}, prompt {d:.3}, state {d:.3}); windows == serial, alone == batched, cut prompts == uncut, bit for bit", .{ w.decode, w.prompt, w.state });
    try timings(gpu, a, s, &k, rng);
}
