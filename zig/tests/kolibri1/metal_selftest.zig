//! tf-kolibri-metal selftest: Kolibri-1's Metal forward on the seeded checkpoint against the float64 reference.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const k1 = @import("kolibri1_metal");

const fwd = k1.forward;

const usage =
    \\usage: tf-kolibri-metal selftest [DIR] [--plant BLOCK:N] [--shape parity|wide]
    \\  DIR    holds the seeded checkpoint (written when DIR has no config.json; default kolibri1-selftest)
    \\  plant  a planted fault: glue:1-3, rows:1-3 or moe:1-3; the run must fail its gate (exit 1)
    \\Exit 0: every gate passed; 1: a gate failed; 2: usage or setup.
    \\
;

const tokens_n = 640; // past one 576-key ring of 16-row calls, so a sliding layer wraps; rows 511..513 see 512..514
const prefix = 320;
const gates = struct {
    const cosine = 0.9999; // every row's logits and each tap row against float64
    const rel = 0.02; // a tap row's relative L2 error
    const tie = 1e-3; // a top-1 swap passes only where float64's own top-2 margin is below this
};

const Stats = struct { worst_cos: f64 = 1, worst_rel: f64 = 0, swaps: usize = 0, ties: usize = 0, rows: usize = 0, ok: bool = true };

fn compare(st: *Stats, got: []const f32, want: []const f64, width: usize, logits: bool) void {
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var err: f64 = 0;
        for (g, w) |a, b| {
            dot += a * b;
            gg += @as(f64, a) * a;
            ww += b * b;
            err += (a - b) * (a - b);
        }
        const cos = dot / @sqrt(gg * ww);
        const rel = @sqrt(err / ww);
        st.worst_cos = @min(st.worst_cos, cos);
        st.worst_rel = @max(st.worst_rel, rel);
        st.rows += 1;
        if (!(cos >= gates.cosine)) st.ok = false;
        if (!logits and !(rel <= gates.rel)) st.ok = false;
        if (!logits) continue;
        const top = std.mem.indexOfMax(f64, w);
        const mine = std.mem.indexOfMax(f32, g);
        if (top == mine) continue;
        var second: f64 = -std.math.inf(f64);
        for (w, 0..) |v, i| if (i != top) {
            second = @max(second, v);
        };
        if (w[top] - second < gates.tie) st.ties += 1 else {
            st.swaps += 1;
            st.ok = false;
        }
    }
}

fn report(name: []const u8, st: Stats) void {
    std.debug.print("{s} {s}: {d} rows, worst cosine {d:.7}, worst rel {e:.3}, top-1 swaps {d} (ties {d})\n", .{ if (st.ok) "PASS" else "FAIL", name, st.rows, st.worst_cos, st.worst_rel, st.swaps, st.ties });
}

fn sameBytes(name: []const u8, a: []const f32, b: []const f32) bool {
    const ok = std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    std.debug.print("{s} {s}: {d} values byte for byte\n", .{ if (ok) "PASS" else "FAIL", name, a.len });
    return ok;
}

/// `ids` through `m` in calls of `chunk` rows (the last ones one row at a time from `singles` on): every row's logits.
fn run(a: std.mem.Allocator, m: *fwd.Model, ids: []const u32, chunk: usize, singles: usize, taps: ?fwd.Taps) ![]f32 {
    const v = m.c.vocab;
    const out = try a.alloc(f32, ids.len * v);
    var at: usize = 0;
    while (at < ids.len) {
        const n = if (at >= singles) 1 else @min(chunk, singles - at);
        const t: ?fwd.Taps = if (taps != null and at == 0 and n == ids.len) taps else null;
        const lg = try m.forward(ids[at..][0..n], true, t);
        @memcpy(out[at * v ..][0 .. n * v], lg);
        at += n;
    }
    return out;
}

fn plant(text: []const u8) !fwd.Ctl {
    const colon = std.mem.indexOfScalar(u8, text, ':') orelse return error.Usage;
    const n = try std.fmt.parseInt(u8, text[colon + 1 ..], 10);
    if (n == 0 or n > 3) return error.Usage;
    const block = text[0..colon];
    if (std.mem.eql(u8, block, "glue")) return .{ .glue = n };
    if (std.mem.eql(u8, block, "rows")) return .{ .rows = n };
    if (std.mem.eql(u8, block, "moe")) return .{ .moe = n };
    return error.Usage;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or !std.mem.eql(u8, args[1], "selftest")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var dir_path: []const u8 = "kolibri1-selftest";
    var ctl: fwd.Ctl = .{};
    var shape = k1.fixture.parity;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--plant") and i + 1 < args.len) {
            i += 1;
            ctl = plant(args[i]) catch {
                std.debug.print("{s}", .{usage});
                return 2;
            };
        } else if (std.mem.eql(u8, args[i], "--shape") and i + 1 < args.len) {
            i += 1;
            shape = if (std.mem.eql(u8, args[i], "wide")) k1.fixture.wide else k1.fixture.parity;
        } else dir_path = args[i];
    }
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    if (dir.access(io, "config.json", .{})) |_| {} else |_| {
        std.debug.print("writing the seeded checkpoint to {s}\n", .{dir_path});
        try k1.fixture.write(gpa, io, dir, shape);
    }
    const device = try mtl.Device.init();
    defer device.deinit();
    var w = try k1.weights.load(gpa, io, device, dir_path);
    defer w.deinit();
    const c = w.config;
    var prng = std.Random.DefaultPrng.init(0x4b31);
    const ids = try a.alloc(u32, tokens_n);
    for (ids) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));
    const other = try a.dupe(u32, ids);
    for (other[prefix..]) |*t| t.* = prng.random().uintLessThan(u32, @intCast(c.vocab));

    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir_path);
    defer ck.close();
    var map = try k1.names.map(gpa, &ck, c);
    defer map.deinit();
    std.debug.print("float64 reference over {d} rows\n", .{ids.len});
    const want = try k1.ref.run(a, &map, c, ids);

    var ok = true;
    // A: the whole prompt in one call, with taps; twice, byte for byte
    var big = try fwd.Model.init(gpa, device, &w, tokens_n, tokens_n, ctl);
    defer big.deinit();
    const taps_buf = try device.buffer(c.layers * 2 * tokens_n * c.hidden * 4, mtl.ResourceOptions.shared);
    defer taps_buf.deinit();
    const one = try run(a, &big, ids, tokens_n, tokens_n, .{ .buf = taps_buf, .rows = tokens_n });
    var st: Stats = .{};
    compare(&st, one, want.logits, c.vocab, true);
    report("A one call: logits", st);
    ok = ok and st.ok;
    const tapped = taps_buf.slice(f32, c.layers * 2 * tokens_n * c.hidden);
    for (0..c.layers) |l| for (0..2) |h| {
        var ts: Stats = .{};
        compare(&ts, tapped[(l * 2 + h) * tokens_n * c.hidden ..][0 .. tokens_n * c.hidden], if (h == 0) want.attn[l] else want.moe[l], c.hidden, false);
        report(try std.fmt.allocPrint(a, "A layer {d} {s} residual", .{ l, if (h == 0) "attention" else "MoE" }), ts);
        ok = ok and ts.ok;
    };
    try big.rewind(0);
    ok = sameBytes("A again", one, try run(a, &big, ids, tokens_n, tokens_n, null)) and ok;

    // B: 16-row calls, then one row at a time past the ring's wrap
    var small = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer small.deinit();
    const chunked = try run(a, &small, ids, 16, 592, null);
    st = .{};
    compare(&st, chunked, want.logits, c.vocab, true);
    report(try std.fmt.allocPrint(a, "B 16-row calls, 1-row from 592 (ring {d})", .{small.ring}), st);
    ok = ok and st.ok;

    // C: 64 rows past a shared prefix, back to the prefix: the same rows again, then another suffix, == fresh
    var fork = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer fork.deinit();
    const first = try run(a, &fork, ids[0 .. prefix + 64], 16, prefix + 64, null);
    try fork.rewind(prefix);
    ok = sameBytes("C same suffix resumed == first", try run(a, &fork, ids[prefix..][0..64], 16, 64, null), first[prefix * c.vocab ..]) and ok;
    try fork.rewind(prefix);
    const forked = try run(a, &fork, other[prefix..], 16, tokens_n - prefix, null);
    var fresh = try fwd.Model.init(gpa, device, &w, tokens_n, 16, ctl);
    defer fresh.deinit();
    const other_fresh = try run(a, &fresh, other, 16, tokens_n, null);
    ok = sameBytes("C other suffix resumed == fresh", forked, other_fresh[prefix * c.vocab ..]) and ok;
    const refused = if (fork.rewind(prefix)) |_| false else |e| e == error.RingOverwritten;
    std.debug.print("{s} C a resume whose window the ring overwrote is refused\n", .{if (refused) "PASS" else "FAIL"});
    ok = ok and refused;

    std.debug.print("{s} kolibri1 metal selftest{s}\n", .{ if (ok) "PASS" else "FAIL", if (ctl.glue + ctl.rows + ctl.moe != 0) " (planted)" else "" });
    return if (ok) 0 else 1;
}
