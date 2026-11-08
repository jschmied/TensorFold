//! core_drafter.zig against the Pass called as lanes.zig calls it: the same drafts, bit for bit, on the CPU twin.
const std = @import("std");
const lanes = @import("lanes");
const twin = @import("twin.zig");
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");
const cd = @import("core_drafter.zig");

const gpa = std.testing.allocator;

fn features(t: iface.Taps) lanes.backend.Features {
    return .{ .buffer = t.address, .rows = t.rows, .row_bytes = t.stride * 4, .space = .host, .dtype = .f32 };
}

/// lanes.zig's ingest: the last `window` rows only.
fn refIngest(p: iface.Pass, shape: dspark.Shape, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) !void {
    const span = iface.ingestSpan(rows.len, shape.window);
    if (span.count > 0) try p.ingest(slot, start + span.skip, taps, rows[span.skip..]);
}

fn drive(smp: ?lanes.Sampling, prompt_len: usize, steps: usize) !void {
    const t = try twin.Twin.init(gpa, .{}, 2);
    defer t.deinit();
    const ref = try twin.Drafter.init(gpa, t, 2);
    defer ref.deinit();
    const dut = try twin.Drafter.init(gpa, t, 2);
    defer dut.deinit();
    const shape = ref.shape;
    const b = shape.block;
    var x = try cd.PassDrafter.init(gpa, dut.pass(), .{ .shape = shape, .slots = 2, .taps = &.{0}, .facts = .{ .depth = b, .step_ms = 1.0 } });
    defer x.deinit();
    const d = x.drafter();
    const prompt = try gpa.alloc(u32, prompt_len);
    defer gpa.free(prompt);
    for (prompt, 0..) |*p, i| p.* = @intCast((i * 7 + 3) % (t.dims.vocab - 1));
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = prompt, .max_new = 64, .eos = &.{}, .sampling = smp });
    defer s.deinit(gpa);
    const tgt = t.target();
    const rp = ref.pass();
    var tok = try tgt.prefill(0, prompt, smp, prompt.len);
    const all = try gpa.alloc(u32, prompt.len);
    defer gpa.free(all);
    for (all, 0..) |*r, i| r.* = @intCast(i);
    rp.reset(0);
    try refIngest(rp, shape, 0, 0, tgt.taps(0), all);
    try d.open(&s);
    try d.absorb(&.{.{ .stream = &s, .features = features(tgt.taps(0)), .start = 0, .follow = &.{} }});
    var len: u64 = prompt.len;
    var compared: usize = 0;
    for (0..steps) |step| {
        var want: [cd.max_block]u32 = undefined;
        var conf: [cd.max_block]f32 = undefined;
        var props = [_]iface.Proposal{.{ .drafts = want[0..b], .conf = conf[0..b] }};
        try rp.propose(&.{.{ .slot = 0, .anchor = tok, .start = len, .params = dspark.Params.of(smp, len) }}, &props);
        try d.hold(&.{.{ .stream = &s, .pending = tok, .position = len + 1, .depth = b }});
        var got: [cd.max_block]u32 = undefined;
        try d.held(&.{&s}, &.{got[0..b]});
        try std.testing.expectEqualSlices(u32, want[0..b], got[0..b]);
        compared += b;
        // verify the pending token and up to three drafts (odd steps: none), keep the accepted chain
        const w: usize = if (step % 2 == 0) 4 else 1;
        var toks: [4]u32 = undefined;
        var draws: [4]u64 = undefined;
        toks[0] = tok;
        @memcpy(toks[1..w], want[0 .. w - 1]);
        for (draws[0..w], 0..) |*dw, i| dw.* = len + 1 + i;
        var c: [4]u32 = undefined;
        var cs = [_][]u32{c[0..w]};
        try tgt.window(&.{.{ .slot = 0, .start = len, .tokens = toks[0..w], .parents = null, .draws = draws[0..w], .sampling = smp }}, &cs);
        var k: usize = 0;
        while (k + 1 < w and toks[k + 1] == c[k]) k += 1;
        const path = [_]u32{ 0, 1, 2, 3 };
        try tgt.keep(0, path[0 .. k + 1]);
        try refIngest(rp, shape, 0, len, tgt.taps(0), path[0 .. k + 1]);
        try d.absorb(&.{.{ .stream = &s, .features = features(tgt.taps(0)), .start = len, .follow = &.{}, .rows = path[0 .. k + 1], .prompt = false }});
        len += k + 1;
        tok = c[k];
    }
    d.release(&s);
    try std.testing.expect(compared == steps * b);
}

test "the adapter drafts what lanes.zig's chain pass drafts: greedy and keyed, short and windowed prompts" {
    for ([_]usize{ 6, 40 }) |n| {
        try drive(null, n, 12);
        try drive(.{ .seed = 11, .temperature = 0.8, .top_k = 12, .top_p = 0.9 }, n, 12);
    }
}

test "the adapter's slots: one a stream, reused after release; held never pads" {
    const t = try twin.Twin.init(gpa, .{}, 1);
    defer t.deinit();
    const p = try twin.Drafter.init(gpa, t, 1);
    defer p.deinit();
    var x = try cd.PassDrafter.init(gpa, p.pass(), .{ .shape = p.shape, .slots = 1, .taps = &.{0}, .facts = .{ .depth = 3, .step_ms = 1.0 } });
    defer x.deinit();
    const d = x.drafter();
    try std.testing.expectEqual(@as(u32, 3), d.facts().depth);
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2, 3 }, .max_new = 4, .eos = &.{} });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{ 1, 2, 3 }, .max_new = 4, .eos = &.{} });
    defer b.deinit(gpa);
    try d.open(&a);
    try std.testing.expectError(error.DraftStreamOpen, d.open(&a));
    try std.testing.expectError(error.TooManySlots, d.open(&b));
    var out: [1]u32 = undefined;
    try std.testing.expectError(error.DraftsShort, d.held(&.{&a}, &.{out[0..1]}));
    try std.testing.expectError(error.DraftDepth, d.hold(&.{.{ .stream = &a, .pending = 1, .position = 4, .depth = p.shape.block + 1 }}));
    d.release(&a);
    try d.open(&b);
    d.release(&b);
    try std.testing.expectError(error.DraftDepth, cd.PassDrafter.init(gpa, p.pass(), .{ .shape = p.shape, .slots = 1, .taps = &.{}, .facts = .{ .depth = p.shape.block + 1, .step_ms = 1.0 } }));
}
