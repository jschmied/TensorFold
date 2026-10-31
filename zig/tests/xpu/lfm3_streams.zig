//! LFM3 streams in shared decode steps against each stream alone, with a stream freed and another prefilled mid-way.

const std = @import("std");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const stop = @import("xpu").stop;
const Model = lfm3.model.Model;
const Seq = lfm3.model.Seq;

const steps = 64;

fn readIds(gpa: std.mem.Allocator, io: std.Io, ref: []const u8, name: []const u8) ![]u32 {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}/meta.json", .{ ref, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    const meta = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    const vals = meta.value.object.get("ids").?.array.items;
    const ids = try gpa.alloc(u32, vals.len);
    for (vals, ids) |v, *o| o.* = @intCast(v.integer);
    return ids;
}

/// The stream's prompt on its own sequence (bound), then its first token.
fn prefill(m: *Model, q: *Seq, ids: []const u32) !u32 {
    m.bind(q);
    try m.reset();
    try m.prefill(ids, null);
    return m.argmax();
}

/// Stream q alone: the prompt, then `steps` greedy tokens.
fn solo(m: *Model, q: *Seq, ids: []const u32, out: []u32) !void {
    out[0] = try prefill(m, q, ids);
    for (1..out.len) |j| {
        try m.forward(out[j - 1 .. j], true, null);
        out[j] = try m.argmax();
    }
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("skipped: usage tf-xpu-lfm3_streams-test MODEL_DIR REF_DIR\n", .{});
        return;
    }
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, io, args[1]);
    defer c.deinit();
    var l = try @import("xpu").loader.Loader.init(gpa, &r, args[1]);
    var m = try Model.load(gpa, &r, &l, &c, 2048, 128);
    const names = [_][]const u8{ "chat", "capitals", "items" };
    var ids: [3][]u32 = undefined;
    for (names, &ids) |n, *v| v.* = try readIds(gpa, io, args[2], n);
    var want: [3][steps]u32 = undefined;
    const q0 = try m.newSeq();
    for (0..3) |k| try solo(&m, q0, ids[k], &want[k]);
    // shared: chat and items decode together; items is freed at step 20 and capitals starts on a new sequence
    const qa = try m.newSeq();
    const qb = try m.newSeq();
    var got: [3][steps]u32 = undefined;
    got[0][0] = try prefill(&m, qa, ids[0]);
    got[2][0] = try prefill(&m, qb, ids[2]);
    var qs: [2]*Seq = .{ qa, qb };
    var toks: [2]u32 = undefined;
    var row: [2]i32 = undefined;
    var ja: usize = 1;
    var jb: usize = 1;
    while (ja < 21) : ({
        ja += 1;
        jb += 1;
    }) {
        toks = .{ got[0][ja - 1], got[2][jb - 1] };
        try m.forwardStreams(&qs, &toks);
        try m.streamTokens(&row);
        got[0][ja] = @intCast(row[0]);
        got[2][jb] = @intCast(row[1]);
    }
    m.freeSeq(qb);
    const qc = try m.newSeq();
    got[1][0] = try prefill(&m, qc, ids[1]);
    qs = .{ qa, qc };
    var jc: usize = 1;
    while (jc < steps - 21) : ({
        ja += 1;
        jc += 1;
    }) {
        toks = .{ got[0][ja - 1], got[1][jc - 1] };
        try m.forwardStreams(&qs, &toks);
        try m.streamTokens(&row);
        got[0][ja] = @intCast(row[0]);
        got[1][jc] = @intCast(row[1]);
    }
    var bad = false;
    // four streams in every step: chat, capitals, items, chat again (a duplicate must equal its twin)
    {
        const order = [_]usize{ 0, 1, 2, 0 };
        var q4: [4]*Seq = undefined;
        var g4: [4][steps]u32 = undefined;
        for (&q4, order, 0..) |*q, k, z| {
            q.* = try m.newSeq();
            g4[z][0] = try prefill(&m, q.*, ids[k]);
        }
        for (1..steps) |j| {
            var t4: [4]u32 = undefined;
            for (0..4) |z| t4[z] = g4[z][j - 1];
            try m.forwardStreams(&q4, &t4);
            var r4: [4]i32 = undefined;
            try m.streamTokens(&r4);
            for (0..4) |z| g4[z][j] = @intCast(r4[z]);
        }
        for (order, 0..) |k, z| {
            const eq = std.mem.eql(u32, &g4[z], &want[k]);
            if (!eq) bad = true;
            std.debug.print("  four streams, row {d} ({s}): {s}\n", .{ z, names[k], if (eq) "equal to alone" else "DIFFERS from alone" });
        }
    }
    // the server's round: haiku (a chat prompt from CHAT_JSON), chat, capitals and items as rows 0..3, 66 steps
    if (args.len > 3) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .unlimited);
        const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        const hv = doc.value.object.get("chat_haiku").?.array.items;
        const hids = try gpa.alloc(u32, hv.len);
        for (hv, hids) |v, *d| d.* = @intCast(v.integer);
        var hwant: [70]u32 = undefined;
        try solo(&m, q0, hids, &hwant);
        const prompts = [_][]const u32{ hids, ids[0], ids[1], ids[2] };
        var q4: [4]*Seq = undefined;
        var g4: [4][70]u32 = undefined;
        for (&q4, prompts, 0..) |*q, pr, z| {
            q.* = try m.newSeq();
            g4[z][0] = try prefill(&m, q.*, pr);
        }
        for (1..70) |j| {
            var t4: [4]u32 = undefined;
            for (0..4) |z| t4[z] = g4[z][j - 1];
            try m.forwardStreams(&q4, &t4);
            var r4: [4]i32 = undefined;
            try m.streamTokens(&r4);
            for (0..4) |z| g4[z][j] = @intCast(r4[z]);
        }
        std.debug.print("  haiku alone, tokens 58..66: {any}\n", .{hwant[58..66]});
        const eq = std.mem.eql(u32, &g4[0], &hwant);
        if (!eq) bad = true;
        const first = std.mem.indexOfDiff(u32, &g4[0], &hwant);
        std.debug.print("  haiku in the server's four-stream round: {s} {any}\n", .{ if (eq) "equal to alone" else "DIFFERS from alone at", first });
        for (1..4) |z| {
            const k = z - 1;
            const e2 = std.mem.eql(u32, g4[z][0..steps], &want[k]);
            if (!e2) bad = true;
            std.debug.print("  row {d} ({s}) in that round: {s}\n", .{ z, names[k], if (e2) "equal to alone" else "DIFFERS from alone" });
        }
    }
    const checks = [_]struct { k: usize, n: usize, what: []const u8 }{
        .{ .k = 0, .n = ja, .what = "chat, shared all the way (a partner freed and a new one prefilled at step 20)" },
        .{ .k = 2, .n = 21, .what = "items, shared until freed at step 20" },
        .{ .k = 1, .n = jc, .what = "capitals, prefilled mid-way, then shared" },
    };
    for (checks) |ck| {
        const eq = std.mem.eql(u32, got[ck.k][0..ck.n], want[ck.k][0..ck.n]);
        if (!eq) bad = true;
        const first = std.mem.indexOfDiff(u32, got[ck.k][0..ck.n], want[ck.k][0..ck.n]);
        std.debug.print("  {s}: {d} tokens {s}{s}\n", .{ ck.what, ck.n, if (eq) "equal to alone" else "DIFFER from alone at ", if (first) |f| try std.fmt.allocPrint(gpa, "{d}", .{f}) else "" });
    }
    std.debug.print("{s}\n", .{if (bad) "lfm3_streams: FAILED" else "lfm3_streams: ok"});
    if (bad) return error.TestFailed;
}
