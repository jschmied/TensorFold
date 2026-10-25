//! `ling-ids-logits DIR IDS OUT [--decode N]`: logits of given i32 ids (prompt pass, then N decode rows) to OUT.logits.
const std = @import("std");
const check = @import("check.zig");
const ling = @import("ling_cuda");
const frig = @import("ling_forward_rig.zig");

const Engine = ling.engine.Engine;

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (rest.len < 3) return error.MissingArgument;
    var decode: usize = 0;
    var i: usize = 3;
    while (i + 1 < rest.len) : (i += 1) if (std.mem.eql(u8, rest[i], "--decode")) {
        decode = try std.fmt.parseInt(usize, rest[i + 1], 10);
    };
    const raw = try std.Io.Dir.cwd().readFileAlloc(gpu.io, rest[1], a, .limited(1 << 26));
    const T = raw.len / 4;
    const ids = try a.alloc(i32, T);
    @memcpy(std.mem.sliceAsBytes(ids), raw[0 .. T * 4]);
    if (decode >= T) return error.BadDecodeCount;
    const P = T - decode;
    const e = try Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, rest[0], .{ .capacity = @max(1024, T + 16), .prompt_rows = @max(512, P), .layers = null, .mtp = false });
    defer e.deinit();
    const V: usize = e.c.vocab;
    const top: usize = e.c.topk;
    var nm: usize = 0;
    for (e.main.layers) |l| nm += @intFromBool(l.ffn == .moe);
    var r: frig.Rig = .{ .gpu = gpu, .a = a, .s = e.stream, .m = &e.m, .g = &e.g };
    defer r.deinit();
    try e.bind(30);
    try e.begin(null);
    const st = &e.cur.st;
    const logits = try a.alloc(u16, T * V);
    const picks = try a.alloc(i32, nm * T * top);
    const pp = try r.zeros(nm * P * (top + 1) * 4);
    @memcpy(logits[0 .. P * V], try r.run(&e.bp, &.{.{ .st = st, .row0 = 0, .rows = P }}, ids[0..P], pp));
    const sp = try frig.sortedPicks(a, try r.back(pp, nm * P * (top + 1) * 4), nm, P, top);
    for (0..nm) |l| @memcpy(picks[l * T * top ..][0 .. P * top], sp[l * P * top ..][0 .. P * top]);
    try r.fwd(&e.bp, null).commit(st, P);
    const dp = try r.zeros(nm * (top + 1) * 4);
    for (P..T) |t| {
        @memcpy(logits[t * V ..][0..V], try r.run(&e.bd, &.{.{ .st = st, .row0 = 0, .rows = 1 }}, ids[t .. t + 1], dp));
        const one = try frig.sortedPicks(a, try r.back(dp, nm * (top + 1) * 4), nm, 1, top);
        for (0..nm) |l| @memcpy(picks[(l * T + t) * top ..][0..top], one[l * top ..][0..top]);
        try r.fwd(&e.bd, null).commit(st, 1);
    }
    e.release(30);
    var name: [512]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fmt.bufPrint(&name, "{s}.logits", .{rest[2]}), .data = std.mem.sliceAsBytes(logits) });
    try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = try std.fmt.bufPrint(&name, "{s}.picks", .{rest[2]}), .data = std.mem.sliceAsBytes(picks) });
    check.pass("ling-ids-logits {t}: {d} ids ({d} prompt rows, {d} decode rows), {d} MoE layers, vocabulary {d}", .{ e.c.quant, T, P, decode, nm, V });
}
