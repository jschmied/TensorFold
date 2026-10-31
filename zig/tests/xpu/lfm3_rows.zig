//! LFM3's N-row decode matvecs (several streams a launch) against the one-row kernels, bit for bit, on random data.

const std = @import("std");
const o = @import("lfm3_ops.zig");
const stop = @import("xpu").stop;
const kernels = @import("xpu").kernels;

const ROWS: u32 = 1024;

fn randWords(r: *o.rt.Runtime, n: usize, rnd: std.Random) !o.Buf {
    const h = try o.gpa.alloc(u32, n);
    for (h) |*v| v.* = rnd.int(u32);
    const b = try r.alloc(n * 4);
    try r.upload(b, std.mem.sliceAsBytes(h));
    return b;
}

fn at(b: o.Buf, off: usize) o.Buf {
    return .{ .rt = b.rt, .ptr = @ptrFromInt(@intFromPtr(b.ptr.?) + off), .len = b.len - off };
}

const Table = struct { w: o.Buf, s: o.Buf, b: o.Buf };

fn table(r: *o.rt.Runtime, rows: u32, in_dim: u32, rnd: std.Random) !Table {
    const sh = try o.gpa.alloc(u16, rows * in_dim / 64);
    const bh = try o.gpa.alloc(u16, sh.len);
    o.randBf(sh, rnd, 0.02);
    o.randBf(bh, rnd, 0.1);
    return .{ .w = try randWords(r, rows * in_dim / 8, rnd), .s = try o.upload(r, sh), .b = try o.upload(r, bh) };
}

fn fetch32(r: *o.rt.Runtime, b: o.Buf, n: usize) ![]u32 {
    const h = try o.gpa.alloc(u32, n);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(h), b);
    try r.sync();
    return h;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main() !void {
    stop.install();
    var r = try o.rt.open();
    defer r.deinit();
    var am = try r.module(kernels.attn);
    var vm = try r.module(kernels.lfm3_mv);
    var pm = try r.module(if (kernels.dpas) kernels.nem_prow else kernels.nem_prow_gen12);
    const tile_outs: u32 = if (kernels.dpas) 8 else 4;
    var k_bf = try am.kernel("qmv4_bf", .{ 64, 1, 1 });
    var k_f32 = try am.kernel("qmv4_f32", .{ 64, 1, 1 });
    var k_bfn = try vm.kernel("qmv4_bf_n", .{ 64, 1, 1 });
    var k_f32n = try vm.kernel("qmv4_f32_n", .{ 64, 1, 1 });
    var k_sw = try vm.kernel("qmv4_swiglu", .{ 64, 1, 1 });
    var k_swn = try vm.kernel("qmv4_swiglu_n", .{ 64, 1, 1 });
    var k_x4 = try vm.kernel("qmv4_bf_x4", .{ 64, 1, 1 });
    var k_x4n = try vm.kernel("qmv4_bf_x4_n", .{ 64, 1, 1 });
    var k_pd = try pm.kernel("pdense", .{ 16 * (16 / tile_outs), 1, 1 });
    var t: o.Tally = .{ .name = "lfm3_rows" };
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    std.debug.print("N-row decode matvecs against the one-row kernels, {d} output rows, bit for bit\n", .{ROWS});
    for ([_]u32{ 2048, 7680 }) |in_dim| {
        const a = try table(&r, ROWS, in_dim, rnd);
        const c = try table(&r, ROWS, in_dim, rnd);
        const xh = try o.gpa.alloc(u16, 8 * in_dim);
        o.randBf(xh, rnd, 1.0);
        const x = try o.upload(&r, xh);
        const y1 = try r.alloc(8 * 4 * ROWS * 4);
        const yn = try r.alloc(8 * 4 * ROWS * 4);
        for ([_]u32{ 1, 2, 3, 4, 8 }) |n| {
            var buf: [96]u8 = undefined;
            // bf16 out
            for (0..n) |z| try o.run(&k_bf, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, x, y1, in_dim, @as(u32, @intCast(z)) * in_dim, @as(u32, @intCast(z)) * ROWS, ROWS });
            try o.run(&k_bfn, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, x, yn, in_dim, ROWS, n, ROWS, @as(u32, 0) });
            var h1 = try o.fetch(&r, y1, n * ROWS);
            var hn = try o.fetch(&r, yn, n * ROWS);
            t.same(try std.fmt.bufPrint(&buf, "qmv4_bf_n     in {d} n {d} vs qmv4_bf", .{ in_dim, n }), std.mem.eql(u16, h1, hn));
            try o.run(&k_pd, .{ ROWS / 16, (n + 7) / 8, 1 }, .{ a.w, a.s, a.b, x, yn, in_dim, ROWS, n, ROWS, @as(u32, 0), @as(u32, 0) });
            hn = try o.fetch(&r, yn, n * ROWS);
            const pd_same = std.mem.eql(u16, h1, hn);
            std.debug.print("  (info) pdense       in {d} n {d} vs qmv4_bf: {s}\n", .{ in_dim, n, if (pd_same) "bitwise identical" else "differs" });
            // fp32 out (the head)
            for (0..n) |z| try o.run(&k_f32, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, x, y1, in_dim, @as(u32, @intCast(z)) * in_dim, @as(u32, @intCast(z)) * ROWS, ROWS });
            try o.run(&k_f32n, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, x, yn, in_dim, ROWS, n });
            const f1 = try fetch32(&r, y1, n * ROWS);
            t.same(try std.fmt.bufPrint(&buf, "qmv4_f32_n    in {d} n {d} vs qmv4_f32", .{ in_dim, n }), std.mem.eql(u32, f1, try fetch32(&r, yn, n * ROWS)));
            // SwiGLU of two tables
            for (0..n) |z| try o.run(&k_sw, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, c.w, c.s, c.b, at(x, z * in_dim * 2), at(y1, z * ROWS * 2), in_dim, ROWS });
            try o.run(&k_swn, .{ ROWS / 4, 1, 1 }, .{ a.w, a.s, a.b, c.w, c.s, c.b, x, yn, in_dim, ROWS, n });
            h1 = try o.fetch(&r, y1, n * ROWS);
            hn = try o.fetch(&r, yn, n * ROWS);
            t.same(try std.fmt.bufPrint(&buf, "qmv4_swiglu_n in {d} n {d} vs qmv4_swiglu", .{ in_dim, n }), std.mem.eql(u16, h1, hn));
            // four tables (two of them repeated), outputs ROWS * 4 apart a row
            const st: u32 = 4 * ROWS;
            for (0..n) |z| try o.run(&k_x4, .{ ROWS, 1, 1 }, .{ a.w, a.s, a.b, at(y1, z * st * 2), ROWS, @as(u32, 0), c.w, c.s, c.b, at(y1, z * st * 2), ROWS, ROWS, a.w, a.s, a.b, at(y1, z * st * 2), ROWS, 2 * ROWS, c.w, c.s, c.b, at(y1, z * st * 2), ROWS, 3 * ROWS, at(x, z * in_dim * 2), in_dim });
            try o.run(&k_x4n, .{ ROWS, 1, 1 }, .{ a.w, a.s, a.b, yn, ROWS, @as(u32, 0), st, c.w, c.s, c.b, yn, ROWS, ROWS, st, a.w, a.s, a.b, yn, ROWS, 2 * ROWS, st, c.w, c.s, c.b, yn, ROWS, 3 * ROWS, st, x, in_dim, n });
            h1 = try o.fetch(&r, y1, n * st);
            hn = try o.fetch(&r, yn, n * st);
            t.same(try std.fmt.bufPrint(&buf, "qmv4_bf_x4_n  in {d} n {d} vs qmv4_bf_x4", .{ in_dim, n }), std.mem.eql(u16, h1, hn));
        }
    }
    try t.finish();
}
