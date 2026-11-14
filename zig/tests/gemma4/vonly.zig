//! Gemma 4 V-only global layers on Metal: rebuilt keys against gemma_qkv_prep's, attention K+V against V-only, timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ks = @import("kernel_sources");

const g4 = tf.gemma4;
const geo = g4.config.dense31b;
const DH = g4.config.global_dim;
const NQ = geo.query_heads;
const NK = geo.global_kv_heads;
const G = NQ / NK;
const W = NQ * DH + NK * DH; // a q|k row; v is k's raw values
const CD = 256; // gemma_decode_attention's keys a chunk
const opts = mtl.ResourceOptions.shared;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

const Pipes = struct { prep: mtl.Pipeline, rebuild: mtl.Pipeline, plant: mtl.Pipeline, decode: mtl.Pipeline, decode_v: mtl.Pipeline, decode_plant: mtl.Pipeline, prompt: mtl.Pipeline };

fn library(device: mtl.Device, a: std.mem.Allocator, plant: bool) !mtl.Library {
    const base = try g4.kernels.promptSource(device, a, geo, true);
    const text = try std.mem.concat(a, u8, &.{ if (plant) "#define TF_KEY_PLANT\n" else "", base, ks.gemma4_vonly_prompt });
    return mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
}

fn pipes(device: mtl.Device, a: std.mem.Allocator) !Pipes {
    const glue = try mtl.Library.fromSource(device, try g4.kernels.glueSource(a, geo, true, false, 4), mtl.CompileOptions.mlx());
    const lib = try library(device, a, false);
    const bad = try library(device, a, true);
    return .{
        .prep = try mtl.Pipeline.init(device, glue, "gemma_qkv_prep", false),
        .rebuild = try mtl.Pipeline.init(device, lib, "gemma_key_rebuild", false),
        .plant = try mtl.Pipeline.init(device, bad, "gemma_key_rebuild", false),
        .decode = try mtl.Pipeline.init(device, lib, "gemma_decode_attention", false),
        .decode_v = try mtl.Pipeline.init(device, lib, "gemma_decode_attention_v", false),
        .decode_plant = try mtl.Pipeline.init(device, bad, "gemma_decode_attention_v", false),
        .prompt = try mtl.Pipeline.init(device, lib, "gemma_prompt_attention", false),
    };
}

/// The layer's fixed inputs: q and k norm weights, the 31B's global inverse frequencies (64 rotated pairs).
const Fixed = struct { qw: mtl.Buffer, kw: mtl.Buffer, invf: mtl.Buffer };

fn fixed(device: mtl.Device, rng: std.Random) !Fixed {
    const f: Fixed = .{ .qw = try device.buffer(DH * 2, opts), .kw = try device.buffer(DH * 2, opts), .invf = try device.buffer(DH / 2 * 4, opts) };
    for (f.qw.slice(u16, DH), f.kw.slice(u16, DH)) |*q, *k| {
        q.* = bf16(0.25 + 1.5 * rng.float(f32));
        k.* = bf16(0.25 + 1.5 * rng.float(f32));
    }
    for (f.invf.slice(f32, DH / 2), 0..) |*v, i| {
        const e: f64 = @as(f64, @floatFromInt(2 * i)) / DH;
        v.* = if (i < g4.config.global_rotated) @floatCast(1.0 / std.math.pow(f64, g4.config.global_theta, e)) else 0;
    }
    return f;
}

/// Raw q|k rows with norms near `scale`; `edges`: rows 0..8 of the k part carry the edge values per head.
fn rawRows(raw: []u16, rng: std.Random, edges: bool) void {
    for (raw) |*x| x.* = bf16(@floatCast(rng.floatNorm(f64) * 1.5));
    if (!edges) return;
    const rows = raw.len / W;
    if (rows < 9) return;
    for (0..NK) |h| {
        const at = struct {
            fn f(r: []u16, row: usize, head: usize) []u16 {
                return r[row * W + NQ * DH + head * DH ..][0..DH];
            }
        }.f;
        @memset(at(raw, 0, h), 0x0000);
        @memset(at(raw, 1, h), 0x8000);
        at(raw, 2, h)[0] = 0x8000;
        at(raw, 3, h)[5] = 0x0001;
        at(raw, 3, h)[300] = 0x8001;
        at(raw, 4, h)[1] = 0x000b;
        at(raw, 5, h)[3] = 0x7fc0;
        at(raw, 6, h)[9] = 0x7f80;
        at(raw, 7, h)[9] = 0xff80;
        at(raw, 8, h)[64] = 0x0080;
    }
}

/// gemma_qkv_prep over `n` raw rows at positions pos: q [n, NQ, DH], k and v [NK, n, DH]; cache: k and v at pos too.
fn prep(queue: mtl.Queue, p: *const Pipes, f: Fixed, raw: mtl.Buffer, pos: mtl.Buffer, n: usize, q: mtl.Buffer, k: mtl.Buffer, v: mtl.Buffer, kc: ?mtl.Buffer, vc: ?mtl.Buffer, cap: usize) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(p.prep);
    e.setBuffer(raw, 0, 0);
    e.setBuffer(f.qw, 0, 1);
    e.setBuffer(f.kw, 0, 2);
    e.setBuffer(f.invf, 0, 3);
    e.setBuffer(pos, 0, 4);
    e.setValue(@as(f32, 1e-6), 5);
    e.setBuffer(q, 0, 6);
    e.setBuffer(k, 0, 7);
    e.setBuffer(v, 0, 8);
    e.setBuffer(kc orelse k, 0, 9);
    e.setBuffer(vc orelse v, 0, 10);
    e.setValue([3]i32{ 0, if (kc != null) @intCast(cap) else 0, 0 }, 11);
    e.dispatchGroups(.{ .width = NQ + 2 * NK, .height = n }, .{ .width = 32 });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

fn rebuild(e: mtl.ComputeEncoder, pipe: mtl.Pipeline, f: Fixed, v: mtl.Buffer, pos: ?mtl.Buffer, n: usize, keys: usize, out: mtl.Buffer) void {
    e.setPipeline(pipe);
    e.setBuffer(v, 0, 0);
    e.setBuffer(f.kw, 0, 1);
    e.setBuffer(f.invf, 0, 2);
    e.setBuffer(pos orelse v, 0, 3);
    e.setValue([2]i32{ @intCast(n), @intFromBool(pos != null) }, 4);
    e.setBuffer(out, 0, 5);
    e.dispatchGroups(.{ .width = keys, .height = NK }, .{ .width = 32 });
}

fn differing(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(x != y);
    return n;
}

var failed = false;

fn verdict(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} " ++ fmt ++ "\n", .{if (ok) "PASS" else "FAIL"} ++ args);
    if (!ok) failed = true;
}

/// Keys rebuilt from gemma_qkv_prep's values against its keys, edges and far positions; the planted rebuild red.
fn keysCheck(device: mtl.Device, queue: mtl.Queue, p: *const Pipes, f: Fixed, rng: std.Random) !void {
    const Case = struct { p0: usize, n: usize, edges: bool };
    const cases = [_]Case{ .{ .p0 = 0, .n = 64, .edges = true }, .{ .p0 = 250, .n = 64, .edges = true }, .{ .p0 = 65530, .n = 64, .edges = true }, .{ .p0 = 70000, .n = 64, .edges = true }, .{ .p0 = 262100, .n = 64, .edges = true }, .{ .p0 = 70000, .n = 512, .edges = false }, .{ .p0 = 0, .n = 3000, .edges = false } };
    for (cases) |c| {
        const raw = try device.buffer(c.n * W * 2, opts);
        const pos = try device.buffer(c.n * 4, opts);
        const q = try device.buffer(c.n * NQ * DH * 2, opts);
        const k = try device.buffer(NK * c.n * DH * 2, opts);
        const v = try device.buffer(NK * c.n * DH * 2, opts);
        const k2 = try device.buffer(NK * c.n * DH * 2, opts);
        defer for ([_]mtl.Buffer{ raw, pos, q, k, v, k2 }) |b| b.deinit();
        rawRows(raw.slice(u16, c.n * W), rng, c.edges);
        for (pos.slice(i32, c.n), 0..) |*x, j| x.* = @intCast(c.p0 + j);
        try prep(queue, p, f, raw, pos, c.n, q, k, v, null, null, 0);
        for ([_]bool{ true, false }) |plant| {
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            rebuild(e, if (plant) p.plant else p.rebuild, f, v, pos, c.n, c.n, k2);
            e.end();
            cb.commit();
            cb.wait();
            const total = NK * c.n * DH;
            const bad = differing(k.slice(u16, total), k2.slice(u16, total));
            if (plant) {
                verdict(bad > 0, "control: planted rebuild red at positions {d}.. ({d} of {d} key values differ)", .{ c.p0, bad, total });
            } else verdict(bad == 0, "keys rebuilt at positions {d}..{d}{s}: {d} of {d} values differ", .{ c.p0, c.p0 + c.n - 1, if (c.edges) " with edge values" else "", bad, total });
        }
    }
}

const Cache = struct { kc: mtl.Buffer, vc: mtl.Buffer, cap: usize };

/// The global layer's cache through position ctx - 1, written by gemma_qkv_prep 512 rows at a time.
fn cache(device: mtl.Device, queue: mtl.Queue, p: *const Pipes, f: Fixed, rng: std.Random, ctx: usize) !Cache {
    const cap = ctx + CD;
    const c: Cache = .{ .kc = try device.buffer(NK * cap * DH * 2, opts), .vc = try device.buffer(NK * cap * DH * 2, opts), .cap = cap };
    @memset(c.kc.slice(u16, NK * cap * DH), 0x7fc0); // never-written rows: NaN, as a fresh cache might hold
    @memset(c.vc.slice(u16, NK * cap * DH), 0x7fc0);
    const n = 512;
    const raw = try device.buffer(n * W * 2, opts);
    const pos = try device.buffer(n * 4, opts);
    const q = try device.buffer(n * NQ * DH * 2, opts);
    const k = try device.buffer(NK * n * DH * 2, opts);
    const v = try device.buffer(NK * n * DH * 2, opts);
    defer for ([_]mtl.Buffer{ raw, pos, q, k, v }) |b| b.deinit();
    rawRows(raw.slice(u16, n * W), rng, true);
    var p0: usize = 0;
    while (p0 < ctx) : (p0 += n) {
        const rows = @min(n, ctx - p0);
        for (pos.slice(i32, n)[0..rows], 0..) |*x, j| x.* = @intCast(p0 + j);
        try prep(queue, p, f, raw, pos, rows, q, k, v, c.kc, c.vc, cap);
    }
    return c;
}

fn median(ms: []f64) f64 {
    std.mem.sort(f64, ms, {}, std.sort.asc(f64));
    return ms[ms.len / 2];
}

const Timed = struct { ms: f64, out: []u32 };

/// Decode partials of `rows` rows ending at ctx - 1: K+V (kc set) or V-only, `reps` launches a buffer, median of 7.
fn decode(device: mtl.Device, queue: mtl.Queue, pipe: mtl.Pipeline, f: Fixed, c: Cache, kc: bool, q: mtl.Buffer, rows: usize, ctx: usize, reps: usize, a: std.mem.Allocator) !Timed {
    const p0 = ctx - rows;
    const nch = (ctx + CD - 1) / CD;
    const parts = NQ * rows * nch;
    const pm = try device.buffer(parts * 4, opts);
    const pl = try device.buffer(parts * 4, opts);
    const po = try device.buffer(parts * DH * 4, opts);
    defer for ([_]mtl.Buffer{ pm, pl, po }) |b| b.deinit();
    var ms: [7]f64 = undefined;
    for (0..8) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| {
            e.setPipeline(pipe);
            e.setBuffer(q, 0, 0);
            e.setBuffer(if (kc) c.kc else c.vc, 0, 1);
            e.setBuffer(c.vc, 0, 2);
            e.setValue([4]i32{ @intCast(rows), @intCast(p0), @intCast(c.cap), @intCast(nch) }, 3);
            e.setBuffer(pm, 0, 4);
            e.setBuffer(pl, 0, 5);
            e.setBuffer(po, 0, 6);
            e.setBuffer(f.kw, 0, 7);
            e.setBuffer(f.invf, 0, 8);
            e.dispatchGroups(.{ .width = nch, .height = NK, .depth = (rows * G + 15) / 16 }, .{ .width = 32 * DH / 128 });
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        if (t > 0) ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
    }
    const out = try a.alloc(u32, parts * (DH + 2));
    @memcpy(out[0..parts], pm.slice(u32, parts));
    @memcpy(out[parts .. 2 * parts], pl.slice(u32, parts));
    @memcpy(out[2 * parts ..], po.slice(u32, parts * DH));
    return .{ .ms = median(&ms), .out = out };
}

/// A prompt chunk's 512 rows ending at ctx - 1 on cached keys, or (scratch set) on keys rebuilt into it first.
fn prompt(device: mtl.Device, queue: mtl.Queue, p: *const Pipes, f: Fixed, c: Cache, scratch: ?mtl.Buffer, q: mtl.Buffer, ctx: usize, a: std.mem.Allocator) !Timed {
    const rows = 512;
    const o = try device.buffer(rows * NQ * DH * 2, opts);
    const hi = try device.buffer(rows * 4, opts);
    defer for ([_]mtl.Buffer{ o, hi }) |b| b.deinit();
    var ms: [7]f64 = undefined;
    for (0..8) |t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        if (scratch) |s| {
            rebuild(e, p.rebuild, f, c.vc, null, c.cap, ctx, s);
            e.barrier();
        }
        e.setPipeline(p.prompt);
        e.setBuffer(q, 0, 0);
        e.setBuffer(scratch orelse c.kc, 0, 1);
        e.setBuffer(c.vc, 0, 2);
        e.setBuffer(o, 0, 3);
        e.setValue([3]i32{ rows, @intCast(ctx - rows), @intCast(c.cap) }, 4);
        e.setBuffer(hi, 0, 5);
        e.dispatchGroups(.{ .width = rows / 32, .height = NQ }, .{ .width = 32 * DH / 64 });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        if (t > 0) ms[t - 1] = (cb.gpuEnd() - cb.gpuStart()) * 1e3;
    }
    const out = try a.alloc(u32, rows * NQ * DH / 2);
    @memcpy(out, o.slice(u32, rows * NQ * DH / 2));
    return .{ .ms = median(&ms), .out = out };
}

fn differ32(x: []const u32, y: []const u32) usize {
    var n: usize = 0;
    for (x, y) |u, v| n += @intFromBool(u != v);
    return n;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("gemma4 V-only on {s}, tensor units {s}: 31B global layer, {d}/{d} heads of {d}\n", .{ device.name(), if (device.tensorUnits()) "yes" else "no", NQ, NK, DH });
    const p = try pipes(device, a);
    var prng = std.Random.DefaultPrng.init(0x3312);
    const rng = prng.random();
    const f = try fixed(device, rng);
    try keysCheck(device, queue, &p, f, rng);
    for ([_]usize{ 8192, 32768 }) |ctx| {
        const c = try cache(device, queue, &p, f, rng, ctx);
        defer for ([_]mtl.Buffer{ c.kc, c.vc }) |b| b.deinit();
        const q = try device.buffer(512 * NQ * DH * 2, opts);
        defer q.deinit();
        for (q.slice(u16, 512 * NQ * DH)) |*x| x.* = bf16(@floatCast(rng.floatNorm(f64)));
        for ([_]usize{ 1, 4 }) |rows| {
            const kv = try decode(device, queue, p.decode, f, c, true, q, rows, ctx, 50, a);
            const vo = try decode(device, queue, p.decode_v, f, c, false, q, rows, ctx, 50, a);
            const bad = try decode(device, queue, p.decode_plant, f, c, false, q, rows, ctx, 1, a);
            verdict(differ32(kv.out, vo.out) == 0, "decode {d} row(s) at {d}: V-only partials equal K+V's ({d} of {d} words differ)", .{ rows, ctx, differ32(kv.out, vo.out), kv.out.len });
            verdict(differ32(kv.out, bad.out) > 0, "control: planted V-only decode red at {d} ({d} words differ)", .{ ctx, differ32(kv.out, bad.out) });
            std.debug.print("TIME decode {d} row(s) at {d}: K+V {d:.4} ms, V-only {d:.4} ms ({d:.1} %)\n", .{ rows, ctx, kv.ms, vo.ms, (vo.ms / kv.ms - 1) * 100 });
        }
        const scratch = try device.buffer(NK * c.cap * DH * 2, opts);
        defer scratch.deinit();
        const kv = try prompt(device, queue, &p, f, c, null, q, ctx, a);
        const vo = try prompt(device, queue, &p, f, c, scratch, q, ctx, a);
        verdict(differ32(kv.out, vo.out) == 0, "prompt chunk of 512 at {d}: rebuilt-key output equals cached-key output ({d} words differ)", .{ ctx, differ32(kv.out, vo.out) });
        std.debug.print("TIME prompt chunk 512 at {d}: K+V {d:.3} ms, V-only (rebuild {d} keys + chunk) {d:.3} ms ({d:.1} %)\n", .{ ctx, kv.ms, ctx, vo.ms, (vo.ms / kv.ms - 1) * 100 });
    }
    if (failed) {
        std.debug.print("VONLY FAIL\n", .{});
        std.process.exit(1);
    }
    std.debug.print("VONLY PASS\n", .{});
}
