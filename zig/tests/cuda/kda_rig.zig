//! The KDA tests' model: random weights, states and rows on the host, their device copies, calls and float64 runs.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const ref = core.kda_ref;
const K = cuda.kda;
pub const D = K.dim;

pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// Heads, slots, the gate (lower_bound < 0 the safe gate, 0 softplus), the pool's rows and a call's most rows.
pub const Cfg = struct { heads: usize, slots: usize, lower_bound: f32, eps: f32 = 1e-6, pool: usize, most: usize };

/// One stream's rows in a call: pool rows, the logged rows it keeps first, and whether it commits.
pub const Part = struct { slot: usize, rows: []const usize, kept: usize = 0, commit: bool = true };

pub const Kind = enum { rows, prompt };

pub const Model = struct {
    cfg: Cfg,
    a: std.mem.Allocator,
    conv: []f32, // [3 H D][4]
    a_log: []f32,
    dt_bias: []f32,
    norm: []f32,
    s0: []f32, // [slots][H][D][D]
    conv0: []u16, // [slots][3][3 H D]
    qkv: []u16, // pool [P][3 H D]
    f: []u16, // [P][H D]
    b: []u16, // [P][H]
    gate: []u16, // [P][H D]

    pub fn init(a: std.mem.Allocator, rng: std.Random, cfg: Cfg) !Model {
        const hd = cfg.heads * D;
        const m: Model = .{ .cfg = cfg, .a = a, .conv = try a.alloc(f32, 3 * hd * 4), .a_log = try a.alloc(f32, cfg.heads), .dt_bias = try a.alloc(f32, hd), .norm = try a.alloc(f32, D), .s0 = try a.alloc(f32, cfg.slots * hd * D), .conv0 = try a.alloc(u16, cfg.slots * 3 * 3 * hd), .qkv = try a.alloc(u16, cfg.pool * 3 * hd), .f = try a.alloc(u16, cfg.pool * hd), .b = try a.alloc(u16, cfg.pool * cfg.heads), .gate = try a.alloc(u16, cfg.pool * hd) };
        for (m.conv) |*x| x.* = rng.floatNorm(f32) * 0.5;
        for (m.a_log) |*x| x.* = rng.float(f32) * 2.5 - 1;
        const spread: f32 = if (cfg.lower_bound < 0) 1 else 3; // softplus: inputs on both sides of its switch at 20
        for (m.dt_bias) |*x| x.* = rng.floatNorm(f32) * 0.5;
        for (m.norm) |*x| x.* = 1 + rng.floatNorm(f32) * 0.1;
        for (m.s0) |*x| x.* = rng.floatNorm(f32) * 0.3;
        for (m.conv0) |*x| x.* = bf16(rng.floatNorm(f32));
        for (m.qkv) |*x| x.* = bf16(rng.floatNorm(f32));
        for (m.f) |*x| x.* = bf16(rng.floatNorm(f32) * spread);
        for (m.b) |*x| x.* = bf16(rng.floatNorm(f32) * 1.5);
        for (m.gate) |*x| x.* = bf16(rng.floatNorm(f32));
        return m;
    }

    /// One (slot, head) from the initial state through `seq`: y [rows][D] unrounded, and its final S.
    pub fn reference(m: *const Model, slot: usize, head: usize, seq: []const usize, y: []f64, tol: []f64, s: []f64) !void {
        const hd = m.cfg.heads * D;
        const a = m.a;
        const conv = try a.alloc(f64, 3 * D * 4);
        const win = try a.alloc([ref.keep]f64, 3 * D);
        for (0..3) |part| for (0..D) |d| {
            const c = part * hd + head * D + d;
            for (0..4) |j| conv[(part * D + d) * 4 + j] = m.conv[c * 4 + j];
            for (0..ref.keep) |j| win[part * D + d][j] = bf(m.conv0[(slot * 3 + j) * 3 * hd + c]);
        };
        const dtb = try a.alloc(f64, D);
        const nw = try a.alloc(f64, D);
        for (dtb, nw, 0..) |*x, *n, d| {
            x.* = m.dt_bias[head * D + d];
            n.* = m.norm[d];
        }
        for (s, m.s0[(slot * m.cfg.heads + head) * D * D ..][0 .. D * D]) |*x, v| x.* = v;
        var h: ref.Head = .{ .s = s, .win = win };
        const w: ref.Weights = .{ .conv = conv, .a_log = m.a_log[head], .dt_bias = dtb, .norm = nw, .lower_bound = m.cfg.lower_bound, .eps = m.cfg.eps };
        const x = try a.alloc(f64, 3 * D);
        const f = try a.alloc(f64, D);
        const g = try a.alloc(f64, D);
        const scratch = try a.alloc(f64, 5 * D);
        for (seq, 0..) |p, r| {
            for (0..3) |part| for (0..D) |d| {
                x[part * D + d] = bf(m.qkv[p * 3 * hd + part * hd + head * D + d]);
            };
            for (0..D) |d| {
                f[d] = bf(m.f[p * hd + head * D + d]);
                g[d] = bf(m.gate[p * hd + head * D + d]);
            }
            h.row(w, x, f, bf(m.b[p * m.cfg.heads + head]), g, scratch, y[r * D ..][0..D], tol[r * D ..][0..D]);
        }
    }
};

/// The device side: weights, the slots' arenas and one call's io, padded strides so rows are not packed.
pub const Dev = struct {
    m: *const Model,
    gpu: check.Gpu,
    s: cuda.Stream,
    k: *const K.Kda,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    w: K.Weights = undefined,
    st: K.State = undefined,
    sb: cuda.DeviceBuffer = undefined,
    cb: cuda.DeviceBuffer = undefined,
    io: [5]cuda.DeviceBuffer = undefined, // qkv, f, b, gate, y
    segs: cuda.DeviceBuffer = undefined,
    last: K.Io = undefined,

    const pad = 8;

    fn buf(dv: *Dev, bytes: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(dv.gpu.d, @max(bytes, 16));
        try dv.bufs.append(dv.gpu.gpa, b);
        return b;
    }

    fn up(dv: *Dev, bytes: []const u8) !u64 {
        const b = try dv.buf(bytes.len);
        try b.upload(0, bytes);
        return b.ptr;
    }

    pub fn init(m: *const Model, gpu: check.Gpu, s: cuda.Stream, k: *const K.Kda) !Dev {
        var dv: Dev = .{ .m = m, .gpu = gpu, .s = s, .k = k };
        const c = m.cfg;
        const sz = K.slotBytes(dv.shape());
        dv.w = .{ .conv = try dv.up(std.mem.sliceAsBytes(m.conv)), .a_log = try dv.up(std.mem.sliceAsBytes(m.a_log)), .dt_bias = try dv.up(std.mem.sliceAsBytes(m.dt_bias)), .norm = try dv.up(std.mem.sliceAsBytes(m.norm)) };
        dv.sb = try dv.buf(c.slots * sz.s);
        dv.cb = try dv.buf(c.slots * sz.conv);
        dv.st = .{ .s = dv.sb.ptr, .conv = dv.cb.ptr, .log_a = (try dv.buf(c.slots * sz.log)).ptr, .log_k = (try dv.buf(c.slots * sz.log)).ptr, .log_u = (try dv.buf(c.slots * sz.log)).ptr, .log_x = (try dv.buf(c.slots * sz.log_x)).ptr };
        const hd = c.heads * D;
        for (&dv.io, [_]usize{ 3 * hd, hd, c.heads, hd, hd }) |*b, width| b.* = try dv.buf(c.most * (width + pad) * 2);
        dv.segs = try dv.buf(c.slots * @sizeOf(K.Seg));
        try dv.reset();
        return dv;
    }

    pub fn deinit(dv: *Dev) void {
        for (dv.bufs.items) |*b| b.free();
        dv.bufs.deinit(dv.gpu.gpa);
    }

    pub fn shape(dv: *const Dev) K.Shape {
        return .{ .heads = dv.m.cfg.heads, .lower_bound = dv.m.cfg.lower_bound, .eps = dv.m.cfg.eps };
    }

    /// Every slot back to its initial state and conv window, logs poisoned.
    pub fn reset(dv: *Dev) !void {
        try dv.sb.upload(0, std.mem.sliceAsBytes(dv.m.s0));
        try dv.cb.upload(0, std.mem.sliceAsBytes(dv.m.conv0));
    }

    /// One call of `kind` over `parts` (a slot at most once); returns each row's output [rows][H D] bf16.
    pub fn call(dv: *Dev, a: std.mem.Allocator, kind: Kind, parts: []const Part) ![]u16 {
        const c = dv.m.cfg;
        const hd = c.heads * D;
        var n: usize = 0;
        for (parts) |p| n += p.rows.len;
        if (n > c.most) return error.TooManyRows;
        const widths = [_]usize{ 3 * hd, hd, c.heads, hd };
        const srcs = [_][]const u16{ dv.m.qkv, dv.m.f, dv.m.b, dv.m.gate };
        var segs: [16]K.Seg = undefined;
        var first: usize = 0;
        for (parts, 0..) |p, i| {
            segs[i] = .{ .first = @intCast(first), .rows = @intCast(p.rows.len), .kept = @intCast(p.kept), .flags = if (p.commit) K.commit else 0, .slot = @intCast(p.slot) };
            first += p.rows.len;
        }
        try dv.segs.upload(0, std.mem.sliceAsBytes(segs[0..parts.len]));
        for (widths, srcs, dv.io[0..4]) |width, src, b| {
            const host = try a.alloc(u16, @max(n, 1) * (width + pad));
            @memset(host, 0x7fc0); // padding reads NaN
            var r: usize = 0;
            for (parts) |p| for (p.rows) |row| {
                @memcpy(host[r * (width + pad) ..][0..width], src[row * width ..][0..width]);
                r += 1;
            };
            try b.upload(0, std.mem.sliceAsBytes(host));
        }
        const ld = struct {
            fn f(w: usize) i64 {
                return @intCast(w + pad);
            }
        }.f;
        const io: K.Io = .{ .qkv = dv.io[0].ptr, .qkv_ld = ld(3 * hd), .f = dv.io[1].ptr, .f_ld = ld(hd), .b = dv.io[2].ptr, .b_ld = ld(c.heads), .gate = dv.io[3].ptr, .gate_ld = ld(hd), .y = dv.io[4].ptr, .y_ld = ld(hd) };
        try dv.io[4].fill8(0xff, dv.s.handle);
        dv.last = io;
        switch (kind) {
            .rows => try dv.k.rows(dv.s, io, dv.w, dv.st, dv.segs.ptr, parts.len, dv.shape()),
            .prompt => try dv.k.prompt(dv.s, io, dv.w, dv.st, dv.segs.ptr, parts.len, dv.shape()),
        }
        try dv.s.synchronize();
        const raw = try a.alloc(u16, @max(n, 1) * (hd + pad));
        try dv.io[4].download(0, std.mem.sliceAsBytes(raw));
        const y = try a.alloc(u16, n * hd);
        for (0..n) |r| @memcpy(y[r * hd ..][0..hd], raw[r * (hd + pad) ..][0..hd]);
        return y;
    }

    /// The last call's launch again on its io and segments, not waited for (timing).
    pub fn again(dv: *Dev, kind: Kind, n: usize) !void {
        switch (kind) {
            .rows => try dv.k.rows(dv.s, dv.last, dv.w, dv.st, dv.segs.ptr, n, dv.shape()),
            .prompt => try dv.k.prompt(dv.s, dv.last, dv.w, dv.st, dv.segs.ptr, n, dv.shape()),
        }
    }

    /// A slot's S [H][D][D] and conv window [3][3 H D].
    pub fn state(dv: *Dev, a: std.mem.Allocator, slot: usize) !struct { s: []f32, conv: []u16 } {
        const sz = K.slotBytes(dv.shape());
        const s = try a.alloc(f32, sz.s / 4);
        const conv = try a.alloc(u16, sz.conv / 2);
        try dv.sb.download(slot * sz.s, std.mem.sliceAsBytes(s));
        try dv.cb.download(slot * sz.conv, std.mem.sliceAsBytes(conv));
        return .{ .s = s, .conv = conv };
    }
};

/// Worst fraction of each element's tolerance (kda_ref's bound) over a head's rows.
pub fn yFraction(got: []const u16, want: []const f64, tol: []const f64) f64 {
    var worst: f64 = 0;
    for (got, want, tol) |g, x, t| worst = @max(worst, @abs(bf(g) - x) / (t + 1e-30));
    return worst;
}

/// Worst fraction of the state's tolerance: 2^-11 max|want| over the head (fp32 sums, a rare bf16 input one ulp off).
pub fn sFraction(got: []const f32, want: []const f64) f64 {
    var most: f64 = 0;
    for (want) |x| most = @max(most, @abs(x));
    var worst: f64 = 0;
    for (got, want) |g, x| worst = @max(worst, @abs(@as(f64, g) - x) / (std.math.pow(f64, 2, -11) * most + 1e-30));
    return worst;
}
