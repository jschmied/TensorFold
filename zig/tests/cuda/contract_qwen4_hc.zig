//! Flash Next's hyper-connection read-out at 4 x 5120 with rank 640 (a provisional Qwen4 dense shape), against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const tk = flash.tile_kernels;

pub const dims = 5120;
pub const streams = 4;
pub const wide = streams * dims;
pub const low = 640;
const n_down = low + streams;
const nc = dims / 256;
const sk = 32;
const eps: f32 = 1e-6;
const u: f64 = 0x1p-8; // bf16's unit roundoff
const f32u: f64 = 0x1p-24;

/// tiles: the fused prompt tiles (hcf_down, hcf_up); decode: the sliced chain (down, upmix).
pub const Form = enum { tiles, decode };

/// The control feeds the device one up-projection word the float64 side does not have.
pub const forms = [_]struct { Form, bool }{ .{ .tiles, false }, .{ .decode, false }, .{ .tiles, true } };

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn spec(f: Form, control: bool) c.Spec {
    const name = if (control) "control qwen4 HC read-out with one wrong up-projection word" else switch (f) {
        .tiles => "cuda qwen4 synthetic HC read-out 4 x 5120 rank 640, prompt tiles (hcf_down, hc_reduce_act, hcf_up, mix_h)",
        .decode => "cuda qwen4 synthetic HC read-out 4 x 5120 rank 640, sliced chain (fn_hc_down, fn_hc_reduce_act, fn_hc_upmix)",
    };
    // rows = token rows, n = hidden, k = the streams' width
    return .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "act, xs, inj, mixed and xsm each in float64's class, stage by stage" }, .tiles = .{ .{ .size = 64 }, .{}, .{} }, .served = &.{ .{ .rows = 64, .n = dims, .k = wide }, .{ .rows = 96, .n = dims, .k = wide } } };
}

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub const Hc = struct {
    rig: rig_.Rig,
    form: Form,
    control: bool = false,
    glue: *const flash.Glue,
    down_fn: cuda.Function,
    up_fn: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    h: []u16 = &.{},
    pss: []f32 = &.{},
    scale: []f32 = &.{},
    dq_down: []f64 = &.{},
    dq_up: []f64 = &.{},
    ins: [9]u64 = @splat(0),
    o: [5]u64 = @splat(0),
    msg: [240]u8 = undefined,

    pub fn deinit(self: *Hc) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "h", "pss", "scale", "dq_down", "dq_up" }) |f| a.free(@field(self, f));
        self.rig.deinit();
    }

    pub fn kernel(self: *Hc) g.Kernel {
        return .{ .spec = spec(self.form, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Hc);

    fn bfs(self: *Hc, n: usize, r: std.Random, lo: f32, hi: f32) ![]u16 {
        const v = try self.rig.gpu.gpa.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16(lo + r.float(f32) * (hi - lo));
        return v;
    }

    /// One quantized layer n x k: the tiled device copies at ins[at..at + 3], the dequantized rows as float64.
    fn layer(self: *Hc, at: usize, n: usize, k: usize, hi_scale: f32, hi_bias: f32, r: std.Random) ![]f64 {
        const a = self.rig.gpu.gpa;
        const words = try a.alloc(u32, n * k / 8);
        defer a.free(words);
        for (words) |*w| w.* = r.int(u32);
        const sc = try self.bfs(n * k / 32, r, 0.001, hi_scale);
        defer a.free(sc);
        const bi = try self.bfs(n * k / 32, r, -hi_bias, hi_bias);
        defer a.free(bi);
        const dq = try a.alloc(f64, n * k);
        errdefer a.free(dq);
        for (0..n) |j| for (0..k) |i| {
            const q: f64 = @floatFromInt((words[j * (k / 8) + i / 8] >> @intCast(4 * (i % 8))) & 0xF);
            dq[j * k + i] = q * bf(sc[j * (k / 32) + i / 32]) + bf(bi[j * (k / 32) + i / 32]);
        };
        if (self.control and at == 6) words[0] ^= 0x0000_00f0; // after the float64 rows were made
        const tw = try flash.hc.tileWords(a, std.mem.sliceAsBytes(words), n, k / 8);
        defer a.free(tw);
        const ts = try flash.hc.transposeBf16(a, std.mem.sliceAsBytes(sc), n, k / 32);
        defer a.free(ts);
        const tb = try flash.hc.transposeBf16(a, std.mem.sliceAsBytes(bi), n, k / 32);
        defer a.free(tb);
        self.ins[at] = try self.rig.input(tw);
        self.ins[at + 1] = try self.rig.input(ts);
        self.ins[at + 2] = try self.rig.input(tb);
        return dq;
    }

    pub fn prepare(self: *Hc, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != dims or shape.k != wide or shape.rows == 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        inline for (.{ "h", "pss", "scale", "dq_down", "dq_up" }) |f| a.free(@field(self, f));
        self.h = &.{};
        self.pss = &.{};
        self.scale = &.{};
        self.dq_down = &.{};
        self.dq_up = &.{};
        self.shape = shape;
        const m = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x71776863);
        const r = prng.random();
        self.h = try self.bfs(m * wide, r, -1, 1);
        if (fill == .nonfinite) self.h[11] = c.toBf16(std.math.nan(f32));
        self.pss = try a.alloc(f32, m * nc * streams);
        for (0..m) |row| for (0..nc) |cc| for (0..streams) |s| {
            var t: f32 = 0;
            for (0..256) |i| {
                const v: f32 = @floatCast(bf(self.h[row * wide + s * dims + cc * 256 + i]));
                t += v * v;
            }
            self.pss[(row * nc + cc) * streams + s] = t;
        };
        self.scale = try a.alloc(f32, wide);
        for (self.scale) |*x| x.* = 0.5 + r.float(f32);
        self.ins[0] = try self.rig.input(std.mem.sliceAsBytes(self.h));
        self.ins[1] = try self.rig.input(std.mem.sliceAsBytes(self.pss));
        self.ins[2] = try self.rig.input(std.mem.sliceAsBytes(self.scale));
        self.dq_down = try self.layer(3, n_down, wide, 0.004, 0.01, r);
        self.dq_up = try self.layer(6, wide, low, 0.05, 0.1, r);
        for ([_]usize{ m * low * 2, m * (low / 32) * 4, m * streams * 2, m * dims * 2, m * (dims / 32) * 4 }, &self.o) |len, *p| p.* = try self.rig.output(len);
        return true;
    }

    fn sweepOf(m: usize, bm: usize, k: usize) usize {
        return @max(@as(usize, 1), @min((m + bm - 1) / bm, (12 << 20) / (bm * k * 2)));
    }

    pub fn run(self: *Hc) !void {
        const s = self.rig.s;
        const m = self.shape.rows;
        const o = self.o;
        const i = self.ins;
        switch (self.form) {
            .tiles => {
                const part = try self.rig.scratch(m * n_down * 4);
                const rinv = try self.rig.scratch(m * streams * 4);
                const up = try self.rig.scratch(m * wide * 2);
                var a: cuda.Args = .{};
                for ([_]u64{ i[0], 0, i[3], i[4], i[5], part }) |v| a.add(v);
                for ([_]usize{ m, n_down, wide, sk, wide, sweepOf(m, 32, wide) }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(i[1]);
                a.add(i[2]);
                a.add(eps);
                for ([_]usize{ dims, nc, streams }) |v| a.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.down_fn, .{ .grid = .{ .x = @intCast((m + 31) / 32 * ((n_down + 191) / 192)) }, .block = .{ .x = 256 }, .shared = tk.hcf_down_smem }, s, &a);
                try self.glue.hcReduceAct(s, part, o[0], o[1], o[2], 1, m, streams, low, n_down, 1);
                var b: cuda.Args = .{};
                for ([_]u64{ o[0], o[1], i[6], i[7], i[8], up }) |v| b.add(v);
                for ([_]usize{ m, wide, low, 1, low, sweepOf(m, 64, low) }) |v| b.add(@as(c_int, @intCast(v)));
                b.add(@as(u64, 0));
                b.add(@as(u64, 0));
                b.add(@as(f32, 0));
                for ([_]usize{ dims, nc, streams }) |v| b.add(@as(c_int, @intCast(v)));
                try cuda.launch.launch(self.up_fn, .{ .grid = .{ .x = @intCast((m + 63) / 64 * (wide / 128)) }, .block = .{ .x = 256 }, .shared = tk.hcf_up_smem }, s, &b);
                try self.glue.hcRinv(s, i[1], rinv, eps, m, dims, nc, streams);
                try self.glue.hcMixH(s, up, i[0], rinv, i[2], o[3], o[4], m, dims, streams);
            },
            .decode => {
                const normed = try self.rig.scratch(m * wide * 2);
                const dn = try self.rig.scratch(m * (low + 4) * 2);
                const part = try self.rig.scratch(sk * m * n_down * 4);
                try self.glue.hcDown(s, i[0], i[1], i[2], normed, i[3], i[4], i[5], dn, part, eps, m, n_down, wide, dims, nc, streams, sk);
                try self.glue.hcReduceAct(s, part, o[0], o[1], o[2], sk, m, streams, low, n_down, 1);
                try self.glue.hcUpmix(s, o[0], o[1], i[6], i[7], i[8], normed, o[3], o[4], m, wide, low, dims, streams);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Hc) usize {
        const m = self.shape.rows;
        return m * low * 2 + m * (low / 32) * 4 + m * streams * 2 + m * dims * 2 + m * (dims / 32) * 4;
    }

    pub fn reference(_: *Hc, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }

    /// The first stage output that leaves float64's class or bound, else null.
    pub fn check(self: *Hc, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const m = self.shape.rows;
        const act = try a.dupe(u8, try self.rig.bytes(self.o[0], m * low * 2));
        defer a.free(act);
        const xs = try a.dupe(u8, try self.rig.bytes(self.o[1], m * (low / 32) * 4));
        defer a.free(xs);
        const inj = try a.dupe(u8, try self.rig.bytes(self.o[2], m * streams * 2));
        defer a.free(inj);
        const mixed = try a.dupe(u8, try self.rig.bytes(self.o[3], m * dims * 2));
        defer a.free(mixed);
        const xsm = try a.dupe(u8, try self.rig.bytes(self.o[4], m * (dims / 32) * 4));
        defer a.free(xsm);
        const normed = try a.alloc(u16, wide);
        defer a.free(normed);
        const down = try a.alloc(f64, n_down);
        defer a.free(down);
        const mag = try a.alloc(f64, n_down);
        defer a.free(mag);
        var planted: usize = 0;
        for (0..m) |row| {
            try flash.hc.normed(self.h[row * wide ..][0..wide], self.pss[row * nc * streams ..][0 .. nc * streams], self.scale, dims, streams, eps, normed);
            for (0..n_down) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                const w = self.dq_down[j * wide ..][0..wide];
                for (normed, w) |x, wv| {
                    const t = bf(x) * wv;
                    sum += t;
                    abs += @abs(t);
                }
                down[j] = sum;
                mag[j] = abs;
            }
            if (try self.stageDown(row, down, mag, act, xs, inj, at, &planted)) |why| return why;
            if (try self.stageUp(row, normed, act, xs, mixed, xsm, at, &planted)) |why| return why;
        }
        if (at == .nonfinite and (planted == 0 or planted == m * (low + streams + dims))) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} reference outputs non-finite: no proof", .{planted});
        return null;
    }

    fn miss(self: *Hc, what: []const u8, row: usize, i: usize, got: f64, want: f64, tol: f64) ![]const u8 {
        return std.fmt.bufPrint(&self.msg, "{s} row {d} element {d} is {e:.6} where float64 gives {e:.6} (bound {e:.3})", .{ what, row, i, got, want, tol });
    }

    /// act, inj and xs of one row: the reduce's rounding points around float64's down sums.
    fn stageDown(self: *Hc, row: usize, down: []const f64, mag: []const f64, act: []const u8, xs: []const u8, inj: []const u8, at: g.Gate, planted: *usize) !?[]const u8 {
        const slices: f64 = if (self.form == .decode) sk else 1;
        for (0..low + streams) |j| {
            const y = down[j] / streams;
            const e = (@as(f64, wide) / 16 + slices) * f32u * mag[j] / streams; // the fp32 sums' allowance
            const sig = sigmoid(y);
            const is_act = j < low;
            const want = if (is_act) y * sig else 2 * sig;
            const slope = if (is_act) @abs(sig * (1 + y * (1 - sig))) else 2 * sig * (1 - sig);
            const got = if (is_act) c.read(.bf16, act, row * low + j) else c.read(.bf16, inj, row * streams + j - low);
            if (c.classOf(got) != c.classOf(want)) return try self.miss(if (is_act) "act class" else "inj class", row, j, got, want, 0);
            if (!std.math.isFinite(want)) {
                planted.* += 1;
                continue;
            }
            const tol = slope * (u * @abs(y) + e) + 2 * u * @abs(want) + 8 * f32u * @abs(want) + 0x1p-133;
            if (at != .nonfinite and !(@abs(got - want) <= tol)) return try self.miss(if (is_act) "act" else "inj", row, j, got, want, tol);
        }
        for (0..low / 32) |gi| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..32) |t| {
                const v = c.read(.bf16, act, row * low + gi * 32 + t);
                sum += v;
                abs += @abs(v);
            }
            const got = c.read(.f32, xs, row * (low / 32) + gi);
            if (c.classOf(got) != c.classOf(sum)) return try self.miss("xs class", row, gi, got, sum, 0);
            if (std.math.isFinite(sum) and at != .nonfinite and !(@abs(got - sum) <= 5 * f32u * abs + 0x1p-149)) return try self.miss("xs", row, gi, got, sum, 5 * f32u * abs);
        }
        return null;
    }

    /// mixed and xsm of one row, from the kernel's own act and xs (teacher forcing): float64's up sums and gates.
    fn stageUp(self: *Hc, row: usize, normed: []const u16, act: []const u8, xs: []const u8, mixed: []const u8, xsm: []const u8, at: g.Gate, planted: *usize) !?[]const u8 {
        _ = xs;
        var a_row: [low]f64 = undefined;
        for (&a_row, 0..) |*x, i| x.* = c.read(.bf16, act, row * low + i);
        for (0..dims) |col| {
            var total: f64 = 0;
            var tol: f64 = 0;
            for (0..streams) |s| {
                const w = self.dq_up[(s * dims + col) * low ..][0..low];
                var up: f64 = 0;
                var abs: f64 = 0;
                for (a_row, w) |x, wv| {
                    up += x * wv;
                    abs += @abs(x * wv);
                }
                const sig = sigmoid(up);
                const n = bf(normed[s * dims + col]);
                total += sig * n;
                const e = (@as(f64, low) / 16 + 8) * f32u * abs;
                tol += sig * (1 - sig) * @abs(n) * (u * @abs(up) + e) + 2 * u * sig * @abs(n);
            }
            const want = total / streams;
            const got = c.read(.bf16, mixed, row * dims + col);
            if (c.classOf(got) != c.classOf(want)) return try self.miss("mixed class", row, col, got, want, 0);
            if (!std.math.isFinite(want)) {
                planted.* += 1;
                continue;
            }
            const bound = tol / streams + u * @abs(want) + 8 * f32u * @abs(want) + 0x1p-133;
            if (at != .nonfinite and !(@abs(got - want) <= bound)) return try self.miss("mixed", row, col, got, want, bound);
        }
        for (0..dims / 32) |gi| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..32) |t| {
                const v = c.read(.bf16, mixed, row * dims + gi * 32 + t);
                sum += v;
                abs += @abs(v);
            }
            const got = c.read(.f32, xsm, row * (dims / 32) + gi);
            if (c.classOf(got) != c.classOf(sum)) return try self.miss("xsm class", row, gi, got, sum, 0);
            if (std.math.isFinite(sum) and at != .nonfinite and !(@abs(got - sum) <= 5 * f32u * abs + 0x1p-149)) return try self.miss("xsm", row, gi, got, sum, 5 * f32u * abs);
        }
        return null;
    }
};
