//! Flash Next's block-sparse attention (fn_attn_chunks, fn_attn_merge) at Q24/KV4, a provisional Qwen4 dense shape.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const qh = 24;
const kvh = 4;
const group = qh / kvh;
const hd = 256;
const chunk = 512;
const scale: f32 = 0.0625;
/// cuda_prompt_wt.zig's block indexer: blocks of 4 keys, the best 512, 5 sparse or 3 dense chunks, the list width.
const ratio = 4;
const top = 512;
const nch = 5;
const dense_nch = 3;
const idw = top * ratio + ratio;
/// Keys a dense row reads at most (cuda_rows.zig), and the sparse row's selected keys plus its tail.
const dense_keys = dense_nch * chunk;
const selected = top * ratio;

pub const Form = struct { sparse: bool, control: bool = false };
pub const forms = [_]Form{ .{ .sparse = false }, .{ .sparse = true }, .{ .sparse = true, .control = true } };

fn keyChain(s: c.Shape) usize {
    return s.k + s.rows + nch;
}

pub fn spec(f: Form) c.Spec {
    const name = if (f.control) "control qwen4 QSA attention dropping each row's last chunk" else if (f.sparse) "cuda qwen4 synthetic QSA attention, sparse rows, Q24/KV4 (fn_attn_chunks, fn_attn_merge)" else "cuda qwen4 synthetic dense attention rows, Q24/KV4 (fn_attn_chunks, fn_attn_merge)";
    // rows = query rows, n = query heads, k = the first row's visible keys
    const served: []const c.Shape = if (f.sparse) &.{ .{ .rows = 1, .n = qh, .k = 5000 }, .{ .rows = 8, .n = qh, .k = 3000 }, .{ .rows = 17, .n = qh, .k = 2100 } } else &.{ .{ .rows = 1, .n = qh, .k = 1400 }, .{ .rows = 8, .n = qh, .k = 1000 }, .{ .rows = 17, .n = qh, .k = 1400 } };
    return .{ .name = name, .accum = .f32, .depth = keyChain, .ops = 3, .out = .bf16, .tiles = .{ .{ .size = 16 }, .{}, .{ .size = chunk } }, .served = served, .sums = .{ .why = attn.why } };
}

pub const Qsa = struct {
    rig: rig_.Rig,
    form: Form,
    glue: *const flash.Glue,
    r: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: std.ArrayList(u16) = .empty,
    kc: std.ArrayList(u16) = .empty,
    vc: std.ArrayList(u16) = .empty,
    ids: std.ArrayList(i32) = .empty,
    nk: std.ArrayList(i32) = .empty,
    spr: std.ArrayList(i32) = .empty,
    dev: [6]u64 = @splat(0),
    part: [3]u64 = @splat(0),
    out: u64 = 0,
    chunks: usize = 0,
    pos0: usize = 0,
    spr_dev: u64 = 0,

    pub fn deinit(self: *Qsa) void {
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        inline for (.{ "q", "kc", "vc", "ids", "nk", "spr" }) |f| @field(self, f).deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Qsa) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }

    const vt = g.hooks(Qsa);

    fn visible(self: *const Qsa, row: usize) usize {
        return self.pos0 + row + 1;
    }

    fn cap(self: *const Qsa) usize {
        return self.pos0 + self.shape.rows;
    }

    pub fn prepare(self: *Qsa, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.n != qh or shape.rows == 0 or shape.k == 0) return false;
        if (!self.form.sparse and shape.k + shape.rows - 1 > dense_keys) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x71736134);
        const rng = prng.random();
        self.shape = shape;
        self.pos0 = shape.k - 1;
        const rows = shape.rows;
        const n_cap = self.cap();
        try self.q.resize(a, rows * qh * hd);
        try self.kc.resize(a, n_cap * kvh * hd);
        try self.vc.resize(a, n_cap * kvh * hd);
        for (self.q.items) |*x| x.* = attn.normal(rng);
        for (self.kc.items) |*x| x.* = attn.normal(rng);
        for (self.vc.items) |*x| x.* = attn.normal(rng);
        try self.ids.resize(a, rows * idw);
        try self.nk.resize(a, rows);
        try self.spr.resize(a, rows);
        @memset(self.ids.items, 0);
        self.chunks = 1;
        for (0..rows) |row| {
            const vis = self.visible(row);
            const n = if (self.form.sparse) @min(vis, selected + row % ratio) else vis;
            self.nk.items[row] = @intCast(n);
            self.spr.items[row] = @intFromBool(self.form.sparse);
            if (self.form.sparse) self.select(row, n, fill == .stress, rng);
            self.chunks = @max(self.chunks, (n + chunk - 1) / chunk);
        }
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, hd, @max(nch * chunk, shape.k + rows));
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q.items));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.kc.items));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.vc.items));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(&[4]i32{ @intCast(self.pos0), 0, 0, 0 }));
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(self.ids.items));
        self.dev[5] = try self.rig.input(std.mem.sliceAsBytes(self.nk.items));
        const flags = try self.rig.input(std.mem.sliceAsBytes(self.spr.items));
        const slots = rows * nch * qh;
        self.part[0] = try self.rig.scratch(slots * hd * 4);
        self.part[1] = try self.rig.scratch(slots * 4);
        self.part[2] = try self.rig.scratch(slots * 4);
        self.out = try self.rig.output(rows * qh * hd * 2);
        self.spr_dev = flags;
        return true;
    }

    /// Row `row`'s `n` selected keys, ascending: its last four visible keys, then others (stress: the first n).
    fn select(self: *Qsa, row: usize, n: usize, first: bool, rng: std.Random) void {
        const vis = self.visible(row);
        const list = self.ids.items[row * idw ..][0..n];
        if (first) {
            for (list, 0..) |*x, i| x.* = @intCast(i);
            return;
        }
        const tail = @min(n, 4);
        const rest = n - tail;
        var at: usize = 0;
        // selection sampling: each earlier key is taken with the probability that keeps `rest` of them
        var left = vis - tail;
        var need = rest;
        for (0..vis - tail) |pos| {
            if (need == 0) break;
            if (rng.uintLessThan(usize, left) < need) {
                list[at] = @intCast(pos);
                at += 1;
                need -= 1;
            }
            left -= 1;
        }
        for (0..tail) |i| list[at + i] = @intCast(vis - tail + i);
    }

    fn put(self: *Qsa, value: bool, h: usize, pos: usize, e: usize, v: f32) void {
        const x = if (value) self.vc.items else self.kc.items;
        x[(pos * kvh + h) * hd + e] = c.toBf16(v);
    }

    /// NaN in a query, then +inf and -inf values (alone, and both in one column) at row 0's last keys.
    fn plant(self: *Qsa) void {
        self.q.items[3] = c.toBf16(std.math.nan(f32)); // row 0, head 0
        const p = self.pos0;
        self.put(true, 0, p, 7, std.math.inf(f32));
        if (p >= 3) {
            self.put(true, 1, p - 1, 11, std.math.inf(f32));
            self.put(true, 1, p - 2, 11, -std.math.inf(f32));
            self.put(true, 2, p - 1, 5, -std.math.inf(f32));
        }
    }

    /// Gate 4: every key of a head alike (each p is 1), values mirrored per column over the first keys, then 0.
    fn stress(self: *Qsa, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const p = @min(self.shape.k, 2048);
        const col = try a.alloc(f32, p);
        defer a.free(col);
        const cp = self.cap();
        for (0..kvh) |h| {
            for (1..cp) |j| @memcpy(self.kc.items[(j * kvh + h) * hd ..][0..hd], self.kc.items[h * hd ..][0..hd]);
            for (0..hd) |e| {
                attn.mirrored(rng, col);
                for (0..cp) |j| self.vc.items[(j * kvh + h) * hd + e] = if (j < p) c.toBf16(col[j]) else 0;
            }
        }
    }

    pub fn run(self: *Qsa) !void {
        const rows = self.shape.rows;
        const z = if (self.form.control) @max(1, self.chunks - 1) else self.chunks;
        const qs = self.form.sparse;
        const d = self.dev;
        try self.glue.attnChunks(self.rig.s, d[0], d[1], d[2], d[3], self.part[0], self.part[1], self.part[2], d[4], d[5], self.spr_dev, rows, kvh, z, qs);
        try self.glue.attnMerge(self.rig.s, self.part[0], self.part[1], self.part[2], d[3], self.out, d[5], self.spr_dev, rows, kvh, qs);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Qsa) usize {
        return self.shape.rows * qh * hd;
    }

    /// Output head i (row i / qh, head i % qh): the keys it reads, in order, from its KV head.
    pub fn load(self: *Qsa, i: usize, h: *attn.Head) !void {
        const row = i / qh;
        const head = i % qh;
        const kh = head / group;
        for (h.q, 0..) |*x, e| x.* = c.fromBf16(self.q.items[(row * qh + head) * hd + e]);
        const n: usize = @intCast(self.nk.items[row]);
        for (0..n) |t| {
            const pos: usize = if (self.form.sparse) @intCast(self.ids.items[row * idw + t]) else t;
            for (0..hd) |e| {
                h.k[t * hd + e] = c.fromBf16(self.kc.items[(pos * kvh + kh) * hd + e]);
                h.v[t * hd + e] = c.fromBf16(self.vc.items[(pos * kvh + kh) * hd + e]);
            }
        }
        h.n = n;
    }

    fn heads(self: *Qsa) usize {
        return self.shape.rows * qh;
    }

    pub fn check(self: *Qsa, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale, try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *Qsa, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale, want, mag);
    }

    pub fn simulate(self: *Qsa, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale, acc, order, out);
    }
};
