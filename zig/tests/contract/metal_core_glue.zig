//! The core row glue (core/row_glue.metal) and sigmoid MoE route (core/moe_sigmoid.metal) under contract gates 1-4.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const nref = contract.norms_ref;
const glue = tf.row_glue;
const moe = tf.moe_sigmoid;
const base = "core glue ";

pub const Kind = enum { embed, rms, add_rms, qk, select, plan, combine };
pub const Variant = struct { kind: Kind, ctl: u8 = 0, score: bool = false };

const eps: f32 = 1e-6;
/// Kolibri-1's widths: hidden 2560 rows, 48 + 2 x 4 heads of 128, 384 or 8 experts top-6 and the shared one.
const row_shapes = [_]c.Shape{ .{ .rows = 16, .n = 1, .k = 2560 }, .{ .rows = 1, .n = 1, .k = 2560 }, .{ .rows = 5, .n = 1, .k = 1536 }, .{ .rows = 3, .n = 1, .k = 256 } };
const qk_shapes = [_]c.Shape{ .{ .rows = 4, .n = 56, .k = 128 }, .{ .rows = 1, .n = 56, .k = 128 }, .{ .rows = 3, .n = 6, .k = 64 } };
const route_shapes = [_]c.Shape{ .{ .rows = 16, .n = 384, .k = 6 }, .{ .rows = 1, .n = 384, .k = 6 }, .{ .rows = 7, .n = 8, .k = 6 }, .{ .rows = 3, .n = 128, .k = 8 } };
const combine_shapes = [_]c.Shape{ .{ .rows = 16, .n = 2560, .k = 7 }, .{ .rows = 1, .n = 2560, .k = 7 }, .{ .rows = 3, .n = 1536, .k = 9 } };

/// A row sum's chain: a thread's k / 256 terms, the 32-lane tree, the 8 simdgroups' partials in order.
fn rowDepth(s: c.Shape) usize {
    return s.k / 256 + 1 + 5 + 8;
}
fn headDepth(s: c.Shape) usize {
    return s.k / 32 + 5;
}
fn slotDepth(s: c.Shape) usize {
    return s.k;
}
fn none(_: c.Shape) usize {
    return 0;
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    const raw: c.Nan = .{ .stated = "exact: picks, groups and copies equal the host rule's bytes" };
    return switch (v.kind) {
        .embed, .select, .plan => .{ .name = name, .accum = .none, .depth = none, .out = .raw, .nan = raw, .served = if (v.kind == .embed) &row_shapes else &route_shapes },
        .rms => .{ .name = name, .accum = .f32, .depth = rowDepth, .ops = 3, .out = .bf16, .served = &row_shapes, .tiles = .{ .{}, .{}, .{ .size = 256, .step = 256, .min = 256 } } },
        .add_rms => .{ .name = name, .accum = .f32, .depth = rowDepth, .ops = 6, .out = .bf16, .served = &row_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "the sandwich's two norms: its bound is norms_ref's, not a sums proof", .exact = false } },
        .qk => .{ .name = name, .accum = .f32, .depth = headDepth, .ops = 6, .out = .bf16, .served = &qk_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = "a head's 128 squares: too short to make bf16 sums miss 100x", .exact = false } },
        .combine => .{ .name = name, .accum = .f32, .depth = slotDepth, .out = .bf16, .served = &combine_shapes },
    };
}

pub const Glue = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    gp: glue.Pipes,
    mp: moe.Pipes,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    rows: ?contract.rows.Rows = null, // rms and add_rms: x and w1 (add_rms: w2 = w1 reversed, res from `aux`)
    aux: []f64 = &.{}, // add_rms: the residual in; qk: positions; combine: weights then y; select: logits then bias
    host: []u8 = &.{}, // the inputs as bound, for the exact checks
    refs: [10]rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Glue {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .gp = try glue.Pipes.init(a, device, if (v.kind == .select or v.kind == .plan or v.kind == .combine) 0 else v.ctl), .mp = try moe.Pipes.init(a, device, if (v.kind == .select or v.kind == .plan or v.kind == .combine) v.ctl else 0) };
    }

    pub fn deinit(p: *Glue) void {
        p.drop();
        p.gp.deinit();
        p.mp.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Glue) void {
        if (p.rows) |r| r.deinit();
        p.rows = null;
        p.a.free(p.aux);
        p.aux = &.{};
        p.a.free(p.host);
        p.host = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Glue) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = if (p.v.ctl != 0) .shapes else null };
    }

    const vt = g.hooks(Glue);

    fn r(x: rig_.Ref) glue.Ref {
        return .{ .buf = x.buf, .off = x.off };
    }

    pub fn route(p: *const Glue) moe.Route {
        const ne = p.shape.n;
        return .{ .experts = ne, .topk = p.shape.k, .shared = true, .score = p.v.score, .normalize = p.v.score, .scale = if (p.v.score) 2.5 else 1, .groups = if (p.v.score) 8 else 1, .keep = if (p.v.score) 4 else 1 };
    }

    pub fn prepare(p: *Glue, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16) return false;
        p.drop();
        p.shape = shape;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x9e37);
        const rng = prng.random();
        const k = shape.k;
        switch (p.v.kind) {
            .embed => {
                const vocab = 37;
                const table = try p.a.alloc(u16, vocab * k);
                defer p.a.free(table);
                for (table) |*t| t.* = rng.int(u16); // every bit pattern, NaNs and infinities among them
                const ids = try p.a.alloc(u32, shape.rows);
                defer p.a.free(ids);
                for (ids) |*i| i.* = rng.uintLessThan(u32, vocab);
                p.host = try std.mem.concat(p.a, u8, &.{ std.mem.sliceAsBytes(ids), std.mem.sliceAsBytes(table) });
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(ids));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(table));
                p.refs[2] = try p.rig.output(shape.rows * k * 4);
            },
            .rms, .add_rms => {
                if (k % 256 != 0) return false;
                var rows = try contract.rows.Rows.init(p.a, .rms, .{ .rows = shape.rows, .n = 1, .k = k });
                rows.fill(fill, seed);
                if (fill == .random) for (rows.x[0..k]) |*x| {
                    x.* = c.toBf16(c.fromBf16(x.*) * 1e-4); // eps sets this row's scale: a norm without it shows
                };
                p.rows = rows;
                const x32 = try p.a.alloc(f32, rows.x.len);
                defer p.a.free(x32);
                for (x32, rows.x) |*o, b| o.* = @floatCast(c.fromBf16(b));
                const w2 = try p.a.alloc(u16, k);
                defer p.a.free(w2);
                for (w2, 0..) |*w, i| w.* = rows.w[k - 1 - i];
                if (p.v.kind == .rms) {
                    p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(x32));
                    p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(rows.w));
                    p.refs[2] = try p.rig.output(shape.rows * k * 2);
                } else {
                    p.aux = try p.a.alloc(f64, rows.x.len);
                    const res = try p.a.alloc(f32, rows.x.len);
                    defer p.a.free(res);
                    for (res, p.aux) |*o, *h| {
                        o.* = (rng.float(f32) * 2 - 1) * 8;
                        h.* = o.*;
                    }
                    p.host = try p.a.dupe(u8, std.mem.sliceAsBytes(res));
                    p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(rows.x));
                    p.refs[1] = try p.rig.scratch(res.len * 4); // in place: run() puts the residual back first
                    p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(rows.w));
                    p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(w2));
                    p.refs[4] = try p.rig.output(shape.rows * k * 2);
                }
            },
            .qk => try p.prepareQk(shape, fill, rng),
            .select, .plan => try p.prepareRoute(shape, fill, rng),
            .combine => {
                const slots = k;
                const n = shape.n;
                p.aux = try p.a.alloc(f64, shape.rows * slots * (1 + n));
                const w = try p.a.alloc(f32, shape.rows * slots);
                defer p.a.free(w);
                const y = try p.a.alloc(f32, shape.rows * slots * n);
                defer p.a.free(y);
                for (w, 0..) |*x, i| x.* = if (fill == .stress) 1 else if (i % slots == slots - 1) 1 else rng.float(f32);
                for (y, 0..) |*x, i| x.* = if (fill == .stress) @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 64))) - 32 + @as(i32, @intFromBool(i % 7 == 0))) else (rng.float(f32) * 2 - 1) * 4;
                if (fill == .nonfinite) for ([_]usize{ 0, n + 1, 2 * n + 2 }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |at, v| {
                    if (at < y.len) y[at] = v;
                };
                for (w, 0..) |x, i| p.aux[i] = x;
                for (y, 0..) |x, i| p.aux[w.len + i] = x;
                p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(y));
                p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(w));
                p.refs[2] = try p.rig.output(shape.rows * n * 2);
            },
        }
        return true;
    }

    fn prepareQk(p: *Glue, shape: c.Shape, fill: g.Fill, rng: std.Random) !void {
        const d = shape.k;
        const hk: usize = 4;
        const h = shape.n - 2 * hk;
        const width = shape.n * d;
        const ring = shape.rows; // every slot written, the rows' positions across the wrap
        const x = try p.a.alloc(u16, shape.rows * width);
        defer p.a.free(x);
        for (x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        if (fill == .nonfinite) for ([_]usize{ 0, d + 5, width + 2 }, [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |at, v| {
            if (at < x.len) x[at] = c.toBf16(v);
        };
        const norms = try p.a.alloc(u16, 2 * d);
        defer p.a.free(norms);
        for (norms) |*v| v.* = c.toBf16(0.5 + rng.float(f32));
        const ctx: usize = 64;
        const cos = try p.a.alloc(f32, ctx * d);
        defer p.a.free(cos);
        const sin = try p.a.alloc(f32, ctx * d);
        defer p.a.free(sin);
        for (0..ctx) |pos| for (0..d / 2) |j| {
            const ang = @as(f64, @floatFromInt(pos)) / std.math.pow(f64, 1e4, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(d)));
            for ([_]usize{ j, j + d / 2 }) |col| {
                cos[pos * d + col] = @floatCast(@cos(ang));
                sin[pos * d + col] = @floatCast(@sin(ang));
            }
        };
        const pos = try p.a.alloc(u32, shape.rows);
        defer p.a.free(pos);
        for (pos, 0..) |*v, i| v.* = @intCast(22 + i);
        _ = h;
        p.host = try std.mem.concat(p.a, u8, &.{ std.mem.sliceAsBytes(x), std.mem.sliceAsBytes(norms), std.mem.sliceAsBytes(cos), std.mem.sliceAsBytes(sin), std.mem.sliceAsBytes(pos) });
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(norms[0..d]));
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(norms[d..]));
        p.refs[3] = try p.rig.input(std.mem.sliceAsBytes(cos));
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(sin));
        p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(pos));
        p.refs[6] = try p.rig.output(shape.rows * (shape.n - 2 * hk) * d * 2);
        p.refs[7] = try p.rig.output(ring * hk * d * 2);
        p.refs[8] = try p.rig.output(ring * hk * d * 2);
    }

    fn prepareRoute(p: *Glue, shape: c.Shape, fill: g.Fill, rng: std.Random) !void {
        const ne = shape.n;
        const logits = try p.a.alloc(f32, shape.rows * ne + ne);
        defer p.a.free(logits);
        for (logits[0 .. shape.rows * ne], 0..) |*v, i| v.* = if (fill == .stress or i % 5 == 0) @floatFromInt(@as(i32, @intCast(i % 3))) else rng.float(f32) * 8 - 4; // exact ties
        for (logits[shape.rows * ne ..]) |*v| v.* = if (rng.uintLessThan(u32, 4) == 0) 0 else rng.float(f32) * 0.5;
        if (fill == .nonfinite) logits[1] = std.math.inf(f32);
        p.host = try p.a.dupe(u8, std.mem.sliceAsBytes(logits));
        const rt = p.route();
        const pairs = shape.rows * rt.slots();
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(logits[0 .. shape.rows * ne]));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(logits[shape.rows * ne ..]));
        p.refs[2] = try p.rig.output(pairs * 4); // picks
        p.refs[3] = try p.rig.output(pairs * 4); // weights
        if (p.v.kind == .plan) {
            p.refs[4] = try p.rig.scratch((ne + 1) * 4);
            p.refs[5] = try p.rig.output(pairs * 4); // members
            p.refs[6] = try p.rig.output(tf.block_rows.maxItems(pairs, ne + 1) * 12);
            p.refs[7] = try p.rig.output(4);
        }
    }

    pub fn run(p: *Glue) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        const ga: glue.Args = .{ .rows = @intCast(s.rows), .width = @intCast(s.k), .eps = eps };
        switch (p.v.kind) {
            .embed => p.gp.embed(e, r(p.refs[0]), r(p.refs[1]), r(p.refs[2]), ga),
            .rms => p.gp.rmsNorm(e, r(p.refs[0]), r(p.refs[1]), r(p.refs[2]), ga),
            .add_rms => {
                @memcpy(p.rig.bytes(p.refs[1], p.host.len), p.host);
                p.gp.addRms(e, r(p.refs[0]), r(p.refs[1]), r(p.refs[2]), r(p.refs[3]), r(p.refs[4]), ga);
            },
            .qk => try p.gp.qk(e, .{ .qkv = r(p.refs[0]), .q_norm = r(p.refs[1]), .k_norm = r(p.refs[2]), .cos = r(p.refs[3]), .sin = r(p.refs[4]), .pos = r(p.refs[5]), .q = r(p.refs[6]), .k_cache = r(p.refs[7]), .v_cache = r(p.refs[8]) }, .{ .rows = @intCast(s.rows), .heads = @intCast(s.n - 8), .kv_heads = 4, .dim = @intCast(s.k), .rope = 1, .ring = @intCast(s.rows), .eps = eps }),
            .select, .plan => {
                const rt = p.route();
                const a = moe.argsOf(rt, s.rows, 1);
                e.setPipeline(p.mp.select);
                for ([_]rig_.Ref{ p.refs[0], p.refs[1] }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
                e.setValue(a, 2);
                e.setBuffer(p.refs[2].buf, p.refs[2].off, 3);
                e.setBuffer(p.refs[3].buf, p.refs[3].off, 4);
                e.dispatchThreads(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(s.rows, 1, 1));
                if (p.v.kind == .plan) {
                    e.barrier();
                    e.setPipeline(p.mp.plan);
                    e.setBuffer(p.refs[2].buf, p.refs[2].off, 0);
                    e.setValue(a, 1);
                    for ([_]rig_.Ref{ p.refs[4], p.refs[5], p.refs[6], p.refs[7] }, 2..) |b, i| e.setBuffer(b.buf, b.off, i);
                    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
                }
            },
            .combine => {
                const a = moe.argsOf(.{ .experts = 1, .topk = s.k - 1, .shared = true }, s.rows, s.n);
                e.setPipeline(p.mp.combine);
                e.setBuffer(p.refs[0].buf, p.refs[0].off, 0);
                e.setBuffer(p.refs[1].buf, p.refs[1].off, 1);
                e.setValue(a, 2);
                e.setBuffer(p.refs[2].buf, p.refs[2].off, 3);
                e.dispatchThreads(mtl.Size.of(s.n, s.rows, 1), mtl.Size.of(256, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Glue) usize {
        const s = p.shape;
        return switch (p.v.kind) {
            .embed => s.rows * s.k * 4,
            .rms, .add_rms => s.rows * s.k,
            .qk => (s.rows * (s.n - 8) + 2 * s.rows * 4) * s.k,
            .select => 2 * s.rows * (s.k + 1) * 4,
            .plan => 3 * s.rows * (s.k + 1) * 4 + tf.block_rows.maxItems(s.rows * (s.k + 1), s.n + 1) * 12 + 4,
            .combine => s.rows * s.n,
        };
    }

    pub const glue_ref = @import("metal_core_glue_ref.zig");

    pub fn reference(p: *Glue, want: []f64, mag: []f64) !void {
        return glue_ref.reference(p, want, mag);
    }

    pub fn bounds(p: *Glue, want: []const f64, mag: []const f64, allow: []f64) !void {
        return glue_ref.bounds(p, want, mag, allow);
    }

    pub fn simulate(p: *Glue, acc: c.Accum, order: g.Order, out: []f64) !void {
        return glue_ref.simulate(p, acc, order, out);
    }

    pub fn check(p: *Glue, at: g.Gate) !?[]const u8 {
        return glue_ref.check(p, at);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Glue) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .kind = .embed }, .name = base ++ "tf_glue_embed" },
        .{ .v = .{ .kind = .rms }, .name = base ++ "tf_glue_rms" },
        .{ .v = .{ .kind = .add_rms }, .name = base ++ "tf_glue_add_rms" },
        .{ .v = .{ .kind = .qk }, .name = base ++ "tf_glue_qk" },
        .{ .v = .{ .kind = .select }, .name = base ++ "tf_moe_select" },
        .{ .v = .{ .kind = .select, .score = true }, .name = base ++ "tf_moe_select score groups normalised" },
        .{ .v = .{ .kind = .plan }, .name = base ++ "tf_moe_plan" },
        .{ .v = .{ .kind = .combine }, .name = base ++ "tf_moe_combine" },
        .{ .v = .{ .kind = .qk, .ctl = 1 }, .name = base ++ "control: RoPE halves swapped" },
        .{ .v = .{ .kind = .rms, .ctl = 2 }, .name = base ++ "control: norm without eps" },
        .{ .v = .{ .kind = .qk, .ctl = 3 }, .name = base ++ "control: cache slot off by one" },
        .{ .v = .{ .kind = .select, .ctl = 1 }, .name = base ++ "control: ties to the higher id" },
        .{ .v = .{ .kind = .select, .ctl = 2 }, .name = base ++ "control: normalising flipped" },
        .{ .v = .{ .kind = .combine, .ctl = 3 }, .name = base ++ "control: the shared slot skipped" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Glue);
            errdefer a.destroy(p);
            p.* = try Glue.init(a, device, queue, entry.v, entry.name);
            errdefer p.deinit();
            try contract.register(a, &s.probes, kernels, p, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
