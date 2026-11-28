//! Kolibri-1's layer graph on Metal, one stream: core blocks only (row glue, block products, attention, MoE).
const std = @import("std");
const mtl = @import("metal");
const blocks = @import("metal_blocks");
const Config = @import("config.zig").Config;
const wts = @import("metal_weights.zig");

const Ref = blocks.block_rows.Ref;
const glue = blocks.row_glue;
const br = blocks.block_rows;
const moe = blocks.moe_sigmoid;
const attn = blocks.shared_attention;

/// Keys a sliding layer's ring holds: a call's rows written before its first row reads back its window, whole 64s.
pub fn ringSize(rows: usize, window: usize) usize {
    return (rows + window + 63) / 64 * 64;
}

/// Residuals after each layer's attention half and MoE half, fp32 [layers][2][rows][D], when a forward asks.
pub const Taps = struct { buf: mtl.Buffer, rows: usize };

pub const Model = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    w: *const wts.Weights,
    c: Config,
    context: usize,
    max_rows: usize,
    ring: usize,
    glue: glue.Pipes,
    rows: br.Pipes, // fp32 out: projections into bf16 scratch go through `rows16`
    rows16: br.Pipes,
    moe: moe.Pipes,
    att: attn.Attention,
    route: moe.Route,
    k: []mtl.Buffer, // per layer: sliding [ring][HK][Dh], full [context][HK][Dh], bf16
    v: []mtl.Buffer,
    cos: mtl.Buffer,
    sin: mtl.Buffer,
    s: Scratch,
    p: usize = 0, // positions the caches hold

    const Scratch = struct { ids: mtl.Buffer, pos: mtl.Buffer, res: mtl.Buffer, x: mtl.Buffer, qkv: mtl.Buffer, q: mtl.Buffer, att: mtl.Buffer, o: mtl.Buffer, mo: mtl.Buffer, mask: mtl.Buffer, logits: mtl.Buffer, moe: [9]mtl.Buffer };

    fn buffer(m: *Model, bytes: usize) !mtl.Buffer {
        return m.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
    }

    /// Caches for `context` positions, RoPE tables and scratch for calls of up to `max_rows` rows.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, w: *const wts.Weights, context: usize, max_rows: usize, ctl: Ctl) !Model {
        const c = w.config;
        var m: Model = .{ .gpa = gpa, .device = device, .queue = try device.queue(), .w = w, .c = c, .context = context, .max_rows = max_rows, .ring = ringSize(max_rows, c.window), .glue = undefined, .rows = undefined, .rows16 = undefined, .moe = undefined, .att = undefined, .route = .{ .experts = c.experts, .topk = c.top_k, .shared = true }, .k = &.{}, .v = &.{}, .cos = undefined, .sin = undefined, .s = undefined };
        try m.route.check();
        m.glue = try glue.Pipes.init(gpa, device, ctl.glue);
        m.rows = try br.Pipes.init(gpa, device, .f32, ctl.rows);
        m.rows16 = try br.Pipes.init(gpa, device, .bf16, ctl.rows);
        m.moe = try moe.Pipes.init(gpa, device, ctl.moe);
        const cap = @max(m.ring, context);
        m.att = try attn.Attention.init(gpa, device, .{ .query_heads = c.heads, .kv_heads = c.kv_heads, .head_dim = c.head_dim, .capacity = cap, .output = .bf16 });
        m.k = try gpa.alloc(mtl.Buffer, c.layers);
        m.v = try gpa.alloc(mtl.Buffer, c.layers);
        const kvd = c.kv_heads * c.head_dim;
        for (0..c.layers) |i| {
            const len = if (c.full[i]) context else m.ring;
            m.k[i] = try m.buffer(len * kvd * 2);
            m.v[i] = try m.buffer(len * kvd * 2);
        }
        try m.rope();
        const r = max_rows;
        const d = c.hidden;
        const z = moe.bytes(m.route, r, d, c.moe_width);
        var ms: [9]mtl.Buffer = undefined;
        for (&ms, [_]usize{ z.logits, z.pick, z.wts, z.bucket, z.members, z.items, z.count, z.act, z.y }) |*b, n| b.* = try m.buffer(n);
        m.s = .{ .ids = try m.buffer(r * 4), .pos = try m.buffer(r * 4), .res = try m.buffer(r * d * 4), .x = try m.buffer(r * d * 2), .qkv = try m.buffer(r * c.qkvDim() * 2), .q = try m.buffer(r * c.heads * c.head_dim * 2), .att = try m.buffer(r * c.heads * c.head_dim * 2), .o = try m.buffer(r * d * 2), .mo = try m.buffer(r * d * 2), .mask = try m.buffer(2 * r * cap), .logits = try m.buffer(r * c.vocab * 4), .moe = ms };
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.k, m.v) |a, b| {
            a.deinit();
            b.deinit();
        }
        m.gpa.free(m.k);
        m.gpa.free(m.v);
        const s = m.s;
        for ([_]mtl.Buffer{ s.ids, s.pos, s.res, s.x, s.qkv, s.q, s.att, s.o, s.mo, s.mask, s.logits, m.cos, m.sin } ++ s.moe) |b| b.deinit();
        m.att.deinit();
        m.moe.deinit();
        m.rows.deinit();
        m.rows16.deinit();
        m.glue.deinit();
        m.queue.deinit();
    }

    /// RoPE tables: angle p / theta^(2j / D) in float64, cos and sin rounded to fp32, the halves repeated (neox).
    fn rope(m: *Model) !void {
        const dh = m.c.head_dim;
        m.cos = try m.buffer(m.context * dh * 4);
        m.sin = try m.buffer(m.context * dh * 4);
        const cos = m.cos.slice(f32, m.context * dh);
        const sin = m.sin.slice(f32, m.context * dh);
        const theta: f64 = m.c.rope_theta;
        for (0..m.context) |p| for (0..dh / 2) |j| {
            const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh)));
            for ([_]usize{ j, j + dh / 2 }) |col| {
                cos[p * dh + col] = @floatCast(@cos(ang));
                sin[p * dh + col] = @floatCast(@sin(ang));
            }
        };
    }

    /// The next forward starts at `p`: a prefix the caches still hold (a sliding ring keeps only its last keys).
    pub fn rewind(m: *Model, p: usize) !void {
        if (p > m.p) return error.PastCachedPositions;
        if (p > 0 and m.p - p + m.c.window > m.ring + 1) return error.RingOverwritten; // the window before p is gone
        m.p = p;
    }

    /// `tokens` at positions m.p.. through every layer; fp32 logits of every row (all) or the last into m.s.logits.
    pub fn forward(m: *Model, tokens: []const u32, all: bool, taps: ?Taps) ![]const f32 {
        const c = m.c;
        const n = tokens.len;
        if (n == 0 or n > m.max_rows) return error.TooManyRows;
        if (m.p + n > m.context) return error.PastContext;
        for (tokens, m.s.ids.slice(u32, n), m.s.pos.slice(u32, n), 0..) |t, *id, *pos, i| {
            if (t >= c.vocab) return error.TokenPastVocab;
            id.* = t;
            pos.* = @intCast(m.p + i);
        }
        if (taps) |t| if (t.rows < n or t.buf.length() < c.layers * 2 * n * c.hidden * 4) return error.TapsTooSmall;
        const cb = m.queue.commandBuffer();
        var e = cb.compute(.serial);
        const d = c.hidden;
        const ga: glue.Args = .{ .rows = @intCast(n), .width = @intCast(d), .eps = c.eps };
        m.glue.embed(e, .{ .buf = m.s.ids }, m.w.embed, .{ .buf = m.s.res }, ga);
        e.barrier();
        m.glue.rmsNorm(e, .{ .buf = m.s.res }, m.w.layers[0].input_norm, .{ .buf = m.s.x }, ga);
        e.barrier();
        // one command buffer reads every mask: sliding layers' at 0, full layers' after, each written once
        const last = m.p + n - 1;
        const mask = m.s.mask.contents();
        masks(mask[0 .. n * m.slidingKeys(last)], m.p, n, m.slidingKeys(last), .{ .window = c.window, .ring = m.ring });
        masks(mask[n * m.ring ..][0 .. n * (last + 1)], m.p, n, last + 1, null);
        const z = m.s.moe;
        const sc: moe.Scratch = .{ .logits = .{ .buf = z[0] }, .pick = .{ .buf = z[1] }, .wts = .{ .buf = z[2] }, .bucket = .{ .buf = z[3] }, .members = .{ .buf = z[4] }, .items = .{ .buf = z[5] }, .count = .{ .buf = z[6] }, .act = .{ .buf = z[7] }, .y = .{ .buf = z[8] } };
        for (m.w.layers, 0..) |L, i| {
            try m.attention(e, L, i, n);
            m.glue.addRms(e, .{ .buf = m.s.o }, .{ .buf = m.s.res }, L.post_attn_norm, L.pre_moe_norm, .{ .buf = m.s.x }, ga);
            e.barrier();
            if (taps) |t| e = tap(cb, e, m.s.res, t.buf, i * 2 * n * d * 4, n * d * 4);
            try moe.encode(e, m.moe, m.rows, m.route, .{ .buf = m.s.x }, d, n, L.router, L.bias, L.experts, sc, .{ .buf = m.s.mo });
            e.barrier();
            const next = if (i + 1 < m.w.layers.len) m.w.layers[i + 1].input_norm else m.w.norm;
            m.glue.addRms(e, .{ .buf = m.s.mo }, .{ .buf = m.s.res }, L.post_moe_norm, next, .{ .buf = m.s.x }, ga);
            e.barrier();
            if (taps) |t| e = tap(cb, e, m.s.res, t.buf, (i * 2 + 1) * n * d * 4, n * d * 4);
        }
        const first = if (all) 0 else n - 1;
        try project(e, m.rows, 4, .{ .buf = m.s.x, .off = first * d * 2 }, .{ m.w.head, m.w.head }, .{ .buf = m.s.logits }, n - first, d, c.vocab, c.vocab, true);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.log.err("kolibri1 metal forward: {s}", .{why});
            return error.GpuFailed;
        }
        m.p += n;
        return m.s.logits.slice(f32, (n - first) * c.vocab);
    }

    /// The residual's rows copied into the taps between two compute passes of the same command buffer.
    fn tap(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder, from: mtl.Buffer, to: mtl.Buffer, at: usize, bytes: usize) mtl.ComputeEncoder {
        e.end();
        const b = cb.blit();
        b.copy(from, 0, to, at, bytes);
        b.end();
        return cb.compute(.serial);
    }

    /// qkv, q/k norms and RoPE into the caches, attention over each row's keys, o: into m.s.o.
    fn attention(m: *Model, e: mtl.ComputeEncoder, L: wts.Layer, i: usize, n: usize) !void {
        const c = m.c;
        const d = c.hidden;
        const dh = c.head_dim;
        const qd = c.heads * dh;
        const kvd = c.kv_heads * dh;
        const ld = c.qkvDim();
        try m.dense16(e, .{ .buf = m.s.x }, L.q, .{ .buf = m.s.qkv }, n, d, qd, ld);
        try m.dense16(e, .{ .buf = m.s.x }, L.k, .{ .buf = m.s.qkv, .off = qd * 2 }, n, d, kvd, ld);
        try m.dense16(e, .{ .buf = m.s.x }, L.v, .{ .buf = m.s.qkv, .off = (qd + kvd) * 2 }, n, d, kvd, ld);
        e.barrier();
        const full = c.full[i];
        try m.glue.qk(e, .{ .qkv = .{ .buf = m.s.qkv }, .q_norm = L.q_norm, .k_norm = L.k_norm, .cos = .{ .buf = m.cos }, .sin = .{ .buf = m.sin }, .pos = .{ .buf = m.s.pos }, .q = .{ .buf = m.s.q }, .k_cache = .{ .buf = m.k[i] }, .v_cache = .{ .buf = m.v[i] } }, .{ .rows = @intCast(n), .heads = @intCast(c.heads), .kv_heads = @intCast(c.kv_heads), .dim = @intCast(dh), .rope = @intFromBool(!full), .ring = if (full) 0 else @intCast(m.ring), .eps = c.eps });
        e.barrier();
        const last = m.p + n - 1;
        const keys = if (full) last + 1 else m.slidingKeys(last);
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dh))));
        const base = if (full) n * m.ring else 0;
        var at: usize = 0;
        while (at < n) : (at += 16) {
            const r = @min(16, n - at);
            try m.att.encodeSplit(e, r, keys, keys, scale, .{ .buf = m.s.q, .off = at * qd * 2 }, .{ .buf = m.k[i] }, .{ .buf = m.v[i] }, .{ .buf = m.s.mask, .off = base + at * keys }, .{ .buf = m.s.att, .off = at * qd * 2 });
            e.barrier();
        }
        try m.dense16(e, .{ .buf = m.s.att }, L.o, .{ .buf = m.s.o }, n, qd, d, d);
        e.barrier();
    }

    fn slidingKeys(m: *const Model, last: usize) usize {
        return @min(m.ring, last + 1);
    }

    fn dense16(m: *Model, e: mtl.ComputeEncoder, x: Ref, w: [2]Ref, out: Ref, n: usize, k: usize, cols: usize, ldy: usize) !void {
        try project(e, m.rows16, 2, x, w, out, n, k, cols, ldy, false);
    }
};

/// x [n, k] bf16 times w [cols, k] (fp8 codes and scales, or bf16) into out (stride ldy, `elt` bytes) by 16 rows.
fn project(e: mtl.ComputeEncoder, p: br.Pipes, elt: usize, x: Ref, w: [2]Ref, out: Ref, n: usize, k: usize, cols: usize, ldy: usize, bf16: bool) !void {
    var at: usize = 0;
    while (at < n) : (at += 16) {
        const r = @min(16, n - at);
        try p.dense(e, .{ .buf = x.buf, .off = x.off + at * k * 2 }, w[0], w[1], .{ .buf = out.buf, .off = out.off + at * ldy * elt }, .{ .rows = @intCast(r), .k = @intCast(k), .n = @intCast(cols), .ldx = @intCast(k), .ldy = @intCast(ldy), .bf16 = @intFromBool(bf16) });
    }
}

/// Planted faults by block (0: none), for the selftest's whole-forward controls.
pub const Ctl = struct { glue: u8 = 0, rows: u8 = 0, moe: u8 = 0 };

pub const Sliding = struct { window: usize, ring: usize };

/// Row r at p sees slot s: full, position s <= p; sliding, the slot's latest position q (mod ring) in (p - window, p].
pub fn masks(out: []u8, p0: usize, n: usize, keys: usize, sliding: ?Sliding) void {
    const last = p0 + n - 1;
    for (0..n) |r| {
        const p = p0 + r;
        for (0..keys) |s| {
            const q: ?usize = if (sliding) |sl| (if (last >= (last + sl.ring - s) % sl.ring) last - (last + sl.ring - s) % sl.ring else null) else s;
            out[r * keys + s] = @intFromBool(if (q) |v| v <= p and (if (sliding) |sl| v + sl.window > p else true) else false);
        }
    }
}

test "Kolibri Metal masks: a sliding row sees exactly its window's slots across a ring wrap, a full row every key" {
    const ring = 8;
    var out: [3 * ring]u8 = undefined;
    masks(&out, 9, 3, ring, .{ .window = 4, .ring = ring }); // rows at 9, 10, 11 after writing slots 1, 2, 3
    // row 9 sees 6, 7, 8, 9 at slots 6, 7, 0, 1; row 11 sees 8 .. 11 at slots 0 .. 3
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 0, 0, 0, 0, 1, 1 }, out[0..8]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 1, 1, 0, 0, 0, 0 }, out[16..24]);
    var full: [2 * 5]u8 = undefined;
    masks(&full, 3, 2, 5, null);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 1, 1, 0, 1, 1, 1, 1, 1 }, &full);
    var early: [4]u8 = undefined;
    masks(&early, 0, 1, 4, .{ .window = 4, .ring = 8 }); // keys only slot 0 at position 0
    try std.testing.expectEqualSlices(u8, &.{ 1, 0, 0, 0 }, &early);
}

test "Kolibri Metal ring: a call's rows and a window, whole 64s" {
    try std.testing.expectEqual(@as(usize, 576), ringSize(16, 513));
    try std.testing.expect(ringSize(520, 513) >= 520 + 513);
}
