//! Kolibri 1's forward over streams' serial chains of rows (forward.py Model.forward), launch for launch.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const wts = @import("weights.zig");
const kern = @import("kernels.zig");
const tree = @import("tree.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const grouped = cuda.grouped;
const experts = cuda.experts;

pub const prompt_chunk = 8192; // prompt rows a forward (forward.py PROMPT_CHUNK)
const bn = 64; // the sliding kernel's key block

/// attention.ring_size: keys a sliding layer keeps when a forward writes `rows` before its first row reads back.
pub fn ringSize(rows: usize, window: usize) usize {
    return (rows + window + bn - 1) / bn * bn;
}

/// A stream's rows in one forward: its cache slot, first position, tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

pub const Model = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const wts.Weights,
    c: Config,
    context: usize,
    slots: usize,
    ring: usize,
    max_rows: usize,
    k: []u64, // per layer: the cache [slots, len, HK, D] bf16 (len: context for full layers, the ring for sliding)
    v: []u64,
    norms: []u64, // per layer: q_norm then k_norm, fp32 [2, D]
    cos: u64,
    sin: u64,
    origin: u64, // the bf16 base cache offsets are measured from (attention.base)
    s: Scratch,
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    const Scratch = struct { ids: u64, pos: u64, slot: u64, res: u64, x: u64, qkv: u64, q: u64, kn: u64, vn: u64, att: u64, o: u64, part: u64, logits_e: u64, pick: u64, wts: u64, plan: grouped.Plan, act: u64, y: u64, mo: u64, flat: u64, paths: u64, depths: u64, po: u64, pm: u64, pl: u64, offs: u64, sel: u64 };

    fn alloc(m: *Model, bytes: usize) !u64 {
        const b = try DeviceBuffer.alloc(m.d, @max(bytes, 256));
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    fn zeros(m: *Model, s: cuda.Stream, bytes: usize) !u64 {
        const p = try m.alloc(bytes);
        try m.d.check(m.d.api.cuMemsetD8Async(p, 0, @max(bytes, 256), s.handle), "cuMemsetD8Async");
        return p;
    }

    /// Caches for `slots` streams of `context` positions, RoPE tables, and scratch for `max_rows` rows a forward.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const wts.Weights, context: usize, slots: usize, max_rows: usize) !Model {
        const c = w.config;
        var m: Model = .{ .gpa = gpa, .d = d, .w = w, .c = c, .context = context, .slots = slots, .ring = ringSize(prompt_chunk, c.window), .max_rows = max_rows, .k = undefined, .v = undefined, .norms = undefined, .cos = 0, .sin = 0, .origin = 0, .s = undefined };
        errdefer m.deinit();
        const dh = c.head_dim;
        m.k = try gpa.alloc(u64, c.layers);
        m.v = try gpa.alloc(u64, c.layers);
        m.norms = try gpa.alloc(u64, c.layers);
        for (0..c.layers) |i| {
            const len = if (c.full[i]) context else m.ring;
            const bytes = slots * len * c.kv_heads * dh * 2;
            m.k[i] = try m.zeros(s, bytes);
            m.v[i] = try m.zeros(s, bytes);
            m.norms[i] = try m.alloc(2 * dh * 4);
            const L = w.layers[i];
            try d.check(d.api.cuMemcpyDtoDAsync_v2(m.norms[i], L.q_norm, dh * 4, s.handle), "cuMemcpyDtoDAsync");
            try d.check(d.api.cuMemcpyDtoDAsync_v2(m.norms[i] + dh * 4, L.k_norm, dh * 4, s.handle), "cuMemcpyDtoDAsync");
        }
        try m.rope(s);
        m.origin = try m.zeros(s, 64 * 2);
        try m.scratch(s);
        try s.synchronize();
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
        m.gpa.free(m.k);
        m.gpa.free(m.v);
        m.gpa.free(m.norms);
    }

    /// forward.py's tables: angle p / theta^(2j / D) in float64, cos and sin rounded to fp32, the halves repeated.
    fn rope(m: *Model, s: cuda.Stream) !void {
        const dh = m.c.head_dim;
        const cos = try m.gpa.alloc(f32, m.context * dh);
        defer m.gpa.free(cos);
        const sin = try m.gpa.alloc(f32, m.context * dh);
        defer m.gpa.free(sin);
        const theta: f64 = m.c.rope_theta;
        for (0..m.context) |p| for (0..dh / 2) |j| {
            const step = @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(dh));
            const ang = @as(f64, @floatFromInt(p)) / std.math.pow(f64, theta, step);
            for ([_]usize{ j, j + dh / 2 }) |col| {
                cos[p * dh + col] = @floatCast(@cos(ang));
                sin[p * dh + col] = @floatCast(@sin(ang));
            }
        };
        m.cos = try m.alloc(cos.len * 4);
        m.sin = try m.alloc(sin.len * 4);
        try m.upload(s, m.cos, std.mem.sliceAsBytes(cos));
        try m.upload(s, m.sin, std.mem.sliceAsBytes(sin));
        try s.synchronize();
    }

    fn upload(m: *Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try m.d.check(m.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
    }

    fn scratch(m: *Model, s: cuda.Stream) !void {
        _ = s;
        const c = m.c;
        const r = m.max_rows;
        const dh = c.head_dim;
        const slots = c.top_k + 1;
        const pairs = r * slots;
        const count = c.experts + 1;
        const wide = pairs > grouped.small;
        const chunks = (m.context + tree.max_nodes) / tree.chunk + 2;
        m.s = .{
            .ids = try m.alloc(r * 4),
            .pos = try m.alloc(r * 4),
            .slot = try m.alloc(r * 4),
            .res = try m.alloc(r * c.hidden * 4),
            .x = try m.alloc(r * c.hidden * 2),
            .qkv = try m.alloc(r * (c.heads + 2 * c.kv_heads) * dh * 2),
            .q = try m.alloc(r * c.heads * dh * 2),
            .kn = try m.alloc(r * c.kv_heads * dh * 2),
            .vn = try m.alloc(r * c.kv_heads * dh * 2),
            .att = try m.alloc(r * c.heads * dh * 2),
            .o = try m.alloc(r * c.hidden * 2),
            .part = try m.alloc(8 * r * @max(c.hidden, (c.heads + 2 * c.kv_heads) * dh) * 4),
            .logits_e = try m.alloc(r * c.experts * 4),
            .pick = try m.alloc(pairs * 4),
            .wts = try m.alloc(pairs * 4),
            .plan = .{
                .members = try m.alloc(pairs * 4),
                .items = try m.alloc(grouped.maxItems(pairs, count, 16) * 12),
                .counts = try m.alloc(8),
                .rank = try m.alloc((if (wide) pairs else 1) * 4),
                .hist = try m.alloc((if (wide) (pairs + 1023) / 1024 * count else 1) * 4),
            },
            .act = try m.alloc(pairs * c.moe_width * 2),
            .y = try m.alloc(pairs * c.hidden * 4),
            .mo = try m.alloc(r * c.hidden * 2),
            .flat = try m.alloc((r * 2 + m.slots * 4 + 3 * r * c.heads / c.kv_heads * chunks) * 4 + 4096),
            .paths = try m.alloc(r * tree.max_nodes * 4),
            .depths = try m.alloc(r * 4),
            .po = try m.alloc(chunks * r * c.heads * dh * 4),
            .pm = try m.alloc(chunks * r * c.heads * 4),
            .pl = try m.alloc(chunks * r * c.heads * 4),
            .offs = try m.alloc(m.slots * 2 * 8),
            .sel = try m.alloc(r * c.hidden * 2),
        };
    }

    /// fp32 logits [rows.len, V] at `out` of the chains' rows `rows` (default: each chain's last), as forward.py.
    pub fn forward(m: *Model, o: kern.Ops, chains: []const Chain, prompt: bool, rows: ?[]const usize, out: u64) !void {
        const c = m.c;
        const gpa = m.gpa;
        const s = o.s;
        if (prompt and chains.len != 1) return error.PromptTakesOneStream;
        var n: usize = 0;
        for (chains) |ch| {
            if (ch.tokens.len > m.ring - c.window) return error.ChainOverrunsRing;
            if (ch.p0 + ch.tokens.len > m.context) return error.PastContext;
            n += ch.tokens.len;
        }
        if (n > m.max_rows) return error.TooManyRows;
        const ids = try gpa.alloc(i32, n);
        defer gpa.free(ids);
        const pos = try gpa.alloc(i32, n);
        defer gpa.free(pos);
        const slot = try gpa.alloc(i32, n);
        defer gpa.free(slot);
        var at: usize = 0;
        for (chains) |ch| for (ch.tokens, 0..) |t, i| {
            ids[at] = @intCast(t);
            pos[at] = @intCast(ch.p0 + i);
            slot[at] = @intCast(ch.slot);
            at += 1;
        };
        try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids));
        try m.upload(s, m.s.pos, std.mem.sliceAsBytes(pos));
        try m.upload(s, m.s.slot, std.mem.sliceAsBytes(slot));
        const w = m.w;
        const h = c.heads;
        const hk = c.kv_heads;
        const dh = c.head_dim;
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dh))));
        try o.embed(w.embed, m.s.ids, n, c.hidden, m.s.res);
        try o.rms(m.s.res, w.layers[0].input_norm, m.s.x, n, c.hidden, c.eps);
        // the tree plan for full layers' decode rows: each chain a serial path after its committed keys
        var t: ?kern.Ops.Tree = null;
        var plan_flat: ?[]i32 = null;
        defer if (plan_flat) |f| gpa.free(f);
        if (!prompt) {
            const windows = try gpa.alloc(tree.Window, chains.len);
            defer gpa.free(windows);
            var parents: std.ArrayList(i32) = .empty;
            defer parents.deinit(gpa);
            for (chains) |ch| for (0..ch.tokens.len) |i| try parents.append(gpa, @as(i32, @intCast(i)) - 1);
            var off: usize = 0;
            for (chains, windows) |ch, *win| {
                win.* = .{ .parents = parents.items[off..][0..ch.tokens.len], .committed = ch.p0 };
                off += ch.tokens.len;
            }
            const p = try tree.build(gpa, windows, h / hk);
            plan_flat = p.flat;
            try m.upload(s, m.s.flat, std.mem.sliceAsBytes(p.flat));
            const base = m.s.flat;
            const rows_at = base;
            const streams_at = base + p.width * 4;
            const items_at = streams_at + p.streams * 16;
            const parents_at = items_at + p.items * 12;
            t = .{ .rows = rows_at, .streams = streams_at, .items = items_at, .parents = parents_at, .paths = m.s.paths, .depths = m.s.depths, .n_items = p.items, .chunks = p.most, .width = p.width };
            try o.paths(t.?);
        }
        const slots_e = c.top_k + 1;
        const pairs = n * slots_e;
        const count = c.experts + 1;
        const tile = if (prompt) experts.promptTile(.fp8g) else 16;
        const items = grouped.maxItems(pairs, count, tile);
        const gx: experts.Grouped = .{ .fp8g = &o.k.fp8x };
        for (w.layers, 0..) |L, i| {
            try o.k.lane.matmul(s, m.s.x, c.hidden, n, L.qkv, m.s.qkv, m.s.part);
            const len = if (c.full[i]) m.context else m.ring;
            try o.qkv(.{ .rows = m.s.qkv, .norms = m.norms[i], .cos = m.cos, .sin = m.sin, .pos = m.s.pos, .slots = m.s.slot, .q = m.s.q, .k = m.s.kn, .v = m.s.vn, .kc = m.k[i], .vc = m.v[i], .len = len }, n, h, hk, dh, c.eps, !c.full[i]);
            if (!c.full[i]) {
                try o.sliding(m.s.q, m.k[i], m.v[i], m.s.att, m.s.pos, m.s.slot, n, h, hk, dh, c.window, m.ring, scale);
            } else if (prompt) {
                const per = m.context * hk * dh * 2;
                const ch = chains[0];
                try o.prompt(m.s.q, m.k[i] + ch.slot * per, m.v[i] + ch.slot * per, m.s.att, ch.p0, n, h, hk, scale);
            } else {
                const per = m.context * hk * dh * 2;
                const offs = try gpa.alloc(i64, 2 * chains.len);
                defer gpa.free(offs);
                for (chains, 0..) |ch, j| {
                    offs[2 * j] = @intCast((m.k[i] + ch.slot * per - m.origin) / 2);
                    offs[2 * j + 1] = @intCast((m.v[i] + ch.slot * per - m.origin) / 2);
                }
                try m.upload(s, m.s.offs, std.mem.sliceAsBytes(offs));
                try o.treeAttention(m.s.q, m.s.kn, m.s.vn, m.origin, m.s.offs, t.?, .{ .o = m.s.po, .m = m.s.pm, .l = m.s.pl }, m.s.att, h, hk, dh, scale);
            }
            try o.k.lane.matmul(s, m.s.att, h * dh, n, L.o, m.s.o, m.s.part);
            try o.addRms(m.s.o, m.s.res, L.post_attn_norm, L.pre_moe_norm, m.s.x, n, c.hidden, c.eps);
            // moe.run: router logits, top-k + the shared slot, the plan, gate-up and down, the weighted sum
            try o.router(m.s.x, c.hidden, L.router, m.s.logits_e, n, c.hidden, c.experts);
            try o.topk(m.s.logits_e, L.bias, m.s.pick, m.s.wts, n, c.experts, c.top_k);
            try o.k.router.route(s, m.s.pick, pairs, count, tile, m.s.plan);
            const up_in: experts.Rows = .{ .x = m.s.x, .stride = c.hidden, .slots = slots_e };
            const down_in: experts.Rows = .{ .x = m.s.act, .stride = c.moe_width };
            if (prompt) {
                try gx.prompt(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, -1);
                try gx.prompt(s, .down_bf16, down_in, L.experts, m.s.plan, m.s.y, items, -1);
            } else {
                try gx.decode(s, .up, up_in, L.experts, m.s.plan, m.s.act, items, -1);
                try gx.decode(s, .down_f32, down_in, L.experts, m.s.plan, m.s.y, items, -1);
            }
            try o.combine(m.s.y, prompt, m.s.wts, m.s.mo, n, slots_e, c.hidden);
            const after = if (i + 1 < w.layers.len) w.layers[i + 1].input_norm else w.norm;
            try o.addRms(m.s.mo, m.s.res, L.post_moe_norm, after, m.s.x, n, c.hidden, c.eps);
        }
        // the head over the chosen rows (default: each chain's last)
        var ends: std.ArrayList(usize) = .empty;
        defer ends.deinit(gpa);
        const pick_rows = rows orelse blk: {
            var e: usize = 0;
            for (chains) |ch| {
                e += ch.tokens.len;
                try ends.append(gpa, e - 1);
            }
            break :blk ends.items;
        };
        for (pick_rows, 0..) |r, j| try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(m.s.sel + j * c.hidden * 2, m.s.x + r * c.hidden * 2, c.hidden * 2, s.handle), "cuMemcpyDtoDAsync");
        try o.router(m.s.sel, c.hidden, w.head, out, pick_rows.len, c.hidden, c.vocab);
    }
};
