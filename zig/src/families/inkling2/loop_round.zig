//! Inkling 2 Mini's forward over several streams' rows: row-wise kernels once, each slot's conv and attention alone.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const engine = @import("loop_engine.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const max_slots = engine.max_slots;

/// One stream's rows in a shared forward: its slot, the position of its first row, its tokens.
pub const Chain = struct { slot: usize, p0: usize, tokens: []const u32 };

const max_tiles = 128;

/// A stream's rows of the shared scratch.
const Part = struct { slot: usize, p0: usize, row0: usize, n: usize };

/// Streams in one attention call: rows <= split_rows, so each global layer takes its solo (split) kernel.
const Group = struct {
    first: usize,
    n: usize,
    row0: usize,
    rows: usize,
    last_pos: usize,
    n_tiles: [2]usize, // sliding, global
};

/// Streams in order, a new group where the rows would pass split_rows.
fn groupsOf(parts: []const Part, out: []Group) []Group {
    var g: usize = 0;
    for (parts, 0..) |p, i| {
        const last = p.p0 + p.n - 1;
        if (g > 0 and out[g - 1].rows + p.n <= attention.split_rows) {
            out[g - 1].n += 1;
            out[g - 1].rows += p.n;
            out[g - 1].last_pos = @max(out[g - 1].last_pos, last);
            continue;
        }
        out[g] = .{ .first = i, .n = 1, .row0 = p.row0, .rows = p.n, .last_pos = last, .n_tiles = undefined };
        g += 1;
    }
    return out[0..g];
}

/// Every chain's rows in one pass; logits of rows [head0, head0 + heads). Each stream's bits are its solo forward's.
pub fn forward(m: *Model, k: *const Kernels, s: cuda.Stream, chains: []const Chain, head0: usize, heads: usize, record: bool) !void {
    const c = m.w.c.text;
    const h: usize = c.hidden;
    if (chains.len == 0 or chains.len > max_slots) return error.BadLoopRows;
    for (chains) |ch| for (ch.tokens) |token| if (token >= c.vocab) return error.InvalidToken; // every stream, before any launch
    var parts: [max_slots]Part = undefined;
    var ids: [prompt_rows]i32 = undefined;
    var pos: [prompt_rows]i32 = undefined;
    var seg_of: [prompt_rows]i32 = undefined;
    var segs: [max_slots]attention.Seg = undefined;
    var conv_segs: [max_slots][4]i32 = undefined;
    var tiles: [max_slots * 2 * max_tiles]attention.Tile = undefined;
    var rows: usize = 0;
    for (chains, 0..) |ch, i| {
        const n = ch.tokens.len;
        if (n == 0 or ch.slot >= m.n_slots or ch.p0 + n > m.context or (record and n > window_rows)) return error.BadLoopRows;
        for (chains[0..i]) |o| if (o.slot == ch.slot) return error.BadLoopRows;
        if (rows + n > prompt_rows or (chains.len > 1 and rows + n > engine.round_rows)) return error.BadLoopRows;
        for (ch.tokens, 0..) |token, j| {
            ids[rows + j] = @intCast(token);
            pos[rows + j] = @intCast(ch.p0 + j);
        }
        parts[i] = .{ .slot = ch.slot, .p0 = ch.p0, .row0 = rows, .n = n };
        conv_segs[i] = .{ @intCast(rows), @intCast(n), @intCast(ch.slot), 0 };
        rows += n;
    }
    // A slot's caches are `off` bf16 elements past slot 0's in Model.caches; a row's segment counts from its group.
    var group_buf: [max_slots]Group = undefined;
    const groups = groupsOf(parts[0..chains.len], &group_buf);
    for (groups, 0..) |*g, gi| {
        for (g.first..g.first + g.n) |i| {
            const p = parts[i];
            segs[i] = .{ .row0 = @intCast(p.row0 - g.row0), .n = @intCast(p.n), .p0 = @intCast(p.p0), .off = @intCast(p.slot * m.stride() / 2) };
            @memset(seg_of[p.row0..][0..p.n], @intCast(i - g.first));
        }
        for (0..2) |kind| {
            const layer = if (kind == 0) c.local else c.global;
            const at = (gi * 2 + kind) * max_tiles;
            g.n_tiles[kind] = try attention.tiles(segs[g.first..][0..g.n], layer.q / layer.kv, tiles[at..][0..max_tiles]);
        }
    }
    if (!record and heads > window_rows * chains.len) return error.BadLoopRows;
    if (head0 + heads > rows) return error.BadLoopRows;
    const ps = parts[0..chains.len];
    const keep = m.slot;
    defer m.slot = keep;
    try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..rows]));
    try m.upload(s, m.s.pos, std.mem.sliceAsBytes(pos[0..rows]));
    try m.upload(s, m.s.seg_of, std.mem.sliceAsBytes(seg_of[0..rows]));
    try m.upload(s, m.s.segs, std.mem.sliceAsBytes(segs[0..chains.len]));
    try m.upload(s, m.s.conv_segs, std.mem.sliceAsBytes(conv_segs[0..chains.len]));
    try m.upload(s, m.s.tiles, std.mem.sliceAsBytes(tiles[0 .. groups.len * 2 * max_tiles]));
    for (ps) |p| {
        m.recorded[p.slot] = record;
        m.slot = p.slot;
        if (record) try m.copy(s, m.slots[p.slot].backup.ptr, m.slotBase() + m.layout.at(0).conv_offset, m.layout.bytes - m.layout.at(0).conv_offset);
    }

    try ops.embed(&k.glue, s, m.s.ids, rows, m.w.embed, m.w.embed_norm, m.s.partial_a, h, c.eps);
    var partial = m.s.partial_a;
    var next = m.s.partial_b;
    var n_blocks: usize = 0;
    const lp = c.loop.?;

    for (m.w.p.visits[0..m.w.p.n_visits]) |visit| {
        const layer = m.w.p.at(visit.layer);
        const lw = &m.w.layers[visit.layer];
        if (visit.pass >= 2 and visit.layer == lp.core_start) {
            try m.copy(s, partial, m.s.prelude, rows * h * 2);
            n_blocks = 1; // block zero is the embedding; later passes never inherit completed blocks
        }
        const messages = visit.hasMessages();
        if (messages) {
            const core_j = visit.layer - lp.core_start;
            try loop_math.messages(&k.loop, s, m.s.bank0, m.s.bank1, m.w.cross_qm + 2 * core_j * h * 4, m.w.cross_kv + 2 * core_j * h * 4, m.s.messages, rows, 1, h, c.eps);
        }
        const nq: usize = if (layer.kind == .global) 3 else 1;
        const query = m.w.attn_queries[visit.layer];
        const blocks = m.s.blocks[0..n_blocks];
        const raw_stride = rows * h * 4;
        const norm_stride = rows * h * 2;
        if (messages) {
            if (nq == 3) {
                try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw, 3, true, rows));
            } else {
                // A sliding layer stores one query; each gated role gets its own AR result from that query.
                for (0..3) |role| try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw + role * raw_stride, 1, true, rows));
            }
            for (0..3) |role| {
                const ctl = lw.controllers[role];
                try loop_math.gate(&k.loop, s, m.s.raw + role * raw_stride, m.s.messages + role * raw_stride, ctl.down, ctl.forget, ctl.gain, m.s.zeta, m.s.raw + role * raw_stride, rows, h, c.eps);
                const norms = [3]u64{ lw.attn.norms[role], lw.attn.norms[role], lw.attn.norms[role] };
                try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw + role * raw_stride, 0, norms, m.s.normalized + role * norm_stride, 1, false, rows));
            }
        } else {
            try ops.mix(&k.glue, s, m.mixArgs(blocks, partial, query, lw.attn.norms, m.s.normalized, nq, true, rows));
        }
        const u_q = m.s.normalized;
        const u_k = if (layer.kind == .global or messages) m.s.normalized + norm_stride else u_q;
        const u_v = if (layer.kind == .global or messages) m.s.normalized + 2 * norm_stride else u_q;
        const q_width: usize = @as(usize, layer.heads.q) * layer.heads.dim;
        const kv_width: usize = @as(usize, layer.heads.kv) * layer.heads.dim;
        const rel_width: usize = @as(usize, layer.heads.q) * c.d_rel;
        const taps: usize = @as(usize, layer.heads.kv) * config.conv_taps;
        try loop_math.projectQkv(&k.loop, s, u_q, u_k, u_v, lw.attn.wq, lw.attn.wk, lw.attn.wv, m.s.q_raw, m.s.k_raw, m.s.v_raw, rows, h, q_width, kv_width);
        try ops.rowdot(&k.glue, s, u_q, lw.attn.wr, m.s.rel_raw, rows, rel_width, h, true, false);
        try ops.rowdot(&k.glue, s, u_q, lw.attn.gate, m.s.gate, rows, q_width, h, true, false);
        try ops.rowdot(&k.glue, s, u_q, lw.attn.sink, m.s.sink, rows, layer.heads.q, h, true, false);

        const geometry = m.layerGeometry(visit.layer);
        const dynamic = layer.kind == .global;
        if (dynamic) for (0..2) |kind| {
            const dyn_out = if (kind == 0) m.s.k_dynamic else m.s.v_dynamic;
            try ops.rowdot(&k.glue, s, if (kind == 0) u_k else u_v, if (kind == 0) lw.attn.k_dynamic else lw.attn.v_dynamic, dyn_out, rows, taps, h, false, false);
            try loop_math.roundBf16(&k.loop, s, dyn_out, dyn_out, rows * taps);
        };
        const kind_i: usize = @intFromBool(dynamic);
        const stride = attention.slotsFor(geometry, m.context);
        const part_capacity = prompt_rows * layer.heads.q * stride;
        const prep: attention.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = @floatFromInt(c.log_floor), .alpha = c.log_alpha, .eps = c.eps };
        for (0..2) |kind| try convAll(m, k, s, ps, record, visit.cache_slot, kind, .{
            .x = if (kind == 0) m.s.k_raw else m.s.v_raw,
            .state = 0,
            .w = if (kind == 0) lw.attn.k_conv else lw.attn.v_conv,
            .dyn = if (dynamic) (if (kind == 0) m.s.k_dynamic else m.s.v_dynamic) else 0,
            .out = if (kind == 0) m.s.k_conv else m.s.v_conv,
            .x_bf16 = 1,
            .heads = @intCast(layer.heads.kv),
            .c = @intCast(kv_width),
            .rows = 0,
            .mode = 0,
        }, .{ .x = kv_width * 2, .dyn = taps * 4, .out = kv_width * 2 });
        m.slot = 0; // every group's caches count from slot 0's (Seg.off)
        const cache = m.cacheOf(visit.cache_slot);
        for (groups, 0..) |g, gi| {
            const r0 = g.row0;
            const r: attention.Rows = .{
                .rows = @intCast(g.rows),
                .heads = layer.heads.q,
                .kv_heads = layer.heads.kv,
                .pos = m.s.pos + r0 * 4,
                .seg_of = m.s.seg_of + r0 * 4,
                .segs = m.s.segs + g.first * @sizeOf(attention.Seg),
                .tiles = m.s.tiles + (gi * 2 + kind_i) * max_tiles * @sizeOf(attention.Tile),
                .n_tiles = @intCast(g.n_tiles[kind_i]),
                .last_pos = @intCast(g.last_pos),
                .streams = @intCast(g.n),
            };
            const q_prepared = m.s.q_prepared + r0 * q_width * 2;
            const rel_prepared = m.s.rel_prepared + r0 * rel_width * 2;
            try attention.prep(&k.attn, s, r, geometry, prep, .{ m.s.q_raw + r0 * q_width * 2, m.s.k_conv + r0 * kv_width * 2, m.s.v_conv + r0 * kv_width * 2, m.s.rel_raw + r0 * rel_width * 2 }, .{ lw.attn.q_scale, lw.attn.rel_scale }, q_prepared, rel_prepared, cache);
            try attention.relTable(&k.attn, s, r, rel_prepared, lw.attn.rel_proj, m.s.rel_table);
            try attention.attend(&k.attn, s, r, geometry, stride, part_capacity, q_prepared, cache, m.s.rel_table, m.s.part_o, m.s.part_ml, m.s.sink + r0 * layer.heads.q * 2, m.s.gate + r0 * q_width * 2, m.s.attn_heads + r0 * q_width * 2);
        }
        try ops.rowdot(&k.glue, s, m.s.attn_heads, lw.attn.wo, m.s.attn_out, rows, h, q_width, true, false);

        if (layer.boundary) {
            try m.copy(s, m.s.blocks[n_blocks], partial, rows * h * 2);
            n_blocks += 1;
        }
        try convAll(m, k, s, ps, record, visit.cache_slot, 2, .{
            .x = m.s.attn_out,
            .state = 0,
            .w = lw.attn_conv,
            .base = if (layer.boundary) 0 else partial,
            .resid = m.s.t,
            .x_bf16 = 1,
            .c = @intCast(h),
            .rows = 0,
            .mode = 1,
        }, .{ .x = h * 2, .base = h * 2, .resid = h * 4 });

        const mlp_query = m.w.mlp_queries[visit.layer];
        const mlp_blocks = m.s.blocks[0..n_blocks];
        if (messages) {
            try loop_math.ar(&k.loop, s, m.arBlocks(mlp_blocks, m.s.t, mlp_query, m.s.raw, 1, false, rows));
            const ctl = lw.controllers[3];
            try loop_math.gate(&k.loop, s, m.s.raw, m.s.messages + 3 * raw_stride, ctl.down, ctl.forget, ctl.gain, m.s.zeta, m.s.raw, rows, h, c.eps);
            try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw, 0, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
        } else {
            try ops.mix(&k.glue, s, m.mixArgs(mlp_blocks, m.s.t, mlp_query, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
        }
        try ops.rowdot(&k.glue, s, m.s.normalized, lw.router, m.s.route_scores, rows, c.experts, h, false, true);
        try ops.topk(&k.glue, s, m.s.route_scores, lw.router_bias, m.s.route_pick, m.s.route_weights, rows, c.experts, c.topk, c.route_scale);
        try modelopt.run(&k.moe, s, lw.experts, .{ .gate = c.gate_cap, .up = c.up_cap }, m.s.normalized, h, m.s.route_pick, m.s.route_weights, rows, c.topk, m.s.moe_out, m.s.moe_scratch);
        try convAll(m, k, s, ps, record, visit.cache_slot, 3, .{
            .x = m.s.moe_out,
            .state = 0,
            .w = lw.mlp_conv,
            .resid = m.s.t,
            .out = next,
            .x_bf16 = 1,
            .c = @intCast(h),
            .rows = 0,
            .mode = 2,
        }, .{ .x = h * 2, .resid = h * 4, .out = h * 2 });
        std.mem.swap(u64, &partial, &next);

        if (visit.layer + 1 == lp.core_start) try m.copy(s, m.s.prelude, partial, rows * h * 2);
        if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) {
            try m.copy(s, m.s.bank0, m.s.blocks[n_blocks - 1], rows * h * 2);
            try m.copy(s, m.s.bank1, partial, rows * h * 2);
        }
    }

    if (heads == 0) return;
    try ops.mix(&k.glue, s, m.mixArgs(m.s.blocks[0..n_blocks], partial, m.w.output_query, .{ m.w.norm, m.w.norm, m.w.norm }, m.s.hidden, 1, true, rows));
    try ops.rowdot(&k.glue, s, m.s.hidden + head0 * h * 2, m.w.unembed, m.s.logits, heads, c.unpadded_vocab, h, false, false);
    try loop_math.roundBf16(&k.loop, s, m.s.logits, m.s.logits, heads * c.unpadded_vocab);
    try ops.softcap(&k.glue, s, m.s.logits, heads * c.unpadded_vocab, c.logit_cap);
}

/// Bytes a row of each of a conv's row arrays (0: the field is unused or null).
const RowBytes = struct { x: usize = 0, dyn: usize = 0, base: usize = 0, resid: usize = 0, out: usize = 0 };

/// A conv site over all streams in one launch (state by slot * stride); with `record` one a stream, for its history.
fn convAll(m: *Model, k: *const Kernels, s: cuda.Stream, ps: []const Part, record: bool, visit: u32, site: usize, a: ops.ConvArgs, rb: RowBytes) !void {
    if (!record) {
        var all = a;
        m.slot = 0;
        all.state = m.convOf(visit, site);
        all.segs = m.s.conv_segs;
        all.state_stride = @intCast(m.stride() / 4);
        var widest: usize = 0;
        for (ps) |p| widest = @max(widest, p.n);
        all.rows = @intCast(widest);
        return ops.conv(&k.glue, s, all, ps.len);
    }
    for (ps) |p| {
        var one = a;
        m.slot = p.slot;
        const at = struct {
            fn f(ptr: u64, bytes: usize, row0: usize) u64 {
                return if (ptr == 0) 0 else ptr + row0 * bytes;
            }
        }.f;
        one.x = at(a.x, rb.x, p.row0);
        one.dyn = at(a.dyn, rb.dyn, p.row0);
        one.base = at(a.base, rb.base, p.row0);
        one.resid = at(a.resid, rb.resid, p.row0);
        one.out = at(a.out, rb.out, p.row0);
        one.state = m.convOf(visit, site);
        one.hist = m.histOf(visit, site);
        one.rows = @intCast(p.n);
        try ops.conv(&k.glue, s, one, 1);
    }
}

test "groupsOf: streams join while their rows fit split_rows, a wider stream alone" {
    const t = std.testing;
    var buf: [max_slots]Group = undefined;
    const ps = [_]Part{
        .{ .slot = 3, .p0 = 10, .row0 = 0, .n = 1 },  .{ .slot = 0, .p0 = 99, .row0 = 1, .n = 16 },
        .{ .slot = 1, .p0 = 5, .row0 = 17, .n = 47 }, .{ .slot = 2, .p0 = 7, .row0 = 64, .n = 1 },
        .{ .slot = 4, .p0 = 0, .row0 = 65, .n = 65 },
    };
    const g = groupsOf(&ps, &buf);
    try t.expectEqual(@as(usize, 3), g.len);
    try t.expectEqual(Group{ .first = 0, .n = 3, .row0 = 0, .rows = 64, .last_pos = 114, .n_tiles = g[0].n_tiles }, g[0]);
    try t.expectEqual(Group{ .first = 3, .n = 1, .row0 = 64, .rows = 1, .last_pos = 7, .n_tiles = g[1].n_tiles }, g[1]);
    try t.expectEqual(Group{ .first = 4, .n = 1, .row0 = 65, .rows = 65, .last_pos = 64, .n_tiles = g[2].n_tiles }, g[2]);
}
