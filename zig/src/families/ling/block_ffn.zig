//! Ling's MLPs on CUDA: SwiGLU over projections in their loaded formats, and the MoE block (router, groups, experts).
const std = @import("std");
const cuda = @import("cuda");
const graph = @import("graph.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const ling = @import("cuda.zig");

const glue = graph.glue;
const Ops = graph.Ops;
const Config = @import("config.zig").Config;

/// Gate and up into [gate | up] rows, SwiGLU with min(silu(g), limit) (0: none), down into `out` [rows][hidden].
pub fn mlp(o: Ops, b: *const state.Buffers, m: weights.Mlp, x: u64, ldx: usize, rows: usize, limit: f32, out: u64) !void {
    const w = m.gate.outputs();
    if (rows == 1) { // one row is its own columns: no copies
        try o.mm(m.gate, x, ldx, 1, b.gu);
        try o.mm(m.up, x, ldx, 1, b.gu + w * 2);
    } else {
        try o.mmCols(m.gate, x, ldx, rows, b.tmp, b.gu, 2 * w, 0);
        try o.mmCols(m.up, x, ldx, rows, b.tmp, b.gu, 2 * w, w);
    }
    try glue.swigluWith(o.m.swiglu, o.s, b.gu, b.act, b.xs, limit, rows, w);
    try o.mm(m.down, b.act, w, rows, out);
}

/// The MoE block on b.normed into b.y: routed experts' slots summed (weights fp32), plus the shared expert.
pub fn moe(o: Ops, b: *const state.Buffers, e: weights.Moe, g: *const graph.Graph, layer: u32, rows: usize, picks: ?u64) !void {
    const c = &g.w.config;
    const D: usize = c.hidden;
    const E: usize = c.experts;
    if (e.experts != E) return error.PartialExperts; // the router picks among every expert
    const top: usize = c.topk;
    const slots = top + 1;
    const lim = ling.limits(c, layer);
    try glue.router(o.set(), o.s, b.normed, D, e.router, b.mlog, b.rpart, rows, D, E);
    try glue.select(o.set(), o.s, b.mlog, if (e.bias != 0) e.bias else g.zero_bias, b.pick, b.wts, rows, top, E, c.routed_scale, c.norm_topk, ling.groups(c));
    if (picks) |p| try o.copy(b.pick, slots * 4, p, slots * 4, rows, slots * 4);
    switch (e.routed) {
        .grouped => |l| try grouped(o, b, l, rows, slots, E),
        .lanes => |ls| if (denseLanes()) {
            for (ls, 0..) |x, j| try mlp(o, b, x, b.normed, D, rows, lim.routed, b.ey + j * rows * D * 2);
        } else try picked(o, b, g, ls, lim.routed, rows, slots),
    }
    if (e.shared) |s| try mlp(o, b, s, b.normed, D, rows, e.shared_limit, b.sh);
    var a: cuda.Args = .{};
    inline for (.{ b.ey, b.pick, b.wts, if (e.shared != null) b.sh else @as(u64, 0), b.y, @as(i64, @intCast(rows)), @as(i64, @intCast(D)), @as(c_int, @intCast(slots)), @as(c_int, @intCast(top)), @as(c_int, @intFromBool(e.routed == .lanes)) }) |v| a.add(v);
    try cuda.launch.launch(o.m.combine, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((D + 255) / 256) }, .block = .{ .x = 256 } }, o.s, &a);
}

/// TF_LING_DENSE_EXPERTS=1 runs every bf16 expert on every row, as before `picked`: the same bytes, far slower.
fn denseLanes() bool {
    const S = struct {
        var read: bool = false;
        var on: bool = false;
    };
    if (!S.read) {
        S.on = if (std.c.getenv("TF_LING_DENSE_EXPERTS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
        S.read = true;
    }
    return S.on;
}

/// bf16 routed experts: each expert some row picked runs on the rows that picked it (gather, run, scatter into `ey`).
fn picked(o: Ops, b: *const state.Buffers, g: *const graph.Graph, ls: []const weights.Mlp, limit: f32, rows: usize, slots: usize) !void {
    const gpa = g.store.gpa;
    const D: usize = g.w.config.hidden;
    const top = slots - 1;
    const E = ls.len;
    const raw = try gpa.alloc(i32, rows * slots);
    defer gpa.free(raw);
    try o.s.synchronize();
    const pick: cuda.DeviceBuffer = .{ .d = g.store.d, .ptr = b.pick, .len = raw.len * 4 };
    try pick.download(0, std.mem.sliceAsBytes(raw));
    const start = try gpa.alloc(u32, E + 1); // expert e's rows are list[start[e] .. start[e + 1]], ascending
    defer gpa.free(start);
    @memset(start, 0);
    for (0..rows) |r| for (0..top) |k| {
        const e: usize = @intCast(raw[r * slots + k]);
        if (e >= E) return error.BadPick;
        start[e + 1] += 1;
    };
    for (0..E) |e| start[e + 1] += start[e];
    const list = try gpa.alloc(i32, start[E]);
    defer gpa.free(list);
    const fill = try gpa.alloc(u32, E);
    defer gpa.free(fill);
    @memset(fill, 0);
    for (0..rows) |r| for (0..top) |k| {
        const e: usize = @intCast(raw[r * slots + k]);
        list[start[e] + fill[e]] = @intCast(r);
        fill[e] += 1;
    };
    const index: cuda.DeviceBuffer = .{ .d = g.store.d, .ptr = b.gi, .len = @max(list.len, 1) * 4 };
    try index.upload(0, std.mem.sliceAsBytes(list));
    const row_bytes: c_int = @intCast(D * 2);
    for (0..E) |e| {
        const n = start[e + 1] - start[e];
        if (n == 0) continue;
        const y = b.ey + e * rows * D * 2;
        if (n == rows) { // every row picked it: its rows are the block's, in order
            try mlp(o, b, ls[e], b.normed, D, rows, limit, y);
            continue;
        }
        const at = b.gi + start[e] * 4;
        var a: cuda.Args = .{};
        inline for (.{ b.normed, at, b.gx, @as(c_int, @intCast(n)), row_bytes }) |v| a.add(v);
        try cuda.launch.launch(o.m.gather_rows, .{ .grid = .{ .x = n }, .block = .{ .x = 128 } }, o.s, &a);
        try mlp(o, b, ls[e], b.gx, D, n, limit, b.gy);
        var s: cuda.Args = .{};
        inline for (.{ b.gy, at, y, @as(i64, @intCast(D * 2)), @as(c_int, @intCast(n)), row_bytes }) |v| s.add(v);
        try cuda.launch.launch(o.m.scatter_rows, .{ .grid = .{ .x = n }, .block = .{ .x = 128 } }, o.s, &s);
    }
}

/// Prompt chunks under this many routed pairs an expert stream FP8 experts faster on the decode kernels (same bits).
const decode_pairs = 16;

/// Grouped experts on every row's top-k slots (the shared slot skipped): gate-up SwiGLU, then down to bf16 slot rows.
fn grouped(o: Ops, b: *const state.Buffers, l: cuda.experts.Layer, rows: usize, slots: usize, count: usize) !void {
    const pairs = rows * slots;
    const prompt = o.prompt and !(l.format == .fp8g and rows * (slots - 1) < decode_pairs * count);
    const tile: usize = if (prompt) cuda.experts.promptTile(l.format) else 16;
    const skip: i32 = @intCast(count);
    try o.m.router.route(o.s, b.pick, pairs, count + 1, tile, b.plan);
    const items = cuda.grouped.maxItems(pairs, count + 1, tile);
    const g = o.m.grouped();
    const up: cuda.experts.Rows = .{ .x = b.normed, .stride = l.dims, .slots = slots };
    const down: cuda.experts.Rows = .{ .x = b.eact, .stride = l.width };
    if (prompt) {
        try g.prompt(o.s, .up, up, l, b.plan, b.eact, items, skip);
        try g.prompt(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip);
    } else {
        try g.decode(o.s, .up, up, l, b.plan, b.eact, items, skip);
        try g.decode(o.s, .down_bf16, down, l, b.plan, b.ey, items, skip);
    }
}
