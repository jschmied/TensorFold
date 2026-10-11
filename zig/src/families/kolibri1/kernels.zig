//! Kolibri 1's kernels: the Python engine's Triton set (captured), its FP8 lane, grouped experts and prompt attention.
const std = @import("std");
const cuda = @import("cuda");
const tree = @import("tree.zig");

const aot = cuda.aot;
const grouped = cuda.grouped;
const experts = cuda.experts;

/// prefill_attention.cu's instance for 48 query heads over 4 KV heads: heads_a_block(12) = 4, eight staging slots.
const pattn_symbol = "_ZN20tf_prefill_attention12pattn_kernelILi128ELi8ELi4ELi8EEEvPK13__nv_bfloat16S3_S3_PS1_iiiiif";
const pattn_smem: u32 = 65536;
const pattn_heads = 4; // query heads a block
const merge_columns = 64; // output columns a merge program folds

pub const Kernels = struct {
    set: aot.Set,
    lane: cuda.qmmf.Lane,
    fp8x: experts.Fp8Experts,
    router: grouped.Router,
    mods: [4]cuda.Module, // experts (the plan kernels), prefill_attention, kolibri_ops, sample
    pattn: cuda.Function,
    embed: cuda.Function,
    argmax_fn: cuda.Function,
    bf16_fn: cuda.Function,
    draws: cuda.sample.Draws,
    norms: cuda.norms.Norms,
    moe: cuda.moe_glue.MoeGlue,
    dense: cuda.dense_rows.DenseRows,

    /// The built-in modules plus the captured Triton set in `set_dir` (tools: capture_kolibri.py, aot_pack.py).
    pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, set_dir: []const u8) !Kernels {
        const d = ctx.d;
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        const images = [_][]const u8{ cuda.kernels.experts, cuda.kernels.prefill_attention, cuda.kernels.kolibri_ops, cuda.kernels.sample };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.router = try grouped.Router.resolve(k.mods[0]);
        k.pattn = try k.mods[1].function(pattn_symbol);
        try k.pattn.allowDynamicShared(pattn_smem);
        k.embed = try k.mods[2].function("tf_kolibri_embed");
        k.argmax_fn = try k.mods[2].function("tf_kolibri_argmax");
        k.bf16_fn = try k.mods[2].function("tf_kolibri_to_bf16");
        k.draws = try cuda.sample.Draws.resolve(k.mods[3]);
        const cap = try ctx.capability();
        k.lane = try cuda.qmmf.Lane.load(d, cap / 10);
        errdefer k.lane.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        k.fp8x = try experts.Fp8Experts.load(d, sms);
        errdefer k.fp8x.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.moe = try cuda.moe_glue.MoeGlue.load(d);
        errdefer k.moe.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        k.set = try aot.Set.load(gpa, io, d, ctx.device, set_dir);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.set.deinit();
        k.norms.unload();
        k.moe.unload();
        k.dense.unload();
        k.fp8x.unload();
        k.lane.unload();
        for (&k.mods) |*m| m.unload();
    }
};

fn int(x: usize) i32 {
    return @intCast(x);
}

/// Launches on one stream, each the Python call it mirrors.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn triton(o: Ops, f: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        try o.k.set.run(o.s, f, .{ @intCast(grid[0]), @intCast(grid[1]), @intCast(grid[2]) }, args, consts);
    }

    /// weights.embed[ids].float(): the fp32 residual of `rows` tokens.
    pub fn embed(o: Ops, table: u64, ids: u64, rows: usize, d: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(ids);
        a.add(int(d));
        a.add(out);
        try cuda.launch.launch(o.k.embed, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// Each row's argmax over `v` fp32 logits, ties to the lowest id (torch.argmax), into `out` (u32 a row).
    pub fn argmax(o: Ops, logits: u64, rows: usize, v: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(int(v));
        a.add(out);
        try cuda.launch.launch(o.k.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, o.s, &a);
    }

    /// bf16 copies of `n` fp32 logits (round to nearest even), as sample.cu's draws read them.
    pub fn toBf16(o: Ops, x: u64, y: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(i64, @intCast(n)));
        try cuda.launch.launch(o.k.bf16_fn, .{ .grid = .{ .x = @intCast(@min((n + 255) / 256, 4096)) }, .block = .{ .x = 256 } }, o.s, &a);
    }

    /// forward.rms (bf16 out): a program a row.
    pub fn rms(o: Ops, x: u64, w: u64, out: u64, rows: usize, n: usize, eps: f32) !void {
        try o.k.norms.rms(o.s, x, .f32, w, out, .bf16, rows, n, eps);
    }

    /// glue.add_rms: res += rms(x) * w1 in place; out = bf16(rms(res) * w2).
    pub fn addRms(o: Ops, x: u64, res: u64, w1: u64, w2: u64, out: u64, rows: usize, n: usize, eps: f32) !void {
        try o.k.norms.addRms(o.s, x, .bf16, res, w1, w2, out, .bf16, rows, n, eps);
    }

    pub const Qkv = struct { rows: u64, norms: u64, cos: u64, sin: u64, pos: u64, slots: u64, q: u64, k: u64, v: u64, kc: u64, vc: u64, len: usize };

    /// glue.qkv: q/k norms, RoPE on sliding layers, k and v into the cache at pos % len.
    pub fn qkv(o: Ops, a: Qkv, n: usize, h: usize, hk: usize, d: usize, eps: f32, rope: bool) !void {
        try o.triton("_qkv", .{ n, h + 2 * hk, 1 }, &.{ aot.ptr("QKV", "*bf16", a.rows), aot.ptr("NW", "*fp32", a.norms), aot.ptr("COS", "*fp32", a.cos), aot.ptr("SIN", "*fp32", a.sin), aot.ptr("POS", "*i32", a.pos), aot.ptr("SLOT", "*i32", a.slots), aot.ptr("Q", "*bf16", a.q), aot.ptr("KO", "*bf16", a.k), aot.ptr("VO", "*bf16", a.v), aot.ptr("KC", "*bf16", a.kc), aot.ptr("VC", "*bf16", a.vc), aot.int("LEN", int(a.len)) }, &.{ aot.ci("H", @intCast(h)), aot.ci("HK", @intCast(hk)), aot.ci("D", @intCast(d)), aot.cf("EPS", eps), aot.ci("ROPE", @intFromBool(rope)) });
    }

    /// attention.sliding: a program a (row, KV head) over the ring's last `window` keys.
    pub fn sliding(o: Ops, q: u64, kc: u64, vc: u64, out: u64, pos: u64, slots: u64, w: usize, h: usize, hk: usize, d: usize, window: usize, ring: usize, scale: f32) !void {
        try o.triton("_sliding", .{ w, hk, 1 }, &.{ aot.ptr("Q", "*bf16", q), aot.ptr("KC", "*bf16", kc), aot.ptr("VC", "*bf16", vc), aot.ptr("OUT", "*bf16", out), aot.ptr("POS", "*i32", pos), aot.ptr("SLOT", "*i32", slots) }, &.{ aot.ci("H", @intCast(h)), aot.ci("HK", @intCast(hk)), aot.ci("D", @intCast(d)), aot.ci("G", @intCast(h / hk)), aot.ci("WIN", @intCast(window)), aot.ci("RING", @intCast(ring)), aot.cf("SCALE", scale), aot.ci("BN", 64), aot.ci("GP", 16) });
    }

    /// prefill_attention.attention: prompt rows [p0, p0 + w) over every key before them (one stream's caches).
    pub fn prompt(o: Ops, q: u64, kc: u64, vc: u64, out: u64, p0: usize, w: usize, h: usize, hk: usize, scale: f32) !void {
        const g = h / hk;
        if (g % pattn_heads != 0 or g % 8 == 0) return error.PromptAttentionInstance; // heads_a_block(g) must be 4
        var a: cuda.Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, w, h, hk, g }) |v| a.add(int(v));
        a.add(scale);
        const rows = 16 * (8 / pattn_heads);
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast((w + rows - 1) / rows), .y = @intCast(hk * (g / pattn_heads)) }, .block = .{ .x = 256 }, .shared = pattn_smem };
        try cuda.launch.launch(o.k.pattn, cfg, o.s, &a);
    }

    pub const Tree = struct { rows: u64, streams: u64, items: u64, parents: u64, paths: u64, depths: u64, n_items: usize, chunks: usize, width: usize };

    /// attention.from_packed's paths: each window row's ancestors.
    pub fn paths(o: Ops, t: Tree) !void {
        try o.triton("_paths", .{ t.width, 1, 1 }, &.{ aot.ptr("PARENTS", "*i32", t.parents), aot.ptr("PATHS", "*i32", t.paths), aot.ptr("DEPTHS", "*i32", t.depths) }, &.{aot.ci("MAXD", tree.max_nodes)});
    }

    pub const Partials = struct { o: u64, m: u64, l: u64 };

    /// attention.attention: committed keys by item (_shared), each row's own path (_tail), then the merge.
    pub fn treeAttention(o: Ops, q: u64, kn: u64, vn: u64, origin: u64, offs: u64, t: Tree, p: Partials, out: u64, h: usize, hk: usize, d: usize, scale: f32) !void {
        const g = h / hk;
        const w = t.width;
        const cs = [_]aot.Const{ aot.ci("H", @intCast(h)), aot.ci("HK", @intCast(hk)), aot.ci("D", @intCast(d)), aot.ci("G", @intCast(g)), aot.ci("CH", tree.chunk), aot.cf("SCALE", scale), aot.ci("GR", tree.group) };
        if (t.n_items > 0) try o.triton("_shared", .{ t.n_items * hk, 1, 1 }, &.{ aot.ptr("Q", "*bf16", q), aot.ptr("KC", "*bf16", origin), aot.ptr("VC", "*bf16", origin), aot.ptr("OFF", "*i64", offs), aot.ptr("STREAM", "*i32", t.streams), aot.ptr("ITEMS", "*i32", t.items), aot.ptr("PO", "*fp32", p.o), aot.ptr("PM", "*fp32", p.m), aot.ptr("PL", "*fp32", p.l), aot.int("W", int(w)) }, &cs);
        const tails = 1 + (tree.max_nodes + tree.chunk - 1) / tree.chunk;
        try o.triton("_tail", .{ w, hk, tails }, &.{ aot.ptr("Q", "*bf16", q), aot.ptr("KN", "*bf16", kn), aot.ptr("VN", "*bf16", vn), aot.ptr("KC", "*bf16", origin), aot.ptr("VC", "*bf16", origin), aot.ptr("OFF", "*i64", offs), aot.ptr("STREAM", "*i32", t.streams), aot.ptr("ROWS", "*i32", t.rows), aot.ptr("PATHS", "*i32", t.paths), aot.ptr("DEPTHS", "*i32", t.depths), aot.ptr("PO", "*fp32", p.o), aot.ptr("PM", "*fp32", p.m), aot.ptr("PL", "*fp32", p.l), aot.int("W", int(w)) }, &(cs ++ [_]aot.Const{aot.ci("MAXD", tree.max_nodes)}));
        try o.triton("_merge", .{ w, hk, d / merge_columns }, &.{ aot.ptr("PO", "*fp32", p.o), aot.ptr("PM", "*fp32", p.m), aot.ptr("PL", "*fp32", p.l), aot.ptr("OUT", "*bf16", out), aot.ptr("STREAM", "*i32", t.streams), aot.ptr("ROWS", "*i32", t.rows), aot.int("W", int(w)) }, &.{ aot.ci("H", @intCast(h)), aot.ci("D", @intCast(d)), aot.ci("G", @intCast(g)), aot.ci("DS", merge_columns), aot.ci("CH", tree.chunk), aot.ci("GR", tree.group) });
    }

    /// cuda.moe.router: x [m, D] bf16 times rows [ne, D] bf16 -> [m, ne] fp32 (the MoE router and the head).
    pub fn router(o: Ops, x: u64, x_stride: usize, rows: u64, out: u64, m: usize, d: usize, ne: usize) !void {
        try o.k.dense.rows(o.s, x, x_stride, rows, out, ne, m, ne, d);
    }

    /// moe._topk: each row's top-k experts by sigmoid + bias, weights, then the shared expert's slot.
    pub fn topk(o: Ops, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, ne: usize, k: usize) !void {
        try o.k.moe.topk(o.s, logits, bias, .{ .experts = ne, .topk = k, .shared = true }, pick, wts, rows);
    }

    /// cuda.moe.combine: y [R, S, D] (fp32 decode, bf16 prompt) weighted by wts [R, S] -> [R, D] bf16.
    pub fn combine(o: Ops, y: u64, y_bf16: bool, wts: u64, out: u64, rows: usize, s: usize, d: usize) !void {
        try o.k.moe.combine(o.s, y, if (y_bf16) .bf16 else .f32, wts, out, .bf16, rows, s, d);
    }
};
