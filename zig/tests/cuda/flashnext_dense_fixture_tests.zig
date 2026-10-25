//! The provisional dense node against an external oracle's tiny dense fixture, read from a directory (not vendored).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const rows = 9;
const hidden = 5120;
const width = 64;

fn load(gpu: check.Gpu, a: std.mem.Allocator, dir: []const u8, name: []const u8, count: usize) ![]u16 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}.bf16", .{ dir, name });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 24));
    try check.expect(bytes.len == count * 2, "fixture {s} holds {d} bytes, want {d}", .{ name, bytes.len, count * 2 });
    const out = try a.alloc(u16, count);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    return out;
}

fn tensor(values: []const u16, n: usize, k: usize) core.safetensors.Tensor {
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(values) };
}

/// Distance in bf16 steps between two bf16 words of one sign (a step across zero counts both sides).
fn steps(a: u16, b: u16) u32 {
    const ia: i32 = if (a & 0x8000 != 0) -@as(i32, a & 0x7fff) else a;
    const ib: i32 = if (b & 0x8000 != 0) -@as(i32, b & 0x7fff) else b;
    return @abs(ia - ib);
}

const Stat = struct {
    off: usize = 0,
    worst: u32 = 0,

    fn add(s: *Stat, got: u16, want: u16) void {
        const d = steps(got, want);
        if (d != 0) s.off += 1;
        s.worst = @max(s.worst, d);
    }

    fn say(s: Stat, what: []const u8, total: usize) void {
        std.debug.print("  {s}: {d} of {d} differ from the oracle, worst {d} bf16 steps\n", .{ what, s.off, total, s.worst });
    }
};

/// The FFN chain on the oracle's weights and input: each stage's bf16 words against the oracle's per-op tensors.
fn ffnOps(gpu: check.Gpu, a: std.mem.Allocator, dir: []const u8) !void {
    var prng = std.Random.DefaultPrng.init(1);
    var r = try rig.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    const x = try load(gpu, a, dir, "x", rows * hidden);
    const wg = try load(gpu, a, dir, "wg", width * hidden);
    const wu = try load(gpu, a, dir, "wu", width * hidden);
    const wd = try load(gpu, a, dir, "wd", hidden * width);
    const want_gate = try load(gpu, a, dir, "gate", rows * width);
    const want_up = try load(gpu, a, dir, "up", rows * width);
    const want_product = try load(gpu, a, dir, "product", rows * width);
    const want_out = try load(gpu, a, dir, "output", rows * hidden);
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var act = try flash.dense_ffn.Act.load(gpu.d);
    defer act.unload();
    var owned = false;
    var gate = try flash.projection.dense(a, gpu.d, tensor(wg, width, hidden));
    errdefer if (!owned) gate.free();
    var up = try flash.projection.dense(a, gpu.d, tensor(wu, width, hidden));
    errdefer if (!owned) up.free();
    var down = try flash.projection.dense(a, gpu.d, tensor(wd, hidden, width));
    errdefer if (!owned) down.free();
    var ffn = try flash.dense_ffn.Ffn.init(gpu.d, &lane, act.f, gate, up, down, rows);
    owned = true;
    defer ffn.free();
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const out = try r.out(rows * hidden * 4);
    try ffn.run(&lane, r.s, dx, hidden, rows, out);
    const gates = try r.back(u16, ffn.gate_out.ptr, rows * width);
    const ups = try r.back(u16, ffn.up_out.ptr, rows * width);
    const prods = try r.back(u16, ffn.act_out.ptr, rows * width);
    const outs = try r.back(f32, out, rows * hidden);
    var s_gate: Stat = .{};
    var s_up: Stat = .{};
    var s_prod: Stat = .{};
    var s_out: Stat = .{};
    for (gates, want_gate) |g, w| s_gate.add(g, w);
    for (ups, want_up) |g, w| s_up.add(g, w);
    for (prods, want_product) |g, w| s_prod.add(g, w);
    // a 64-term sum cancels: judge the output against the size of its terms (the oracle's own product and weights)
    var worst_q: f64 = 0;
    for (0..rows) |row| for (0..hidden) |col| {
        var mag: f64 = 0;
        for (0..width) |j| mag += @abs(rig.bf(want_product[row * width + j]) * rig.bf(wd[col * width + j]));
        const want = rig.bf(want_out[row * hidden + col]);
        const got = rig.bf(rig.bf16(outs[row * hidden + col]));
        s_out.add(rig.bf16(outs[row * hidden + col]), want_out[row * hidden + col]);
        worst_q = @max(worst_q, @abs(got - want) / (mag * 0x1p-7 + @abs(want) * 0x1p-8 + 1e-30));
    };
    std.debug.print("dense FFN at hidden {d}, width {d}, {d} rows against the oracle's per-op tensors\n", .{ hidden, width, rows });
    s_gate.say("gate (bf16)", gates.len);
    s_up.say("up (bf16)", ups.len);
    s_prod.say("silu(gate) * up (one bf16 rounding here, the oracle rounds silu first)", prods.len);
    s_out.say("down output (fp32 rounded to bf16; steps are large where the sum cancels)", outs.len);
    std.debug.print("  down output worst |got - oracle| = {d:.3} of (2^-7 * sum|terms| + 2^-8 * |oracle|)\n", .{worst_q});
    // the single-rounding product from the oracle's own gate and up, in float64: what this kernel is meant to compute
    var single: Stat = .{};
    for (want_gate, want_up, prods) |g, u, got| {
        const gv = rig.bf(g);
        single.add(got, rig.bf16(@floatCast(gv / (1 + @exp(-gv)) * rig.bf(u))));
    }
    single.say("product against float64 silu(oracle gate) * oracle up rounded once", prods.len);
    try check.expect(s_gate.worst <= 1 and s_up.worst <= 1, "gate and up within one bf16 step of the oracle", .{});
    try check.expect(worst_q <= 1, "down output within its term-size bound of the oracle", .{});
}

/// Which limits the oracle's tiny shape breaks, one config field at a time (each set to the dense value in turn).
fn tinyShape(gpu: check.Gpu, a: std.mem.Allocator, dir: []const u8) !void {
    const path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
    const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    std.debug.print("tiny dense fixture against the compiled shape table\n", .{});
    if (flash.config.parse(a, bytes)) |parsed| {
        var c = parsed;
        c.deinit();
    } else |e| std.debug.print("  config.parse refuses it: {t} (the fixture is unquantized BF16; the CUDA path loads 4-bit affine)\n", .{e});
    const root = (try std.json.parseFromSlice(std.json.Value, a, bytes, .{})).value.object;
    const t = root.get("text_config").?.object;
    const n = struct {
        fn of(o: std.json.ObjectMap, k: []const u8) usize {
            return @intCast(o.get(k).?.integer);
        }
    }.of;
    var s = flash.shape.qwen4_dense;
    s.layers = n(t, "num_hidden_layers");
    s.dims = n(t, "hidden_size");
    s.streams = n(t, "hc_count");
    s.low = n(t, "hc_lowrank");
    s.vocab = n(t, "vocab_size");
    s.ffn = n(t, "intermediate_size");
    s.q_heads = n(t, "num_attention_heads");
    s.kv_heads = n(t, "num_key_value_heads");
    s.head_dim = n(t, "head_dim");
    s.index_heads = n(t, "indexer_n_heads");
    s.index_dim = n(t, "indexer_head_dim");
    s.index_ratio = n(t, "indexer_compress_ratio");
    s.index_top = n(t, "indexer_budget") / s.index_ratio;
    s.gdn_key_heads = n(t, "linear_num_key_heads");
    s.gdn_value_heads = n(t, "linear_num_value_heads");
    s.gdn_key_dim = n(t, "linear_key_head_dim");
    s.gdn_value_dim = n(t, "linear_value_head_dim");
    s.taps = n(t, "linear_conv_kernel_dim");
    s.full_every = n(t, "full_attention_interval");
    s.rotary_pairs = @intFromFloat(t.get("rope_parameters").?.object.get("partial_rotary_factor").?.float * @as(f64, @floatFromInt(s.head_dim)) / 2);
    const want = flash.shape.qwen4_dense;
    for (0..32) |_| {
        const field = s.limits() orelse break;
        std.debug.print("  blocked by {s}\n", .{field});
        const before = s;
        inline for (.{ .{ "hidden_size", "dims" }, .{ "hc_count", "streams" }, .{ "hc_lowrank", "low" }, .{ "head_dim", "head_dim" }, .{ "num_key_value_heads", "kv_heads" }, .{ "num_attention_heads", "q_heads" }, .{ "indexer_head_dim", "index_dim" }, .{ "indexer_budget", "index_top" }, .{ "linear_num_key_heads", "gdn_key_heads" }, .{ "linear_num_value_heads", "gdn_value_heads" }, .{ "linear_key_head_dim", "gdn_key_dim" }, .{ "linear_value_head_dim", "gdn_value_dim" }, .{ "linear_conv_kernel_dim", "taps" }, .{ "intermediate_size", "ffn" }, .{ "num_hidden_layers", "layers" }, .{ "full_attention_interval", "full_every" }, .{ "partial_rotary_factor", "rotary_pairs" }, .{ "num_experts_per_tok", "top_k" } }) |pair| {
            if (std.mem.eql(u8, field, pair[0])) @field(s, pair[1]) = @field(want, pair[1]);
        }
        if (std.mem.eql(u8, field, "indexer_budget")) s.index_ratio = want.index_ratio;
        if (std.meta.eql(before, s)) break; // a limit this table cannot clear
    }
    std.debug.print("  the tiny shape (hidden {d}, HC {d}, head {d}, GDN {d}/{d}) cannot run on the compiled table; ops are compared instead\n", .{ t.get("hidden_size").?.integer, t.get("hc_count").?.integer, t.get("head_dim").?.integer, t.get("linear_num_key_heads").?.integer, t.get("linear_num_value_heads").?.integer });
}

/// What the shared pass's hook needs of a pass: the mixed row, the fp32 branch, the lane, the pending flag.
const Pass = struct {
    s: struct { mixed: u64, moe_f32: u64 },
    k: struct { native_lane: cuda.qmmf.Lane },
    stream: cuda.Stream,
    rows: usize,
    pending_f32: bool = false,

    pub fn addAt(_: *const Pass, _: usize) ?u8 {
        return null;
    }
};

/// The shared pass's hook on many rows and the row path's step on each row alone: the same fp32 words.
fn hooks(gpu: check.Gpu, a: std.mem.Allocator) !void {
    const dims = flash.prompt_wt.dims;
    const ffn_w = 192;
    var prng = std.Random.DefaultPrng.init(7);
    var r = try rig.Rig.init(gpu, a, prng.random());
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var act = try flash.dense_ffn.Act.load(gpu.d);
    defer act.unload();
    const x = r.bfs(rows * dims, 1);
    var owned = false;
    var gate = try flash.projection.dense(a, gpu.d, tensor(r.bfs(ffn_w * dims, 0.04), ffn_w, dims));
    errdefer if (!owned) gate.free();
    var up = try flash.projection.dense(a, gpu.d, tensor(r.bfs(ffn_w * dims, 0.04), ffn_w, dims));
    errdefer if (!owned) up.free();
    var down = try flash.projection.dense(a, gpu.d, tensor(r.bfs(dims * ffn_w, 0.04), dims, ffn_w));
    errdefer if (!owned) down.free();
    var ffn = try flash.dense_ffn.Ffn.init(gpu.d, &lane, act.f, gate, up, down, rows);
    owned = true;
    defer ffn.free();
    var scratch = try flash.prompt_step.Scratch.alloc(@constCast(gpu.d), 1);
    defer scratch.free();
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const branch = try r.out(rows * dims * 4);
    var pass: Pass = .{ .s = .{ .mixed = dx, .moe_f32 = branch }, .k = .{ .native_lane = lane }, .stream = r.s, .rows = rows };
    try flash.batch_dense.block(&pass, &ffn, 0, .{});
    try check.expect(pass.pending_f32, "the hook leaves the fp32 branch pending for the write-back", .{});
    if (flash.batch_dense.block(&pass, &ffn, 0, .{ .moe_branch = 1 })) |_| return error.TestFailed else |e| {
        try check.expect(e == error.UnexpectedTensor, "the hook refuses a layer that keeps MoE tensors, got {t}", .{e});
    }
    const shared = try r.back(f32, branch, rows * dims);
    for (0..rows) |t| {
        try scratch.mixed.upload(0, std.mem.sliceAsBytes(x[t * dims ..][0..dims]));
        try flash.prompt_step.denseStep(&r.s, &lane, &ffn, &scratch, scratch.y.ptr);
        try r.sameRows("row path against the shared pass", f32, scratch.y.ptr, shared[t * dims ..][0..dims]);
    }
    std.debug.print("dense node: the shared pass's hook ({d} rows) and the row path (one row at a time) give the same fp32 words\n", .{rows});
}

pub fn run(gpu: check.Gpu, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    try hooks(gpu, arena.allocator());
    try tinyShape(gpu, arena.allocator(), args[0]);
    try ffnOps(gpu, arena.allocator(), args[1]);
}
