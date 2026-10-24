//! The provisional dense feed-forward node on synthetic weights (hidden 5120, FFN 14336) against an FP64 chain.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const bf = rig.bf;
const rbf = rig.rbf;
const hidden = 5120;
const width = 14336;
const most = 9;

fn tensor(values: []const u16, n: usize, k: usize) core.safetensors.Tensor {
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(values) };
}

/// An fp32 sum over k products on 16-wide mma steps: one rounding a step and a few after.
fn sumTol(magnitude: f64, k: usize) f64 {
    return magnitude * @as(f64, @floatFromInt(k / 16 + 16)) * 0x1p-24 + 1e-30;
}

fn dot(x: []const u16, w: []const u16, mag: *f64) f64 {
    var sum: f64 = 0;
    var abs: f64 = 0;
    for (x, w) |a, b| {
        const t = bf(a) * bf(b);
        sum += t;
        abs += @abs(t);
    }
    mag.* = abs;
    return sum;
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0xd3a5e);
    var r = try rig.Rig.init(gpu, arena.allocator(), prng.random());
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, (try gpu.ctx.capability()) / 10);
    defer lane.unload();
    var act = try flash.dense_ffn.Act.load(gpu.d);
    defer act.unload();
    const x = r.bfs(most * hidden, 1);
    const gw = r.bfs(width * hidden, 0.04);
    const uw = r.bfs(width * hidden, 0.04);
    const dw = r.bfs(hidden * width, 0.02);
    var gate = try flash.projection.dense(r.a, gpu.d, tensor(gw, width, hidden));
    errdefer gate.free();
    var up = try flash.projection.dense(r.a, gpu.d, tensor(uw, width, hidden));
    errdefer up.free();
    var down = try flash.projection.dense(r.a, gpu.d, tensor(dw, hidden, width));
    errdefer down.free();
    var ffn = try flash.dense_ffn.Ffn.init(gpu.d, &lane, act.f, gate, up, down, most);
    defer ffn.free();
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const out = try r.out(most * hidden * 4);
    var gates: []u16 = undefined;
    var ups: []u16 = undefined;
    var acts: []u16 = undefined;
    var outs: []f32 = undefined;
    for ([_]usize{ most, 1, 5 }) |rows| {
        try ffn.run(&lane, r.s, dx, hidden, rows, out);
        if (rows == most) {
            gates = try r.back(u16, ffn.gate_out.ptr, most * width);
            ups = try r.back(u16, ffn.up_out.ptr, most * width);
            acts = try r.back(u16, ffn.act_out.ptr, most * width);
            outs = try r.back(f32, out, most * hidden);
        } else {
            try r.sameRows("dense FFN gate BF16", u16, ffn.gate_out.ptr, gates[0 .. rows * width]);
            try r.sameRows("dense FFN up BF16", u16, ffn.up_out.ptr, ups[0 .. rows * width]);
            try r.sameRows("dense FFN activation", u16, ffn.act_out.ptr, acts[0 .. rows * width]);
            try r.sameRows("dense FFN down FP32", f32, out, outs[0 .. rows * hidden]);
        }
    }
    for (0..most) |row| {
        for (0..width) |col| {
            var mag: f64 = 0;
            const g = dot(x[row * hidden ..][0..hidden], gw[col * hidden ..][0..hidden], &mag);
            try r.within("dense FFN gate BF16", bf(gates[row * width + col]), g, rig.ulps(g, 0.5) + sumTol(mag, hidden));
            const u = dot(x[row * hidden ..][0..hidden], uw[col * hidden ..][0..hidden], &mag);
            try r.within("dense FFN up BF16", bf(ups[row * width + col]), u, rig.ulps(u, 0.5) + sumTol(mag, hidden));
            const gv = bf(gates[row * width + col]);
            const want = rbf(gv / (1 + @exp(-gv)) * bf(ups[row * width + col]));
            try r.within("dense FFN activation", bf(acts[row * width + col]), want, rig.ulps(want, 1));
        }
        for (0..hidden) |col| {
            var mag: f64 = 0;
            const sum = dot(acts[row * width ..][0..width], dw[col * width ..][0..width], &mag);
            try r.within("dense FFN down FP32", outs[row * hidden + col], sum, sumTol(mag, width));
        }
    }
    r.report("dense FFN gate, up, SwiGLU and down against FP64");
}
