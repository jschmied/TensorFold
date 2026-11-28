//! Products over 128-column blocks on Metal at runtime shapes (block-FP8 or bf16 weights): dense rows, expert pairs.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Out = enum { f32, bf16 };

pub const Args = extern struct { rows: u32, k: u32, n: u32, ldx: u32, ldy: u32, bf16: u32 = 0 };
pub const PickArgs = extern struct { k: u32, n: u32, ldx: u32, slots: u32, bf16: u32 = 0 };
/// Byte offsets into one weights buffer of an expert's tensors (the checkpoint's own, mapped): kernels' TfExpertAt.
pub const ExpertAt = extern struct { gate: u64, gate_s: u64, up: u64, up_s: u64, down: u64, down_s: u64 };

/// The library's text: `ctl` plants a contract control (0: none), `out` the dense rows' output type.
pub fn source(a: std.mem.Allocator, out: Out, ctl: u8) ![]u8 {
    return std.fmt.allocPrint(a, "#define TF_CTL {d}\n#define TF_OUT {s}\n{s}", .{ ctl, if (out == .f32) "float" else "bfloat", ks.core_block_rows });
}

/// Refuses what the kernels' block walk cannot take: K in whole 128 blocks, 1-16 rows, no zero width.
pub fn check(rows: usize, k: usize, n: usize) !void {
    if (rows == 0 or rows > 16 or k == 0 or k % 128 != 0 or n == 0 or k > std.math.maxInt(u32) or n > std.math.maxInt(u32))
        return error.UnsupportedRowsShape;
}

pub const Pipes = struct {
    dense_pipe: mtl.Pipeline,
    glu_pipe: mtl.Pipeline,
    down_pipe: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, out: Out, ctl: u8) !Pipes {
        const text = try source(a, out, ctl);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const dense_p = try mtl.Pipeline.init(device, lib, "tf_block_rows", false);
        errdefer dense_p.deinit();
        const glu_p = try mtl.Pipeline.init(device, lib, "tf_block_glu", false);
        errdefer glu_p.deinit();
        const down_p = try mtl.Pipeline.init(device, lib, "tf_block_down", false);
        errdefer down_p.deinit();
        if (dense_p.simdWidth() != 32 or glu_p.simdWidth() != 32 or down_p.simdWidth() != 32) return error.UnsupportedRowsPipeline;
        return .{ .dense_pipe = dense_p, .glu_pipe = glu_p, .down_pipe = down_p };
    }

    pub fn deinit(p: Pipes) void {
        p.dense_pipe.deinit();
        p.glu_pipe.deinit();
        p.down_pipe.deinit();
    }

    /// y [rows, n] (stride ldy) = x [rows, k] (stride ldx) times w [n, k]: codes with scales s [n/128, k/128], or bf16.
    pub fn dense(p: Pipes, e: mtl.ComputeEncoder, x: Ref, w: Ref, s: Ref, y: Ref, a: Args) !void {
        try check(a.rows, a.k, a.n);
        e.setPipeline(p.dense_pipe);
        for ([_]Ref{ x, w, s }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(a, 3);
        e.setBuffer(y.buf, y.off, 4);
        e.dispatchGroups(mtl.Size.of((a.n + 3) / 4, 1, 1), mtl.Size.of(128, 1, 1));
    }

    /// The plan's groups: (pair, expert) items of at most 16 pairs, and how many the plan wrote.
    pub const Plan = struct { items: Ref, members: Ref, count: Ref, most: usize };

    /// act [pairs, n] = bf16(silu(x . gate) * (x . up)) at each pair's expert; `at` [experts] offsets into `w`.
    pub fn gateUp(p: Pipes, e: mtl.ComputeEncoder, x: Ref, w: Ref, at: Ref, plan: Plan, act: Ref, a: PickArgs) !void {
        try p.pick(e, p.glu_pipe, x, w, at, plan, act, a);
    }

    /// y [pairs, n] fp32 = act [pairs, k] times each pair's expert's down weights.
    pub fn down(p: Pipes, e: mtl.ComputeEncoder, act: Ref, w: Ref, at: Ref, plan: Plan, y: Ref, a: PickArgs) !void {
        try p.pick(e, p.down_pipe, act, w, at, plan, y, a);
    }

    fn pick(_: Pipes, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, x: Ref, w: Ref, at: Ref, plan: Plan, out: Ref, a: PickArgs) !void {
        try check(1, a.k, a.n);
        e.setPipeline(pipe);
        for ([_]Ref{ x, w, at, plan.items, plan.members, plan.count }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(a, 6);
        e.setBuffer(out.buf, out.off, 7);
        e.dispatchGroups(mtl.Size.of((a.n + 3) / 4, plan.most, 1), mtl.Size.of(128, 1, 1));
    }
};

/// Items a plan of `pairs` pairs over `groups` experts can write at most: every group's partial item, then sixteens.
pub fn maxItems(pairs: usize, groups: usize) usize {
    return pairs / 16 + @min(pairs, groups);
}

test "block rows shapes: whole 128 blocks, 1-16 rows; item bound holds for any grouping" {
    try check(16, 2560, 7168);
    try std.testing.expectError(error.UnsupportedRowsShape, check(17, 2560, 1));
    try std.testing.expectError(error.UnsupportedRowsShape, check(1, 2500, 1));
    try std.testing.expectError(error.UnsupportedRowsShape, check(1, 128, 0));
    // 7 pairs in 7 groups: 7 items; 33 in one group: 3 items (16, 16, 1)
    try std.testing.expect(maxItems(7, 385) >= 7);
    try std.testing.expect(maxItems(33, 1) >= 3);
    try std.testing.expect(maxItems(3640, 9) >= 3640 / 16 + 9);
}
