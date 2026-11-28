//! Row glue on Metal at runtime widths (kernels/metal/core/row_glue.metal): embed, RMS norms, q/k norms with RoPE.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

pub const Ref = @import("block_rows.zig").Ref;
pub const Args = extern struct { rows: u32, width: u32, eps: f32 };
pub const QkArgs = extern struct { rows: u32, heads: u32, kv_heads: u32, dim: u32, rope: u32, ring: u32, eps: f32 };

pub fn source(a: std.mem.Allocator, ctl: u8) ![]u8 {
    return std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, ks.core_row_glue });
}

/// Refuses head layouts the q/k kernel's lanes cannot hold: dim a multiple of 64 up to 256.
pub fn checkHeads(q: QkArgs) !void {
    if (q.dim == 0 or q.dim % 64 != 0 or q.dim > 256 or q.heads == 0 or q.kv_heads == 0 or q.heads % q.kv_heads != 0)
        return error.UnsupportedHeads;
}

pub const Pipes = struct {
    embed_pipe: mtl.Pipeline,
    rms_pipe: mtl.Pipeline,
    add_rms_pipe: mtl.Pipeline,
    qk_pipe: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !Pipes {
        const text = try source(a, ctl);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var out: Pipes = undefined;
        var made: usize = 0;
        const names = [_][:0]const u8{ "tf_glue_embed", "tf_glue_rms", "tf_glue_add_rms", "tf_glue_qk" };
        const slots = [_]*mtl.Pipeline{ &out.embed_pipe, &out.rms_pipe, &out.add_rms_pipe, &out.qk_pipe };
        errdefer for (slots[0..made]) |s| s.deinit();
        for (names, slots) |n, s| {
            s.* = try mtl.Pipeline.init(device, lib, n, false);
            made += 1;
        }
        if (out.qk_pipe.simdWidth() != 32) return error.UnsupportedGluePipeline;
        return out;
    }

    pub fn deinit(p: Pipes) void {
        for ([_]mtl.Pipeline{ p.embed_pipe, p.rms_pipe, p.add_rms_pipe, p.qk_pipe }) |q| q.deinit();
    }

    /// res [rows, width] fp32 = table [V, width] bf16 rows at ids [rows] u32.
    pub fn embed(p: Pipes, e: mtl.ComputeEncoder, ids: Ref, table: Ref, res: Ref, a: Args) void {
        e.setPipeline(p.embed_pipe);
        e.setBuffer(ids.buf, ids.off, 0);
        e.setBuffer(table.buf, table.off, 1);
        e.setValue(a, 2);
        e.setBuffer(res.buf, res.off, 3);
        e.dispatchThreads(mtl.Size.of(a.width, a.rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// out bf16 = rms(x fp32) * w.
    pub fn rmsNorm(p: Pipes, e: mtl.ComputeEncoder, x: Ref, w: Ref, out: Ref, a: Args) void {
        e.setPipeline(p.rms_pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setValue(a, 2);
        e.setBuffer(out.buf, out.off, 3);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// res += rms(x bf16) * w1 in place, then out bf16 = rms(res) * w2.
    pub fn addRms(p: Pipes, e: mtl.ComputeEncoder, x: Ref, res: Ref, w1: Ref, w2: Ref, out: Ref, a: Args) void {
        e.setPipeline(p.add_rms_pipe);
        for ([_]Ref{ x, res, w1, w2 }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(a, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(a.rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    /// The q/k/v rows of qkv: q normed (and RoPE'd) into q, k likewise and v as is into the cache at pos (% ring).
    pub const Qk = struct { qkv: Ref, q_norm: Ref, k_norm: Ref, cos: Ref, sin: Ref, pos: Ref, q: Ref, k_cache: Ref, v_cache: Ref };

    pub fn qk(p: Pipes, e: mtl.ComputeEncoder, b: Qk, a: QkArgs) !void {
        try checkHeads(a);
        e.setPipeline(p.qk_pipe);
        for ([_]Ref{ b.qkv, b.q_norm, b.k_norm, b.cos, b.sin, b.pos }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(a, 6);
        for ([_]Ref{ b.q, b.k_cache, b.v_cache }, 7..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.dispatchGroups(mtl.Size.of(a.heads + 2 * a.kv_heads, a.rows, 1), mtl.Size.of(32, 1, 1));
    }
};

test "row glue heads: Kolibri's 48 / 4 at dim 128 pass, odd layouts are refused" {
    try checkHeads(.{ .rows = 1, .heads = 48, .kv_heads = 4, .dim = 128, .rope = 1, .ring = 1088, .eps = 1e-6 });
    try std.testing.expectError(error.UnsupportedHeads, checkHeads(.{ .rows = 1, .heads = 48, .kv_heads = 5, .dim = 128, .rope = 1, .ring = 0, .eps = 1e-6 }));
    try std.testing.expectError(error.UnsupportedHeads, checkHeads(.{ .rows = 1, .heads = 4, .kv_heads = 1, .dim = 96, .rope = 1, .ring = 0, .eps = 1e-6 }));
}
