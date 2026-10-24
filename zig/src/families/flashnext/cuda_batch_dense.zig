//! The shared pass's dense feed-forward (provisional): the node's fp32 branch is what the next write-back adds.
const batch = @import("cuda_batch.zig");
const dense = @import("cuda_dense_ffn.zig");
const dims = @import("cuda_prompt_wt.zig").dims;

/// After a layer's mixed read-out: the dense branch into moe_f32, which the write-back rounds once as it does a MoE's.
pub fn block(self: anytype, ffn: *const dense.Ffn, layer: usize, keep: batch.LayerKeep) !void {
    if (self.addAt(layer) != null or keep.moe_branch != 0 or keep.moe_x != 0 or keep.moe_ids != 0 or keep.moe_wts != 0) return error.UnexpectedTensor;
    const s = self.s;
    try ffn.run(&self.k.native_lane, self.stream, s.mixed, dims, self.rows, s.moe_f32);
    self.pending_f32 = true;
}
