//! Synthetic records at a provisional dense Qwen4 shape (hidden 5120, HC 4 x 5120 rank 640, Q24/KV4, FFN 14336).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const swiglu_k = @import("contract_qwen4_swiglu.zig");
const qsa_k = @import("contract_qwen4_qsa.zig");
const hc_k = @import("contract_qwen4_hc.zig");
const flash = @import("flashnext");

pub const Group = struct {
    swiglu: swiglu_k.Swiglu,
    qsa: [qsa_k.forms.len]qsa_k.Qsa,
    hc: [hc_k.forms.len]hc_k.Hc,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, q35: cuda.Module, glue: *const flash.Glue, hcf: cuda.Module) !Group {
        var group: Group = .{ .swiglu = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try q35.function("tf_q35_mlp_act") }, .qsa = undefined, .hc = undefined };
        for (qsa_k.forms, &group.qsa) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f, .glue = glue };
        const tk = flash.tile_kernels;
        const down = try hcf.function(tk.cubin_hcf_down);
        const up = try hcf.function(tk.cubin_hcf_up);
        try down.allowDynamicShared(tk.hcf_down_smem);
        try up.allowDynamicShared(tk.hcf_up_smem);
        for (hc_k.forms, &group.hc) |f, *x| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .form = f[0], .control = f[1], .glue = glue, .down_fn = down, .up_fn = up };
        return group;
    }

    pub fn append(group: *Group, a: std.mem.Allocator, kernels: *std.ArrayList(core.contract.gates.Kernel)) !void {
        try kernels.append(a, group.swiglu.kernel());
        for (&group.qsa) |*x| try kernels.append(a, x.kernel());
        for (&group.hc) |*x| try kernels.append(a, x.kernel());
    }

    pub fn deinit(group: *Group) void {
        group.swiglu.deinit();
        for (&group.qsa) |*x| x.deinit();
        for (&group.hc) |*x| x.deinit();
    }
};
