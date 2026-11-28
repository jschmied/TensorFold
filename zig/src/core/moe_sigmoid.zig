//! A sigmoid + bias MoE layer on Metal at runtime shapes: router, picks, expert groups, gate/up, down, combine.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const rows_ = @import("block_rows.zig");

const Ref = rows_.Ref;

/// The route, as cuda/moe_glue.zig's Route: rank logit + bias (or sigmoid + bias), groups, normalise, scale.
pub const Route = struct {
    experts: usize,
    topk: usize,
    shared: bool,
    score: bool = false,
    normalize: bool = false,
    scale: f32 = 1,
    groups: usize = 1,
    keep: usize = 1,

    pub fn slots(r: Route) usize {
        return r.topk + @intFromBool(r.shared);
    }

    pub fn check(r: Route) !void {
        if (r.experts == 0 or r.experts > 4096 or r.topk == 0 or r.topk > r.experts or r.groups == 0 or r.groups > 32 or
            r.experts % r.groups != 0 or r.keep == 0 or r.keep > r.groups or (r.groups > 1 and r.topk > r.keep * (r.experts / r.groups)) or
            !std.math.isFinite(r.scale)) return error.UnsupportedRoute;
    }
};

pub const Args = extern struct { rows: u32, experts: u32, topk: u32, shared: u32, ldy: u32, score: u32, normalize: u32, groups: u32, keep: u32, scale: f32 };

/// The kernels' arguments for `rows` rows of y stride `ldy` (the contract binds them too).
pub fn argsOf(r: Route, rows: usize, ldy: usize) Args {
    return .{ .rows = @intCast(rows), .experts = @intCast(r.experts), .topk = @intCast(r.topk), .shared = @intFromBool(r.shared), .ldy = @intCast(ldy), .score = @intFromBool(r.score), .normalize = @intFromBool(r.normalize), .groups = @intCast(r.groups), .keep = @intCast(r.keep), .scale = r.scale };
}

pub fn source(a: std.mem.Allocator, ctl: u8) ![]u8 {
    return std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, ks.core_moe_sigmoid });
}

pub const Pipes = struct {
    select: mtl.Pipeline,
    plan: mtl.Pipeline,
    combine: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !Pipes {
        const text = try source(a, ctl);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const select = try mtl.Pipeline.init(device, lib, "tf_moe_select", false);
        errdefer select.deinit();
        const plan = try mtl.Pipeline.init(device, lib, "tf_moe_plan", false);
        errdefer plan.deinit();
        const combine = try mtl.Pipeline.init(device, lib, "tf_moe_combine", false);
        return .{ .select = select, .plan = plan, .combine = combine };
    }

    pub fn deinit(p: Pipes) void {
        p.select.deinit();
        p.plan.deinit();
        p.combine.deinit();
    }
};

/// One layer's experts, the shared one last: `at` [E(+1)] byte offsets of each one's tensors into `w`.
pub const Experts = struct { w: Ref, at: Ref, width: usize, bf16: bool = false };

/// Scratch a layer of `rows` rows takes, as views of caller buffers (sizes from `bytes`).
pub const Scratch = struct { logits: Ref, pick: Ref, wts: Ref, bucket: Ref, members: Ref, items: Ref, count: Ref, act: Ref, y: Ref };

pub const Sizes = struct { logits: usize, pick: usize, wts: usize, bucket: usize, members: usize, items: usize, count: usize, act: usize, y: usize };

/// Bytes of each scratch view for `rows` rows at hidden `d` and expert width `width`.
pub fn bytes(r: Route, rows: usize, d: usize, width: usize) Sizes {
    const pairs = rows * r.slots();
    return .{ .logits = rows * r.experts * 4, .pick = pairs * 4, .wts = pairs * 4, .bucket = (r.experts + 1) * 4, .members = pairs * 4, .items = rows_.maxItems(pairs, r.experts + 1) * 12, .count = 4, .act = pairs * width * 2, .y = pairs * d * 4 };
}

/// out [rows, d] bf16 = the layer's MoE of x [rows, d] bf16; router [E, d] bf16 rows, bias [E] fp32.
pub fn encode(e: mtl.ComputeEncoder, p: Pipes, br: rows_.Pipes, r: Route, x: Ref, d: usize, rows: usize, router: Ref, bias: Ref, ex: Experts, s: Scratch, out: Ref) !void {
    try r.check();
    if (rows == 0) return error.EmptyRows;
    const a = argsOf(r, rows, d);
    var at: usize = 0;
    while (at < rows) : (at += 16) { // the router's 16-row launches
        const n = @min(16, rows - at);
        try br.dense(e, .{ .buf = x.buf, .off = x.off + at * d * 2 }, router, router, .{ .buf = s.logits.buf, .off = s.logits.off + at * r.experts * 4 }, .{ .rows = @intCast(n), .k = @intCast(d), .n = @intCast(r.experts), .ldx = @intCast(d), .ldy = @intCast(r.experts), .bf16 = 1 });
    }
    e.barrier();
    e.setPipeline(p.select);
    for ([_]Ref{ s.logits, bias }, 0..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.setValue(a, 2);
    e.setBuffer(s.pick.buf, s.pick.off, 3);
    e.setBuffer(s.wts.buf, s.wts.off, 4);
    e.dispatchThreads(mtl.Size.of(rows, 1, 1), mtl.Size.of(@min(rows, 64), 1, 1));
    e.barrier();
    e.setPipeline(p.plan);
    e.setBuffer(s.pick.buf, s.pick.off, 0);
    e.setValue(a, 1);
    for ([_]Ref{ s.bucket, s.members, s.items, s.count }, 2..) |b, i| e.setBuffer(b.buf, b.off, i);
    e.dispatchThreads(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    e.barrier();
    const plan: rows_.Pipes.Plan = .{ .items = s.items, .members = s.members, .count = s.count, .most = rows_.maxItems(rows * r.slots(), r.experts + 1) };
    const up: rows_.PickArgs = .{ .k = @intCast(d), .n = @intCast(ex.width), .ldx = @intCast(d), .slots = @intCast(r.slots()), .bf16 = @intFromBool(ex.bf16) };
    try br.gateUp(e, x, ex.w, ex.at, plan, s.act, up);
    e.barrier();
    var dn = up;
    dn.k = @intCast(ex.width);
    dn.n = @intCast(d);
    try br.down(e, s.act, ex.w, ex.at, plan, s.y, dn);
    e.barrier();
    e.setPipeline(p.combine);
    e.setBuffer(s.y.buf, s.y.off, 0);
    e.setBuffer(s.wts.buf, s.wts.off, 1);
    e.setValue(a, 2);
    e.setBuffer(out.buf, out.off, 3);
    e.dispatchThreads(mtl.Size.of(d, rows, 1), mtl.Size.of(256, 1, 1));
}

test "MoE route: Kolibri's and Ling's shapes pass, others are refused before a launch" {
    try (Route{ .experts = 384, .topk = 6, .shared = true }).check();
    try (Route{ .experts = 128, .topk = 8, .shared = false, .score = true, .normalize = true, .scale = 2.5, .groups = 8, .keep = 4 }).check();
    try std.testing.expectError(error.UnsupportedRoute, (Route{ .experts = 8, .topk = 9, .shared = true }).check());
    try std.testing.expectError(error.UnsupportedRoute, (Route{ .experts = 10, .topk = 2, .shared = true, .groups = 3 }).check());
    try std.testing.expectEqual(@as(usize, 7), (Route{ .experts = 384, .topk = 6, .shared = true }).slots());
}
