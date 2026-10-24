//! The dense feed-forward node, provisional (no released Qwen4 checkpoint): gate and up, the dense lane's SwiGLU, down.
const std = @import("std");
const cuda = @import("cuda");
const projection = @import("cuda_projection.zig");

/// The qwen3_5 dense lane's tf_q35_mlp_act: out = bf16(silu(gate) * up), four values a thread.
pub const Act = struct {
    mod: cuda.Module,
    f: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Act {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.qwen35);
        errdefer mod.unload();
        return .{ .mod = mod, .f = try mod.function("tf_q35_mlp_act") };
    }

    pub fn unload(a: *Act) void {
        a.mod.unload();
    }
};

/// Gate, up and down as native lane projections, and the work buffers a launch of up to `cap` rows needs.
pub const Ffn = struct {
    gate: projection.Projection,
    up: projection.Projection,
    down: projection.Projection,
    act: cuda.Function,
    width: usize,
    cap: usize,
    gate_out: cuda.DeviceBuffer,
    up_out: cuda.DeviceBuffer,
    act_out: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,

    /// Takes the three projections (and frees them with the node); `act` is Act.f, whose module the caller keeps.
    pub fn init(d: *const cuda.Driver, lane: *const cuda.qmmf.Lane, act: cuda.Function, gate: projection.Projection, up: projection.Projection, down: projection.Projection, cap: usize) !Ffn {
        const g = gate.native orelse return error.UnexpectedTensor;
        const u = up.native orelse return error.UnexpectedTensor;
        const w = down.native orelse return error.UnexpectedTensor;
        const width = g.n;
        const hidden = g.k;
        if (cap == 0 or width == 0 or width % 64 != 0 or u.n != width or u.k != hidden or w.n != hidden or w.k != width) return error.UnexpectedTensor;
        var part: usize = 0;
        for (1..cap + 1) |m| part = @max(part, lane.partBytes(m, g), lane.partBytes(m, u), lane.partBytes(m, w));
        var gate_out = try cuda.DeviceBuffer.alloc(d, cap * width * 2);
        errdefer gate_out.free();
        var up_out = try cuda.DeviceBuffer.alloc(d, cap * width * 2);
        errdefer up_out.free();
        var act_out = try cuda.DeviceBuffer.alloc(d, cap * width * 2);
        errdefer act_out.free();
        const scratch = try cuda.DeviceBuffer.alloc(d, @max(part, 16));
        return .{ .gate = gate, .up = up, .down = down, .act = act, .width = width, .cap = cap, .gate_out = gate_out, .up_out = up_out, .act_out = act_out, .part = scratch };
    }

    pub fn free(self: *Ffn) void {
        for ([_]*cuda.DeviceBuffer{ &self.gate_out, &self.up_out, &self.act_out, &self.part }) |b| b.free();
        for ([_]*projection.Projection{ &self.gate, &self.up, &self.down }) |p| p.free();
    }

    /// out [rows, hidden] fp32 = down(silu(gate x) * up x) over x [rows, hidden] bf16 (rows `ldx` apart).
    pub fn run(self: *const Ffn, lane: *const cuda.qmmf.Lane, s: cuda.Stream, x: u64, ldx: usize, rows: usize, out: u64) !void {
        if (rows == 0 or rows > self.cap) return error.UnexpectedTensor;
        try lane.matmul(s, x, ldx, rows, self.gate.native.?, self.gate_out.ptr, self.part.ptr);
        try lane.matmul(s, x, ldx, rows, self.up.native.?, self.up_out.ptr, self.part.ptr);
        const total = rows * self.width;
        var a: cuda.Args = .{};
        a.add(self.gate_out.ptr);
        a.add(self.up_out.ptr);
        a.add(self.act_out.ptr);
        a.add(@as(c_longlong, @intCast(total)));
        a.add(@as(u64, 0)); // no row-count pointer
        a.add(@as(c_int, @intCast(self.width)));
        a.add(@as(u64, 0)); // no group sums
        try cuda.launch.launch(self.act, .{ .grid = .{ .x = @intCast((total + 1023) / 1024) }, .block = .{ .x = 256 } }, s, &a);
        try lane.matmulF32(s, self.act_out.ptr, self.width, rows, self.down.native.?, out, self.part.ptr);
    }
};
