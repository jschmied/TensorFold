//! Grouped block-FP8 experts (fp8/experts.cu) on grouped.zig's plan: the Python packing, kernel and grid choice.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const cols = 32; // output columns a warp
pub const block = 128; // the checkpoint's scale block, both ways
pub const prompt_tile = 64; // pairs a prompt item holds

/// e4m3 [n, k] of one expert as int32 [n/32][k/32][32][2][4]: lane (gq, t)'s half h, tile j (fp8/experts.py pack).
pub fn packOne(out: []u32, w: []const u8, n: usize, k: usize) void {
    std.debug.assert(n % block == 0 and k % block == 0 and out.len == n * k / 4 and w.len == n * k);
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    var i: usize = 0;
    for (0..n / cols) |cb| for (0..k / 32) |g| for (0..8) |gq| for (0..4) |t| for (0..2) |h| for (0..4) |j| {
        const row = cb * cols + j * 8 + gq;
        out[i] = words[row * (k / 4) + g * 8 + h * 4 + t];
        i += 1;
    };
}

/// Gate and up interleaved per (column block, k32 group) as make() stacks them: [n/32][k/32][2][32][2][4].
pub fn packGateUp(out: []u32, gate: []const u8, up: []const u8, n: usize, k: usize) !void {
    const half = n * k / 4;
    std.debug.assert(out.len == 2 * half);
    const gpa = std.heap.page_allocator;
    const g = try gpa.alloc(u32, half);
    defer gpa.free(g);
    const u = try gpa.alloc(u32, half);
    defer gpa.free(u);
    packOne(g, gate, n, k);
    packOne(u, up, n, k);
    const unit = 32 * 2 * 4; // a lane block of one k32 group
    for (0..half / unit) |b| {
        @memcpy(out[(2 * b) * unit ..][0..unit], g[b * unit ..][0..unit]);
        @memcpy(out[(2 * b + 1) * unit ..][0..unit], u[b * unit ..][0..unit]);
    }
}

pub const symbols = struct {
    pub const gate_up = "_ZN14tf_fp8_experts17fp8_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const down_f32 = "_ZN14tf_fp8_experts17fp8_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const down_bf16 = "_ZN14tf_fp8_experts17fp8_expert_kernelILi1ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const p_gate_up = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi2ELi2ELi2ELi2EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const p_down_f32 = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi1ELi0ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const p_down_bf16 = "_ZN14tf_fp8_experts24fp8_expert_prompt_kernelILi1ELi3ELi4ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
};

/// One layer's experts on the device (the shared one last), as make() lays them out.
pub const Layer = struct {
    up: u64, // [E][NI/32][D/32][2][32][2][4] int32
    down: u64, // [E][D/32][NI/32][1][32][2][4]
    up_scale: u64, // [E][2][NI/128][D/128] fp32 (gate, up)
    down_scale: u64, // [E][1][D/128][NI/128]
    width: usize, // NI
    dims: usize, // D
    experts: usize,
    limit: f32 = 0.0,
};

/// Which epilogue a call takes (fp8_experts_cuda's epi): SwiGLU gate-up, down to fp32 or to bf16.
pub const Epi = enum { gate_up, down_f32, down_bf16 };

pub const Experts = struct {
    mod: Module,
    fns: [2][3]Function, // decode, prompt; by Epi
    resident: [2][3]usize, // blocks the grid may hold: per SM times SMs

    pub fn load(d: *const Driver, sms: usize) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.fp8_experts);
        errdefer mod.unload();
        var e: Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        const names = [2][3][:0]const u8{ .{ symbols.gate_up, symbols.down_f32, symbols.down_bf16 }, .{ symbols.p_gate_up, symbols.p_down_f32, symbols.p_down_bf16 } };
        for (names, 0..) |set, i| for (set, 0..) |name, j| {
            e.fns[i][j] = try mod.function(name);
            e.resident[i][j] = @max(1, try e.fns[i][j].occupancy(128, 0)) * sms;
        };
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// One call of fp8/experts.py's _run: gate_up reads token rows (`slots` pairs a row), down reads pair rows.
    pub fn run(e: Experts, s: Stream, epi: Epi, x: u64, x_stride: usize, rows: usize, slots: usize, l: Layer, p: *const grouped.Plan, prompt: bool, out: u64, skip: i32) !void {
        const gate = epi == .gate_up;
        const w = if (gate) l.up else l.down;
        const scale = if (gate) l.up_scale else l.down_scale;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;
        const n = if (gate) l.width else l.dims;
        const t: usize = if (prompt) prompt_tile else grouped.tile;
        const units = grouped.maxItems(rows * slots, l.experts, t) * nb; // the plan's pairs, both calls
        const pi: usize = @intFromBool(prompt);
        const ei: usize = @backingInt(epi);
        const grid: usize = if (prompt) blk: {
            const cw: usize = if (gate) 2 else 4;
            break :blk @min(units / nb * (nb / cw), e.resident[pi][ei]);
        } else @min((units + 3) / 4, e.resident[pi][ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_stride)));
        a.add(@as(i32, @intCast(if (gate) slots else 0)));
        a.add(w);
        a.add(scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        a.add(p.items.ptr);
        a.add(p.counts.ptr);
        a.add(p.members.ptr);
        a.add(out);
        a.add(@as(i32, @intCast(n)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        try launch_.launch(e.fns[pi][ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
};

test "a packed expert puts each word where fp8/experts.py's pack does" {
    const gpa = std.testing.allocator;
    const n = 128;
    const k = 128;
    const w = try gpa.alloc(u8, n * k);
    defer gpa.free(w);
    for (w, 0..) |*b, i| b.* = @truncate(i * 31 + 7);
    const out = try gpa.alloc(u32, n * k / 4);
    defer gpa.free(out);
    packOne(out, w, n, k);
    // (cb 1, g 2, gq 5, t 3, h 1, j 2): row 32 + 16 + 5 = 53, word 2 * 8 + 4 + 3 = 23
    const idx = (((((1 * 4 + 2) * 8 + 5) * 4 + 3) * 2 + 1) * 4) + 2;
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, w));
    try std.testing.expectEqual(words[53 * (k / 4) + 23], out[idx]);
}
