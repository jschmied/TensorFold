//! Grouped NVFP4 experts (nvfp4/experts.cu) on grouped.zig's plan: the Python packing, kernel and grid choice.
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

pub const cols = 32; // output columns a block
pub const words = 144; // int32 a (32 columns, 32 inputs) block: 128 code words, then 16 of e4m3 scales
pub const tile = grouped.tile; // prompt items hold 16 pairs too (PREFILL_TILE): 64 ran slower

/// Where input 16h + 4t + q of a lane's word sits: nibble 2h + q/2 + 4(q%2) (fp4pair's field order).
fn slot(h: usize, q: usize) u5 {
    return @intCast(4 * (2 * h + q / 2 + 4 * (q % 2)));
}

/// One expert's words [n, k/2] (low nibble first) and e4m3 scales [n, k/16] as blocks [n/32][k/32][144] (_pack).
pub fn packOne(out: []u32, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
    const kg = k / 32;
    std.debug.assert(n % cols == 0 and k % 32 == 0 and out.len == n / cols * kg * words);
    std.debug.assert(codes.len == n * k / 2 and scales.len == n * k / 16);
    for (0..n / cols) |cb| for (0..kg) |g| {
        const block = out[(cb * kg + g) * words ..][0..words];
        for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
            const row = cb * cols + j * 8 + gq;
            var w: u32 = 0;
            for (0..2) |h| for (0..4) |q| {
                const input = g * 32 + h * 16 + t * 4 + q;
                const byte = codes[row * (k / 2) + input / 2];
                const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
                w |= nib << slot(h, q);
            };
            block[(gq * 4 + t) * 4 + j] = w;
        };
        const sc = std.mem.sliceAsBytes(block[128..]);
        for (0..4) |t| for (0..2) |h| for (0..4) |j| for (0..2) |c| {
            const row = cb * cols + j * 8 + t * 2 + c;
            sc[((t * 2 + h) * 4 + j) * 2 + c] = scales[row * (k / 16) + g * 2 + h];
        };
    };
}

/// Gate and up interleaved per (column block, k32 group) as make() stacks them: [n/32][k/32][2][144].
pub fn packGateUp(out: []u32, gate: [2][]const u8, up: [2][]const u8, n: usize, k: usize) !void {
    const half = n / cols * (k / 32) * words;
    std.debug.assert(out.len == 2 * half);
    const gpa = std.heap.page_allocator;
    const g = try gpa.alloc(u32, half);
    defer gpa.free(g);
    const u = try gpa.alloc(u32, half);
    defer gpa.free(u);
    packOne(g, gate[0], gate[1], n, k);
    packOne(u, up[0], up[1], n, k);
    for (0..half / words) |b| {
        @memcpy(out[(2 * b) * words ..][0..words], g[b * words ..][0..words]);
        @memcpy(out[(2 * b + 1) * words ..][0..words], u[b * words ..][0..words]);
    }
}

pub const symbols = struct {
    pub const gate_up = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const down_f32 = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    pub const down_bf16 = "_ZN16tf_nvfp4_experts19nvfp4_expert_kernelILi1ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
};

/// One layer's experts on the device, as make() lays them out.
pub const Layer = struct {
    up: u64, // [E][NI/32][D/32][2][144] int32
    down: u64, // [E][D/32][NI/32][1][144]
    up_scale: u64, // [E][2] fp32 (gate, up)
    down_scale: u64, // [E][1]
    width: usize, // NI
    dims: usize, // D
    experts: usize,
    limit: f32 = 0.0,
};

/// Which epilogue a call takes (nvfp4_experts_cuda's epi): SwiGLU gate-up, down to fp32 or to bf16.
pub const Epi = enum { gate_up, down_f32, down_bf16 };

pub const Experts = struct {
    mod: Module,
    fns: [3]Function, // by Epi
    resident: [3]usize, // blocks the grid may hold: per SM times SMs

    pub fn load(d: *const Driver, sms: usize) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.nvfp4_experts);
        errdefer mod.unload();
        var e: Experts = .{ .mod = mod, .fns = undefined, .resident = undefined };
        for ([_][:0]const u8{ symbols.gate_up, symbols.down_f32, symbols.down_bf16 }, 0..) |name, i| {
            e.fns[i] = try mod.function(name);
            e.resident[i] = @max(1, try e.fns[i].occupancy(128, 0)) * sms;
        }
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// One call of nvfp4/experts.py's _run: gate_up reads token rows (`slots` pairs a row), down reads pair rows.
    pub fn run(e: Experts, s: Stream, epi: Epi, x: u64, x_stride: usize, rows: usize, slots: usize, l: Layer, p: *const grouped.Plan, out: u64, skip: i32) !void {
        const gate = epi == .gate_up;
        const kg = (if (gate) l.dims else l.width) / 32;
        const nb = (if (gate) l.width else l.dims) / cols;
        const units = grouped.maxItems(rows * slots, l.experts, tile) * nb;
        const ei: usize = @backingInt(epi);
        const grid = @min((units + 3) / 4, e.resident[ei]);
        if (grid < 1) return;
        var a: launch_.Args = .{};
        a.add(x);
        a.add(@as(i32, @intCast(x_stride)));
        a.add(@as(i32, @intCast(if (gate) slots else 0)));
        a.add(if (gate) l.up else l.down);
        a.add(if (gate) l.up_scale else l.down_scale);
        a.add(@as(i32, @intCast(kg)));
        a.add(@as(i32, @intCast(nb)));
        a.add(p.items.ptr);
        a.add(p.counts.ptr);
        a.add(p.members.ptr);
        a.add(out);
        a.add(@as(i32, @intCast(if (gate) l.width else l.dims)));
        a.add(if (gate) l.limit else @as(f32, 0.0));
        a.add(skip);
        try launch_.launch(e.fns[ei], .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }
};

test "a packed block puts each nibble and scale where nvfp4/experts.py's _pack does" {
    const gpa = std.testing.allocator;
    const n = 64;
    const k = 64;
    const codes = try gpa.alloc(u8, n * k / 2);
    defer gpa.free(codes);
    for (codes, 0..) |*c, i| c.* = @truncate(i * 29 + 3);
    var scales: [n * k / 16]u8 = undefined;
    for (&scales, 0..) |*c, i| c.* = @truncate(i * 7 + 1);
    var out: [n / cols * (k / 32) * words]u32 = undefined;
    packOne(&out, codes, &scales, n, k);
    // block (cb 1, g 1), lane gq 3, t 2, tile j 1: row 32 + 8 + 3 = 43; input 32 + 16 + 8 + 3 (h 1, q 3) -> slot 7
    const w = out[(1 * 2 + 1) * words + (3 * 4 + 2) * 4 + 1];
    try std.testing.expectEqual(@as(u32, codes[43 * 32 + 59 / 2] >> 4), (w >> slot(1, 3)) & 0xF);
    // its scale byte (t 2, h 1, j 1, c 0): row 32 + 8 + 4 = 44, k16 group 3
    const sc = std.mem.sliceAsBytes(out[(1 * 2 + 1) * words + 128 ..][0..16]);
    try std.testing.expectEqual(scales[44 * 4 + 3], sc[((2 * 2 + 1) * 4 + 1) * 2]);
}
