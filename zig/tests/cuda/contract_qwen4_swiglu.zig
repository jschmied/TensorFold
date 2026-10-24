//! The dense lane's SwiGLU, tf_q35_mlp_act, at a provisional Qwen4 dense FFN width, against float64's silu(gate) * up.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

/// The provisional dense width (Intel's kernels imply 14336; no checkpoint states it).
pub const width = 14336;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// expf, the division and the product are fp32 steps after no sum; the output is the one bf16 rounding.
pub fn spec() c.Spec {
    return .{
        .name = "cuda qwen4 synthetic dense SwiGLU tf_q35_mlp_act (FFN 14336, provisional)",
        .accum = .none,
        .depth = noSums,
        .ops = 8,
        .out = .bf16,
        // rows = token rows, n = the FFN width; the kernel walks gate and up four values a thread
        .tiles = .{ .{}, .{ .size = 1024, .step = 4, .min = 4 }, .{} },
        .served = &.{ .{ .rows = 1, .n = width, .k = 1 }, .{ .rows = 16, .n = width, .k = 1 }, .{ .rows = 64, .n = width, .k = 1 } },
    };
}

pub const Swiglu = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    gate: []u16 = &.{},
    up: []u16 = &.{},
    ptrs: [3]u64 = @splat(0),

    pub fn deinit(self: *Swiglu) void {
        self.rig.gpu.gpa.free(self.gate);
        self.rig.gpu.gpa.free(self.up);
        self.rig.deinit();
    }

    pub fn kernel(self: *Swiglu) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Swiglu);

    pub fn prepare(self: *Swiglu, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const total = shape.rows * shape.n;
        if (total == 0 or total % 4 != 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.gate);
        a.free(self.up);
        self.gate = &.{};
        self.up = &.{};
        self.shape = shape;
        self.gate = try a.alloc(u16, total);
        self.up = try a.alloc(u16, total);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x71776534);
        const rng = prng.random();
        for (self.gate, self.up, 0..) |*gv, *uv, i| {
            const big: f32 = if (i % 64 == 7) 20 else 1; // gates to about -80, where expf stays finite
            gv.* = c.toBf16((rng.float(f32) * 8 - 4) * big);
            uv.* = c.toBf16(rng.float(f32) * 4 - 2);
        }
        if (fill == .nonfinite) {
            const n = shape.n;
            const planted = [_]struct { usize, usize, f32, f32 }{
                .{ 0, n / 3, std.math.nan(f32), 1 },
                .{ 1, n / 2, std.math.inf(f32), 1 },
                .{ 2, n - 1, -std.math.inf(f32), 1 },
                .{ 3, 0, std.math.inf(f32), 0 },
                .{ 3, 1, 1, std.math.nan(f32) },
                .{ 4, 2, 1, -std.math.inf(f32) },
            };
            for (planted) |p| if (p[0] < shape.rows) {
                self.gate[p[0] * n + p[1]] = c.toBf16(p[2]);
                self.up[p[0] * n + p[1]] = c.toBf16(p[3]);
            };
        }
        self.ptrs[0] = try self.rig.input(std.mem.sliceAsBytes(self.gate));
        self.ptrs[1] = try self.rig.input(std.mem.sliceAsBytes(self.up));
        self.ptrs[2] = try self.rig.output(total * 2);
        return true;
    }

    pub fn run(self: *Swiglu) !void {
        const total = self.shape.rows * self.shape.n;
        var args: cuda.Args = .{};
        for (self.ptrs) |p| args.add(p);
        args.add(@as(c_longlong, @intCast(total)));
        args.add(@as(u64, 0)); // no row-count pointer
        args.add(@as(c_int, @intCast(self.shape.n)));
        args.add(@as(u64, 0)); // no group sums
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((total + 1023) / 1024) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Swiglu) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Swiglu, out: []f64, mag: []f64) !void {
        for (self.gate, self.up, out) |gv, uv, *o| {
            const x: f64 = c.fromBf16(gv);
            o.* = x / (1 + @exp(-x)) * @as(f64, c.fromBf16(uv));
        }
        @memset(mag, 0);
    }
};
