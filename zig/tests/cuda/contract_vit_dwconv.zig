//! vit.cu's gated 3x3 depthwise conv (LFM3-VL) under the contract: an exact host replay, its raster-order control.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const c = core.contract.spec;
const g = core.contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A shape is a picture: rows = gh * gw patches in merge-block order, n channels, k = gw (both sides even).
pub fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) "control: vit 3x3 depthwise conv reading neighbours in raster order" else "cuda vit gated 3x3 depthwise conv tf_vit_dwconv3 (merge-block rows, zero edges)",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = "the host's replay of each row's fp32 and bf16 steps, bit for bit (NaN as NaN)" },
        .tiles = .{ .{}, .{ .size = 256, .step = 64, .min = 64 }, .{} },
        // first a 4 x 2 picture: the single-shape gates clamp the first shape's rows into 5 to 8 and 1 to 64
        .served = &.{ .{ .rows = 8, .n = 1024, .k = 2 }, .{ .rows = 280, .n = 1024, .k = 20 }, .{ .rows = 4480, .n = 1024, .k = 80 } },
    };
}

pub const Conv = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    p: []u16 = &.{},
    w: []u16 = &.{},
    dev: [3]u64 = @splat(0),
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Conv) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Conv) void {
        const a = self.rig.gpu.gpa;
        a.free(self.p);
        a.free(self.w);
        self.p = &.{};
        self.w = &.{};
    }

    pub fn kernel(self: *Conv) g.Kernel {
        return .{ .spec = spec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Conv);

    pub fn prepare(self: *Conv, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const gw = shape.k;
        if (shape.n == 0 or gw < 2 or gw % 2 != 0 or shape.rows == 0 or shape.rows % (2 * gw) != 0) return false;
        self.rig.reset();
        self.free();
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const r = shape.rows;
        const d = shape.n;
        self.p = try a.alloc(u16, r * 3 * d);
        self.w = try a.alloc(u16, d * 9);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.p) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        for (self.w) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, d / 3 }, .{ 1, d + 1 }, .{ 2, 2 * d + 5 }, .{ 3, 0 }, .{ 3, 3 * d - 1 } };
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (at, put) |q, v| if (q[0] < r) {
                self.p[q[0] * 3 * d + q[1]] = c.toBf16(v);
            };
        }
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.p));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.w));
        self.dev[2] = try self.rig.output(r * d * 2);
        return true;
    }

    pub fn run(self: *Conv) !void {
        const d = self.shape.n;
        var a: cuda.Args = .{};
        for (self.dev) |p| a.add(p);
        for ([_]usize{ self.shape.rows / self.shape.k, self.shape.k, d, 2 }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((d + 255) / 256), .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Conv) usize {
        return self.shape.rows * self.shape.n * 9 * 2;
    }

    pub fn reference(_: *Conv, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    fn val(h: u16) f32 {
        return c.fromBf16(h);
    }

    /// The merge-block row at grid position (y, x), merge 2.
    fn rowAt(y: usize, x: usize, gw: usize) usize {
        return ((y / 2) * (gw / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
    }

    pub fn check(self: *Conv, _: g.Gate) !?[]const u8 {
        const out = try self.rig.outputBytes();
        const d = self.shape.n;
        const gw = self.shape.k;
        const gh = self.shape.rows / gw;
        for (0..gh) |y| for (0..gw) |x| {
            const row = rowAt(y, x, gw);
            for (0..d) |ch| {
                var acc: f32 = 0;
                for (0..3) |ky| for (0..3) |kx| {
                    if (y + ky < 1 or y + ky > gh or x + kx < 1 or x + kx > gw) continue;
                    const m = rowAt(y + ky - 1, x + kx - 1, gw);
                    const bx = val(c.toBf16(val(self.p[m * 3 * d + ch]) * val(self.p[m * 3 * d + 2 * d + ch])));
                    acc += val(self.w[ch * 9 + ky * 3 + kx]) * bx;
                };
                const want = c.toBf16(val(self.p[row * 3 * d + d + ch]) * val(c.toBf16(acc)));
                const got = std.mem.readInt(u16, out[2 * (row * d + ch) ..][0..2], .little);
                const same = got == want or (std.math.isNan(val(got)) and std.math.isNan(val(want)));
                if (!same) return try std.fmt.bufPrint(&self.msg, "grid ({d}, {d}) channel {d}: 0x{x:0>4} where the replay gives 0x{x:0>4}", .{ y, x, ch, got, want });
            }
        };
        return null;
    }
};

/// The kernel and its control, registered onto `kernels`; the control from the contract_controls image `ctl`.
pub const Set = struct {
    mod: ?cuda.Module = null,
    conv: [2]Conv = undefined,

    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, ctl: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !void {
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.vit);
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&self.conv, [_]bool{ false, true }) |*x, ctrl| x.* = .{ .rig = rig, .f = try (if (ctrl) ctl else self.mod.?).function(if (ctrl) "tf_ctl_vit_dwconv3" else "tf_vit_dwconv3"), .control = ctrl };
        for (&self.conv) |*x| try kernels.append(gpu.gpa, x.kernel());
    }

    pub fn deinit(self: *Set) void {
        var m = self.mod orelse return;
        for (&self.conv) |*x| x.deinit();
        m.unload();
    }
};

test "the rig's merge-block rows: block by block, each block's two rows" {
    try std.testing.expectEqual(@as(usize, 0), Conv.rowAt(0, 0, 4));
    try std.testing.expectEqual(@as(usize, 1), Conv.rowAt(0, 1, 4));
    try std.testing.expectEqual(@as(usize, 2), Conv.rowAt(1, 0, 4));
    try std.testing.expectEqual(@as(usize, 4), Conv.rowAt(0, 2, 4));
    try std.testing.expectEqual(@as(usize, 8), Conv.rowAt(2, 0, 4));
}
