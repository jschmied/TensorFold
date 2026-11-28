//! The core block products (core/block_rows.metal) under contract gates 1-4: dense rows, expert down, gate/up.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const br = tf.block_rows;
const base = "core rows ";

pub const Kind = enum { dense, down, glu };
pub const Variant = struct { kind: Kind, bf16: bool = false, ctl: u8 = 0 };

/// Kolibri-1's projections: qkv and o, an expert's gate/up and down, the router and a head slice; every row count.
const served = [_]c.Shape{
    .{ .rows = 16, .n = 512, .k = 2560 }, // k, v, an expert's gate or up
    .{ .rows = 1, .n = 6144, .k = 2560 }, // q
    .{ .rows = 4, .n = 2560, .k = 6144 }, // o
    .{ .rows = 7, .n = 2560, .k = 512 }, // an expert's down
    .{ .rows = 16, .n = 384, .k = 2560 }, // the router
    .{ .rows = 3, .n = 1029, .k = 2560 }, // a head slice, an odd column count
} ++ everyRow();

fn everyRow() [16]c.Shape {
    var out: [16]c.Shape = undefined;
    for (&out, 1..) |*s, r| s.* = .{ .rows = r, .n = 9, .k = 384 };
    return out;
}

/// A lane's 4-term chain, the 32-lane tree, then one add a 128-column block.
fn depth(s: c.Shape) usize {
    return 4 + 5 + s.k / 128;
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depth,
        .ops = if (v.bf16) 0 else 1, // the block scale's product
        .mids = if (v.kind == .glu) 2 else 0,
        .out = if (v.kind == .glu) .bf16 else .f32,
        .tiles = .{ .{ .size = 16 }, .{ .size = 4 }, .{ .size = 256, .step = 128, .min = 128 } },
        .served = &served,
    };
}

/// An e4m3 code's value (no NaN codes are drawn): (1 + m/8) 2^(e-7), subnormals m/8 2^-6.
fn e4m3(code: u8) f64 {
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    const v = if (e == 0) m / 8 * 0x1p-6 else (1 + m / 8) * std.math.ldexp(@as(f64, 1), e - 7);
    return if (code & 0x80 != 0) -v else v;
}

pub const Rows = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: br.Pipes,
    d: ?Dense = null,
    u: ?Dense = null, // glu: the up projection's terms beside the gate's in `d`
    fill: g.Fill = .random,
    experts: usize = 3,
    pick: []u32 = &.{}, // a pair's (row's) expert
    refs: [8]rig_.Ref = undefined, // x, w, s or at, y, items, members, count

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Rows {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = try br.Pipes.init(a, device, .f32, v.ctl) };
    }

    pub fn deinit(p: *Rows) void {
        p.drop();
        p.pipes.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Rows) void {
        if (p.d) |d| d.deinit();
        if (p.u) |d| d.deinit();
        p.d = null;
        p.u = null;
        p.a.free(p.pick);
        p.pick = &.{};
        p.rig.reset();
    }

    pub fn kernel(p: *Rows) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = if (p.v.ctl != 0) .shapes else null };
    }

    const vt = g.hooks(Rows);

    /// Weights [E][n][k]: codes and unequal block scales (random), or small integers at scale 1 (gate 4), or bf16.
    fn weights(p: *Rows, d: Dense, e: usize, rng: std.Random, codes: []u8, scales: []f32, w16: []u16) void {
        const n = d.shape.n;
        const k = d.shape.k;
        const kb = k / 128;
        const nb = (n + 127) / 128;
        for (0..nb * kb) |i| scales[e * nb * kb + i] = if (p.fill == .stress) 1 else (0.5 + rng.float(f32)) * 0x1p-6;
        for (0..n) |j| for (0..k) |i| {
            const at = (e * n + j) * k + i;
            if (p.v.bf16) {
                w16[at] = c.toBf16(if (p.fill == .stress) @floatFromInt(1 + rng.uintLessThan(u32, 6)) else rng.float(f32) * 2 - 1);
                continue;
            }
            const stress = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c }; // 1 .. 6
            codes[at] = if (p.fill == .stress) stress[rng.uintLessThan(usize, 6)] else (if (rng.boolean()) @as(u8, 0x80) else 0) | @as(u8, @intCast(rng.uintLessThan(u32, 0x7f)));
        };
    }

    fn widen(p: *const Rows, d: Dense, e: usize, codes: []const u8, scales: []const f32, w16: []const u16) void {
        const n = d.shape.n;
        const k = d.shape.k;
        const kb = k / 128;
        for (0..n) |j| for (0..k) |i| {
            const at = (e * n + j) * k + i;
            d.w[j * k + i] = if (p.v.bf16) c.fromBf16(w16[at]) else e4m3(codes[at]) * scales[e * ((n + 127) / 128) * kb + (j / 128) * kb + i / 128];
        };
    }

    pub fn prepare(p: *Rows, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 16 or shape.k % 128 != 0 or shape.n == 0) return false;
        p.drop();
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6b31);
        const rng = prng.random();
        const kinds = if (p.v.kind == .dense) 1 else p.experts;
        const n = shape.n;
        const k = shape.k;
        const per = n * k;
        const codes = try p.a.alignedAlloc(u8, .@"4", kinds * per * @as(usize, if (p.v.bf16) 2 else 1) * @as(usize, if (p.v.kind == .glu) 2 else 1));
        defer p.a.free(codes);
        const nb = (n + 127) / 128;
        const scales = try p.a.alloc(f32, 2 * kinds * nb * (k / 128));
        defer p.a.free(scales);
        const w16: []u16 = @alignCast(std.mem.bytesAsSlice(u16, codes[0 .. codes.len / 2 * 2]));
        p.d = try Dense.init(p.a, shape, 1);
        p.d.?.fillX(fill, seed, 1);
        if (p.v.kind == .glu) {
            p.u = try Dense.init(p.a, shape, 1);
            @memcpy(p.u.?.x, p.d.?.x);
        }
        p.pick = try p.a.alloc(u32, shape.rows);
        for (p.pick, 0..) |*e, r| e.* = @intCast((r * 7 + seed) % kinds); // uneven groups, every expert used
        // the weights each row multiplies by, as the reference reads them: row r's expert's
        const halves: usize = if (p.v.kind == .glu) 2 else 1;
        for (0..kinds * halves) |e| p.weights(p.d.?, e, rng, if (p.v.bf16) &.{} else codes, scales, if (p.v.bf16) w16 else &.{});
        const at = try p.a.alloc(br.ExpertAt, kinds);
        defer p.a.free(at);
        const wbytes = per * @as(usize, if (p.v.bf16) 2 else 1);
        const sbytes = nb * (k / 128) * 4;
        for (at, 0..) |*o, e| o.* = .{ .gate = e * halves * wbytes, .gate_s = codes.len + e * halves * sbytes, .up = (e * halves + 1) * wbytes, .up_s = codes.len + (e * halves + 1) * sbytes, .down = e * wbytes, .down_s = codes.len + e * sbytes };
        const blob = try p.a.alloc(u8, codes.len + scales.len * 4);
        defer p.a.free(blob);
        @memcpy(blob[0..codes.len], codes);
        @memcpy(blob[codes.len..], std.mem.sliceAsBytes(scales));
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.d.?.x));
        p.refs[1] = try p.rig.input(blob);
        if (p.v.kind == .dense) {
            p.widen(p.d.?, 0, codes, scales, w16);
            p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(scales));
            p.refs[3] = try p.rig.output(shape.rows * n * 4);
            return true;
        }
        p.refs[2] = try p.rig.input(std.mem.sliceAsBytes(at));
        p.refs[3] = try p.rig.output(shape.rows * n * @as(usize, if (p.v.kind == .glu) 2 else 4));
        try p.groups(kinds);
        return true;
    }

    /// The groups tf_moe_plan makes: pairs sorted by expert, at most 16 a group.
    fn groups(p: *Rows, kinds: usize) !void {
        var items: std.ArrayList(u32) = .empty;
        defer items.deinit(p.a);
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(p.a);
        var made: u32 = 0;
        for (0..kinds) |e| {
            const first: u32 = @intCast(members.items.len);
            for (p.pick, 0..) |x, r| if (x == e) try members.append(p.a, @intCast(r));
            const c_e: u32 = @as(u32, @intCast(members.items.len)) - first;
            var i: u32 = 0;
            while (i < c_e) : (i += 16) {
                try items.appendSlice(p.a, &.{ @intCast(e), first + i, @min(16, c_e - i) });
                made += 1;
            }
        }
        p.refs[4] = try p.rig.input(std.mem.sliceAsBytes(items.items));
        p.refs[5] = try p.rig.input(std.mem.sliceAsBytes(members.items));
        p.refs[6] = try p.rig.input(std.mem.asBytes(&made));
    }

    fn view(x: rig_.Ref) br.Ref {
        return .{ .buf = x.buf, .off = x.off };
    }

    pub fn run(p: *Rows) !void {
        const s = p.d.?.shape;
        const pass = p.rig.begin();
        const bf: u32 = @intFromBool(p.v.bf16);
        switch (p.v.kind) {
            .dense => try p.pipes.dense(pass.e, view(p.refs[0]), view(p.refs[1]), view(p.refs[2]), view(p.refs[3]), .{ .rows = @intCast(s.rows), .k = @intCast(s.k), .n = @intCast(s.n), .ldx = @intCast(s.k), .ldy = @intCast(s.n), .bf16 = bf }),
            .down, .glu => {
                const plan: br.Pipes.Plan = .{ .items = view(p.refs[4]), .members = view(p.refs[5]), .count = view(p.refs[6]), .most = br.maxItems(s.rows, p.experts) };
                const a: br.PickArgs = .{ .k = @intCast(s.k), .n = @intCast(s.n), .ldx = @intCast(s.k), .slots = 1, .bf16 = bf };
                if (p.v.kind == .down) try p.pipes.down(pass.e, view(p.refs[0]), view(p.refs[1]), view(p.refs[2]), plan, view(p.refs[3]), a) else try p.pipes.gateUp(pass.e, view(p.refs[0]), view(p.refs[1]), view(p.refs[2]), plan, view(p.refs[3]), a);
            },
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Rows) usize {
        return p.d.?.outputs();
    }

    /// Row r's weights: its expert's (down, glu), from the blob the kernel reads.
    fn rowWeights(p: *Rows, d: Dense, row: usize, half: usize, w: []f64) !void {
        const blob = p.rig.bytes(p.refs[1], p.refs[1].buf.length() - 2 * rig_.guard);
        const n = d.shape.n;
        const k = d.shape.k;
        const at = std.mem.bytesAsSlice(br.ExpertAt, p.rig.bytes(p.refs[2], p.experts * @sizeOf(br.ExpertAt)))[p.pick[row]];
        const w_off: usize = @intCast(switch (p.v.kind) {
            .glu => if (half == 0) at.gate else at.up,
            else => at.down,
        });
        const s_off: usize = @intCast(switch (p.v.kind) {
            .glu => if (half == 0) at.gate_s else at.up_s,
            else => at.down_s,
        });
        const kb = k / 128;
        for (0..n) |j| for (0..k) |i| {
            w[j * k + i] = if (p.v.bf16) c.fromBf16(std.mem.readInt(u16, blob[w_off + 2 * (j * k + i) ..][0..2], .little)) else e4m3(blob[w_off + j * k + i]) * @as(f64, @as(f32, @bitCast(std.mem.readInt(u32, blob[s_off + 4 * ((j / 128) * kb + i / 128) ..][0..4], .little))));
        };
    }

    /// Each row's outputs through its own expert: a one-row Dense at a time.
    fn perRow(p: *Rows, comptime f: anytype, args: anytype) !void {
        const s = p.d.?.shape;
        for (0..s.rows) |row| for (0..@as(usize, if (p.v.kind == .glu) 2 else 1)) |half| {
            var one = try Dense.init(p.a, .{ .rows = 1, .n = s.n, .k = s.k }, 1);
            defer one.deinit();
            @memcpy(one.x, p.d.?.x[row * s.k ..][0..s.k]);
            try p.rowWeights(one, row, half, one.w);
            try f(p, one, row, half, args);
        };
    }

    pub fn reference(p: *Rows, want: []f64, mag: []f64) !void {
        if (p.v.kind == .dense) return p.d.?.reference(want, mag);
        const s = p.d.?.shape;
        const tmp = try p.a.alloc(f64, 4 * s.n);
        defer p.a.free(tmp);
        const Ctx = struct { want: []f64, mag: []f64, tmp: []f64 };
        try p.perRow(struct {
            fn one(q: *Rows, d: Dense, row: usize, half: usize, x: Ctx) !void {
                const n = d.shape.n;
                try d.reference(x.tmp[half * n ..][0..n], x.tmp[(2 + half) * n ..][0..n]);
                if (q.v.kind == .down) {
                    @memcpy(x.want[row * n ..][0..n], x.tmp[0..n]);
                    @memcpy(x.mag[row * n ..][0..n], x.tmp[2 * n ..][0..n]);
                    return;
                }
                if (half == 0) return;
                for (0..n) |j| {
                    const gv = x.tmp[j];
                    const uv = x.tmp[n + j];
                    const sg = 1.0 / (1.0 + @exp(-gv));
                    x.want[row * n + j] = gv * sg * uv;
                    // |d out / d gate| times the gate's terms plus |d out / d up| times the up's: the sums' errors
                    const dg = @abs(uv * (sg + gv * sg * (1 - sg)));
                    x.mag[row * n + j] = dg * x.tmp[2 * n + j] + @abs(gv * sg) * x.tmp[3 * n + j];
                }
            }
        }.one, Ctx{ .want = want, .mag = mag, .tmp = tmp });
    }

    pub fn bounds(p: *Rows, want: []const f64, _: []const f64, allow: []f64) !void {
        if (p.v.kind != .glu) return;
        // the sigmoid's exp and divide and the two products: four fp32 roundings of the value
        for (allow, want) |*l, w| l.* += 4 * 0x1p-24 * @abs(w);
    }

    pub fn simulate(p: *Rows, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (p.v.kind == .dense) return p.d.?.simulate(acc, order, out);
        const s = p.d.?.shape;
        const tmp = try p.a.alloc(f64, 2 * s.n);
        defer p.a.free(tmp);
        const Ctx = struct { out: []f64, tmp: []f64, acc: c.Accum, order: g.Order };
        try p.perRow(struct {
            fn one(q: *Rows, d: Dense, row: usize, half: usize, x: Ctx) !void {
                const n = d.shape.n;
                try d.simulate(x.acc, x.order, x.tmp[half * n ..][0..n]);
                if (q.v.kind == .down) return @memcpy(x.out[row * n ..][0..n], x.tmp[0..n]);
                if (half == 0) return;
                for (0..n) |j| {
                    const gv = x.tmp[j];
                    x.out[row * n + j] = gv / (1.0 + @exp(-gv)) * x.tmp[n + j];
                }
            }
        }.one, Ctx{ .out = out, .tmp = tmp, .acc = acc, .order = order });
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Rows) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .kind = .dense }, .name = base ++ "fp8 tf_block_rows" },
        .{ .v = .{ .kind = .dense, .bf16 = true }, .name = base ++ "bf16 tf_block_rows" },
        .{ .v = .{ .kind = .down }, .name = base ++ "fp8 tf_block_down" },
        .{ .v = .{ .kind = .down, .bf16 = true }, .name = base ++ "bf16 tf_block_down" },
        .{ .v = .{ .kind = .glu }, .name = base ++ "fp8 tf_block_glu" },
        .{ .v = .{ .kind = .glu, .bf16 = true }, .name = base ++ "bf16 tf_block_glu" },
        .{ .v = .{ .kind = .dense, .ctl = 1 }, .name = base ++ "control: the next block's scale" },
        .{ .v = .{ .kind = .down, .ctl = 2 }, .name = base ++ "control: codes' signs dropped" },
        .{ .v = .{ .kind = .glu, .ctl = 3 }, .name = base ++ "control: gate and up swapped" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Rows);
            errdefer a.destroy(p);
            p.* = try Rows.init(a, device, queue, entry.v, entry.name);
            errdefer p.deinit();
            try contract.register(a, &s.probes, kernels, p, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
