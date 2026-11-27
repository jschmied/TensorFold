//! A Mini message site's role launches (gate halves, mix) under the contract, each role bit-equal to its launch.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_inkling2_ops_ref.zig");
const base = @import("contract_inkling2_loop_math.zig");
const mix = @import("contract_inkling2_ops_mix.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lm = ink.loop_math_cuda;
const ops = ref.ops;
const mini_h = base.mini_h;

/// Which launch a registration batches: the two gate_*_roles halves, or tf_ink2_mix_roles.
pub const Kind = enum { gate, mix };

/// The served launch, or its control: two roles bound to one buffer that each role must own.
pub const Fault = enum { none, shared };

// The hazard rows; the mix's roles kernel serves calls of 16 rows or fewer (ops.mixRoles), so it stops there.
const gate_shapes = rowsAt(&.{ 1, 2, 3, 4, 7, 8, 13, 16 }, mini_h, 2 * mini_h) ++ rowsAt(&.{ 31, 32, 65 }, 256, 512);
const mix_shapes = rowsAt(&.{ 1, 2, 3, 4, 7, 8, 13, 16 }, mini_h, 1);

fn rowsAt(comptime rows: []const usize, comptime n: usize, comptime k: usize) [rows.len]c.Shape {
    var out: [rows.len]c.Shape = undefined;
    for (rows, &out) |r, *s| s.* = .{ .rows = r, .n = n, .k = k };
    return out;
}

/// Three independent calls of `kind` on distinct inputs, launched as one; the baseline launches each alone.
pub fn Roles(comptime kind: Kind) type {
    const T = switch (kind) {
        .gate => base.Gate,
        .mix => mix.Mix,
    };
    return struct {
        const Self = @This();
        parts: [3]T,
        fault: Fault = .none,
        alone: [3]u64 = @splat(0), // each role's baseline output (scratch)
        want: std.ArrayList(u8) = .empty,

        pub const sums = {};

        pub fn init(kern: anytype, fault: Fault) Self {
            const p: T = switch (kind) {
                .gate => .{ .kern = kern },
                .mix => .{ .kern = kern, .form = .message, .narrow = true },
            };
            return .{ .parts = .{ p, p, p }, .fault = fault };
        }

        pub fn spec(self: *const Self) c.Spec {
            var s = self.parts[0].spec();
            const served: []const c.Shape = switch (kind) {
                .gate => &gate_shapes,
                .mix => &mix_shapes,
            };
            s.served = if (self.fault == .none) served else served[0..2];
            s.replaces = switch (kind) {
                .gate => "tf_ink2_loop_gate_down + _up",
                .mix => "tf_ink2_mix",
            };
            s.name = switch (kind) {
                .gate => if (self.fault == .none) "inkling2 loop tf_ink2_loop_gate_down_roles + _up_roles (3 message gates)" else "control: inkling2 loop gate roles sharing one zeta scratch",
                .mix => if (self.fault == .none) "inkling2 ops tf_ink2_mix_roles (3 message roles, FP32 state, no query)" else "control: inkling2 ops mix roles with role 2 writing role 1's output",
            };
            return s;
        }

        pub fn control(self: *const Self) ?g.Gate {
            return if (self.fault == .none) null else .shapes;
        }

        pub fn free(self: *Self, a: std.mem.Allocator) void {
            for (&self.parts) |*p| p.free(a);
            self.want.deinit(a);
            self.want = .empty;
        }

        pub fn prepare(self: *Self, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
            for (&self.parts) |*p| if (!try p.prepare(a, rig, shape, fill, rng)) return false;
            for (&self.parts, &self.alone) |*p, *o| o.* = if (kind == .mix) p.base else try rig.scratch(p.count() * 4);
            return true;
        }

        fn gateRole(p: *const base.Gate, zeta: u64, out: u64) lm.GateRole {
            const d = p.dev;
            return .{ .state_f32 = d[0], .message_f32 = d[1], .down_bf16 = d[2], .forget_bf16 = d[3], .gain_bf16 = d[4], .zeta_bf16 = zeta, .out_f32 = out };
        }

        fn launchMix(f: cuda.Function, s: cuda.Stream, rows: usize, roles: usize, args: anytype) !void {
            var a: cuda.Args = .{};
            a.add(args);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .z = @intCast(roles) }, .block = .{ .x = 512 } }, s, &a);
        }

        pub fn run(self: *Self, rig: *rig_.Rig) !void {
            const p = &self.parts;
            const shared = self.fault == .shared;
            switch (kind) {
                .gate => {
                    var r: [3]lm.GateRole = undefined;
                    for (&r, p) |*x, *q| x.* = gateRole(q, if (shared) p[0].zeta else q.zeta, q.out);
                    try lm.gateRoles(p[0].kern, rig.s, &r, p[0].shape.rows, lm.rank / 2, p[0].shape.n, base.eps);
                },
                .mix => {
                    var r: [3]ops.MixArgs = undefined;
                    for (&r, p) |*x, *q| x.* = q.args;
                    if (shared) r[2].out = r[1].out;
                    // the roles kernel itself at every listed row count (ops.mixRoles sends wider calls to mix_rows)
                    try launchMix(p[0].kern.mix_roles, rig.s, p[0].shape.rows, 3, r);
                },
            }
        }

        /// Each role's own launch, as the engine ran it before: the gate halves, tf_ink2_mix.
        pub fn replaced(self: *Self, rig: *rig_.Rig) ![]const u8 {
            const a = rig.gpu.gpa;
            self.want.clearRetainingCapacity();
            for (&self.parts, self.alone) |*q, out| {
                const n = q.count() * @as(usize, if (kind == .mix) 2 else 4);
                switch (kind) {
                    .gate => {
                        try lm.gateDown(q.kern, rig.s, q.dev[0], q.dev[1], q.dev[2], q.zeta, q.shape.rows, lm.rank / 2, q.shape.n, base.eps);
                        try lm.gateUp(q.kern, rig.s, q.dev[0], q.dev[1], q.zeta, q.dev[3], q.dev[4], out, q.shape.rows, q.shape.n);
                    },
                    .mix => {
                        var m = q.args;
                        m.out = out;
                        try launchMix(q.kern.mix, rig.s, q.shape.rows, 1, m);
                    },
                }
                try self.want.appendSlice(a, try rig.bytes(out, n));
            }
            return self.want.items;
        }

        pub fn count(self: *const Self) usize {
            return 3 * self.parts[0].count();
        }

        pub fn solve(self: *const Self, a: std.mem.Allocator, s: ref.Sums, out: []ref.Iv) !void {
            const n = self.parts[0].count();
            for (&self.parts, 0..) |*p, i| try p.solve(a, s, out[i * n ..][0..n]);
        }

        pub fn check(self: *const Self, a: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
            if (comptime !@hasDecl(T, "check")) return null;
            const n = self.parts[0].count() * 4;
            for (&self.parts, 0..) |*p, i| if (try p.check(a, got[i * n ..][0..n], msg)) |why| return why;
            return null;
        }
    };
}
