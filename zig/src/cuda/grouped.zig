//! experts.route's plan on experts.cu: routed pairs grouped by expert into items of 16 (decode) or 64 pairs.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const tile = 16; // pairs a decode item holds
pub const small = 1024; // pairs the one-block plan takes

/// Items a plan of `pairs` can hold: one per used expert, plus one per `t` pairs past its first (experts.max_items).
pub fn maxItems(pairs: usize, experts: usize, t: usize) usize {
    return @min(pairs, experts) + pairs / t;
}

/// The plan's scratch: members [pairs], items [maxItems][3], counts [2], and for wide plans rank and a histogram.
pub const Plan = struct {
    members: memory.DeviceBuffer,
    items: memory.DeviceBuffer,
    counts: memory.DeviceBuffer,
    rank: memory.DeviceBuffer,
    hist: memory.DeviceBuffer,
    pairs: usize,
    experts: usize,

    pub fn init(d: *const Driver, rows: usize, slots: usize, experts: usize) !Plan {
        const pairs = rows * slots;
        const wide = pairs > small;
        var p: Plan = undefined;
        p.pairs = pairs;
        p.experts = experts;
        p.members = try memory.DeviceBuffer.alloc(d, pairs * 4);
        errdefer p.members.free();
        p.items = try memory.DeviceBuffer.alloc(d, maxItems(pairs, experts, tile) * 3 * 4);
        errdefer p.items.free();
        p.counts = try memory.DeviceBuffer.alloc(d, 2 * 4);
        errdefer p.counts.free();
        p.rank = try memory.DeviceBuffer.alloc(d, (if (wide) pairs else 1) * 4);
        errdefer p.rank.free();
        p.hist = try memory.DeviceBuffer.alloc(d, (if (wide) (pairs + 1023) / 1024 * experts else 1) * 4);
        return p;
    }

    pub fn deinit(p: *Plan) void {
        for ([_]*memory.DeviceBuffer{ &p.members, &p.items, &p.counts, &p.rank, &p.hist }) |b| b.free();
    }
};

pub const symbols = struct {
    pub const plan = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    pub const rank = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    pub const offsets = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    pub const scatter = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
};

pub const Router = struct {
    mod: Module,
    small_plan: Function,
    rank: Function,
    offsets: Function,
    scatter: Function,

    pub fn load(d: *const Driver) !Router {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.experts);
        errdefer mod.unload();
        return .{ .mod = mod, .small_plan = try mod.function(symbols.plan), .rank = try mod.function(symbols.rank), .offsets = try mod.function(symbols.offsets), .scatter = try mod.function(symbols.scatter) };
    }

    pub fn unload(r: *Router) void {
        r.mod.unload();
    }

    /// experts.route: `picks` [pairs] int32 (each pair's expert) grouped into items of at most `t` pairs.
    pub fn route(r: Router, s: Stream, picks: u64, pairs: usize, p: *const Plan, t: usize) !void {
        const one = launch_.Dim3{ .x = 1, .y = 1, .z = 1 };
        if (pairs <= small) {
            var a: launch_.Args = .{};
            a.add(picks);
            for ([_]usize{ pairs, p.experts, t }) |v| a.add(@as(i32, @intCast(v)));
            for ([_]u64{ p.members.ptr, p.items.ptr, p.counts.ptr }) |v| a.add(v);
            return launch_.launch(r.small_plan, .{ .grid = one, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
        }
        const nblk = (pairs + 1023) / 1024;
        var a: launch_.Args = .{};
        a.add(picks);
        a.add(@as(i32, @intCast(pairs)));
        a.add(@as(i32, @intCast(p.experts)));
        a.add(p.rank.ptr);
        a.add(p.hist.ptr);
        try launch_.launch(r.rank, .{ .grid = .{ .x = @intCast(nblk), .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &a);
        var b: launch_.Args = .{};
        for ([_]usize{ nblk, p.experts, t }) |v| b.add(@as(i32, @intCast(v)));
        for ([_]u64{ p.hist.ptr, p.items.ptr, p.counts.ptr }) |v| b.add(v);
        try launch_.launch(r.offsets, .{ .grid = one, .block = .{ .x = 1024, .y = 1, .z = 1 } }, s, &b);
        var c: launch_.Args = .{};
        c.add(picks);
        c.add(@as(i32, @intCast(pairs)));
        c.add(@as(i32, @intCast(p.experts)));
        for ([_]u64{ p.rank.ptr, p.hist.ptr, p.members.ptr }) |v| c.add(v);
        try launch_.launch(r.scatter, .{ .grid = .{ .x = @intCast((pairs + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &c);
    }
};

test "plan capacity as experts.max_items" {
    try std.testing.expectEqual(@as(usize, 7 + 0), maxItems(7, 385, 16));
    try std.testing.expectEqual(@as(usize, 385 + 2100 / 64), maxItems(2100, 385, 64));
    try std.testing.expectEqual(@as(usize, 21 + 1), maxItems(21, 385, 16));
}
