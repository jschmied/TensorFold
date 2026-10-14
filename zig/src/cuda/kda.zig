//! kda.cu and kda_prompt.cu's Kimi Delta Attention for every CUDA family: decode rows, verify windows, prompt chunks.
const std = @import("std");
const kernels = @import("kernels.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");

const Driver = @import("driver.zig").Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Key and value dims a head, conv taps, raw rows the conv window keeps, and the prompt kernel's chunk.
pub const dim = 128;
pub const taps = 4;
pub const keep = taps - 1;
pub const chunk = 16;

/// A call's rows (addresses; strides in elements), as kda.cuh's Io; every input bf16.
pub const Io = extern struct {
    qkv: u64, // [rows, qkv_ld]: raw q | k | v channels before the conv, heads * dim each
    qkv_ld: i64,
    f: u64, // [rows, f_ld]: forget-gate inputs, heads * dim (a low-rank gate already applied)
    f_ld: i64,
    b: u64, // [rows, b_ld]: beta logits, heads
    b_ld: i64,
    gate: u64, // [rows, gate_ld]: output-gate logits, heads * dim
    gate_ld: i64,
    y: u64, // [rows, y_ld]: output, heads * dim
    y_ld: i64,
};

/// fp32 weights: conv [3 heads dim][taps] (q, k, v channels), A_log [heads], dt_bias [heads dim], output norm [dim].
pub const Weights = extern struct { conv: u64, a_log: u64, dt_bias: u64, norm: u64 };

/// Slot-major arenas (sizes from `slotBytes`); the logs hold a window's rows until the next call keeps some.
pub const State = extern struct {
    s: u64, // [slots][heads][dim v][dim k] fp32
    conv: u64, // [slots][keep][3 heads dim] bf16, oldest raw row first
    log_a: u64, // [slots][log_rows][heads][dim] fp32 decay factors
    log_k: u64, // [slots][log_rows][heads][dim] fp32 normalized keys
    log_u: u64, // [slots][log_rows][heads][dim] fp32 beta (v - S^T k)
    log_x: u64, // [slots][log_rows][3 heads dim] bf16 raw rows
};

/// One stream's rows in a call: replay `kept` rows of its last window, then rows [first, first + rows) of the call.
pub const Seg = extern struct { first: i32, rows: i32, kept: i32 = 0, flags: i32 = 0, slot: i32 };

/// Seg.flags: write the state after the rows (decode, prompt); without it a window logs its rows instead.
pub const commit: i32 = 1;

const Params = extern struct { heads: i32, log_rows: i32, lower_bound: f32, eps: f32 };

/// lower_bound < 0: g = lower_bound sigmoid(e^A (f + dt_bias)) (the safe gate); 0: g = -e^A softplus(f + dt_bias).
pub const Shape = struct { heads: usize, log_rows: usize = 16, lower_bound: f32, eps: f32 };

/// A slot's bytes in each arena.
pub const SlotBytes = struct { s: usize, conv: usize, log: usize, log_x: usize };

pub fn slotBytes(sh: Shape) SlotBytes {
    const hd = sh.heads * dim;
    return .{ .s = hd * dim * 4, .conv = keep * 3 * hd * 2, .log = sh.log_rows * hd * 4, .log_x = sh.log_rows * 3 * hd * 2 };
}

pub const Kda = struct {
    mod_rows: Module,
    mod_prompt: Module,
    rows_fn: Function,
    prompt_fn: Function,

    pub fn load(d: *const Driver) !Kda {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mr = try Module.load(d, kernels.kda);
        errdefer mr.unload();
        var mp = try Module.load(d, kernels.kda_prompt);
        errdefer mp.unload();
        return .{ .mod_rows = mr, .mod_prompt = mp, .rows_fn = try mr.function("tf_kda_rows"), .prompt_fn = try mp.function("tf_kda_prompt") };
    }

    pub fn unload(k: *Kda) void {
        k.mod_rows.unload();
        k.mod_prompt.unload();
    }

    /// Decode rows and verify windows: each segment's rows one at a time, a window's rows (<= log_rows) logged.
    pub fn rows(k: *const Kda, s: Stream, io: Io, w: Weights, st: State, segs: u64, n: usize, sh: Shape) !void {
        return run(k.rows_fn, 512, s, io, w, st, segs, n, sh);
    }

    /// Prompt rows: each segment's rows in 16-row chunks from its first row, always committed (flags ignored).
    pub fn prompt(k: *const Kda, s: Stream, io: Io, w: Weights, st: State, segs: u64, n: usize, sh: Shape) !void {
        return run(k.prompt_fn, 256, s, io, w, st, segs, n, sh);
    }

    fn run(f: Function, threads: u32, s: Stream, io: Io, w: Weights, st: State, segs: u64, n: usize, sh: Shape) !void {
        if (sh.heads == 0 or sh.log_rows == 0 or !(sh.lower_bound <= 0)) return error.KdaShape;
        if (n == 0) return;
        var a: launch_.Args = .{};
        a.add(io);
        a.add(w);
        a.add(st);
        a.add(segs);
        a.add(Params{ .heads = @intCast(sh.heads), .log_rows = @intCast(sh.log_rows), .lower_bound = sh.lower_bound, .eps = sh.eps });
        try launch_.launch(f, .{ .grid = .{ .x = @intCast(sh.heads), .y = @intCast(n), .z = 1 }, .block = .{ .x = threads } }, s, &a);
    }
};

test "a slot's bytes at Ling-3.1's 64 heads" {
    const b = slotBytes(.{ .heads = 64, .lower_bound = -5, .eps = 1e-6 });
    try std.testing.expectEqual(@as(usize, 4 << 20), b.s);
    try std.testing.expectEqual(@as(usize, 3 * 3 * 64 * 128 * 2), b.conv);
    try std.testing.expectEqual(@as(usize, 16 * 64 * 128 * 4), b.log);
}
