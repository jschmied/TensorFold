//! Kolibri 1's forward, kernels and lane backend behind the native family interface (native/cuda.zig's registry).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const weights = @import("weights.zig");
const kern = @import("kernels.zig");
const fwd = @import("forward.zig");
const be = @import("backend.zig");

pub const model_type = "kolibri1";
pub const formats: []const []const u8 = &.{"fp8-block"};
pub const default_context: i64 = 65536;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = be.prompt_rows;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

/// A lone drafted stream's own driver (none here: every stream runs in the lane core).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives. Every stream's caches are allocated at open (`free_streams` of them).
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    learn_bytes: usize = 0,
};

/// No --slide here, so nothing is ever written in place.
pub fn linked(_: std.mem.Allocator, _: std.Io, _: []const u8) !?[]const u8 {
    return null;
}

const Owned = struct { gpa: std.mem.Allocator, w: weights.Weights, k: kern.Kernels, s: cuda.Stream, m: fwd.Model, kb: be.Kolibri };

/// Streams whose caches the open allocates: TENSORFOLD_KOLIBRI_STREAMS, else 4.
fn streams() usize {
    const v = std.c.getenv("TENSORFOLD_KOLIBRI_STREAMS") orelse return 4;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4;
}

/// The checkpoint, our own kernels and caches.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // the registry's set is Nemotron's glue; Kolibri names its own
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.w = try weights.load(gpa, io, ctx.d, dir, null);
    errdefer own.w.deinit();
    own.k = try kern.Kernels.load(ctx);
    errdefer own.k.deinit();
    own.s = try cuda.Stream.init(ctx.d, false);
    errdefer own.s.deinit();
    const context = (o.context + 1023) / 1024 * 1024; // whole 1,024s
    const n = streams();
    own.m = try fwd.Model.init(gpa, ctx.d, own.s, &own.w, context, n, be.prompt_rows);
    errdefer own.m.deinit();
    own.kb = try be.Kolibri.init(gpa, ctx.d, &own.m, &own.k, own.s);
    return .{
        .backend = own.kb.backend(),
        .facts = own.kb.facts(),
        .rows = 16,
        .stream_bytes = streamBytes(own.w.config, context, own.m.ring),
        .free_streams = @intCast(n),
        .ctx = own,
        .deinit = release,
    };
}

/// One stream's caches: full layers hold the context, sliding ones the ring.
fn streamBytes(c: @import("config.zig").Config, context: usize, ring: usize) usize {
    var total: usize = 0;
    for (c.full[0..c.layers]) |full| total += 2 * (if (full) context else ring) * c.kv_heads * c.head_dim * 2;
    return total;
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PastContext => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream's caches are in use: retry once another request ends",
        error.NoDraftHead => "Kolibri 1 has no draft head on this engine: copy drafts only",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    own.kb.deinit();
    own.m.deinit();
    own.s.deinit();
    own.k.deinit();
    own.w.deinit();
    own.gpa.destroy(own);
}
