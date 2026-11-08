//! A DSpark `Pass` as the core lanes `Drafter`, chains only; trees, siblings and spec passes stay in lanes.zig.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const dr = lanes.drafter;
const Stream = lanes.Stream;
const iface = @import("iface.zig");
const dspark = @import("dspark.zig");

pub const max_block = 16;

pub const Options = struct {
    shape: dspark.Shape,
    slots: u32, // the pass's slots, one a stream
    taps: []const u32, // the target layers the pass reads, in its taps' order
    facts: dr.Facts, // the row policy as a parameter: depth (<= shape.block), step_ms, prior
};

const Lane = struct { slot: u32, held: [max_block]u32 = undefined, n: u32 = 0 };

pub const PassDrafter = struct {
    gpa: Allocator,
    pass: iface.Pass,
    opts: Options,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    free: std.ArrayList(u32) = .empty, // slots not held by a stream
    rows: std.ArrayList(u32) = .empty, // scratch: an absorb's taps rows

    pub fn init(gpa: Allocator, pass: iface.Pass, opts: Options) !PassDrafter {
        if (opts.shape.block > max_block or opts.facts.depth > opts.shape.block) return error.DraftDepth;
        var x: PassDrafter = .{ .gpa = gpa, .pass = pass, .opts = opts };
        errdefer x.free.deinit(gpa);
        var s = opts.slots;
        while (s > 0) : (s -= 1) try x.free.append(gpa, s - 1);
        return x;
    }

    pub fn deinit(x: *PassDrafter) void {
        x.lanes.deinit(x.gpa);
        x.free.deinit(x.gpa);
        x.rows.deinit(x.gpa);
    }

    pub fn drafter(x: *PassDrafter) dr.Drafter {
        return .{ .kind = "DSpark", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *PassDrafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        return cast(ptr).opts.taps;
    }

    fn factsFn(ptr: *anyopaque) dr.Facts {
        return cast(ptr).opts.facts;
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = cast(ptr);
        if (x.lanes.contains(s)) return error.DraftStreamOpen;
        const slot = x.free.pop() orelse return error.TooManySlots;
        errdefer x.free.appendAssumeCapacity(slot);
        try x.lanes.put(x.gpa, s, .{ .slot = slot });
        x.pass.reset(slot); // lanes.zig's prefill: the slot's context restarts at position 0
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const x = cast(ptr);
        const l = x.lanes.fetchRemove(s) orelse return;
        x.pass.reset(l.value.slot);
        x.free.appendAssumeCapacity(l.value.slot);
    }

    fn elemBytes(t: lanes.backend.Features.Dtype) u32 {
        return switch (t) {
            .bf16, .f16 => 2,
            .f32, .u32 => 4,
        };
    }

    /// Rows land at start + i (taps row `rows[i]`), only the last `window` of them (iface.ingestSpan, as lanes.zig).
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = cast(ptr);
        for (items) |it| {
            const l = x.lanes.getPtr(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.ready != .none) return error.DraftFeaturesUnsupported;
            const e = elemBytes(f.dtype);
            if (f.row_bytes % e != 0) return error.DraftFeaturesUnsupported;
            const taps: iface.Taps = .{ .address = f.buffer + f.offset, .rows = f.rows, .stride = f.row_bytes / e };
            x.rows.clearRetainingCapacity();
            if (it.rows) |r| try x.rows.appendSlice(x.gpa, r) else for (0..f.rows) |r| try x.rows.append(x.gpa, @intCast(r));
            l.n = 0;
            const span = iface.ingestSpan(x.rows.items.len, x.opts.shape.window);
            if (span.count == 0) continue;
            try x.pass.ingest(l.slot, it.start + span.skip, taps, x.rows.items[span.skip..]);
        }
    }

    /// One pass over every asking stream (lanes.zig's draft): the pending token at position - 1 is the anchor.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const x = cast(ptr);
        const b = x.opts.shape.block;
        var asks: [max_block]iface.Ask = undefined;
        var whose: [max_block]*Lane = undefined;
        var depths: [max_block]u32 = undefined;
        var drafts: [max_block * max_block]u32 = undefined;
        var conf: [max_block * max_block]f32 = undefined;
        var props: [max_block]iface.Proposal = undefined;
        var n: usize = 0;
        for (items) |it| {
            const l = x.lanes.getPtr(it.stream) orelse return error.DraftStreamUnknown;
            l.n = 0;
            if (it.depth == 0) continue;
            if (it.depth > b or it.position == 0 or it.lanes != null) return error.DraftDepth;
            if (n == asks.len) return error.TooManySlots;
            const at = it.position - 1;
            asks[n] = .{ .slot = l.slot, .anchor = it.pending, .start = at, .params = dspark.Params.of(it.stream.sampling, at) };
            props[n] = .{ .drafts = drafts[n * b ..][0..b], .conf = conf[n * b ..][0..b] };
            whose[n] = l;
            depths[n] = it.depth;
            n += 1;
        }
        if (n == 0) return;
        try x.pass.propose(asks[0..n], props[0..n]);
        for (props[0..n], whose[0..n], depths[0..n]) |p, l, d| {
            l.n = d;
            @memcpy(l.held[0..d], p.drafts[0..d]);
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = cast(ptr);
        for (streams, out) |s, o| {
            const l = x.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.n) return error.DraftsShort;
            @memcpy(o, l.held[0..o.len]);
        }
    }
};
