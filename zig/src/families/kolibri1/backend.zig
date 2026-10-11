//! Kolibri 1 behind lanes.Backend: a cache slot a stream, one forward a round over every window, draws on the GPU.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const kern = @import("kernels.zig");

const be = lanes.backend;
const Stream = lanes.Stream;

/// Prompt rows a prefill forward takes (well under forward.zig's ring limit).
pub const prompt_rows = 2048;
/// Window rows a round takes over every stream: the forward's scratch holds this many beside a prompt chunk.
pub const round_rows = 64;
/// Drawn tokens a handle names (a ring; read before the round loop could wrap it).
const ring = 1024;

const Lane = struct {
    slot: usize,
    len: u64 = 0, // the cache length: positions [0, len) hold this stream's keys
    base: u64 = 0, // the cache length the last verify started from
    rows: std.ArrayList(u32) = .empty, // the last verify's row tokens
    last: u64, // the prompt's last row's fp32 logits [V] (the first draw reads them)
    rule: u64, // sample.cu's Rule, when the stream samples
    sampled: bool = false,
};

pub const Kolibri = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *fwd.Model,
    k: *const kern.Kernels,
    s: cuda.Stream,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    free: []bool, // slot i is free
    logits: cuda.DeviceBuffer, // fp32 [round_rows, V]
    logits16: cuda.DeviceBuffer, // their bf16 copy for keyed draws
    tok: cuda.DeviceBuffer, // drawn tokens: the handle ring, then a round's rows
    meta: cuda.DeviceBuffer, // a draw's first position minus one, a window each
    lane_bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    next: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *fwd.Model, k: *const kern.Kernels, s: cuda.Stream) !Kolibri {
        const v = m.c.vocab;
        const free = try gpa.alloc(bool, m.slots);
        @memset(free, true);
        errdefer gpa.free(free);
        var logits = try cuda.DeviceBuffer.alloc(d, round_rows * v * 4);
        errdefer logits.free();
        var logits16 = try cuda.DeviceBuffer.alloc(d, round_rows * v * 2);
        errdefer logits16.free();
        var tok = try cuda.DeviceBuffer.alloc(d, (ring + round_rows) * 4);
        errdefer tok.free();
        const meta = try cuda.DeviceBuffer.alloc(d, round_rows * 4 * 4);
        return .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .free = free, .logits = logits, .logits16 = logits16, .tok = tok, .meta = meta };
    }

    pub fn deinit(x: *Kolibri) void {
        var it = x.lanes.valueIterator();
        while (it.next()) |l| l.rows.deinit(x.gpa);
        x.lanes.deinit(x.gpa);
        for (x.lane_bufs.items) |*b| b.free();
        x.lane_bufs.deinit(x.gpa);
        x.gpa.free(x.free);
        x.logits.free();
        x.logits16.free();
        x.tok.free();
        x.meta.free();
    }

    pub fn backend(x: *Kolibri) be.Backend {
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    /// What the round loop reads: windows up to 16 rows (every kernel is row-invariant), several streams a forward.
    pub fn facts(x: *const Kolibri) lanes.Model {
        return .{ .exact_width = 16, .gpu_tokens = false, .mtp = false, .speculate = false, .hidden_rows = true, .batch_rows = round_rows, .max_streams = @intCast(x.m.slots) };
    }

    fn self(ptr: *anyopaque) *Kolibri {
        return @ptrCast(@alignCast(ptr));
    }

    fn ops(x: *const Kolibri) kern.Ops {
        return .{ .k = x.k, .s = x.s };
    }

    fn laneBuf(x: *Kolibri, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(x.d, bytes);
        try x.lane_bufs.append(x.gpa, b);
        return b.ptr;
    }

    fn upload(x: *Kolibri, dst: u64, bytes: []const u8) !void {
        try x.d.check(x.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, x.s.handle), "cuMemcpyHtoDAsync");
    }

    /// Rows [row0, row0 + rows) of `logits` drawn into tok[at..], row r at `position0 + r` (argmax when greedy).
    fn drawRows(x: *Kolibri, l: *const Lane, logits: u64, row0: usize, rows: usize, position0: u64, at: usize, meta_at: usize) !void {
        const v = x.m.c.vocab;
        const o = x.ops();
        const src = logits + row0 * v * 4;
        if (!l.sampled) return o.argmax(src, rows, v, x.tok.ptr + at * 4);
        try o.toBf16(src, x.logits16.ptr, rows * v);
        const meta = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        const mp = x.meta.ptr + meta_at * 16;
        try x.upload(mp, std.mem.asBytes(&meta));
        try x.k.draws.draw(x.s, x.logits16.ptr, v, l.rule, mp, 0, x.tok.ptr + at * 4, rows, null, null);
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        const got = try x.lanes.getOrPut(x.gpa, s);
        if (got.found_existing) return error.StreamAlreadyPrefilled;
        errdefer _ = x.lanes.remove(s);
        const slot = for (x.free, 0..) |f, i| {
            if (f) break i;
        } else return error.NoFreeSlot;
        const v = x.m.c.vocab;
        got.value_ptr.* = .{ .slot = slot, .last = try x.laneBuf(v * 4), .rule = try x.laneBuf(@sizeOf(cuda.sample.Rule)) };
        x.free[slot] = false;
        const l = got.value_ptr;
        if (s.sampling) |sp| if (sp.temperature > 0) {
            const r = cuda.sample.rule(sp.seed, sp.temperature, sp.top_p, sp.minLog(), sp.top_k);
            try x.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        };
        const p = s.prompt();
        if (p.len == 0) return error.EmptyPrompt;
        var at: usize = 0;
        while (at < p.len) : (at += prompt_rows) {
            if (s.isCancelled()) return error.Cancelled;
            const end = @min(p.len, at + prompt_rows);
            try x.m.forward(x.ops(), &.{.{ .slot = slot, .p0 = at, .tokens = p[at..end] }}, true, null, l.last);
        }
        l.len = p.len;
        s.cached = 0;
    }

    fn handle(x: *Kolibri) usize {
        const h = x.next;
        x.next += 1;
        return @intCast(h % ring);
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        if (position != l.len) return error.PositionMismatch;
        const h = x.next;
        try x.drawRows(l, l.last, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn value(x: *Kolibri, feed: be.Feed) !u32 {
        return switch (feed) {
            .value => |v| v,
            .handle => |h| read(x, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        const x = self(ptr);
        const l = x.lanes.getPtr(s) orelse return error.UnknownStream;
        const t = [1]u32{try x.value(feed)};
        if (position != l.len + 1) return error.PositionMismatch;
        try x.m.forward(x.ops(), &.{.{ .slot = l.slot, .p0 = l.len, .tokens = &t }}, false, null, x.logits.ptr);
        l.len += 1;
        const h = x.next;
        try x.drawRows(l, x.logits.ptr, 0, 1, position, x.handle(), 0);
        return h;
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const x = self(ptr);
        var out: [1]u32 = undefined;
        try x.s.synchronize();
        try x.tok.download(@intCast((h % ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        const chains = try x.gpa.alloc(fwd.Chain, windows.len);
        defer x.gpa.free(chains);
        var total: usize = 0;
        for (windows, chains) |w, *c| {
            if (w.held != 0) return error.NoHeldDrafts; // no draft head: every draft comes from the host
            if (w.parents != null) return error.ChainsOnly;
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(x.gpa, w.pending);
            try l.rows.appendSlice(x.gpa, w.tokens);
            for (w.positions, 0..) |pos, r| if (pos != l.base + r + 1) return error.PositionMismatch;
            c.* = .{ .slot = l.slot, .p0 = l.base, .tokens = l.rows.items };
            total += l.rows.items.len;
        }
        if (total > round_rows) return error.RoundTooWide;
        const rows = try x.gpa.alloc(usize, total);
        defer x.gpa.free(rows);
        for (rows, 0..) |*r, i| r.* = i;
        try x.m.forward(x.ops(), chains, false, rows, x.logits.ptr);
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            const l = x.lanes.getPtr(w.stream).?;
            try x.drawRows(l, x.logits.ptr, at, l.rows.items.len, w.positions[0], ring + at, i);
            at += l.rows.items.len;
        }
        const got = try x.gpa.alloc(u32, total);
        defer x.gpa.free(got);
        try x.s.synchronize();
        try x.tok.download(ring * 4, std.mem.sliceAsBytes(got));
        at = 0;
        for (windows, out) |w, o| {
            const l = x.lanes.getPtr(w.stream).?;
            const n = l.rows.items.len;
            @memcpy(o.sampled[0..n], got[at..][0..n]);
            @memcpy(o.drafts, l.rows.items[1..]);
            l.len = l.base + n; // a chain keeps every row unless rolled back
            at += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        for (windows, paths) |w, path| {
            const l = x.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            for (path, 0..) |r, i| if (r != i) return error.ChainsOnly; // a chain keeps a prefix of its rows
            l.len = l.base + path.len; // caches are positional: later rows overwrite what the rejected ones wrote
        }
    }

    fn draft(_: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        for (requests) |r| if (r.depth != 0) return error.NoDraftHead;
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.lanes.fetchRemove(s)) |kv| {
            var l = kv.value;
            x.free[l.slot] = true;
            l.rows.deinit(x.gpa);
        }
    }
};
