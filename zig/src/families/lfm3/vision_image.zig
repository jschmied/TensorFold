//! LFM3-VL's images as its processor makes them: core/picture/patches with torchvision's resize, fused normalization.
const std = @import("std");
const core = @import("core");
const patches = core.picture.patches;

/// The tower's fixed shape (the admission checks config.json's vision_config against it).
pub const hidden = 1024;
pub const heads = 16;
pub const kv_heads = 8;
pub const layers = 14;
pub const intermediate = 2560;
pub const patch = 16;
pub const temporal = 2;
pub const merge = 2;

pub const Error = error{VisionConfig};

/// The marker ids a prompt's image rows sit between, and the one each image row holds (config.json).
pub const Marks = struct { image: u32, start: u32, end: u32 };

pub fn marks(arena: std.mem.Allocator, config_json: []const u8) Error!Marks {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, config_json, .{}) catch return error.VisionConfig;
    if (doc != .object) return error.VisionConfig;
    return .{ .image = try id(doc.object, "image_token_id"), .start = try id(doc.object, "vision_start_token_id"), .end = try id(doc.object, "vision_end_token_id") };
}

/// processor_config.json's image_processor, which must be the processor this port matches bit for bit.
pub fn processor(arena: std.mem.Allocator, processor_json: []const u8) Error!patches.Patches {
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, processor_json, .{}) catch return error.VisionConfig;
    if (doc != .object) return error.VisionConfig;
    const ip = doc.object.get("image_processor") orelse return error.VisionConfig;
    if (ip != .object) return error.VisionConfig;
    const o = ip.object;
    const want = [_]struct { []const u8, i64 }{ .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "merge_size", merge }, .{ "resample", 3 } };
    for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.VisionConfig;
        if (v != .integer or v.integer != kv[1]) return error.VisionConfig;
    }
    const size = o.get("size") orelse return error.VisionConfig;
    if (size != .object) return error.VisionConfig;
    var p: patches.Patches = .{ .patch = patch, .temporal = temporal, .merge = merge, .min_pixels = try id(size.object, "shortest_edge"), .max_pixels = try id(size.object, "longest_edge"), .rescale = try number(o, "rescale_factor"), .mean = undefined, .std = undefined, .resize = .torch, .fused = true };
    inline for (.{ "image_mean", "image_std" }, .{ &p.mean, &p.std }) |key, dst| {
        const v = o.get(key) orelse return error.VisionConfig;
        if (v != .array or v.array.items.len != 3) return error.VisionConfig;
        for (v.array.items, 0..) |x, i| dst[i] = @floatCast(switch (x) {
            .float => |f| f,
            .integer => |n| @as(f64, @floatFromInt(n)),
            else => return error.VisionConfig,
        });
    }
    for (p.std) |x| if (x == 0) return error.VisionConfig;
    if (p.min_pixels > p.max_pixels or p.rescale <= 0) return error.VisionConfig;
    return p;
}

fn id(o: std.json.ObjectMap, key: []const u8) Error!u32 {
    const v = o.get(key) orelse return error.VisionConfig;
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.VisionConfig;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, key: []const u8) Error!f64 {
    return switch (o.get(key) orelse return error.VisionConfig) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => error.VisionConfig,
    };
}

test {
    _ = @import("vision_image_test.zig");
}
