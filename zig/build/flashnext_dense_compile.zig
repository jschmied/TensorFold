//! Compiles the Flash Next CUDA diagnostics for the provisional dense shape, so a MoE-only path cannot creep back in.
const std = @import("std");

pub fn steps(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, family: *std.Build.Module) void {
    const all = b.step("compile-flashnext-dense", "Compile the Flash Next CUDA diagnostics with cuda_shape.compiled = qwen4_dense");
    for ([_]struct { name: []const u8, root: []const u8 }{
        .{ .name = "embed", .root = "cuda_embed_main.zig" },
        .{ .name = "prompt", .root = "cuda_prompt_main.zig" },
        .{ .name = "compare", .root = "cuda_compare_main.zig" },
        .{ .name = "learn", .root = "cuda_learn_main.zig" },
    }) |check| {
        const root = b.createModule(.{ .root_source_file = b.path(b.fmt("zig/src/families/flashnext/{s}", .{check.root})), .target = target, .optimize = optimize, .link_libc = true });
        root.addImport("cuda", cuda);
        root.addImport("core", core);
        root.addImport("flashnext", family);
        all.dependOn(&b.addExecutable(.{ .name = b.fmt("flashnext-dense-{s}", .{check.name}), .root_module = root }).step);
    }
}
