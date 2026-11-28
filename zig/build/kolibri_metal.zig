//! Kolibri-1 on Metal: `tf-kolibri-metal` (the seeded-checkpoint selftest), its module tests and a compile check.
const std = @import("std");

fn modules(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module) [3]*std.Build.Module {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "tokenizer", .module = tokenizer }} });
    const blocks = b.createModule(.{ .root_source_file = b.path("zig/src/core/metal_blocks.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources } } });
    const family = b.createModule(.{ .root_source_file = b.path("zig/src/families/kolibri1/metal.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "metal_blocks", .module = blocks } } });
    const tool = b.createModule(.{ .root_source_file = b.path("zig/tests/kolibri1/metal_selftest.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "core", .module = core }, .{ .name = "kolibri1_metal", .module = family } } });
    return .{ family, tool, blocks };
}

/// macOS: `tf-kolibri-metal`, `test-kolibri-metal` (masks, ring, shapes; no shader) and `compile-kolibri-metal`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, sources: *std.Build.Module, test_step: *std.Build.Step) void {
    const m = modules(b, target, optimize, metal, sources);
    const exe = b.addExecutable(.{ .name = "tf-kolibri-metal", .root_module = m[1] });
    b.step("tf-kolibri-metal", "Kolibri-1 on Metal: the seeded checkpoint's forward against float64, resume and planted faults").dependOn(&b.addInstallArtifact(exe, .{}).step);
    const step = b.step("test-kolibri-metal", "Kolibri-1's Metal masks and ring, the core Metal blocks' shape refusals; no shader");
    for ([_]*std.Build.Module{ m[0], m[2] }) |root| {
        const run = b.addRunArtifact(b.addTest(.{ .root_module = root }));
        step.dependOn(&run.step);
        test_step.dependOn(&run.step);
    }
    // the check's own metal module links no framework: it compiles on a host without the macOS SDK too
    const bare = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const obj = b.addObject(.{ .name = "kolibri-metal-check", .root_module = modules(b, target, optimize, bare, sources)[1] });
    b.step("compile-kolibri-metal", "Compile tf-kolibri-metal without linking or running a shader").dependOn(&obj.step);
    const blocks = b.createModule(.{ .root_source_file = b.path("zig/src/core/metal_blocks.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = bare }, .{ .name = "kernel_sources", .module = sources } } });
    const suite = b.createModule(.{ .root_source_file = b.path("zig/tests/contract/core_blocks_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = bare }, .{ .name = "tensorfold", .module = blocks }, .{ .name = "contract", .module = @import("kernel_contract.zig").module(b, target, optimize) }, .{ .name = "kernel_sources", .module = sources } } });
    b.step("compile-core-blocks-contract", "Compile the core Metal blocks' contract suites without linking or a Metal compiler").dependOn(&b.addObject(.{ .name = "core-blocks-contract", .root_module = suite }).step);
}
