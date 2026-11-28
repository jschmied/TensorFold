//! Compile-only root of the core Metal blocks' contract suites: every hook analysed, nothing run.
comptime {
    _ = &@import("metal_core_rows.zig").Suite.create;
    _ = &@import("metal_core_glue.zig").Suite.create;
}
