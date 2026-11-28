//! Runtime-shaped Metal blocks a family wires together: block products, a sigmoid MoE layer, row glue, attention.
pub const block_rows = @import("block_rows.zig");
pub const moe_sigmoid = @import("moe_sigmoid.zig");
pub const row_glue = @import("row_glue.zig");
pub const shared_attention = @import("shared_attention.zig");

test {
    _ = block_rows;
    _ = moe_sigmoid;
    _ = row_glue;
}
