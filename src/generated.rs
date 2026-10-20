/// FlatBuffers bindings for `schema/tosa.fbs`. The file is checked in and
/// regenerated with `cargo xtask codegen` when the schema changes.
#[allow(
    unknown_lints, // mismatched_lifetime_syntaxes is newer than the MSRV
    unused_imports,
    dead_code,
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn,
    mismatched_lifetime_syntaxes
)]
pub mod tosa {
    include!("generated/tosa_generated.rs");
}

pub use tosa::tosa::*;
