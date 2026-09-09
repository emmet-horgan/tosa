#[allow(
    unused_imports,
    dead_code,
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals
)]
pub mod tosa {
    include!(concat!(env!("OUT_DIR"), "/tosa_generated.rs"));
}

pub use tosa::tosa::*;
