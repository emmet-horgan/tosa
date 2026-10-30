//! The TFLite frontend: parsing `.tflite` flatbuffers into a
//! [`SourceGraph`](crate::source::SourceGraph) and lowering TFLite's builtin
//! operators to TOSA. See [`loader`] for the parsing rules (including how
//! uint8 tensors are handled) and [`lowerings`] for the operator mappings.

/// FlatBuffers bindings for `schema/tflite.fbs` (TensorFlow v2.19.0),
/// regenerated with `cargo xtask codegen`.
#[allow(
    unknown_lints,
    unused_imports,
    dead_code,
    clippy::all,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unsafe_op_in_unsafe_fn,
    mismatched_lifetime_syntaxes
)]
pub(crate) mod generated {
    include!("generated/tflite_generated.rs");
}

pub mod loader;
pub mod lowerings;

pub(crate) use generated::tflite as fb;

use crate::error::Result;
use crate::frontend::Frontend;
use crate::lower::LoweringRegistry;
use crate::source::SourceGraph;

pub struct TfliteFrontend {
    lowerings: LoweringRegistry,
}

impl TfliteFrontend {
    pub fn new() -> Self {
        TfliteFrontend { lowerings: lowerings::registry() }
    }
}

impl Default for TfliteFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend for TfliteFrontend {
    fn name(&self) -> &str {
        "tflite"
    }

    fn extensions(&self) -> &[&str] {
        &["tflite", "lite"]
    }

    fn sniff(&self, bytes: &[u8]) -> bool {
        fb::model_buffer_has_identifier(bytes)
    }

    fn load(&self, bytes: &[u8]) -> Result<SourceGraph> {
        loader::load(bytes)
    }

    fn lowerings(&self) -> &LoweringRegistry {
        &self.lowerings
    }
}
