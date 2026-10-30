//! The ONNX frontend: a hand-written protobuf reader for `onnx.proto3`
//! ([`proto`]), a [`loader`] into the source graph, and [`lowerings`] for the
//! default-domain operators of float models.

pub mod loader;
pub mod lowerings;
pub mod proto;

use crate::error::Result;
use crate::frontend::Frontend;
use crate::lower::LoweringRegistry;
use crate::source::SourceGraph;

pub struct OnnxFrontend {
    lowerings: LoweringRegistry,
}

impl OnnxFrontend {
    pub fn new() -> Self {
        OnnxFrontend { lowerings: lowerings::registry() }
    }
}

impl Default for OnnxFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontend for OnnxFrontend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn extensions(&self) -> &[&str] {
        &["onnx"]
    }

    /// Protobuf has no magic number; a `ModelProto` starts with its
    /// `ir_version` field (tag 0x08) and must parse.
    fn sniff(&self, bytes: &[u8]) -> bool {
        bytes.first() == Some(&0x08) && proto::parse(bytes).map(|m| !m.graph.node.is_empty()).unwrap_or(false)
    }

    fn load(&self, bytes: &[u8]) -> Result<SourceGraph> {
        loader::load(bytes)
    }

    fn lowerings(&self) -> &LoweringRegistry {
        &self.lowerings
    }
}
