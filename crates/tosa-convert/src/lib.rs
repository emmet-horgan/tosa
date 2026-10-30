//! Lowering other model formats to TOSA.
//!
//! The crate is a framework plus frontends:
//!
//! - A [`Frontend`] parses one file format into a [`SourceGraph`], a plain
//!   description of the model's tensors and nodes in the format's own terms,
//!   and owns a [`LoweringRegistry`] mapping the format's operators to TOSA.
//! - [`lower_graph`] drives the lowering: it declares the inputs, calls each
//!   node's lowering with a [`LowerCtx`] that resolves source tensors to TOSA
//!   tensors and materializes constants on demand, marks the outputs, and
//!   runs the cleanup [`passes`].
//! - [`quant`] and [`layout`] hold the arithmetic every frontend needs:
//!   RESCALE multipliers and shifts in the TensorFlow legalization's
//!   convention, and channels-first to channels-last conversions.
//! - [`analyze`] reports which operators of a model are supported before
//!   any work is done.
//!
//! Frontends are compiled in behind Cargo features and registered in
//! [`default_frontends`].

pub mod error;
pub mod frontend;
pub mod layout;
pub mod lower;
pub mod passes;
pub mod quant;
pub mod source;

pub use error::{ConvertError, Result};
pub use frontend::{Frontend, FrontendSet};
pub use lower::{
    Conversion, ConvertOptions, LowerCtx, LowerFn, LoweringRegistry, Report, analyze, convert, lower_graph,
};
pub use source::{AttrValue, Attrs, Quantization, SourceDType, SourceGraph, SourceNode, SourceTensor, TensorId};

/// The frontends built into this crate (selected by Cargo features).
pub fn default_frontends() -> FrontendSet {
    FrontendSet::new()
}
