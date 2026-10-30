//! The source graph: a frontend-neutral description of a model to lower.
//!
//! Each frontend parses its file format into a [`SourceGraph`]: tensors with
//! a type, shape, optional quantization parameters and optional constant
//! data, and nodes that name their operator as the source format spells it
//! (`"CONV_2D"`, `"Conv"`, `"aten.convolution"`, ...) with the format's own
//! attributes. Lowerings then map each node onto TOSA operators. Nothing in
//! here tries to unify the formats' operator sets: that is the lowerings'
//! job, so that each frontend's semantics stay exact.

use std::collections::BTreeMap;

use tosa_ir::{DType, TensorData};

use crate::error::{ConvertError, Result};

/// Index of a tensor in [`SourceGraph::tensors`].
pub type TensorId = usize;

/// Element type of a source tensor. Covers types TOSA has no counterpart for
/// so that frontends can describe a model faithfully and lowerings decide
/// how to handle them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceDType {
    Bool,
    I4,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    Str,
    Other(String),
}

impl SourceDType {
    /// The TOSA element type with the same representation, if there is one.
    pub fn to_tosa(&self) -> Option<DType> {
        Some(match self {
            SourceDType::Bool => DType::Bool,
            SourceDType::I4 => DType::I4,
            SourceDType::I8 => DType::I8,
            SourceDType::I16 => DType::I16,
            SourceDType::I32 => DType::I32,
            SourceDType::I64 => DType::I64,
            SourceDType::F16 => DType::F16,
            SourceDType::Bf16 => DType::Bf16,
            SourceDType::F32 => DType::F32,
            _ => return None,
        })
    }

    pub fn is_float(&self) -> bool {
        matches!(self, SourceDType::F16 | SourceDType::Bf16 | SourceDType::F32 | SourceDType::F64)
    }
}

/// Affine quantization: `real = scale * (q - zero_point)`, per tensor or per
/// channel along `axis`.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantization {
    pub scales: Vec<f64>,
    pub zero_points: Vec<i64>,
    /// The channel axis for per-channel parameters; `None` for per-tensor.
    pub axis: Option<usize>,
}

impl Quantization {
    pub fn per_tensor(scale: f64, zero_point: i64) -> Self {
        Quantization { scales: vec![scale], zero_points: vec![zero_point], axis: None }
    }

    pub fn is_per_channel(&self) -> bool {
        self.axis.is_some() && self.scales.len() > 1
    }

    /// The single scale of a per-tensor quantization.
    pub fn scale(&self) -> Option<f64> {
        (!self.is_per_channel()).then(|| self.scales[0])
    }

    /// The single zero point of a per-tensor quantization.
    pub fn zero_point(&self) -> Option<i64> {
        (!self.is_per_channel()).then(|| self.zero_points[0])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceTensor {
    pub name: String,
    pub dtype: SourceDType,
    /// `-1` marks a dynamic dimension.
    pub shape: Vec<i64>,
    pub quant: Option<Quantization>,
    /// Constant data (weights, biases, shape operands).
    pub constant: Option<TensorData>,
}

impl SourceTensor {
    pub fn new(name: &str, dtype: SourceDType, shape: &[i64]) -> Self {
        SourceTensor { name: name.to_string(), dtype, shape: shape.to_vec(), quant: None, constant: None }
    }

    pub fn with_constant(mut self, data: TensorData) -> Self {
        self.constant = Some(data);
        self
    }

    pub fn with_quant(mut self, quant: Quantization) -> Self {
        self.quant = Some(quant);
        self
    }

    pub fn is_constant(&self) -> bool {
        self.constant.is_some()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|&d| d >= 0)
    }

    /// The shape as TOSA declares it (`i32` dimensions, `-1` for dynamic).
    pub fn tosa_shape(&self) -> Vec<i32> {
        self.shape.iter().map(|&d| d.clamp(-1, i32::MAX as i64) as i32).collect()
    }
}

/// An attribute value of a source node.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Bytes(Vec<u8>),
}

pub type Attrs = BTreeMap<String, AttrValue>;

#[derive(Debug, Clone, PartialEq)]
pub struct SourceNode {
    /// The operator as the source format names it.
    pub op: String,
    /// The node's own name, for diagnostics and operator locations.
    pub name: String,
    /// Inputs in the source format's operand order; `None` for an omitted
    /// optional operand.
    pub inputs: Vec<Option<TensorId>>,
    pub outputs: Vec<TensorId>,
    pub attrs: Attrs,
}

impl SourceNode {
    pub fn new(op: &str, name: &str, inputs: Vec<Option<TensorId>>, outputs: Vec<TensorId>) -> Self {
        SourceNode { op: op.to_string(), name: name.to_string(), inputs, outputs, attrs: Attrs::new() }
    }

    pub fn with_attr(mut self, key: &str, value: AttrValue) -> Self {
        self.attrs.insert(key.to_string(), value);
        self
    }

    fn lowering_error(&self, reason: String) -> ConvertError {
        ConvertError::Lowering { op: self.op.clone(), node: self.name.clone(), reason }
    }

    /// The `i`th input, which must be present.
    pub fn input(&self, i: usize) -> Result<TensorId> {
        self.inputs.get(i).copied().flatten().ok_or_else(|| self.lowering_error(format!("input {i} is missing")))
    }

    /// The `i`th input if present.
    pub fn optional_input(&self, i: usize) -> Option<TensorId> {
        self.inputs.get(i).copied().flatten()
    }

    pub fn output(&self, i: usize) -> Result<TensorId> {
        self.outputs.get(i).copied().ok_or_else(|| self.lowering_error(format!("output {i} is missing")))
    }

    pub fn attr(&self, key: &str) -> Result<&AttrValue> {
        self.attrs.get(key).ok_or_else(|| self.lowering_error(format!("attribute '{key}' is missing")))
    }

    pub fn attr_int(&self, key: &str) -> Result<i64> {
        match self.attr(key)? {
            AttrValue::Int(v) => Ok(*v),
            AttrValue::Bool(b) => Ok(*b as i64),
            other => Err(self.lowering_error(format!("attribute '{key}' is {other:?}, expected an integer"))),
        }
    }

    pub fn attr_int_or(&self, key: &str, default: i64) -> Result<i64> {
        if self.attrs.contains_key(key) { self.attr_int(key) } else { Ok(default) }
    }

    pub fn attr_float(&self, key: &str) -> Result<f64> {
        match self.attr(key)? {
            AttrValue::Float(v) => Ok(*v),
            AttrValue::Int(v) => Ok(*v as f64),
            other => Err(self.lowering_error(format!("attribute '{key}' is {other:?}, expected a float"))),
        }
    }

    pub fn attr_bool(&self, key: &str) -> Result<bool> {
        match self.attr(key)? {
            AttrValue::Bool(v) => Ok(*v),
            AttrValue::Int(v) => Ok(*v != 0),
            other => Err(self.lowering_error(format!("attribute '{key}' is {other:?}, expected a bool"))),
        }
    }

    pub fn attr_str(&self, key: &str) -> Result<&str> {
        match self.attr(key)? {
            AttrValue::Str(v) => Ok(v),
            other => Err(self.lowering_error(format!("attribute '{key}' is {other:?}, expected a string"))),
        }
    }

    pub fn attr_ints(&self, key: &str) -> Result<&[i64]> {
        match self.attr(key)? {
            AttrValue::Ints(v) => Ok(v),
            other => Err(self.lowering_error(format!("attribute '{key}' is {other:?}, expected integers"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SourceGraph {
    pub name: String,
    pub tensors: Vec<SourceTensor>,
    /// Nodes in a valid execution order.
    pub nodes: Vec<SourceNode>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl SourceGraph {
    pub fn new(name: &str) -> Self {
        SourceGraph { name: name.to_string(), ..Default::default() }
    }

    pub fn add_tensor(&mut self, tensor: SourceTensor) -> TensorId {
        self.tensors.push(tensor);
        self.tensors.len() - 1
    }

    pub fn add_node(&mut self, node: SourceNode) {
        self.nodes.push(node);
    }

    pub fn tensor(&self, id: TensorId) -> &SourceTensor {
        &self.tensors[id]
    }

    /// How many nodes use each operator.
    pub fn op_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for n in &self.nodes {
            *counts.entry(n.op.clone()).or_insert(0) += 1;
        }
        counts
    }
}
