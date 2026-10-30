//! Constructing graphs programmatically.
//!
//! A [`BlockBuilder`] accumulates the tensors and operators of one block and
//! hands out [`TensorRef`]s, which carry a tensor's name, element type and
//! shape so that code emitting operators can reason about what it has
//! produced. A [`GraphBuilder`] assembles blocks into a [`Graph`].
//!
//! ```
//! use tosa_ir::{BlockBuilder, DType, GraphBuilder, Op, OpAttribute, TensorData};
//!
//! let mut b = BlockBuilder::new("main");
//! let x = b.input("x", DType::F32, &[1, 4]);
//! let one = b.constant("one", &[1, 1], TensorData::F32(vec![1.0])).unwrap();
//! let y = b.op1(Op::Add, OpAttribute::None, &[&x, &one], DType::F32, &[1, 4], "y");
//! b.output(&y);
//! let graph = GraphBuilder::new(b.build()).build();
//! let bytes = graph.to_flatbuffer().unwrap();
//! assert_eq!(tosa_ir::Graph::from_flatbuffer(&bytes).unwrap(), graph);
//! ```

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::ir::{BasicBlock, DType, Graph, Op, OpAttribute, Operator, Region, TensorDef, Version};
use crate::tensor::{TensorData, checked_element_count};

/// A tensor declared in a block under construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub dtype: DType,
    /// Declared shape; `-1` marks a dynamic dimension.
    pub shape: Vec<i32>,
}

impl TensorRef {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Element count, or `None` if any dimension is dynamic.
    pub fn element_count(&self) -> Option<usize> {
        if self.shape.iter().any(|&d| d < 0) {
            return None;
        }
        checked_element_count(&self.shape.iter().map(|&d| d as usize).collect::<Vec<_>>()).ok()
    }
}

/// Builder for one block (the main graph or a control-flow sub-graph).
#[derive(Debug)]
pub struct BlockBuilder {
    name: String,
    tensors: Vec<TensorDef>,
    operators: Vec<Operator>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    names: HashSet<String>,
    /// Location recorded on operators added from now on (see [`BlockBuilder::set_location`]).
    location: Option<String>,
}

impl BlockBuilder {
    pub fn new(name: &str) -> Self {
        BlockBuilder {
            name: name.to_string(),
            tensors: Vec::new(),
            operators: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            names: HashSet::new(),
            location: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// A tensor name that is unused in this block: `hint` itself if free,
    /// otherwise `hint`, `_1`, `_2`, ... appended.
    pub fn unique_name(&mut self, hint: &str) -> String {
        let hint = if hint.is_empty() { "t" } else { hint };
        let mut name = hint.to_string();
        let mut n = 0;
        while !self.names.insert(name.clone()) {
            n += 1;
            name = format!("{hint}_{n}");
        }
        name
    }

    /// Record `location` on every operator added until the next call
    /// (`None` to stop). Converters use it for the source node's name.
    pub fn set_location(&mut self, location: Option<String>) {
        self.location = location;
    }

    fn declare_def(&mut self, hint: &str, dtype: DType, shape: &[i32], data: Option<Vec<u8>>) -> TensorRef {
        let name = self.unique_name(hint);
        self.tensors.push(TensorDef {
            name: name.clone(),
            shape: shape.to_vec(),
            dtype,
            data,
            is_variable: false,
            is_unranked: false,
            variable_name: None,
        });
        TensorRef { name, dtype, shape: shape.to_vec() }
    }

    /// Declare a block input.
    pub fn input(&mut self, hint: &str, dtype: DType, shape: &[i32]) -> TensorRef {
        let t = self.declare_def(hint, dtype, shape, None);
        self.inputs.push(t.name.clone());
        t
    }

    /// Declare an intermediate tensor (to be produced by an operator added
    /// with [`BlockBuilder::op`]).
    pub fn declare(&mut self, hint: &str, dtype: DType, shape: &[i32]) -> TensorRef {
        self.declare_def(hint, dtype, shape, None)
    }

    /// Add a constant tensor: its declaration with data, and the CONST
    /// operator producing it.
    pub fn constant(&mut self, hint: &str, shape: &[i32], data: TensorData) -> Result<TensorRef> {
        if shape.iter().any(|&d| d < 0) {
            return Err(Error::ShapeMismatch(format!("constant '{hint}' has a dynamic shape {shape:?}")));
        }
        let count = checked_element_count(&shape.iter().map(|&d| d as usize).collect::<Vec<_>>())?;
        if data.len() != count {
            return Err(Error::ShapeMismatch(format!(
                "constant '{hint}' has {} elements but shape {shape:?}",
                data.len()
            )));
        }
        let dtype = data.dtype();
        let t = self.declare_def(hint, dtype, shape, Some(data.to_raw_bytes()));
        self.push_op(Op::Const, OpAttribute::None, vec![], vec![t.name.clone()]);
        Ok(t)
    }

    /// Add a constant SHAPE value (a rank-1 list of dimensions) and the
    /// CONST_SHAPE operator producing it.
    pub fn constant_shape(&mut self, hint: &str, values: &[i64]) -> TensorRef {
        let data = TensorData::I64(values.to_vec()).to_raw_bytes();
        let t = self.declare_def(hint, DType::Shape, &[values.len() as i32], Some(data));
        self.push_op(Op::ConstShape, OpAttribute::None, vec![], vec![t.name.clone()]);
        t
    }

    /// Add a constant holding a single element of `dtype`, as operators such
    /// as zero points and MUL's shift expect (shape `[1]`).
    pub fn scalar(&mut self, hint: &str, data: TensorData) -> Result<TensorRef> {
        self.constant(hint, &[1], data)
    }

    fn push_op(&mut self, op: Op, attribute: OpAttribute, inputs: Vec<String>, outputs: Vec<String>) -> &mut Operator {
        let mut operator = Operator::new(op, inputs, outputs, attribute);
        operator.location = self.location.clone();
        self.operators.push(operator);
        self.operators.last_mut().expect("just pushed")
    }

    /// Add an operator reading `inputs` and producing the already declared
    /// `outputs`. Returns the operator for further adjustment.
    pub fn op(
        &mut self,
        op: Op,
        attribute: OpAttribute,
        inputs: &[&TensorRef],
        outputs: &[&TensorRef],
    ) -> &mut Operator {
        let inputs = inputs.iter().map(|t| t.name.clone()).collect();
        let outputs = outputs.iter().map(|t| t.name.clone()).collect();
        self.push_op(op, attribute, inputs, outputs)
    }

    /// Add a single-output operator, declaring its output as `out_dtype` of
    /// `out_shape` under a name derived from `hint`.
    pub fn op1(
        &mut self,
        op: Op,
        attribute: OpAttribute,
        inputs: &[&TensorRef],
        out_dtype: DType,
        out_shape: &[i32],
        hint: &str,
    ) -> TensorRef {
        let out = self.declare(hint, out_dtype, out_shape);
        self.op(op, attribute, inputs, &[&out]);
        out
    }

    /// Mark `t` as a block output.
    pub fn output(&mut self, t: &TensorRef) {
        self.outputs.push(t.name.clone());
    }

    /// Whether `name` is declared in this block.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// Rename tensor `t` to `new_name`, updating every operator that refers
    /// to it. Returns `None`, changing nothing, if `new_name` is taken.
    pub fn rename(&mut self, t: &TensorRef, new_name: &str) -> Option<TensorRef> {
        if new_name == t.name {
            return Some(t.clone());
        }
        if !self.names.insert(new_name.to_string()) {
            return None;
        }
        self.names.remove(&t.name);
        for def in &mut self.tensors {
            if def.name == t.name {
                def.name = new_name.to_string();
            }
        }
        for op in &mut self.operators {
            for name in op.inputs.iter_mut().chain(op.outputs.iter_mut()) {
                if *name == t.name {
                    *name = new_name.to_string();
                }
            }
        }
        for name in self.inputs.iter_mut().chain(self.outputs.iter_mut()) {
            if *name == t.name {
                *name = new_name.to_string();
            }
        }
        Some(TensorRef { name: new_name.to_string(), dtype: t.dtype, shape: t.shape.clone() })
    }

    pub fn build(self) -> BasicBlock {
        BasicBlock {
            name: self.name,
            operators: self.operators,
            tensors: self.tensors,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }
}

/// Builder for a [`Graph`]: a main block plus any control-flow sub-graphs,
/// all in one region.
#[derive(Debug)]
pub struct GraphBuilder {
    version: Version,
    region: String,
    blocks: Vec<BasicBlock>,
}

impl GraphBuilder {
    /// Start a graph whose entry block is `main`.
    pub fn new(main: BasicBlock) -> Self {
        GraphBuilder { version: Version::CURRENT, region: "main".to_string(), blocks: vec![main] }
    }

    pub fn version(mut self, version: Version) -> Self {
        self.version = version;
        self
    }

    pub fn region_name(mut self, name: &str) -> Self {
        self.region = name.to_string();
        self
    }

    /// Add a sub-graph for COND_IF / WHILE_LOOP to refer to by its block name.
    pub fn add_block(&mut self, block: BasicBlock) -> &mut Self {
        self.blocks.push(block);
        self
    }

    pub fn build(self) -> Graph {
        Graph { version: self.version, regions: vec![Region { name: self.region, blocks: self.blocks }] }
    }
}
