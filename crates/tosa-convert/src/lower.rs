//! Lowering a [`SourceGraph`] to a TOSA [`Graph`].
//!
//! A [`LoweringRegistry`] maps source operator names to lowering functions.
//! The driver ([`lower_graph`]) declares the graph inputs, walks the nodes in
//! order, calls each node's lowering with a [`LowerCtx`] that resolves source
//! tensors to TOSA tensors (materializing constants on demand), marks the
//! outputs, and finally runs the cleanup passes.

use std::collections::{BTreeMap, HashMap};

use tosa_ir::{BlockBuilder, DType, Graph, GraphBuilder, Op, OpAttribute, TensorData, TensorRef};

use crate::error::{ConvertError, Result};
use crate::frontend::Frontend;
use crate::passes;
use crate::source::{SourceGraph, SourceNode, SourceTensor, TensorId};

/// A lowering: emits the TOSA operators for one source node into the context.
pub type LowerFn = Box<dyn Fn(&SourceNode, &mut LowerCtx<'_>) -> Result<()> + Send + Sync>;

/// A hook on a graph input or output: given the source tensor and the TOSA
/// tensor holding it (the declared input, or the lowered output), returns
/// the tensor to use instead (e.g. after a layout or representation change).
pub type IoHook = Box<dyn Fn(TensorId, TensorRef, &mut LowerCtx<'_>) -> Result<TensorRef> + Send + Sync>;

/// Lowerings keyed by source operator name, plus optional hooks on graph
/// inputs and outputs.
#[derive(Default)]
pub struct LoweringRegistry {
    lowerings: BTreeMap<String, LowerFn>,
    input_hook: Option<IoHook>,
    output_hook: Option<IoHook>,
}

impl LoweringRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the lowering for `op`, replacing any previous one.
    pub fn register<F>(&mut self, op: &str, f: F) -> &mut Self
    where
        F: Fn(&SourceNode, &mut LowerCtx<'_>) -> Result<()> + Send + Sync + 'static,
    {
        self.lowerings.insert(op.to_string(), Box::new(f));
        self
    }

    /// Run `f` on every graph input after it is declared.
    pub fn set_input_hook<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(TensorId, TensorRef, &mut LowerCtx<'_>) -> Result<TensorRef> + Send + Sync + 'static,
    {
        self.input_hook = Some(Box::new(f));
        self
    }

    /// Run `f` on every graph output before it is marked.
    pub fn set_output_hook<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(TensorId, TensorRef, &mut LowerCtx<'_>) -> Result<TensorRef> + Send + Sync + 'static,
    {
        self.output_hook = Some(Box::new(f));
        self
    }

    pub fn supports(&self, op: &str) -> bool {
        self.lowerings.contains_key(op)
    }

    pub fn get(&self, op: &str) -> Option<&LowerFn> {
        self.lowerings.get(op)
    }

    /// The supported operators, sorted.
    pub fn ops(&self) -> Vec<&str> {
        self.lowerings.keys().map(String::as_str).collect()
    }
}

/// Knobs for [`lower_graph`].
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Run the cleanup passes (transpose cancellation, constant folding,
    /// dead-code elimination) on the result.
    pub cleanup: bool,
    /// Largest constant, in bytes, that constant folding may create.
    pub max_folded_bytes: usize,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        ConvertOptions { cleanup: true, max_folded_bytes: 64 << 20 }
    }
}

/// Which source operators a registry covers, for a given graph.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Report {
    /// Supported operators and their node counts.
    pub supported: BTreeMap<String, usize>,
    /// Unsupported operators and their node counts.
    pub unsupported: BTreeMap<String, usize>,
}

impl Report {
    pub fn is_complete(&self) -> bool {
        self.unsupported.is_empty()
    }

    fn unsupported_error(&self) -> ConvertError {
        ConvertError::Unsupported(self.unsupported.iter().map(|(k, v)| (k.clone(), *v)).collect())
    }
}

/// Check which of `source`'s operators `registry` can lower, without lowering.
pub fn analyze(source: &SourceGraph, registry: &LoweringRegistry) -> Report {
    let mut report = Report::default();
    for (op, n) in source.op_counts() {
        let bucket = if registry.supports(&op) { &mut report.supported } else { &mut report.unsupported };
        bucket.insert(op, n);
    }
    report
}

/// The result of a conversion.
#[derive(Debug)]
pub struct Conversion {
    pub graph: Graph,
    pub report: Report,
}

/// Context handed to lowerings: the source graph, the TOSA block being built
/// and the mapping between them.
pub struct LowerCtx<'a> {
    pub source: &'a SourceGraph,
    pub block: BlockBuilder,
    values: HashMap<TensorId, TensorRef>,
    /// Constants computed during lowering (shape arithmetic resolved at
    /// conversion time), consulted before the source tensors' own data.
    computed: HashMap<TensorId, TensorData>,
    node: Option<&'a SourceNode>,
}

impl<'a> LowerCtx<'a> {
    fn new(source: &'a SourceGraph, block: BlockBuilder) -> Self {
        LowerCtx { source, block, values: HashMap::new(), computed: HashMap::new(), node: None }
    }

    /// The node being lowered.
    pub fn node(&self) -> Option<&'a SourceNode> {
        self.node
    }

    /// Record that source tensor `id` has the constant value `data`
    /// (computed by a lowering, e.g. a Shape operator on a static tensor).
    /// Later lowerings see it through [`LowerCtx::constant`], and
    /// [`LowerCtx::value`] materializes it as a CONST when needed.
    pub fn define_constant(&mut self, id: TensorId, data: TensorData) {
        self.computed.insert(id, data);
    }

    /// Whether source tensor `id` has a value known at conversion time.
    pub fn is_constant(&self, id: TensorId) -> bool {
        self.computed.contains_key(&id) || self.tensor(id).is_constant()
    }

    /// Whether source tensor `id` is already bound to a TOSA tensor (an
    /// input, an operator output or a materialized constant).
    pub fn is_available_as_value(&self, id: TensorId) -> bool {
        self.values.contains_key(&id)
    }

    /// The source tensor `id`.
    pub fn tensor(&self, id: TensorId) -> &'a SourceTensor {
        &self.source.tensors[id]
    }

    /// An error attributed to the node being lowered.
    pub fn fail<T>(&self, reason: impl Into<String>) -> Result<T> {
        let (op, node) = self.node.map(|n| (n.op.clone(), n.name.clone())).unwrap_or_default();
        Err(ConvertError::Lowering { op, node, reason: reason.into() })
    }

    /// The TOSA element type of source tensor `id`, which must have one.
    pub fn dtype(&self, id: TensorId) -> Result<DType> {
        let t = self.tensor(id);
        match t.dtype.to_tosa() {
            Some(d) => Ok(d),
            None => self.fail(format!("tensor '{}' has type {:?}, which TOSA lacks", t.name, t.dtype)),
        }
    }

    /// Whether source tensor `id` has been bound to a TOSA tensor (or is a
    /// constant, which can be materialized).
    pub fn is_available(&self, id: TensorId) -> bool {
        self.values.contains_key(&id) || self.is_constant(id)
    }

    /// The TOSA tensor holding source tensor `id`: a graph input, a value
    /// produced by an earlier node, or a constant (materialized as a CONST
    /// operator on first use).
    pub fn value(&mut self, id: TensorId) -> Result<TensorRef> {
        if let Some(v) = self.values.get(&id) {
            return Ok(v.clone());
        }
        let t = self.tensor(id);
        if let Some(data) = self.computed.get(&id) {
            // Shape of a computed constant: the source shape if static, else flat.
            let shape = if t.is_static() && t.rank() > 0 { t.tosa_shape() } else { vec![data.len() as i32] };
            let v = self.block.constant(&t.name, &shape, data.clone())?;
            self.values.insert(id, v.clone());
            return Ok(v);
        }
        let Some(data) = &t.constant else {
            return self.fail(format!("tensor '{}' is used before it is produced", t.name));
        };
        let dtype = self.dtype(id)?;
        if data.dtype() != dtype {
            return self.fail(format!(
                "constant '{}' holds {:?} data but is declared {:?}",
                t.name,
                data.dtype(),
                dtype
            ));
        }
        let v = self.block.constant(&t.name, &t.tosa_shape(), data.clone())?;
        self.values.insert(id, v.clone());
        Ok(v)
    }

    /// The constant data of source tensor `id`, for lowerings that need to
    /// read an operand (shapes, axes, quantization tables) at conversion time.
    pub fn constant(&self, id: TensorId) -> Result<&TensorData> {
        if let Some(d) = self.computed.get(&id) {
            return Ok(d);
        }
        let t = self.tensor(id);
        match &t.constant {
            Some(d) => Ok(d),
            None => self.fail(format!("operand '{}' must be a constant", t.name)),
        }
    }

    /// Integer constant operand `id` as `i64` values.
    pub fn constant_ints(&self, id: TensorId) -> Result<Vec<i64>> {
        let data = self.constant(id)?;
        let values = match data {
            TensorData::I4(v) | TensorData::I8(v) => v.iter().map(|&x| x as i64).collect(),
            TensorData::I16(v) => v.iter().map(|&x| x as i64).collect(),
            TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
            TensorData::I48(v) | TensorData::I64(v) => v.clone(),
            TensorData::Bool(v) => v.iter().map(|&x| x as i64).collect(),
            _ => return self.fail(format!("operand '{}' must be an integer constant", self.tensor(id).name)),
        };
        Ok(values)
    }

    /// Bind source tensor `id` to the TOSA tensor `value`.
    pub fn bind(&mut self, id: TensorId, value: TensorRef) {
        self.values.insert(id, value);
    }

    /// Emit a single-output operator whose output is source tensor `id`,
    /// declared with the source tensor's type and shape, and bind it.
    pub fn emit(&mut self, op: Op, attribute: OpAttribute, inputs: &[&TensorRef], id: TensorId) -> Result<TensorRef> {
        let t = self.tensor(id);
        let dtype = self.dtype(id)?;
        let out = self.block.op1(op, attribute, inputs, dtype, &t.tosa_shape(), &t.name);
        self.values.insert(id, out.clone());
        Ok(out)
    }

    /// Emit a single-output operator whose output is source tensor `id`, with
    /// an explicitly computed `shape` (for formats that do not record
    /// intermediate shapes) and `dtype`.
    pub fn emit_shaped(
        &mut self,
        op: Op,
        attribute: OpAttribute,
        inputs: &[&TensorRef],
        id: TensorId,
        dtype: DType,
        shape: &[i32],
    ) -> TensorRef {
        let name = self.tensor(id).name.clone();
        let out = self.block.op1(op, attribute, inputs, dtype, shape, &name);
        self.values.insert(id, out.clone());
        out
    }

    /// Emit a single-output operator producing an intermediate tensor (not a
    /// source tensor) of the given type and shape.
    pub fn emit_temp(
        &mut self,
        op: Op,
        attribute: OpAttribute,
        inputs: &[&TensorRef],
        dtype: DType,
        shape: &[i32],
        hint: &str,
    ) -> TensorRef {
        self.block.op1(op, attribute, inputs, dtype, shape, hint)
    }

    /// A SHAPE constant.
    pub fn shape_const(&mut self, values: &[i64], hint: &str) -> TensorRef {
        self.block.constant_shape(hint, values)
    }

    /// A single-element constant of `data`'s type (zero points, MUL's shift, ...).
    pub fn scalar(&mut self, data: TensorData, hint: &str) -> Result<TensorRef> {
        Ok(self.block.scalar(hint, data)?)
    }

    /// A zero constant of `dtype` with shape `[1]`, as zero-point operands of
    /// float operators require.
    pub fn zero(&mut self, dtype: DType, hint: &str) -> Result<TensorRef> {
        let data = TensorData::zeros(dtype, 1)?;
        self.scalar(data, hint)
    }
}

/// Lower `source` with `registry`.
pub fn lower_graph(source: &SourceGraph, registry: &LoweringRegistry, options: &ConvertOptions) -> Result<Conversion> {
    let report = analyze(source, registry);
    if !report.is_complete() {
        return Err(report.unsupported_error());
    }

    let mut ctx = LowerCtx::new(source, BlockBuilder::new("main"));
    for &id in &source.inputs {
        let t = ctx.tensor(id);
        let dtype = ctx.dtype(id)?;
        let mut v = ctx.block.input(&t.name, dtype, &t.tosa_shape());
        if let Some(hook) = &registry.input_hook {
            ctx.block.set_location(Some(t.name.clone()));
            v = hook(id, v, &mut ctx)?;
        }
        ctx.bind(id, v);
    }

    for node in &source.nodes {
        let lowering = registry.get(&node.op).expect("checked by analyze");
        ctx.node = Some(node);
        ctx.block.set_location(Some(node.name.clone()));
        lowering(node, &mut ctx).map_err(|e| match e {
            ConvertError::Lowering { .. } | ConvertError::Unsupported(_) => e,
            other => ConvertError::Lowering { op: node.op.clone(), node: node.name.clone(), reason: other.to_string() },
        })?;
        for &out in &node.outputs {
            if !ctx.values.contains_key(&out) && !ctx.computed.contains_key(&out) {
                let name = ctx.tensor(out).name.clone();
                return Err(ConvertError::Lowering {
                    op: node.op.clone(),
                    node: node.name.clone(),
                    reason: format!("lowering did not produce output '{name}'"),
                });
            }
        }
    }
    ctx.node = None;
    ctx.block.set_location(None);

    let mut emitted = std::collections::HashSet::new();
    for &id in &source.outputs {
        let mut v = ctx.value(id)?;
        if let Some(hook) = &registry.output_hook {
            ctx.block.set_location(Some(ctx.tensor(id).name.clone()));
            v = hook(id, v, &mut ctx)?;
            ctx.block.set_location(None);
        }
        let t = ctx.tensor(id);
        // Each graph output must be a distinct tensor that is not also an
        // input, and should carry the source model's output name.
        if !emitted.insert(v.name.clone()) || source.inputs.contains(&id) {
            v = ctx.block.op1(Op::Identity, OpAttribute::None, &[&v], v.dtype, &t.tosa_shape(), &t.name);
        } else if v.name != t.name
            && let Some(renamed) = ctx.block.rename(&v, &t.name)
        {
            v = renamed;
        }
        ctx.block.output(&v);
    }

    let mut graph = GraphBuilder::new(ctx.block.build()).build();
    if options.cleanup {
        passes::cleanup(&mut graph, options.max_folded_bytes)?;
    }
    Ok(Conversion { graph, report })
}

/// Parse `bytes` with `frontend` and lower the result.
pub fn convert(frontend: &dyn Frontend, bytes: &[u8], options: &ConvertOptions) -> Result<Conversion> {
    let source = frontend.load(bytes)?;
    lower_graph(&source, frontend.lowerings(), options)
}
