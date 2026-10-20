use crate::ir::{DType, Op};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub struct OpInfo {
    pub op: Op,
    pub op_index: usize,
}

pub trait InterpreterHook {
    fn pre_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor]) {}
    fn post_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor], _outputs: &[&Tensor]) {}
    fn on_tensor_alloc(&mut self, _name: &str, _shape: &[usize], _dtype: DType, _bytes: usize) {}
    fn on_tensor_free(&mut self, _name: &str, _bytes: usize) {}
}

/// A collection of hooks that dispatches to all of them in order.
pub struct HookSet {
    hooks: Vec<Box<dyn InterpreterHook>>,
}

impl HookSet {
    pub fn new() -> Self {
        HookSet { hooks: Vec::new() }
    }

    pub fn add<H: InterpreterHook + 'static>(&mut self, hook: H) {
        self.hooks.push(Box::new(hook));
    }

    pub fn hooks(&self) -> &[Box<dyn InterpreterHook>] {
        &self.hooks
    }

    pub fn hooks_mut(&mut self) -> &mut Vec<Box<dyn InterpreterHook>> {
        &mut self.hooks
    }

    pub fn into_hooks(self) -> Vec<Box<dyn InterpreterHook>> {
        self.hooks
    }
}

impl Default for HookSet {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpreterHook for HookSet {
    fn pre_op(&mut self, op: &OpInfo, inputs: &[&Tensor]) {
        for hook in &mut self.hooks {
            hook.pre_op(op, inputs);
        }
    }

    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        for hook in &mut self.hooks {
            hook.post_op(op, inputs, outputs);
        }
    }

    fn on_tensor_alloc(&mut self, name: &str, shape: &[usize], dtype: DType, bytes: usize) {
        for hook in &mut self.hooks {
            hook.on_tensor_alloc(name, shape, dtype, bytes);
        }
    }

    fn on_tensor_free(&mut self, name: &str, bytes: usize) {
        for hook in &mut self.hooks {
            hook.on_tensor_free(name, bytes);
        }
    }
}

// Also implement for Vec<Box<dyn InterpreterHook>> directly
impl InterpreterHook for Vec<Box<dyn InterpreterHook>> {
    fn pre_op(&mut self, op: &OpInfo, inputs: &[&Tensor]) {
        for hook in self.iter_mut() {
            hook.pre_op(op, inputs);
        }
    }

    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        for hook in self.iter_mut() {
            hook.post_op(op, inputs, outputs);
        }
    }

    fn on_tensor_alloc(&mut self, name: &str, shape: &[usize], dtype: DType, bytes: usize) {
        for hook in self.iter_mut() {
            hook.on_tensor_alloc(name, shape, dtype, bytes);
        }
    }

    fn on_tensor_free(&mut self, name: &str, bytes: usize) {
        for hook in self.iter_mut() {
            hook.on_tensor_free(name, bytes);
        }
    }
}

pub struct NoopHook;
impl InterpreterHook for NoopHook {}

// ---------------------------------------------------------------------------
// Built-in collectors
// ---------------------------------------------------------------------------

/// Counts ops by type, tracks timing and element throughput.
#[derive(Debug, Clone, Default)]
pub struct OpStats {
    pub count: u64,
    pub total_time: Duration,
    pub total_input_elements: u64,
    pub total_output_elements: u64,
}

#[derive(Debug, Clone)]
pub struct StatsCollector {
    pub op_stats: HashMap<&'static str, OpStats>,
    pub peak_memory_bytes: usize,
    pub current_memory_bytes: usize,
    pub total_ops_executed: u64,
    current_op_start: Option<Instant>,
}

impl StatsCollector {
    pub fn new() -> Self {
        StatsCollector {
            op_stats: HashMap::new(),
            peak_memory_bytes: 0,
            current_memory_bytes: 0,
            total_ops_executed: 0,
            current_op_start: None,
        }
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Total ops executed: {}", self.total_ops_executed));
        lines.push(format!("Peak memory: {} bytes", self.peak_memory_bytes));
        lines.push(String::new());

        let mut sorted: Vec<_> = self.op_stats.iter().collect();
        sorted.sort_by_key(|(name, _)| *name);

        for (name, stats) in sorted {
            lines.push(format!(
                "  {:<16} count={:<6} time={:>8.3}ms  elements_in={:<10} elements_out={:<10}",
                name,
                stats.count,
                stats.total_time.as_secs_f64() * 1000.0,
                stats.total_input_elements,
                stats.total_output_elements,
            ));
        }
        lines.join("\n")
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl InterpreterHook for StatsCollector {
    fn pre_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor]) {
        self.current_op_start = Some(Instant::now());
    }

    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        let elapsed = self.current_op_start.map(|s| s.elapsed()).unwrap_or_default();

        let entry = self.op_stats.entry(op.op.name()).or_default();
        entry.count += 1;
        entry.total_time += elapsed;
        entry.total_input_elements += inputs.iter().map(|t| t.element_count() as u64).sum::<u64>();
        entry.total_output_elements += outputs.iter().map(|t| t.element_count() as u64).sum::<u64>();

        self.total_ops_executed += 1;
    }

    fn on_tensor_alloc(&mut self, _name: &str, _shape: &[usize], _dtype: DType, bytes: usize) {
        self.current_memory_bytes += bytes;
        self.peak_memory_bytes = self.peak_memory_bytes.max(self.current_memory_bytes);
    }

    fn on_tensor_free(&mut self, _name: &str, bytes: usize) {
        self.current_memory_bytes = self.current_memory_bytes.saturating_sub(bytes);
    }
}

/// Tracks estimated FLOPs for compute-heavy ops.
#[derive(Debug, Clone, Default)]
pub struct FlopCounter {
    pub total_flops: u64,
    pub flops_by_op: HashMap<&'static str, u64>,
}

impl FlopCounter {
    pub fn new() -> Self {
        Self::default()
    }
}

impl InterpreterHook for FlopCounter {
    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        let flops = estimate_flops(op.op, inputs, outputs);
        self.total_flops += flops;
        *self.flops_by_op.entry(op.op.name()).or_default() += flops;
    }
}

fn estimate_flops(op: Op, inputs: &[&Tensor], outputs: &[&Tensor]) -> u64 {
    match op {
        // Element-wise: 1 flop per output element
        Op::Add | Op::Sub | Op::Mul | Op::Abs | Op::Negate | Op::Maximum | Op::Minimum => {
            outputs.first().map(|t| t.element_count() as u64).unwrap_or(0)
        }
        // Transcendentals: ~10 flops per element (approximation)
        Op::Exp
        | Op::Log
        | Op::Sin
        | Op::Cos
        | Op::Tanh
        | Op::Sigmoid
        | Op::Erf
        | Op::Rsqrt
        | Op::Reciprocal
        | Op::Pow => outputs.first().map(|t| t.element_count() as u64 * 10).unwrap_or(0),
        // MATMUL: [N, H, C] x [N, C, W] → 2*N*H*W*C flops
        Op::MatMul => {
            if inputs.len() >= 2 && inputs[0].shape.len() == 3 {
                let n = inputs[0].shape[0] as u64;
                let h = inputs[0].shape[1] as u64;
                let c = inputs[0].shape[2] as u64;
                let w = inputs[1].shape[2] as u64;
                2 * n * h * w * c
            } else {
                0
            }
        }
        // CONV2D: 2 * N * OH * OW * OC * KH * KW * IC
        Op::Conv2d => {
            if inputs.len() >= 2 && !outputs.is_empty() {
                let out = &outputs[0];
                if out.shape.len() == 4 && inputs[1].shape.len() == 4 {
                    let n = out.shape[0] as u64;
                    let oh = out.shape[1] as u64;
                    let ow = out.shape[2] as u64;
                    let oc = out.shape[3] as u64;
                    let kh = inputs[1].shape[1] as u64;
                    let kw = inputs[1].shape[2] as u64;
                    let ic = inputs[1].shape[3] as u64;
                    2 * n * oh * ow * oc * kh * kw * ic
                } else {
                    0
                }
            } else {
                0
            }
        }
        _ => 0,
    }
}

/// Records a trace of every op execution with tensor shapes.
#[derive(Debug, Clone, Default)]
pub struct OpTrace {
    pub entries: Vec<TraceEntry>,
}

#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub op: &'static str,
    pub op_index: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub elapsed: Duration,
}

impl OpTrace {
    pub fn new() -> Self {
        Self::default()
    }
}

impl InterpreterHook for OpTrace {
    fn pre_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor]) {}

    fn post_op(&mut self, op: &OpInfo, inputs: &[&Tensor], outputs: &[&Tensor]) {
        self.entries.push(TraceEntry {
            op: op.op.name(),
            op_index: op.op_index,
            input_shapes: inputs.iter().map(|t| t.shape.clone()).collect(),
            output_shapes: outputs.iter().map(|t| t.shape.clone()).collect(),
            elapsed: Duration::ZERO, // timing is handled by StatsCollector
        });
    }
}
