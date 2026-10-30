//! Cleanup passes on lowered graphs.
//!
//! Lowerings emit straightforward code; these passes tidy the result:
//!
//! - [`cancel_transposes`]: a TRANSPOSE of a TRANSPOSE becomes one TRANSPOSE
//!   with the composed permutation, or disappears when that is the identity.
//!   Layout conversions inserted around every channels-first operator cancel
//!   this way.
//! - [`fold_constants`]: an operator whose inputs are all constants is
//!   evaluated with the interpreter and replaced by a CONST (weight layout
//!   transposes, reshaped biases, computed shape operands).
//! - [`eliminate_dead_code`]: operators whose outputs nobody uses, and
//!   tensor declarations nobody refers to, are removed.

use std::collections::{HashMap, HashSet};

use tosa_interp::{Interpreter, NoopHook};
use tosa_ir::tensor::checked_element_count;
use tosa_ir::{BasicBlock, DType, Graph, Op, OpAttribute, Operator, Region, TensorDef, Version};

use crate::error::Result;
use crate::layout::{compose_perms, inverse_perms, is_identity};

/// Run every cleanup pass on every block of `graph`.
pub fn cleanup(graph: &mut Graph, max_folded_bytes: usize) -> Result<()> {
    for region in &mut graph.regions {
        for block in &mut region.blocks {
            loop {
                let changed = cancel_transposes(block) + sink_transposes(block);
                if changed == 0 {
                    break;
                }
            }
            fold_constants(block, max_folded_bytes)?;
            eliminate_dead_code(block);
        }
    }
    Ok(())
}

/// Operators that act element by element, so a TRANSPOSE commutes with them
/// (every data operand permuted the same way; rank-1 single-element operands
/// such as MUL's shift and zero points are left alone).
fn is_elementwise(op: &Operator) -> bool {
    match op.op {
        Op::Abs
        | Op::Negate
        | Op::Ceil
        | Op::Floor
        | Op::Exp
        | Op::Log
        | Op::Sin
        | Op::Cos
        | Op::Reciprocal
        | Op::Rsqrt
        | Op::Erf
        | Op::Sigmoid
        | Op::Tanh
        | Op::Clz
        | Op::BitwiseNot
        | Op::LogicalNot
        | Op::Clamp
        | Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Maximum
        | Op::Minimum
        | Op::Pow
        | Op::Intdiv
        | Op::LogicalAnd
        | Op::LogicalOr
        | Op::LogicalXor
        | Op::BitwiseAnd
        | Op::BitwiseOr
        | Op::BitwiseXor
        | Op::Equal
        | Op::Greater
        | Op::GreaterEqual
        | Op::Select
        | Op::Cast
        | Op::Table => true,
        Op::Rescale => matches!(op.attribute, OpAttribute::Rescale { per_channel: false, .. }),
        _ => false,
    }
}

/// Move TRANSPOSE operators from the inputs of element-wise operators to
/// their outputs: `E(T(x), T(y))` becomes `T(E(x, y))`. Constant operands of
/// the same rank are transposed instead (and folded later); operands that
/// are single elements are left as they are. Layout conversions inserted
/// around channels-first operators then meet and cancel. Returns how many
/// operators were rewritten.
pub fn sink_transposes(block: &mut BasicBlock) -> usize {
    let mut changed = 0;
    loop {
        let producers: HashMap<String, usize> = producers(block).into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let defs: HashMap<String, usize> = block.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        let constants: HashSet<String> =
            block.operators.iter().filter(|op| op.op == Op::Const).flat_map(|op| op.outputs.iter().cloned()).collect();
        let uses = use_counts(block);

        // Find an element-wise operator with at least one TRANSPOSE operand
        // and every data operand either transposed by the same permutation
        // or constant.
        let mut candidate = None;
        'ops: for (i, op) in block.operators.iter().enumerate() {
            if !is_elementwise(op) || op.outputs.len() != 1 {
                continue;
            }
            let out_rank = defs.get(&op.outputs[0]).map(|&d| block.tensors[d].shape.len());
            let Some(out_rank) = out_rank else { continue };
            let mut perms: Option<Vec<i32>> = None;
            for input in &op.inputs {
                let rank = defs.get(input).map(|&d| block.tensors[d].shape.len());
                if rank != Some(out_rank) {
                    // A single-element operand (shift, zero point): untouched.
                    if rank == Some(1) && defs.get(input).map(|&d| block.tensors[d].shape == [1]).unwrap_or(false) {
                        continue;
                    }
                    continue 'ops;
                }
                if constants.contains(input) {
                    continue;
                }
                let Some(&p) = producers.get(input) else { continue 'ops };
                let t = &block.operators[p];
                let OpAttribute::Transpose { perms: tp } = &t.attribute else { continue 'ops };
                if t.op != Op::Transpose || uses.get(input.as_str()).copied().unwrap_or(0) != 1 {
                    continue 'ops;
                }
                match &perms {
                    None => perms = Some(tp.clone()),
                    Some(existing) if existing == tp => {}
                    Some(_) => continue 'ops,
                }
            }
            if let Some(p) = perms {
                candidate = Some((i, p));
                break;
            }
        }
        let Some((i, perms)) = candidate else { break };
        changed += 1;
        let inverse = inverse_perms(&perms);

        // Rewrite: operands become the transposes' sources (constants get an
        // explicit transpose), the operator's output takes the untransposed
        // shape under a new name, and a transpose restores the old output.
        let out_name = block.operators[i].outputs[0].clone();
        let out_def = defs[&out_name];
        let out_dtype = block.tensors[out_def].dtype;
        let old_shape = block.tensors[out_def].shape.clone();
        let new_shape: Vec<i32> = inverse.iter().map(|&p| old_shape[p as usize]).collect();
        let inner_name = unique_name(block, &format!("{out_name}_pre"));
        let mut new_ops: Vec<Operator> = Vec::new();
        let inputs = block.operators[i].inputs.clone();
        let mut new_inputs = Vec::with_capacity(inputs.len());
        for input in inputs {
            let rank = block.tensors[defs[&input]].shape.len();
            if rank != old_shape.len() {
                new_inputs.push(input);
            } else if constants.contains(&input) {
                let c_shape = block.tensors[defs[&input]].shape.clone();
                let t_shape: Vec<i32> = inverse.iter().map(|&p| c_shape[p as usize]).collect();
                let t_name = unique_name(block, &format!("{input}_t"));
                block.tensors.push(TensorDef {
                    name: t_name.clone(),
                    shape: t_shape,
                    dtype: block.tensors[defs[&input]].dtype,
                    data: None,
                    is_variable: false,
                    is_unranked: false,
                    variable_name: None,
                });
                new_ops.push(Operator::new(
                    Op::Transpose,
                    vec![input],
                    vec![t_name.clone()],
                    OpAttribute::Transpose { perms: inverse.clone() },
                ));
                new_inputs.push(t_name);
            } else {
                let p = producers[&input];
                new_inputs.push(block.operators[p].inputs[0].clone());
            }
        }
        block.tensors.push(TensorDef {
            name: inner_name.clone(),
            shape: new_shape,
            dtype: out_dtype,
            data: None,
            is_variable: false,
            is_unranked: false,
            variable_name: None,
        });
        let location = block.operators[i].location.clone();
        block.operators[i].inputs = new_inputs;
        block.operators[i].outputs = vec![inner_name.clone()];
        let mut restore = Operator::new(
            Op::Transpose,
            vec![inner_name],
            vec![out_name],
            OpAttribute::Transpose { perms: perms.clone() },
        );
        restore.location = location.clone();
        for op in &mut new_ops {
            op.location = location.clone();
        }
        // Constant transposes go before the operator, the restoring transpose after it.
        let n = new_ops.len();
        for (k, op) in new_ops.into_iter().enumerate() {
            block.operators.insert(i + k, op);
        }
        block.operators.insert(i + n + 1, restore);
        eliminate_dead_code(block);
    }
    changed
}

/// A tensor name not yet used in `block`.
fn unique_name(block: &BasicBlock, hint: &str) -> String {
    let taken: HashSet<&str> = block.tensors.iter().map(|t| t.name.as_str()).collect();
    if !taken.contains(hint) {
        return hint.to_string();
    }
    (1..).map(|n| format!("{hint}_{n}")).find(|c| !taken.contains(c.as_str())).unwrap()
}

/// Operator index producing each tensor.
fn producers(block: &BasicBlock) -> HashMap<&str, usize> {
    let mut map = HashMap::new();
    for (i, op) in block.operators.iter().enumerate() {
        for out in &op.outputs {
            map.insert(out.as_str(), i);
        }
    }
    map
}

/// How many operator inputs and block outputs refer to each tensor.
fn use_counts(block: &BasicBlock) -> HashMap<&str, usize> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for op in &block.operators {
        for input in &op.inputs {
            *counts.entry(input.as_str()).or_default() += 1;
        }
    }
    for out in &block.outputs {
        *counts.entry(out.as_str()).or_default() += 1;
    }
    counts
}

/// Rename every reference to tensor `from` as `to` and drop `from`'s declaration.
fn replace_tensor(block: &mut BasicBlock, from: &str, to: &str) {
    for op in &mut block.operators {
        for name in op.inputs.iter_mut().chain(op.outputs.iter_mut()) {
            if name == from {
                *name = to.to_string();
            }
        }
    }
    for name in block.inputs.iter_mut().chain(block.outputs.iter_mut()) {
        if name == from {
            *name = to.to_string();
        }
    }
    block.tensors.retain(|t| t.name != from);
}

/// Operators with effects beyond their outputs, which no pass may remove or fold.
fn has_side_effects(op: Op) -> bool {
    matches!(
        op,
        Op::Variable
            | Op::VariableRead
            | Op::VariableWrite
            | Op::AssertEqualShape
            | Op::Custom
            | Op::CondIf
            | Op::WhileLoop
            | Op::Unknown(_)
    )
}

/// Merge chains of TRANSPOSE operators. Returns how many were rewritten.
pub fn cancel_transposes(block: &mut BasicBlock) -> usize {
    let mut changed = 0;
    let block_inputs: HashSet<String> = block.inputs.iter().cloned().collect();
    let block_outputs: HashSet<String> = block.outputs.iter().cloned().collect();
    loop {
        let producers: HashMap<String, usize> = producers(block).into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let mut rewrite = None;
        for (i, op) in block.operators.iter().enumerate() {
            let OpAttribute::Transpose { perms: second } = &op.attribute else { continue };
            let Some(&j) = op.inputs.first().and_then(|n| producers.get(n)) else { continue };
            let OpAttribute::Transpose { perms: first } = &block.operators[j].attribute else { continue };
            if block.operators[j].op != Op::Transpose || op.op != Op::Transpose {
                continue;
            }
            let source = block.operators[j].inputs[0].clone();
            rewrite = Some((i, source, compose_perms(first, second)));
            break;
        }
        let Some((i, source, composed)) = rewrite else { break };
        changed += 1;
        let output = block.operators[i].outputs[0].clone();
        if is_identity(&composed) {
            // The transpose is a no-op: merge its output with its source.
            // Block output names are kept; a block output that would then
            // alias a block input or another output becomes an IDENTITY.
            if !block_outputs.contains(&output) {
                block.operators.remove(i);
                replace_tensor(block, &output, &source);
            } else if !(block_inputs.contains(&source) || block_outputs.contains(&source)) {
                block.operators.remove(i);
                replace_tensor(block, &source, &output);
            } else {
                let op = &mut block.operators[i];
                op.op = Op::Identity;
                op.attribute = OpAttribute::None;
                op.inputs = vec![source];
            }
        } else {
            let op = &mut block.operators[i];
            op.inputs = vec![source];
            op.attribute = OpAttribute::Transpose { perms: composed };
        }
    }
    changed
}

/// Evaluate operators whose inputs are all constants and replace them with
/// CONST operators. Returns how many were folded.
pub fn fold_constants(block: &mut BasicBlock, max_bytes: usize) -> Result<usize> {
    let mut folded = 0;
    let mut defs: HashMap<String, usize> = block.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
    let mut constants: HashSet<String> = block
        .operators
        .iter()
        .filter(|op| matches!(op.op, Op::Const | Op::ConstShape))
        .flat_map(|op| op.outputs.iter().cloned())
        .collect();

    for i in 0..block.operators.len() {
        let op = &block.operators[i];
        if matches!(op.op, Op::Const | Op::ConstShape | Op::Identity) || has_side_effects(op.op) {
            continue;
        }
        if op.inputs.is_empty() || op.outputs.len() != 1 || !op.inputs.iter().all(|n| constants.contains(n)) {
            continue;
        }
        let Some(&out_def) = defs.get(&op.outputs[0]) else { continue };
        let out = &block.tensors[out_def];
        if out.shape.iter().any(|&d| d < 0) || out.dtype == DType::Unknown {
            continue;
        }
        let shape: Vec<usize> = out.shape.iter().map(|&d| d as usize).collect();
        let Ok(count) = checked_element_count(&shape) else { continue };
        if count.saturating_mul(out.dtype.element_size_bytes()) > max_bytes {
            continue;
        }

        // A graph holding just this operator and its constant inputs.
        let mut tensors: Vec<TensorDef> = Vec::new();
        let mut operators = Vec::new();
        for input in &op.inputs {
            let Some(&d) = defs.get(input) else { continue };
            if tensors.iter().any(|t| &t.name == input) {
                continue;
            }
            let def = &block.tensors[d];
            let const_op = if def.dtype == DType::Shape { Op::ConstShape } else { Op::Const };
            tensors.push(def.clone());
            operators.push(Operator::new(const_op, vec![], vec![input.clone()], OpAttribute::None));
        }
        tensors.push(out.clone());
        operators.push(op.clone());
        let mini = Graph {
            version: Version::CURRENT,
            regions: vec![Region {
                name: "fold".into(),
                blocks: vec![BasicBlock {
                    name: "fold".into(),
                    operators,
                    tensors,
                    inputs: vec![],
                    outputs: vec![op.outputs[0].clone()],
                }],
            }],
        };
        let result = Interpreter::new(mini).and_then(|mut interp| interp.run(HashMap::new(), &mut NoopHook));
        let Ok(mut outputs) = result else { continue };
        let Some(value) = outputs.remove(&op.outputs[0]) else { continue };
        if value.dtype() != out.dtype && !(out.dtype == DType::Shape && value.dtype() == DType::I64) {
            continue;
        }

        let name = op.outputs[0].clone();
        let is_shape = out.dtype == DType::Shape;
        block.tensors[out_def].data = Some(value.data.to_raw_bytes());
        block.tensors[out_def].shape = value.shape.iter().map(|&d| d as i32).collect();
        let location = block.operators[i].location.clone();
        block.operators[i] = Operator::new(
            if is_shape { Op::ConstShape } else { Op::Const },
            vec![],
            vec![name.clone()],
            OpAttribute::None,
        );
        block.operators[i].location = location;
        constants.insert(name);
        folded += 1;
        defs = block.tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
    }
    Ok(folded)
}

/// Remove operators whose outputs are unused and declarations nobody refers
/// to. Returns how many operators were removed.
pub fn eliminate_dead_code(block: &mut BasicBlock) -> usize {
    let mut removed = 0;
    loop {
        let uses = use_counts(block);
        let dead: Vec<usize> = block
            .operators
            .iter()
            .enumerate()
            .filter(|(_, op)| {
                !has_side_effects(op.op)
                    && !op.outputs.is_empty()
                    && op.outputs.iter().all(|o| !uses.contains_key(o.as_str()))
            })
            .map(|(i, _)| i)
            .collect();
        if dead.is_empty() {
            break;
        }
        for &i in dead.iter().rev() {
            block.operators.remove(i);
        }
        removed += dead.len();
    }
    let mut referenced: HashSet<&str> = HashSet::new();
    for op in &block.operators {
        referenced.extend(op.inputs.iter().chain(&op.outputs).map(String::as_str));
    }
    referenced.extend(block.inputs.iter().chain(&block.outputs).map(String::as_str));
    let keep: Vec<bool> = block.tensors.iter().map(|t| referenced.contains(t.name.as_str())).collect();
    let mut idx = 0;
    block.tensors.retain(|_| {
        idx += 1;
        keep[idx - 1]
    });
    removed
}
