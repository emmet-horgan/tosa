//! Execution plans: the operators of a block in a valid order, and the
//! tensors that can be dropped after each one.
//!
//! A plan is computed once per block when the [`Interpreter`](crate::Interpreter)
//! is created, so malformed graphs (undefined tensors, two producers of one
//! tensor, dependency cycles) are rejected before anything runs, and running
//! a block is a straight loop over `order`.
//!
//! Ordering rules:
//! - An operator runs after the producers of its input tensors.
//! - VARIABLE, VARIABLE_READ and VARIABLE_WRITE operators on the same variable
//!   run in program order, as they are side effects rather than dataflow.
//! - Among operators that are ready at the same time, program order wins, so
//!   a block serialized in execution order runs in exactly that order.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::error::{Error, Result};
use crate::ir::{BasicBlock, Op, Operator};

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Operator indices (into the block's operator list) in execution order.
    pub order: Vec<usize>,
    /// For every position in `order`, the tensors whose last use is that
    /// operator (operator outputs nobody reads are freed right after it).
    /// Block outputs are never listed.
    pub free_after: Vec<Vec<String>>,
    /// Block inputs no operator reads; they can be dropped before executing.
    pub unused_inputs: Vec<String>,
    /// Names the plan treats as persistent variables rather than tensors.
    pub variables: HashSet<String>,
}

/// Whether `op` is one of the variable (state) operators.
pub fn is_variable_op(op: Op) -> bool {
    matches!(op, Op::Variable | Op::VariableRead | Op::VariableWrite)
}

/// Names the variable operators of `block` operate on: outputs of VARIABLE
/// operators, tensors declared as variables, and the first input of a
/// VARIABLE_READ / VARIABLE_WRITE when it is not otherwise defined.
fn variable_names(block: &BasicBlock, producers: &HashMap<&str, usize>, inputs: &HashSet<&str>) -> HashSet<String> {
    let mut vars: HashSet<String> = block.tensors.iter().filter(|t| t.is_variable).map(|t| t.name.clone()).collect();
    for op in &block.operators {
        match op.op {
            Op::Variable => vars.extend(op.outputs.iter().cloned()),
            Op::VariableRead | Op::VariableWrite => {
                if let Some(first) = op.inputs.first()
                    && !producers.contains_key(first.as_str())
                    && !inputs.contains(first.as_str())
                {
                    vars.insert(first.clone());
                }
            }
            _ => {}
        }
    }
    vars
}

/// The variables `op` reads or writes.
fn touched_variables<'a>(op: &'a Operator, vars: &HashSet<String>) -> impl Iterator<Item = &'a String> {
    let outputs = if op.op == Op::Variable { &op.outputs[..] } else { &[] };
    op.inputs.iter().chain(outputs).filter(move |n| vars.contains(*n))
}

impl ExecutionPlan {
    pub fn new(block: &BasicBlock) -> Result<Self> {
        let n = block.operators.len();
        let inputs: HashSet<&str> = block.inputs.iter().map(String::as_str).collect();
        let outputs: HashSet<&str> = block.outputs.iter().map(String::as_str).collect();

        // Producers of ordinary tensors (variable operators produce state, not tensors).
        let mut producers: HashMap<&str, usize> = HashMap::new();
        for (i, op) in block.operators.iter().enumerate() {
            if op.op == Op::Variable {
                continue;
            }
            for out in &op.outputs {
                if producers.insert(out.as_str(), i).is_some() {
                    return Err(Error::GraphError(format!(
                        "block '{}': tensor '{out}' is produced by more than one operator",
                        block.name
                    )));
                }
                if inputs.contains(out.as_str()) {
                    return Err(Error::GraphError(format!(
                        "block '{}': tensor '{out}' is both a block input and an operator output",
                        block.name
                    )));
                }
            }
        }
        let variables = variable_names(block, &producers, &inputs);

        // Dependencies: data producers plus the previous operator on each variable.
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut last_variable_op: HashMap<&str, usize> = HashMap::new();
        for (i, op) in block.operators.iter().enumerate() {
            for name in &op.inputs {
                if variables.contains(name) {
                    continue;
                }
                match producers.get(name.as_str()) {
                    Some(&p) if p != i => deps[i].push(p),
                    Some(_) => {
                        return Err(Error::GraphError(format!(
                            "block '{}': operator {i} ({}) consumes its own output '{name}'",
                            block.name,
                            op.op.name()
                        )));
                    }
                    None if inputs.contains(name.as_str()) => {}
                    None => {
                        return Err(Error::GraphError(format!(
                            "block '{}': operator {i} ({}) reads undefined tensor '{name}'",
                            block.name,
                            op.op.name()
                        )));
                    }
                }
            }
            if is_variable_op(op.op) {
                for v in touched_variables(op, &variables) {
                    if let Some(prev) = last_variable_op.insert(v.as_str(), i) {
                        deps[i].push(prev);
                    }
                }
            }
        }

        // Kahn's algorithm, lowest index first.
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut pending = vec![0usize; n];
        for (i, d) in deps.iter().enumerate() {
            let mut d: Vec<usize> = d.clone();
            d.sort_unstable();
            d.dedup();
            pending[i] = d.len();
            for &p in &d {
                consumers[p].push(i);
            }
        }
        let mut ready: BinaryHeap<Reverse<usize>> = (0..n).filter(|&i| pending[i] == 0).map(Reverse).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(Reverse(i)) = ready.pop() {
            order.push(i);
            for &c in &consumers[i] {
                pending[c] -= 1;
                if pending[c] == 0 {
                    ready.push(Reverse(c));
                }
            }
        }
        if order.len() < n {
            let stuck: Vec<String> =
                (0..n).filter(|&i| pending[i] > 0).map(|i| format!("{} ({i})", block.operators[i].op.name())).collect();
            return Err(Error::GraphError(format!(
                "block '{}': dependency cycle among operators {}",
                block.name,
                stuck.join(", ")
            )));
        }

        // Last use of every tensor: the latest position that produces or consumes it.
        let mut last_use: HashMap<&str, usize> = HashMap::new();
        for (pos, &i) in order.iter().enumerate() {
            let op = &block.operators[i];
            let tensors = op.inputs.iter().chain(if op.op == Op::Variable { &[][..] } else { &op.outputs[..] });
            for name in tensors.filter(|n| !variables.contains(*n)) {
                last_use.insert(name.as_str(), pos);
            }
        }
        let mut free_after = vec![Vec::new(); n];
        let mut freed: HashSet<&str> = HashSet::new();
        for (&name, &pos) in &last_use {
            if !outputs.contains(name) && freed.insert(name) {
                free_after[pos].push(name.to_string());
            }
        }
        for list in &mut free_after {
            list.sort();
        }
        let unused_inputs =
            block.inputs.iter().filter(|n| !last_use.contains_key(n.as_str()) && !outputs.contains(n.as_str()));

        Ok(ExecutionPlan { order, free_after, unused_inputs: unused_inputs.cloned().collect(), variables })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::OpAttribute;

    fn op(op: Op, inputs: &[&str], outputs: &[&str]) -> Operator {
        Operator::new(
            op,
            inputs.iter().map(|s| s.to_string()).collect(),
            outputs.iter().map(|s| s.to_string()).collect(),
            OpAttribute::None,
        )
    }

    fn block(inputs: &[&str], outputs: &[&str], operators: Vec<Operator>) -> BasicBlock {
        BasicBlock {
            name: "main".into(),
            operators,
            tensors: vec![],
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn reorders_by_dependencies_and_frees_at_last_use() {
        // c = a + b is listed before the CONST that defines b.
        let b = block(
            &["a"],
            &["d"],
            vec![
                op(Op::Add, &["a", "b"], &["c"]),
                op(Op::Const, &[], &["b"]),
                op(Op::Abs, &["c"], &["d"]),
                op(Op::Negate, &["a"], &["unused"]),
            ],
        );
        let plan = ExecutionPlan::new(&b).unwrap();
        assert_eq!(plan.order, vec![1, 0, 2, 3]);
        assert_eq!(
            plan.free_after,
            vec![vec![], vec!["b".to_string()], vec!["c".to_string()], vec!["a".to_string(), "unused".to_string()]]
        );
        assert!(plan.unused_inputs.is_empty());
    }

    #[test]
    fn variable_operators_keep_program_order() {
        let b = block(
            &["x"],
            &["r2"],
            vec![
                op(Op::Variable, &[], &["v"]),
                op(Op::VariableRead, &["v"], &["r1"]),
                op(Op::Add, &["r1", "x"], &["s"]),
                op(Op::VariableWrite, &["v", "s"], &[]),
                op(Op::VariableRead, &["v"], &["r2"]),
            ],
        );
        let plan = ExecutionPlan::new(&b).unwrap();
        assert_eq!(plan.order, vec![0, 1, 2, 3, 4]);
        assert!(plan.variables.contains("v"));
    }

    #[test]
    fn rejects_undefined_tensors_duplicate_producers_and_cycles() {
        let undefined = block(&[], &["b"], vec![op(Op::Abs, &["a"], &["b"])]);
        assert!(ExecutionPlan::new(&undefined).unwrap_err().to_string().contains("undefined tensor 'a'"));

        let twice = block(&["a"], &["b"], vec![op(Op::Abs, &["a"], &["b"]), op(Op::Negate, &["a"], &["b"])]);
        assert!(ExecutionPlan::new(&twice).unwrap_err().to_string().contains("more than one operator"));

        let cycle = block(&[], &["b"], vec![op(Op::Abs, &["c"], &["b"]), op(Op::Negate, &["b"], &["c"])]);
        assert!(ExecutionPlan::new(&cycle).unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn unused_inputs_are_reported() {
        let b = block(&["a", "b"], &["c"], vec![op(Op::Abs, &["a"], &["c"])]);
        assert_eq!(ExecutionPlan::new(&b).unwrap().unused_inputs, vec!["b".to_string()]);
    }
}
