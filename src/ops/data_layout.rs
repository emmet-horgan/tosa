//! Data layout and gather/scatter operators: CONCAT, PAD, RESHAPE, REVERSE,
//! SLICE, TILE, TRANSPOSE, GATHER and SCATTER.
//!
//! Each operator only moves elements, so it is expressed as an index map
//! from output elements to input elements and applied with
//! [`TensorData::gather`]; this works for every element type.
//!
//! Shape-valued operands (PAD's padding, RESHAPE's shape, SLICE's start and
//! size, TILE's multiples) are SHAPE tensors.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::common::{check_declared_output, expect_operands, expect_rank, int_values, invalid};
use crate::tensor::{Tensor, TensorData};

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Call `f` with every multi-index of `shape` in row-major order.
fn for_each_index(shape: &[usize], mut f: impl FnMut(&[usize])) {
    let count: usize = shape.iter().product();
    let mut index = vec![0; shape.len()];
    for _ in 0..count {
        f(&index);
        for d in (0..shape.len()).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Values of a shape operand.
fn shape_operand(state: &InterpreterState, op: &str, name: &str) -> Result<Vec<i64>> {
    let t = state.get_tensor(name)?;
    int_values(&t.data).ok_or_else(|| invalid(op, format!("operand '{name}' must be a shape, got {:?}", t.dtype())))
}

/// Check the result against the graph's declaration and store it. The output
/// must have the element type of `dtype`.
fn finish(state: &mut InterpreterState, op: &Operator, name: &str, shape: Vec<usize>, data: TensorData) -> Result<()> {
    let output = &op.outputs[0];
    if let Some(declared) = check_declared_output(state, name, output, &shape)? {
        if declared != data.dtype() {
            return Err(Error::DTypeMismatch(format!(
                "{name}: output dtype {declared:?} must match input dtype {:?}",
                data.dtype()
            )));
        }
    }
    state.set_tensor(Tensor::new(output.clone(), shape, data));
    Ok(())
}

fn require_rank_at_least_one(op: &str, t: &Tensor) -> Result<()> {
    if t.shape.is_empty() {
        return Err(Error::ShapeMismatch(format!("{op}: input must have rank at least 1")));
    }
    Ok(())
}

fn axis_attribute(op: &str, axis: i32, rank: usize) -> Result<usize> {
    if axis < 0 || axis as usize >= rank {
        return Err(invalid(op, format!("axis {axis} out of range for rank {rank}")));
    }
    Ok(axis as usize)
}

pub fn eval_concat(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CONCAT";
    let OpAttribute::Concat { axis } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    if op.inputs.is_empty() || op.outputs.len() != 1 {
        return Err(invalid(OP, "expected at least one input and one output"));
    }
    let inputs: Vec<&Tensor> = op.inputs.iter().map(|n| state.get_tensor(n)).collect::<Result<_>>()?;
    let first = inputs[0];
    require_rank_at_least_one(OP, first)?;
    let rank = first.shape.len();
    let axis = axis_attribute(OP, axis, rank)?;
    for t in &inputs {
        let same_dims = t.shape.len() == rank && (0..rank).all(|d| d == axis || t.shape[d] == first.shape[d]);
        if !same_dims {
            return Err(Error::ShapeMismatch(format!(
                "{OP}: input shapes {:?} and {:?} are incompatible on axis {axis}",
                first.shape, t.shape
            )));
        }
    }
    let mut out_shape = first.shape.clone();
    out_shape[axis] = inputs.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = first.shape[..axis].iter().product();
    let inner: usize = first.shape[axis + 1..].iter().product();
    let joined = TensorData::concat(&inputs.iter().map(|t| &t.data).collect::<Vec<_>>())?;
    let mut offsets = Vec::with_capacity(inputs.len());
    let mut offset = 0;
    for t in &inputs {
        offsets.push(offset);
        offset += t.data.len();
    }
    let mut indices = Vec::with_capacity(joined.len());
    for o in 0..outer {
        for (t, &start) in inputs.iter().zip(&offsets) {
            let run = t.shape[axis] * inner;
            indices.extend(start + o * run..start + (o + 1) * run);
        }
    }
    let data = joined.gather(&indices);
    finish(state, op, OP, out_shape, data)
}

pub fn eval_pad(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "PAD";
    expect_operands(op, OP, 3, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    require_rank_at_least_one(OP, input)?;
    let padding = shape_operand(state, OP, &op.inputs[1])?;
    let pad_const = state.get_tensor(&op.inputs[2])?;
    let rank = input.shape.len();
    if padding.len() != 2 * rank {
        return Err(invalid(OP, format!("padding needs {} values, got {}", 2 * rank, padding.len())));
    }
    if padding.iter().any(|&p| p < 0) {
        return Err(invalid(OP, format!("padding {padding:?} must be non-negative")));
    }
    if pad_const.data.len() != 1 || pad_const.dtype() != input.dtype() {
        return Err(invalid(OP, "pad_const must be a single element of the input type"));
    }
    let out_shape: Vec<usize> =
        (0..rank).map(|d| input.shape[d] + padding[2 * d] as usize + padding[2 * d + 1] as usize).collect();

    // Index `input.len()` refers to the pad value appended after the input.
    let in_strides = strides(&input.shape);
    let pad_index = input.data.len();
    let mut indices = Vec::with_capacity(out_shape.iter().product());
    for_each_index(&out_shape, |idx| {
        let mut flat = 0;
        for d in 0..rank {
            let i = idx[d] as i64 - padding[2 * d];
            if i < 0 || i >= input.shape[d] as i64 {
                indices.push(pad_index);
                return;
            }
            flat += i as usize * in_strides[d];
        }
        indices.push(flat);
    });
    let data = TensorData::concat(&[&input.data, &pad_const.data])?.gather(&indices);
    finish(state, op, OP, out_shape, data)
}

pub fn eval_reshape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "RESHAPE";
    expect_operands(op, OP, 2, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    let new_shape = shape_operand(state, OP, &op.inputs[1])?;
    if new_shape.iter().any(|&d| d < 0) {
        return Err(invalid(OP, format!("shape {new_shape:?} must be non-negative")));
    }
    let out_shape: Vec<usize> = new_shape.iter().map(|&d| d as usize).collect();
    if out_shape.iter().product::<usize>() != input.shape.iter().product::<usize>() {
        return Err(Error::ShapeMismatch(format!(
            "{OP}: cannot reshape {:?} to {out_shape:?}",
            input.shape
        )));
    }
    let data = input.data.clone();
    finish(state, op, OP, out_shape, data)
}

pub fn eval_reverse(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "REVERSE";
    let OpAttribute::Reverse { axis } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    require_rank_at_least_one(OP, input)?;
    let axis = axis_attribute(OP, axis, input.shape.len())?;
    let s = strides(&input.shape);
    let mut indices = Vec::with_capacity(input.data.len());
    for_each_index(&input.shape, |idx| {
        let flat: usize = idx.iter().zip(&s).map(|(i, s)| i * s).sum();
        let mirrored = input.shape[axis] - 1 - idx[axis];
        indices.push(flat - idx[axis] * s[axis] + mirrored * s[axis]);
    });
    let (shape, data) = (input.shape.clone(), input.data.gather(&indices));
    finish(state, op, OP, shape, data)
}

pub fn eval_slice(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "SLICE";
    expect_operands(op, OP, 3, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    require_rank_at_least_one(OP, input)?;
    let start = shape_operand(state, OP, &op.inputs[1])?;
    let size = shape_operand(state, OP, &op.inputs[2])?;
    let rank = input.shape.len();
    if start.len() != rank || size.len() != rank {
        return Err(invalid(OP, format!("start and size need {rank} values")));
    }
    for d in 0..rank {
        let dim = input.shape[d] as i64;
        if start[d] < 0 || start[d] >= dim || size[d] <= 0 || start[d] + size[d] > dim {
            return Err(invalid(OP, format!("start {start:?} / size {size:?} out of bounds for {:?}", input.shape)));
        }
    }
    let out_shape: Vec<usize> = size.iter().map(|&s| s as usize).collect();
    let s = strides(&input.shape);
    let mut indices = Vec::with_capacity(out_shape.iter().product());
    for_each_index(&out_shape, |idx| {
        indices.push((0..rank).map(|d| (idx[d] + start[d] as usize) * s[d]).sum());
    });
    let data = input.data.gather(&indices);
    finish(state, op, OP, out_shape, data)
}

pub fn eval_tile(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "TILE";
    expect_operands(op, OP, 2, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    require_rank_at_least_one(OP, input)?;
    let multiples = shape_operand(state, OP, &op.inputs[1])?;
    let rank = input.shape.len();
    if multiples.len() != rank || multiples.iter().any(|&m| m < 1) {
        return Err(invalid(OP, format!("multiples {multiples:?} must be {rank} positive values")));
    }
    let out_shape: Vec<usize> = (0..rank).map(|d| input.shape[d] * multiples[d] as usize).collect();
    let s = strides(&input.shape);
    let mut indices = Vec::with_capacity(out_shape.iter().product());
    for_each_index(&out_shape, |idx| {
        indices.push((0..rank).map(|d| (idx[d] % input.shape[d]) * s[d]).sum());
    });
    let data = input.data.gather(&indices);
    finish(state, op, OP, out_shape, data)
}

pub fn eval_transpose(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "TRANSPOSE";
    let OpAttribute::Transpose { perms } = &op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    require_rank_at_least_one(OP, input)?;
    let rank = input.shape.len();
    let mut used = vec![false; rank];
    for &p in perms {
        if p < 0 || p as usize >= rank || std::mem::replace(&mut used[p as usize], true) {
            return Err(invalid(OP, format!("perms {perms:?} must be a permutation of 0..{rank}")));
        }
    }
    if perms.len() != rank {
        return Err(invalid(OP, format!("perms {perms:?} must have {rank} entries")));
    }
    let perms: Vec<usize> = perms.iter().map(|&p| p as usize).collect();
    let out_shape: Vec<usize> = perms.iter().map(|&p| input.shape[p]).collect();
    let s = strides(&input.shape);
    let mut indices = Vec::with_capacity(input.data.len());
    for_each_index(&out_shape, |idx| {
        indices.push(idx.iter().zip(&perms).map(|(&i, &p)| i * s[p]).sum());
    });
    let data = input.data.gather(&indices);
    finish(state, op, OP, out_shape, data)
}

fn index_values(op: &str, t: &Tensor) -> Result<Vec<i64>> {
    if !matches!(t.dtype(), DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch(format!("{op}: indices must be INT32 or INT64, got {:?}", t.dtype())));
    }
    Ok(int_values(&t.data).unwrap())
}

/// GATHER: values [N,K,C], indices [N,W] -> output [N,W,C] with
/// `output[n,w,c] = values[n, indices[n,w], c]`.
pub fn eval_gather(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "GATHER";
    expect_operands(op, OP, 2, 1)?;
    let values = state.get_tensor(&op.inputs[0])?;
    let indices = state.get_tensor(&op.inputs[1])?;
    expect_rank(OP, "values", values, 3)?;
    expect_rank(OP, "indices", indices, 2)?;
    let [n, k, c] = values.shape[..] else { unreachable!() };
    let [ni, w] = indices.shape[..] else { unreachable!() };
    if ni != n {
        return Err(Error::ShapeMismatch(format!("{OP}: batch {ni} of indices != {n} of values")));
    }
    let idx = index_values(OP, indices)?;
    let mut map = Vec::with_capacity(n * w * c);
    for b in 0..n {
        for j in 0..w {
            let row = idx[b * w + j];
            if row < 0 || row as usize >= k {
                return Err(invalid(OP, format!("index {row} out of range [0, {k})")));
            }
            map.extend((0..c).map(|ch| (b * k + row as usize) * c + ch));
        }
    }
    let data = values.data.gather(&map);
    finish(state, op, OP, vec![n, w, c], data)
}

/// SCATTER: values_in [N,K,C], indices [N,W], input [N,W,C] -> values_out
/// [N,K,C], a copy of values_in with `values_out[n, indices[n,w], c] =
/// input[n,w,c]`. Each row may be written at most once.
pub fn eval_scatter(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "SCATTER";
    expect_operands(op, OP, 3, 1)?;
    let values_in = state.get_tensor(&op.inputs[0])?;
    let indices = state.get_tensor(&op.inputs[1])?;
    let input = state.get_tensor(&op.inputs[2])?;
    expect_rank(OP, "values_in", values_in, 3)?;
    expect_rank(OP, "indices", indices, 2)?;
    expect_rank(OP, "input", input, 3)?;
    let [n, k, c] = values_in.shape[..] else { unreachable!() };
    if indices.shape[0] != n || input.shape[0] != n || input.shape[1] != indices.shape[1] || input.shape[2] != c {
        return Err(Error::ShapeMismatch(format!(
            "{OP}: incompatible shapes values_in {:?}, indices {:?}, input {:?}",
            values_in.shape, indices.shape, input.shape
        )));
    }
    let w = indices.shape[1];
    let idx = index_values(OP, indices)?;
    // Indices >= values_in.len() refer to `input`, appended after values_in.
    let mut map: Vec<usize> = (0..values_in.data.len()).collect();
    let mut written = vec![false; n * k];
    for b in 0..n {
        for j in 0..w {
            let row = idx[b * w + j];
            if row < 0 || row as usize >= k {
                return Err(invalid(OP, format!("index {row} out of range [0, {k})")));
            }
            let row = row as usize;
            if std::mem::replace(&mut written[b * k + row], true) {
                return Err(invalid(OP, format!("row {row} of batch {b} is written more than once")));
            }
            for ch in 0..c {
                map[(b * k + row) * c + ch] = values_in.data.len() + (b * w + j) * c + ch;
            }
        }
    }
    let data = TensorData::concat(&[&values_in.data, &input.data])?.gather(&map);
    finish(state, op, OP, vec![n, k, c], data)
}
