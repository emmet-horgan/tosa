//! Shape operators (the `tosa-ext-shape` extension): CONST_SHAPE is handled
//! with CONST; this module covers DIM, the element-wise shape arithmetic,
//! CONCAT_SHAPE, SLICE_SHAPE and ASSERT_EQUAL_SHAPE.
//!
//! A SHAPE value of rank r is a list of r int64 dimensions. At run time it is
//! held as an INT64 tensor of shape [r]; whether a tensor is a SHAPE is
//! checked against the graph's declarations.
//!
//! Arithmetic follows the reference model, including its C-style integer
//! division: DIV_FLOOR_SHAPE truncates towards zero, DIV_CEIL_SHAPE adds one
//! to the truncated quotient when there is a remainder, and MOD_SHAPE is the
//! truncated remainder.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::common::{expect_operands, int_values, invalid};
use crate::tensor::{Tensor, TensorData};

/// Largest exponent EXP2_SHAPE accepts (MAX_LOG2_SIZE of TOSA level 8K).
const MAX_LOG2_SIZE: i64 = 63;

fn declared_dtype(state: &InterpreterState, name: &str) -> Option<DType> {
    state.find_tensor_def(name).map(|d| d.dtype)
}

/// The values of a SHAPE operand.
fn shape_input(state: &InterpreterState, op: &str, name: &str) -> Result<Vec<i64>> {
    if declared_dtype(state, name).is_some_and(|d| d != DType::Shape) {
        return Err(Error::DTypeMismatch(format!("{op}: input '{name}' must be a SHAPE")));
    }
    let t = state.get_tensor(name)?;
    int_values(&t.data).ok_or_else(|| Error::DTypeMismatch(format!("{op}: input '{name}' must be a SHAPE")))
}

/// Store a SHAPE result, checking the declared output type and length.
fn shape_output(state: &mut InterpreterState, op: &Operator, name: &str, values: Vec<i64>) -> Result<()> {
    let output = &op.outputs[0];
    if let Some(def) = state.find_tensor_def(output) {
        if def.dtype != DType::Shape {
            return Err(Error::DTypeMismatch(format!("{name}: output must be a SHAPE, got {:?}", def.dtype)));
        }
        if def.shape != [values.len() as i32] {
            return Err(Error::ShapeMismatch(format!(
                "{name}: output has {} values but is declared with shape {:?}",
                values.len(),
                def.shape
            )));
        }
    }
    state.set_tensor(Tensor::new(output.clone(), vec![values.len()], TensorData::I64(values)));
    Ok(())
}

fn eval_unary_shape(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    f: impl Fn(i64) -> Result<i64>,
) -> Result<()> {
    expect_operands(op, name, 1, 1)?;
    let a = shape_input(state, name, &op.inputs[0])?;
    let out = a.into_iter().map(f).collect::<Result<_>>()?;
    shape_output(state, op, name, out)
}

fn eval_binary_shape(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    f: impl Fn(i64, i64) -> Result<i64>,
) -> Result<()> {
    expect_operands(op, name, 2, 1)?;
    let a = shape_input(state, name, &op.inputs[0])?;
    let b = shape_input(state, name, &op.inputs[1])?;
    if a.len() != b.len() {
        return Err(Error::ShapeMismatch(format!("{name}: inputs have ranks {} and {}", a.len(), b.len())));
    }
    let out = a.into_iter().zip(b).map(|(x, y)| f(x, y)).collect::<Result<_>>()?;
    shape_output(state, op, name, out)
}

fn nonzero(name: &str, b: i64) -> Result<i64> {
    if b == 0 {
        return Err(invalid(name, "division by zero"));
    }
    Ok(b)
}

pub fn eval_add_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_shape(state, op, "ADD_SHAPE", |a, b| Ok(a.wrapping_add(b)))
}

pub fn eval_sub_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_shape(state, op, "SUB_SHAPE", |a, b| Ok(a.wrapping_sub(b)))
}

pub fn eval_mul_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_shape(state, op, "MUL_SHAPE", |a, b| Ok(a.wrapping_mul(b)))
}

pub fn eval_max_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_shape(state, op, "MAX_SHAPE", |a, b| Ok(a.max(b)))
}

pub fn eval_min_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_binary_shape(state, op, "MIN_SHAPE", |a, b| Ok(a.min(b)))
}

pub fn eval_mod_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "MOD_SHAPE";
    eval_binary_shape(state, op, OP, |a, b| Ok(a.wrapping_rem(nonzero(OP, b)?)))
}

pub fn eval_div_floor_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "DIV_FLOOR_SHAPE";
    eval_binary_shape(state, op, OP, |a, b| Ok(a.wrapping_div(nonzero(OP, b)?)))
}

pub fn eval_div_ceil_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "DIV_CEIL_SHAPE";
    eval_binary_shape(state, op, OP, |a, b| {
        let b = nonzero(OP, b)?;
        Ok(a.wrapping_div(b) + (a.wrapping_rem(b) != 0) as i64)
    })
}

pub fn eval_exp2_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "EXP2_SHAPE";
    eval_unary_shape(state, op, OP, |a| {
        if !(0..MAX_LOG2_SIZE).contains(&a) {
            return Err(invalid(OP, format!("exponent {a} out of range")));
        }
        Ok(1i64 << a)
    })
}

fn log2_floor(op: &str, a: i64) -> Result<i64> {
    if a < 1 {
        return Err(invalid(op, format!("log2 of {a} is undefined")));
    }
    Ok(63 - a.leading_zeros() as i64)
}

pub fn eval_log2_floor_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval_unary_shape(state, op, "LOG2_FLOOR_SHAPE", |a| log2_floor("LOG2_FLOOR_SHAPE", a))
}

pub fn eval_log2_ceil_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "LOG2_CEIL_SHAPE";
    eval_unary_shape(state, op, OP, |a| {
        let f = log2_floor(OP, a)?;
        Ok(f + (a != 1 << f) as i64)
    })
}

pub fn eval_concat_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "CONCAT_SHAPE";
    if op.inputs.is_empty() || op.outputs.len() != 1 {
        return Err(invalid(OP, "expected at least one input and one output"));
    }
    let mut out = Vec::new();
    for name in &op.inputs {
        let v = shape_input(state, OP, name)?;
        if v.is_empty() {
            return Err(invalid(OP, "rank-0 shapes cannot be concatenated"));
        }
        out.extend(v);
    }
    shape_output(state, op, OP, out)
}

/// SLICE_SHAPE: `size` values of the input starting at `start` (both INT32 [1]).
pub fn eval_slice_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "SLICE_SHAPE";
    expect_operands(op, OP, 3, 1)?;
    let input = shape_input(state, OP, &op.inputs[0])?;
    let scalar = |name: &str| -> Result<i64> {
        let t = state.get_tensor(name)?;
        match (&t.data, t.data.len()) {
            (TensorData::I32(v), 1) => Ok(v[0] as i64),
            _ => Err(Error::DTypeMismatch(format!("{OP}: start and size must be INT32 [1]"))),
        }
    };
    let (start, size) = (scalar(&op.inputs[1])?, scalar(&op.inputs[2])?);
    if start < 0 || size <= 0 || (start + size) as usize > input.len() {
        return Err(invalid(OP, format!("start {start} / size {size} out of bounds for {} values", input.len())));
    }
    let out = input[start as usize..(start + size) as usize].to_vec();
    shape_output(state, op, OP, out)
}

/// DIM: the size of dimension `axis` of the input, as a SHAPE of rank 1.
pub fn eval_dim(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "DIM";
    let OpAttribute::Dim { axis } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 1, 1)?;
    let input = state.get_tensor(&op.inputs[0])?;
    if axis < 0 || axis as usize >= input.shape.len() {
        return Err(invalid(OP, format!("axis {axis} out of range for rank {}", input.shape.len())));
    }
    let dim = input.shape[axis as usize] as i64;
    shape_output(state, op, OP, vec![dim])
}

/// ASSERT_EQUAL_SHAPE: no outputs; the graph is invalid unless the two
/// shapes are equal (or, with `allow_broadcast`, broadcast-compatible).
pub fn eval_assert_equal_shape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "ASSERT_EQUAL_SHAPE";
    let OpAttribute::AssertEqualShape { allow_broadcast } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 2, 0)?;
    let a = shape_input(state, OP, &op.inputs[0])?;
    let b = shape_input(state, OP, &op.inputs[1])?;
    let compatible = a.len() == b.len()
        && a.iter().zip(&b).all(|(&x, &y)| x == y || (allow_broadcast && (x == 1 || y == 1)));
    if !compatible {
        return Err(invalid(OP, format!("shapes {a:?} and {b:?} differ")));
    }
    Ok(())
}
