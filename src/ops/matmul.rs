//! MATMUL: `output[n,h,w] = sum_c (a[n,h,c] - a_zp) * (b[n,c,w] - b_zp)`.
//!
//! Operands: a [N,H,C], b [N,C,W], a_zp [1], b_zp [1]; output [N,H,W]. As in
//! the reference model, the accumulator has the output type: inputs are
//! converted to it, zero points subtracted, and products summed in `c`
//! order. INT48 results are clamped to 48 bits.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, Operator};
use crate::ops::accum::{AccF16UnroundedProduct, AccF32, AccI32, AccI64, Accumulator};
use crate::ops::common::{
    check_declared_output, expect_operands, expect_rank, float_data, float_values, int_values,
    invalid, single_value,
};
use crate::tensor::{Tensor, TensorData};

const OP: &str = "MATMUL";

const INT48_MIN: i64 = -(1 << 47);
const INT48_MAX: i64 = (1 << 47) - 1;

/// The reference model's (a, b, output) type combinations.
fn is_supported(a: DType, b: DType, out: DType) -> bool {
    use DType::*;
    let fp8 = |t: DType| matches!(t, Fp8E4M3 | Fp8E5M2);
    match (a, b, out) {
        (I8, I8, I32) | (I16, I16, I48) => true,
        (F16, F16, F16 | F32) | (Bf16, Bf16, F32) | (F32, F32, F32) => true,
        (a, b, F16 | F32) => fp8(a) && fp8(b),
        _ => false,
    }
}

/// Output dtype when the graph does not declare one.
fn default_output(a: DType) -> DType {
    match a {
        DType::I8 => DType::I32,
        DType::I16 => DType::I48,
        DType::F16 | DType::Fp8E4M3 | DType::Fp8E5M2 => DType::F16,
        _ => DType::F32,
    }
}

fn matmul<A: Accumulator>(n: usize, h: usize, c: usize, w: usize, a: &[A::Operand], b: &[A::Operand]) -> Vec<A> {
    let mut out = Vec::with_capacity(n * h * w);
    for bn in 0..n {
        for row in 0..h {
            let a_row = &a[(bn * h + row) * c..][..c];
            for col in 0..w {
                let mut acc = A::zero();
                for (k, &x) in a_row.iter().enumerate() {
                    acc = acc.mac(x, b[(bn * c + k) * w + col]);
                }
                out.push(acc);
            }
        }
    }
    out
}

pub fn eval_matmul(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    expect_operands(op, OP, 4, 1)?;
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let a_zp = single_value(OP, "a_zp", state.get_tensor(&op.inputs[2])?)?;
    let b_zp = single_value(OP, "b_zp", state.get_tensor(&op.inputs[3])?)?;
    let output_name = &op.outputs[0];

    expect_rank(OP, "a", a, 3)?;
    expect_rank(OP, "b", b, 3)?;
    let [n, h, c] = a.shape[..] else { unreachable!() };
    let [bn, bc, w] = b.shape[..] else { unreachable!() };
    if bn != n || bc != c {
        return Err(Error::ShapeMismatch(format!(
            "{OP}: b shape {:?} incompatible with a shape {:?}",
            b.shape, a.shape
        )));
    }
    let out_shape = vec![n, h, w];

    let (a_t, b_t) = (a.dtype(), b.dtype());
    let out_t = check_declared_output(state, OP, output_name, &out_shape)?.unwrap_or_else(|| default_output(a_t));
    if !is_supported(a_t, b_t, out_t) {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: unsupported types a={a_t:?} b={b_t:?} output={out_t:?}"
        )));
    }
    if a_zp.dtype() != a_t || b_zp.dtype() != b_t {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: zero points must match input dtypes, got {:?}/{:?}",
            a_zp.dtype(),
            b_zp.dtype()
        )));
    }

    let output = if out_t.is_int() {
        let a_zp = int_values(&a_zp).unwrap()[0];
        let b_zp = int_values(&b_zp).unwrap()[0];
        if (a_zp != 0 && a_t != DType::I8) || (b_zp != 0 && b_t != DType::I8) {
            return Err(invalid(OP, "zero points must be zero for non-int8 inputs"));
        }
        let av: Vec<i64> = int_values(&a.data).unwrap().iter().map(|&v| v - a_zp).collect();
        let bv: Vec<i64> = int_values(&b.data).unwrap().iter().map(|&v| v - b_zp).collect();
        if out_t == DType::I32 {
            let to_i32 = |v: Vec<i64>| -> Vec<i32> { v.into_iter().map(|x| x as i32).collect() };
            TensorData::I32(matmul::<AccI32>(n, h, c, w, &to_i32(av), &to_i32(bv)).iter().map(|x| x.0).collect())
        } else {
            let acc = matmul::<AccI64>(n, h, c, w, &av, &bv);
            TensorData::I48(acc.iter().map(|x| x.0.clamp(INT48_MIN, INT48_MAX)).collect())
        }
    } else {
        let a_zp = float_values(&a_zp).unwrap()[0];
        let b_zp = float_values(&b_zp).unwrap()[0];
        if a_zp != 0.0 || b_zp != 0.0 {
            return Err(invalid(OP, "zero points must be zero for floating-point inputs"));
        }
        let av: Vec<f32> = float_values(&a.data).unwrap().iter().map(|&v| v - a_zp).collect();
        let bv: Vec<f32> = float_values(&b.data).unwrap().iter().map(|&v| v - b_zp).collect();
        let acc: Vec<f32> = match out_t {
            DType::F32 => matmul::<AccF32>(n, h, c, w, &av, &bv).iter().map(|x| x.0).collect(),
            DType::F16 => matmul::<AccF16UnroundedProduct>(n, h, c, w, &av, &bv).iter().map(|x| x.0).collect(),
            _ => unreachable!("is_supported admits only F16/F32 float outputs"),
        };
        float_data(out_t, acc)?
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, output));
    Ok(())
}
