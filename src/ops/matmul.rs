//! MATMUL and MATMUL_T.
//!
//! MATMUL: a [N,H,C], b [N,C,W]; MATMUL_T: a [N,H,C], b [D,W,C] with D equal
//! to N or 1 (broadcast). Both take zero points a_zp [1] and b_zp [1] and
//! produce `output[n,h,w] = sum_c (a[n,h,c] - a_zp) * (b[.,c,w] - b_zp)`.
//!
//! As in the reference model the accumulator has the output type: operands
//! are converted to it (which rounds, e.g., FP16 operands of a BF16 MATMUL_T),
//! zero points subtracted, and products summed in `c` order with
//! `acc = acc + a * b`. INT48 results are clamped to 48 bits.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, Operator};
use crate::ops::accum::{AccBf16UnroundedProduct, AccF16UnroundedProduct, AccF32, AccI32, AccI64, Accumulator};
use crate::ops::common::{
    check_declared_output, expect_operands, expect_rank, float_data, float_values, int_values, invalid,
    round_to, single_value,
};
use crate::tensor::{Tensor, TensorData};

const INT48_MIN: i64 = -(1 << 47);
const INT48_MAX: i64 = (1 << 47) - 1;

fn is_fp8(t: DType) -> bool {
    matches!(t, DType::Fp8E4M3 | DType::Fp8E5M2)
}

/// MATMUL's (a, b, output) type combinations.
fn matmul_types(a: DType, b: DType, out: DType) -> bool {
    use DType::*;
    match (a, b, out) {
        (I8, I8, I32) | (I16, I16, I48) => true,
        (F16, F16, F16 | F32) | (Bf16, Bf16, F32) | (F32, F32, F32) => true,
        (a, b, F16 | F32) => is_fp8(a) && is_fp8(b),
        _ => false,
    }
}

/// MATMUL_T's (a, b, output) type combinations: any mix of FP8/FP16/BF16/FP32
/// operands into BF16 or FP32 (FP32 operands need FP32), plus FP16 x FP16 -> FP16.
fn matmul_t_types(a: DType, b: DType, out: DType) -> bool {
    use DType::*;
    let float = |t: DType| is_fp8(t) || matches!(t, F16 | Bf16 | F32);
    match (a, b, out) {
        (I8, I8, I32) | (I16, I16, I48) | (F16, F16, F16) => true,
        (a, b, F32) => float(a) && float(b),
        (a, b, Bf16) => float(a) && float(b) && a != F32 && b != F32 && (a, b) != (F16, F16),
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

struct Dims {
    n: usize,
    h: usize,
    c: usize,
    w: usize,
    /// b's batch is 1 and broadcast (MATMUL_T only).
    b_broadcast: bool,
    /// b is laid out [D,W,C] rather than [N,C,W].
    b_transposed: bool,
}

impl Dims {
    fn b_index(&self, n: usize, k: usize, col: usize) -> usize {
        let n = if self.b_broadcast { 0 } else { n };
        if self.b_transposed {
            (n * self.w + col) * self.c + k
        } else {
            (n * self.c + k) * self.w + col
        }
    }

    fn run<A: Accumulator>(&self, a: &[A::Operand], b: &[A::Operand]) -> Vec<A> {
        let mut out = Vec::with_capacity(self.n * self.h * self.w);
        for bn in 0..self.n {
            for row in 0..self.h {
                let a_row = &a[(bn * self.h + row) * self.c..][..self.c];
                for col in 0..self.w {
                    let mut acc = A::zero();
                    for (k, &x) in a_row.iter().enumerate() {
                        acc = acc.mac(x, b[self.b_index(bn, k, col)]);
                    }
                    out.push(acc);
                }
            }
        }
        out
    }
}

fn eval(
    state: &mut InterpreterState,
    op: &Operator,
    name: &str,
    transposed: bool,
    types_ok: fn(DType, DType, DType) -> bool,
) -> Result<()> {
    expect_operands(op, name, 4, 1)?;
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let a_zp = single_value(name, "a_zp", state.get_tensor(&op.inputs[2])?)?;
    let b_zp = single_value(name, "b_zp", state.get_tensor(&op.inputs[3])?)?;
    let output_name = &op.outputs[0];

    expect_rank(name, "a", a, 3)?;
    expect_rank(name, "b", b, 3)?;
    let [n, h, c] = a.shape[..] else { unreachable!() };
    let (bn, bc, w) = if transposed {
        (b.shape[0], b.shape[2], b.shape[1])
    } else {
        (b.shape[0], b.shape[1], b.shape[2])
    };
    if bc != c || (bn != n && !(transposed && bn == 1)) {
        return Err(Error::ShapeMismatch(format!(
            "{name}: b shape {:?} incompatible with a shape {:?}",
            b.shape, a.shape
        )));
    }
    let dims = Dims { n, h, c, w, b_broadcast: bn != n, b_transposed: transposed };
    let out_shape = vec![n, h, w];

    let (a_t, b_t) = (a.dtype(), b.dtype());
    let out_t = check_declared_output(state, name, output_name, &out_shape)?.unwrap_or_else(|| default_output(a_t));
    if !types_ok(a_t, b_t, out_t) {
        return Err(Error::DTypeMismatch(format!(
            "{name}: unsupported types a={a_t:?} b={b_t:?} output={out_t:?}"
        )));
    }
    if a_zp.dtype() != a_t || b_zp.dtype() != b_t {
        return Err(Error::DTypeMismatch(format!(
            "{name}: zero points must match input dtypes, got {:?}/{:?}",
            a_zp.dtype(),
            b_zp.dtype()
        )));
    }

    let output = if out_t.is_int() {
        let a_zp = int_values(&a_zp).unwrap()[0];
        let b_zp = int_values(&b_zp).unwrap()[0];
        if (a_zp != 0 && a_t != DType::I8) || (b_zp != 0 && b_t != DType::I8) {
            return Err(invalid(name, "zero points must be zero for non-int8 inputs"));
        }
        let av: Vec<i64> = int_values(&a.data).unwrap().iter().map(|&v| v - a_zp).collect();
        let bv: Vec<i64> = int_values(&b.data).unwrap().iter().map(|&v| v - b_zp).collect();
        if out_t == DType::I32 {
            let to_i32 = |v: Vec<i64>| -> Vec<i32> { v.into_iter().map(|x| x as i32).collect() };
            TensorData::I32(dims.run::<AccI32>(&to_i32(av), &to_i32(bv)).iter().map(|x| x.0).collect())
        } else {
            let acc = dims.run::<AccI64>(&av, &bv);
            TensorData::I48(acc.iter().map(|x| x.0.clamp(INT48_MIN, INT48_MAX)).collect())
        }
    } else {
        let a_zp = float_values(&a_zp).unwrap()[0];
        let b_zp = float_values(&b_zp).unwrap()[0];
        if a_zp != 0.0 || b_zp != 0.0 {
            return Err(invalid(name, "zero points must be zero for floating-point inputs"));
        }
        let convert = |v: Vec<f32>| -> Vec<f32> { v.into_iter().map(|x| round_to(out_t, x)).collect() };
        let av = convert(float_values(&a.data).unwrap());
        let bv = convert(float_values(&b.data).unwrap());
        let acc: Vec<f32> = match out_t {
            DType::F32 => dims.run::<AccF32>(&av, &bv).iter().map(|x| x.0).collect(),
            DType::F16 => dims.run::<AccF16UnroundedProduct>(&av, &bv).iter().map(|x| x.0).collect(),
            DType::Bf16 => dims.run::<AccBf16UnroundedProduct>(&av, &bv).iter().map(|x| x.0).collect(),
            _ => unreachable!("the type tables admit only F16/BF16/F32 float outputs"),
        };
        float_data(out_t, acc)?
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, output));
    Ok(())
}

pub fn eval_matmul(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval(state, op, "MATMUL", false, matmul_types)
}

pub fn eval_matmul_t(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    eval(state, op, "MATMUL_T", true, matmul_t_types)
}
