//! CONV2D.
//!
//! Operands: input [N,IH,IW,IC], weight [OC,KH,KW,IC], bias [OC] or [1],
//! input_zp [1], weight_zp [1]. Output: [N,OH,OW,OC].
//!
//! Semantics follow the TOSA reference model: the accumulator starts at zero,
//! padding contributes nothing, and the bias is added after the accumulator
//! has been converted to the output type: `out = cast<out>(acc) + bias`.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::accum::{AccBf16, AccF16, AccF32, AccI32, AccI64, Accumulator};
use crate::ops::common::{
    check_declared_output, expect_operands, expect_rank, float_data, float_values, int_values,
    invalid, round_to,
};
use crate::tensor::{Tensor, TensorData};

const OP: &str = "CONV2D";

const INT48_MIN: i64 = -(1 << 47);
const INT48_MAX: i64 = (1 << 47) - 1;

/// Shape and attribute parameters, validated.
pub struct Conv2dGeometry {
    pub n: usize,
    pub ih: usize,
    pub iw: usize,
    pub ic: usize,
    pub oc: usize,
    pub kh: usize,
    pub kw: usize,
    pub oh: usize,
    pub ow: usize,
    pub pad_top: usize,
    pub pad_left: usize,
    pub stride_y: usize,
    pub stride_x: usize,
    pub dilation_y: usize,
    pub dilation_x: usize,
}

/// Compute one output extent, requiring the window to tile the padded input exactly.
fn output_extent(
    what: &str,
    input: usize,
    pad_a: i64,
    pad_b: i64,
    kernel: usize,
    dilation: i64,
    stride: i64,
) -> Result<usize> {
    let full = input as i64 - 1 + pad_a + pad_b - (kernel as i64 - 1) * dilation;
    if full % stride != 0 {
        return Err(invalid(
            OP,
            format!("{what}: parameters must yield an exact integer output dimension"),
        ));
    }
    let out = full / stride + 1;
    if out < 1 {
        return Err(invalid(OP, format!("{what}: output dimension {out} must be positive")));
    }
    Ok(out as usize)
}

fn geometry(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    pad: &[i32],
    stride: &[i32],
    dilation: &[i32],
) -> Result<Conv2dGeometry> {
    expect_rank(OP, "input", input, 4)?;
    expect_rank(OP, "weight", weight, 4)?;
    expect_rank(OP, "bias", bias, 1)?;

    if pad.len() != 4 || stride.len() != 2 || dilation.len() != 2 {
        return Err(invalid(OP, "pad must have 4 elements, stride and dilation 2"));
    }
    if pad.iter().any(|&p| p < 0) {
        return Err(invalid(OP, format!("pad {pad:?} must be non-negative")));
    }
    if stride.iter().chain(dilation).any(|&s| s < 1) {
        return Err(invalid(OP, "stride and dilation must be at least 1"));
    }

    let [n, ih, iw, ic] = input.shape[..] else { unreachable!() };
    let [oc, kh, kw, kic] = weight.shape[..] else { unreachable!() };
    if kic != ic {
        return Err(Error::ShapeMismatch(format!(
            "{OP}: weight input channels {kic} != input channels {ic}"
        )));
    }
    let bc = bias.shape[0];
    if bc != oc && bc != 1 {
        return Err(Error::ShapeMismatch(format!(
            "{OP}: bias channels {bc} must be 1 or match output channels {oc}"
        )));
    }

    let p = |i: usize| pad[i] as i64;
    let oh = output_extent("height", ih, p(0), p(1), kh, dilation[0] as i64, stride[0] as i64)?;
    let ow = output_extent("width", iw, p(2), p(3), kw, dilation[1] as i64, stride[1] as i64)?;

    Ok(Conv2dGeometry {
        n,
        ih,
        iw,
        ic,
        oc,
        kh,
        kw,
        oh,
        ow,
        pad_top: pad[0] as usize,
        pad_left: pad[2] as usize,
        stride_y: stride[0] as usize,
        stride_x: stride[1] as usize,
        dilation_y: dilation[0] as usize,
        dilation_x: dilation[1] as usize,
    })
}

/// Accumulate every output element, in the reference model's loop order
/// (ky, kx, ic). Positions in the padding are skipped.
fn accumulate<A: Accumulator>(g: &Conv2dGeometry, input: &[A::Operand], weight: &[A::Operand]) -> Vec<A> {
    let mut out = Vec::with_capacity(g.n * g.oh * g.ow * g.oc);
    for b in 0..g.n {
        for oy in 0..g.oh {
            let iy0 = (oy * g.stride_y) as isize - g.pad_top as isize;
            for ox in 0..g.ow {
                let ix0 = (ox * g.stride_x) as isize - g.pad_left as isize;
                for o in 0..g.oc {
                    let mut acc = A::zero();
                    for ky in 0..g.kh {
                        let y = iy0 + (ky * g.dilation_y) as isize;
                        if y < 0 || y >= g.ih as isize {
                            continue;
                        }
                        for kx in 0..g.kw {
                            let x = ix0 + (kx * g.dilation_x) as isize;
                            if x < 0 || x >= g.iw as isize {
                                continue;
                            }
                            let i_base = ((b * g.ih + y as usize) * g.iw + x as usize) * g.ic;
                            let w_base = ((o * g.kh + ky) * g.kw + kx) * g.ic;
                            for c in 0..g.ic {
                                acc = acc.mac(input[i_base + c], weight[w_base + c]);
                            }
                        }
                    }
                    out.push(acc);
                }
            }
        }
    }
    out
}

/// The reference model's supported (input, weight, acc, output) type combinations.
fn is_supported(input: DType, weight: DType, acc: DType, out: DType) -> bool {
    use DType::*;
    let fp8 = |t: DType| matches!(t, Fp8E4M3 | Fp8E5M2);
    match (input, weight, acc, out) {
        (I8, I8 | I4, I32, I32) | (I16, I8, I48, I48) => true,
        (i, w, F32, F32) => [i, w].iter().all(|&t| fp8(t) || matches!(t, F16 | Bf16 | F32)),
        (i, w, F16 | F32, F16) => [i, w].iter().all(|&t| fp8(t) || t == F16),
        (i, w, Bf16 | F32, Bf16) => [i, w].iter().all(|&t| fp8(t) || t == Bf16),
        _ => false,
    }
}

fn single_value(op_name: &str, what: &str, t: &Tensor) -> Result<TensorData> {
    expect_rank(op_name, what, t, 1)?;
    if t.data.len() != 1 {
        return Err(Error::ShapeMismatch(format!(
            "{op_name}: {what} must have exactly one element, got shape {:?}",
            t.shape
        )));
    }
    Ok(t.data.clone())
}

pub fn eval_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let OpAttribute::Conv2d { pad, stride, dilation, acc_type } = &op.attribute else {
        return Err(Error::GraphError(format!("{OP}: missing attribute")));
    };
    expect_operands(op, OP, 5, 1)?;

    let input = state.get_tensor(&op.inputs[0])?;
    let weight = state.get_tensor(&op.inputs[1])?;
    let bias = state.get_tensor(&op.inputs[2])?;
    let input_zp = single_value(OP, "input_zp", state.get_tensor(&op.inputs[3])?)?;
    let weight_zp = single_value(OP, "weight_zp", state.get_tensor(&op.inputs[4])?)?;
    let output_name = &op.outputs[0];

    let g = geometry(input, weight, bias, pad, stride, dilation)?;
    let out_shape = vec![g.n, g.oh, g.ow, g.oc];

    let (in_t, w_t, acc_t, out_t) = (input.dtype(), weight.dtype(), *acc_type, bias.dtype());
    if let Some(declared) = check_declared_output(state, OP, output_name, &out_shape)? {
        if declared != out_t {
            return Err(Error::DTypeMismatch(format!(
                "{OP}: output dtype {declared:?} does not match bias dtype {out_t:?}"
            )));
        }
    }
    if !is_supported(in_t, w_t, acc_t, out_t) {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: unsupported types input={in_t:?} weight={w_t:?} acc={acc_t:?} output={out_t:?}"
        )));
    }
    if input_zp.dtype() != in_t || weight_zp.dtype() != w_t {
        return Err(Error::DTypeMismatch(format!(
            "{OP}: zero points must match input/weight dtypes, got {:?}/{:?}",
            input_zp.dtype(),
            weight_zp.dtype()
        )));
    }

    let bias_idx = |o: usize| if bias.data.len() == 1 { 0 } else { o };
    let output = if acc_t.is_int() {
        // Every element of the tensors is an integer here; see is_supported.
        let in_zp = int_values(&input_zp).unwrap()[0];
        let w_zp = int_values(&weight_zp).unwrap()[0];
        if in_zp != 0 && in_t != DType::I8 {
            return Err(invalid(OP, "input zero point must be zero for non-int8 input"));
        }
        if w_zp != 0 && w_t != DType::I8 {
            return Err(invalid(OP, "weight zero point must be zero for non-int8 weight"));
        }
        let inp: Vec<i64> = int_values(&input.data).unwrap().iter().map(|&v| v - in_zp).collect();
        let wgt: Vec<i64> = int_values(&weight.data).unwrap().iter().map(|&v| v - w_zp).collect();
        let bis = int_values(&bias.data).unwrap();

        if acc_t == DType::I32 {
            let to_i32 = |v: Vec<i64>| -> Vec<i32> { v.into_iter().map(|x| x as i32).collect() };
            let acc = accumulate::<AccI32>(&g, &to_i32(inp), &to_i32(wgt));
            let out = acc
                .iter()
                .enumerate()
                .map(|(i, a)| a.0.wrapping_add(bis[bias_idx(i % g.oc)] as i32))
                .collect();
            TensorData::I32(out)
        } else {
            let acc = accumulate::<AccI64>(&g, &inp, &wgt);
            let out = acc
                .iter()
                .enumerate()
                .map(|(i, a)| a.0.wrapping_add(bis[bias_idx(i % g.oc)]).clamp(INT48_MIN, INT48_MAX))
                .collect();
            TensorData::I48(out)
        }
    } else {
        // Floats: zero points must be zero; the subtraction is still evaluated
        // (in f32) to mirror the reference.
        let in_zp = float_values(&input_zp).unwrap()[0];
        let w_zp = float_values(&weight_zp).unwrap()[0];
        if in_zp != 0.0 || w_zp != 0.0 {
            return Err(invalid(OP, "zero points must be zero for floating-point operands"));
        }
        let inp: Vec<f32> = float_values(&input.data).unwrap().iter().map(|&v| v - in_zp).collect();
        let wgt: Vec<f32> = float_values(&weight.data).unwrap().iter().map(|&v| v - w_zp).collect();
        let bis = float_values(&bias.data).unwrap();

        let acc: Vec<f32> = match acc_t {
            DType::F32 => accumulate::<AccF32>(&g, &inp, &wgt).iter().map(|a| a.0).collect(),
            DType::F16 => accumulate::<AccF16>(&g, &inp, &wgt).iter().map(|a| a.0).collect(),
            DType::Bf16 => accumulate::<AccBf16>(&g, &inp, &wgt).iter().map(|a| a.0).collect(),
            _ => unreachable!("is_supported admits only float accumulators here"),
        };
        let out = acc
            .iter()
            .enumerate()
            .map(|(i, &a)| round_to(out_t, round_to(out_t, a) + bis[bias_idx(i % g.oc)]))
            .collect();
        float_data(out_t, out)?
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, output));
    Ok(())
}
