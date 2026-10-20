//! FFT2D and RFFT2D: direct (O(n^2)) discrete Fourier transforms over the
//! H and W dimensions of [N,H,W] FP32 tensors, evaluated exactly as the
//! reference model does: the angle `2*pi*(ay/H + ax/W)` uses integer-reduced
//! indices and is formed in double precision then rounded to FP32, and the
//! sums accumulate in FP32 in input order.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::common::{check_declared_output, expect_operands, expect_rank, invalid};
use crate::tensor::{Tensor, TensorData};

fn f32_input<'a>(state: &'a InterpreterState, op: &str, name: &str) -> Result<(&'a [f32], &'a [usize])> {
    let t = state.get_tensor(name)?;
    expect_rank(op, "input", t, 3)?;
    match &t.data {
        TensorData::F32(v) => Ok((v, &t.shape)),
        other => Err(Error::DTypeMismatch(format!("{op}: inputs must be FP32, got {:?}", other.dtype()))),
    }
}

fn is_power_of_two(n: usize) -> bool {
    n.is_power_of_two()
}

/// The angle for input position (iy, ix) and output position (oy, ox).
fn angle(sign: f32, iy: usize, ix: usize, oy: usize, ox: usize, h: usize, w: usize) -> f32 {
    let ay = (iy * oy) % h;
    let ax = (ix * ox) % w;
    let frac = ay as f32 / h as f32 + ax as f32 / w as f32;
    ((sign * 2.0) as f64 * PI * frac as f64) as f32
}

fn store(state: &mut InterpreterState, op: &Operator, name: &str, shape: Vec<usize>, real: Vec<f32>, imag: Vec<f32>) -> Result<()> {
    for out in &op.outputs {
        if let Some(d) = check_declared_output(state, name, out, &shape)?
            && d != DType::F32
        {
            return Err(Error::DTypeMismatch(format!("{name}: outputs must be FP32, got {d:?}")));
        }
    }
    state.set_tensor(Tensor::new(op.outputs[0].clone(), shape.clone(), TensorData::F32(real)));
    state.set_tensor(Tensor::new(op.outputs[1].clone(), shape, TensorData::F32(imag)));
    Ok(())
}

pub fn eval_fft2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "FFT2D";
    let OpAttribute::Fft2d { inverse } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    expect_operands(op, OP, 2, 2)?;
    let (re, shape) = f32_input(state, OP, &op.inputs[0])?;
    let (im, im_shape) = f32_input(state, OP, &op.inputs[1])?;
    if shape != im_shape {
        return Err(Error::ShapeMismatch(format!("{OP}: real {shape:?} and imaginary {im_shape:?} shapes differ")));
    }
    let [n, h, w] = shape[..] else { unreachable!() };
    if !is_power_of_two(h) || !is_power_of_two(w) {
        return Err(invalid(OP, "height and width must be powers of two"));
    }
    let sign = if inverse { -1.0 } else { 1.0 };
    let mut out_re = Vec::with_capacity(n * h * w);
    let mut out_im = Vec::with_capacity(n * h * w);
    for b in 0..n {
        for oy in 0..h {
            for ox in 0..w {
                let (mut sum_re, mut sum_im) = (0f32, 0f32);
                for iy in 0..h {
                    for ix in 0..w {
                        let i = (b * h + iy) * w + ix;
                        let a = angle(sign, iy, ix, oy, ox, h, w);
                        let (s, c) = (a.sin(), a.cos());
                        sum_re += re[i] * c + im[i] * s;
                        sum_im += -re[i] * s + im[i] * c;
                    }
                }
                out_re.push(sum_re);
                out_im.push(sum_im);
            }
        }
    }
    let shape = shape.to_vec();
    store(state, op, OP, shape, out_re, out_im)
}

/// RFFT2D: real input [N,H,W] -> real and imaginary parts [N,H,W/2+1].
pub fn eval_rfft2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "RFFT2D";
    expect_operands(op, OP, 1, 2)?;
    let (x, shape) = f32_input(state, OP, &op.inputs[0])?;
    let [n, h, w] = shape[..] else { unreachable!() };
    if !is_power_of_two(h) || !is_power_of_two(w) {
        return Err(invalid(OP, "height and width must be powers of two"));
    }
    let ow = w / 2 + 1;
    let (half_h, half_w) = (h / 2, w / 2);
    let mut out_re = Vec::with_capacity(n * h * ow);
    let mut out_im = Vec::with_capacity(n * h * ow);
    for b in 0..n {
        for oy in 0..h {
            for ox in 0..ow {
                let (mut sum_re, mut sum_im) = (0f32, 0f32);
                for iy in 0..h {
                    for ix in 0..w {
                        let v = x[(b * h + iy) * w + ix];
                        let a = angle(1.0, iy, ix, oy, ox, h, w);
                        sum_re += v * a.cos();
                        // The imaginary parts at (0,0), (0,W/2), (H/2,0) and
                        // (H/2,W/2) are exactly zero; sin(pi) is not, so they
                        // are skipped.
                        let in_scope = (h > 1 && ((iy * oy) % h) % half_h > 0)
                            || (w > 1 && ((ix * ox) % w) % half_w > 0);
                        if in_scope {
                            sum_im += -v * a.sin();
                        }
                    }
                }
                out_re.push(sum_re);
                out_im.push(sum_im);
            }
        }
    }
    store(state, op, OP, vec![n, h, ow], out_re, out_im)
}
