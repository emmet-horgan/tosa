//! RESIZE: input [N,IH,IW,C] -> output [N,OH,OW,C] with SHAPE operands
//! scale [y_n, y_d, x_n, x_d], offset [y, x] and border [y, x].
//!
//! Output position `o` samples input coordinate `(o * d + offset) / n`.
//! Integer inputs use exact fixed-point arithmetic (BILINEAR widens INT8 to
//! INT32 and INT16 to INT48, scaled by `y_n * x_n`). Float inputs follow the
//! reference model's rounding: the fractional position is rounded to the
//! element type, each bilinear term is rounded to the output type, and the
//! terms are summed with a rounding after every addition.

use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::ops::common::{
    check_declared_output, expect_operands, expect_rank, float_data, float_values, int_data, int_values,
    invalid, round_to,
};
use crate::tensor::Tensor;

const NEAREST: u32 = 1;
const BILINEAR: u32 = 2;

fn shape_values(state: &InterpreterState, name: &str, len: usize, what: &str) -> Result<Vec<i64>> {
    let t = state.get_tensor(name)?;
    let v = int_values(&t.data).ok_or_else(|| invalid("RESIZE", format!("{what} must be a shape")))?;
    if v.len() != len {
        return Err(invalid("RESIZE", format!("{what} must have {len} elements")));
    }
    Ok(v)
}

/// Sampling parameters along one axis.
struct Axis {
    n: i64,
    d: i64,
    offset: i64,
    size: usize,
}

impl Axis {
    /// (floor index, numerator remainder, float position) of output index `o`.
    fn position(&self, o: usize) -> (i64, i64, f32) {
        let v = o as i64 * self.d + self.offset;
        let i = v.div_euclid(self.n);
        (i, v - i * self.n, v as f32 / self.n as f32)
    }

    fn clamp(&self, i: i64) -> usize {
        i.clamp(0, self.size as i64 - 1) as usize
    }
}

pub fn eval_resize(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    const OP: &str = "RESIZE";
    let OpAttribute::Resize { mode } = op.attribute else {
        return Err(invalid(OP, "missing attribute"));
    };
    if mode != NEAREST && mode != BILINEAR {
        return Err(invalid(OP, format!("unknown mode {mode}")));
    }
    expect_operands(op, OP, 4, 1)?;
    let scale = shape_values(state, &op.inputs[1], 4, "scale")?;
    let offset = shape_values(state, &op.inputs[2], 2, "offset")?;
    let border = shape_values(state, &op.inputs[3], 2, "border")?;
    let input = state.get_tensor(&op.inputs[0])?;
    expect_rank(OP, "input", input, 4)?;
    let [n, ih, iw, c] = input.shape[..] else { unreachable!() };

    let [y_n, y_d, x_n, x_d] = scale[..] else { unreachable!() };
    if scale.iter().any(|&s| s <= 0) || y_n > 1 << 11 || x_n > 1 << 11 || y_d >= 16 * y_n || x_d >= 16 * x_n {
        return Err(invalid(OP, format!("invalid scale {scale:?}")));
    }
    for (i, (sn, off, bord)) in [(y_n, offset[0], border[0]), (x_n, offset[1], border[1])].into_iter().enumerate() {
        if off < -sn || off >= 16 * sn || bord < -16 * sn || bord >= sn {
            return Err(invalid(OP, format!("invalid offset/border for axis {i}")));
        }
    }
    let extent = |size: usize, sn: i64, sd: i64, off: i64, bord: i64| -> Result<usize> {
        let num = (size as i64 - 1) * sn - off + bord;
        if num % sd != 0 {
            return Err(invalid(OP, "scale, offset and border must yield an exact output size"));
        }
        let out = num / sd + 1;
        if out < 1 {
            return Err(invalid(OP, "output size must be positive"));
        }
        Ok(out as usize)
    };
    let oh = extent(ih, y_n, y_d, offset[0], border[0])?;
    let ow = extent(iw, x_n, x_d, offset[1], border[1])?;
    if ih.max(iw).max(oh).max(ow) >= 16384 {
        return Err(invalid(OP, "dimensions must be below 16384"));
    }
    let out_shape = vec![n, oh, ow, c];
    let in_t = input.dtype();
    let out_t = check_declared_output(state, OP, &op.outputs[0], &out_shape)?.unwrap_or(in_t);
    use DType::*;
    let types_ok = match (in_t, out_t) {
        (I8, I8) | (I16, I16) | (F16, F16) | (Bf16, Bf16) | (F32, F32) | (Fp8E4M3, Fp8E4M3) | (Fp8E5M2, Fp8E5M2) => true,
        (I8, I32) | (I16, I48) => mode == BILINEAR,
        _ => false,
    };
    if !types_ok || (in_t.is_int() && mode == NEAREST && in_t != out_t) {
        return Err(Error::DTypeMismatch(format!("{OP}: unsupported types {in_t:?} -> {out_t:?}")));
    }

    let ya = Axis { n: y_n, d: y_d, offset: offset[0], size: ih };
    let xa = Axis { n: x_n, d: x_d, offset: offset[1], size: iw };
    let at = |b: usize, y: usize, x: usize, ch: usize| ((b * ih + y) * iw + x) * c + ch;
    let count = n * oh * ow * c;

    let data = if let Some(v) = int_values(&input.data) {
        let mut out = Vec::with_capacity(count);
        for b in 0..n {
            for oy in 0..oh {
                let (iy, dy, _) = ya.position(oy);
                let (y0, y1) = (ya.clamp(iy), ya.clamp(iy + 1));
                for ox in 0..ow {
                    let (ix, dx, _) = xa.position(ox);
                    let (x0, x1) = (xa.clamp(ix), xa.clamp(ix + 1));
                    for ch in 0..c {
                        out.push(if mode == NEAREST {
                            let y = if 2 * dy >= y_n { y1 } else { y0 };
                            let x = if 2 * dx >= x_n { x1 } else { x0 };
                            v[at(b, y, x, ch)]
                        } else {
                            v[at(b, y0, x0, ch)] * (y_n - dy) * (x_n - dx)
                                + v[at(b, y0, x1, ch)] * (y_n - dy) * dx
                                + v[at(b, y1, x0, ch)] * dy * (x_n - dx)
                                + v[at(b, y1, x1, ch)] * dy * dx
                        });
                    }
                }
            }
        }
        int_data(out_t, out)?
    } else {
        let v = float_values(&input.data).unwrap();
        // FP8 positions are computed in FP32; FP16/BF16 in their own type.
        let frac_t = if matches!(in_t, Fp8E4M3 | Fp8E5M2) { F32 } else { in_t };
        let r = |x: f32| round_to(out_t, x);
        let mut out = Vec::with_capacity(count);
        for b in 0..n {
            for oy in 0..oh {
                let (iy, _, fy) = ya.position(oy);
                let dy = round_to(frac_t, fy - iy as f32);
                let (y0, y1) = (ya.clamp(iy), ya.clamp(iy + 1));
                for ox in 0..ow {
                    let (ix, _, fx) = xa.position(ox);
                    let dx = round_to(frac_t, fx - ix as f32);
                    let (x0, x1) = (xa.clamp(ix), xa.clamp(ix + 1));
                    for ch in 0..c {
                        out.push(if mode == NEAREST {
                            let y = if dy >= 0.5 { y1 } else { y0 };
                            let x = if dx >= 0.5 { x1 } else { x0 };
                            v[at(b, y, x, ch)]
                        } else {
                            let mut acc = r(v[at(b, y0, x0, ch)] * (1.0 - dy) * (1.0 - dx));
                            acc = r(acc + r(v[at(b, y0, x1, ch)] * (1.0 - dy) * dx));
                            acc = r(acc + r(v[at(b, y1, x0, ch)] * dy * (1.0 - dx)));
                            r(acc + r(v[at(b, y1, x1, ch)] * dy * dx))
                        });
                    }
                }
            }
        }
        float_data(out_t, out)?
    };
    state.set_tensor(Tensor::new(op.outputs[0].clone(), out_shape, data));
    Ok(())
}
