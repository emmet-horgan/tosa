use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

pub fn eval_matmul(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let a = state.get_tensor(&op.inputs[0])?;
    let b = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    // A: [N, H, C], B: [N, C, W] -> Output: [N, H, W]
    let (n, h, c) = (a.shape[0], a.shape[1], a.shape[2]);
    let w = b.shape[2];
    let out_shape = vec![n, h, w];

    match (&a.data, &b.data) {
        (TensorData::F32(av), TensorData::F32(bv)) => {
            let mut output = vec![0.0f32; n * h * w];
            for batch in 0..n {
                for row in 0..h {
                    for col in 0..w {
                        let mut acc = 0.0f32;
                        for k in 0..c {
                            let a_idx = batch * h * c + row * c + k;
                            let b_idx = batch * c * w + k * w + col;
                            acc += av[a_idx] * bv[b_idx];
                        }
                        output[batch * h * w + row * w + col] = acc;
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        (TensorData::I8(av), TensorData::I8(bv)) => {
            let mut output = vec![0i32; n * h * w];
            for batch in 0..n {
                for row in 0..h {
                    for col in 0..w {
                        let mut acc = 0i64;
                        for k in 0..c {
                            let a_idx = batch * h * c + row * c + k;
                            let b_idx = batch * c * w + k * w + col;
                            acc += av[a_idx] as i64 * bv[b_idx] as i64;
                        }
                        output[batch * h * w + row * w + col] = acc as i32;
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::I32(output)));
        }
        (TensorData::I16(av), TensorData::I8(bv)) => {
            let mut output = vec![0i64; n * h * w];
            for batch in 0..n {
                for row in 0..h {
                    for col in 0..w {
                        let mut acc = 0i64;
                        for k in 0..c {
                            let a_idx = batch * h * c + row * c + k;
                            let b_idx = batch * c * w + k * w + col;
                            acc += av[a_idx] as i64 * bv[b_idx] as i64;
                        }
                        output[batch * h * w + row * w + col] = acc;
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::I48(output)));
        }
        _ => {
            return Err(Error::DTypeMismatch(
                "MATMUL: unsupported dtype combination".into(),
            ))
        }
    }

    Ok(())
}

pub fn eval_avg_pool2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (kernel, stride, pad) = match &op.attribute {
        OpAttribute::Pool2d {
            kernel,
            stride,
            pad,
        } => (kernel.clone(), stride.clone(), pad.clone()),
        _ => return Err(Error::GraphError("AVG_POOL2D: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    let (n, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let kh = kernel[0] as usize;
    let kw = kernel[1] as usize;
    let sh = stride[0] as usize;
    let sw = stride[1] as usize;
    let pt = pad[0] as usize;
    let pb = pad[1] as usize;
    let pl = pad[2] as usize;
    let pr = pad[3] as usize;

    let oh = (ih + pt + pb - kh) / sh + 1;
    let ow = (iw + pl + pr - kw) / sw + 1;
    let out_shape = vec![n, oh, ow, ic];

    match &input.data {
        TensorData::F32(inp) => {
            let mut output = vec![0.0f32; n * oh * ow * ic];
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        for c in 0..ic {
                            let mut sum = 0.0f32;
                            let mut count = 0u32;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * sh + ky) as isize - pt as isize;
                                    let ix = (x * sw + kx) as isize - pl as isize;
                                    if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                        let idx = batch * ih * iw * ic
                                            + iy as usize * iw * ic
                                            + ix as usize * ic
                                            + c;
                                        sum += inp[idx];
                                        count += 1;
                                    }
                                }
                            }
                            let out_idx = batch * oh * ow * ic + y * ow * ic + x * ic + c;
                            output[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        _ => {
            return Err(Error::DTypeMismatch(
                "AVG_POOL2D: unsupported dtype".into(),
            ))
        }
    }
    Ok(())
}

pub fn eval_max_pool2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (kernel, stride, pad) = match &op.attribute {
        OpAttribute::Pool2d {
            kernel,
            stride,
            pad,
        } => (kernel.clone(), stride.clone(), pad.clone()),
        _ => return Err(Error::GraphError("MAX_POOL2D: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    let (n, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let kh = kernel[0] as usize;
    let kw = kernel[1] as usize;
    let sh = stride[0] as usize;
    let sw = stride[1] as usize;
    let pt = pad[0] as usize;
    let pb = pad[1] as usize;
    let pl = pad[2] as usize;
    let pr = pad[3] as usize;

    let oh = (ih + pt + pb - kh) / sh + 1;
    let ow = (iw + pl + pr - kw) / sw + 1;
    let out_shape = vec![n, oh, ow, ic];

    match &input.data {
        TensorData::F32(inp) => {
            let mut output = vec![f32::NEG_INFINITY; n * oh * ow * ic];
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        for c in 0..ic {
                            let mut max_val = f32::NEG_INFINITY;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * sh + ky) as isize - pt as isize;
                                    let ix = (x * sw + kx) as isize - pl as isize;
                                    if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                        let idx = batch * ih * iw * ic
                                            + iy as usize * iw * ic
                                            + ix as usize * ic
                                            + c;
                                        max_val = max_val.max(inp[idx]);
                                    }
                                }
                            }
                            let out_idx = batch * oh * ow * ic + y * ow * ic + x * ic + c;
                            output[out_idx] = max_val;
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        TensorData::I8(inp) => {
            let mut output = vec![i8::MIN; n * oh * ow * ic];
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        for c in 0..ic {
                            let mut max_val = i8::MIN;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let iy = (y * sh + ky) as isize - pt as isize;
                                    let ix = (x * sw + kx) as isize - pl as isize;
                                    if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                        let idx = batch * ih * iw * ic
                                            + iy as usize * iw * ic
                                            + ix as usize * ic
                                            + c;
                                        max_val = max_val.max(inp[idx]);
                                    }
                                }
                            }
                            let out_idx = batch * oh * ow * ic + y * ow * ic + x * ic + c;
                            output[out_idx] = max_val;
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::I8(output)));
        }
        _ => {
            return Err(Error::DTypeMismatch(
                "MAX_POOL2D: unsupported dtype".into(),
            ))
        }
    }
    Ok(())
}

pub fn eval_argmax(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let axis = match &op.attribute {
        OpAttribute::ArgMax { axis } => *axis as usize,
        _ => return Err(Error::GraphError("ARGMAX: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];
    let in_shape = &input.shape;

    if axis >= in_shape.len() {
        return Err(Error::GraphError(format!(
            "ARGMAX: axis {} >= rank {}",
            axis,
            in_shape.len()
        )));
    }

    // Output shape: input shape with axis dimension removed
    let mut out_shape: Vec<usize> = in_shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    let axis_size = in_shape[axis];
    let outer: usize = in_shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = in_shape[axis + 1..].iter().product::<usize>().max(1);

    match &input.data {
        TensorData::F32(inp) => {
            let mut output = vec![0i32; out_count];
            for o in 0..outer {
                for i in 0..inner {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0i32;
                    for a in 0..axis_size {
                        let idx = o * axis_size * inner + a * inner + i;
                        if inp[idx] > max_val {
                            max_val = inp[idx];
                            max_idx = a as i32;
                        }
                    }
                    output[o * inner + i] = max_idx;
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::I32(output)));
        }
        TensorData::I32(inp) => {
            let mut output = vec![0i32; out_count];
            for o in 0..outer {
                for i in 0..inner {
                    let mut max_val = i32::MIN;
                    let mut max_idx = 0i32;
                    for a in 0..axis_size {
                        let idx = o * axis_size * inner + a * inner + i;
                        if inp[idx] > max_val {
                            max_val = inp[idx];
                            max_idx = a as i32;
                        }
                    }
                    output[o * inner + i] = max_idx;
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::I32(output)));
        }
        _ => {
            return Err(Error::DTypeMismatch("ARGMAX: unsupported dtype".into()))
        }
    }
    Ok(())
}

pub fn eval_depthwise_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (pad, stride, dilation) = match &op.attribute {
        OpAttribute::DepthwiseConv2d {
            pad, stride, dilation, ..
        } => (pad.clone(), stride.clone(), dilation.clone()),
        _ => return Err(Error::GraphError("DEPTHWISE_CONV2D: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let weight = state.get_tensor(&op.inputs[1])?;
    let bias = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    // Input: [N, IH, IW, C], Weight: [KH, KW, C, M], Bias: [C*M], Output: [N, OH, OW, C*M]
    let (n, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let (kh, kw, _wc, m) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);

    let pt = pad[0] as usize;
    let pb = pad[1] as usize;
    let pl = pad[2] as usize;
    let pr = pad[3] as usize;
    let sh = stride[0] as usize;
    let sw = stride[1] as usize;
    let dh = dilation[0] as usize;
    let dw = dilation[1] as usize;

    let oh = (ih + pt + pb - (kh - 1) * dh - 1) / sh + 1;
    let ow = (iw + pl + pr - (kw - 1) * dw - 1) / sw + 1;
    let oc = ic * m;
    let out_shape = vec![n, oh, ow, oc];

    match (&input.data, &weight.data, &bias.data) {
        (TensorData::F32(inp), TensorData::F32(wgt), TensorData::F32(bis)) => {
            let mut output = vec![0.0f32; n * oh * ow * oc];
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        for c in 0..ic {
                            for mi in 0..m {
                                let oc_idx = c * m + mi;
                                let mut acc = bis[oc_idx];
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (y * sh + ky * dh) as isize - pt as isize;
                                        let ix = (x * sw + kx * dw) as isize - pl as isize;
                                        if iy >= 0 && iy < ih as isize && ix >= 0 && ix < iw as isize {
                                            let iy = iy as usize;
                                            let ix = ix as usize;
                                            let i_idx = batch * ih * iw * ic + iy * iw * ic + ix * ic + c;
                                            let w_idx = ky * kw * ic * m + kx * ic * m + c * m + mi;
                                            acc += inp[i_idx] * wgt[w_idx];
                                        }
                                    }
                                }
                                output[batch * oh * ow * oc + y * ow * oc + x * oc + oc_idx] = acc;
                            }
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        _ => return Err(Error::DTypeMismatch("DEPTHWISE_CONV2D: unsupported dtype".into())),
    }
    Ok(())
}

pub fn eval_transpose_conv2d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (out_pad, stride) = match &op.attribute {
        OpAttribute::TransposeConv2d { out_pad, stride, .. } => (out_pad.clone(), stride.clone()),
        _ => return Err(Error::GraphError("TRANSPOSE_CONV2D: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let weight = state.get_tensor(&op.inputs[1])?;
    let bias = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    // Input: [N, IH, IW, IC], Weight: [OC, KH, KW, IC], Bias: [OC]
    let (n, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let (oc, kh, kw, _wic) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);

    let sh = stride[0] as usize;
    let sw = stride[1] as usize;
    let op_top = out_pad[0] as usize;
    let op_bottom = out_pad[1] as usize;
    let op_left = out_pad[2] as usize;
    let op_right = out_pad[3] as usize;

    let oh = (ih - 1) * sh + kh - op_top - op_bottom;
    let ow = (iw - 1) * sw + kw - op_left - op_right;
    let out_shape = vec![n, oh, ow, oc];

    match (&input.data, &weight.data, &bias.data) {
        (TensorData::F32(inp), TensorData::F32(wgt), TensorData::F32(bis)) => {
            let mut output = vec![0.0f32; n * oh * ow * oc];
            // Initialize with bias
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        for oc_idx in 0..oc {
                            output[batch * oh * ow * oc + y * ow * oc + x * oc + oc_idx] = bis[oc_idx];
                        }
                    }
                }
            }
            // Scatter input contributions
            for batch in 0..n {
                for iy in 0..ih {
                    for ix in 0..iw {
                        for oc_idx in 0..oc {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let oy = iy * sh + ky;
                                    let ox = ix * sw + kx;
                                    let oy = oy as isize - op_top as isize;
                                    let ox = ox as isize - op_left as isize;
                                    if oy >= 0 && oy < oh as isize && ox >= 0 && ox < ow as isize {
                                        let oy = oy as usize;
                                        let ox = ox as usize;
                                        for c in 0..ic {
                                            let i_idx = batch * ih * iw * ic + iy * iw * ic + ix * ic + c;
                                            let w_idx = oc_idx * kh * kw * ic + ky * kw * ic + kx * ic + c;
                                            let o_idx = batch * oh * ow * oc + oy * ow * oc + ox * oc + oc_idx;
                                            output[o_idx] += inp[i_idx] * wgt[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        _ => return Err(Error::DTypeMismatch("TRANSPOSE_CONV2D: unsupported dtype".into())),
    }
    Ok(())
}

pub fn eval_conv3d(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let (pad, stride, dilation) = match &op.attribute {
        OpAttribute::Conv3d { pad, stride, dilation, .. } => (pad.clone(), stride.clone(), dilation.clone()),
        _ => return Err(Error::GraphError("CONV3D: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    let weight = state.get_tensor(&op.inputs[1])?;
    let bias = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    // Input: [N, ID, IH, IW, IC], Weight: [OC, KD, KH, KW, IC], Bias: [OC]
    let (n, id, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3], input.shape[4]);
    let (oc, kd, kh, kw, _) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4]);

    let (pd_f, pd_b) = (pad[0] as usize, pad[1] as usize);
    let (pt, pb) = (pad[2] as usize, pad[3] as usize);
    let (pl, pr) = (pad[4] as usize, pad[5] as usize);
    let (sd, sh, sw) = (stride[0] as usize, stride[1] as usize, stride[2] as usize);
    let (dd, dh, dw) = (dilation[0] as usize, dilation[1] as usize, dilation[2] as usize);

    let od = (id + pd_f + pd_b - (kd - 1) * dd - 1) / sd + 1;
    let oh = (ih + pt + pb - (kh - 1) * dh - 1) / sh + 1;
    let ow = (iw + pl + pr - (kw - 1) * dw - 1) / sw + 1;
    let out_shape = vec![n, od, oh, ow, oc];

    match (&input.data, &weight.data, &bias.data) {
        (TensorData::F32(inp), TensorData::F32(wgt), TensorData::F32(bis)) => {
            let mut output = vec![0.0f32; n * od * oh * ow * oc];
            for batch in 0..n {
                for d in 0..od {
                    for y in 0..oh {
                        for x in 0..ow {
                            for oc_idx in 0..oc {
                                let mut acc = bis[oc_idx];
                                for kdi in 0..kd {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let iz = (d * sd + kdi * dd) as isize - pd_f as isize;
                                            let iy = (y * sh + ky * dh) as isize - pt as isize;
                                            let ix = (x * sw + kx * dw) as isize - pl as isize;
                                            if iz >= 0 && iz < id as isize
                                                && iy >= 0 && iy < ih as isize
                                                && ix >= 0 && ix < iw as isize
                                            {
                                                let iz = iz as usize;
                                                let iy = iy as usize;
                                                let ix = ix as usize;
                                                for c in 0..ic {
                                                    let i_idx = batch * id * ih * iw * ic + iz * ih * iw * ic + iy * iw * ic + ix * ic + c;
                                                    let w_idx = oc_idx * kd * kh * kw * ic + kdi * kh * kw * ic + ky * kw * ic + kx * ic + c;
                                                    acc += inp[i_idx] * wgt[w_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                                let o_idx = batch * od * oh * ow * oc + d * oh * ow * oc + y * ow * oc + x * oc + oc_idx;
                                output[o_idx] = acc;
                            }
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        _ => return Err(Error::DTypeMismatch("CONV3D: unsupported dtype".into())),
    }
    Ok(())
}

pub fn eval_table(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let table = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];
    let shape = input.shape.clone();

    match (&input.data, &table.data) {
        // INT8 input, 256-entry INT8 table -> INT8 output
        (TensorData::I8(inp), TensorData::I8(tbl)) => {
            let result: Vec<i8> = inp.iter().map(|&x| {
                let idx = (x as u8) as usize;
                tbl[idx]
            }).collect();
            state.set_tensor(Tensor::new(output_name.clone(), shape, TensorData::I8(result)));
        }
        // INT16 input, 513-entry INT16 table -> INT32 output (interpolation)
        (TensorData::I16(inp), TensorData::I16(tbl)) => {
            let result: Vec<i32> = inp.iter().map(|&x| {
                // TOSA TABLE for int16: upper 9 bits index, lower 7 bits fraction
                let x32 = x as i32;
                let integer_bits = 9;
                let fraction_bits = 7;

                let clamp_min = 0i32;
                let clamp_max = (1 << integer_bits) - 1;

                let idx = ((x32 + 32768) >> fraction_bits).clamp(clamp_min, clamp_max) as usize;
                let frac = (x32 + 32768) & ((1 << fraction_bits) - 1);

                let base = tbl[idx] as i32;
                let next = if idx < tbl.len() - 1 { tbl[idx + 1] as i32 } else { base };
                let interp = base + ((next - base) * frac >> fraction_bits);
                interp
            }).collect();
            state.set_tensor(Tensor::new(output_name.clone(), shape, TensorData::I32(result)));
        }
        _ => return Err(Error::DTypeMismatch("TABLE: unsupported dtype combination".into())),
    }
    Ok(())
}

pub fn eval_resize(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let mode = match &op.attribute {
        OpAttribute::Resize { mode } => *mode,
        _ => return Err(Error::GraphError("RESIZE: missing attribute".into())),
    };

    let input = state.get_tensor(&op.inputs[0])?;
    // Scale, offset, border from input tensors 1, 2, 3
    let scale_tensor = state.get_tensor(&op.inputs[1])?;
    let offset_tensor = state.get_tensor(&op.inputs[2])?;
    let border_tensor = state.get_tensor(&op.inputs[3])?;
    let output_name = &op.outputs[0];

    let out_def = state.find_tensor_def(output_name)
        .ok_or_else(|| Error::TensorNotFound(output_name.clone()))?;
    let out_shape = crate::tensor::shape_from_ir(&out_def.shape);

    // Input: [N, IH, IW, C], Output: [N, OH, OW, C]
    let (n, ih, iw, ic) = (input.shape[0], input.shape[1], input.shape[2], input.shape[3]);
    let (oh, ow) = (out_shape[1], out_shape[2]);

    let scale: Vec<i64> = match &scale_tensor.data {
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I64(v) => v.clone(),
        _ => return Err(Error::DTypeMismatch("RESIZE: scale must be integer".into())),
    };
    let offset: Vec<i64> = match &offset_tensor.data {
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I64(v) => v.clone(),
        _ => return Err(Error::DTypeMismatch("RESIZE: offset must be integer".into())),
    };
    let _border: Vec<i64> = match &border_tensor.data {
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I64(v) => v.clone(),
        _ => return Err(Error::DTypeMismatch("RESIZE: border must be integer".into())),
    };

    // scale = [scale_y_n, scale_y_d, scale_x_n, scale_x_d]
    let (sy_n, sy_d, sx_n, sx_d) = (scale[0], scale[1], scale[2], scale[3]);
    let (off_y, off_x) = (offset[0], offset[1]);

    let is_nearest = mode == 1; // ResizeMode::NEAREST = 1

    match &input.data {
        TensorData::F32(inp) => {
            let mut output = vec![0.0f32; n * oh * ow * ic];
            for batch in 0..n {
                for y in 0..oh {
                    for x in 0..ow {
                        // Compute input coordinate
                        let iy_num = y as i64 * sy_d + off_y;
                        let ix_num = x as i64 * sx_d + off_x;

                        if is_nearest {
                            let iy = (iy_num / sy_n).clamp(0, ih as i64 - 1) as usize;
                            let ix = (ix_num / sx_n).clamp(0, iw as i64 - 1) as usize;
                            for c in 0..ic {
                                let i_idx = batch * ih * iw * ic + iy * iw * ic + ix * ic + c;
                                let o_idx = batch * oh * ow * ic + y * ow * ic + x * ic + c;
                                output[o_idx] = inp[i_idx];
                            }
                        } else {
                            // Bilinear
                            let iy_f = iy_num as f64 / sy_n as f64;
                            let ix_f = ix_num as f64 / sx_n as f64;
                            let iy0 = iy_f.floor() as i64;
                            let ix0 = ix_f.floor() as i64;
                            let dy = iy_f - iy0 as f64;
                            let dx = ix_f - ix0 as f64;

                            for c in 0..ic {
                                let mut val = 0.0f64;
                                for jy in 0..=1i64 {
                                    for jx in 0..=1i64 {
                                        let cy = (iy0 + jy).clamp(0, ih as i64 - 1) as usize;
                                        let cx = (ix0 + jx).clamp(0, iw as i64 - 1) as usize;
                                        let i_idx = batch * ih * iw * ic + cy * iw * ic + cx * ic + c;
                                        let wy = if jy == 0 { 1.0 - dy } else { dy };
                                        let wx = if jx == 0 { 1.0 - dx } else { dx };
                                        val += inp[i_idx] as f64 * wy * wx;
                                    }
                                }
                                let o_idx = batch * oh * ow * ic + y * ow * ic + x * ic + c;
                                output[o_idx] = val as f32;
                            }
                        }
                    }
                }
            }
            state.set_tensor(Tensor::new(output_name.clone(), out_shape, TensorData::F32(output)));
        }
        _ => return Err(Error::DTypeMismatch("RESIZE: unsupported dtype".into())),
    }
    Ok(())
}
