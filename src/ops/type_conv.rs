use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{DType, OpAttribute, Operator};
use crate::tensor::{Tensor, TensorData};

pub fn eval_cast(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    let out_def = state
        .find_tensor_def(output_name)
        .ok_or_else(|| Error::TensorNotFound(output_name.clone()))?;
    let out_dtype = out_def.dtype;
    let shape = input.shape.clone();

    let data = cast_tensor_data(&input.data, out_dtype)?;
    state.set_tensor(Tensor::new(output_name.clone(), shape, data));
    Ok(())
}

fn cast_tensor_data(src: &TensorData, dst_dtype: DType) -> Result<TensorData> {
    match (src, dst_dtype) {
        // F32 -> integer: TOSA spec uses round-to-nearest-even
        (TensorData::F32(v), DType::I32) => Ok(TensorData::I32(v.iter().map(|&x| x.round() as i32).collect())),
        (TensorData::F32(v), DType::I16) => Ok(TensorData::I16(v.iter().map(|&x| x.round() as i16).collect())),
        (TensorData::F32(v), DType::I8) => Ok(TensorData::I8(v.iter().map(|&x| x.round() as i8).collect())),
        (TensorData::F32(v), DType::F16) => Ok(TensorData::F16(v.iter().map(|&x| half::f16::from_f32(x)).collect())),
        (TensorData::F32(v), DType::Bf16) => Ok(TensorData::Bf16(v.iter().map(|&x| half::bf16::from_f32(x)).collect())),
        (TensorData::F32(v), DType::F64) => Ok(TensorData::F64(v.iter().map(|&x| x as f64).collect())),
        (TensorData::F32(v), DType::Bool) => Ok(TensorData::Bool(v.iter().map(|&x| x != 0.0).collect())),
        (TensorData::F32(_), DType::F32) => Ok(src.clone()),

        // I32 -> other
        (TensorData::I32(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|&x| x as f32).collect())),
        (TensorData::I32(v), DType::I8) => Ok(TensorData::I8(v.iter().map(|&x| x as i8).collect())),
        (TensorData::I32(v), DType::I16) => Ok(TensorData::I16(v.iter().map(|&x| x as i16).collect())),
        (TensorData::I32(v), DType::I64) => Ok(TensorData::I64(v.iter().map(|&x| x as i64).collect())),
        (TensorData::I32(v), DType::Bool) => Ok(TensorData::Bool(v.iter().map(|&x| x != 0).collect())),
        (TensorData::I32(_), DType::I32) => Ok(src.clone()),

        // I8 -> other
        (TensorData::I8(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|&x| x as f32).collect())),
        (TensorData::I8(v), DType::I16) => Ok(TensorData::I16(v.iter().map(|&x| x as i16).collect())),
        (TensorData::I8(v), DType::I32) => Ok(TensorData::I32(v.iter().map(|&x| x as i32).collect())),
        (TensorData::I8(_), DType::I8) => Ok(src.clone()),

        // I16 -> other
        (TensorData::I16(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|&x| x as f32).collect())),
        (TensorData::I16(v), DType::I32) => Ok(TensorData::I32(v.iter().map(|&x| x as i32).collect())),
        (TensorData::I16(v), DType::I8) => Ok(TensorData::I8(v.iter().map(|&x| x as i8).collect())),
        (TensorData::I16(_), DType::I16) => Ok(src.clone()),

        // I64 -> other
        (TensorData::I64(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|&x| x as f32).collect())),
        (TensorData::I64(v), DType::I32) => Ok(TensorData::I32(v.iter().map(|&x| x as i32).collect())),
        (TensorData::I64(_), DType::I64) => Ok(src.clone()),

        // F16 -> other
        (TensorData::F16(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|x| x.to_f32()).collect())),
        (TensorData::F16(_), DType::F16) => Ok(src.clone()),

        // BF16 -> other
        (TensorData::Bf16(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|x| x.to_f32()).collect())),
        (TensorData::Bf16(_), DType::Bf16) => Ok(src.clone()),

        // Bool -> other
        (TensorData::Bool(v), DType::I32) => Ok(TensorData::I32(v.iter().map(|&x| x as i32).collect())),
        (TensorData::Bool(v), DType::I8) => Ok(TensorData::I8(v.iter().map(|&x| x as i8).collect())),
        (TensorData::Bool(v), DType::F32) => Ok(TensorData::F32(v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect())),
        (TensorData::Bool(_), DType::Bool) => Ok(src.clone()),

        _ => Err(Error::DTypeMismatch(format!(
            "CAST: unsupported conversion from {:?} to {:?}",
            src.dtype(),
            dst_dtype
        ))),
    }
}

pub fn eval_rescale(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    // RESCALE inputs: [input, multiplier, shift, input_zp, output_zp]
    let multiplier = state.get_tensor(&op.inputs[1])?;
    let shift = state.get_tensor(&op.inputs[2])?;
    let input_zp = state.get_tensor(&op.inputs[3])?;
    let output_zp = state.get_tensor(&op.inputs[4])?;

    let _attr = match &op.attribute {
        OpAttribute::Rescale { scale32, per_channel, input_unsigned, output_unsigned } => {
            (*scale32, *per_channel, *input_unsigned, *output_unsigned)
        }
        _ => (true, false, false, false),
    };

    let out_def = state
        .find_tensor_def(output_name)
        .ok_or_else(|| Error::TensorNotFound(output_name.clone()))?;
    let out_dtype = out_def.dtype;
    let shape = input.shape.clone();

    // Get zero points as i32
    let in_zp = match &input_zp.data {
        TensorData::I32(v) => v.first().copied().unwrap_or(0) as i64,
        TensorData::I8(v) => v.first().copied().unwrap_or(0) as i64,
        _ => 0i64,
    };
    let out_zp = match &output_zp.data {
        TensorData::I32(v) => v.first().copied().unwrap_or(0) as i64,
        TensorData::I8(v) => v.first().copied().unwrap_or(0) as i64,
        _ => 0i64,
    };

    let mult_vals: Vec<i32> = match &multiplier.data {
        TensorData::I32(v) => v.clone(),
        TensorData::I16(v) => v.iter().map(|&x| x as i32).collect(),
        _ => vec![1],
    };
    let shift_vals: Vec<i32> = match &shift.data {
        TensorData::I32(v) => v.clone(),
        TensorData::I8(v) => v.iter().map(|&x| x as i32).collect(),
        _ => vec![0],
    };

    // Get input values as i64
    let in_vals: Vec<i64> = match &input.data {
        TensorData::I8(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I16(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I48(v) => v.clone(),
        _ => return Err(Error::DTypeMismatch("RESCALE: input must be integer".into())),
    };

    let num_channels = if _attr.1 { mult_vals.len() } else { 1 };
    let channel_size = if num_channels > 1 {
        in_vals.len() / num_channels
    } else {
        in_vals.len()
    };

    let mut out_vals = Vec::with_capacity(in_vals.len());
    for (i, &val) in in_vals.iter().enumerate() {
        let ch = if _attr.1 { i / channel_size } else { 0 };
        let ch = ch.min(mult_vals.len() - 1);

        let m = mult_vals[ch] as i64;
        let s = shift_vals[ch.min(shift_vals.len() - 1)];

        let shifted = val - in_zp;
        // Multiply and shift
        let result = if s > 0 {
            let product = shifted * m;
            let round = 1i64 << (s - 1);
            ((product + round) >> s) + out_zp
        } else {
            shifted * m + out_zp
        };
        out_vals.push(result);
    }

    let data = match out_dtype {
        DType::I8 => TensorData::I8(out_vals.iter().map(|&x| x.clamp(-128, 127) as i8).collect()),
        DType::I16 => TensorData::I16(out_vals.iter().map(|&x| x.clamp(-32768, 32767) as i16).collect()),
        DType::I32 => TensorData::I32(out_vals.iter().map(|&x| x as i32).collect()),
        _ => return Err(Error::DTypeMismatch("RESCALE: output must be integer".into())),
    };

    state.set_tensor(Tensor::new(output_name.clone(), shape, data));
    Ok(())
}
