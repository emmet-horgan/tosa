use crate::broadcast::flat_to_nd;
use crate::error::{Error, Result};
use crate::interpreter::InterpreterState;
use crate::ir::{OpAttribute, Operator};
use crate::tensor::{shape_from_ir, Tensor, TensorData};

pub fn eval_reshape(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    // The output shape comes from the tensor definition in the block
    let tensor_def = state
        .find_tensor_def(output_name)
        .ok_or_else(|| Error::TensorNotFound(output_name.clone()))?;

    let new_shape = shape_from_ir(&tensor_def.shape);

    let expected_count: usize = new_shape.iter().product::<usize>().max(1);
    let actual_count = input.element_count();

    if expected_count != actual_count {
        return Err(Error::ShapeMismatch(format!(
            "RESHAPE: input has {} elements but output shape {:?} requires {}",
            actual_count, new_shape, expected_count
        )));
    }

    let data = input.data.clone();
    state.set_tensor(Tensor::new(output_name.clone(), new_shape, data));
    Ok(())
}

/// TRANSPOSE: permute dimensions according to `perms`. If perms is empty, reverse dimensions.
pub fn eval_transpose(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];
    let rank = input.shape.len();

    let perms: Vec<usize> = match &op.attribute {
        OpAttribute::Transpose { perms } if !perms.is_empty() => {
            perms.iter().map(|&p| p as usize).collect()
        }
        _ => {
            // Default: reverse dimensions
            (0..rank).rev().collect()
        }
    };

    if perms.len() != rank {
        return Err(Error::ShapeMismatch(format!(
            "TRANSPOSE: perms length {} does not match input rank {}",
            perms.len(),
            rank
        )));
    }

    // Compute output shape
    let out_shape: Vec<usize> = perms.iter().map(|&p| input.shape[p]).collect();
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    // Compute input strides
    let mut in_strides = vec![1usize; rank];
    for dim in (0..rank.saturating_sub(1)).rev() {
        in_strides[dim] = in_strides[dim + 1] * input.shape[dim + 1];
    }

    // For each output element, compute the corresponding input flat index
    macro_rules! transpose_typed {
        ($variant:ident, $v:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for out_flat in 0..out_count {
                let out_nd = flat_to_nd(out_flat, &out_shape);
                let mut in_flat = 0usize;
                for (out_dim, &out_idx) in out_nd.iter().enumerate() {
                    let in_dim = perms[out_dim];
                    in_flat += out_idx * in_strides[in_dim];
                }
                result.push($v[in_flat]);
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &input.data {
        TensorData::F32(v) => transpose_typed!(F32, v),
        TensorData::I32(v) => transpose_typed!(I32, v),
        TensorData::I8(v) => transpose_typed!(I8, v),
        TensorData::I16(v) => transpose_typed!(I16, v),
        TensorData::I64(v) => transpose_typed!(I64, v),
        TensorData::Bool(v) => transpose_typed!(Bool, v),
        TensorData::I48(v) => transpose_typed!(I48, v),
        TensorData::F16(v) => transpose_typed!(F16, v),
        TensorData::Bf16(v) => transpose_typed!(Bf16, v),
        TensorData::F64(v) => transpose_typed!(F64, v),
        TensorData::I4(v) => transpose_typed!(I4, v),
        TensorData::Fp8E4M3(v) => transpose_typed!(Fp8E4M3, v),
        TensorData::Fp8E5M2(v) => transpose_typed!(Fp8E5M2, v),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// CONCAT: concatenate multiple input tensors along the given axis.
pub fn eval_concat(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let axis = match &op.attribute {
        OpAttribute::Concat { axis } => *axis as usize,
        _ => {
            return Err(Error::ShapeMismatch(
                "CONCAT: missing Concat attribute".to_string(),
            ));
        }
    };

    let output_name = &op.outputs[0];

    if op.inputs.is_empty() {
        return Err(Error::ShapeMismatch(
            "CONCAT: no inputs provided".to_string(),
        ));
    }

    // Gather all input tensors' shapes and validate compatibility
    let first = state.get_tensor(&op.inputs[0])?;
    let rank = first.shape.len();

    if axis >= rank {
        return Err(Error::ShapeMismatch(format!(
            "CONCAT: axis {} out of range for rank {}",
            axis, rank
        )));
    }

    // Compute output shape
    let mut out_shape = first.shape.clone();
    let mut total_axis_dim = first.shape[axis];
    for input_name in op.inputs.iter().skip(1) {
        let t = state.get_tensor(input_name)?;
        if t.shape.len() != rank {
            return Err(Error::ShapeMismatch(format!(
                "CONCAT: all inputs must have the same rank, got {} and {}",
                rank,
                t.shape.len()
            )));
        }
        for (d, (&a, &b)) in first.shape.iter().zip(t.shape.iter()).enumerate() {
            if d != axis && a != b {
                return Err(Error::ShapeMismatch(format!(
                    "CONCAT: dimension {} mismatch: {} vs {}",
                    d, a, b
                )));
            }
        }
        total_axis_dim += t.shape[axis];
    }
    out_shape[axis] = total_axis_dim;

    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    // Build the output by iterating over the output space and finding which input
    // each element belongs to
    macro_rules! concat_typed {
        ($variant:ident) => {{
            let mut result = Vec::with_capacity(out_count);
            for out_flat in 0..out_count {
                let out_nd = flat_to_nd(out_flat, &out_shape);
                // Determine which input this index corresponds to along the concat axis
                let mut axis_idx = out_nd[axis];
                let mut found = false;
                for input_name in &op.inputs {
                    let t = state.get_tensor(input_name)?;
                    let t_axis_dim = t.shape[axis];
                    if axis_idx < t_axis_dim {
                        // This element comes from this input
                        let mut in_nd = out_nd.clone();
                        in_nd[axis] = axis_idx;
                        let in_flat = nd_to_flat_shape(&in_nd, &t.shape);
                        match &t.data {
                            TensorData::$variant(v) => result.push(v[in_flat]),
                            _ => {
                                return Err(Error::DTypeMismatch(
                                    "CONCAT: all inputs must have the same dtype".to_string(),
                                ));
                            }
                        }
                        found = true;
                        break;
                    }
                    axis_idx -= t_axis_dim;
                }
                if !found {
                    return Err(Error::ShapeMismatch(
                        "CONCAT: internal indexing error".to_string(),
                    ));
                }
            }
            TensorData::$variant(result)
        }};
    }

    let first = state.get_tensor(&op.inputs[0])?;
    let data = match &first.data {
        TensorData::F32(_) => concat_typed!(F32),
        TensorData::I32(_) => concat_typed!(I32),
        TensorData::I8(_) => concat_typed!(I8),
        TensorData::I16(_) => concat_typed!(I16),
        TensorData::I64(_) => concat_typed!(I64),
        TensorData::Bool(_) => concat_typed!(Bool),
        TensorData::I48(_) => concat_typed!(I48),
        TensorData::F16(_) => concat_typed!(F16),
        TensorData::Bf16(_) => concat_typed!(Bf16),
        TensorData::F64(_) => concat_typed!(F64),
        TensorData::I4(_) => concat_typed!(I4),
        TensorData::Fp8E4M3(_) => concat_typed!(Fp8E4M3),
        TensorData::Fp8E5M2(_) => concat_typed!(Fp8E5M2),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// SLICE: extract a sub-tensor. Start indices and sizes come from input[1] and input[2]
/// (SHAPE-type tensors containing i32 values).
pub fn eval_slice(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let start_tensor = state.get_tensor(&op.inputs[1])?;
    let size_tensor = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    let starts: Vec<usize> = start_tensor
        .data
        .as_i32_slice()
        .ok_or_else(|| Error::DTypeMismatch("SLICE: start tensor must be I32/SHAPE".to_string()))?
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();

    let sizes: Vec<usize> = size_tensor
        .data
        .as_i32_slice()
        .ok_or_else(|| Error::DTypeMismatch("SLICE: size tensor must be I32/SHAPE".to_string()))?
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();

    let rank = input.shape.len();
    if starts.len() != rank || sizes.len() != rank {
        return Err(Error::ShapeMismatch(format!(
            "SLICE: start/size length ({}/{}) must match input rank {}",
            starts.len(),
            sizes.len(),
            rank
        )));
    }

    // Validate bounds
    for d in 0..rank {
        if starts[d] + sizes[d] > input.shape[d] {
            return Err(Error::ShapeMismatch(format!(
                "SLICE: start[{}]={} + size[{}]={} exceeds input dim {}",
                d, starts[d], d, sizes[d], input.shape[d]
            )));
        }
    }

    let out_shape = sizes.clone();
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! slice_typed {
        ($variant:ident, $v:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for out_flat in 0..out_count {
                let out_nd = flat_to_nd(out_flat, &out_shape);
                let in_nd: Vec<usize> = out_nd.iter().enumerate().map(|(d, &i)| i + starts[d]).collect();
                let in_flat = nd_to_flat_shape(&in_nd, &input.shape);
                result.push($v[in_flat]);
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &input.data {
        TensorData::F32(v) => slice_typed!(F32, v),
        TensorData::I32(v) => slice_typed!(I32, v),
        TensorData::I8(v) => slice_typed!(I8, v),
        TensorData::I16(v) => slice_typed!(I16, v),
        TensorData::I64(v) => slice_typed!(I64, v),
        TensorData::Bool(v) => slice_typed!(Bool, v),
        TensorData::I48(v) => slice_typed!(I48, v),
        TensorData::F16(v) => slice_typed!(F16, v),
        TensorData::Bf16(v) => slice_typed!(Bf16, v),
        TensorData::F64(v) => slice_typed!(F64, v),
        TensorData::I4(v) => slice_typed!(I4, v),
        TensorData::Fp8E4M3(v) => slice_typed!(Fp8E4M3, v),
        TensorData::Fp8E5M2(v) => slice_typed!(Fp8E5M2, v),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// PAD: zero-pad a tensor. Padding values come from input[1] (SHAPE tensor with layout
/// [before0, after0, before1, after1, ...]).
pub fn eval_pad(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let pad_tensor = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let padding: Vec<usize> = match &pad_tensor.data {
        TensorData::I32(v) => v.iter().map(|&x| x.max(0) as usize).collect(),
        TensorData::I64(v) => v.iter().map(|&x| x.max(0) as usize).collect(),
        _ => return Err(Error::DTypeMismatch("PAD: padding tensor must be I32/I64/SHAPE".to_string())),
    };

    let rank = input.shape.len();
    if padding.len() != rank * 2 {
        return Err(Error::ShapeMismatch(format!(
            "PAD: padding length {} must be 2 * rank {}",
            padding.len(),
            rank
        )));
    }

    // Compute output shape
    let mut out_shape = Vec::with_capacity(rank);
    for d in 0..rank {
        let before = padding[d * 2];
        let after = padding[d * 2 + 1];
        out_shape.push(input.shape[d] + before + after);
    }

    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! pad_typed {
        ($variant:ident, $v:expr, $zero:expr) => {{
            let mut result = vec![$zero; out_count];
            for out_flat in 0..out_count {
                let out_nd = flat_to_nd(out_flat, &out_shape);
                let mut in_bounds = true;
                let mut in_nd = Vec::with_capacity(rank);
                for d in 0..rank {
                    let before = padding[d * 2];
                    if out_nd[d] < before || out_nd[d] >= before + input.shape[d] {
                        in_bounds = false;
                        break;
                    }
                    in_nd.push(out_nd[d] - before);
                }
                if in_bounds {
                    let in_flat = nd_to_flat_shape(&in_nd, &input.shape);
                    result[out_flat] = $v[in_flat];
                }
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &input.data {
        TensorData::F32(v) => pad_typed!(F32, v, 0.0f32),
        TensorData::I32(v) => pad_typed!(I32, v, 0i32),
        TensorData::I8(v) => pad_typed!(I8, v, 0i8),
        TensorData::I16(v) => pad_typed!(I16, v, 0i16),
        TensorData::I64(v) => pad_typed!(I64, v, 0i64),
        TensorData::Bool(v) => pad_typed!(Bool, v, false),
        TensorData::I48(v) => pad_typed!(I48, v, 0i64),
        TensorData::F16(v) => pad_typed!(F16, v, half::f16::ZERO),
        TensorData::Bf16(v) => pad_typed!(Bf16, v, half::bf16::ZERO),
        TensorData::F64(v) => pad_typed!(F64, v, 0.0f64),
        TensorData::I4(v) => pad_typed!(I4, v, 0i8),
        TensorData::Fp8E4M3(v) => pad_typed!(Fp8E4M3, v, crate::float8::Fp8E4M3::ZERO),
        TensorData::Fp8E5M2(v) => pad_typed!(Fp8E5M2, v, crate::float8::Fp8E5M2::ZERO),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// TILE: repeat tensor along each dimension. Multipliers come from input[1] (SHAPE tensor).
pub fn eval_tile(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let multiples_tensor = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    let multiples: Vec<usize> = match &multiples_tensor.data {
        TensorData::I32(v) => v.iter().map(|&x| x.max(1) as usize).collect(),
        TensorData::I64(v) => v.iter().map(|&x| x.max(1) as usize).collect(),
        _ => return Err(Error::DTypeMismatch("TILE: multiples tensor must be I32/I64/SHAPE".to_string())),
    };

    let rank = input.shape.len();
    if multiples.len() != rank {
        return Err(Error::ShapeMismatch(format!(
            "TILE: multiples length {} must match input rank {}",
            multiples.len(),
            rank
        )));
    }

    // Compute output shape
    let out_shape: Vec<usize> = input
        .shape
        .iter()
        .zip(multiples.iter())
        .map(|(&dim, &m)| dim * m)
        .collect();
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! tile_typed {
        ($variant:ident, $v:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for out_flat in 0..out_count {
                let out_nd = flat_to_nd(out_flat, &out_shape);
                // Map output indices back to input indices via modulo
                let in_nd: Vec<usize> = out_nd
                    .iter()
                    .enumerate()
                    .map(|(d, &idx)| idx % input.shape[d])
                    .collect();
                let in_flat = nd_to_flat_shape(&in_nd, &input.shape);
                result.push($v[in_flat]);
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &input.data {
        TensorData::F32(v) => tile_typed!(F32, v),
        TensorData::I32(v) => tile_typed!(I32, v),
        TensorData::I8(v) => tile_typed!(I8, v),
        TensorData::I16(v) => tile_typed!(I16, v),
        TensorData::I64(v) => tile_typed!(I64, v),
        TensorData::Bool(v) => tile_typed!(Bool, v),
        TensorData::I48(v) => tile_typed!(I48, v),
        TensorData::F16(v) => tile_typed!(F16, v),
        TensorData::Bf16(v) => tile_typed!(Bf16, v),
        TensorData::F64(v) => tile_typed!(F64, v),
        TensorData::I4(v) => tile_typed!(I4, v),
        TensorData::Fp8E4M3(v) => tile_typed!(Fp8E4M3, v),
        TensorData::Fp8E5M2(v) => tile_typed!(Fp8E5M2, v),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// REVERSE: reverse elements along the specified axis.
pub fn eval_reverse(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let input = state.get_tensor(&op.inputs[0])?;
    let output_name = &op.outputs[0];

    let axis = match &op.attribute {
        OpAttribute::Reverse { axis } => {
            let a = *axis;
            if a < 0 {
                (input.shape.len() as i32 + a) as usize
            } else {
                a as usize
            }
        }
        _ => {
            return Err(Error::ShapeMismatch(
                "REVERSE: missing Reverse attribute".to_string(),
            ));
        }
    };

    let rank = input.shape.len();
    if axis >= rank {
        return Err(Error::ShapeMismatch(format!(
            "REVERSE: axis {} out of range for rank {}",
            axis, rank
        )));
    }

    let out_shape = input.shape.clone();
    let out_count: usize = out_shape.iter().product::<usize>().max(1);

    macro_rules! reverse_typed {
        ($variant:ident, $v:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for out_flat in 0..out_count {
                let mut nd = flat_to_nd(out_flat, &out_shape);
                // Reverse the index along the specified axis
                nd[axis] = input.shape[axis] - 1 - nd[axis];
                let in_flat = nd_to_flat_shape(&nd, &input.shape);
                result.push($v[in_flat]);
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &input.data {
        TensorData::F32(v) => reverse_typed!(F32, v),
        TensorData::I32(v) => reverse_typed!(I32, v),
        TensorData::I8(v) => reverse_typed!(I8, v),
        TensorData::I16(v) => reverse_typed!(I16, v),
        TensorData::I64(v) => reverse_typed!(I64, v),
        TensorData::Bool(v) => reverse_typed!(Bool, v),
        TensorData::I48(v) => reverse_typed!(I48, v),
        TensorData::F16(v) => reverse_typed!(F16, v),
        TensorData::Bf16(v) => reverse_typed!(Bf16, v),
        TensorData::F64(v) => reverse_typed!(F64, v),
        TensorData::I4(v) => reverse_typed!(I4, v),
        TensorData::Fp8E4M3(v) => reverse_typed!(Fp8E4M3, v),
        TensorData::Fp8E5M2(v) => reverse_typed!(Fp8E5M2, v),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// GATHER: inputs [values(N,K,C), indices(N,W)], output (N,W,C).
/// output[n,w,c] = values[n, indices[n,w], c]
pub fn eval_gather(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let values = state.get_tensor(&op.inputs[0])?;
    let indices = state.get_tensor(&op.inputs[1])?;
    let output_name = &op.outputs[0];

    // values shape: (N, K, C)
    if values.shape.len() != 3 {
        return Err(Error::ShapeMismatch(format!(
            "GATHER: values must be rank 3, got rank {}",
            values.shape.len()
        )));
    }
    // indices shape: (N, W)
    if indices.shape.len() != 2 {
        return Err(Error::ShapeMismatch(format!(
            "GATHER: indices must be rank 2, got rank {}",
            indices.shape.len()
        )));
    }

    let n = values.shape[0];
    let k = values.shape[1];
    let c = values.shape[2];
    let w = indices.shape[1];

    if indices.shape[0] != n {
        return Err(Error::ShapeMismatch(format!(
            "GATHER: indices batch dim {} must match values batch dim {}",
            indices.shape[0],
            n
        )));
    }

    let idx_data = indices
        .data
        .as_i32_slice()
        .ok_or_else(|| Error::DTypeMismatch("GATHER: indices must be I32".to_string()))?;

    let out_shape = vec![n, w, c];
    let out_count = n * w * c;

    macro_rules! gather_typed {
        ($variant:ident, $v:expr) => {{
            let mut result = Vec::with_capacity(out_count);
            for ni in 0..n {
                for wi in 0..w {
                    let idx = idx_data[ni * w + wi] as usize;
                    let idx_clamped = idx.min(k.saturating_sub(1));
                    for ci in 0..c {
                        let in_flat = ni * k * c + idx_clamped * c + ci;
                        result.push($v[in_flat]);
                    }
                }
            }
            TensorData::$variant(result)
        }};
    }

    let data = match &values.data {
        TensorData::F32(v) => gather_typed!(F32, v),
        TensorData::I32(v) => gather_typed!(I32, v),
        TensorData::I8(v) => gather_typed!(I8, v),
        TensorData::I16(v) => gather_typed!(I16, v),
        TensorData::I64(v) => gather_typed!(I64, v),
        TensorData::Bool(v) => gather_typed!(Bool, v),
        TensorData::I48(v) => gather_typed!(I48, v),
        TensorData::F16(v) => gather_typed!(F16, v),
        TensorData::Bf16(v) => gather_typed!(Bf16, v),
        TensorData::F64(v) => gather_typed!(F64, v),
        TensorData::I4(v) => gather_typed!(I4, v),
        TensorData::Fp8E4M3(v) => gather_typed!(Fp8E4M3, v),
        TensorData::Fp8E5M2(v) => gather_typed!(Fp8E5M2, v),
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// SCATTER: inputs [values_in(N,K,C), indices(N,W), input(N,W,C)], output (N,K,C).
/// output = values_in.clone(); output[n, indices[n,w], c] = input[n,w,c]
pub fn eval_scatter(state: &mut InterpreterState, op: &Operator) -> Result<()> {
    let values_in = state.get_tensor(&op.inputs[0])?;
    let indices = state.get_tensor(&op.inputs[1])?;
    let input = state.get_tensor(&op.inputs[2])?;
    let output_name = &op.outputs[0];

    // values_in shape: (N, K, C)
    if values_in.shape.len() != 3 {
        return Err(Error::ShapeMismatch(format!(
            "SCATTER: values_in must be rank 3, got rank {}",
            values_in.shape.len()
        )));
    }
    // indices shape: (N, W)
    if indices.shape.len() != 2 {
        return Err(Error::ShapeMismatch(format!(
            "SCATTER: indices must be rank 2, got rank {}",
            indices.shape.len()
        )));
    }
    // input shape: (N, W, C)
    if input.shape.len() != 3 {
        return Err(Error::ShapeMismatch(format!(
            "SCATTER: input must be rank 3, got rank {}",
            input.shape.len()
        )));
    }

    let n = values_in.shape[0];
    let k = values_in.shape[1];
    let c = values_in.shape[2];
    let w = indices.shape[1];

    if indices.shape[0] != n || input.shape[0] != n {
        return Err(Error::ShapeMismatch(
            "SCATTER: batch dimension mismatch".to_string(),
        ));
    }
    if input.shape[1] != w || input.shape[2] != c {
        return Err(Error::ShapeMismatch(
            "SCATTER: input shape must be (N,W,C)".to_string(),
        ));
    }

    let idx_data = indices
        .data
        .as_i32_slice()
        .ok_or_else(|| Error::DTypeMismatch("SCATTER: indices must be I32".to_string()))?;

    let out_shape = vec![n, k, c];

    macro_rules! scatter_typed {
        ($variant:ident, $values_v:expr, $input_v:expr) => {{
            let mut result = $values_v.clone();
            for ni in 0..n {
                for wi in 0..w {
                    let idx = idx_data[ni * w + wi] as usize;
                    let idx_clamped = idx.min(k.saturating_sub(1));
                    for ci in 0..c {
                        let out_flat = ni * k * c + idx_clamped * c + ci;
                        let in_flat = ni * w * c + wi * c + ci;
                        result[out_flat] = $input_v[in_flat];
                    }
                }
            }
            TensorData::$variant(result)
        }};
    }

    let data = match (&values_in.data, &input.data) {
        (TensorData::F32(vv), TensorData::F32(iv)) => scatter_typed!(F32, vv, iv),
        (TensorData::I32(vv), TensorData::I32(iv)) => scatter_typed!(I32, vv, iv),
        (TensorData::I8(vv), TensorData::I8(iv)) => scatter_typed!(I8, vv, iv),
        (TensorData::I16(vv), TensorData::I16(iv)) => scatter_typed!(I16, vv, iv),
        (TensorData::I64(vv), TensorData::I64(iv)) => scatter_typed!(I64, vv, iv),
        (TensorData::Bool(vv), TensorData::Bool(iv)) => scatter_typed!(Bool, vv, iv),
        (TensorData::I48(vv), TensorData::I48(iv)) => scatter_typed!(I48, vv, iv),
        (TensorData::F16(vv), TensorData::F16(iv)) => scatter_typed!(F16, vv, iv),
        (TensorData::Bf16(vv), TensorData::Bf16(iv)) => scatter_typed!(Bf16, vv, iv),
        (TensorData::F64(vv), TensorData::F64(iv)) => scatter_typed!(F64, vv, iv),
        _ => {
            return Err(Error::DTypeMismatch(
                "SCATTER: values_in and input must have the same dtype".to_string(),
            ));
        }
    };

    state.set_tensor(Tensor::new(output_name.clone(), out_shape, data));
    Ok(())
}

/// Helper: compute a flat index from N-d indices and a shape.
fn nd_to_flat_shape(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for dim in (0..shape.len()).rev() {
        flat += indices[dim] * stride;
        stride *= shape[dim];
    }
    flat
}
