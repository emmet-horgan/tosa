//! Parsing a `.tflite` flatbuffer into a [`SourceGraph`].
//!
//! Only the main (single) subgraph is supported. Tensors keep TFLite's
//! names, shapes (a dynamic graph input dimension becomes [`SYMBOLIC_DIM`];
//! dynamic intermediate shapes are inferred by [`super::shapes`]),
//! quantization parameters and constant buffers. Nodes carry the
//! builtin operator's name (`CONV_2D`, ...) and the fields of its options
//! table as attributes.
//!
//! TFLite's uint8 tensors are re-expressed as int8 with every value and zero
//! point shifted by -128, which leaves `scale * (q - zero_point)` unchanged.
//! The original type is recorded in [`SourceTensor::original_dtype`]; the
//! graph input and output hooks restore the unsigned representation with
//! RESCALE operators.

use tosa_ir::TensorData;

use super::fb;
use crate::error::{ConvertError, Result};
use crate::source::{AttrValue, Attrs, Quantization, SourceDType, SourceGraph, SourceNode, SourceTensor, TensorId};

/// Size substituted for a dynamic dimension of a graph input, such as an
/// unknown batch size.
pub const SYMBOLIC_DIM: i64 = 1;

pub(super) fn err(reason: impl Into<String>) -> ConvertError {
    ConvertError::Frontend { format: "tflite".into(), reason: reason.into() }
}

fn dtype_of(t: fb::TensorType) -> SourceDType {
    match t {
        fb::TensorType::FLOAT32 => SourceDType::F32,
        fb::TensorType::FLOAT16 => SourceDType::F16,
        fb::TensorType::BFLOAT16 => SourceDType::Bf16,
        fb::TensorType::FLOAT64 => SourceDType::F64,
        fb::TensorType::INT4 => SourceDType::I4,
        fb::TensorType::INT8 => SourceDType::I8,
        fb::TensorType::INT16 => SourceDType::I16,
        fb::TensorType::INT32 => SourceDType::I32,
        fb::TensorType::INT64 => SourceDType::I64,
        fb::TensorType::UINT8 => SourceDType::U8,
        fb::TensorType::UINT16 => SourceDType::U16,
        fb::TensorType::UINT32 => SourceDType::U32,
        fb::TensorType::UINT64 => SourceDType::U64,
        fb::TensorType::BOOL => SourceDType::Bool,
        fb::TensorType::STRING => SourceDType::Str,
        other => SourceDType::Other(other.variant_name().unwrap_or("?").to_string()),
    }
}

/// The builtin operator of an operator code: the newer 32-bit field when it
/// is set, else the deprecated 8-bit one (as TFLite's `GetBuiltinCode`).
fn builtin_code(code: fb::OperatorCode<'_>) -> fb::BuiltinOperator {
    let deprecated = code.deprecated_builtin_code() as i32;
    fb::BuiltinOperator(deprecated.max(code.builtin_code().0))
}

/// Decode a constant buffer as `dtype`; uint8 data becomes shifted int8.
fn decode_constant(dtype: &SourceDType, bytes: &[u8], count: usize, name: &str) -> Result<TensorData> {
    if *dtype == SourceDType::U8 {
        if bytes.len() < count {
            return Err(err(format!("constant '{name}' has {} bytes for {count} elements", bytes.len())));
        }
        return Ok(TensorData::I8(bytes[..count].iter().map(|&b| (b as i32 - 128) as i8).collect()));
    }
    let Some(tosa) = dtype.to_tosa() else {
        return Err(err(format!("constant '{name}' has type {dtype:?}, which cannot be represented")));
    };
    TensorData::from_raw_bytes(tosa, bytes, count).map_err(|e| err(format!("constant '{name}': {e}")))
}

fn activation_name(a: fb::ActivationFunctionType) -> &'static str {
    a.variant_name().unwrap_or("NONE")
}

fn padding_name(p: fb::Padding) -> &'static str {
    p.variant_name().unwrap_or("SAME")
}

/// The attributes of an operator: the fields of its builtin options table.
fn attributes(op: &fb::Operator<'_>) -> Attrs {
    let mut a = Attrs::new();
    let mut int = |k: &str, v: i64| {
        a.insert(k.to_string(), AttrValue::Int(v));
    };
    if let Some(o) = op.builtin_options_as_conv_2_doptions() {
        int("stride_h", o.stride_h() as i64);
        int("stride_w", o.stride_w() as i64);
        int("dilation_h", o.dilation_h_factor().max(1) as i64);
        int("dilation_w", o.dilation_w_factor().max(1) as i64);
        a.insert("padding".into(), AttrValue::Str(padding_name(o.padding()).into()));
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_depthwise_conv_2_doptions() {
        int("stride_h", o.stride_h() as i64);
        int("stride_w", o.stride_w() as i64);
        int("dilation_h", o.dilation_h_factor().max(1) as i64);
        int("dilation_w", o.dilation_w_factor().max(1) as i64);
        int("depth_multiplier", o.depth_multiplier() as i64);
        a.insert("padding".into(), AttrValue::Str(padding_name(o.padding()).into()));
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_pool_2_doptions() {
        int("stride_h", o.stride_h() as i64);
        int("stride_w", o.stride_w() as i64);
        int("filter_height", o.filter_height() as i64);
        int("filter_width", o.filter_width() as i64);
        a.insert("padding".into(), AttrValue::Str(padding_name(o.padding()).into()));
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_fully_connected_options() {
        a.insert("keep_num_dims".into(), AttrValue::Bool(o.keep_num_dims()));
        a.insert(
            "weights_format".into(),
            AttrValue::Str(o.weights_format().variant_name().unwrap_or("DEFAULT").into()),
        );
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_add_options() {
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_sub_options() {
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_mul_options() {
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_concatenation_options() {
        int("axis", o.axis() as i64);
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_reshape_options() {
        if let Some(s) = o.new_shape() {
            a.insert("new_shape".into(), AttrValue::Ints(s.iter().map(|d| d as i64).collect()));
        }
    } else if let Some(o) = op.builtin_options_as_softmax_options() {
        a.insert("beta".into(), AttrValue::Float(o.beta() as f64));
    } else if let Some(o) = op.builtin_options_as_reducer_options() {
        a.insert("keep_dims".into(), AttrValue::Bool(o.keep_dims()));
    } else if let Some(o) = op.builtin_options_as_squeeze_options() {
        if let Some(d) = o.squeeze_dims() {
            a.insert("squeeze_dims".into(), AttrValue::Ints(d.iter().map(|d| d as i64).collect()));
        }
    } else if let Some(o) = op.builtin_options_as_leaky_relu_options() {
        a.insert("alpha".into(), AttrValue::Float(o.alpha() as f64));
    } else if let Some(o) = op.builtin_options_as_resize_bilinear_options() {
        a.insert("align_corners".into(), AttrValue::Bool(o.align_corners()));
        a.insert("half_pixel_centers".into(), AttrValue::Bool(o.half_pixel_centers()));
    } else if let Some(o) = op.builtin_options_as_resize_nearest_neighbor_options() {
        a.insert("align_corners".into(), AttrValue::Bool(o.align_corners()));
        a.insert("half_pixel_centers".into(), AttrValue::Bool(o.half_pixel_centers()));
    } else if let Some(o) = op.builtin_options_as_transpose_conv_options() {
        int("stride_h", o.stride_h() as i64);
        int("stride_w", o.stride_w() as i64);
        a.insert("padding".into(), AttrValue::Str(padding_name(o.padding()).into()));
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_strided_slice_options() {
        int("begin_mask", o.begin_mask() as i64);
        int("end_mask", o.end_mask() as i64);
        int("ellipsis_mask", o.ellipsis_mask() as i64);
        int("new_axis_mask", o.new_axis_mask() as i64);
        int("shrink_axis_mask", o.shrink_axis_mask() as i64);
    } else if let Some(o) = op.builtin_options_as_split_options() {
        int("num_splits", o.num_splits() as i64);
    } else if let Some(o) = op.builtin_options_as_split_voptions() {
        int("num_splits", o.num_splits() as i64);
    } else if let Some(o) = op.builtin_options_as_depth_to_space_options() {
        int("block_size", o.block_size() as i64);
    } else if let Some(o) = op.builtin_options_as_space_to_depth_options() {
        int("block_size", o.block_size() as i64);
    } else if let Some(o) = op.builtin_options_as_pack_options() {
        int("values_count", o.values_count() as i64);
        int("axis", o.axis() as i64);
    } else if let Some(o) = op.builtin_options_as_unpack_options() {
        int("num", o.num() as i64);
        int("axis", o.axis() as i64);
    } else if let Some(o) = op.builtin_options_as_gather_options() {
        int("axis", o.axis() as i64);
        int("batch_dims", o.batch_dims() as i64);
    } else if let Some(o) = op.builtin_options_as_batch_mat_mul_options() {
        a.insert("adj_x".into(), AttrValue::Bool(o.adj_x()));
        a.insert("adj_y".into(), AttrValue::Bool(o.adj_y()));
    } else if let Some(o) = op.builtin_options_as_div_options() {
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_l2_norm_options() {
        a.insert(
            "fused_activation_function".into(),
            AttrValue::Str(activation_name(o.fused_activation_function()).into()),
        );
    } else if let Some(o) = op.builtin_options_as_gelu_options() {
        a.insert("approximate".into(), AttrValue::Bool(o.approximate()));
    }
    a
}

pub fn load(bytes: &[u8]) -> Result<SourceGraph> {
    let model = fb::root_as_model(bytes).map_err(|e| err(format!("not a TFLite flatbuffer: {e}")))?;
    let subgraphs = model.subgraphs().ok_or_else(|| err("the model has no subgraphs"))?;
    if subgraphs.len() != 1 {
        return Err(err(format!("{} subgraphs; only single-subgraph models are supported", subgraphs.len())));
    }
    let sg = subgraphs.get(0);
    let buffers = model.buffers().ok_or_else(|| err("the model has no buffers"))?;
    let codes = model.operator_codes().ok_or_else(|| err("the model has no operator codes"))?;

    let mut graph = SourceGraph::new(sg.name().filter(|n| !n.is_empty()).unwrap_or("main"));
    let graph_inputs: std::collections::HashSet<usize> =
        sg.inputs().map(|v| v.iter().map(|x| x as usize).collect()).unwrap_or_default();
    let mut names = std::collections::HashSet::new();
    for (i, t) in sg.tensors().into_iter().flatten().enumerate() {
        let mut name = t.name().filter(|n| !n.is_empty()).map(str::to_string).unwrap_or_else(|| format!("tensor_{i}"));
        while !names.insert(name.clone()) {
            name = format!("{name}_{i}");
        }
        let dtype = dtype_of(t.type_());
        // `shape_signature` marks dynamic dimensions with -1; `shape` then
        // holds whatever the converter allocated, which for intermediate
        // tensors is not reliable. Dynamic shapes are inferred afterwards
        // (see [`super::shapes`]); a dynamic graph input dimension (an
        // unknown batch) is taken as [`SYMBOLIC_DIM`].
        let concrete: Vec<i64> = t.shape().map(|s| s.iter().map(|d| d as i64).collect()).unwrap_or_default();
        let signature: Option<Vec<i64>> = t.shape_signature().map(|s| s.iter().map(|d| d as i64).collect());
        let mut shape: Vec<i64> = match signature {
            Some(sig) if sig.len() == concrete.len() => sig,
            _ => concrete,
        };
        if graph_inputs.contains(&i) {
            for d in &mut shape {
                if *d < 0 {
                    *d = SYMBOLIC_DIM;
                }
            }
        }
        let quant = t.quantization().and_then(|q| {
            let scales: Vec<f64> = q.scale()?.iter().map(|s| s as f64).collect();
            if scales.is_empty() {
                return None;
            }
            let mut zero_points: Vec<i64> = q.zero_point().map(|z| z.iter().collect()).unwrap_or_default();
            if zero_points.is_empty() {
                zero_points = vec![0; scales.len()];
            }
            if dtype == SourceDType::U8 {
                for z in &mut zero_points {
                    *z -= 128;
                }
            }
            let axis = (scales.len() > 1).then_some(q.quantized_dimension() as usize);
            Some(Quantization { scales, zero_points, axis })
        });
        let constant = match buffers.get(t.buffer() as usize).data() {
            Some(data) if !data.is_empty() => {
                if shape.iter().any(|&d| d < 0) {
                    return Err(err(format!("constant '{name}' has a dynamic shape {shape:?}")));
                }
                let count = shape.iter().map(|&d| d as usize).product();
                Some(decode_constant(&dtype, data.bytes(), count, &name)?)
            }
            _ => None,
        };
        let (dtype, original_dtype) =
            if dtype == SourceDType::U8 { (SourceDType::I8, Some(SourceDType::U8)) } else { (dtype, None) };
        graph.add_tensor(SourceTensor { name, dtype, shape, quant, constant, original_dtype });
    }

    for (i, op) in sg.operators().into_iter().flatten().enumerate() {
        let code = codes.get(op.opcode_index() as usize);
        let builtin = builtin_code(code);
        let op_name = match builtin.variant_name() {
            Some("CUSTOM") => format!("CUSTOM:{}", code.custom_code().unwrap_or("?")),
            Some(n) => n.to_string(),
            None => format!("BUILTIN_{}", builtin.0),
        };
        let to_ids = |v: Option<flatbuffers::Vector<'_, i32>>| -> Vec<Option<TensorId>> {
            v.map(|v| v.iter().map(|x| (x >= 0).then_some(x as TensorId)).collect()).unwrap_or_default()
        };
        let inputs = to_ids(op.inputs());
        let outputs: Vec<TensorId> = to_ids(op.outputs()).into_iter().flatten().collect();
        for &id in inputs.iter().flatten().chain(&outputs) {
            if id >= graph.tensors.len() {
                return Err(err(format!("operator {i} ({op_name}) refers to tensor {id}, which does not exist")));
            }
        }
        let name = outputs.first().map(|&o| graph.tensors[o].name.clone()).unwrap_or_else(|| format!("{op_name}_{i}"));
        let attrs = attributes(&op);
        graph.add_node(SourceNode { op: op_name, name, inputs, outputs, attrs });
    }

    let ids = |v: Option<flatbuffers::Vector<'_, i32>>| -> Vec<TensorId> {
        v.map(|v| v.iter().map(|x| x as TensorId).collect()).unwrap_or_default()
    };
    graph.inputs = ids(sg.inputs());
    graph.outputs = ids(sg.outputs());
    for &id in graph.inputs.iter().chain(&graph.outputs) {
        if id >= graph.tensors.len() {
            return Err(err(format!("graph input/output tensor {id} does not exist")));
        }
    }
    super::shapes::infer_shapes(&mut graph)?;
    Ok(graph)
}
