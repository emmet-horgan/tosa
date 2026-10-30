//! Parsing an ONNX `ModelProto` into a [`SourceGraph`].
//!
//! Tensors come from the graph inputs (types and shapes declared; symbolic
//! dimensions become 1, see [`SYMBOLIC_DIM`]), the initializers (constants),
//! `Constant` nodes (turned into constant tensors) and every node output
//! (type and shape from `value_info` when the model records them; the
//! lowerings otherwise derive shapes from their operands). Nodes keep
//! ONNX's operator names (`Conv`, `Gemm`, ...) and attributes. The default
//! opset version is recorded as the graph's `"opset"` metadata.

use half::f16;
use tosa_ir::TensorData;

use super::proto::{self, AttributeValue, data_type};
use crate::error::{ConvertError, Result};
use crate::source::{AttrValue, Attrs, SourceDType, SourceGraph, SourceNode, SourceTensor, TensorId};

/// Size substituted for symbolic (named) dimensions of graph inputs, such
/// as a `batch_size` of unknown value.
pub const SYMBOLIC_DIM: i64 = 1;

pub(super) fn err(reason: impl Into<String>) -> ConvertError {
    ConvertError::Frontend { format: "onnx".into(), reason: reason.into() }
}

fn dtype_of(t: i64) -> SourceDType {
    match t {
        data_type::FLOAT => SourceDType::F32,
        data_type::UINT8 => SourceDType::U8,
        data_type::INT8 => SourceDType::I8,
        data_type::UINT16 => SourceDType::U16,
        data_type::INT16 => SourceDType::I16,
        data_type::INT32 => SourceDType::I32,
        data_type::INT64 => SourceDType::I64,
        data_type::STRING => SourceDType::Str,
        data_type::BOOL => SourceDType::Bool,
        data_type::FLOAT16 => SourceDType::F16,
        data_type::DOUBLE => SourceDType::F64,
        data_type::UINT32 => SourceDType::U32,
        data_type::UINT64 => SourceDType::U64,
        data_type::BFLOAT16 => SourceDType::Bf16,
        data_type::INT4 => SourceDType::I4,
        other => SourceDType::Other(format!("onnx data type {other}")),
    }
}

/// The constant data of a `TensorProto`, from `raw_data` or the typed fields.
fn tensor_data(t: &proto::Tensor) -> Result<TensorData> {
    if t.external {
        return Err(err(format!("tensor '{}' uses external data, which is not supported", t.name)));
    }
    let count: usize = t.dims.iter().map(|&d| d.max(0) as usize).product();
    let dtype = dtype_of(t.data_type);
    if !t.raw_data.is_empty() {
        return match dtype {
            SourceDType::U8 => Ok(TensorData::I8(t.raw_data.iter().map(|&b| (b as i32 - 128) as i8).collect())),
            SourceDType::F64 => {
                Ok(TensorData::F64(t.raw_data.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect()))
            }
            _ => {
                let tosa = dtype.to_tosa().ok_or_else(|| {
                    err(format!("tensor '{}' has type {dtype:?}, which cannot be represented", t.name))
                })?;
                TensorData::from_raw_bytes(tosa, &t.raw_data, count)
                    .map_err(|e| err(format!("tensor '{}': {e}", t.name)))
            }
        };
    }
    Ok(match dtype {
        SourceDType::F32 => TensorData::F32(t.float_data.clone()),
        SourceDType::F64 => TensorData::F64(t.double_data.clone()),
        SourceDType::I64 => TensorData::I64(t.int64_data.clone()),
        SourceDType::I32 => TensorData::I32(t.int32_data.iter().map(|&v| v as i32).collect()),
        SourceDType::I16 => TensorData::I16(t.int32_data.iter().map(|&v| v as i16).collect()),
        SourceDType::I8 => TensorData::I8(t.int32_data.iter().map(|&v| v as i8).collect()),
        SourceDType::U8 => TensorData::I8(t.int32_data.iter().map(|&v| (v - 128) as i8).collect()),
        SourceDType::Bool => TensorData::Bool(t.int32_data.iter().map(|&v| v != 0).collect()),
        SourceDType::F16 => TensorData::F16(t.int32_data.iter().map(|&v| f16::from_bits(v as u16)).collect()),
        other => return Err(err(format!("tensor '{}' has type {other:?}, which cannot be represented", t.name))),
    })
}

fn attribute_value(a: &AttributeValue) -> Option<AttrValue> {
    Some(match a {
        AttributeValue::Float(v) => AttrValue::Float(*v as f64),
        AttributeValue::Int(v) => AttrValue::Int(*v),
        AttributeValue::String(s) => AttrValue::Str(String::from_utf8_lossy(s).into_owned()),
        AttributeValue::Floats(v) => AttrValue::Floats(v.iter().map(|&f| f as f64).collect()),
        AttributeValue::Ints(v) => AttrValue::Ints(v.clone()),
        AttributeValue::Strings(v) => {
            AttrValue::Str(v.iter().map(|s| String::from_utf8_lossy(s).into_owned()).collect::<Vec<_>>().join(","))
        }
        AttributeValue::Tensor(_) | AttributeValue::Unsupported(_) => return None,
    })
}

pub fn load(bytes: &[u8]) -> Result<SourceGraph> {
    let model = proto::parse(bytes)?;
    let g = &model.graph;
    let opset = model
        .opset_import
        .iter()
        .find(|(domain, _)| domain.is_empty() || domain == "ai.onnx")
        .map(|(_, v)| *v)
        .unwrap_or(0);
    let mut graph = SourceGraph::new(if g.name.is_empty() { "main" } else { &g.name });
    graph.metadata.insert("opset".into(), AttrValue::Int(opset));
    graph.metadata.insert("ir_version".into(), AttrValue::Int(model.ir_version));
    graph.metadata.insert("producer".into(), AttrValue::Str(model.producer_name.clone()));

    let mut ids: std::collections::HashMap<String, TensorId> = std::collections::HashMap::new();
    let tensor =
        |graph: &mut SourceGraph, ids: &mut std::collections::HashMap<String, TensorId>, t: SourceTensor| -> TensorId {
            let name = t.name.clone();
            let id = graph.add_tensor(t);
            ids.insert(name, id);
            id
        };

    // Initializers (constants).
    for t in &g.initializer {
        let dtype = dtype_of(t.data_type);
        let data = tensor_data(t)?;
        let (dtype, original) =
            if dtype == SourceDType::U8 { (SourceDType::I8, Some(SourceDType::U8)) } else { (dtype, None) };
        let mut st = SourceTensor::new(&t.name, dtype, &t.dims).with_constant(data);
        st.original_dtype = original;
        tensor(&mut graph, &mut ids, st);
    }
    // Graph inputs that are not initializers.
    for vi in &g.input {
        if ids.contains_key(&vi.name) {
            continue;
        }
        let dtype = vi.elem_type.map(dtype_of).unwrap_or(SourceDType::F32);
        let shape: Vec<i64> =
            vi.shape.clone().unwrap_or_default().into_iter().map(|d| d.unwrap_or(SYMBOLIC_DIM)).collect();
        let id = tensor(&mut graph, &mut ids, SourceTensor::new(&vi.name, dtype, &shape));
        graph.inputs.push(id);
    }
    // Declared intermediate and output types.
    let mut declared: std::collections::HashMap<&str, &proto::ValueInfo> = std::collections::HashMap::new();
    for vi in g.value_info.iter().chain(&g.output) {
        declared.insert(vi.name.as_str(), vi);
    }

    for (i, n) in g.node.iter().enumerate() {
        let name = if n.name.is_empty() {
            n.output.first().cloned().unwrap_or_else(|| format!("node_{i}"))
        } else {
            n.name.clone()
        };
        if n.op_type == "Constant" && n.domain.is_empty() {
            // A constant tensor rather than an operator.
            let out = n.output.first().ok_or_else(|| err(format!("Constant node '{name}' has no output")))?;
            let attr = n.attribute.iter().find(|a| a.name == "value").map(|a| &a.value);
            let (dtype, shape, data) = match attr {
                Some(AttributeValue::Tensor(t)) => (dtype_of(t.data_type), t.dims.clone(), tensor_data(t)?),
                _ => match n.attribute.first().map(|a| (&a.name[..], &a.value)) {
                    Some(("value_float", AttributeValue::Float(v))) => {
                        (SourceDType::F32, vec![], TensorData::F32(vec![*v]))
                    }
                    Some(("value_int", AttributeValue::Int(v))) => {
                        (SourceDType::I64, vec![], TensorData::I64(vec![*v]))
                    }
                    Some(("value_floats", AttributeValue::Floats(v))) => {
                        (SourceDType::F32, vec![v.len() as i64], TensorData::F32(v.clone()))
                    }
                    Some(("value_ints", AttributeValue::Ints(v))) => {
                        (SourceDType::I64, vec![v.len() as i64], TensorData::I64(v.clone()))
                    }
                    _ => return Err(err(format!("Constant node '{name}' has an unsupported value"))),
                },
            };
            let (dtype, original) =
                if dtype == SourceDType::U8 { (SourceDType::I8, Some(SourceDType::U8)) } else { (dtype, None) };
            let mut st = SourceTensor::new(out, dtype, &shape).with_constant(data);
            st.original_dtype = original;
            tensor(&mut graph, &mut ids, st);
            continue;
        }
        let mut inputs = Vec::with_capacity(n.input.len());
        for input in &n.input {
            if input.is_empty() {
                inputs.push(None);
                continue;
            }
            let id = *ids
                .get(input)
                .ok_or_else(|| err(format!("node '{name}' reads '{input}', which is not defined before it")))?;
            inputs.push(Some(id));
        }
        let mut outputs = Vec::with_capacity(n.output.len());
        for output in &n.output {
            if output.is_empty() {
                // An omitted optional output still occupies its position.
                let id = tensor(
                    &mut graph,
                    &mut ids,
                    SourceTensor::new(&format!("{name}:unused{}", outputs.len()), SourceDType::F32, &[]),
                );
                outputs.push(id);
                continue;
            }
            let (dtype, shape) = match declared.get(output.as_str()) {
                Some(vi) => (
                    vi.elem_type.map(dtype_of).unwrap_or(SourceDType::F32),
                    vi.shape.clone().unwrap_or_default().into_iter().map(|d| d.unwrap_or(-1)).collect::<Vec<i64>>(),
                ),
                None => (SourceDType::F32, Vec::new()),
            };
            let id = tensor(&mut graph, &mut ids, SourceTensor::new(output, dtype, &shape));
            outputs.push(id);
        }
        let mut attrs = Attrs::new();
        for a in &n.attribute {
            if let Some(v) = attribute_value(&a.value) {
                attrs.insert(a.name.clone(), v);
            }
        }
        let op = if n.domain.is_empty() || n.domain == "ai.onnx" {
            n.op_type.clone()
        } else {
            format!("{}.{}", n.domain, n.op_type)
        };
        graph.add_node(SourceNode { op, name, inputs, outputs, attrs });
    }

    for vi in &g.output {
        let id = *ids.get(&vi.name).ok_or_else(|| err(format!("graph output '{}' is never produced", vi.name)))?;
        graph.outputs.push(id);
    }
    Ok(graph)
}
