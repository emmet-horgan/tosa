//! Shape inference for TFLite graphs.
//!
//! Models exported with dynamic dimensions (a `-1` batch or spatial size in
//! `shape_signature`) do not record usable shapes for their intermediate
//! tensors: the file holds placeholders that LiteRT recomputes when it
//! allocates tensors. Conversion needs static shapes, so this pass fills in
//! every dynamic output shape from the operator's inputs and options,
//! following TFLite's own shape rules. Static shapes recorded in the file are
//! kept as they are.

use tosa_ir::TensorData;

use super::loader::err;
use crate::error::Result;
use crate::source::{SourceGraph, SourceNode, TensorId};

type Shape = Vec<i64>;
type Infer<T> = std::result::Result<T, String>;

/// Replace the dynamic output shapes of `graph`'s nodes with inferred ones.
pub(super) fn infer_shapes(graph: &mut SourceGraph) -> Result<()> {
    for i in 0..graph.nodes.len() {
        if graph.nodes[i].outputs.iter().all(|&o| graph.tensors[o].is_static()) {
            continue;
        }
        let node = &graph.nodes[i];
        let shapes = infer_node(graph, node).map_err(|reason| {
            err(format!("cannot infer the output shape of {} node '{}': {reason}", node.op, node.name))
        })?;
        if shapes.len() < node.outputs.len() {
            return Err(err(format!(
                "cannot infer the output shape of {} node '{}': {} of {} outputs inferred",
                node.op,
                node.name,
                shapes.len(),
                node.outputs.len()
            )));
        }
        let outputs = node.outputs.clone();
        for (k, &o) in outputs.iter().enumerate() {
            let t = &mut graph.tensors[o];
            if !t.is_static() {
                t.shape = shapes[k].clone();
            }
        }
    }
    Ok(())
}

fn input_shape(graph: &SourceGraph, node: &SourceNode, i: usize) -> Infer<Shape> {
    let id = node.inputs.get(i).copied().flatten().ok_or_else(|| format!("input {i} is missing"))?;
    let t = graph.tensor(id);
    if t.is_static() { Ok(t.shape.clone()) } else { Err(format!("input '{}' has a dynamic shape", t.name)) }
}

fn constant_ints(graph: &SourceGraph, node: &SourceNode, i: usize) -> Infer<Vec<i64>> {
    let id: TensorId = node.inputs.get(i).copied().flatten().ok_or_else(|| format!("input {i} is missing"))?;
    let t = graph.tensor(id);
    let data = t.constant.as_ref().ok_or_else(|| format!("operand '{}' must be a constant", t.name))?;
    Ok(match data {
        TensorData::I8(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I16(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I32(v) => v.iter().map(|&x| x as i64).collect(),
        TensorData::I64(v) => v.clone(),
        _ => return Err(format!("operand '{}' must be an integer constant", t.name)),
    })
}

fn int_attr(node: &SourceNode, key: &str, default: i64) -> i64 {
    node.attr_int_or(key, default).unwrap_or(default)
}

fn normalize_axis(axis: i64, rank: usize) -> Infer<usize> {
    let a = if axis < 0 { axis + rank as i64 } else { axis };
    if a < 0 || a >= rank as i64 {
        return Err(format!("axis {axis} is out of range for rank {rank}"));
    }
    Ok(a as usize)
}

/// NumPy-style broadcast of two shapes.
fn broadcast(a: &[i64], b: &[i64]) -> Infer<Shape> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = if i + a.len() >= rank { a[i + a.len() - rank] } else { 1 };
        let db = if i + b.len() >= rank { b[i + b.len() - rank] } else { 1 };
        out.push(match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return Err(format!("shapes {a:?} and {b:?} do not broadcast")),
        });
    }
    Ok(out)
}

/// Output extent of a window of effective size `k` sliding with stride `s`
/// over `size` samples (TFLite's `ComputeOutSize`).
fn windowed(size: i64, k: i64, s: i64, dilation: i64, same: bool) -> Infer<i64> {
    if s <= 0 {
        return Err("stride must be positive".into());
    }
    let effective = (k - 1) * dilation + 1;
    let out = if same { (size + s - 1) / s } else { (size + s - effective) / s };
    if out <= 0 {
        return Err(format!("window {effective} with stride {s} does not fit in {size}"));
    }
    Ok(out)
}

fn spatial(node: &SourceNode, input: &[i64], kernel: [i64; 2], dilation: [i64; 2]) -> Infer<[i64; 2]> {
    if input.len() != 4 {
        return Err(format!("input must be 4-D, got {input:?}"));
    }
    let same = match node.attr_str("padding") {
        Ok("SAME") => true,
        Ok("VALID") => false,
        Ok(other) => return Err(format!("padding mode {other} is not supported")),
        Err(_) => return Err("padding mode is missing".into()),
    };
    let stride = [int_attr(node, "stride_h", 1), int_attr(node, "stride_w", 1)];
    Ok([
        windowed(input[1], kernel[0], stride[0], dilation[0], same)?,
        windowed(input[2], kernel[1], stride[1], dilation[1], same)?,
    ])
}

const SAME_AS_INPUT: &[&str] = &[
    "ABS",
    "CAST",
    "CEIL",
    "COS",
    "DEQUANTIZE",
    "ELU",
    "EXP",
    "FLOOR",
    "GELU",
    "HARD_SWISH",
    "L2_NORMALIZATION",
    "LEAKY_RELU",
    "LOG",
    "LOGICAL_NOT",
    "LOGISTIC",
    "LOG_SOFTMAX",
    "NEG",
    "QUANTIZE",
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "RELU_N1_TO_1",
    "ROUND",
    "RSQRT",
    "SIN",
    "SOFTMAX",
    "SQRT",
    "SQUARE",
    "TANH",
    "MIRROR_PAD_IDENTITY",
];

const BROADCAST_BINARY: &[&str] = &[
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "MAXIMUM",
    "MINIMUM",
    "POW",
    "SQUARED_DIFFERENCE",
    "PRELU",
    "EQUAL",
    "NOT_EQUAL",
    "GREATER",
    "GREATER_EQUAL",
    "LESS",
    "LESS_EQUAL",
    "LOGICAL_AND",
    "LOGICAL_OR",
    "FLOOR_DIV",
    "FLOOR_MOD",
];

fn infer_node(g: &SourceGraph, node: &SourceNode) -> Infer<Vec<Shape>> {
    let op = node.op.as_str();
    if SAME_AS_INPUT.contains(&op) {
        return Ok(vec![input_shape(g, node, 0)?]);
    }
    if BROADCAST_BINARY.contains(&op) {
        return Ok(vec![broadcast(&input_shape(g, node, 0)?, &input_shape(g, node, 1)?)?]);
    }
    let one = |s: Shape| Ok(vec![s]);
    match op {
        "CONV_2D" => {
            let x = input_shape(g, node, 0)?;
            let w = input_shape(g, node, 1)?;
            if w.len() != 4 {
                return Err(format!("weights must be 4-D, got {w:?}"));
            }
            let dilation = [int_attr(node, "dilation_h", 1), int_attr(node, "dilation_w", 1)];
            let [h, wd] = spatial(node, &x, [w[1], w[2]], dilation)?;
            one(vec![x[0], h, wd, w[0]])
        }
        "DEPTHWISE_CONV_2D" => {
            let x = input_shape(g, node, 0)?;
            let w = input_shape(g, node, 1)?;
            if w.len() != 4 {
                return Err(format!("weights must be 4-D, got {w:?}"));
            }
            let dilation = [int_attr(node, "dilation_h", 1), int_attr(node, "dilation_w", 1)];
            let [h, wd] = spatial(node, &x, [w[1], w[2]], dilation)?;
            one(vec![x[0], h, wd, w[3]])
        }
        "TRANSPOSE_CONV" => one(constant_ints(g, node, 0)?),
        "AVERAGE_POOL_2D" | "MAX_POOL_2D" | "L2_POOL_2D" => {
            let x = input_shape(g, node, 0)?;
            let kernel = [int_attr(node, "filter_height", 1), int_attr(node, "filter_width", 1)];
            let [h, wd] = spatial(node, &x, kernel, [1, 1])?;
            one(vec![x[0], h, wd, x[3]])
        }
        "FULLY_CONNECTED" => {
            let x = input_shape(g, node, 0)?;
            let w = input_shape(g, node, 1)?;
            if w.len() != 2 {
                return Err(format!("weights must be 2-D, got {w:?}"));
            }
            let (out_channels, k) = (w[0], w[1]);
            if node.attr_bool("keep_num_dims").unwrap_or(false) {
                let mut s = x.clone();
                *s.last_mut().ok_or("input must have a feature dimension")? = out_channels;
                return one(s);
            }
            let total: i64 = x.iter().product();
            if k == 0 || total % k != 0 {
                return Err(format!("input {x:?} is not a multiple of {k} features"));
            }
            one(vec![total / k, out_channels])
        }
        "RESHAPE" => {
            let x = input_shape(g, node, 0)?;
            let mut shape: Shape = match node.inputs.get(1).copied().flatten() {
                Some(id) if g.tensor(id).is_constant() => constant_ints(g, node, 1)?,
                _ => node.attr_ints("new_shape").map_err(|_| "RESHAPE needs a constant shape".to_string())?.to_vec(),
            };
            let total: i64 = x.iter().product();
            if let Some(i) = shape.iter().position(|&d| d == -1) {
                let known: i64 = shape.iter().filter(|&&d| d != -1).product();
                if known == 0 || total % known != 0 {
                    return Err(format!("cannot infer the -1 dimension of {shape:?} from {total} elements"));
                }
                shape[i] = total / known;
            }
            one(shape)
        }
        "SQUEEZE" => {
            let x = input_shape(g, node, 0)?;
            let dims: Vec<usize> = match node.attr_ints("squeeze_dims") {
                Ok(d) if !d.is_empty() => d.iter().map(|&a| normalize_axis(a, x.len())).collect::<Infer<_>>()?,
                _ => (0..x.len()).filter(|&i| x[i] == 1).collect(),
            };
            one((0..x.len()).filter(|i| !dims.contains(i)).map(|i| x[i]).collect())
        }
        "EXPAND_DIMS" => {
            let mut x = input_shape(g, node, 0)?;
            let axis = constant_ints(g, node, 1)?.first().copied().ok_or("axis operand is empty")?;
            let axis = normalize_axis(axis, x.len() + 1)?;
            x.insert(axis, 1);
            one(x)
        }
        "PAD" | "PADV2" | "MIRROR_PAD" => {
            let x = input_shape(g, node, 0)?;
            let pads = constant_ints(g, node, 1)?;
            if pads.len() != 2 * x.len() {
                return Err(format!("paddings has {} values for rank {}", pads.len(), x.len()));
            }
            one(x.iter().enumerate().map(|(i, &d)| d + pads[2 * i] + pads[2 * i + 1]).collect())
        }
        "MEAN" | "SUM" | "REDUCE_MAX" | "REDUCE_MIN" | "REDUCE_PROD" | "REDUCE_ANY" | "REDUCE_ALL" => {
            let x = input_shape(g, node, 0)?;
            let mut axes: Vec<usize> =
                constant_ints(g, node, 1)?.iter().map(|&a| normalize_axis(a, x.len())).collect::<Infer<_>>()?;
            axes.sort_unstable();
            axes.dedup();
            let keep = node.attr_bool("keep_dims").unwrap_or(false);
            one(if keep {
                (0..x.len()).map(|i| if axes.contains(&i) { 1 } else { x[i] }).collect()
            } else {
                (0..x.len()).filter(|i| !axes.contains(i)).map(|i| x[i]).collect()
            })
        }
        "ARG_MAX" | "ARG_MIN" => {
            let x = input_shape(g, node, 0)?;
            let axis = constant_ints(g, node, 1)?.first().copied().unwrap_or(0);
            let axis = normalize_axis(axis, x.len())?;
            one((0..x.len()).filter(|&i| i != axis).map(|i| x[i]).collect())
        }
        "CONCATENATION" => {
            let shapes: Vec<Shape> = (0..node.inputs.len()).map(|i| input_shape(g, node, i)).collect::<Infer<_>>()?;
            let first = shapes.first().ok_or("CONCATENATION has no inputs")?;
            let axis = normalize_axis(int_attr(node, "axis", 0), first.len())?;
            let mut out = first.clone();
            out[axis] = shapes.iter().map(|s| s.get(axis).copied().unwrap_or(0)).sum();
            one(out)
        }
        "PACK" => {
            let x = input_shape(g, node, 0)?;
            let axis = normalize_axis(int_attr(node, "axis", 0), x.len() + 1)?;
            let mut out = x;
            out.insert(axis, node.inputs.len() as i64);
            one(out)
        }
        "UNPACK" => {
            let x = input_shape(g, node, 0)?;
            let axis = normalize_axis(int_attr(node, "axis", 0), x.len())?;
            let piece: Shape = (0..x.len()).filter(|&i| i != axis).map(|i| x[i]).collect();
            Ok(vec![piece; node.outputs.len()])
        }
        "SPLIT" => {
            let x = input_shape(g, node, 1)?;
            let axis = normalize_axis(constant_ints(g, node, 0)?.first().copied().unwrap_or(0), x.len())?;
            let n = node.outputs.len() as i64;
            if n == 0 || x[axis] % n != 0 {
                return Err(format!("cannot split {} into {n} equal parts", x[axis]));
            }
            let mut piece = x.clone();
            piece[axis] /= n;
            Ok(vec![piece; node.outputs.len()])
        }
        "SPLIT_V" => {
            let x = input_shape(g, node, 0)?;
            let mut sizes = constant_ints(g, node, 1)?;
            let axis = normalize_axis(constant_ints(g, node, 2)?.first().copied().unwrap_or(0), x.len())?;
            if let Some(i) = sizes.iter().position(|&s| s == -1) {
                let known: i64 = sizes.iter().filter(|&&s| s != -1).sum();
                sizes[i] = x[axis] - known;
            }
            Ok(sizes
                .iter()
                .map(|&s| {
                    let mut piece = x.clone();
                    piece[axis] = s;
                    piece
                })
                .collect())
        }
        "RESIZE_BILINEAR" | "RESIZE_NEAREST_NEIGHBOR" => {
            let x = input_shape(g, node, 0)?;
            let size = constant_ints(g, node, 1)?;
            if x.len() != 4 || size.len() != 2 {
                return Err("RESIZE needs a 4-D input and a [height, width] size".into());
            }
            one(vec![x[0], size[0], size[1], x[3]])
        }
        "TRANSPOSE" => {
            let x = input_shape(g, node, 0)?;
            let perm = constant_ints(g, node, 1)?;
            if perm.len() != x.len() {
                return Err(format!("perm {perm:?} does not match rank {}", x.len()));
            }
            perm.iter()
                .map(|&p| x.get(p as usize).copied().ok_or_else(|| format!("perm {perm:?} is invalid")))
                .collect::<Infer<Shape>>()
                .map(|s| vec![s])
        }
        "SLICE" => {
            let x = input_shape(g, node, 0)?;
            let begin = constant_ints(g, node, 1)?;
            let size = constant_ints(g, node, 2)?;
            if begin.len() != x.len() || size.len() != x.len() {
                return Err("SLICE begin/size do not match the input rank".into());
            }
            one((0..x.len()).map(|i| if size[i] == -1 { x[i] - begin[i] } else { size[i] }).collect())
        }
        "STRIDED_SLICE" => strided_slice(g, node),
        "TILE" => {
            let x = input_shape(g, node, 0)?;
            let m = constant_ints(g, node, 1)?;
            if m.len() != x.len() {
                return Err("TILE multiples do not match the input rank".into());
            }
            one(x.iter().zip(&m).map(|(&d, &k)| d * k).collect())
        }
        "GATHER" => {
            let params = input_shape(g, node, 0)?;
            let indices = input_shape(g, node, 1)?;
            let axis = normalize_axis(int_attr(node, "axis", 0), params.len())?;
            let batch_dims = int_attr(node, "batch_dims", 0) as usize;
            let mut out = params[..axis].to_vec();
            out.extend_from_slice(&indices[batch_dims.min(indices.len())..]);
            out.extend_from_slice(&params[axis + 1..]);
            one(out)
        }
        "DEPTH_TO_SPACE" => {
            let x = input_shape(g, node, 0)?;
            let b = int_attr(node, "block_size", 1);
            if x.len() != 4 || b <= 0 || x[3] % (b * b) != 0 {
                return Err("DEPTH_TO_SPACE needs a 4-D input with channels divisible by block_size^2".into());
            }
            one(vec![x[0], x[1] * b, x[2] * b, x[3] / (b * b)])
        }
        "SPACE_TO_DEPTH" => {
            let x = input_shape(g, node, 0)?;
            let b = int_attr(node, "block_size", 1);
            if x.len() != 4 || b <= 0 || x[1] % b != 0 || x[2] % b != 0 {
                return Err("SPACE_TO_DEPTH needs a 4-D input with spatial size divisible by block_size".into());
            }
            one(vec![x[0], x[1] / b, x[2] / b, x[3] * b * b])
        }
        "SELECT" | "SELECT_V2" => {
            let c = input_shape(g, node, 0)?;
            let a = input_shape(g, node, 1)?;
            let b = input_shape(g, node, 2)?;
            one(broadcast(&broadcast(&c, &a)?, &b)?)
        }
        "BATCH_MATMUL" => {
            let a = input_shape(g, node, 0)?;
            let b = input_shape(g, node, 1)?;
            if a.len() < 2 || b.len() < 2 {
                return Err("BATCH_MATMUL operands must be at least 2-D".into());
            }
            let adj_x = node.attr_bool("adj_x").unwrap_or(false);
            let adj_y = node.attr_bool("adj_y").unwrap_or(false);
            let m = if adj_x { a[a.len() - 1] } else { a[a.len() - 2] };
            let n = if adj_y { b[b.len() - 2] } else { b[b.len() - 1] };
            let mut out = broadcast(&a[..a.len() - 2], &b[..b.len() - 2])?;
            out.push(m);
            out.push(n);
            one(out)
        }
        "SHAPE" => {
            let x = input_shape(g, node, 0)?;
            one(vec![x.len() as i64])
        }
        other => Err(format!("no shape rule for {other}")),
    }
}

fn strided_slice(g: &SourceGraph, node: &SourceNode) -> Infer<Vec<Shape>> {
    let x = input_shape(g, node, 0)?;
    let begin = constant_ints(g, node, 1)?;
    let end = constant_ints(g, node, 2)?;
    let strides = constant_ints(g, node, 3)?;
    if int_attr(node, "ellipsis_mask", 0) != 0 || int_attr(node, "new_axis_mask", 0) != 0 {
        return Err("ellipsis_mask and new_axis_mask are not supported".into());
    }
    let begin_mask = int_attr(node, "begin_mask", 0);
    let end_mask = int_attr(node, "end_mask", 0);
    let shrink_mask = int_attr(node, "shrink_axis_mask", 0);
    let n = begin.len().min(end.len()).min(strides.len());
    let mut out = Vec::with_capacity(x.len());
    for (i, &dim) in x.iter().enumerate() {
        if i >= n {
            out.push(dim);
            continue;
        }
        let stride = strides[i];
        if stride == 0 {
            return Err("stride must not be zero".into());
        }
        let clamp = |v: i64, lo: i64, hi: i64| v.max(lo).min(hi);
        let wrap = |v: i64| if v < 0 { v + dim } else { v };
        let (lo, hi) = if stride > 0 { (0, dim) } else { (-1, dim - 1) };
        let mut b =
            if begin_mask >> i & 1 == 1 { if stride > 0 { 0 } else { dim - 1 } } else { clamp(wrap(begin[i]), lo, hi) };
        if shrink_mask >> i & 1 == 1 {
            b = clamp(wrap(begin[i]), 0, dim - 1);
            // The axis is removed from the output.
            continue;
        }
        let e = if end_mask >> i & 1 == 1 { if stride > 0 { dim } else { -1 } } else { clamp(wrap(end[i]), lo, hi) };
        let len = if stride > 0 { (e - b + stride - 1) / stride } else { (b - e + (-stride) - 1) / (-stride) };
        out.push(len.max(0));
    }
    Ok(vec![out])
}
