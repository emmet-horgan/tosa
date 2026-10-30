//! End-to-end tests of the lowering framework with a toy frontend: source
//! graphs built in memory, lowered through a small registry, serialized,
//! and executed with the interpreter.

use std::collections::HashMap;
use std::path::Path;

use tosa_convert::quant::{RescaleSpec, emit_rescale};
use tosa_convert::{
    AttrValue, ConvertError, ConvertOptions, Frontend, FrontendSet, LowerCtx, LoweringRegistry, SourceDType,
    SourceGraph, SourceNode, SourceTensor, analyze, convert, layout, lower_graph,
};
use tosa_interp::{Interpreter, NoopHook, Tensor, TensorData};
use tosa_ir::{DType, Graph, NanMode, Op, OpAttribute};

// ---------------------------------------------------------------------------
// Toy lowerings
// ---------------------------------------------------------------------------

fn lower_add(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> tosa_convert::Result<()> {
    let a = ctx.value(node.input(0)?)?;
    let b = ctx.value(node.input(1)?)?;
    ctx.emit(Op::Add, OpAttribute::None, &[&a, &b], node.output(0)?)?;
    Ok(())
}

fn lower_relu(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> tosa_convert::Result<()> {
    let x = ctx.value(node.input(0)?)?;
    if x.dtype != DType::F32 {
        return ctx.fail("only f32 is supported");
    }
    let attr = OpAttribute::Clamp {
        min_val: 0f32.to_le_bytes().to_vec(),
        max_val: f32::MAX.to_le_bytes().to_vec(),
        nan_mode: NanMode::Propagate,
    };
    ctx.emit(Op::Clamp, attr, &[&x], node.output(0)?)?;
    Ok(())
}

fn lower_transpose(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> tosa_convert::Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let perms: Vec<i32> = node.attr_ints("perms")?.iter().map(|&p| p as i32).collect();
    let t = layout::transpose(ctx, &x, &perms, "transposed");
    ctx.bind(node.output(0)?, t);
    Ok(())
}

fn lower_requantize(node: &SourceNode, ctx: &mut LowerCtx<'_>) -> tosa_convert::Result<()> {
    let x = ctx.value(node.input(0)?)?;
    let out = node.output(0)?;
    let spec = RescaleSpec::per_tensor(node.attr_float("scale")?, 0, node.attr_int("zero_point")?);
    let r = emit_rescale(ctx, &x, ctx.dtype(out)?, &spec, "requant")?;
    ctx.bind(out, r);
    Ok(())
}

fn registry() -> LoweringRegistry {
    let mut r = LoweringRegistry::new();
    r.register("add", lower_add);
    r.register("relu", lower_relu);
    r.register("transpose", lower_transpose);
    r.register("requantize", lower_requantize);
    r
}

fn f32_tensor(name: &str, shape: &[i64]) -> SourceTensor {
    SourceTensor::new(name, SourceDType::F32, shape)
}

fn run(graph: &Graph, inputs: Vec<(&str, Vec<usize>, TensorData)>) -> HashMap<String, Tensor> {
    // Through the serializer, so the test covers what a file would hold.
    let bytes = graph.to_flatbuffer().unwrap();
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let inputs = inputs
        .into_iter()
        .map(|(name, shape, data)| (name.to_string(), Tensor::new(name.to_string(), shape, data)))
        .collect();
    interp.run(inputs, &mut NoopHook).unwrap()
}

fn ops(graph: &Graph) -> Vec<&'static str> {
    graph.main_block().unwrap().operators.iter().map(|o| o.op.name()).collect()
}

// ---------------------------------------------------------------------------
// Lowering
// ---------------------------------------------------------------------------

#[test]
fn lowers_a_small_graph_and_records_locations() {
    // y = relu(x + c)
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[1, 4]));
    let c = g.add_tensor(f32_tensor("c", &[1, 4]).with_constant(TensorData::F32(vec![-5.0, 0.0, 1.0, 2.0])));
    let s = g.add_tensor(f32_tensor("sum", &[1, 4]));
    let y = g.add_tensor(f32_tensor("y", &[1, 4]));
    g.add_node(SourceNode::new("add", "add_0", vec![Some(x), Some(c)], vec![s]));
    g.add_node(SourceNode::new("relu", "relu_0", vec![Some(s)], vec![y]));
    g.inputs = vec![x];
    g.outputs = vec![y];

    let conv = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap();
    assert_eq!(conv.report.supported.len(), 2);
    assert!(conv.report.is_complete());
    assert_eq!(ops(&conv.graph), vec!["CONST", "ADD", "CLAMP"]);
    let block = conv.graph.main_block().unwrap();
    assert_eq!(block.inputs, vec!["x"]);
    assert_eq!(block.outputs, vec!["y"]);
    let locations: Vec<_> = block.operators.iter().map(|o| o.location.as_deref()).collect();
    assert_eq!(locations, vec![Some("add_0"), Some("add_0"), Some("relu_0")]);

    let out = run(&conv.graph, vec![("x", vec![1, 4], TensorData::F32(vec![1.0, 1.0, 1.0, 1.0]))]);
    assert_eq!(out["y"].data.as_f32_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn unsupported_operators_are_reported_before_lowering() {
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[2]));
    let y = g.add_tensor(f32_tensor("y", &[2]));
    let z = g.add_tensor(f32_tensor("z", &[2]));
    g.add_node(SourceNode::new("softmax", "s0", vec![Some(x)], vec![y]));
    g.add_node(SourceNode::new("softmax", "s1", vec![Some(y)], vec![z]));
    g.add_node(SourceNode::new("relu", "r", vec![Some(z)], vec![z]));
    g.inputs = vec![x];
    g.outputs = vec![z];

    let report = analyze(&g, &registry());
    assert_eq!(report.unsupported.get("softmax"), Some(&2));
    assert_eq!(report.supported.get("relu"), Some(&1));
    let err = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap_err();
    assert!(matches!(err, ConvertError::Unsupported(ref v) if v == &[("softmax".to_string(), 2)]), "{err}");
    assert!(err.to_string().contains("softmax (x2)"));
}

#[test]
fn lowering_errors_name_the_node() {
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(SourceTensor::new("x", SourceDType::I32, &[2]));
    let y = g.add_tensor(SourceTensor::new("y", SourceDType::I32, &[2]));
    g.add_node(SourceNode::new("relu", "the_relu", vec![Some(x)], vec![y]));
    g.inputs = vec![x];
    g.outputs = vec![y];
    let err = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap_err();
    assert!(
        matches!(err, ConvertError::Lowering { ref op, ref node, .. } if op == "relu" && node == "the_relu"),
        "{err}"
    );

    // A type TOSA lacks.
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(SourceTensor::new("x", SourceDType::U8, &[2]));
    g.inputs = vec![x];
    g.outputs = vec![x];
    let err = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap_err();
    assert!(err.to_string().contains("U8"), "{err}");
}

#[test]
fn outputs_that_alias_inputs_or_each_other_get_identity_copies() {
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[2]));
    g.inputs = vec![x];
    g.outputs = vec![x, x];
    let conv = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap();
    assert_eq!(ops(&conv.graph), vec!["IDENTITY", "IDENTITY"]);
    let out = run(&conv.graph, vec![("x", vec![2], TensorData::F32(vec![1.0, 2.0]))]);
    assert_eq!(out.len(), 2);
    for t in out.values() {
        assert_eq!(t.data.as_f32_slice().unwrap(), &[1.0, 2.0]);
    }
}

// ---------------------------------------------------------------------------
// Cleanup passes
// ---------------------------------------------------------------------------

fn transpose_chain(cleanup: bool) -> Graph {
    // relu(T2(T1(x))) with T2 = inverse of T1, then another pair feeding the output.
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[1, 2, 3, 4]));
    let t1 = g.add_tensor(f32_tensor("t1", &[1, 3, 4, 2]));
    let t2 = g.add_tensor(f32_tensor("t2", &[1, 2, 3, 4]));
    let r = g.add_tensor(f32_tensor("r", &[1, 2, 3, 4]));
    let t3 = g.add_tensor(f32_tensor("t3", &[1, 3, 4, 2]));
    let y = g.add_tensor(f32_tensor("y", &[1, 2, 3, 4]));
    let perms = |p: &[i64]| AttrValue::Ints(p.to_vec());
    g.add_node(SourceNode::new("transpose", "T1", vec![Some(x)], vec![t1]).with_attr("perms", perms(&[0, 2, 3, 1])));
    g.add_node(SourceNode::new("transpose", "T2", vec![Some(t1)], vec![t2]).with_attr("perms", perms(&[0, 3, 1, 2])));
    g.add_node(SourceNode::new("relu", "relu", vec![Some(t2)], vec![r]));
    g.add_node(SourceNode::new("transpose", "T3", vec![Some(r)], vec![t3]).with_attr("perms", perms(&[0, 2, 3, 1])));
    g.add_node(SourceNode::new("transpose", "T4", vec![Some(t3)], vec![y]).with_attr("perms", perms(&[0, 3, 1, 2])));
    g.inputs = vec![x];
    g.outputs = vec![y];
    lower_graph(&g, &registry(), &ConvertOptions { cleanup, ..Default::default() }).unwrap().graph
}

#[test]
fn transpose_pairs_cancel() {
    let before = transpose_chain(false);
    assert_eq!(ops(&before).iter().filter(|o| **o == "TRANSPOSE").count(), 4);
    let after = transpose_chain(true);
    assert_eq!(ops(&after), vec!["CLAMP"]);
    let block = after.main_block().unwrap();
    assert_eq!(block.operators[0].inputs, vec!["x"]);
    assert_eq!(block.operators[0].outputs, vec!["y"]);
    assert_eq!(block.outputs, vec!["y"]);
    assert_eq!(block.tensors.len(), 2, "{:?}", block.tensors);

    let data: Vec<f32> = (0..24).map(|i| i as f32 - 12.0).collect();
    let expected: Vec<f32> = data.iter().map(|v| v.max(0.0)).collect();
    let a = run(&before, vec![("x", vec![1, 2, 3, 4], TensorData::F32(data.clone()))]);
    let b = run(&after, vec![("x", vec![1, 2, 3, 4], TensorData::F32(data))]);
    assert_eq!(a["y"].data.as_f32_slice().unwrap(), &expected[..]);
    assert_eq!(b["y"].data.as_f32_slice().unwrap(), &expected[..]);
}

#[test]
fn non_inverse_transposes_merge_into_one() {
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[2, 3, 4]));
    let t1 = g.add_tensor(f32_tensor("t1", &[3, 2, 4]));
    let y = g.add_tensor(f32_tensor("y", &[4, 3, 2]));
    g.add_node(
        SourceNode::new("transpose", "T1", vec![Some(x)], vec![t1]).with_attr("perms", AttrValue::Ints(vec![1, 0, 2])),
    );
    g.add_node(
        SourceNode::new("transpose", "T2", vec![Some(t1)], vec![y]).with_attr("perms", AttrValue::Ints(vec![2, 0, 1])),
    );
    g.inputs = vec![x];
    g.outputs = vec![y];
    let graph = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap().graph;
    assert_eq!(ops(&graph), vec!["TRANSPOSE"]);
    let OpAttribute::Transpose { perms } = &graph.main_block().unwrap().operators[0].attribute else { panic!() };
    assert_eq!(perms, &[2, 1, 0]);
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let out = run(&graph, vec![("x", vec![2, 3, 4], TensorData::F32(data.clone()))]);
    // out[k][j][i] = x[i][j][k]
    let got = out["y"].data.as_f32_slice().unwrap();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(got[(k * 3 + j) * 2 + i], data[(i * 3 + j) * 4 + k]);
            }
        }
    }
}

#[test]
fn constant_subgraphs_are_folded() {
    // y = x + (a + b) with a, b constants: the inner add folds to one CONST.
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(f32_tensor("x", &[2]));
    let a = g.add_tensor(f32_tensor("a", &[2]).with_constant(TensorData::F32(vec![1.0, 2.0])));
    let b = g.add_tensor(f32_tensor("b", &[2]).with_constant(TensorData::F32(vec![10.0, 20.0])));
    let ab = g.add_tensor(f32_tensor("ab", &[2]));
    let y = g.add_tensor(f32_tensor("y", &[2]));
    g.add_node(SourceNode::new("add", "inner", vec![Some(a), Some(b)], vec![ab]));
    g.add_node(SourceNode::new("add", "outer", vec![Some(x), Some(ab)], vec![y]));
    g.inputs = vec![x];
    g.outputs = vec![y];

    let graph = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap().graph;
    assert_eq!(ops(&graph), vec!["CONST", "ADD"]);
    let block = graph.main_block().unwrap();
    let folded = block.tensors.iter().find(|t| t.name == "ab").unwrap();
    assert_eq!(folded.data.as_deref(), Some(&TensorData::F32(vec![11.0, 22.0]).to_raw_bytes()[..]));
    // The folded constant keeps the location of the node it came from.
    assert_eq!(block.operators[0].location.as_deref(), Some("inner"));
    let out = run(&graph, vec![("x", vec![2], TensorData::F32(vec![0.5, 0.5]))]);
    assert_eq!(out["y"].data.as_f32_slice().unwrap(), &[11.5, 22.5]);

    // Folding respects the size limit.
    let graph =
        lower_graph(&g, &registry(), &ConvertOptions { max_folded_bytes: 4, ..Default::default() }).unwrap().graph;
    assert_eq!(ops(&graph), vec!["CONST", "CONST", "ADD", "ADD"]);
}

// ---------------------------------------------------------------------------
// Quantization helpers
// ---------------------------------------------------------------------------

#[test]
fn rescale_emission_matches_the_interpreter() {
    let mut g = SourceGraph::new("toy");
    let x = g.add_tensor(SourceTensor::new("x", SourceDType::I32, &[4]));
    let y = g.add_tensor(SourceTensor::new("y", SourceDType::I8, &[4]));
    g.add_node(
        SourceNode::new("requantize", "rq", vec![Some(x)], vec![y])
            .with_attr("scale", AttrValue::Float(0.5))
            .with_attr("zero_point", AttrValue::Int(3)),
    );
    g.inputs = vec![x];
    g.outputs = vec![y];
    let graph = lower_graph(&g, &registry(), &ConvertOptions::default()).unwrap().graph;
    let rescale = graph.main_block().unwrap().operators.iter().find(|o| o.op == Op::Rescale).unwrap();
    assert!(matches!(rescale.attribute, OpAttribute::Rescale { scale32: true, per_channel: false, .. }));

    let out = run(&graph, vec![("x", vec![4], TensorData::I32(vec![3, -3, 100, 1000]))]);
    // round-half-up(x / 2) + 3, saturated to int8: 2+3, -1+3, 50+3, 127.
    assert_eq!(out["y"].data, TensorData::I8(vec![5, 2, 53, 127]));
}

// ---------------------------------------------------------------------------
// Frontends
// ---------------------------------------------------------------------------

/// A frontend for files holding the text `TOY!` followed by a vector length:
/// the model adds one to its input.
struct ToyFrontend {
    lowerings: LoweringRegistry,
}

impl Frontend for ToyFrontend {
    fn name(&self) -> &str {
        "toy"
    }
    fn extensions(&self) -> &[&str] {
        &["toy"]
    }
    fn sniff(&self, bytes: &[u8]) -> bool {
        bytes.starts_with(b"TOY!")
    }
    fn load(&self, bytes: &[u8]) -> tosa_convert::Result<SourceGraph> {
        let text = std::str::from_utf8(bytes).ok().and_then(|s| s.strip_prefix("TOY!"));
        let n: i64 = text
            .and_then(|s| s.trim().parse().ok())
            .ok_or_else(|| ConvertError::Frontend { format: "toy".into(), reason: "bad header".into() })?;
        let mut g = SourceGraph::new("toy");
        let x = g.add_tensor(f32_tensor("x", &[n]));
        let one = g.add_tensor(f32_tensor("one", &[1]).with_constant(TensorData::F32(vec![1.0])));
        let y = g.add_tensor(f32_tensor("y", &[n]));
        g.add_node(SourceNode::new("add", "plus_one", vec![Some(x), Some(one)], vec![y]));
        g.inputs = vec![x];
        g.outputs = vec![y];
        Ok(g)
    }
    fn lowerings(&self) -> &LoweringRegistry {
        &self.lowerings
    }
}

#[test]
fn frontends_are_selected_by_name_contents_or_extension() {
    let mut set = FrontendSet::new();
    set.register(ToyFrontend { lowerings: registry() });
    assert_eq!(set.names(), vec!["toy"]);
    assert!(set.by_name("tflite").is_err());
    assert!(set.for_path(Path::new("model.onnx")).is_err());
    assert_eq!(set.for_path(Path::new("model.TOY")).unwrap().name(), "toy");
    assert_eq!(set.sniff(b"TOY! 3").unwrap().name(), "toy");
    assert!(set.sniff(b"nope").is_err());
    assert_eq!(set.select(None, Path::new("x.bin"), b"TOY! 3").unwrap().name(), "toy");
    assert_eq!(set.select(None, Path::new("x.toy"), b"garbage").unwrap().name(), "toy");

    let frontend = set.by_name("toy").unwrap();
    assert!(matches!(frontend.load(b"TOY! x").unwrap_err(), ConvertError::Frontend { .. }));
    let conv = convert(frontend, b"TOY! 3", &ConvertOptions::default()).unwrap();
    let out = run(&conv.graph, vec![("x", vec![3], TensorData::F32(vec![1.0, 2.0, 3.0]))]);
    assert_eq!(out["y"].data.as_f32_slice().unwrap(), &[2.0, 3.0, 4.0]);
}
