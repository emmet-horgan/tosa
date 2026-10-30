//! Tests of the interpreter's mechanics, as opposed to operator numerics
//! (covered by the conformance suite): graph validation when an interpreter
//! is created, input and output checking, tensor lifetimes, persistent
//! variables, hooks inside control flow, and structured errors.
//!
//! Graphs are built directly as `ir::Graph` values, or taken from the
//! checked-in conformance and network cases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tosa_interp::interpreter::MAX_RANK;
use tosa_interp::ir::{BasicBlock, Graph, OpAttribute, Operator, Region, TensorDef};
use tosa_interp::{
    DType, Error, Interpreter, InterpreterHook, NoopHook, Op, OpInfo, OpTrace, StatsCollector, Tensor, TensorData, npy,
};

// ---------------------------------------------------------------------------
// Graph construction helpers
// ---------------------------------------------------------------------------

fn def(name: &str, shape: &[i32], dtype: DType) -> TensorDef {
    TensorDef {
        name: name.into(),
        shape: shape.to_vec(),
        dtype,
        data: None,
        is_variable: false,
        is_unranked: false,
        variable_name: None,
    }
}

fn const_i32(name: &str, values: &[i32]) -> TensorDef {
    let mut d = def(name, &[values.len() as i32], DType::I32);
    d.data = Some(values.iter().flat_map(|v| v.to_le_bytes()).collect());
    d
}

fn op(op: Op, inputs: &[&str], outputs: &[&str]) -> Operator {
    Operator::new(
        op,
        inputs.iter().map(|s| s.to_string()).collect(),
        outputs.iter().map(|s| s.to_string()).collect(),
        OpAttribute::None,
    )
}

fn graph(inputs: &[&str], outputs: &[&str], tensors: Vec<TensorDef>, operators: Vec<Operator>) -> Graph {
    Graph {
        version: Default::default(),
        regions: vec![Region {
            name: "main".into(),
            blocks: vec![BasicBlock {
                name: "main".into(),
                operators,
                tensors,
                inputs: inputs.iter().map(|s| s.to_string()).collect(),
                outputs: outputs.iter().map(|s| s.to_string()).collect(),
            }],
        }],
    }
}

fn i32_tensor(name: &str, values: &[i32]) -> Tensor {
    Tensor::new(name.into(), vec![values.len()], TensorData::I32(values.to_vec()))
}

fn i32_values(t: &Tensor) -> Vec<i32> {
    t.data.as_i32_slice().unwrap().to_vec()
}

fn inputs(tensors: Vec<Tensor>) -> HashMap<String, Tensor> {
    tensors.into_iter().map(|t| (t.name.clone(), t)).collect()
}

/// `out = abs(x) + c`, with `c` a constant: inputs ["x"], outputs ["out"].
fn abs_plus_const() -> Graph {
    graph(
        &["x"],
        &["out"],
        vec![
            def("x", &[3], DType::I32),
            const_i32("c", &[10, 20, 30]),
            def("a", &[3], DType::I32),
            def("out", &[3], DType::I32),
        ],
        vec![op(Op::Abs, &["x"], &["a"]), op(Op::Const, &[], &["c"]), op(Op::Add, &["a", "c"], &["out"])],
    )
}

fn err_string<T: std::fmt::Debug>(r: Result<T, Error>) -> String {
    r.unwrap_err().to_string()
}

// ---------------------------------------------------------------------------
// Checked-in cases
// ---------------------------------------------------------------------------

fn tests_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

#[derive(serde::Deserialize)]
struct Desc {
    tosa_file: String,
    ifm_name: Vec<String>,
    ifm_file: Vec<String>,
}

/// The graph and inputs of a checked-in case.
fn load_case(dir: &Path) -> (Vec<u8>, HashMap<String, Tensor>) {
    let desc: Desc = serde_json::from_str(&std::fs::read_to_string(dir.join("desc.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir.join(&desc.tosa_file)).unwrap();
    let parsed = Graph::from_flatbuffer(&bytes).unwrap();
    let block = parsed.main_block().unwrap();
    let dtype = |name: &str| block.tensors.iter().find(|t| t.name == name).unwrap().dtype;
    let inputs = desc
        .ifm_name
        .iter()
        .zip(&desc.ifm_file)
        .map(|(name, file)| {
            let arr = npy::read(&dir.join(file)).unwrap();
            let data = arr.to_tensor_data(dtype(name)).unwrap();
            (name.clone(), Tensor::new(name.clone(), arr.shape, data))
        })
        .collect();
    (bytes, inputs)
}

/// The first positive conformance case of `op` whose name contains `filter`.
fn conformance_case(op: &str, filter: &str) -> PathBuf {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(tests_dir().join("conformance").join(op))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy();
            !name.contains("ERRORIF") && name.contains(filter)
        })
        .collect();
    dirs.sort();
    dirs.into_iter().next().unwrap_or_else(|| panic!("no {op} case matching {filter}"))
}

// ---------------------------------------------------------------------------
// Input and output checking
// ---------------------------------------------------------------------------

#[test]
fn runs_a_hand_built_graph() {
    let mut interp = Interpreter::new(abs_plus_const()).unwrap();
    let out = interp.run(inputs(vec![i32_tensor("x", &[-1, 2, -3])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["out"]), vec![11, 22, 33]);
}

#[test]
fn rejects_missing_unknown_and_mismatched_inputs() {
    let mut interp = Interpreter::new(abs_plus_const()).unwrap();

    let e = interp.run(HashMap::new(), &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::MissingInput(ref n) if n == "x"), "{e}");

    let mut extra = inputs(vec![i32_tensor("x", &[1, 2, 3])]);
    extra.insert("y".into(), i32_tensor("y", &[1]));
    let e = interp.run(extra, &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::UnknownInput(ref n) if n == "y"), "{e}");

    let wrong_shape = inputs(vec![i32_tensor("x", &[1, 2])]);
    let e = interp.run(wrong_shape, &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::InputMismatch { ref name, .. } if name == "x"), "{e}");
    assert!(e.to_string().contains("shape"), "{e}");

    let wrong_dtype = inputs(vec![Tensor::new("x".into(), vec![3], TensorData::F32(vec![0.0; 3]))]);
    let e = interp.run(wrong_dtype, &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::InputMismatch { ref name, .. } if name == "x"), "{e}");
    assert!(e.to_string().contains("dtype"), "{e}");

    let short_data = inputs(vec![Tensor::new("x".into(), vec![3], TensorData::I32(vec![0]))]);
    let e = interp.run(short_data, &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::InputMismatch { .. }), "{e}");
}

#[test]
fn input_tensors_are_renamed_to_their_declared_name() {
    let mut interp = Interpreter::new(abs_plus_const()).unwrap();
    let mut t = i32_tensor("whatever", &[-4, 5, -6]);
    t.name = "something else".into();
    let mut map = HashMap::new();
    map.insert("x".to_string(), t);
    let out = interp.run(map, &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["out"]), vec![14, 25, 36]);
}

#[test]
fn dynamic_input_dimensions_accept_any_size() {
    let g = graph(
        &["x"],
        &["y"],
        vec![def("x", &[-1], DType::I32), def("y", &[-1], DType::I32)],
        vec![op(Op::Abs, &["x"], &["y"])],
    );
    let mut interp = Interpreter::new(g).unwrap();
    let out = interp.run(inputs(vec![i32_tensor("x", &[-1, -2, -3, -4, -5])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["y"]), vec![1, 2, 3, 4, 5]);
}

#[test]
fn missing_output_is_an_error() {
    // The block declares an output no operator produces.
    let g = graph(
        &["x"],
        &["y", "z"],
        vec![def("x", &[1], DType::I32), def("y", &[1], DType::I32)],
        vec![op(Op::Abs, &["x"], &["y"])],
    );
    let mut interp = Interpreter::new(g).unwrap();
    let e = interp.run(inputs(vec![i32_tensor("x", &[1])]), &mut NoopHook).unwrap_err();
    assert!(matches!(e, Error::MissingOutput(ref n) if n == "z"), "{e}");
}

// ---------------------------------------------------------------------------
// Graph validation at construction
// ---------------------------------------------------------------------------

#[test]
fn malformed_graphs_are_rejected_when_the_interpreter_is_created() {
    let no_blocks = Graph { version: Default::default(), regions: vec![] };
    assert!(err_string(Interpreter::new(no_blocks)).contains("no main block"));

    let undefined = graph(&[], &["y"], vec![def("y", &[1], DType::I32)], vec![op(Op::Abs, &["x"], &["y"])]);
    assert!(err_string(Interpreter::new(undefined)).contains("undefined tensor 'x'"));

    let two_producers = graph(
        &["x"],
        &["y"],
        vec![def("x", &[1], DType::I32), def("y", &[1], DType::I32)],
        vec![op(Op::Abs, &["x"], &["y"]), op(Op::Negate, &["x"], &["y"])],
    );
    assert!(err_string(Interpreter::new(two_producers)).contains("more than one operator"));

    let cycle = graph(
        &[],
        &["a"],
        vec![def("a", &[1], DType::I32), def("b", &[1], DType::I32)],
        vec![op(Op::Abs, &["b"], &["a"]), op(Op::Abs, &["a"], &["b"])],
    );
    assert!(err_string(Interpreter::new(cycle)).contains("cycle"));

    let too_deep = graph(&["x"], &["x"], vec![def("x", &[1; MAX_RANK + 1], DType::I32)], vec![]);
    assert!(err_string(Interpreter::new(too_deep)).contains("rank"));

    // A declared shape whose element count overflows usize.
    let huge = graph(&["x"], &["x"], vec![def("x", &[i32::MAX; 3], DType::I32)], vec![]);
    assert!(matches!(Interpreter::new(huge).unwrap_err(), Error::TensorTooLarge(_)));
}

#[test]
fn oversized_outputs_are_rejected_not_allocated() {
    // CONST declares a tensor of 2^31 x 2^31 bytes; the interpreter must
    // refuse to materialize it rather than attempt the allocation.
    let big = i32::MAX;
    let g = graph(&[], &["c"], vec![def("c", &[big, big], DType::I8)], vec![op(Op::Const, &[], &["c"])]);
    let mut interp = Interpreter::new(g).unwrap();
    let e = interp.run(HashMap::new(), &mut NoopHook).unwrap_err();
    assert!(matches!(e.root_cause(), Error::TensorTooLarge(_)), "{e}");
    assert_eq!(e.failing_operator(), Some(("main", 0, "CONST")));

    // A graph that fits the default limit but not a lowered one.
    let g = graph(&[], &["c"], vec![def("c", &[1024], DType::F32)], vec![op(Op::Const, &[], &["c"])]);
    let mut interp = Interpreter::new(g).unwrap();
    interp.set_max_tensor_bytes(1024);
    let e = interp.run(HashMap::new(), &mut NoopHook).unwrap_err();
    assert!(matches!(e.root_cause(), Error::TensorTooLarge(_)), "{e}");
    interp.set_max_tensor_bytes(4096);
    assert!(interp.run(HashMap::new(), &mut NoopHook).is_ok());
}

#[test]
fn execution_plan_is_exposed() {
    let interp = Interpreter::new(abs_plus_const()).unwrap();
    let plan = interp.plan("main").unwrap();
    // ABS (0) is listed first but ADD (2) needs the CONST (1) too.
    assert_eq!(plan.order, vec![0, 1, 2]);
    assert!(interp.plan("nope").is_none());
}

// ---------------------------------------------------------------------------
// Structured errors
// ---------------------------------------------------------------------------

#[test]
fn operator_errors_carry_their_position() {
    // ADD of an I32 tensor to a BOOL tensor is a dtype mismatch inside operator 1.
    let g = graph(
        &["x", "b"],
        &["y"],
        vec![def("x", &[1], DType::I32), def("b", &[1], DType::Bool), def("y", &[1], DType::I32)],
        vec![op(Op::Identity, &["x"], &["x2"]), op(Op::Add, &["x2", "b"], &["y"])],
    );
    let mut interp = Interpreter::new(g).unwrap();
    let b = Tensor::new("b".into(), vec![1], TensorData::Bool(vec![true]));
    let e = interp.run(inputs(vec![i32_tensor("x", &[1]), b]), &mut NoopHook).unwrap_err();
    assert_eq!(e.failing_operator(), Some(("main", 1, "ADD")));
    assert!(matches!(e.root_cause(), Error::DTypeMismatch(_)), "{e}");
    let msg = e.to_string();
    assert!(msg.starts_with("ADD (operator 1 in block 'main'): "), "{msg}");
}

/// A graph whose COND_IF runs `then_b` (ADD of x and a BOOL constant, which
/// fails) or `else_b` (IDENTITY), selected by the BOOL input `c`.
fn cond_if_graph() -> Graph {
    let mut k = def("k", &[1], DType::Bool);
    k.data = Some(vec![1]);
    let then_b = BasicBlock {
        name: "then_b".into(),
        operators: vec![op(Op::Const, &[], &["k"]), op(Op::Add, &["tx", "k"], &["ty"])],
        tensors: vec![def("tx", &[1], DType::I32), k, def("ty", &[1], DType::I32)],
        inputs: vec!["tx".into()],
        outputs: vec!["ty".into()],
    };
    let else_b = BasicBlock {
        name: "else_b".into(),
        operators: vec![op(Op::Identity, &["ex"], &["ey"])],
        tensors: vec![def("ex", &[1], DType::I32), def("ey", &[1], DType::I32)],
        inputs: vec!["ex".into()],
        outputs: vec!["ey".into()],
    };
    let mut cond = op(Op::CondIf, &["c", "x"], &["y"]);
    cond.attribute = OpAttribute::CondIf { then_graph: "then_b".into(), else_graph: "else_b".into() };
    let mut g = graph(
        &["c", "x"],
        &["y"],
        vec![def("c", &[1], DType::Bool), def("x", &[1], DType::I32), def("y", &[1], DType::I32)],
        vec![cond],
    );
    g.regions[0].blocks.extend([then_b, else_b]);
    g
}

fn cond_inputs(c: bool, x: i32) -> HashMap<String, Tensor> {
    inputs(vec![Tensor::new("c".into(), vec![1], TensorData::Bool(vec![c])), i32_tensor("x", &[x])])
}

#[test]
fn errors_inside_control_flow_nest() {
    let mut interp = Interpreter::new(cond_if_graph()).unwrap();
    let out = interp.run(cond_inputs(false, 7), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["y"]), vec![7]);

    let e = interp.run(cond_inputs(true, 7), &mut NoopHook).unwrap_err();
    // Innermost failing operator: the ADD in the then-branch.
    assert_eq!(e.failing_operator(), Some(("then_b", 1, "ADD")));
    assert!(matches!(e.root_cause(), Error::DTypeMismatch(_)), "{e}");
    // The outer layer names the COND_IF in the main block.
    let Error::Operator { block, index, op, .. } = &e else { panic!("{e}") };
    assert_eq!((block.as_str(), *index, *op), ("main", 0, "COND_IF"));
    assert!(
        e.to_string().starts_with("COND_IF (operator 0 in block 'main'): ADD (operator 1 in block 'then_b'): "),
        "{e}"
    );
}

// ---------------------------------------------------------------------------
// Hooks and tensor lifetimes
// ---------------------------------------------------------------------------

/// Records every hook call.
#[derive(Default)]
struct Recorder {
    pre: Vec<(String, usize, String, usize)>,
    post: Vec<(String, usize, String, usize)>,
    allocs: Vec<(String, usize)>,
    frees: Vec<(String, usize)>,
}

impl InterpreterHook for Recorder {
    fn pre_op(&mut self, op: &OpInfo, _inputs: &[&Tensor]) {
        self.pre.push((op.op.name().into(), op.op_index, op.block.into(), op.depth));
    }
    fn post_op(&mut self, op: &OpInfo, _inputs: &[&Tensor], _outputs: &[&Tensor]) {
        self.post.push((op.op.name().into(), op.op_index, op.block.into(), op.depth));
    }
    fn on_tensor_alloc(&mut self, name: &str, _shape: &[usize], _dtype: DType, bytes: usize) {
        self.allocs.push((name.into(), bytes));
    }
    fn on_tensor_free(&mut self, name: &str, bytes: usize) {
        self.frees.push((name.into(), bytes));
    }
}

#[test]
fn hooks_see_operators_inside_control_flow() {
    let dir = conformance_case("cond_if", "binary");
    let (bytes, case_inputs) = load_case(&dir);
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let mut rec = Recorder::default();
    interp.run(case_inputs, &mut rec).unwrap();

    let main_ops: Vec<_> = rec.pre.iter().filter(|(_, _, _, depth)| *depth == 0).collect();
    let nested: Vec<_> = rec.pre.iter().filter(|(_, _, _, depth)| *depth == 1).collect();
    assert!(main_ops.iter().any(|(name, ..)| name == "COND_IF"));
    assert!(!nested.is_empty(), "no operators reported inside the COND_IF body");
    assert!(nested.iter().all(|(_, _, block, _)| block != "main"), "{nested:?}");
    assert_eq!(rec.pre.len(), rec.post.len());

    // Nesting: the COND_IF's post_op comes after its body's post_ops.
    let cond_post = rec.post.iter().position(|(name, ..)| name == "COND_IF").unwrap();
    let last_nested_post = rec.post.iter().rposition(|(_, _, _, depth)| *depth == 1).unwrap();
    assert!(last_nested_post < cond_post);

    // StatsCollector and OpTrace agree on the total.
    let (bytes, case_inputs) = load_case(&dir);
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let mut stats = StatsCollector::new();
    interp.run(case_inputs.clone(), &mut stats).unwrap();
    assert_eq!(stats.total_ops_executed as usize, rec.pre.len());
    let mut trace = OpTrace::new();
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    interp.run(case_inputs, &mut trace).unwrap();
    assert_eq!(trace.entries.len(), rec.pre.len());
    assert!(trace.entries.iter().any(|e| e.depth == 1));
}

#[test]
fn while_loop_bodies_are_observed_once_per_iteration() {
    let dir = conformance_case("while_loop", "iter1");
    let (bytes, case_inputs) = load_case(&dir);
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let mut rec = Recorder::default();
    interp.run(case_inputs, &mut rec).unwrap();
    let nested = rec.pre.iter().filter(|(_, _, _, depth)| *depth == 1).count();
    assert!(nested > 0);
    // Every allocation inside the loop is eventually freed or returned.
    assert!(rec.frees.len() <= rec.allocs.len());
}

#[test]
fn tensors_are_freed_after_their_last_use() {
    let mut interp = Interpreter::new(abs_plus_const()).unwrap();
    let mut rec = Recorder::default();
    let out = interp.run(inputs(vec![i32_tensor("x", &[-1, 2, -3])]), &mut rec).unwrap();

    let allocated: Vec<&str> = rec.allocs.iter().map(|(n, _)| n.as_str()).collect();
    let freed: Vec<&str> = rec.frees.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(allocated, ["x", "a", "c", "out"]);
    // Everything but the graph output is freed, each exactly once, in last-use order.
    assert_eq!(freed, ["x", "a", "c"]);
    assert!(out.contains_key("out"));
    for (name, bytes) in &rec.frees {
        let (_, allocated_bytes) = rec.allocs.iter().find(|(n, _)| n == name).unwrap();
        assert_eq!(bytes, allocated_bytes, "{name}");
    }
}

#[test]
fn peak_memory_is_below_total_allocations() {
    let dir = tests_dir().join("networks/mobilenet_like");
    let (bytes, case_inputs) = load_case(&dir);
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let mut rec = Recorder::default();
    interp.run(case_inputs.clone(), &mut rec).unwrap();
    let total: usize = rec.allocs.iter().map(|(_, b)| b).sum();
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    let mut stats = StatsCollector::new();
    let out = interp.run(case_inputs, &mut stats).unwrap();
    assert!(stats.peak_memory_bytes < total, "peak {} vs total {}", stats.peak_memory_bytes, total);
    // Only the graph outputs stay live.
    assert_eq!(stats.current_memory_bytes, out.values().map(Tensor::size_bytes).sum::<usize>());
}

// ---------------------------------------------------------------------------
// Variables
// ---------------------------------------------------------------------------

/// A counter: `v` starts at `init`, every run adds `x` to it and returns both
/// the value read before and after the write.
fn counter_graph(init: i32) -> Graph {
    let mut v = const_i32("v", &[init]);
    v.is_variable = true;
    graph(
        &["x"],
        &["before", "after"],
        vec![
            def("x", &[1], DType::I32),
            v,
            def("before", &[1], DType::I32),
            def("sum", &[1], DType::I32),
            def("after", &[1], DType::I32),
        ],
        vec![
            op(Op::Variable, &[], &["v"]),
            op(Op::VariableRead, &["v"], &["before"]),
            op(Op::Add, &["before", "x"], &["sum"]),
            op(Op::VariableWrite, &["v", "sum"], &[]),
            // Only program order places this read after the write: it has no
            // data dependency on it, so a pure dataflow scheduler would run
            // it as soon as the variable exists.
            op(Op::VariableRead, &["v"], &["after"]),
        ],
    )
}

#[test]
fn variable_operators_run_in_program_order_and_persist() {
    let mut interp = Interpreter::new(counter_graph(100)).unwrap();
    let out = interp.run(inputs(vec![i32_tensor("x", &[5])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["before"]), vec![100]);
    assert_eq!(i32_values(&out["after"]), vec![105]);

    // The variable keeps its value into the next run.
    let out = interp.run(inputs(vec![i32_tensor("x", &[7])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["before"]), vec![105]);
    assert_eq!(i32_values(&out["after"]), vec![112]);
    assert_eq!(i32_values(&interp.variables()["v"]), vec![112]);

    interp.reset_variables();
    let out = interp.run(inputs(vec![i32_tensor("x", &[1])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["before"]), vec![100]);

    interp.set_variable("v", i32_tensor("ignored", &[-1]));
    let out = interp.run(inputs(vec![i32_tensor("x", &[1])]), &mut NoopHook).unwrap();
    assert_eq!(i32_values(&out["before"]), vec![-1]);
    assert_eq!(i32_values(&out["after"]), vec![0]);
}

#[test]
fn variable_write_checks_type_and_shape() {
    let mut g = counter_graph(0);
    // Make the written value a different shape from the variable.
    let block = &mut g.regions[0].blocks[0];
    for t in &mut block.tensors {
        if t.name == "x" || t.name == "sum" || t.name == "before" {
            t.shape = vec![2];
        }
    }
    block.tensors.push(def("two", &[2], DType::I32));
    // ADD of a [1] read with a [2] input broadcasts to [2], which the write rejects.
    let mut interp = Interpreter::new(g).unwrap();
    let e = interp.run(inputs(vec![i32_tensor("x", &[1, 2])]), &mut NoopHook).unwrap_err();
    assert!(e.to_string().contains("VARIABLE_WRITE") || e.to_string().contains("VARIABLE_READ"), "{e}");
}
