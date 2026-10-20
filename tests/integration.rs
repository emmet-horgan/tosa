//! Tests of the public API: running graphs through the convenience functions
//! and the interpreter hooks (statistics, FLOP counting, tracing).
//!
//! Operator results are covered by the conformance suite (`conformance.rs`);
//! these tests reuse its multi-operator cases in `tests/networks`.

use std::cell::Cell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use tosa_interp::stats::OpInfo;
use tosa_interp::{
    FlopCounter, Graph, HookSet, InterpreterHook, OpTrace, StatsCollector, Tensor, npy, run_graph,
    run_graph_with_hooks, run_graph_with_stats,
};

fn case_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/networks").join(name)
}

#[derive(serde::Deserialize)]
struct Desc {
    tosa_file: String,
    ifm_name: Vec<String>,
    ifm_file: Vec<String>,
    ofm_name: Vec<String>,
    ofm_file: Vec<String>,
}

/// A network test case: its graph, inputs and reference outputs.
struct Case {
    graph: Vec<u8>,
    inputs: HashMap<String, Tensor>,
    expected: HashMap<String, Tensor>,
}

fn load_case(name: &str) -> Case {
    let dir = case_dir(name);
    let desc: Desc = serde_json::from_str(&std::fs::read_to_string(dir.join("desc.json")).unwrap()).unwrap();
    let graph = std::fs::read(dir.join(&desc.tosa_file)).unwrap();
    let parsed = Graph::from_flatbuffer(&graph).unwrap();
    let block = parsed.main_block().unwrap();
    let dtype = |name: &str| block.tensors.iter().find(|t| t.name == name).unwrap().dtype;
    let load = |names: &[String], files: &[String]| -> HashMap<String, Tensor> {
        names
            .iter()
            .zip(files)
            .map(|(name, file)| {
                let arr = npy::read(&dir.join(file)).unwrap();
                let data = arr.to_tensor_data(dtype(name)).unwrap();
                (name.clone(), Tensor::new(name.clone(), arr.shape, data))
            })
            .collect()
    };
    Case {
        inputs: load(&desc.ifm_name, &desc.ifm_file),
        expected: load(&desc.ofm_name, &desc.ofm_file),
        graph,
    }
}

/// Number of operators in the graph's main block.
fn operator_count(graph: &[u8]) -> usize {
    Graph::from_flatbuffer(graph).unwrap().main_block().unwrap().operators.len()
}

#[test]
fn run_graph_matches_reference() {
    let case = load_case("tf_add_f32");
    let outputs = run_graph(&case.graph, case.inputs).unwrap();
    assert_eq!(outputs.len(), case.expected.len());
    for (name, expected) in &case.expected {
        let actual = &outputs[name];
        assert_eq!(actual.shape, expected.shape, "{name}");
        let bits = |t: &Tensor| t.data.as_f32_slice().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(actual), bits(expected), "{name}");
    }
}

#[test]
fn run_graph_rejects_missing_input() {
    let case = load_case("tf_add_f32");
    let mut inputs = case.inputs;
    let removed = inputs.keys().next().unwrap().clone();
    inputs.remove(&removed);
    assert!(run_graph(&case.graph, inputs).is_err());
}

#[test]
fn run_graph_rejects_invalid_flatbuffer() {
    assert!(run_graph(b"not a tosa flatbuffer", HashMap::new()).is_err());
    assert!(run_graph(&[], HashMap::new()).is_err());
}

#[test]
fn stats_collector_counts_every_operator() {
    let case = load_case("mobilenet_like");
    let ops = operator_count(&case.graph);
    let (outputs, stats) = run_graph_with_stats(&case.graph, case.inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(stats.total_ops_executed as usize, ops);
    assert_eq!(stats.op_stats.values().map(|s| s.count).sum::<u64>() as usize, ops);
    assert!(stats.op_stats.contains_key("CONV2D"));
    assert!(stats.peak_memory_bytes > 0);
    assert!(stats.summary().contains("CONV2D"));
}

#[test]
fn op_trace_records_every_operator_once() {
    let case = load_case("conv_matmul");
    let graph = Graph::from_flatbuffer(&case.graph).unwrap();
    let operators = &graph.main_block().unwrap().operators;
    let mut trace = OpTrace::new();
    run_graph_with_hooks(&case.graph, case.inputs, &mut trace).unwrap();

    // Operators run in dataflow order; `op_index` is the position in the graph.
    let mut seen: Vec<usize> = trace.entries.iter().map(|e| e.op_index).collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..operators.len()).collect::<Vec<_>>());
    for e in &trace.entries {
        assert_eq!(e.op, operators[e.op_index].op.name());
        assert_eq!(e.output_shapes.len(), operators[e.op_index].outputs.len(), "{}", e.op);
    }
}

#[test]
fn flop_counter_matches_operator_shapes() {
    let case = load_case("conv_matmul");
    let mut trace = OpTrace::new();
    run_graph_with_hooks(&case.graph, case.inputs.clone(), &mut trace).unwrap();
    let mut flops = FlopCounter::new();
    run_graph_with_hooks(&case.graph, case.inputs, &mut flops).unwrap();

    // MATMUL [N, H, C] x [N, C, W]: 2*N*H*W*C; CONV2D: 2*N*OH*OW*OC*KH*KW*IC.
    let expected = |op: &str| -> u64 {
        trace
            .entries
            .iter()
            .filter(|e| e.op == op)
            .map(|e| {
                let p = |v: &[usize]| v.iter().map(|&d| d as u64).product::<u64>();
                let (i, w, o) = (&e.input_shapes[0], &e.input_shapes[1], &e.output_shapes[0]);
                match op {
                    "MATMUL" => 2 * p(&i[..3]) * w[2] as u64,
                    _ => 2 * p(o) * p(&w[1..]),
                }
            })
            .sum()
    };
    for op in ["MATMUL", "CONV2D"] {
        assert!(expected(op) > 0, "{op} missing from the graph");
        assert_eq!(flops.flops_by_op[op], expected(op), "{op}");
    }
    assert_eq!(flops.total_flops, flops.flops_by_op.values().sum::<u64>());
}

/// Counts hook calls into counters shared with the test.
struct CallCounter {
    pre: Rc<Cell<usize>>,
    post: Rc<Cell<usize>>,
}

impl InterpreterHook for CallCounter {
    fn pre_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor]) {
        self.pre.set(self.pre.get() + 1);
    }
    fn post_op(&mut self, _op: &OpInfo, _inputs: &[&Tensor], _outputs: &[&Tensor]) {
        self.post.set(self.post.get() + 1);
    }
}

#[test]
fn hookset_dispatches_to_every_hook() {
    let case = load_case("mobilenet_like");
    let ops = operator_count(&case.graph);
    let counters: Vec<_> = (0..2).map(|_| (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)))).collect();
    let mut hooks = HookSet::new();
    for (pre, post) in &counters {
        hooks.add(CallCounter { pre: pre.clone(), post: post.clone() });
    }
    hooks.add(StatsCollector::new());
    assert_eq!(hooks.hooks().len(), 3);

    let outputs = run_graph_with_hooks(&case.graph, case.inputs, &mut hooks).unwrap();
    assert_eq!(outputs.len(), 1);
    for (pre, post) in &counters {
        assert_eq!((pre.get(), post.get()), (ops, ops));
    }
}
