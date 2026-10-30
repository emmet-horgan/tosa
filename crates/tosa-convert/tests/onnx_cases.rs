//! The checked-in ONNX cases (`tests/onnx/*`, built by
//! `scripts/onnx_testcases.py` with onnxruntime's outputs recorded).

mod common;

use common::{Tolerance, check_case, load_cases};
use std::path::Path;
use tosa_convert::default_frontends;

fn cases() -> Vec<common::Case> {
    load_cases(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/onnx"))
}

#[test]
fn checked_in_onnx_models_match_onnxruntime() {
    let cases = cases();
    assert!(cases.len() >= 3, "expected the checked-in cases under tests/onnx");
    let mut failures = Vec::new();
    for case in &cases {
        failures.extend(check_case(case, &Tolerance::onnxruntime()));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn onnx_frontend_is_detected_and_layout_transposes_cancel() {
    let set = default_frontends();
    let frontend = set.by_name("onnx").unwrap();
    let case = cases().into_iter().find(|c| c.name == "conv_bn_relu").unwrap();
    assert!(frontend.sniff(&case.model));
    let source = frontend.load(&case.model).unwrap();
    let counts = source.op_counts();
    assert_eq!(counts.get("Conv"), Some(&2));
    assert!(frontend.load(b"not a model").is_err());

    // Two convolutions with BatchNormalization and Relu between them: after
    // cleanup, only the graph-boundary layout changes remain.
    let conversion = common::convert_case(&case);
    let block = conversion.graph.main_block().unwrap();
    let transposes = block.operators.iter().filter(|o| o.op == tosa_ir::Op::Transpose).count();
    assert!(transposes <= 2, "{transposes} transposes left:\n{}", conversion.graph);
}
