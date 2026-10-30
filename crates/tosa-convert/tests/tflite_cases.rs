//! The checked-in TFLite cases (`tests/tflite/*`, built by
//! `scripts/tflite_testcases.py`): float and quantized convolutional
//! networks and element-wise operators, converted and compared with
//! LiteRT's outputs.

mod common;

use common::{Tolerance, check_case, checked_in_cases};
use tosa_convert::default_frontends;

#[test]
fn checked_in_models_match_litert() {
    let cases = checked_in_cases();
    assert!(cases.len() >= 4, "expected the checked-in cases under tests/tflite");
    let mut failures = Vec::new();
    for case in &cases {
        failures.extend(check_case(case, &Tolerance::default()));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn frontend_is_detected_and_reports_structure() {
    let set = default_frontends();
    let frontend = set.by_name("tflite").unwrap();
    let case = checked_in_cases().into_iter().find(|c| c.name == "conv_f32").unwrap();
    assert!(frontend.sniff(&case.model));
    assert!(set.select(None, &case.dir.join("model.tflite"), &case.model).is_ok());
    let source = frontend.load(&case.model).unwrap();
    let counts = source.op_counts();
    assert_eq!(counts.get("CONV_2D"), Some(&1));
    assert_eq!(counts.get("SOFTMAX"), Some(&1));
    assert_eq!(source.inputs.len(), 1);
    assert_eq!(source.tensor(source.inputs[0]).name, "input");
    let report = tosa_convert::analyze(&source, frontend.lowerings());
    assert!(report.is_complete(), "{report:?}");
    assert!(frontend.load(b"not a model").is_err());
}

#[test]
fn uint8_models_present_unsigned_bits_at_the_boundary() {
    let case = checked_in_cases().into_iter().find(|c| c.name == "conv_uint8").unwrap();
    let conversion = common::convert_case(&case);
    let block = conversion.graph.main_block().unwrap();
    let input = block.tensors.iter().find(|t| t.name == "input").unwrap();
    assert_eq!(input.dtype, tosa_ir::DType::I8);
    let rescales: Vec<_> = block
        .operators
        .iter()
        .filter_map(|o| match &o.attribute {
            tosa_ir::OpAttribute::Rescale { input_unsigned, output_unsigned, .. } => {
                Some((*input_unsigned, *output_unsigned))
            }
            _ => None,
        })
        .collect();
    assert!(rescales.contains(&(true, false)), "no unsigned-input RESCALE: {rescales:?}");
    assert!(rescales.contains(&(false, true)), "no unsigned-output RESCALE: {rescales:?}");
    assert_eq!(block.outputs, vec!["output"]);
}
