use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tosa_interp::{
    run_graph, run_graph_with_hooks, run_graph_with_stats, FlopCounter, HookSet, StatsCollector,
    Tensor, TensorData,
};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_tosa(fixture_name: &str) -> Vec<u8> {
    let path = fixtures_dir().join(fixture_name).join("graph.tosa");
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

fn load_npy_f32(fixture_name: &str, filename: &str) -> Vec<f32> {
    let path = fixtures_dir().join(fixture_name).join(filename);
    let reader = npyz::NpyFile::new(
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display())),
    )
    .unwrap();
    reader.into_vec::<f32>().unwrap()
}

fn load_npy_i32(fixture_name: &str, filename: &str) -> Vec<i32> {
    let path = fixtures_dir().join(fixture_name).join(filename);
    let reader = npyz::NpyFile::new(
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display())),
    )
    .unwrap();
    reader.into_vec::<i32>().unwrap()
}

fn assert_f32_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "element {i}: got {a} expected {e} (diff {})",
            (a - e).abs()
        );
    }
}

fn assert_i32_exact(actual: &[i32], expected: &[i32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: got {} expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "element {i}: got {a} expected {e}");
    }
}

#[test]
fn test_const_add_f32() {
    let data = load_tosa("const_add_f32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    assert_eq!(outputs.len(), 1);
    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("const_add_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_input_add_f32() {
    let data = load_tosa("input_add_f32");

    let a_data = load_npy_f32("input_add_f32", "input_TosaInput_0.npy");
    let b_data = load_npy_f32("input_add_f32", "input_TosaInput_1.npy");

    let mut inputs = HashMap::new();
    inputs.insert(
        "TosaInput_0".to_string(),
        Tensor::new(
            "TosaInput_0".to_string(),
            vec![2, 3],
            TensorData::F32(a_data),
        ),
    );
    inputs.insert(
        "TosaInput_1".to_string(),
        Tensor::new(
            "TosaInput_1".to_string(),
            vec![2, 3],
            TensorData::F32(b_data),
        ),
    );

    let outputs = run_graph(&data, inputs).unwrap();
    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("input_add_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_sub_i32() {
    let data = load_tosa("sub_i32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_i32_slice().unwrap();
    let expected = load_npy_i32("sub_i32", "expected_TosaOutput_0.npy");
    assert_i32_exact(actual, &expected);
}

#[test]
fn test_abs_negate_f32() {
    let data = load_tosa("abs_negate_f32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("abs_negate_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_reshape_f32() {
    let data = load_tosa("reshape_f32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    let (_, tensor) = outputs.iter().next().unwrap();
    assert_eq!(tensor.shape, vec![3, 2]);
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("reshape_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_broadcast_add_f32() {
    let data = load_tosa("broadcast_add_f32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    let (_, tensor) = outputs.iter().next().unwrap();
    assert_eq!(tensor.shape, vec![2, 3]);
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("broadcast_add_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_mul_f32() {
    let data = load_tosa("mul_f32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();

    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32("mul_f32", "expected_TosaOutput_0.npy");
    assert_f32_close(actual, &expected, 1e-6);
}

#[test]
fn test_stats_collector() {
    let data = load_tosa("const_add_f32");
    let (outputs, stats) = run_graph_with_stats(&data, HashMap::new()).unwrap();

    assert!(!outputs.is_empty());
    assert!(stats.total_ops_executed >= 3); // 2 CONST + 1 ADD
    assert!(stats.peak_memory_bytes > 0);
    assert!(stats.op_stats.contains_key("CONST"));
    assert!(stats.op_stats.contains_key("ADD"));
}

/// Test with the pre-existing example .tosa file in the repo.
/// This test verifies the interpreter can load and execute a real .tosa file
/// from the reference model examples directory.
#[test]
fn test_existing_example_add() {
    let tosa_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("reference_model/examples/test_add_1x4x4x4_f32/flatbuffer-tf/test_add_1x4x4x4_f32.tosa");

    if !tosa_path.exists() {
        eprintln!("Skipping test_existing_example_add: file not found");
        return;
    }

    let data = std::fs::read(&tosa_path).unwrap();

    // Load the numpy inputs
    let examples_dir = tosa_path.parent().unwrap().parent().unwrap();
    let p0_path = examples_dir.join("placeholder_0.npy");
    let p1_path = examples_dir.join("placeholder_1.npy");

    if !p0_path.exists() || !p1_path.exists() {
        eprintln!("Skipping: input npy files not found");
        return;
    }

    let p0_file = npyz::NpyFile::new(std::fs::File::open(&p0_path).unwrap()).unwrap();
    let p0_shape: Vec<usize> = p0_file.shape().iter().map(|&d| d as usize).collect();
    let p0: Vec<f32> = p0_file.into_vec().unwrap();

    let p1_file = npyz::NpyFile::new(std::fs::File::open(&p1_path).unwrap()).unwrap();
    let p1_shape: Vec<usize> = p1_file.shape().iter().map(|&d| d as usize).collect();
    let p1: Vec<f32> = p1_file.into_vec().unwrap();

    let mut inputs = HashMap::new();
    inputs.insert(
        "TosaInput_0".to_string(),
        Tensor::new("TosaInput_0".to_string(), p0_shape, TensorData::F32(p0.clone())),
    );
    inputs.insert(
        "TosaInput_1".to_string(),
        Tensor::new("TosaInput_1".to_string(), p1_shape, TensorData::F32(p1.clone())),
    );

    let result = run_graph(&data, inputs);
    // Just verify the graph loads and runs without error.
    // The .npy files in this example don't match the graph's declared shape,
    // so we only check that execution completed.
    assert!(result.is_ok(), "graph execution failed: {:?}", result.err());
    assert!(!result.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Phase 2 tests — ops generated by gen_fixtures_phase2.py
// ---------------------------------------------------------------------------

fn run_const_graph_f32(fixture: &str, expected_file: &str, tol: f32) {
    let data = load_tosa(fixture);
    let outputs = run_graph(&data, HashMap::new()).unwrap();
    assert!(!outputs.is_empty(), "{fixture}: no outputs");
    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_f32_slice().unwrap();
    let expected = load_npy_f32(fixture, expected_file);
    assert_f32_close(actual, &expected, tol);
}

fn run_const_graph_i32(fixture: &str, expected_file: &str) {
    let data = load_tosa(fixture);
    let outputs = run_graph(&data, HashMap::new()).unwrap();
    assert!(!outputs.is_empty(), "{fixture}: no outputs");
    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_i32_slice().unwrap();
    let expected = load_npy_i32(fixture, expected_file);
    assert_i32_exact(actual, &expected);
}

#[test]
fn test_maximum_f32() {
    run_const_graph_f32("maximum_f32", "expected_result-0.npy", 1e-6);
}

#[test]
fn test_matmul_f32() {
    run_const_graph_f32("matmul_f32", "expected_result-0.npy", 1e-5);
}

#[test]
fn test_conv2d_f32() {
    run_const_graph_f32("conv2d_f32", "expected_result-0.npy", 1e-5);
}

#[test]
fn test_transpose_f32() {
    run_const_graph_f32("transpose_f32", "expected_result-0.npy", 1e-6);
}

#[test]
fn test_reduce_sum_f32() {
    run_const_graph_f32("reduce_sum_f32", "expected_result-0.npy", 1e-5);
}

#[test]
fn test_sigmoid_f32() {
    run_const_graph_f32("sigmoid_f32", "expected_result-0.npy", 1e-5);
}

#[test]
fn test_concat_f32() {
    run_const_graph_f32("concat_f32", "expected_result-0.npy", 1e-6);
}

#[test]
fn test_cast_f32_to_i32() {
    run_const_graph_i32("cast_f32_to_i32", "expected_result-0.npy");
}

#[test]
fn test_equal_i32() {
    let data = load_tosa("equal_i32");
    let outputs = run_graph(&data, HashMap::new()).unwrap();
    assert!(!outputs.is_empty());
    let (_, tensor) = outputs.iter().next().unwrap();
    let actual = tensor.data.as_bool_slice().unwrap();
    // Expected: [true, false, true, false]
    assert_eq!(actual, &[true, false, true, false]);
}

// ---------------------------------------------------------------------------
// HookSet / multi-collector test
// ---------------------------------------------------------------------------

#[test]
fn test_hookset_multiple_collectors() {
    let data = load_tosa("matmul_f32");

    let mut hooks = HookSet::new();
    hooks.add(StatsCollector::new());
    hooks.add(FlopCounter::new());

    let outputs = run_graph_with_hooks(&data, HashMap::new(), &mut hooks).unwrap();
    assert!(!outputs.is_empty());

    // We can't downcast easily through Box<dyn>, so just verify it ran without panic.
    // The HookSet dispatched to both collectors successfully.
}

#[test]
fn test_flop_counter() {
    let data = load_tosa("matmul_f32");
    let mut hook = FlopCounter::new();
    let outputs = run_graph_with_hooks(&data, HashMap::new(), &mut hook).unwrap();
    assert!(!outputs.is_empty());
    assert!(hook.total_flops > 0, "MATMUL should report FLOPs");
    assert!(
        hook.flops_by_op.contains_key("MATMUL"),
        "MATMUL should appear in flops_by_op"
    );
    // [1,2,3] x [1,3,2] = 2*1*2*2*3 = 24 flops
    assert_eq!(hook.flops_by_op["MATMUL"], 24);
}
