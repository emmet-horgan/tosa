//! Data-driven equivalence tests.
//!
//! Scans `tests/fixtures/equiv_*` directories, each containing:
//!   - graph.tosa
//!   - meta.json (with input_names, output_names)
//!   - input_0.npy, input_1.npy, ...
//!   - expected_0.npy, expected_1.npy, ...
//!
//! Runs each graph through our interpreter and verifies output matches
//! the reference model's output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tosa_interp::{run_graph, Tensor, TensorData};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[derive(serde::Deserialize)]
struct Meta {
    input_names: Vec<String>,
    output_names: Vec<String>,
    #[allow(dead_code)]
    dtype: Option<String>,
    #[allow(dead_code)]
    op: Option<String>,
}

fn load_npy_as_tensor(path: &Path, name: &str) -> Tensor {
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(file).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let type_str = match npy.dtype() {
        npyz::DType::Plain(ts) => ts,
        other => panic!("unsupported npy DType: {:?}", other),
    };
    let type_char = type_str.type_char();
    let num_bytes = type_str.num_bytes().unwrap_or(0);

    let data = match (type_char, num_bytes) {
        (npyz::TypeChar::Float, 2) => {
            // f16 — npyz can't read f16 directly, read raw bytes and convert
            let raw = std::fs::read(path).unwrap();
            // Skip numpy header: find the \n after the header dict
            let header_end = raw.windows(1).enumerate()
                .filter(|(i, w)| *i >= 10 && w[0] == b'\n')
                .map(|(i, _)| i + 1)
                .next()
                .unwrap_or(128);
            let body = &raw[header_end..];
            let f16_data: Vec<half::f16> = body
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect();
            // Store as F32 for comparison (promote)
            TensorData::F32(f16_data.iter().map(|x| x.to_f32()).collect())
        }
        (npyz::TypeChar::Float, 4) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::F32(npy.into_vec::<f32>().unwrap())
        }
        (npyz::TypeChar::Float, 8) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::F64(npy.into_vec::<f64>().unwrap())
        }
        (npyz::TypeChar::Int, 1) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::I8(npy.into_vec::<i8>().unwrap())
        }
        (npyz::TypeChar::Int, 2) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::I16(npy.into_vec::<i16>().unwrap())
        }
        (npyz::TypeChar::Int, 4) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::I32(npy.into_vec::<i32>().unwrap())
        }
        (npyz::TypeChar::Int, 8) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            TensorData::I64(npy.into_vec::<i64>().unwrap())
        }
        (npyz::TypeChar::Uint, 1) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            let bytes: Vec<u8> = npy.into_vec::<u8>().unwrap();
            TensorData::Bool(bytes.iter().map(|&b| b != 0).collect())
        }
        (npyz::TypeChar::Bool, _) => {
            let npy = npyz::NpyFile::new(std::fs::File::open(path).unwrap()).unwrap();
            let bools: Vec<bool> = npy.into_vec::<bool>().unwrap();
            TensorData::Bool(bools)
        }
        _ => panic!(
            "unsupported npy dtype: type_char={:?} num_bytes={}",
            type_char, num_bytes
        ),
    };

    Tensor::new(name.to_string(), shape, data)
}

fn compare_tensors(actual: &Tensor, expected: &Tensor, fixture_name: &str, tol: f32) {
    assert_eq!(
        actual.data.len(),
        expected.data.len(),
        "{fixture_name}: element count mismatch: got {} expected {}",
        actual.data.len(),
        expected.data.len()
    );

    match (&actual.data, &expected.data) {
        (TensorData::F32(a), TensorData::F32(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                assert!(
                    (av - ev).abs() <= tol || (av.is_nan() && ev.is_nan()),
                    "{fixture_name}: element {i}: got {av} expected {ev} (diff {})",
                    (av - ev).abs()
                );
            }
        }
        // F16 actual vs F32 expected (npy loader promotes f16 to f32)
        (TensorData::F16(a), TensorData::F32(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                let av_f32 = av.to_f32();
                assert!(
                    (av_f32 - ev).abs() <= 1e-2 || (av_f32.is_nan() && ev.is_nan()),
                    "{fixture_name}: element {i}: got {av_f32} expected {ev} (diff {})",
                    (av_f32 - ev).abs()
                );
            }
        }
        (TensorData::I32(a), TensorData::I32(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                assert_eq!(av, ev, "{fixture_name}: element {i}: got {av} expected {ev}");
            }
        }
        (TensorData::I8(a), TensorData::I8(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                let diff = (*av as i32 - *ev as i32).abs();
                assert!(
                    diff <= 1,
                    "{fixture_name}: element {i}: got {av} expected {ev} (diff {diff} > 1)"
                );
            }
        }
        (TensorData::I16(a), TensorData::I16(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                assert_eq!(av, ev, "{fixture_name}: element {i}: got {av} expected {ev}");
            }
        }
        (TensorData::I64(a), TensorData::I64(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                assert_eq!(av, ev, "{fixture_name}: element {i}: got {av} expected {ev}");
            }
        }
        (TensorData::Bool(a), TensorData::Bool(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                assert_eq!(av, ev, "{fixture_name}: element {i}: got {av} expected {ev}");
            }
        }
        // BF16 actual vs F32 expected
        (TensorData::Bf16(a), TensorData::F32(e)) => {
            for (i, (av, ev)) in a.iter().zip(e.iter()).enumerate() {
                let av_f32 = av.to_f32();
                assert!(
                    (av_f32 - ev).abs() <= 1e-1 || (av_f32.is_nan() && ev.is_nan()),
                    "{fixture_name}: element {i}: got {av_f32} expected {ev}",
                );
            }
        }
        _ => {
            panic!(
                "{fixture_name}: dtype mismatch: actual {:?} vs expected {:?}",
                actual.data.dtype(),
                expected.data.dtype()
            );
        }
    }
}

fn run_single_fixture(fixture_dir: &Path) {
    let fixture_name = fixture_dir.file_name().unwrap().to_str().unwrap();

    let meta_path = fixture_dir.join("meta.json");
    if !meta_path.exists() {
        return; // Not an equivalence fixture
    }

    let meta: Meta =
        serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();

    let graph_data = std::fs::read(fixture_dir.join("graph.tosa")).unwrap();

    // Load inputs
    let mut inputs = HashMap::new();
    for (i, name) in meta.input_names.iter().enumerate() {
        let npy_path = fixture_dir.join(format!("input_{i}.npy"));
        if npy_path.exists() {
            inputs.insert(name.clone(), load_npy_as_tensor(&npy_path, name));
        }
    }

    // Run interpreter
    let outputs = run_graph(&graph_data, inputs)
        .unwrap_or_else(|e| panic!("{fixture_name}: interpreter error: {e}"));

    // Load and compare expected outputs
    for (i, name) in meta.output_names.iter().enumerate() {
        let expected_path = fixture_dir.join(format!("expected_{i}.npy"));
        if !expected_path.exists() {
            panic!("{fixture_name}: missing expected_{i}.npy");
        }

        let expected = load_npy_as_tensor(&expected_path, name);
        let actual = outputs
            .get(name)
            .unwrap_or_else(|| panic!("{fixture_name}: output '{name}' not found in results"));

        compare_tensors(actual, &expected, fixture_name, 1e-5);
    }
}

#[test]
fn test_all_equivalence_fixtures() {
    let dir = fixtures_dir();
    if !dir.exists() {
        eprintln!("Fixtures directory not found, skipping equivalence tests");
        return;
    }

    let mut fixture_dirs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir() && {
                let name = p.file_name().unwrap().to_str().unwrap();
                (name.starts_with("equiv_") || name.starts_with("network_") || name.starts_with("model_"))
                    && p.join("meta.json").exists()
            }
        })
        .collect();

    fixture_dirs.sort();

    if fixture_dirs.is_empty() {
        eprintln!("No equiv_* fixtures found. Run test_equivalence.py first.");
        return;
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut failures = Vec::new();

    for fixture_dir in &fixture_dirs {
        let name = fixture_dir.file_name().unwrap().to_str().unwrap();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_single_fixture(fixture_dir);
        })) {
            Ok(()) => {
                passed += 1;
            }
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "unknown panic".to_string()
                };
                failures.push(format!("{name}: {msg}"));
                failed += 1;
            }
        }
    }

    eprintln!(
        "\nEquivalence tests: {passed} passed, {failed} failed out of {}",
        fixture_dirs.len()
    );
    if !failures.is_empty() {
        eprintln!("\nFailures:");
        for f in &failures {
            eprintln!("  {f}");
        }
        panic!(
            "{failed} equivalence test(s) failed out of {}",
            fixture_dirs.len()
        );
    }
}
