//! Conformance tests against the TOSA reference model.
//!
//! Every directory below `tests/conformance/` that contains a `desc.json`
//! is one test case in the format produced by tosa-tools'
//! `tosa_verif_build_tests`: a `.tosa` flatbuffer, one `.npy` per graph input
//! and, for positive tests, one `.npy` per output holding the reference
//! model's result. Negative tests (`"expected_failure": true`) must be
//! rejected by the interpreter.
//!
//! Outputs are compared bit for bit; any two NaNs are considered equal. The
//! one exception is a tensor whose compliance mode (in the test's `meta`) is
//! `ULP` with a bound of at least one ULP: the specification then allows
//! implementations to differ, and the reference model itself uses the
//! platform's libm (e.g. ERF), so a difference of up to that many ULPs from
//! the reference output is accepted. Set `TOSA_CONFORMANCE_STRICT=1` to
//! require bit-exact results everywhere.
//!
//! Set `TOSA_CONFORMANCE_DIR` (a list separated like `PATH`) to run test
//! cases from other directories instead, e.g. the full suite generated with
//! `scripts/gen_conformance.py --full` into `target/conformance-full`.
//! Regenerate the checked-in cases with `scripts/gen_conformance.py`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use libtest_mimic::{Arguments, Failed, Trial};
use tosa_interp::{DType, Graph, Interpreter, NoopHook, Tensor, TensorData, npy};

#[derive(serde::Deserialize)]
struct Desc {
    tosa_file: String,
    ifm_name: Vec<String>,
    ifm_file: Vec<String>,
    ofm_name: Vec<String>,
    ofm_file: Vec<String>,
    #[serde(default)]
    expected_failure: bool,
    #[serde(default)]
    meta: serde_json::Value,
}

/// Accepted ULP difference for output tensor `name` (0 = bit-exact).
fn ulp_tolerance(desc: &Desc, name: &str) -> u64 {
    if std::env::var_os("TOSA_CONFORMANCE_STRICT").is_some() {
        return 0;
    }
    let c = &desc.meta["compliance"]["tensors"][name];
    if c["mode"] != "ULP" {
        return 0;
    }
    c["ulp_info"]["ulp"].as_f64().map_or(0, |u| u.floor() as u64)
}

/// Distance in representable values between two floats given as sign-magnitude bits.
fn ulp_distance(a: u64, b: u64, sign_bit: u64) -> u64 {
    let ordinal = |x: u64| -> i128 {
        if x & sign_bit != 0 { -((x & !sign_bit) as i128) } else { x as i128 }
    };
    (ordinal(a) - ordinal(b)).unsigned_abs() as u64
}

fn roots() -> Vec<PathBuf> {
    match std::env::var("TOSA_CONFORMANCE_DIR") {
        Ok(v) => std::env::split_paths(&v).collect(),
        Err(_) => vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/conformance")],
    }
}

fn collect_cases(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.join("desc.json").is_file() {
                out.push(p);
            } else {
                collect_cases(&p, out);
            }
        }
    }
}

fn tensor_dtypes(graph: &Graph) -> HashMap<&str, DType> {
    graph
        .main_block()
        .map(|b| b.tensors.iter().map(|t| (t.name.as_str(), t.dtype)).collect())
        .unwrap_or_default()
}

/// Element-wise bit comparison; returns a description of the first mismatches.
fn compare(actual: &TensorData, expected: &TensorData, ulps: u64) -> std::result::Result<(), String> {
    fn diff<T: PartialEq + std::fmt::Debug>(
        a: &[T],
        e: &[T],
        same: impl Fn(&T, &T) -> bool,
    ) -> std::result::Result<(), String> {
        if a.len() != e.len() {
            return Err(format!("element count {} != expected {}", a.len(), e.len()));
        }
        let bad: Vec<usize> = (0..a.len()).filter(|&i| !same(&a[i], &e[i])).collect();
        if bad.is_empty() {
            return Ok(());
        }
        let shown: Vec<String> = bad
            .iter()
            .take(8)
            .map(|&i| format!("  [{i}] got {:?}, expected {:?}", a[i], e[i]))
            .collect();
        Err(format!("{} of {} elements differ:\n{}", bad.len(), a.len(), shown.join("\n")))
    }
    // Equal bits, both NaN, or (when allowed) within `ulps` of each other.
    let close = |a: u64, b: u64, a_nan: bool, b_nan: bool, sign: u64| {
        a == b || (a_nan && b_nan) || (ulps > 0 && !a_nan && !b_nan && ulp_distance(a, b, sign) <= ulps)
    };
    use TensorData as T;
    match (actual, expected) {
        (T::Bool(a), T::Bool(e)) => diff(a, e, PartialEq::eq),
        (T::I4(a), T::I4(e)) | (T::I8(a), T::I8(e)) => diff(a, e, PartialEq::eq),
        (T::I16(a), T::I16(e)) => diff(a, e, PartialEq::eq),
        (T::I32(a), T::I32(e)) => diff(a, e, PartialEq::eq),
        (T::I48(a), T::I48(e)) | (T::I64(a), T::I64(e)) => diff(a, e, PartialEq::eq),
        (T::Fp8E4M3(a), T::Fp8E4M3(e)) => {
            diff(a, e, |x, y| x == y || (x.is_nan() && y.is_nan()))
        }
        (T::Fp8E5M2(a), T::Fp8E5M2(e)) => {
            diff(a, e, |x, y| x == y || (x.is_nan() && y.is_nan()))
        }
        (T::F16(a), T::F16(e)) => {
            diff(a, e, |x, y| close(x.to_bits() as u64, y.to_bits() as u64, x.is_nan(), y.is_nan(), 0x8000))
        }
        (T::Bf16(a), T::Bf16(e)) => {
            diff(a, e, |x, y| close(x.to_bits() as u64, y.to_bits() as u64, x.is_nan(), y.is_nan(), 0x8000))
        }
        (T::F32(a), T::F32(e)) => {
            diff(a, e, |x, y| close(x.to_bits() as u64, y.to_bits() as u64, x.is_nan(), y.is_nan(), 0x8000_0000))
        }
        (T::F64(a), T::F64(e)) => {
            diff(a, e, |x, y| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
        }
        _ => Err(format!("dtype {:?} != expected {:?}", actual.dtype(), expected.dtype())),
    }
}

fn run_case(dir: &Path) -> std::result::Result<(), Failed> {
    let desc: Desc = serde_json::from_str(&std::fs::read_to_string(dir.join("desc.json"))?)?;
    let graph_bytes = std::fs::read(dir.join(&desc.tosa_file))?;

    let run = || -> tosa_interp::Result<HashMap<String, Tensor>> {
        let interp = Interpreter::from_flatbuffer(&graph_bytes)?;
        let dtypes = tensor_dtypes(interp.graph());
        let mut inputs = HashMap::new();
        for (name, file) in desc.ifm_name.iter().zip(&desc.ifm_file) {
            let dtype = *dtypes
                .get(name.as_str())
                .ok_or_else(|| tosa_interp::Error::TensorNotFound(name.clone()))?;
            let arr = npy::read(&dir.join(file))?;
            let data = arr.to_tensor_data(dtype)?;
            inputs.insert(name.clone(), Tensor::new(name.clone(), arr.shape, data));
        }
        interp.run(inputs, &mut NoopHook)
    };
    let result = run();

    if desc.expected_failure {
        return match result {
            Ok(_) => Err("expected the graph to be rejected, but it ran successfully".into()),
            Err(e) => {
                if std::env::var_os("TOSA_CONFORMANCE_VERBOSE").is_some() {
                    eprintln!("{}: rejected: {e}", dir.display());
                }
                Ok(())
            }
        };
    }
    let outputs = result.map_err(|e| format!("interpreter error: {e}"))?;

    let graph = Graph::from_flatbuffer(&graph_bytes)?;
    let dtypes = tensor_dtypes(&graph);
    for (name, file) in desc.ofm_name.iter().zip(&desc.ofm_file) {
        let actual = outputs.get(name).ok_or_else(|| format!("output {name} was not produced"))?;
        let dtype = dtypes[name.as_str()];
        let arr = npy::read(&dir.join(file))?;
        if actual.shape != arr.shape {
            return Err(format!("{name}: shape {:?} != expected {:?}", actual.shape, arr.shape).into());
        }
        let expected = arr.to_tensor_data(dtype)?;
        compare(&actual.data, &expected, ulp_tolerance(&desc, name)).map_err(|m| format!("{name}: {m}"))?;
    }
    Ok(())
}

fn main() {
    let args = Arguments::from_args();
    let mut cases = Vec::new();
    for root in roots() {
        let mut found = Vec::new();
        collect_cases(&root, &mut found);
        cases.extend(found.into_iter().map(|p| {
            let name = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("::");
            (name, p)
        }));
    }
    let trials = cases
        .into_iter()
        .map(|(name, dir)| Trial::test(name, move || run_case(&dir)))
        .collect();
    libtest_mimic::run(&args, trials).exit();
}
