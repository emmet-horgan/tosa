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
//! exceptions are operators the reference model evaluates with the host C
//! library's transcendental functions (ERF, LOG, POW, SIN, COS, FFT2D,
//! RFFT2D): their reference outputs differ by a few ULPs between Windows,
//! Linux and macOS, so a small, spec-sized error is accepted (see
//! [`tolerance`]). Set `TOSA_CONFORMANCE_STRICT=1` to require bit-exact
//! results everywhere, and `TOSA_CONFORMANCE_VERBOSE=1` to print the largest
//! deviation of every inexact output.
//!
//! `tests/networks/` holds multi-operator graphs (small networks and models
//! converted by TensorFlow / TFLite legalization) in the same format; their
//! tests are named `networks::<case>` and must match exactly. Refresh their
//! reference outputs with `scripts/refresh_networks.py`.
//!
//! Set `TOSA_CONFORMANCE_DIR` (a list separated like `PATH`) to run test
//! cases from other directories instead, e.g. the full suite generated with
//! `scripts/gen_conformance.py --full` into `target/conformance-full`.
//! Regenerate the checked-in cases with `scripts/gen_conformance.py`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use libtest_mimic::{Arguments, Failed, Trial};
use tosa_interp::{DType, Graph, Interpreter, NoopHook, Op, Tensor, TensorData, npy};

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

/// How closely an output must match the reference model.
#[derive(Clone, Copy, Debug)]
enum Tolerance {
    /// Bit-exact (any two NaNs compare equal).
    Exact,
    /// Within this many units in the last place of the expected value.
    Ulp(u64),
    /// `|actual - expected| <= bound`.
    Abs(f64),
}

/// Operators whose reference result depends on the host C library's
/// transcendental functions (`erf`, `logf`, `sinf`, `cosf`), which differ by a
/// few ULPs between platforms (glibc, Apple libm and the MSVC CRT). The
/// interpreter uses the portable `libm` crate instead, so it cannot match every
/// platform bit for bit; the specification only requires these to be within
/// an error bound of a few ULPs. The bound is the larger of the one in the
/// test's compliance meta (ULP mode) and the figure below. EXP, SIGMOID and
/// TANH are not listed: the reference bundles its own `expf`, which the
/// interpreter reproduces exactly.
const ULP_OPS: &[(&str, u64)] = &[("ERF", 5), ("LOG", 2), ("POW", 2), ("SIN", 2), ("COS", 2)];

/// FFT outputs are sums of products with twiddle factors; the allowed error
/// scales with the magnitude of the inputs (2^-20 x sum of |x|).
const FFT_REL_EPS: f64 = 1.0 / (1u64 << 20) as f64;

/// The single non-constant operator of a test graph, if there is exactly one.
fn tested_op(graph: &Graph) -> Option<&'static str> {
    let block = graph.main_block()?;
    let mut ops = block.operators.iter().filter(|o| !matches!(o.op, Op::Const | Op::ConstShape));
    let op = ops.next()?;
    ops.next().is_none().then(|| op.op.name())
}

/// Tolerance for output `name` of a single-operator test of `op`; graphs with
/// several operators (e.g. networks) must match exactly.
fn tolerance(desc: &Desc, op: Option<&str>, name: &str, input_abs_sum: f64) -> Tolerance {
    if std::env::var_os("TOSA_CONFORMANCE_STRICT").is_some() {
        return Tolerance::Exact;
    }
    match op {
        Some("FFT2D" | "RFFT2D") => Tolerance::Abs(FFT_REL_EPS * input_abs_sum),
        Some(op) => match ULP_OPS.iter().find(|(o, _)| *o == op) {
            Some(&(_, default)) => {
                let c = &desc.meta["compliance"]["tensors"][name];
                let meta = (c["mode"] == "ULP").then(|| c["ulp_info"]["ulp"].as_f64()).flatten();
                Tolerance::Ulp(meta.map_or(default, |u| (u.floor() as u64).max(default)))
            }
            None => Tolerance::Exact,
        },
        None => Tolerance::Exact,
    }
}

/// Sum of the absolute values of a floating-point tensor (0 for other types).
fn abs_sum(data: &TensorData) -> f64 {
    use TensorData as T;
    match data {
        T::F16(v) => v.iter().map(|x| f64::from(x.to_f32()).abs()).sum(),
        T::Bf16(v) => v.iter().map(|x| f64::from(x.to_f32()).abs()).sum(),
        T::F32(v) => v.iter().map(|x| f64::from(*x).abs()).sum(),
        T::F64(v) => v.iter().map(|x| x.abs()).sum(),
        _ => 0.0,
    }
}

/// Distance in representable values between two floats given as sign-magnitude bits.
fn ulp_distance(a: u64, b: u64, sign_bit: u64) -> u64 {
    let ordinal = |x: u64| -> i128 { if x & sign_bit != 0 { -((x & !sign_bit) as i128) } else { x as i128 } };
    (ordinal(a) - ordinal(b)).unsigned_abs() as u64
}

/// Directories to search for test cases, each with a prefix for the test names.
fn roots() -> Vec<(PathBuf, &'static str)> {
    match std::env::var("TOSA_CONFORMANCE_DIR") {
        Ok(v) => std::env::split_paths(&v).map(|p| (p, "")).collect(),
        Err(_) => {
            let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
            vec![(tests.join("conformance"), ""), (tests.join("networks"), "networks::")]
        }
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
    graph.main_block().map(|b| b.tensors.iter().map(|t| (t.name.as_str(), t.dtype)).collect()).unwrap_or_default()
}

/// Largest deviation seen while comparing a tensor (for `TOSA_CONFORMANCE_VERBOSE`).
#[derive(Default)]
struct Deviation {
    ulps: u64,
    abs: f64,
}

/// One floating-point element: raw bits, its format's sign bit and its value.
struct Fp {
    bits: u64,
    sign: u64,
    value: f64,
}

/// Element-wise comparison; returns a description of the first mismatches.
fn compare(
    actual: &TensorData,
    expected: &TensorData,
    tol: Tolerance,
    dev: &mut Deviation,
) -> std::result::Result<(), String> {
    fn diff<T: std::fmt::Debug>(
        a: &[T],
        e: &[T],
        mut same: impl FnMut(&T, &T) -> bool,
    ) -> std::result::Result<(), String> {
        if a.len() != e.len() {
            return Err(format!("element count {} != expected {}", a.len(), e.len()));
        }
        let bad: Vec<usize> = (0..a.len()).filter(|&i| !same(&a[i], &e[i])).collect();
        if bad.is_empty() {
            return Ok(());
        }
        let shown: Vec<String> =
            bad.iter().take(8).map(|&i| format!("  [{i}] got {:?}, expected {:?}", a[i], e[i])).collect();
        Err(format!("{} of {} elements differ:\n{}", bad.len(), a.len(), shown.join("\n")))
    }
    let mut close = |a: Fp, e: Fp| -> bool {
        let (a_nan, e_nan) = (a.value.is_nan(), e.value.is_nan());
        if a.bits == e.bits || (a_nan && e_nan) {
            return true;
        }
        if a_nan || e_nan {
            return false;
        }
        let ulps = ulp_distance(a.bits, e.bits, a.sign);
        let abs = (a.value - e.value).abs();
        dev.ulps = dev.ulps.max(ulps);
        dev.abs = dev.abs.max(abs);
        match tol {
            Tolerance::Exact => false,
            Tolerance::Ulp(n) => ulps <= n,
            Tolerance::Abs(bound) => abs <= bound,
        }
    };
    use TensorData as T;
    match (actual, expected) {
        (T::Bool(a), T::Bool(e)) => diff(a, e, PartialEq::eq),
        (T::I4(a), T::I4(e)) | (T::I8(a), T::I8(e)) => diff(a, e, PartialEq::eq),
        (T::I16(a), T::I16(e)) => diff(a, e, PartialEq::eq),
        (T::I32(a), T::I32(e)) => diff(a, e, PartialEq::eq),
        (T::I48(a), T::I48(e)) | (T::I64(a), T::I64(e)) => diff(a, e, PartialEq::eq),
        (T::Fp8E4M3(a), T::Fp8E4M3(e)) => diff(a, e, |x, y| x == y || (x.is_nan() && y.is_nan())),
        (T::Fp8E5M2(a), T::Fp8E5M2(e)) => diff(a, e, |x, y| x == y || (x.is_nan() && y.is_nan())),
        (T::F16(a), T::F16(e)) => diff(a, e, |x, y| {
            let fp = |v: &half::f16| Fp { bits: v.to_bits().into(), sign: 0x8000, value: v.to_f64() };
            close(fp(x), fp(y))
        }),
        (T::Bf16(a), T::Bf16(e)) => diff(a, e, |x, y| {
            let fp = |v: &half::bf16| Fp { bits: v.to_bits().into(), sign: 0x8000, value: v.to_f64() };
            close(fp(x), fp(y))
        }),
        (T::F32(a), T::F32(e)) => diff(a, e, |x, y| {
            let fp = |v: &f32| Fp { bits: v.to_bits().into(), sign: 0x8000_0000, value: f64::from(*v) };
            close(fp(x), fp(y))
        }),
        (T::F64(a), T::F64(e)) => diff(a, e, |x, y| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())),
        _ => Err(format!("dtype {:?} != expected {:?}", actual.dtype(), expected.dtype())),
    }
}

fn run_case(dir: &Path) -> std::result::Result<(), Failed> {
    let desc: Desc = serde_json::from_str(&std::fs::read_to_string(dir.join("desc.json"))?)?;
    let graph_bytes = std::fs::read(dir.join(&desc.tosa_file))?;

    let mut input_abs_sum = 0.0;
    let mut run = || -> tosa_interp::Result<HashMap<String, Tensor>> {
        let interp = Interpreter::from_flatbuffer(&graph_bytes)?;
        let dtypes = tensor_dtypes(interp.graph());
        let mut inputs = HashMap::new();
        for (name, file) in desc.ifm_name.iter().zip(&desc.ifm_file) {
            let dtype = *dtypes.get(name.as_str()).ok_or_else(|| tosa_interp::Error::TensorNotFound(name.clone()))?;
            let arr = npy::read(&dir.join(file))?;
            let data = arr.to_tensor_data(dtype)?;
            input_abs_sum += abs_sum(&data);
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
    let op = tested_op(&graph);
    let verbose = std::env::var_os("TOSA_CONFORMANCE_VERBOSE").is_some();
    for (name, file) in desc.ofm_name.iter().zip(&desc.ofm_file) {
        let actual = outputs.get(name).ok_or_else(|| format!("output {name} was not produced"))?;
        let dtype = dtypes[name.as_str()];
        let arr = npy::read(&dir.join(file))?;
        if actual.shape != arr.shape {
            return Err(format!("{name}: shape {:?} != expected {:?}", actual.shape, arr.shape).into());
        }
        let expected = arr.to_tensor_data(dtype)?;
        let tol = tolerance(&desc, op, name, input_abs_sum);
        let mut dev = Deviation::default();
        let res = compare(&actual.data, &expected, tol, &mut dev);
        if verbose && (dev.ulps > 0 || dev.abs > 0.0) {
            let rel = if input_abs_sum > 0.0 { dev.abs / input_abs_sum } else { 0.0 };
            eprintln!(
                "{}: {name}: max deviation {} ulp, {:e} abs ({:e} x sum|inputs|), allowed {tol:?}",
                dir.display(),
                dev.ulps,
                dev.abs,
                rel
            );
        }
        res.map_err(|m| format!("{name}: {m} (allowed {tol:?})"))?;
    }
    Ok(())
}

fn main() {
    let args = Arguments::from_args();
    let mut cases = Vec::new();
    for (root, prefix) in roots() {
        let mut found = Vec::new();
        collect_cases(&root, &mut found);
        cases.extend(found.into_iter().map(|p| {
            let name = prefix.to_string()
                + &p.strip_prefix(&root)
                    .unwrap_or(&p)
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("::");
            (name, p)
        }));
    }
    let trials = cases.into_iter().map(|(name, dir)| Trial::test(name, move || run_case(&dir))).collect();
    libtest_mimic::run(&args, trials).exit();
}
