//! Shared harness for TFLite case directories: `desc.json` names a
//! `.tflite` model, its inputs and LiteRT's outputs (see
//! `scripts/models.py` and `scripts/tflite_testcases.py`). A case is
//! converted, serialized, run with the interpreter and compared.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tosa_convert::{Conversion, ConvertOptions, convert, default_frontends};
use tosa_interp::{Interpreter, NoopHook, Tensor, TensorData};
use tosa_ir::npy::{self, NpyArray};
use tosa_ir::{DType, Graph};

#[derive(serde::Deserialize)]
pub struct Desc {
    /// The model file (`tflite_file` or `onnx_file` in `desc.json`).
    #[serde(alias = "tflite_file", alias = "onnx_file")]
    pub model_file: String,
    pub ifm_name: Vec<String>,
    pub ifm_file: Vec<String>,
    pub ofm_name: Vec<String>,
    pub ofm_file: Vec<String>,
    /// Looser float bounds for models whose reference arithmetic TOSA cannot
    /// reproduce bit for bit (recorded by `scripts/models.py`, with the
    /// reason in `meta.tolerance_note`).
    #[serde(default)]
    pub tolerance: Option<ToleranceOverride>,
}

#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct ToleranceOverride {
    pub max_abs: Option<f64>,
    pub rel: Option<f64>,
}

pub struct Case {
    pub name: String,
    pub dir: PathBuf,
    pub desc: Desc,
    pub model: Vec<u8>,
}

/// Every directory below `root` with a `desc.json` naming a `tflite_file`.
pub fn load_cases(root: &Path) -> Vec<Case> {
    let mut dirs: Vec<PathBuf> = match std::fs::read_dir(root) {
        Ok(entries) => entries.flatten().map(|e| e.path()).filter(|p| p.join("desc.json").is_file()).collect(),
        Err(_) => return Vec::new(),
    };
    dirs.sort();
    dirs.into_iter()
        .filter_map(|dir| {
            let text = std::fs::read_to_string(dir.join("desc.json")).ok()?;
            let desc: Desc = serde_json::from_str(&text).ok()?;
            let model = std::fs::read(dir.join(&desc.model_file)).ok()?;
            let name = dir.file_name().unwrap().to_string_lossy().into_owned();
            Some(Case { name, dir, desc, model })
        })
        .collect()
}

/// The checked-in small cases.
pub fn checked_in_cases() -> Vec<Case> {
    load_cases(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/tflite"))
}

/// Directory of the downloaded models: `TOSA_MODELS_DIR` or `target/models`.
pub fn models_dir() -> PathBuf {
    match std::env::var_os("TOSA_MODELS_DIR") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models"),
    }
}

pub fn convert_case(case: &Case) -> Conversion {
    try_convert_case(case).unwrap_or_else(|e| panic!("{}: {e}", case.name))
}

pub fn try_convert_case(case: &Case) -> Result<Conversion, String> {
    let set = default_frontends();
    let path = case.dir.join(&case.desc.model_file);
    let frontend = set.select(None, &path, &case.model).map_err(|e| e.to_string())?;
    convert(frontend, &case.model, &ConvertOptions::default()).map_err(|e| e.to_string())
}

/// The frontend a case uses, by its model file's extension.
pub fn frontend_name(case: &Case) -> String {
    let set = default_frontends();
    let path = case.dir.join(&case.desc.model_file);
    set.select(None, &path, &case.model).map(|f| f.name().to_string()).unwrap_or_default()
}

/// The case's inputs as tensors of the graph's declared types (uint8 files
/// feed INT8 inputs as their bit patterns).
pub fn inputs(case: &Case, graph: &Graph) -> HashMap<String, Tensor> {
    let block = graph.main_block().unwrap();
    let dtype = |name: &str| -> DType {
        block.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("{}: no tensor {name}", case.name)).dtype
    };
    case.desc
        .ifm_name
        .iter()
        .zip(&case.desc.ifm_file)
        .map(|(name, file)| {
            let arr = npy::read(&case.dir.join(file)).unwrap();
            let data = arr.to_tensor_data(dtype(name)).unwrap();
            (name.clone(), Tensor::new(name.clone(), arr.shape, data))
        })
        .collect()
}

pub fn run(graph: &Graph, inputs: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let bytes = graph.to_flatbuffer().unwrap();
    let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
    interp.run(inputs, &mut NoopHook).unwrap_or_else(|e| panic!("{e}"))
}

/// How an output differed from LiteRT's.
#[derive(Debug, Default, Clone)]
pub struct Deviation {
    pub elements: usize,
    pub max_abs: f64,
    /// Largest `|a - e| / max(|e|, 1e-30)` over float elements.
    pub max_rel: f64,
    pub max_int: i64,
    pub exact: usize,
    /// Float elements outside `max_abs + rel * |e|` of the tolerance.
    pub failing: usize,
}

impl Deviation {
    pub fn exact_fraction(&self) -> f64 {
        if self.elements == 0 { 1.0 } else { self.exact as f64 / self.elements as f64 }
    }
}

/// Compare an interpreter output with a LiteRT output file. uint8 LiteRT
/// outputs are compared with the INT8 output's bits read as unsigned.
pub fn compare(actual: &Tensor, expected: &NpyArray, tol: &Tolerance) -> Deviation {
    assert_eq!(actual.shape, expected.shape, "shape");
    let mut dev = Deviation { elements: actual.element_count(), ..Default::default() };
    let unsigned = expected.descr.kind == 'u';
    match &actual.data {
        TensorData::F32(v) => {
            let e = expected.to_tensor_data(DType::F32).unwrap();
            let TensorData::F32(e) = e else { unreachable!() };
            for (a, b) in v.iter().zip(&e) {
                if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
                    dev.exact += 1;
                    continue;
                }
                let diff = (*a as f64 - *b as f64).abs();
                dev.max_abs = dev.max_abs.max(diff);
                dev.max_rel = dev.max_rel.max(diff / (*b as f64).abs().max(1e-30));
                if diff > tol.max_abs + tol.rel * (*b as f64).abs() {
                    dev.failing += 1;
                }
            }
        }
        TensorData::I8(v) => {
            let e = expected.to_tensor_data(DType::I8).unwrap();
            let TensorData::I8(e) = e else { unreachable!() };
            for (a, b) in v.iter().zip(&e) {
                let (a, b) = if unsigned { (*a as u8 as i64, *b as u8 as i64) } else { (*a as i64, *b as i64) };
                if a == b {
                    dev.exact += 1;
                }
                dev.max_int = dev.max_int.max((a - b).abs());
            }
        }
        TensorData::I32(v) => {
            let e = expected.to_tensor_data(DType::I32).unwrap();
            let TensorData::I32(e) = e else { unreachable!() };
            for (a, b) in v.iter().zip(&e) {
                if a == b {
                    dev.exact += 1;
                }
                dev.max_int = dev.max_int.max((*a as i64 - *b as i64).abs());
            }
        }
        other => panic!("unsupported output dtype {:?}", other.dtype()),
    }
    dev
}

/// Tolerances: a float element passes when `|a - e| <= max_abs + rel * |e|`;
/// integer outputs must be within `max_int` units with at least `min_exact`
/// of the elements identical.
///
/// The defaults expect what the lowering delivers against TFLite's reference
/// kernels: bit-exact quantized results, and float results that differ only
/// through the transcendental functions (EXP in SOFTMAX).
pub struct Tolerance {
    pub max_abs: f64,
    pub rel: f64,
    pub max_int: i64,
    pub min_exact: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance { max_abs: 1e-6, rel: 0.0, max_int: 0, min_exact: 1.0 }
    }
}

impl Tolerance {
    /// For ONNX models checked against onnxruntime, whose GEMM and
    /// convolution kernels accumulate in a blocked order: results agree to
    /// float32 rounding, not bit for bit.
    pub fn onnxruntime() -> Self {
        Tolerance { max_abs: 1e-5, rel: 1e-4, max_int: 0, min_exact: 1.0 }
    }

    /// The tolerance for a case, by the frontend its model uses.
    pub fn for_case(case: &Case) -> Self {
        let mut tol = if frontend_name(case) == "onnx" { Tolerance::onnxruntime() } else { Tolerance::default() };
        if let Some(o) = &case.desc.tolerance {
            if let Some(a) = o.max_abs {
                tol.max_abs = a;
            }
            if let Some(r) = o.rel {
                tol.rel = r;
            }
        }
        tol
    }
}

/// Convert, run and compare one case; prints the deviations. Returns the
/// failures as messages.
pub fn check_case(case: &Case, tol: &Tolerance) -> Vec<String> {
    let conversion = match try_convert_case(case) {
        Ok(c) => c,
        Err(e) => return vec![format!("{}: conversion failed: {e}", case.name)],
    };
    let graph = &conversion.graph;
    let bytes = graph.to_flatbuffer().unwrap();
    let mut interp = match Interpreter::from_flatbuffer(&bytes) {
        Ok(i) => i,
        Err(e) => return vec![format!("{}: converted graph rejected: {e}", case.name)],
    };
    let outputs = match interp.run(inputs(case, graph), &mut NoopHook) {
        Ok(o) => o,
        Err(e) => return vec![format!("{}: interpreter error: {e}", case.name)],
    };
    let mut failures = Vec::new();
    for (name, file) in case.desc.ofm_name.iter().zip(&case.desc.ofm_file) {
        let Some(actual) = outputs.get(name) else {
            failures.push(format!("{}: output '{name}' missing", case.name));
            continue;
        };
        let expected = npy::read(&case.dir.join(file)).unwrap();
        let dev = compare(actual, &expected, tol);
        let ops = graph.main_block().unwrap().operators.len();
        println!(
            "{}: {name}: {} elements, {:.1}% exact, max abs {:.3e}, max rel {:.3e}, max int {} ({ops} TOSA operators)",
            case.name,
            dev.elements,
            dev.exact_fraction() * 100.0,
            dev.max_abs,
            dev.max_rel,
            dev.max_int
        );
        if dev.failing > 0 {
            failures.push(format!(
                "{}: {name}: {} elements differ by more than {:.1e} + {:.1e} x |expected| (max abs {:.3e}, max rel {:.3e})",
                case.name, dev.failing, tol.max_abs, tol.rel, dev.max_abs, dev.max_rel
            ));
        }
        if dev.max_int > tol.max_int {
            failures.push(format!("{}: {name}: max integer difference {} > {}", case.name, dev.max_int, tol.max_int));
        }
        if matches!(actual.data, TensorData::I8(_) | TensorData::I32(_)) && dev.exact_fraction() < tol.min_exact {
            failures.push(format!("{}: {name}: only {:.1}% exact", case.name, dev.exact_fraction() * 100.0));
        }
    }
    failures
}
