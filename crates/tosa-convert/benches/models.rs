//! Conversion and interpretation throughput on the well-known models
//! fetched by `scripts/models.py` (release mode, criterion):
//!
//! ```sh
//! uv run scripts/models.py
//! cargo bench -p tosa-convert
//! ```
//!
//! Models that have not been fetched are skipped.

#[path = "../tests/common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use tosa_convert::{ConvertOptions, convert, default_frontends};
use tosa_interp::{Interpreter, NoopHook};

fn models(c: &mut Criterion) {
    let cases = common::load_cases(&common::models_dir());
    if cases.is_empty() {
        eprintln!("no models under {}; run `uv run scripts/models.py` (skipping)", common::models_dir().display());
        return;
    }
    let set = default_frontends();
    for case in &cases {
        let path = case.dir.join(&case.desc.model_file);
        let frontend = set.select(None, &path, &case.model).unwrap();
        c.bench_function(&format!("convert/{}", case.name), |b| {
            b.iter(|| convert(frontend, &case.model, &ConvertOptions::default()).unwrap())
        });
        let graph = convert(frontend, &case.model, &ConvertOptions::default()).unwrap().graph;
        let bytes = graph.to_flatbuffer().unwrap();
        let inputs = common::inputs(case, &graph);
        c.bench_function(&format!("load/{}", case.name), |b| b.iter(|| Interpreter::from_flatbuffer(&bytes).unwrap()));
        let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
        let mut group = c.benchmark_group("run");
        group.sample_size(10);
        group.bench_function(&case.name, |b| b.iter(|| interp.run(inputs.clone(), &mut NoopHook).unwrap()));
        group.finish();
    }
}

criterion_group!(benches, models);
criterion_main!(benches);
