//! Well-known models fetched by `scripts/models.py` (TFLite: MobileNet v1
//! and v2, EfficientNet-Lite0, float and quantized; ONNX: MobileNet v2,
//! SqueezeNet 1.1, ResNet-18), converted and compared with the native
//! runtime's outputs. Skipped, with a message, when the models have not
//! been fetched, so `cargo test` works offline. Run in release mode:
//!
//! ```sh
//! uv run scripts/models.py
//! cargo test --release -p tosa-convert --test models -- --nocapture
//! ```

mod common;

use common::{Tolerance, check_case, load_cases, models_dir};

#[test]
fn well_known_models_match_litert() {
    if cfg!(debug_assertions) {
        eprintln!("the well-known models take minutes unoptimized; run with --release (skipping)");
        return;
    }
    let dir = models_dir();
    let cases = load_cases(&dir);
    if cases.is_empty() {
        eprintln!("no models under {}; run `uv run scripts/models.py` to fetch them (skipping)", dir.display());
        return;
    }
    let mut failures = Vec::new();
    for case in &cases {
        let start = std::time::Instant::now();
        failures.extend(check_case(case, &Tolerance::for_case(case)));
        println!("{}: {:.2?}", case.name, start.elapsed());
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
