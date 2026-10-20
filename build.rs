use std::path::PathBuf;
use std::process::Command;

/// Locate `flatc`: `$FLATC`, then the repo's `.venv` (the tosa-tools wheel
/// ships one), then `PATH`.
fn flatc() -> PathBuf {
    println!("cargo::rerun-if-env-changed=FLATC");
    if let Some(p) = std::env::var_os("FLATC") {
        return PathBuf::from(p);
    }
    let venv = PathBuf::from(".venv");
    for candidate in [venv.join("Scripts").join("flatc.exe"), venv.join("bin").join("flatc")] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("flatc")
}

fn main() {
    let schema = "schema/tosa.fbs";
    let out_dir = std::env::var("OUT_DIR").unwrap();

    println!("cargo::rerun-if-changed={schema}");

    let flatc = flatc();
    let status = Command::new(&flatc)
        .args(["--rust", "-o", &out_dir, schema])
        .status()
        .unwrap_or_else(|e| {
            panic!("failed to run {} ({e}) — install tosa-tools into .venv or set FLATC", flatc.display())
        });

    if !status.success() {
        panic!("flatc failed with status: {status}");
    }
}
