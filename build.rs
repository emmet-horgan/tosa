use std::process::Command;

fn main() {
    let schema = "schema/tosa.fbs";
    let out_dir = std::env::var("OUT_DIR").unwrap();

    println!("cargo::rerun-if-changed={schema}");

    let status = Command::new("flatc")
        .args(["--rust", "-o", &out_dir, schema])
        .status()
        .expect("failed to run flatc — is it installed?");

    if !status.success() {
        panic!("flatc failed with status: {status}");
    }
}
