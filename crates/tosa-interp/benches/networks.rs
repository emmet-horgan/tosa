//! Release-mode throughput of the interpreter on the checked-in network
//! cases, so performance regressions in the operators or the scheduler show
//! up: `cargo bench`.

use std::collections::HashMap;
use std::path::Path;

use criterion::{Criterion, criterion_group, criterion_main};
use tosa_interp::{Graph, Interpreter, NoopHook, Tensor, npy};

#[derive(serde::Deserialize)]
struct Desc {
    tosa_file: String,
    ifm_name: Vec<String>,
    ifm_file: Vec<String>,
}

fn load(name: &str) -> (Vec<u8>, HashMap<String, Tensor>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/networks").join(name);
    let desc: Desc = serde_json::from_str(&std::fs::read_to_string(dir.join("desc.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir.join(&desc.tosa_file)).unwrap();
    let graph = Graph::from_flatbuffer(&bytes).unwrap();
    let block = graph.main_block().unwrap();
    let inputs = desc
        .ifm_name
        .iter()
        .zip(&desc.ifm_file)
        .map(|(name, file)| {
            let dtype = block.tensors.iter().find(|t| &t.name == name).unwrap().dtype;
            let arr = npy::read(&dir.join(file)).unwrap();
            let data = arr.to_tensor_data(dtype).unwrap();
            (name.clone(), Tensor::new(name.clone(), arr.shape, data))
        })
        .collect();
    (bytes, inputs)
}

fn networks(c: &mut Criterion) {
    for name in ["mobilenet_like", "conv_matmul", "tflite_conv2d_qi8", "tf_conv2d_f32"] {
        let (bytes, inputs) = load(name);
        c.bench_function(&format!("load/{name}"), |b| b.iter(|| Interpreter::from_flatbuffer(&bytes).unwrap()));
        let mut interp = Interpreter::from_flatbuffer(&bytes).unwrap();
        c.bench_function(&format!("run/{name}"), |b| b.iter(|| interp.run(inputs.clone(), &mut NoopHook).unwrap()));
    }
}

criterion_group!(benches, networks);
criterion_main!(benches);
