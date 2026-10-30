//! Every checked-in `.tosa` graph (the conformance and network cases of
//! tosa-interp) must survive parse -> serialize -> parse unchanged, which
//! exercises the serializer on every operator and attribute the reference
//! model emits.

use std::path::{Path, PathBuf};

use tosa_ir::{Graph, Version};

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "tosa") {
            out.push(p);
        }
    }
}

#[test]
fn every_checked_in_graph_round_trips() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tosa-interp/tests");
    let mut files = Vec::new();
    collect(&root.join("conformance"), &mut files);
    collect(&root.join("networks"), &mut files);
    assert!(files.len() > 3000, "found only {} graphs under {}", files.len(), root.display());

    let mut versions = std::collections::BTreeSet::new();
    for file in &files {
        let bytes = std::fs::read(file).unwrap();
        let graph = Graph::from_flatbuffer(&bytes).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        versions.insert(format!("{:?}", graph.version));
        let again = graph.to_flatbuffer().unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        let reparsed = Graph::from_flatbuffer(&again).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        assert_eq!(reparsed, graph, "{}", file.display());
        // Serializing the same IR twice is deterministic.
        assert_eq!(graph.to_flatbuffer().unwrap(), again, "{}", file.display());
    }
    // The generator writes the current version; the legalized TensorFlow
    // examples carry the older 1.0 draft.
    assert!(versions.contains(&format!("{:?}", Version::CURRENT)), "{versions:?}");
    assert!(versions.iter().all(|v| v.starts_with("Version { major: 1,")), "{versions:?}");
}
