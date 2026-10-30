#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-edge-litert>=1.0", "onnxruntime>=1.17", "numpy"]
# ///
"""Fetch well-known TFLite and ONNX models and record the native runtime's outputs.

Each model becomes a test case directory in the conformance format used by
the Rust tests (``desc.json`` naming the model, one ``.npy`` per input and
one ``ref-<output>.npy`` per output), under ``target/models/<name>/`` by
default. TFLite models are run with LiteRT's reference kernels, ONNX models
with onnxruntime's CPU provider. The conversion tests (``cargo test -p tosa-convert --test models``)
and the model benchmarks (``cargo bench -p tosa-convert``) pick them up from
there, or from ``TOSA_MODELS_DIR``; both skip models that are absent, so the
plain ``cargo test`` stays offline.

Downloads are pinned by SHA-256 and cached; nothing is re-fetched when the
file is already present and correct.

    uv run scripts/models.py                 # fetch everything, record outputs
    uv run scripts/models.py --only mobilenet_v1_1.0_224 efficientnet_lite0_int8
    uv run scripts/models.py --verify        # also convert with the tosa CLI and
                                             # run the TOSA reference model on it

``--verify`` needs the TOSA reference model (``tosa-tools``); run it as
``uv run --with tosa-tools==2026.8.0 scripts/models.py --verify``.

LiteRT wheels exist for Linux and macOS; on Windows, generate the cases on
another platform (they are plain files) or set ``TOSA_MODELS_DIR`` to a copy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "target" / "models"
CACHE = REPO / "target" / "models-download"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    url: str
    sha256: str
    """Hash of the downloaded file (archive or model)."""
    member: str | None = None
    """Path of the .tflite inside the archive, for .tgz downloads."""
    tflite_sha256: str | None = None
    """Hash of the extracted .tflite (defaults to ``sha256`` for direct downloads)."""
    note: str = ""
    max_int_diff: int = 0
    """Largest accepted difference from LiteRT for integer outputs (see docs/CONVERSION.md)."""


TF_MODELS = "https://storage.googleapis.com/download.tensorflow.org/models"
MEDIAPIPE = "https://storage.googleapis.com/mediapipe-models/image_classifier"

ONNX_ZOO = "https://github.com/onnx/models/raw/main/validated/vision/classification"

MODELS = [
    ModelSpec(
        "mobilenet_v1_1.0_224",
        f"{TF_MODELS}/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        "2fadeabb9968ec6833bee903900dda6e61b3947200535874ce2fe42a8493abc0",
        member="mobilenet_v1_1.0_224.tflite",
        tflite_sha256="d95b3c5ea86750cef882fa867ca357dfe4d265d0b80b67e83277a0bda310cfbb",
        note="float32 ImageNet classifier (TensorFlow hosted models)",
    ),
    ModelSpec(
        "mobilenet_v1_1.0_224_quant",
        f"{TF_MODELS}/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        "d32432d28673a936b2d6281ab0600c71cf7226dfe4cdcef3012555f691744166",
        member="mobilenet_v1_1.0_224_quant.tflite",
        tflite_sha256="ecc3a67c47c5a609ec35f6a58a7d97532834e43df4cb7d3f1204a8164b7d20dd",
        note="uint8 per-tensor quantized, uint8 SOFTMAX",
    ),
    ModelSpec(
        "mobilenet_v2_1.0_224",
        f"{TF_MODELS}/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
        "a9fce7e2db6389dfa1e640a9c98a6f29a55e482e463c5f01c377a19806f66ee2",
        member="mobilenet_v2_1.0_224.tflite",
        tflite_sha256="9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303",
        note="float32, with residual ADDs",
    ),
    ModelSpec(
        "mobilenet_v2_1.0_224_quant",
        f"{TF_MODELS}/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
        "d6a04d780f76f656c902413be432eb349ec4a458240e3739119eb44977f77a79",
        member="mobilenet_v2_1.0_224_quant.tflite",
        tflite_sha256="f08d447cde49b4e0446428aa921aff0a14ea589fa9c5817b31f83128e9a43c1d",
        note="uint8 per-tensor quantized, quantized ADDs",
    ),
    ModelSpec(
        "efficientnet_lite0_float32",
        f"{MEDIAPIPE}/efficientnet_lite0/float32/1/efficientnet_lite0.tflite",
        "6c7ab0a6e5dcbf38a8c33b960996a55a3b4300b36a018c4545801de3a3c8bde0",
        note="float32 (MediaPipe)",
    ),
    ModelSpec(
        "efficientnet_lite0_int8",
        f"{MEDIAPIPE}/efficientnet_lite0/int8/1/efficientnet_lite0.tflite",
        "bc2ffe19c1118de0c0c2a9088992da5589722656e0fba81421385300a4a34b16",
        note="int8 per-channel quantized with uint8 inputs and outputs (MediaPipe)",
    ),
    ModelSpec(
        "onnx_mobilenetv2-12",
        f"{ONNX_ZOO}/mobilenet/model/mobilenetv2-12.onnx",
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
        note="ONNX float32 (ONNX model zoo, opset 12)",
    ),
    ModelSpec(
        "onnx_squeezenet1.1-7",
        f"{ONNX_ZOO}/squeezenet/model/squeezenet1.1-7.onnx",
        "1eeff551a67ae8d565ca33b572fc4b66e3ef357b0eb2863bb9ff47a918cc4088",
        note="ONNX float32 (ONNX model zoo, opset 7)",
    ),
    ModelSpec(
        "onnx_resnet18-v1-7",
        f"{ONNX_ZOO}/resnet/model/resnet18-v1-7.onnx",
        "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c",
        note="ONNX float32 (ONNX model zoo, opset 7)",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    if dest.is_file() and sha256(dest) == expected:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    actual = sha256(tmp)
    if actual != expected:
        tmp.unlink()
        sys.exit(f"error: {url}: sha256 {actual} != expected {expected}")
    tmp.replace(dest)


def fetch(spec: ModelSpec, out: Path) -> Path:
    """Download (and extract) the model; returns the .tflite path in its case directory."""
    case = out / spec.name
    case.mkdir(parents=True, exist_ok=True)
    model = case / f"model{Path(spec.member or spec.url).suffix}"
    want = spec.tflite_sha256 or spec.sha256
    if model.is_file() and sha256(model) == want:
        return model
    if spec.member:
        archive = CACHE / Path(spec.url).name
        download(spec.url, archive, spec.sha256)
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.lstrip("./") == spec.member)
            data = tar.extractfile(member).read()
        model.write_bytes(data)
    else:
        download(spec.url, model, spec.sha256)
    if sha256(model) != want:
        sys.exit(f"error: {model}: unexpected contents")
    return model


def seeded_input(detail: dict, rng: np.random.Generator) -> np.ndarray:
    shape = [int(d) for d in detail["shape"]]
    dtype = detail["dtype"]
    if dtype == np.float32:
        # Classifiers expect roughly [-1, 1]; cover the range.
        return rng.uniform(-1.0, 1.0, shape).astype(np.float32)
    if dtype == np.uint8:
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == np.int8:
        return rng.integers(-128, 128, shape, dtype=np.int8)
    if dtype == np.int32:
        return rng.integers(-1000, 1000, shape, dtype=np.int32)
    if dtype == np.bool_:
        return rng.integers(0, 2, shape).astype(np.bool_)
    sys.exit(f"error: no input generator for dtype {dtype}")


def record_outputs(model: Path, case: Path, seed: int, meta: dict) -> dict:
    """Run the model's native runtime on seeded inputs and write the case files.
    Returns the desc."""
    if model.suffix == ".onnx":
        return record_onnx_outputs(model, case, seed, meta)
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    # TFLite's *reference* kernels are the oracle: they define the quantized
    # arithmetic the TOSA lowering follows. The optimized kernels (and the
    # XNNPACK delegate) differ from them by one unit in rare elements.
    interp = Interpreter(model_path=str(model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    rng = np.random.default_rng(seed)
    desc = {
        "tflite_file": model.name,
        "ifm_name": [],
        "ifm_file": [],
        "ofm_name": [],
        "ofm_file": [],
        "expected_failure": False,
        "meta": meta,
    }
    for i, d in enumerate(interp.get_input_details()):
        value = seeded_input(d, rng)
        interp.set_tensor(d["index"], value)
        file = f"input-{i}.npy"
        np.save(case / file, value)
        desc["ifm_name"].append(d["name"])
        desc["ifm_file"].append(file)
    interp.invoke()
    for i, d in enumerate(interp.get_output_details()):
        value = interp.get_tensor(d["index"])
        file = f"ref-output-{i}.npy"
        np.save(case / file, value)
        desc["ofm_name"].append(d["name"])
        desc["ofm_file"].append(file)
    (case / "desc.json").write_text(json.dumps(desc, indent=1) + "\n", newline="\n")
    return desc


ONNX_NP = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32,
           "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(bool)": np.bool_, "tensor(float16)": np.float16}


def record_onnx_outputs(model: Path, case: Path, seed: int, meta: dict) -> dict:
    """As ``record_outputs`` for ONNX, with onnxruntime's CPU provider. Symbolic
    input dimensions become 1, as the converter assumes."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    desc = {
        "onnx_file": model.name,
        "ifm_name": [],
        "ifm_file": [],
        "ofm_name": [],
        "ofm_file": [],
        "expected_failure": False,
        "meta": meta,
    }
    feeds = {}
    for i, inp in enumerate(session.get_inputs()):
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = ONNX_NP.get(inp.type)
        if dtype is None:
            sys.exit(f"error: no input generator for ONNX type {inp.type}")
        value = seeded_input({"shape": shape, "dtype": dtype}, rng)
        feeds[inp.name] = value
        file = f"input-{i}.npy"
        np.save(case / file, value)
        desc["ifm_name"].append(inp.name)
        desc["ifm_file"].append(file)
    outputs = session.run(None, feeds)
    for i, (out, value) in enumerate(zip(session.get_outputs(), outputs)):
        file = f"ref-output-{i}.npy"
        np.save(case / file, np.asarray(value))
        desc["ofm_name"].append(out.name)
        desc["ofm_file"].append(file)
    (case / "desc.json").write_text(json.dumps(desc, indent=1) + "\n", newline="\n")
    return desc


def verify(case: Path, desc: dict, max_int_diff: int) -> bool:
    """Convert with the tosa CLI and run the TOSA reference model on the result.

    The reference model reads inputs with the TOSA tensor's type, so uint8
    model inputs (which the converted graph takes as INT8 tensors holding the
    uint8 bits) are rewritten as int8 views.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_conformance import find_ref_model, run_ref_model  # noqa: E402

    tosa_file = case / "model.tosa"
    model_file = desc.get("tflite_file") or desc.get("onnx_file")
    cmd = ["cargo", "run", "-q", "--release", "-p", "tosa-cli", "--", "convert", str(case / model_file), "-o", str(tosa_file)]
    print("+", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=REPO).returncode != 0:
        print(f"{case.name}: conversion FAILED")
        return False
    ref_dir = case / "reference"
    ref_dir.mkdir(exist_ok=True)
    ref_desc = dict(desc, tosa_file="../model.tosa", ifm_file=[], ofm_file=[])
    for name, file in zip(desc["ifm_name"], desc["ifm_file"]):
        arr = np.load(case / file)
        if arr.dtype == np.uint8:
            arr = arr.view(np.int8)
        np.save(ref_dir / file, arr)
        ref_desc["ifm_file"].append(file)
    ref_desc["ofm_file"] = [f"tosa-{f}" for f in desc["ofm_file"]]
    ref_desc.pop("tflite_file", None)
    ref_desc.pop("onnx_file", None)
    (ref_dir / "desc.json").write_text(json.dumps(ref_desc, indent=1) + "\n", newline="\n")
    status = run_ref_model(find_ref_model(), ref_dir)
    if status != 0:
        print(f"{case.name}: reference model FAILED (exit code {status})")
        return False
    ok = True
    model_file = desc.get("tflite_file") or desc.get("onnx_file")
    for name, ref_file, tosa_file_ in zip(desc["ofm_name"], desc["ofm_file"], ref_desc["ofm_file"]):
        expected = np.load(case / ref_file)
        actual = np.load(ref_dir / tosa_file_)
        if expected.dtype == np.uint8:
            actual = actual.view(np.uint8)
        if expected.dtype.kind == "f":
            diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
            rel = diff / np.maximum(np.abs(expected.astype(np.float64)), 1e-30)
            print(f"{case.name}: {name}: max abs difference vs native runtime {diff.max():.3e}, rel {rel.max():.3e} (reference model)")
            # onnxruntime accumulates in a blocked order (float32 rounding differences); LiteRT reference kernels match closely.
            tol_abs, tol_rel = (1e-5, 1e-4) if model_file.endswith(".onnx") else (1e-6, 0.0)
            ok &= bool(np.all((diff <= tol_abs) | (rel <= tol_rel)))
        else:
            diff = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
            print(f"{case.name}: {name}: max |diff| {diff.max()}, {np.mean(diff == 0) * 100:.1f}% exact (reference model)")
            ok &= bool(diff.max() <= max_int_diff)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("TOSA_MODELS_DIR", DEFAULT_OUT)))
    parser.add_argument("--only", nargs="*", help="model names to process (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true", help="convert and run the TOSA reference model")
    parser.add_argument("--list", action="store_true", help="list the models and exit")
    args = parser.parse_args()

    specs = MODELS if not args.only else [m for m in MODELS if m.name in args.only]
    if args.only:
        missing = set(args.only) - {m.name for m in specs}
        if missing:
            sys.exit(f"error: unknown model(s): {', '.join(sorted(missing))}")
    if args.list:
        for m in MODELS:
            print(f"{m.name:<32} {m.note}")
        return

    failed = []
    for spec in specs:
        model = fetch(spec, args.out)
        meta = {"source_url": spec.url, "sha256": sha256(model), "note": spec.note, "seed": args.seed}
        desc = record_outputs(model, args.out / spec.name, args.seed, meta)
        print(f"{spec.name}: {len(desc['ifm_name'])} input(s), {len(desc['ofm_name'])} output(s) recorded")
        if args.verify and not verify(args.out / spec.name, desc, spec.max_int_diff):
            failed.append(spec.name)
    if failed:
        sys.exit(f"error: verification failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
