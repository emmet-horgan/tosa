#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["ai-edge-litert>=1.0", "onnxruntime>=1.17", "numpy"]
# ///
"""Cross-runtime matrix: every well-known model through its native runtime,
tosa-interp and the TOSA reference model, with errors and timings.

For each model (see ``scripts/models.py``) and each seed, the same random
input is run through:

- the native runtime: LiteRT with reference kernels and LiteRT with its
  default optimized kernels for ``.tflite`` models, onnxruntime (CPU) for
  ``.onnx`` models;
- ``tosa-interp`` via the ``tosa`` CLI on the graph produced by
  ``tosa convert`` (conversion is timed too);
- the TOSA reference model (``tosa_reference_model`` from tosa-tools) on the
  same graph, when it is available.

Outputs are compared pairwise against the native reference run (max
absolute and relative difference for float outputs, max integer difference
and percentage of identical elements for quantized ones), and the worst case
over seeds is reported. Timings are the minimum and median over ``--runs``
repetitions. Results go to ``target/runtime-matrix/results.md`` and
``results.json``; the Markdown is also printed.

    uv run scripts/runtime_matrix.py                       # all models, 3 seeds, 10 runs
    uv run scripts/runtime_matrix.py --only mobilenet_v1_1.0_224_quant --seeds 5
    uv run --with tosa-tools==2026.8.0 scripts/runtime_matrix.py   # include the reference model

Reproducible by construction: the models are pinned by hash, the inputs
come from the seeds, and the package versions used are recorded in the JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MODELS, fetch, seeded_input  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "target" / "runtime-matrix"


def timed(fn, runs: int) -> tuple[object, list[float]]:
    """Call `fn` `runs` times; return its last result and the times in ms."""
    times = []
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return result, times


def summarize(times: list[float]) -> dict:
    return {"min_ms": min(times), "median_ms": statistics.median(times), "runs": len(times)}


def merge(summaries: list[dict]) -> dict:
    """Over seeds: the smallest minimum and the median of the medians."""
    return {"min_ms": min(s["min_ms"] for s in summaries),
            "median_ms": statistics.median(s["median_ms"] for s in summaries),
            "runs": sum(s["runs"] for s in summaries)}


def compare(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Error metrics of `actual` against `expected` (same dtype class)."""
    if expected.dtype.kind == "f":
        a = actual.astype(np.float64).ravel()
        e = expected.astype(np.float64).ravel()
        diff = np.abs(a - e)
        rel = diff / np.maximum(np.abs(e), 1e-30)
        return {"kind": "float", "max_abs": float(diff.max()), "max_rel": float(rel.max()),
                "exact_pct": float(np.mean(a == e) * 100)}
    a = actual.astype(np.int64).ravel()
    e = expected.astype(np.int64).ravel()
    diff = np.abs(a - e)
    return {"kind": "int", "max_int": int(diff.max()), "mean_abs": float(diff.mean()),
            "exact_pct": float(np.mean(diff == 0) * 100)}


def worst(metrics: list[dict]) -> dict:
    """The worst case over seeds."""
    if metrics[0]["kind"] == "float":
        return {"kind": "float", "max_abs": max(m["max_abs"] for m in metrics),
                "max_rel": max(m["max_rel"] for m in metrics), "exact_pct": min(m["exact_pct"] for m in metrics)}
    return {"kind": "int", "max_int": max(m["max_int"] for m in metrics),
            "mean_abs": max(m["mean_abs"] for m in metrics), "exact_pct": min(m["exact_pct"] for m in metrics)}


# ---------------------------------------------------------------------------
# Runtimes
# ---------------------------------------------------------------------------

class LiteRT:
    def __init__(self, model: Path, reference: bool):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType

        kwargs = {"experimental_op_resolver_type": OpResolverType.BUILTIN_REF} if reference else {}
        self.interp = Interpreter(model_path=str(model), **kwargs)
        self.interp.allocate_tensors()
        self.inputs = self.interp.get_input_details()
        self.outputs = self.interp.get_output_details()
        self.name = "litert-ref" if reference else "litert-opt"

    def input_specs(self):
        return [{"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": d["dtype"]} for d in self.inputs]

    def run(self, feeds: dict) -> dict:
        for d in self.inputs:
            self.interp.set_tensor(d["index"], feeds[d["name"]])
        self.interp.invoke()
        return {d["name"]: self.interp.get_tensor(d["index"]) for d in self.outputs}


class OnnxRuntime:
    name = "onnxruntime"
    NP = {"tensor(float)": np.float32, "tensor(int64)": np.int64, "tensor(int32)": np.int32,
          "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(bool)": np.bool_}

    def __init__(self, model: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])

    def input_specs(self):
        return [{"name": i.name, "shape": [d if isinstance(d, int) else 1 for d in i.shape], "dtype": self.NP[i.type]}
                for i in self.session.get_inputs()]

    def run(self, feeds: dict) -> dict:
        outs = self.session.run(None, feeds)
        return {o.name: np.asarray(v) for o, v in zip(self.session.get_outputs(), outs)}


def tosa_cli() -> Path:
    exe = REPO / "target" / "release" / ("tosa.exe" if os.name == "nt" else "tosa")
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "tosa-cli"], cwd=REPO, check=True)
    return exe


def convert(cli: Path, model: Path, out: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([str(cli), "convert", str(model), "-o", str(out)], check=True, stdout=subprocess.DEVNULL)
    return (time.perf_counter() - t0) * 1e3


def run_interp(cli: Path, graph: Path, inputs: dict[str, Path], out_dir: Path, runs: int) -> tuple[dict, dict]:
    cmd = [str(cli), "run", str(graph), "--repeat", str(runs), "--json", "--output-dir", str(out_dir)]
    for name, file in inputs.items():
        cmd += ["--input", f"{name}={file}"]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    summary = json.loads(proc.stdout)
    outputs = {o["name"]: np.load(o["file"]) for o in summary["outputs"]}
    return outputs, {"min_ms": summary["min_ms"], "median_ms": summary["median_ms"], "runs": runs}


def reference_model_exe() -> Path | None:
    try:
        from gen_conformance import find_ref_model
        return find_ref_model()
    except SystemExit:
        return None


def run_reference(exe: Path, graph: Path, inputs: dict[str, Path], native_outputs: dict, work: Path) -> tuple[dict, dict]:
    """Run the TOSA reference model; inputs holding uint8 are viewed as int8."""
    work.mkdir(parents=True, exist_ok=True)
    desc = {"tosa_file": str(graph), "ifm_name": [], "ifm_file": [], "ofm_name": list(native_outputs), "ofm_file": [],
            "expected_failure": False}
    for name, file in inputs.items():
        arr = np.load(file)
        if arr.dtype == np.uint8:
            arr = arr.view(np.int8)
        f = work / f"{name.replace('/', '_')}.npy"
        np.save(f, arr)
        desc["ifm_name"].append(name)
        desc["ifm_file"].append(f.name)
    desc["ofm_file"] = [f"ref-{i}.npy" for i in range(len(native_outputs))]
    (work / "desc.json").write_text(json.dumps(desc, indent=1))
    t0 = time.perf_counter()
    proc = subprocess.run([str(exe), "--test_desc", str(work / "desc.json"), "--tosa_level", "NONE"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = (time.perf_counter() - t0) * 1e3
    if proc.returncode != 0:
        raise RuntimeError(f"reference model failed with exit code {proc.returncode}")
    outputs = {}
    for name, f in zip(native_outputs, desc["ofm_file"]):
        arr = np.load(work / f)
        if native_outputs[name].dtype == np.uint8:
            arr = arr.view(np.uint8)
        outputs[name] = arr
    return outputs, {"min_ms": elapsed, "median_ms": elapsed, "runs": 1}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def evaluate(spec, args, cli: Path, ref_exe: Path | None) -> dict:
    model = fetch(spec, args.models_dir)
    work = OUT / spec.name
    work.mkdir(parents=True, exist_ok=True)
    graph = work / "model.tosa"
    convert_ms = convert(cli, model, graph)

    if model.suffix == ".tflite":
        native = LiteRT(model, reference=True)
        extra = [LiteRT(model, reference=False)]
    else:
        native = OnnxRuntime(model)
        extra = []
    specs = native.input_specs()

    timings: dict[str, list[dict]] = {}
    errors: dict[str, list[dict]] = {}
    for seed in range(args.seeds):
        rng = np.random.default_rng(1000 + seed)
        feeds = {s["name"]: seeded_input(s, rng) for s in specs}
        seed_dir = work / f"seed{seed}"
        seed_dir.mkdir(exist_ok=True)
        files = {}
        for name, value in feeds.items():
            f = seed_dir / f"in-{len(files)}.npy"
            np.save(f, value)
            files[name] = f

        base, t = timed(lambda: native.run(feeds), args.runs)
        timings.setdefault(native.name, []).append(summarize(t))
        results = {}
        for rt in extra:
            out, t = timed(lambda rt=rt: rt.run(feeds), args.runs)
            timings.setdefault(rt.name, []).append(summarize(t))
            results[rt.name] = out
        out, t = run_interp(cli, graph, files, seed_dir / "tosa-interp", args.runs)
        # The converted graph presents uint8 model outputs as INT8 bit patterns.
        out = {k: (v.view(np.uint8) if base[k].dtype == np.uint8 and v.dtype == np.int8 else v) for k, v in out.items()}
        timings.setdefault("tosa-interp", []).append(t)
        results["tosa-interp"] = out
        if ref_exe is not None:
            try:
                out, t = run_reference(ref_exe, graph, files, base, seed_dir / "tosa-ref")
                timings.setdefault("tosa-reference", []).append(t)
                results["tosa-reference"] = out
            except RuntimeError as e:
                print(f"{spec.name}: {e}", file=sys.stderr)

        for rt_name, out in results.items():
            for oname, value in base.items():
                errors.setdefault(f"{rt_name} vs {native.name}", []).append(compare(out[oname], value))
        if "tosa-reference" in results:
            for oname in base:
                errors.setdefault("tosa-interp vs tosa-reference", []).append(
                    compare(results["tosa-interp"][oname], results["tosa-reference"][oname]))

    timing_summary = {name: merge(per_seed) for name, per_seed in timings.items()}
    return {
        "model": spec.name,
        "file": model.name,
        "note": spec.note,
        "convert_ms": convert_ms,
        "tosa_operators": None,
        "timings": timing_summary,
        "errors": {k: worst(v) for k, v in errors.items()},
    }


def fmt_err(m: dict) -> str:
    if m["kind"] == "float":
        return f"max abs {m['max_abs']:.2e}, max rel {m['max_rel']:.2e}"
    return f"max {m['max_int']} unit(s), {m['exact_pct']:.2f}% exact"


def markdown(results: list[dict], args, versions: dict) -> str:
    lines = [
        "# Runtime matrix",
        "",
        f"{args.seeds} seed(s) per model, timings are min / median over {args.runs} runs per seed "
        f"(reference model: one run). Errors are the worst case over seeds, against the native "
        f"reference run (LiteRT reference kernels, or onnxruntime).",
        "",
        "Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
        "",
        "## Timings (ms)",
        "",
    ]
    runtimes = []
    for r in results:
        for k in r["timings"]:
            if k not in runtimes:
                runtimes.append(k)
    lines.append("| Model | convert | " + " | ".join(runtimes) + " |")
    lines.append("|---|---|" + "---|" * len(runtimes))
    for r in results:
        cells = []
        for k in runtimes:
            t = r["timings"].get(k)
            cells.append("" if t is None else f"{t['min_ms']:.1f} / {t['median_ms']:.1f}")
        lines.append(f"| {r['model']} | {r['convert_ms']:.0f} | " + " | ".join(cells) + " |")
    lines += ["", "## Errors", ""]
    pairs = []
    for r in results:
        for k in r["errors"]:
            if k not in pairs:
                pairs.append(k)
    lines.append("| Model | " + " | ".join(pairs) + " |")
    lines.append("|---|" + "---|" * len(pairs))
    for r in results:
        lines.append(f"| {r['model']} | " + " | ".join(fmt_err(r["errors"][k]) if k in r["errors"] else "" for k in pairs) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", help="model names (default: all)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--models-dir", type=Path, default=Path(os.environ.get("TOSA_MODELS_DIR", REPO / "target" / "models")))
    args = parser.parse_args()

    specs = MODELS if not args.only else [m for m in MODELS if m.name in args.only]
    missing = set(args.only or []) - {m.name for m in specs}
    if missing:
        sys.exit(f"error: unknown model(s): {', '.join(sorted(missing))}")
    cli = tosa_cli()
    ref_exe = reference_model_exe()
    if ref_exe is None:
        print("note: tosa_reference_model not found; run with `uv run --with tosa-tools==2026.8.0` to include it",
              file=sys.stderr)

    import ai_edge_litert
    import onnxruntime
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ai-edge-litert": getattr(ai_edge_litert, "__version__", "?"),
        "onnxruntime": onnxruntime.__version__,
        "tosa-tools": "2026.8.0" if ref_exe else "absent",
        "tosa-cli": subprocess.run([str(cli), "--version"], capture_output=True, text=True).stdout.strip(),
        "platform": platform.platform(),
    }
    results = [evaluate(spec, args, cli, ref_exe) for spec in specs]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps({"versions": versions, "args": vars(args) | {"models_dir": str(args.models_dir)}, "results": results}, indent=1, default=str))
    md = markdown(results, args, versions)
    (OUT / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
