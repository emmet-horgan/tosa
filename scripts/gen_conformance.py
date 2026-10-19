#!/usr/bin/env python3
"""Generate conformance test cases for tosa-interp from the TOSA reference model.

For each requested operator this script:

1. Runs tosa-tools' ``tosa_verif_build_tests`` to build test graphs and input
   data (positive tests and, optionally, ERROR_IF negative tests).
2. Runs the reference model executable on every positive test in its
   *native-precision* mode (no ``--precise_mode``), so the recorded outputs are
   exactly what the reference implementation produces and can be compared bit
   for bit.
3. Copies each test case into ``tests/conformance/<op>/<test>/``.

Tests the reference model reports as UNPREDICTABLE are dropped, as are
positive tests it fails to run. MX (microscaling) extension tests are not
generated. The generator produces thousands of cases per operator, so by
default a sample is kept (see ``--max-positive`` / ``--max-negative``);
``--full`` keeps every case, writing to the untracked ``target/conformance-full``.

Run it with a Python environment that has the ``tosa-tools`` wheel installed
(it provides the generator, the reference model executable and its data
generation library), e.g. from the repository root:

    uv venv .venv && uv pip install --python .venv tosa-tools
    .venv/bin/python scripts/gen_conformance.py --op conv2d        # Linux/macOS
    .venv\\Scripts\\python scripts\\gen_conformance.py --op conv2d   # Windows

Set ``TOSA_REF_MODEL`` / ``TOSA_GENERATE_LIB`` to override the binaries used.

Each ``--op`` value names a generator operator (e.g. ``conv2d``) and is
also the output directory name; ``OP_FILTERS`` can map it to a different
generator filter regex.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import random
import re
import subprocess
import sys
import sysconfig
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "conformance"
# Untracked location for complete (unsampled) suites; run them with
# TOSA_CONFORMANCE_DIR=target/conformance-full cargo test --test conformance
FULL_OUT = REPO / "target" / "conformance-full"
# Raw generator output, cached so re-sampling does not rerun the generator.
GEN_CACHE = REPO / "target" / "conformance-gen"

# Generator operator-name regexes, for operators whose generator name differs
# from the name used for the output directory. Defaults to ``^<op>$``.
OP_FILTERS: dict[str, str] = {}

# Every extension except the MX (microscaling) ones, which tosa-interp does
# not implement. Tests needing an unlisted extension are not generated.
EXTENSIONS = [
    "tosa-ext-bf16",
    "tosa-ext-controlflow",
    "tosa-ext-doubleround",
    "tosa-ext-dynamic",
    "tosa-ext-fft",
    "tosa-ext-fp8e4m3",
    "tosa-ext-fp8e5m2",
    "tosa-ext-inexactround",
    "tosa-ext-int16",
    "tosa-ext-int4",
    "tosa-ext-variable",
    "tosa-ext-int64",
    "tosa-ext-shape",
]

# Generator runs per operator: each entry is one set of extra arguments for
# tosa_verif_build_tests. Several runs are needed when, e.g., float and
# integer types require different shapes. Chosen to keep data small while
# covering every type combination and attribute edge case.
DEFAULT_GEN_ARGS = [["--tensor-dim-range", "1,16"]]

_INT_TYPES = ["--target-dtype", "int8", "int16", "int4"]
_FP_TYPES = ["--target-dtype", "fp32", "fp16", "bf16", "fp8e4m3", "fp8e5m2"]
_CONV_LIMITS = ["--max-conv-padding", "1", "--max-conv-dilation", "2", "--max-conv-stride", "2"]

OP_GEN_ARGS = {
    "conv2d": [
        [
            "--tensor-dim-range", "1,12",
            "--max-conv-padding", "2",
            "--max-conv-dilation", "2",
            "--max-conv-stride", "2",
            "--max-batch-size", "2",
        ],
        # Each test draws its mixed float types at random; a 16-channel input
        # with one kernel shape enumerates every mix cheaply.
        [*_FP_TYPES, "--target-shape", "1,6,7,16", "--conv-kernel", "3,3", "--max-conv-padding", "1",
         "--max-conv-dilation", "1", "--max-conv-stride", "1", "--test-type", "positive"],
        # FP8 mixed with wider types; this seed's draw includes every combination.
        ["--target-dtype", "fp8e4m3", "fp8e5m2", "--target-shape", "1,6,7,16", "--conv-kernel", "3,3", "2,2",
         "1,1", "--max-conv-padding", "1", "--max-conv-dilation", "1", "--max-conv-stride", "1",
         "--test-type", "positive", "--seed", "1"],
    ],
    "conv3d": [
        [*_INT_TYPES, "--target-shape", "2,3,4,3,2", "--tensor-dim-range", "1,8", *_CONV_LIMITS],
        [*_FP_TYPES, "--target-shape", "1,4,5,6,3", "--tensor-dim-range", "1,8", *_CONV_LIMITS],
        # Without a target shape, the generator also emits WrongRank tests.
        ["--tensor-dim-range", "1,8", "--test-type", "negative", *_CONV_LIMITS],
    ],
    "depthwise_conv2d": [
        [*_INT_TYPES, "--target-shape", "2,5,6,2", "--tensor-dim-range", "1,10", *_CONV_LIMITS],
        [*_FP_TYPES, "--target-shape", "1,7,9,3", "--conv-kernel", "3,3", "1,2", *_CONV_LIMITS],
        ["--tensor-dim-range", "1,10", "--test-type", "negative", *_CONV_LIMITS],
    ],
    "transpose_conv2d": [
        [*_INT_TYPES, "--target-shape", "2,4,3,2", "--tensor-dim-range", "1,10", "--max-conv-stride", "2"],
        [*_FP_TYPES, "--target-shape", "1,5,6,3", "--tensor-dim-range", "1,10", "--max-conv-stride", "2"],
        ["--target-dtype", "bf16", "--target-shape", "1,5,6,3", "--max-conv-stride", "2", "--test-type", "positive"],
        ["--tensor-dim-range", "1,10", "--test-type", "negative", "--max-conv-stride", "2"],
    ],
    # Float dot-product tests need the generator's default shapes.
    "matmul": [[]],
    "matmul_t": [
        [],
        [*_FP_TYPES, "--tensor-dim-range", "7,24", "--test-type", "positive"],
    ],
    # Float data generation needs larger windows: 16 channels and pooling
    # limits keep the float tests few and small.
    "avg_pool2d": [
        [*_INT_TYPES[:-1], "--tensor-dim-range", "1,16"],
        [*_FP_TYPES, "--target-shape", "2,11,7,16", "--max-pooling-kernel", "3", "--max-pooling-stride", "2",
         "--max-pooling-padding", "1", "--test-type", "positive"],
    ],
    "avg_pool2d_adaptive": [
        [*_INT_TYPES[:-1], "--tensor-dim-range", "1,16"],
        [*_FP_TYPES, "--target-shape", "2,11,7,16", "--max-pooling-kernel", "3", "--max-pooling-stride", "2",
         "--max-pooling-padding", "1", "--test-type", "positive"],
    ],
    "reduce_sum": [
        ["--tensor-dim-range", "1,16"],
        ["--target-dtype", "fp32", "fp16", "bf16", "--test-type", "positive"],
    ],
    # Size-1 dimensions make the generator emit a zero-sized split, and its
    # FP8 ConcatInputDimMismatch ERROR_IF test has a zero dimension, so FP8
    # gets positive tests only.
    "concat": [
        ["--tensor-dim-range", "2,16", "--test-type", "positive"],
        ["--tensor-dim-range", "2,16", "--test-type", "negative",
         "--target-dtype", "bool", "int8", "int16", "int32", "int64", "fp16", "bf16", "fp32"],
    ],
    # A rank-0 TensorSizeInputOutputMismatch ERROR_IF test fails the
    # generator's own validation, so negative tests use ranks 2-4.
    "reshape": [
        ["--tensor-dim-range", "1,16", "--test-type", "positive"],
        ["--tensor-dim-range", "1,16", "--test-type", "negative", "--target-rank", "2", "3", "4"],
    ],
}

# Per-operator cap on positive tests kept per sampling key, for operators
# whose tests are large (the generator needs big shapes for them).
OP_MAX_POSITIVE = {"matmul": 1, "matmul_t": 1, "resize": 2, "fft2d": -1, "rfft2d": -1}  # -1 keeps all


def _write_custom_test(dest: Path, tosa: bytes, inputs: dict, outputs: list[str]) -> None:
    """Write a test in the generator's layout; the reference model fills in the outputs."""
    import numpy as np

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "test.tosa").write_bytes(tosa)
    for name, value in inputs.items():
        np.save(dest / f"{name}.npy", value)
    desc = {
        "tosa_file": "test.tosa",
        "ifm_name": list(inputs),
        "ifm_file": [f"{n}.npy" for n in inputs],
        "ofm_name": outputs,
        "ofm_file": [f"ref-{n}.npy" for n in outputs],
        "expected_failure": False,
    }
    (dest / "desc.json").write_text(json.dumps(desc, indent=1) + "\n")


def _fft_cases(seed: int):
    """(name suffix, shape, scale) for FFT tests: every power-of-two edge size."""
    rng = random.Random(seed)
    shapes = [(1, 1, 1), (1, 1, 8), (1, 4, 1), (1, 2, 2), (2, 4, 8), (1, 8, 4), (1, 16, 16), (3, 2, 32)]
    for shape in shapes:
        for scale in (1.0, 1000.0):
            yield f"{'x'.join(map(str, shape))}_f32_sc{int(scale)}", shape, scale, rng.randrange(1 << 32)


def build_fft2d_tests(out: Path, seed: int) -> None:
    """Positive FFT2D tests (the generator emits only ERROR_IF tests for it)."""
    import numpy as np
    import tosa_serializer as ts
    from tosa.DType import DType
    from tosa.Op import Op

    for suffix, shape, scale, case_seed in _fft_cases(seed):
        for inverse in (False, True):
            name = f"fft2d_{suffix}_inv{int(inverse)}"
            data = np.random.default_rng(case_seed).uniform(-scale, scale, (2, *shape)).astype(np.float32)
            ser = ts.TosaSerializer(str(out / name))
            re = ser.addPlaceholder(list(shape), DType.FP32, None)
            im = ser.addPlaceholder(list(shape), DType.FP32, None)
            out_re = ser.addOutput(list(shape), DType.FP32)
            out_im = ser.addOutput(list(shape), DType.FP32)
            attr = ts.TosaSerializerAttribute()
            attr.FFT2dAttribute(inverse, False)
            ser.addOperator(Op.FFT2D, [re.name, im.name], [out_re.name, out_im.name], attr)
            _write_custom_test(out / name, ser.serialize(), {re.name: data[0], im.name: data[1]}, [out_re.name, out_im.name])


def build_rfft2d_tests(out: Path, seed: int) -> None:
    """Positive RFFT2D tests (the generator emits only ERROR_IF tests for it)."""
    import numpy as np
    import tosa_serializer as ts
    from tosa.DType import DType
    from tosa.Op import Op

    for suffix, shape, scale, case_seed in _fft_cases(seed):
        name = f"rfft2d_{suffix}"
        data = np.random.default_rng(case_seed).uniform(-scale, scale, shape).astype(np.float32)
        out_shape = [shape[0], shape[1], shape[2] // 2 + 1]
        ser = ts.TosaSerializer(str(out / name))
        x = ser.addPlaceholder(list(shape), DType.FP32, None)
        out_re = ser.addOutput(out_shape, DType.FP32)
        out_im = ser.addOutput(out_shape, DType.FP32)
        attr = ts.TosaSerializerAttribute()
        attr.RFFT2dAttribute(False)
        ser.addOperator(Op.RFFT2D, [x.name], [out_re.name, out_im.name], attr)
        _write_custom_test(out / name, ser.serialize(), {x.name: data}, [out_re.name, out_im.name])


# Operators whose positive tests the generator does not produce; built here instead.
CUSTOM_BUILDERS = {"fft2d": build_fft2d_tests, "rfft2d": build_rfft2d_tests}

# Test-name tokens describing element types, e.g. ``i8xi4``, ``accf32``, ``outbf16``.
_DTYPE = r"(?:bool|b|i\d+|u\d+|f\d+|bf16|f8e4m3|f8e5m2)"
DTYPE_TOKEN = re.compile(rf"^(?:acc|out)?{_DTYPE}(?:x{_DTYPE})*$")


def type_tokens(params: str) -> list[str]:
    return [t for t in params.split("_") if DTYPE_TOKEN.match(t)]


def positive_key(test_name: str) -> str:
    """Group key for sampling positive tests: the type tokens, and whether the
    test uses special (e.g. NaN/infinity/extreme) data, marked by an ``_s<N>`` suffix."""
    _, _, params = test_name.partition("_")
    key = type_tokens(params)
    if re.fullmatch(r"s\d+", test_name.rsplit("_", 1)[-1]):
        key.append("special")
    return "_".join(key)


def negative_kind(test_name: str) -> tuple[str, str]:
    """The ERROR_IF kind and the type signature of a negative test."""
    _, _, rest = test_name.partition("_ERRORIF_")
    kind, _, params = rest.partition("_")
    return kind, "_".join(type_tokens(params))


SITE_PACKAGES = Path(sysconfig.get_path("purelib"))
SCRIPTS_DIR = Path(sysconfig.get_path("scripts"))


def tool(name: str) -> Path:
    path = shutil.which(name, path=str(SCRIPTS_DIR))
    if not path:
        sys.exit(f"error: {name} not found in {SCRIPTS_DIR}; is tosa-tools installed in this environment?")
    return Path(path)


def find_file(env_var: str, candidates: list[Path], what: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    for c in candidates:
        if c.is_file():
            return c
    sys.exit(f"error: cannot locate the {what}; set {env_var}")


def find_ref_model() -> Path:
    bin_dir = SITE_PACKAGES / "bin"
    return find_file(
        "TOSA_REF_MODEL",
        [bin_dir / "tosa_reference_model.exe", bin_dir / "tosa_reference_model"],
        "tosa_reference_model executable",
    )


def find_generate_lib() -> Path:
    lib_dir = SITE_PACKAGES / "tosa_reference_model"
    names = ["tosa_reference_generate_lib.dll", "libtosa_reference_generate_lib.so", "libtosa_reference_generate_lib.dylib"]
    return find_file("TOSA_GENERATE_LIB", [lib_dir / n for n in names], "tosa_reference_generate_lib library")


def run_ref_model(ref_model: Path, test_dir: Path) -> int:
    proc = subprocess.run(
        [str(ref_model), "--test_desc", str(test_dir / "desc.json"), "--tosa_level", "NONE"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode


def select_tests(tests: list[Path], max_positive: int, max_negative: int, seed: int) -> list[Path]:
    """Sample the generated tests (a negative limit keeps everything).

    Positive tests: up to ``max_positive`` per ``positive_key``, so every type
    combination keeps both normal and special-value coverage.
    ERROR_IF tests: up to ``max_negative`` per error kind, drawn from as many
    different type signatures as possible.
    """
    rng = random.Random(seed)
    positive_groups: dict[str, list[Path]] = defaultdict(list)
    negative_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for t in tests:
        if "_ERRORIF_" in t.name:
            kind, types = negative_kind(t.name)
            negative_groups[kind][types].append(t)
        else:
            positive_groups[positive_key(t.name)].append(t)

    selected = []
    for key in sorted(positive_groups):
        group = positive_groups[key]
        selected += group if max_positive < 0 else rng.sample(group, min(max_positive, len(group)))
    for kind in sorted(negative_groups):
        by_types = negative_groups[kind]
        if max_negative < 0:
            selected += [t for types in sorted(by_types) for t in by_types[types]]
            continue
        # Round-robin over shuffled type signatures.
        pools = [rng.sample(by_types[k], len(by_types[k])) for k in sorted(by_types)]
        rng.shuffle(pools)
        picked: list[Path] = []
        while len(picked) < max_negative and any(pools):
            for pool in pools:
                if pool and len(picked) < max_negative:
                    picked.append(pool.pop())
        selected += picked
    return sorted(selected)


def keep_test(test: Path, ref_model: Path, dest_root: Path) -> bool:
    desc = json.loads((test / "desc.json").read_text())
    negative = desc.get("expected_failure", False)
    for f in desc["ofm_file"]:
        (test / f).unlink(missing_ok=True)  # stale output from an earlier run of a cached test
    status = run_ref_model(ref_model, test)
    if negative:
        # The reference model must reject the graph; otherwise the generator
        # produced something we cannot use as an oracle.
        if status == 0:
            return False
    elif status != 0 or any(not (test / f).exists() for f in desc["ofm_file"]):
        return False

    # Keep only what the Rust harness needs.
    desc = {
        k: desc[k]
        for k in ("tosa_file", "ifm_name", "ifm_file", "ofm_name", "ofm_file", "expected_failure", "meta")
        if k in desc
    }
    dest = dest_root / test.name
    dest.mkdir(parents=True, exist_ok=True)
    files = [desc["tosa_file"], *desc["ifm_file"]]
    if not negative:
        files += desc["ofm_file"]
    for f in files:
        shutil.copy2(test / f, dest / f)
    (dest / "desc.json").write_text(json.dumps(desc, indent=1) + "\n")
    return True


def generate_op(args, op: str, ref_model: Path, generate_lib: Path) -> tuple[int, int]:
    gen_dir = GEN_CACHE / op
    if args.regenerate or not (gen_dir / ".complete").exists():
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
        gen_dir.mkdir(parents=True)
        runs = OP_GEN_ARGS.get(op, DEFAULT_GEN_ARGS)
        for i, run_args in enumerate(runs):
            out = gen_dir / str(i)
            out.mkdir(parents=True)
            cmd = [
                str(tool("tosa_verif_build_tests")),
                "--generate-lib-path", str(generate_lib),
                "-o", str(out),
                "--filter", OP_FILTERS.get(op, f"^{op}$"),
                "--extension", *EXTENSIONS,
                "--seed", str(args.seed),
                "--test-type", "both",
                *run_args,
                *args.gen_arg,
            ]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        if op in CUSTOM_BUILDERS:
            CUSTOM_BUILDERS[op](gen_dir / "custom", args.seed)
        (gen_dir / ".complete").touch()
    else:
        print(f"{op}: reusing generator output in {gen_dir} (--regenerate to rebuild)", flush=True)

    # Runs may produce identically named tests (typically ERROR_IF ones); keep the first.
    tests = list({p.parent.name: p.parent for p in sorted(gen_dir.rglob("desc.json"), reverse=True)}.values())
    tests.sort(key=lambda p: p.name)
    if args.test_type != "both":
        want_negative = args.test_type == "negative"
        tests = [t for t in tests if ("_ERRORIF_" in t.name) == want_negative]
    if not tests:
        print(f"warning: no tests generated for {op}", file=sys.stderr)
        return 0, 0
    max_positive = args.max_positive
    if max_positive >= 0 and op in OP_MAX_POSITIVE:
        max_positive = min(max_positive, OP_MAX_POSITIVE[op])
    selected = select_tests(tests, max_positive, args.max_negative, args.seed)
    print(f"{op}: generated {len(tests)} tests, checking {len(selected)} against the reference model", flush=True)

    dest_root = args.out / op
    if dest_root.exists() and not args.keep:
        shutil.rmtree(dest_root)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda t: keep_test(t, ref_model, dest_root), selected))
    kept = sum(results)
    return kept, len(results) - kept

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", action="append", required=True, help="operator to generate (repeatable)")
    parser.add_argument("--out", type=Path, help=f"output root (default {DEFAULT_OUT}, or {FULL_OUT} with --full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-type", default="both", choices=["positive", "negative", "both"])
    parser.add_argument(
        "--max-positive",
        type=int,
        default=3,
        help="positive tests kept per (type signature, special data); -1 keeps all",
    )
    parser.add_argument(
        "--max-negative",
        type=int,
        default=3,
        help="ERROR_IF tests kept per error kind; -1 keeps all",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="keep every generated test (implies --max-positive -1 --max-negative -1)",
    )
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--regenerate", action="store_true", help="rerun the generator even if cached")
    parser.add_argument("--keep", action="store_true", help="do not delete existing cases for the op")
    parser.add_argument(
        "--gen-arg", action="append", default=[], help="extra argument passed to tosa_verif_build_tests"
    )
    args = parser.parse_args()
    if args.full:
        args.max_positive = args.max_negative = -1
    if args.out is None:
        args.out = FULL_OUT if args.full else DEFAULT_OUT

    ref_model = find_ref_model()
    generate_lib = find_generate_lib()
    print(f"reference model: {ref_model}")
    failed = []
    for op in args.op:
        try:
            kept, dropped = generate_op(args, op, ref_model, generate_lib)
        except subprocess.CalledProcessError as e:
            print(f"{op}: FAILED: {e}", file=sys.stderr)
            failed.append(op)
            continue
        print(f"{op}: kept {kept} test cases, dropped {dropped}")
    if failed:
        sys.exit(f"generation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()