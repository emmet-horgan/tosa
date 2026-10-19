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
positive tests it fails to run.

Usage (from the tosa-interp checkout, with a tosa-tools checkout whose Python
package -- including the reference model -- is installed into its ``.venv``):

    ../tosa-tools/.venv/bin/python scripts/gen_conformance.py \\
        --tosa-tools ../tosa-tools --op conv2d

Each ``--op`` value is a regular expression matched against the generator's
operator names (``conv2d`` expands to ``^conv2d_\\d+x\\d+$`` style templated
names automatically, see ``OP_FILTERS``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "tests" / "conformance"

# Generator operator-name regexes for operators whose tests are templated.
OP_FILTERS = {
    "conv2d": r"^conv2d_\d+x\d+$",
    "conv3d": r"^conv3d_\d+x\d+x\d+$",
    "depthwise_conv2d": r"^depthwise_conv2d_\d+x\d+$",
    "transpose_conv2d": r"^transpose_conv2d_\d+x\d+$",
}

# Default generator options per operator, chosen to keep the checked-in data
# small while still covering every type combination and attribute edge case.
OP_GEN_ARGS = {
    "conv2d": [
        "--tensor-dim-range", "1,12",
        "--max-conv-padding", "2",
        "--max-conv-dilation", "2",
        "--max-conv-stride", "2",
        "--max-batch-size", "2",
    ],
}


def tool(venv_bin: Path, name: str) -> Path:
    path = venv_bin / name
    if not path.exists():
        sys.exit(f"error: {path} not found; is tosa-tools installed into its venv?")
    return path


def find_ref_model(tosa_tools: Path) -> Path:
    candidates = [
        os.environ.get("TOSA_REF_MODEL"),
        *sorted(str(p) for p in (tosa_tools / ".venv").glob("lib/python*/site-packages/bin/tosa_reference_model")),
        str(tosa_tools / "build-py" / "skbuild" / "reference_model" / "reference_model" / "tosa_reference_model"),
        str(tosa_tools / "build" / "reference_model" / "tosa_reference_model"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    sys.exit("error: cannot locate the tosa_reference_model executable; set TOSA_REF_MODEL")


def run_ref_model(ref_model: Path, test_dir: Path) -> int:
    proc = subprocess.run(
        [str(ref_model), "--test_desc", str(test_dir / "desc.json"), "--tosa_level", "NONE"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode


def generate_op(args, op: str, venv_bin: Path, ref_model: Path) -> tuple[int, int]:
    op_filter = OP_FILTERS.get(op, f"^{op}$")
    gen_args = OP_GEN_ARGS.get(op, []) + args.gen_arg
    dest_root = args.out / op
    with tempfile.TemporaryDirectory(prefix=f"tosa-gen-{op}-") as tmp:
        cmd = [
            str(tool(venv_bin, "tosa_verif_build_tests")),
            "-o", tmp,
            "--filter", op_filter,
            "--seed", str(args.seed),
            "--test-type", args.test_type,
            *gen_args,
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        tests = sorted(p.parent for p in Path(tmp).rglob("desc.json"))
        if not tests:
            print(f"warning: no tests generated for {op}", file=sys.stderr)
            return 0, 0

        if dest_root.exists() and not args.keep:
            shutil.rmtree(dest_root)
        kept = dropped = 0
        for test in tests:
            desc_path = test / "desc.json"
            desc = json.loads(desc_path.read_text())
            negative = desc.get("expected_failure", False)
            status = run_ref_model(ref_model, test)
            if negative:
                # The reference model must reject the graph; otherwise the
                # generator produced something we cannot use as an oracle.
                if status == 0:
                    dropped += 1
                    continue
            elif status != 0:
                dropped += 1
                continue
            missing = [f for f in desc["ofm_file"] if not (test / f).exists()]
            if not negative and missing:
                dropped += 1
                continue

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
            kept += 1
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tosa-tools", type=Path, default=REPO.parent / "tosa-tools")
    parser.add_argument("--op", action="append", required=True, help="operator to generate (repeatable)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-type", default="both", choices=["positive", "negative", "both"])
    parser.add_argument("--keep", action="store_true", help="do not delete existing cases for the op")
    parser.add_argument(
        "--gen-arg", action="append", default=[], help="extra argument passed to tosa_verif_build_tests"
    )
    args = parser.parse_args()

    venv_bin = args.tosa_tools / ".venv" / "bin"
    ref_model = find_ref_model(args.tosa_tools)
    print(f"reference model: {ref_model}")
    for op in args.op:
        kept, dropped = generate_op(args, op, venv_bin, ref_model)
        print(f"{op}: kept {kept} test cases, dropped {dropped}")


if __name__ == "__main__":
    main()
