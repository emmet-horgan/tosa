#!/usr/bin/env python3
"""Summarise conformance coverage per operator for docs/OPERATOR_STATUS.md.

For every operator with a suite under ``tests/conformance`` this reports:

* checked-in positive / ERROR_IF cases and how many pass;
* the size and pass count of the full suite in ``target/conformance-full``
  (if it has been generated with ``gen_conformance.py --full``);
* type signatures and ERROR_IF kinds covered, relative to every signature and
  kind the tosa-tools generator defines for the operator across *all*
  profiles and extensions (including the MX extensions tosa-interp does not
  implement) -- our proxy for "the specification".

The generated table replaces the text between the ``BEGIN GENERATED`` and
``END GENERATED`` markers of the status document. Run from the repository
root with the tosa-tools environment:

    .venv\\Scripts\\python scripts\\conformance_report.py      # Windows
    .venv/bin/python scripts/conformance_report.py            # Linux/macOS

Spec listings are cached in ``target/conformance-spec.json``; pass
``--refresh-spec`` to rebuild them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_conformance as gen  # noqa: E402

REPO = gen.REPO
DOC = REPO / "docs" / "OPERATOR_STATUS.md"
SPEC_CACHE = REPO / "target" / "conformance-spec.json"
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"

# Type tokens, including MX block-scaled ones such as ``bs32-f8ue8m0-f8e4m3``.
_DT = r"(?:bool|b|i\d+|u\d+|f\d+|bf16|f8e4m3|f8e5m2|f6e2m3|f6e3m2|f4e2m1|f8ue8m0|mxint8|bs\d+-[a-z0-9-]+)"
TYPE_TOKEN = re.compile(rf"^(?:acc|out)?{_DT}(?:x{_DT})*$")
ALL_EXTENSIONS = gen.EXTENSIONS + [
    "tosa-ext-mx-common", "tosa-ext-mx-fp4e2m1", "tosa-ext-mx-fp6e2m3", "tosa-ext-mx-fp6e3m2",
    "tosa-ext-mx-fp8e4m3", "tosa-ext-mx-fp8e5m2", "tosa-ext-mx-int8", "tosa-ext-mxfp-conv",
]


def signature(test_name: str) -> str:
    """Type signature of a positive test name."""
    _, _, params = test_name.partition("_")
    return "_".join(t for t in params.split("_") if TYPE_TOKEN.match(t))


def error_kind(test_name: str) -> str:
    return test_name.partition("_ERRORIF_")[2].partition("_")[0]


def classify(names):
    sigs, kinds = set(), set()
    for n in names:
        if "_ERRORIF_" in n:
            kinds.add(error_kind(n))
        else:
            sigs.add(signature(n))
    return sigs, kinds


def spec_listing(op: str, generate_lib: Path) -> list[str]:
    cmd = [
        str(gen.tool("tosa_verif_build_tests")), "--generate-lib-path", str(generate_lib),
        "--filter", gen.OP_FILTERS.get(op, f"^{op}$"), "--extension", *ALL_EXTENSIONS,
        "--test-type", "both", "--list-tests",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def run_suite(root: Path) -> set[str] | None:
    """Names (``op::test``) of failing cases when running the suite in ``root``."""
    if not root.is_dir():
        return None
    env = dict(os.environ, TOSA_CONFORMANCE_DIR=str(root))
    proc = subprocess.run(
        ["cargo", "test", "--release", "--test", "conformance", "--", "-q"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    out = proc.stdout + proc.stderr
    if "test result:" not in out:
        sys.exit(f"cargo test failed to run:\n{out[-2000:]}")
    return set(re.findall(r"^    ([a-z0-9_]+::\S+)$", out, re.MULTILINE))


def cases(root: Path, op: str) -> list[str]:
    d = root / op
    return sorted(p.parent.name for p in d.rglob("desc.json")) if d.is_dir() else []


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-spec", action="store_true")
    parser.add_argument("--no-run", action="store_true", help="do not run the suites (assume all pass)")
    args = parser.parse_args()

    ops = sorted(p.name for p in gen.DEFAULT_OUT.iterdir() if p.is_dir())
    spec = json.loads(SPEC_CACHE.read_text()) if SPEC_CACHE.exists() and not args.refresh_spec else {}
    missing = [op for op in ops if op not in spec]
    if missing:
        lib = gen.find_generate_lib()
        for op in missing:
            print(f"listing spec tests for {op}", file=sys.stderr)
            spec[op] = spec_listing(op, lib)
        SPEC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SPEC_CACHE.write_text(json.dumps(spec, indent=0))

    failing = set() if args.no_run else run_suite(gen.DEFAULT_OUT)
    failing_full = None if args.no_run else run_suite(gen.FULL_OUT)

    rows = []
    totals = defaultdict(int)
    for op in ops:
        names = cases(gen.DEFAULT_OUT, op)
        pos = [n for n in names if "_ERRORIF_" not in n]
        neg = [n for n in names if "_ERRORIF_" in n]
        bad = {n for n in names if f"{op}::{n}" in (failing or set())}
        sigs, kinds = classify(names)
        spec_sigs, spec_kinds = classify(spec.get(op, []))
        # Signatures also come from full-suite generation, which may cover more.
        full = cases(gen.FULL_OUT, op)
        if full:
            fs, fk = classify(full)
            sigs |= fs
            kinds |= fk
        sig_cov = len(sigs & spec_sigs) if spec_sigs else len(sigs)
        kind_cov = len(kinds & spec_kinds) if spec_kinds else len(kinds)
        if full and failing_full is not None:
            full_bad = sum(1 for n in full if f"{op}::{n}" in failing_full)
            full_txt = f"{len(full) - full_bad}/{len(full)}"
        else:
            full_txt = "-"
        # Coverage estimate: type signatures weighted 3:1 against ERROR_IF kinds.
        sig_ratio = sig_cov / len(spec_sigs) if spec_sigs else 1.0
        kind_ratio = kind_cov / len(spec_kinds) if spec_kinds else 1.0
        estimate = (3 * sig_ratio + kind_ratio) / 4
        rows.append(
            f"| {op.upper()} | {len(pos) - len(bad & set(pos))}/{len(pos)} | "
            f"{len(neg) - len(bad & set(neg))}/{len(neg)} | {full_txt} | "
            f"{sig_cov}/{len(spec_sigs)} | {kind_cov}/{len(spec_kinds)} | {estimate:.0%} |"
        )
        totals["pos"] += len(pos)
        totals["neg"] += len(neg)
        totals["bad"] += len(bad)

    table = "\n".join(
        [
            "| Operator | Positive (pass/total) | ERROR_IF (pass/total) | Full suite (pass/total) "
            "| Type signatures (tested/spec) | ERROR_IF kinds (tested/spec) | Coverage estimate |",
            "|---|---|---|---|---|---|---|",
            *rows,
        ]
    )
    summary = (
        f"Checked-in suite: {totals['pos'] + totals['neg'] - totals['bad']}/{totals['pos'] + totals['neg']} "
        f"cases pass ({totals['pos']} positive, {totals['neg']} ERROR_IF) across {len(ops)} operators."
    )
    block = f"{BEGIN}\n{summary}\n\n{table}\n{END}"
    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        sys.exit(f"{DOC} lacks the {BEGIN} / {END} markers")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + block + tail, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
