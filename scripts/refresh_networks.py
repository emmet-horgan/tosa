#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["tosa-tools==2026.8.0"]
# ///
"""Refresh the reference outputs of the multi-operator test cases in ``tests/networks``.

Every directory below ``tests/networks`` containing a ``desc.json`` is a test
case in the conformance format (see ``tests/conformance.rs``): a ``.tosa``
graph, one ``.npy`` per input and one ``ref-<output>.npy`` per output. The
graphs and inputs are checked in; this script reruns the TOSA reference model
on each case to regenerate the expected outputs, e.g. after upgrading the
reference model.

Run it from any OS with uv, which installs ``tosa-tools`` on the fly:

    uv run scripts/refresh_networks.py

or with a Python environment that already has ``tosa-tools`` installed:

    python scripts/refresh_networks.py

To add a case, create ``tests/networks/<name>/`` with the graph, the input
``.npy`` files and a ``desc.json`` naming them, then run this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_conformance import REPO, find_ref_model, run_ref_model  # noqa: E402

NETWORKS = REPO / "tests" / "networks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="case names to refresh (default: all)")
    args = parser.parse_args()

    ref_model = find_ref_model()
    cases = sorted(p.parent for p in NETWORKS.glob("*/desc.json"))
    if args.cases:
        cases = [c for c in cases if c.name in args.cases]
        missing = set(args.cases) - {c.name for c in cases}
        if missing:
            sys.exit(f"error: unknown case(s): {', '.join(sorted(missing))}")

    failed = []
    for case in cases:
        desc = json.loads((case / "desc.json").read_text())
        for f in desc["ofm_file"]:
            (case / f).unlink(missing_ok=True)
        status = run_ref_model(ref_model, case)
        produced = all((case / f).is_file() for f in desc["ofm_file"])
        print(f"{case.name}: {'ok' if status == 0 and produced else f'FAILED (exit code {status})'}")
        if status != 0 or not produced:
            failed.append(case.name)
    if failed:
        sys.exit(f"error: the reference model failed on: {', '.join(failed)}")


if __name__ == "__main__":
    main()
