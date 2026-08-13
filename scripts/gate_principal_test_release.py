#!/usr/bin/env python3
"""Keep the principal test closed unless every frozen artefact passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from airhockey_distill.principal_release import evaluate_test_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    result = evaluate_test_release(args.protocol, args.evidence_manifest)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
