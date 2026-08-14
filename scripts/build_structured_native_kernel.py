#!/usr/bin/env python3
"""Build and record the exact structured-policy pairwise reduction kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path


COMPILER_FLAGS = (
    "-O3",
    "-shared",
    "-fPIC",
    "-ffp-contract=off",
    "-fno-fast-math",
    "-fexcess-precision=standard",
)
MODULE_NAME = "_airhockey_pairwise_float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--compiler", default="gcc")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if platform.system() != "Linux":
        raise RuntimeError("the release kernel must be built on Linux")
    if len(args.code_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.code_commit
    ):
        raise ValueError("code commit must be a full lowercase Git hash")

    repository = Path(__file__).resolve().parents[1]
    source = repository / "native" / f"{MODULE_NAME}.c"
    compiler = shutil.which(args.compiler)
    include = sysconfig.get_config_var("INCLUDEPY")
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if compiler is None:
        raise RuntimeError(f"compiler {args.compiler!r} is unavailable")
    if not include or not (Path(include) / "Python.h").is_file():
        raise RuntimeError("the matching CPython development headers are unavailable")
    if not extension_suffix:
        raise RuntimeError("CPython extension suffix is unavailable")

    output_directory = args.output_directory.resolve()
    manifest = args.manifest.resolve()
    output = output_directory / f"{MODULE_NAME}{extension_suffix}"
    if output.exists() or manifest.exists():
        raise FileExistsError("native kernel output already exists")
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    command = [
        compiler,
        *COMPILER_FLAGS,
        f"-I{include}",
        str(source),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    compiler_path = Path(compiler).resolve()
    compiler_version = subprocess.run(
        [compiler, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "implementation": "native_pairwise_float32_v1",
        "code_commit": args.code_commit,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "binary": str(output),
        "binary_sha256": sha256_file(output),
        "compiler": str(compiler_path),
        "compiler_sha256": sha256_file(compiler_path),
        "compiler_version": compiler_version,
        "compiler_flags": list(COMPILER_FLAGS),
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "extension_suffix": extension_suffix,
        "platform": platform.platform(),
    }
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
