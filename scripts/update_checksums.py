#!/usr/bin/env python3
"""Regenerate checksums for distributed code and data files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "checksums.sha256"
EXCLUDED_PREFIXES = (".git/", "figures/", "statistics/", "work/")


def distributed_files() -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        relative_paths = [Path(line) for line in result.stdout.splitlines() if line]
    except (FileNotFoundError, subprocess.CalledProcessError):
        relative_paths = [path.relative_to(ROOT) for path in ROOT.rglob("*") if path.is_file()]

    selected = []
    for relative in relative_paths:
        name = relative.as_posix()
        if name == OUTPUT.name or name.endswith(".pyc"):
            continue
        if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        selected.append(ROOT / relative)
    return sorted(set(selected), key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = distributed_files()
    lines = [f"{sha256(path)}  ./{path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
