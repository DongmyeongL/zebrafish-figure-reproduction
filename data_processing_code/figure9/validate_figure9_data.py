#!/usr/bin/env python3
"""Compatibility wrapper for the public Figure 9 validator."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    derived = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", ROOT / "derived_data"))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_derived.py"),
            "figure9",
            "--derived-root",
            str(derived),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
