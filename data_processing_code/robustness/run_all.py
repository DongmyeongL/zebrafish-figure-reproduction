#!/usr/bin/env python3
"""Run all public-release SI robustness summaries."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def main() -> None:
    scripts = sorted(HERE.glob("[0-9][0-9]_*.py"))
    for script in scripts:
        print(f"\n==> {script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
