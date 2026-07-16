#!/usr/bin/env python3
"""Check that the documented release inputs and figure outputs are present."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
EXPECTED_FIGURES = [
    "figure9_final_v2.png",
    "figure12_final_v2.png",
    "figure13_final_v2.png",
    "figure_fcv_fcs_sc_corr_forest.png",
    "figure_invertebrate_oo_fcv_relationships.png",
    "figure_stimulus_delta_fcv_acd_combined.png",
    "figure_invertebrate_anatomical_group_summary.png",
    "figure_stimulus_condition_region_profiles.png",
    "figure_supplement_te_structural_controls.png",
    "figure_supply_1.png",
    "figure_supply_2_proc.png",
    "figure_supply_5.png",
    "figure_supply_10_proc.png",
    "figure_supply_13.png",
    "figure_supply_15.png",
]


def main() -> None:
    failures = []
    for name in EXPECTED_FIGURES:
        path = ROOT / "figures" / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(name)
        else:
            print(f"OK  {name:58s} {path.stat().st_size / 1_000_000:7.2f} MB")

    if failures:
        print("Missing or empty outputs: " + ", ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"Validated {len(EXPECTED_FIGURES)} figure files.")


if __name__ == "__main__":
    main()
