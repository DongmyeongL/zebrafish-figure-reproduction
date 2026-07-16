#!/usr/bin/env python3
"""Save regions complete in all seven animals for both Figures 9 and 12."""

from __future__ import annotations

from pathlib import Path
import os

import pandas as pd


PACK_ROOT = Path(__file__).resolve().parents[1]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
OUTPUT_DIR = DERIVED_ROOT / "common"
OUTPUT = OUTPUT_DIR / "figure9_figure12_complete_all7_regions.csv"


def summarize(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return (
        table.groupby(["root_area_id", "node", "anatomy_group"], as_index=False)
        .agg(
            **{
                f"n_observations_{prefix}": ("recording_id", "nunique"),
                f"n_complete_{prefix}": ("complete_five_measures", "sum"),
            }
        )
    )


def main() -> None:
    figure9 = pd.read_csv(
        DERIVED_ROOT / "figure9" / "figure9_recording_region_measures.csv"
    )
    figure12 = pd.read_csv(
        DERIVED_ROOT / "figure12" / "figure12_subject_region_structural_measures.csv"
    )
    summary9 = summarize(figure9, "figure9")
    summary12 = summarize(figure12, "figure12")
    common = summary9.merge(
        summary12,
        on=["root_area_id", "node", "anatomy_group"],
        how="inner",
        validate="one_to_one",
    )
    keep = (
        common["n_observations_figure9"].eq(7)
        & common["n_complete_figure9"].eq(7)
        & common["n_observations_figure12"].eq(7)
        & common["n_complete_figure12"].eq(7)
    )
    common = common.loc[keep].sort_values("root_area_id").reset_index(drop=True)
    common.insert(0, "common_region_index", range(1, len(common) + 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    common.to_csv(OUTPUT, index=False)
    print(f"Saved {len(common)} regions to {OUTPUT}")
    print(common[["root_area_id", "node", "anatomy_group"]].to_string(index=False))


if __name__ == "__main__":
    main()
