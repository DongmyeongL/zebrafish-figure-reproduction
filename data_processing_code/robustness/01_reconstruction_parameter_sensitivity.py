#!/usr/bin/env python3
"""Validate and summarize the 5 x 6 SC-reconstruction sensitivity grid.

The input contains correlations already calculated after rebuilding SC for
each anatomical-unit target size and endpoint-to-soma matching radius. Full SC
generation is handled by ``generate_full_reconstruction_controls.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import DATA, OUTPUT


TARGET_SIZES = (250, 300, 400, 500, 600)
ENDPOINT_RADII = (10.0, 11.0, 12.0, 13.0, 13.86, 15.0)
EXPECTED_METRICS = ("OO_fraction", "DCApost")
DEFAULT_INPUT = DATA / "reconstruction_grid_correlations.csv"
DEFAULT_OUTPUT = OUTPUT / "reconstruction_parameter_sensitivity_summary.csv"
REQUIRED_COLUMNS = {
    "target_size",
    "endpoint_radius",
    "metric",
    "n_regions",
    "pearson_r",
    "pearson_p",
}


def expected_configurations() -> set[tuple[int, float]]:
    return {
        (target_size, radius)
        for target_size in TARGET_SIZES
        for radius in ENDPOINT_RADII
    }


def validate_grid(table: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(table.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    duplicate = table.duplicated(["metric", "target_size", "endpoint_radius"])
    if duplicate.any():
        rows = table.loc[duplicate, ["metric", "target_size", "endpoint_radius"]]
        raise ValueError(f"Duplicate metric/configuration rows:\n{rows.to_string(index=False)}")

    expected = expected_configurations()
    observed_metrics = set(table["metric"])
    if observed_metrics != set(EXPECTED_METRICS):
        raise ValueError(
            f"Expected metrics {list(EXPECTED_METRICS)}, found {sorted(observed_metrics)}"
        )

    for metric, frame in table.groupby("metric"):
        observed = set(zip(frame["target_size"].astype(int), frame["endpoint_radius"].astype(float)))
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(f"{metric}: incomplete grid; missing={missing}, extra={extra}")


def summarize_grid(
    table: pd.DataFrame,
    primary_target_size: int,
    primary_radius: float,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for metric, frame in table.groupby("metric", sort=True):
        primary = frame[
            frame["target_size"].eq(primary_target_size)
            & frame["endpoint_radius"].eq(primary_radius)
        ]
        if len(primary) != 1:
            raise ValueError(
                f"{metric}: expected one primary row for target={primary_target_size}, "
                f"radius={primary_radius}; found {len(primary)}"
            )
        primary_row = primary.iloc[0]
        rows.append({
            "metric": metric,
            "n_configurations": len(frame),
            "n_regions_min": int(frame["n_regions"].min()),
            "n_regions_max": int(frame["n_regions"].max()),
            "pearson_r_min": frame["pearson_r"].min(),
            "pearson_r_max": frame["pearson_r"].max(),
            "n_positive": int(frame["pearson_r"].gt(0).sum()),
            "n_nominal_p_below_alpha": int(frame["pearson_p"].lt(alpha).sum()),
            "nominal_alpha": alpha,
            "primary_target_size": primary_target_size,
            "primary_endpoint_radius": primary_radius,
            "primary_n_regions": int(primary_row["n_regions"]),
            "primary_pearson_r": primary_row["pearson_r"],
            "primary_pearson_p": primary_row["pearson_p"],
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-target-size", type=int, default=400)
    parser.add_argument("--primary-radius", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input)
    validate_grid(table)
    summary = summarize_grid(
        table,
        primary_target_size=args.primary_target_size,
        primary_radius=args.primary_radius,
        alpha=args.alpha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
