#!/usr/bin/env python3
"""Summarize the anatomical-unit-size and endpoint-radius sensitivity grid."""

from __future__ import annotations

import pandas as pd

from common import DATA, save_table


TARGET_SIZES = {250, 300, 400, 500, 600}
RADII = {10.0, 11.0, 12.0, 13.0, 13.86, 15.0}
PRIMARY = (400, 12.0)


def main() -> None:
    table = pd.read_csv(DATA / "reconstruction_grid_correlations.csv")
    combinations = set(zip(table["target_size"], table["endpoint_radius"]))
    expected = {(size, radius) for size in TARGET_SIZES for radius in RADII}
    if combinations != expected:
        raise RuntimeError(f"Incomplete parameter grid: expected {len(expected)}, found {len(combinations)}")

    rows = []
    for metric, frame in table.groupby("metric", sort=True):
        primary = frame[
            frame["target_size"].eq(PRIMARY[0]) & frame["endpoint_radius"].eq(PRIMARY[1])
        ].iloc[0]
        rows.append({
            "metric": metric,
            "n_configurations": len(frame),
            "pearson_r_min": frame["pearson_r"].min(),
            "pearson_r_max": frame["pearson_r"].max(),
            "n_positive": int((frame["pearson_r"] > 0).sum()),
            "n_nominal_p_below_0_05": int((frame["pearson_p"] < 0.05).sum()),
            "primary_target_size": PRIMARY[0],
            "primary_endpoint_radius": PRIMARY[1],
            "primary_pearson_r": primary["pearson_r"],
            "primary_pearson_p": primary["pearson_p"],
        })
    summary = pd.DataFrame(rows)
    path = save_table(summary, "reconstruction_parameter_sensitivity_summary.csv")
    print(summary.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
