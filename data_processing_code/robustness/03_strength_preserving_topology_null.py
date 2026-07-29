#!/usr/bin/env python3
"""Evaluate observed FCV correlations against strength-preserving nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, save_table


def main() -> None:
    nulls = pd.read_csv(DATA / "topology_null_correlations.csv")
    nulls = nulls[nulls["null_type"].eq("strength_preserving")].copy()
    observed = pd.read_csv(DATA / "topology_null_observed_correlations.csv")
    observed = observed[observed["support_threshold"].eq(1)].set_index("metric")

    rows = []
    for metric, frame in nulls.groupby("metric", sort=True):
        values = frame["pearson_r"].to_numpy(float)
        observed_r = float(observed.loc[metric, "pearson_r"])
        null_mean = float(values.mean())
        centered_p = (
            1 + np.count_nonzero(np.abs(values - null_mean) >= abs(observed_r - null_mean))
        ) / (len(values) + 1)
        upper_p = (1 + np.count_nonzero(values >= observed_r)) / (len(values) + 1)
        rows.append({
            "metric": metric,
            "n_null": len(values),
            "observed_pearson_r": observed_r,
            "null_mean_r": null_mean,
            "null_ci_low": np.quantile(values, 0.025),
            "null_ci_high": np.quantile(values, 0.975),
            "empirical_centered_two_sided_p": centered_p,
            "empirical_upper_tail_p": upper_p,
            "median_max_relative_margin_error": frame["max_relative_margin_error"].median(),
        })
    summary = pd.DataFrame(rows)
    path = save_table(summary, "strength_preserving_topology_null_summary.csv")
    print(summary.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
