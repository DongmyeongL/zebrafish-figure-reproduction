#!/usr/bin/env python3
"""Reproduce SI summaries for reconstructed-morphology subsampling."""

from __future__ import annotations

import pandas as pd

from common import DATA, save_table


def main() -> None:
    iterations = pd.read_csv(DATA / "morphology_subsampling_iterations.csv")
    rows = []
    for (fraction, metric), frame in iterations.groupby(["subset_fraction", "metric"]):
        rows.append({
            "subset_fraction": fraction,
            "metric": metric,
            "n_iterations": len(frame),
            "n_morphologies": int(frame["n_morphologies"].median()),
            "median_full_reproducibility_r": frame["full_reproducibility_r"].median(),
            "full_reproducibility_ci_low": frame["full_reproducibility_r"].quantile(0.025),
            "full_reproducibility_ci_high": frame["full_reproducibility_r"].quantile(0.975),
            "median_fcv_pearson_r": frame["fcv_pearson_r"].median(),
            "fcv_pearson_r_ci_low": frame["fcv_pearson_r"].quantile(0.025),
            "fcv_pearson_r_ci_high": frame["fcv_pearson_r"].quantile(0.975),
            "proportion_fcv_r_positive": (frame["fcv_pearson_r"] > 0).mean(),
            "proportion_fcv_p_below_0_05": (frame["fcv_pearson_p"] < 0.05).mean(),
        })
    summary = pd.DataFrame(rows).sort_values(["metric", "subset_fraction"])
    path = save_table(summary, "morphology_subsampling_summary.csv")
    print(summary.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
