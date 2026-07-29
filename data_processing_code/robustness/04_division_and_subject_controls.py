#!/usr/bin/env python3
"""Reproduce division-adjusted and subject-specific FCV controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr

from common import DATA, random_effects_summary, residualize_by_group, save_table


SEED = 20_260_726
N_PERMUTATIONS = 20_000
MEASURES = {"OO fraction": "Hard_OO_fraction", "DCA_post": "FU_DCApost"}


def division_adjusted() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "division_adjusted_region_input.csv").dropna()
    y = frame["EdgeStdFCV"].to_numpy(float)
    y_residual = residualize_by_group(y, frame["anatomy_group"])
    indices = [np.asarray(x, dtype=int) for x in frame.groupby("anatomy_group").indices.values()]
    rng = np.random.default_rng(SEED)
    rows = []
    for measure, column in MEASURES.items():
        x = frame[column].to_numpy(float)
        x_residual = residualize_by_group(x, frame["anatomy_group"])
        partial = pearsonr(x_residual, y_residual)
        null = np.empty(N_PERMUTATIONS)
        for iteration in range(N_PERMUTATIONS):
            shuffled = x.copy()
            for group_indices in indices:
                shuffled[group_indices] = rng.permutation(shuffled[group_indices])
            null[iteration] = pearsonr(
                residualize_by_group(shuffled, frame["anatomy_group"]), y_residual
            ).statistic
        rows.append({
            "measure": measure,
            "n_regions": len(frame),
            "unadjusted_pearson_r": pearsonr(x, y).statistic,
            "within_division_partial_r": partial.statistic,
            "within_division_partial_p_parametric": partial.pvalue,
            "within_division_permutation_p": (
                1 + np.count_nonzero(np.abs(null) >= abs(partial.statistic))
            ) / (N_PERMUTATIONS + 1),
            "n_permutations": N_PERMUTATIONS,
            "random_seed": SEED,
        })
    return pd.DataFrame(rows)


def subject_specific() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(DATA / "subject_specific_region_input.csv").dropna()
    rows = []
    for measure, column in MEASURES.items():
        for subject, group in frame.groupby("Subject"):
            result = pearsonr(group[column], group["EdgeStdFCV"])
            fisher_z = np.arctanh(np.clip(result.statistic, -0.999999, 0.999999))
            half_width = norm.ppf(0.975) / np.sqrt(len(group) - 3)
            rows.append({
                "measure": measure,
                "Subject": subject,
                "n_regions": len(group),
                "pearson_r": result.statistic,
                "pearson_p": result.pvalue,
                "pearson_ci_low": np.tanh(fisher_z - half_width),
                "pearson_ci_high": np.tanh(fisher_z + half_width),
                "fisher_z": fisher_z,
                "fisher_variance": 1.0 / (len(group) - 3),
            })
    effects = pd.DataFrame(rows)
    summaries = []
    for measure, group in effects.groupby("measure", sort=False):
        summaries.append({
            "measure": measure,
            "common_regions": int(group["n_regions"].iloc[0]),
            **random_effects_summary(group),
        })
    return effects, pd.DataFrame(summaries)


def main() -> None:
    division = division_adjusted()
    effects, meta = subject_specific()
    paths = [
        save_table(division, "division_adjusted_fcv_structure.csv"),
        save_table(effects, "subject_specific_fcv_structure_correlations.csv"),
        save_table(meta, "subject_specific_meta_analysis.csv"),
    ]
    print("Division-adjusted controls\n", division.to_string(index=False))
    print("\nSubject-specific random-effects summaries\n", meta.to_string(index=False))
    print("Saved " + ", ".join(map(str, paths)))


if __name__ == "__main__":
    main()
