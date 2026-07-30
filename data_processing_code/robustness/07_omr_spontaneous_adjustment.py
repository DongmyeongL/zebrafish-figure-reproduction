#!/usr/bin/env python3
"""Reproduce spontaneous-FCV and division-adjusted OMR associations."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from common import RELEASE, save_table


INPUT = (
    RELEASE
    / "derived_data"
    / "common"
    / "figure_stimulus_delta_fcv_acd_combined_skeleton_kmeans_nearest_r12_two_way_subsampling_input.csv"
)
N_PERMUTATIONS = int(os.environ.get("ZF_OMR_ADJUSTMENT_PERMUTATIONS", "20000"))
SEED = 20260730


def zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - values.mean()) / values.std(ddof=0)


def design_matrix(frame, controls):
    columns = []
    if "spont_FCV" in controls:
        columns.append(zscore(frame["spont_FCV"]))
    if "division" in controls:
        dummies = pd.get_dummies(
            frame["anatomy_group"], drop_first=True, dtype=float
        )
        columns.extend(dummies.to_numpy().T)
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(columns)


def partial_correlation(frame, x_column, y_column, controls, rng):
    x = zscore(frame[x_column])
    y = zscore(frame[y_column])
    covariates = design_matrix(frame, controls)
    design = np.column_stack((np.ones(len(frame)), covariates))
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r = float(np.corrcoef(x_residual, y_residual)[0, 1])
    df = len(frame) - covariates.shape[1] - 2
    t_value = r * np.sqrt(df / max(1.0 - r * r, np.finfo(float).eps))
    p_value = float(2.0 * stats.t.sf(abs(t_value), df))

    permutation_r = np.empty(N_PERMUTATIONS, dtype=float)
    groups = frame["anatomy_group"].to_numpy()
    group_indices = [
        np.flatnonzero(groups == group) for group in np.unique(groups)
    ]
    for iteration in range(N_PERMUTATIONS):
        if "division" in controls:
            shuffled = x_residual.copy()
            for indices in group_indices:
                shuffled[indices] = x_residual[indices][
                    rng.permutation(len(indices))
                ]
        else:
            shuffled = x_residual[rng.permutation(len(x_residual))]
        permutation_r[iteration] = np.corrcoef(shuffled, y_residual)[0, 1]
    permutation_p = float(
        (1 + np.count_nonzero(np.abs(permutation_r) >= abs(r)))
        / (N_PERMUTATIONS + 1)
    )
    return r, p_value, permutation_p, df


def main():
    frame = pd.read_csv(INPUT).replace([np.inf, -np.inf], np.nan)
    required = [
        "spont_FCV",
        "stim_FCV",
        "stim_FCV_condition_sd",
        "FU_DCApost",
        "Hard_OO_fraction",
        "anatomy_group",
    ]
    frame = frame.dropna(subset=required).copy()
    frame["stim_minus_spont_FCV_z"] = (
        zscore(frame["stim_FCV"]) - zscore(frame["spont_FCV"])
    )
    rng = np.random.default_rng(SEED)

    analyses = [
        ("state_correspondence", "spont_FCV", "stim_FCV"),
        ("DCApost_mean_OMR", "FU_DCApost", "stim_FCV"),
        ("OO_mean_OMR", "Hard_OO_fraction", "stim_FCV"),
        (
            "DCApost_OMR_condition_SD",
            "FU_DCApost",
            "stim_FCV_condition_sd",
        ),
        (
            "OO_OMR_condition_SD",
            "Hard_OO_fraction",
            "stim_FCV_condition_sd",
        ),
        ("DCApost_stim_minus_spont", "FU_DCApost", "stim_minus_spont_FCV_z"),
        ("OO_stim_minus_spont", "Hard_OO_fraction", "stim_minus_spont_FCV_z"),
    ]
    rows = []
    for analysis, x_column, y_column in analyses:
        control_sets = (
            [()]
            if analysis == "state_correspondence"
            else [(), ("spont_FCV",), ("division",), ("spont_FCV", "division")]
        )
        for controls in control_sets:
            r, p_value, permutation_p, df = partial_correlation(
                frame, x_column, y_column, controls, rng
            )
            rows.append(
                {
                    "analysis": analysis,
                    "x": x_column,
                    "y": y_column,
                    "controls": "+".join(controls) if controls else "none",
                    "n_regions": len(frame),
                    "partial_r": r,
                    "p_value": p_value,
                    "permutation_p_two_sided": permutation_p,
                    "df": df,
                    "n_permutations": N_PERMUTATIONS,
                }
            )

    output = pd.DataFrame(rows)
    path = save_table(output, "omr_spontaneous_adjustment.csv")
    print(output.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
