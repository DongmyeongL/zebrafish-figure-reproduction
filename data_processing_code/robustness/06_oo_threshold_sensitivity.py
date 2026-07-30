#!/usr/bin/env python3
"""Reproduce OO-threshold, near-zero exclusion, and soft-OO controls."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA, save_table


INPUT = DATA / "oo_threshold_sensitivity"
N_RESAMPLES = int(os.environ.get("ZF_OO_SENSITIVITY_RESAMPLES", "10000"))
SEED = 20260730


def correlation_summary(x, y, rng):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, dtype=float)[valid], np.asarray(y, dtype=float)[valid]
    r, p_value = stats.pearsonr(x, y)
    n = len(x)

    indices = rng.integers(0, n, size=(N_RESAMPLES, n))
    x_boot, y_boot = x[indices], y[indices]
    x_boot -= x_boot.mean(axis=1, keepdims=True)
    y_boot -= y_boot.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.square(x_boot).sum(axis=1) * np.square(y_boot).sum(axis=1)
    )
    bootstrap_r = np.divide(
        (x_boot * y_boot).sum(axis=1),
        denominator,
        out=np.full(N_RESAMPLES, np.nan),
        where=denominator > 0,
    )

    permutation_r = np.empty(N_RESAMPLES, dtype=float)
    x_centered = x - x.mean()
    x_norm = np.sqrt(np.square(x_centered).sum())
    for iteration in range(N_RESAMPLES):
        shuffled = y[rng.permutation(n)]
        shuffled -= shuffled.mean()
        permutation_r[iteration] = np.dot(x_centered, shuffled) / (
            x_norm * np.sqrt(np.square(shuffled).sum())
        )
    permutation_p = float(
        (1 + np.count_nonzero(np.abs(permutation_r) >= abs(r)))
        / (N_RESAMPLES + 1)
    )
    low, high = np.nanpercentile(bootstrap_r, [2.5, 97.5])
    return {
        "n_regions": n,
        "pearson_r": float(r),
        "pearson_p": float(p_value),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "permutation_p_two_sided": permutation_p,
    }


def partial_correlation(x, y, covariate, rng):
    covariate = np.asarray(covariate, dtype=float)
    if covariate.ndim == 1:
        covariate = covariate[:, None]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(covariate).all(axis=1)
    x, y, z = np.asarray(x)[valid], np.asarray(y)[valid], covariate[valid]
    design = np.column_stack((np.ones(len(z)), z))
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return correlation_summary(x_residual, y_residual, rng)


def load_inputs():
    hard = pd.read_csv(INPUT / "oo_threshold_subject_region_values.csv")
    exclusion = pd.read_csv(INPUT / "oo_near_zero_exclusion_subject_region_values.csv")
    soft = pd.read_csv(INPUT / "soft_oo_subject_region_values.csv")
    primary = pd.read_csv(INPUT / "oo_primary_decomposition_region_input.csv")
    fcv = primary[["root_area_id", "node", "EdgeStdFCV"]].copy()
    return hard, exclusion, soft, fcv


def summarize_correlations(hard, exclusion, soft, fcv):
    rng = np.random.default_rng(SEED)
    rows = []

    hard_mean = hard.groupby(
        ["threshold_type", "threshold_value", "measure", "RegionID", "node"],
        as_index=False,
    )["value"].mean()
    for keys, group in hard_mean.groupby(
        ["threshold_type", "threshold_value", "measure"]
    ):
        threshold_type, threshold_value, measure = keys
        merged = fcv.merge(
            group,
            left_on=["root_area_id", "node"],
            right_on=["RegionID", "node"],
            how="left",
        )
        summary = correlation_summary(
            merged["value"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng
        )
        rows.append(
            {
                "analysis": threshold_type,
                "parameter": threshold_value,
                "measure": measure,
                **summary,
            }
        )

    exclusion_mean = exclusion.groupby(
        ["exclusion_band_sd", "RegionID", "node"], as_index=False
    ).agg(
        value=("OO", "mean"),
        retained_edge_fraction=("retained_edge_fraction", "mean"),
    )
    for band, group in exclusion_mean.groupby("exclusion_band_sd"):
        merged = fcv.merge(
            group,
            left_on=["root_area_id", "node"],
            right_on=["RegionID", "node"],
            how="left",
        )
        summary = correlation_summary(
            merged["value"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng
        )
        rows.append(
            {
                "analysis": "near_zero_exclusion",
                "parameter": band,
                "measure": "OO",
                "retained_edge_fraction": group["retained_edge_fraction"].mean(),
                **summary,
            }
        )

    soft_mean = soft.groupby(
        ["temperature_sd", "RegionID", "node"], as_index=False
    )["soft_OO"].mean()
    for temperature, group in soft_mean.groupby("temperature_sd"):
        merged = fcv.merge(
            group,
            left_on=["root_area_id", "node"],
            right_on=["RegionID", "node"],
            how="left",
        )
        summary = correlation_summary(
            merged["soft_OO"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng
        )
        rows.append(
            {
                "analysis": "soft_oo",
                "parameter": temperature,
                "measure": "soft_OO",
                **summary,
            }
        )

    primary = hard_mean[
        hard_mean["threshold_type"].eq("dca_sd")
        & hard_mean["threshold_value"].eq(0.0)
    ].pivot(index=["RegionID", "node"], columns="measure", values="value").reset_index()
    primary = fcv.merge(
        primary,
        left_on=["root_area_id", "node"],
        right_on=["RegionID", "node"],
        how="left",
    )
    measures = [
        "OO",
        "source_positive",
        "target_positive",
        "conditional_target_positive",
        "OO_minus_independence",
    ]
    for measure in measures:
        summary = correlation_summary(
            primary[measure].to_numpy(), primary["EdgeStdFCV"].to_numpy(), rng
        )
        rows.append(
            {
                "analysis": "primary_decomposition",
                "parameter": 0.0,
                "measure": measure,
                **summary,
            }
        )

    controls = {
        "OO_controlling_source_positive": primary[["source_positive"]].to_numpy(),
        "OO_controlling_target_positive": primary[["target_positive"]].to_numpy(),
        "OO_controlling_source_and_target_positive": primary[
            ["source_positive", "target_positive"]
        ].to_numpy(),
    }
    for label, covariates in controls.items():
        summary = partial_correlation(
            primary["OO"].to_numpy(),
            primary["EdgeStdFCV"].to_numpy(),
            covariates,
            rng,
        )
        rows.append(
            {
                "analysis": "primary_partial",
                "parameter": 0.0,
                "measure": label,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def main():
    hard, exclusion, soft, fcv = load_inputs()
    output = summarize_correlations(hard, exclusion, soft, fcv)
    path = save_table(output, "oo_threshold_sensitivity.csv")
    print(output.to_string(index=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
