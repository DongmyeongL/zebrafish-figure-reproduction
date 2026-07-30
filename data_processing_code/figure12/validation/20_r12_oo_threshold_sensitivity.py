#!/usr/bin/env python3
"""Test whether the primary zebrafish OO--FCV result depends on DCA > 0."""

from __future__ import annotations

import pickle
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve()
FIGURE12_CODE = HERE.parents[1]
PACK_ROOT = HERE.parents[3]
if str(FIGURE12_CODE) not in sys.path:
    sys.path.insert(0, str(FIGURE12_CODE))

from common import (  # noqa: E402
    CONFIG,
    N_REGIONS,
    REGION_NAMES,
    compact_sc_path,
    functional_unit_dca_path,
    original_sc_path,
)


SC_SOURCE = "fcs_calibrated_skeleton_kmeans_nearest_r12"
OUT_DIR = Path(
    os.environ.get(
        "ZF_OO_SENSITIVITY_DERIVED",
        PACK_ROOT
        / "derived_data"
        / "figure12"
        / "validation"
        / "r12_primary"
        / "oo_threshold_sensitivity",
    )
)
STATS_DIR = Path(
    os.environ.get(
        "ZF_OO_SENSITIVITY_STATS", PACK_ROOT / "statistics" / "r12_primary"
    )
)
FIGURE_DIR = Path(
    os.environ.get(
        "ZF_OO_SENSITIVITY_FIGURES",
        PACK_ROOT / "figures" / "validation" / "r12_primary",
    )
)
REGION_LIST = (
    PACK_ROOT
    / "derived_data"
    / "common"
    / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
FCV_FILE = PACK_ROOT / "derived_data" / "figure9" / "figure9_region_summary.csv"

SD_THRESHOLDS = np.array(
    [-0.50, -0.35, -0.25, -0.15, -0.10, -0.05, 0.0,
     0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
)
PERCENTILE_THRESHOLDS = np.array([25, 35, 40, 45, 50, 55, 60, 65, 75])
EXCLUSION_BANDS = np.array([0.0, 0.025, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50])
SOFT_TEMPERATURES = np.array([0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0])
N_RESAMPLES = 10_000
SEED = 20260730


def functional_unit_membership(subject: int, neuron_region: np.ndarray):
    with original_sc_path(subject).open("rb") as handle:
        raw = pickle.load(handle)
    membership = np.full(len(neuron_region), -1, dtype=np.int32)
    unit_region = np.asarray(raw["root_area"], dtype=np.int64)
    for unit_id, cells in enumerate(raw["final_id_cluster"]):
        membership[np.asarray(cells, dtype=np.int64)] = unit_id
    if (membership < 0).any():
        raise ValueError(f"Subject {subject}: incomplete anatomical-unit membership")
    if not np.array_equal(unit_region[membership], neuron_region):
        raise ValueError(f"Subject {subject}: anatomical-unit membership mismatch")
    return membership, unit_region


def aggregate_unit_adjacency(edges, membership, unit_region, chunk_size=2_000_000):
    n_units = len(unit_region)
    counts = np.zeros(n_units * n_units, dtype=np.float64)
    for start in range(0, len(edges), chunk_size):
        chunk = edges[start:start + chunk_size]
        source = membership[chunk[:, 0]]
        target = membership[chunk[:, 1]]
        keep = (source != target) & (unit_region[source] != unit_region[target])
        code = source[keep].astype(np.int64) * n_units + target[keep]
        counts += np.bincount(code, minlength=n_units * n_units)
    return counts.reshape(n_units, n_units)


def weighted_region_mean(unit_values, unit_region, unit_weights):
    valid = np.isfinite(unit_values) & np.isfinite(unit_weights) & (unit_weights > 0)
    numerator = np.bincount(
        unit_region,
        weights=np.where(valid, unit_values * unit_weights, 0.0),
        minlength=N_REGIONS,
    )
    denominator = np.bincount(
        unit_region,
        weights=np.where(valid, unit_weights, 0.0),
        minlength=N_REGIONS,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(N_REGIONS, np.nan),
        where=denominator > 0,
    )


def hard_metrics(adjacency, dca, unit_region, threshold):
    positive = dca > threshold
    source_weight = adjacency.sum(axis=1)
    target_positive_weight = adjacency @ positive.astype(float)
    oo_unit_weight = positive.astype(float) * target_positive_weight
    source_positive = weighted_region_mean(positive.astype(float), unit_region, source_weight)
    target_positive = weighted_region_mean(
        np.divide(
            target_positive_weight,
            source_weight,
            out=np.full(len(dca), np.nan),
            where=source_weight > 0,
        ),
        unit_region,
        source_weight,
    )
    oo = weighted_region_mean(
        np.divide(
            oo_unit_weight,
            source_weight,
            out=np.full(len(dca), np.nan),
            where=source_weight > 0,
        ),
        unit_region,
        source_weight,
    )
    conditional = np.divide(
        oo,
        source_positive,
        out=np.full(N_REGIONS, np.nan),
        where=source_positive > 0,
    )
    expected = source_positive * target_positive
    return {
        "OO": oo,
        "source_positive": source_positive,
        "target_positive": target_positive,
        "conditional_target_positive": conditional,
        "OO_minus_independence": oo - expected,
    }


def exclusion_metric(adjacency, dca, unit_region, band):
    eligible = np.abs(dca) > band
    positive = dca > band
    eligible_edges = adjacency * np.outer(eligible, eligible)
    oo_edges = eligible_edges * np.outer(positive, positive)
    denominator_unit = eligible_edges.sum(axis=1)
    oo_unit = oo_edges.sum(axis=1)
    oo = weighted_region_mean(
        np.divide(
            oo_unit,
            denominator_unit,
            out=np.full(len(dca), np.nan),
            where=denominator_unit > 0,
        ),
        unit_region,
        denominator_unit,
    )
    return oo, float(eligible_edges.sum() / adjacency.sum())


def soft_metric(adjacency, dca, unit_region, temperature):
    scaled = np.clip(dca / temperature, -40, 40)
    output_probability = 1.0 / (1.0 + np.exp(-scaled))
    source_weight = adjacency.sum(axis=1)
    soft_target = adjacency @ output_probability
    soft_oo_weight = output_probability * soft_target
    return weighted_region_mean(
        np.divide(
            soft_oo_weight,
            source_weight,
            out=np.full(len(dca), np.nan),
            where=source_weight > 0,
        ),
        unit_region,
        source_weight,
    )


def correlation_summary(x, y, rng):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    r, p = stats.pearsonr(x, y)
    n = len(x)
    indices = rng.integers(0, n, size=(N_RESAMPLES, n))
    xb, yb = x[indices], y[indices]
    xb -= xb.mean(axis=1, keepdims=True)
    yb -= yb.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.square(xb).sum(axis=1) * np.square(yb).sum(axis=1))
    boot = np.divide((xb * yb).sum(axis=1), denom, out=np.full(N_RESAMPLES, np.nan), where=denom > 0)
    perm = np.empty(N_RESAMPLES)
    x_centered = x - x.mean()
    x_norm = np.sqrt(np.square(x_centered).sum())
    for iteration in range(N_RESAMPLES):
        shuffled = y[rng.permutation(n)]
        shuffled -= shuffled.mean()
        perm[iteration] = np.dot(x_centered, shuffled) / (x_norm * np.sqrt(np.square(shuffled).sum()))
    permutation_p = (1 + np.count_nonzero(np.abs(perm) >= abs(r))) / (N_RESAMPLES + 1)
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "n_regions": n,
        "pearson_r": float(r),
        "pearson_p": float(p),
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "permutation_p_two_sided": float(permutation_p),
    }


def partial_correlation(x, y, covariate, rng):
    covariate = np.asarray(covariate)
    if covariate.ndim == 1:
        covariate = covariate[:, None]
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(covariate).all(axis=1)
    x, y, z = np.asarray(x)[valid], np.asarray(y)[valid], covariate[valid]
    design = np.column_stack((np.ones(len(z)), z))
    x_resid = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return correlation_summary(x_resid, y_resid, rng)


def subject_metrics():
    hard_rows, exclusion_rows, soft_rows = [], [], []
    for subject in CONFIG["subjects"]:
        print(f"[OO sensitivity] subject {subject}", flush=True)
        payload = np.load(compact_sc_path(subject, SC_SOURCE), allow_pickle=False)
        edges = np.asarray(payload["edges"])
        neuron_region = np.asarray(payload["neuron_region"], dtype=np.int64)
        membership, unit_region = functional_unit_membership(subject, neuron_region)
        adjacency = aggregate_unit_adjacency(edges, membership, unit_region)
        unit_data = np.load(functional_unit_dca_path(subject, SC_SOURCE), allow_pickle=False)
        dca = np.asarray(unit_data["dca"], dtype=float)
        dca_scale = float(np.std(dca))

        for threshold_sd in SD_THRESHOLDS:
            threshold = threshold_sd * dca_scale
            for measure, values in hard_metrics(adjacency, dca, unit_region, threshold).items():
                for region_id, value in enumerate(values):
                    hard_rows.append({
                        "Subject": subject,
                        "RegionID": region_id,
                        "node": REGION_NAMES[region_id],
                        "threshold_type": "dca_sd",
                        "threshold_value": threshold_sd,
                        "measure": measure,
                        "value": value,
                    })

        for percentile in PERCENTILE_THRESHOLDS:
            threshold = float(np.percentile(dca, percentile))
            for measure, values in hard_metrics(adjacency, dca, unit_region, threshold).items():
                for region_id, value in enumerate(values):
                    hard_rows.append({
                        "Subject": subject,
                        "RegionID": region_id,
                        "node": REGION_NAMES[region_id],
                        "threshold_type": "dca_percentile",
                        "threshold_value": percentile,
                        "measure": measure,
                        "value": value,
                    })

        for band_sd in EXCLUSION_BANDS:
            values, retained = exclusion_metric(adjacency, dca, unit_region, band_sd * dca_scale)
            for region_id, value in enumerate(values):
                exclusion_rows.append({
                    "Subject": subject,
                    "RegionID": region_id,
                    "node": REGION_NAMES[region_id],
                    "exclusion_band_sd": band_sd,
                    "OO": value,
                    "retained_edge_fraction": retained,
                })

        for temperature_sd in SOFT_TEMPERATURES:
            values = soft_metric(adjacency, dca, unit_region, temperature_sd * dca_scale)
            for region_id, value in enumerate(values):
                soft_rows.append({
                    "Subject": subject,
                    "RegionID": region_id,
                    "node": REGION_NAMES[region_id],
                    "temperature_sd": temperature_sd,
                    "soft_OO": value,
                })
    return pd.DataFrame(hard_rows), pd.DataFrame(exclusion_rows), pd.DataFrame(soft_rows)


def matched_fcv():
    regions = pd.read_csv(REGION_LIST)[["legacy_order", "root_area_id", "node", "anatomy_group"]]
    fcv = pd.read_csv(FCV_FILE)[["root_area_id", "node", "EdgeStdFCV"]]
    return regions.merge(fcv, on=["root_area_id", "node"], how="left", validate="one_to_one")


def summarize_correlations(hard, exclusion, soft, fcv):
    rng = np.random.default_rng(SEED)
    rows = []

    hard_mean = hard.groupby(
        ["threshold_type", "threshold_value", "measure", "RegionID", "node"], as_index=False
    )["value"].mean()
    for keys, group in hard_mean.groupby(["threshold_type", "threshold_value", "measure"]):
        threshold_type, threshold_value, measure = keys
        merged = fcv.merge(group, left_on=["root_area_id", "node"], right_on=["RegionID", "node"], how="left")
        summary = correlation_summary(merged["value"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng)
        rows.append({"analysis": threshold_type, "parameter": threshold_value, "measure": measure, **summary})

    exclusion_mean = exclusion.groupby(["exclusion_band_sd", "RegionID", "node"], as_index=False).agg(
        value=("OO", "mean"), retained_edge_fraction=("retained_edge_fraction", "mean")
    )
    for band, group in exclusion_mean.groupby("exclusion_band_sd"):
        merged = fcv.merge(group, left_on=["root_area_id", "node"], right_on=["RegionID", "node"], how="left")
        summary = correlation_summary(merged["value"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng)
        rows.append({
            "analysis": "near_zero_exclusion",
            "parameter": band,
            "measure": "OO",
            "retained_edge_fraction": group["retained_edge_fraction"].mean(),
            **summary,
        })

    soft_mean = soft.groupby(["temperature_sd", "RegionID", "node"], as_index=False)["soft_OO"].mean()
    for temperature, group in soft_mean.groupby("temperature_sd"):
        merged = fcv.merge(group, left_on=["root_area_id", "node"], right_on=["RegionID", "node"], how="left")
        summary = correlation_summary(merged["soft_OO"].to_numpy(), merged["EdgeStdFCV"].to_numpy(), rng)
        rows.append({"analysis": "soft_oo", "parameter": temperature, "measure": "soft_OO", **summary})

    primary = hard_mean[
        hard_mean["threshold_type"].eq("dca_sd")
        & hard_mean["threshold_value"].eq(0.0)
    ].pivot(index=["RegionID", "node"], columns="measure", values="value").reset_index()
    primary = fcv.merge(primary, left_on=["root_area_id", "node"], right_on=["RegionID", "node"], how="left")
    for measure in ["OO", "source_positive", "target_positive", "conditional_target_positive", "OO_minus_independence"]:
        summary = correlation_summary(primary[measure].to_numpy(), primary["EdgeStdFCV"].to_numpy(), rng)
        rows.append({"analysis": "primary_decomposition", "parameter": 0.0, "measure": measure, **summary})
    controls = {
        "OO_controlling_source_positive": primary[["source_positive"]].to_numpy(),
        "OO_controlling_target_positive": primary[["target_positive"]].to_numpy(),
        "OO_controlling_source_and_target_positive": primary[
            ["source_positive", "target_positive"]
        ].to_numpy(),
    }
    for label, covariates in controls.items():
        partial = partial_correlation(
            primary["OO"].to_numpy(),
            primary["EdgeStdFCV"].to_numpy(),
            covariates,
            rng,
        )
        rows.append({
            "analysis": "primary_partial",
            "parameter": 0.0,
            "measure": label,
            **partial,
        })
    return pd.DataFrame(rows), primary


def plot_summary(summary):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)

    def curve(ax, frame, x_label, title):
        frame = frame.sort_values("parameter")
        ax.fill_between(frame["parameter"], frame["bootstrap_ci_low"], frame["bootstrap_ci_high"], color="#8ecae6", alpha=0.35)
        ax.plot(frame["parameter"], frame["pearson_r"], color="#126782", marker="o", ms=4)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(xlabel=x_label, ylabel="Pearson r with FCV", title=title)

    curve(axes[0, 0], summary.query("analysis == 'dca_sd' and measure == 'OO'"), "DCA threshold (subject SD)", "A  Absolute threshold")
    axes[0, 0].axvline(0, color="#d62828", ls="--", lw=1)
    curve(axes[0, 1], summary.query("analysis == 'dca_percentile' and measure == 'OO'"), "DCA percentile threshold", "B  Percentile threshold")
    curve(axes[0, 2], summary.query("analysis == 'near_zero_exclusion'"), "Excluded |DCA| band (subject SD)", "C  Near-zero exclusion")
    curve(axes[1, 0], summary.query("analysis == 'soft_oo'"), "Sigmoid temperature (subject SD)", "D  Continuous soft OO")

    decomposition = summary.query("analysis == 'primary_decomposition'").copy()
    order = ["OO", "source_positive", "target_positive", "conditional_target_positive", "OO_minus_independence"]
    labels = ["Hard OO", "Positive source", "Positive target", "Target | source+", "OO enrichment"]
    decomposition["order"] = decomposition["measure"].map(dict(zip(order, range(len(order)))))
    decomposition = decomposition.sort_values("order")
    y = np.arange(len(decomposition))
    axes[1, 1].errorbar(
        decomposition["pearson_r"], y,
        xerr=np.vstack((decomposition["pearson_r"] - decomposition["bootstrap_ci_low"], decomposition["bootstrap_ci_high"] - decomposition["pearson_r"])),
        fmt="o", color="#126782", capsize=2,
    )
    axes[1, 1].axvline(0, color="0.5", lw=0.8)
    axes[1, 1].set(yticks=y, yticklabels=labels, xlabel="Pearson r with FCV", title="E  Primary OO decomposition")
    axes[1, 1].invert_yaxis()

    pframe = summary.query("analysis == 'primary_partial'")
    hard = summary.query("analysis == 'primary_decomposition' and measure == 'OO'")
    partial_labels = {
        "OO_controlling_source_positive": "OO | source+",
        "OO_controlling_target_positive": "OO | target+",
        "OO_controlling_source_and_target_positive": "OO | source+, target+",
    }
    pframe = pframe.assign(label=pframe["measure"].map(partial_labels))
    values = pd.concat((hard.assign(label="Hard OO"), pframe), ignore_index=True)
    y = np.arange(len(values))
    axes[1, 2].errorbar(
        values["pearson_r"], y,
        xerr=np.vstack((values["pearson_r"] - values["bootstrap_ci_low"], values["bootstrap_ci_high"] - values["pearson_r"])),
        fmt="o", color="#6a994e", capsize=2,
    )
    axes[1, 2].axvline(0, color="0.5", lw=0.8)
    axes[1, 2].set(yticks=y, yticklabels=values["label"], xlabel="Pearson / partial r", title="F  Composition controls")
    axes[1, 2].invert_yaxis()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    hard, exclusion, soft = subject_metrics()
    fcv = matched_fcv()
    summary, primary = summarize_correlations(hard, exclusion, soft, fcv)

    hard.to_csv(OUT_DIR / "oo_threshold_subject_region_values.csv", index=False)
    exclusion.to_csv(OUT_DIR / "oo_near_zero_exclusion_subject_region_values.csv", index=False)
    soft.to_csv(OUT_DIR / "soft_oo_subject_region_values.csv", index=False)
    primary.to_csv(OUT_DIR / "oo_primary_decomposition_region_input.csv", index=False)
    summary_path = STATS_DIR / "r12_oo_threshold_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    figure_path = FIGURE_DIR / "figure12_oo_threshold_sensitivity.png"
    figure = plot_summary(summary)
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
