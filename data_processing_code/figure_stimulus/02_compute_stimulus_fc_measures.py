#!/usr/bin/env python3
"""Compute OMR FCV and signed FCS from compact root-area traces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import CONFIG, DERIVED_DIR, ensure_output_dirs


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=1, keepdims=True)
    sd = np.nanstd(matrix, axis=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = np.nan
    return (matrix - mean) / sd


def zscore_vector(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(np.nan, index=values.index)
    return (values - values.mean()) / sd


def compute_fcv_fcs(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    traces = np.asarray(traces, dtype=float)
    n_regions = traces.shape[0]
    valid = np.isfinite(traces).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    valid[valid_indices] &= np.std(traces[valid_indices], axis=1) > 1e-12
    valid_traces = zscore_rows(traces[valid])
    window = int(CONFIG["fc_window_frames"])
    step = int(CONFIG["fc_step_frames"])
    correlations = [
        np.corrcoef(valid_traces[:, start : start + window])
        for start in range(0, valid_traces.shape[1] - window + 1, step)
    ]
    if not correlations:
        return np.full(n_regions, np.nan), np.full(n_regions, np.nan), 0

    stack = np.asarray(correlations)
    edge_sd = np.nanstd(stack, axis=0)
    edge_mean = np.nanmean(stack, axis=0)
    np.fill_diagonal(edge_sd, np.nan)
    np.fill_diagonal(edge_mean, np.nan)
    fcv = np.full(n_regions, np.nan)
    fcs = np.full(n_regions, np.nan)
    fcv[valid] = np.nanmean(edge_sd, axis=1)
    fcs[valid] = np.nanmean(edge_mean, axis=1)
    return fcv, fcs, len(stack)


def main() -> None:
    ensure_output_dirs()
    rows = []
    for subject in CONFIG["subjects"]:
        source = (
            DERIVED_DIR / "region_traces" / f"subject_{subject}_stimulus_region_traces.npz"
        )
        with np.load(source, allow_pickle=False) as data:
            region_ids = data["region_ids"].astype(int)
            region_names = data["region_names"].astype(str)
            traces = data["region_traces"].astype(float)
            neuron_counts = data["neuron_counts"].astype(int)
            stimulus = data["stimulus_array"].astype(int)

        for stimulus_index in CONFIG["stimulus_indices"]:
            frame_indices = np.flatnonzero(stimulus == int(stimulus_index))
            fcv, fcs, n_windows = compute_fcv_fcs(traces[:, frame_indices])
            for position, (region_id, node) in enumerate(zip(region_ids, region_names)):
                rows.append(
                    {
                        "subject": int(subject),
                        "stimulus_index": int(stimulus_index),
                        "stimulus_label": CONFIG["stimulus_labels"][str(stimulus_index)],
                        "root_area_id": int(region_id),
                        "node": str(node),
                        "n_neurons": int(neuron_counts[position]),
                        "n_frames": int(frame_indices.size),
                        "n_windows": int(n_windows),
                        "FCV_raw": float(fcv[position]),
                        "FCS_raw": float(fcs[position]),
                    }
                )

    measures = pd.DataFrame(rows)
    keys = ["subject", "stimulus_index"]
    measures["FCV_z"] = measures.groupby(keys, group_keys=False)["FCV_raw"].transform(
        zscore_vector
    )
    measures["FCS_z"] = measures.groupby(keys, group_keys=False)["FCS_raw"].transform(
        zscore_vector
    )
    measures.to_csv(
        DERIVED_DIR / "stimulus_fc_measures_subject_condition_region.csv", index=False
    )

    summary = (
        measures.groupby(["root_area_id", "node"], as_index=False)
        .agg(
            n_subject_condition=("FCV_z", "count"),
            n_subjects=("subject", "nunique"),
            n_conditions=("stimulus_index", "nunique"),
            FCV=("FCV_z", "mean"),
            FCS=("FCS_z", "mean"),
            FCV_raw_mean=("FCV_raw", "mean"),
            FCS_raw_mean=("FCS_raw", "mean"),
        )
        .sort_values("root_area_id")
    )
    summary["complete_all7_all3"] = (
        summary["n_subject_condition"].eq(21)
        & summary["n_subjects"].eq(7)
        & summary["n_conditions"].eq(3)
        & summary[["FCV", "FCS"]].notna().all(axis=1)
    )
    summary.to_csv(DERIVED_DIR / "stimulus_fc_region_summary.csv", index=False)
    print(
        f"Saved {len(measures)} subject-condition-region rows and "
        f"{len(summary)} region summaries"
    )


if __name__ == "__main__":
    main()
