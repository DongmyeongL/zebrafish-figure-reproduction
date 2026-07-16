#!/usr/bin/env python3
"""Compute the three Figure 9 FC measures from spatial functional units."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from common import CONFIG, DERIVED_DIR, ensure_output_dirs


def highpass_filter(traces: np.ndarray, rate_hz: float, cutoff_hz: float) -> np.ndarray:
    nyquist = 0.5 * rate_hz
    b, a = butter(2, cutoff_hz / nyquist, btype="high")
    return filtfilt(b, a, traces, axis=1)


def zscore_rows(traces: np.ndarray) -> np.ndarray:
    mean = np.nanmean(traces, axis=1, keepdims=True)
    sd = np.nanstd(traces, axis=1, keepdims=True)
    sd[sd <= 1e-12] = np.nan
    return np.nan_to_num((traces - mean) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def iter_corr_windows(traces: np.ndarray, window: int, step: int, diagonal: float):
    for start in range(0, traces.shape[1] - window + 1, step):
        corr = np.corrcoef(traces[:, start : start + window])
        np.fill_diagonal(corr, diagonal)
        yield corr.astype(np.float64, copy=False)


def row_corr_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_valid = a.shape[0] - 1
    sum_a = a.sum(axis=1)
    sum_b = b.sum(axis=1)
    sum_aa = np.square(a).sum(axis=1)
    sum_bb = np.square(b).sum(axis=1)
    sum_ab = (a * b).sum(axis=1)
    numerator = sum_ab - sum_a * sum_b / n_valid
    denominator = np.sqrt(
        np.maximum(sum_aa - sum_a * sum_a / n_valid, 0.0)
        * np.maximum(sum_bb - sum_b * sum_b / n_valid, 0.0)
    )
    corr = np.full(a.shape[0], np.nan)
    valid = denominator > 1e-12
    corr[valid] = np.clip(numerator[valid] / denominator[valid], -1.0, 1.0)
    return 1.0 - corr


def compute_measures(traces: np.ndarray, window: int, step: int) -> tuple[pd.DataFrame, int]:
    n_units = traces.shape[0]
    fc_sum = np.zeros((n_units, n_units), dtype=np.float64)
    fc2_sum = np.zeros_like(fc_sum)
    n_windows = 0
    for corr in iter_corr_windows(traces, window, step, diagonal=0.0):
        fc_sum += corr
        fc2_sum += corr * corr
        n_windows += 1
    if n_windows < 3:
        raise ValueError("At least three FC windows are required")

    mean_fc = fc_sum / n_windows
    edge_std = np.sqrt(np.maximum(fc2_sum / n_windows - mean_fc * mean_fc, 0.0))
    np.fill_diagonal(edge_std, 0.0)
    fcv = edge_std.sum(axis=1) / max(n_units - 1, 1)

    corr_dist_sum = np.zeros(n_units)
    corr_dist_count = np.zeros(n_units, dtype=int)
    fcs_windows = []
    for corr_zero_diag, corr_nan_diag in zip(
        iter_corr_windows(traces, window, step, diagonal=0.0),
        iter_corr_windows(traces, window, step, diagonal=np.nan),
    ):
        distance = row_corr_distance(corr_zero_diag, mean_fc)
        valid = np.isfinite(distance)
        corr_dist_sum[valid] += distance[valid]
        corr_dist_count[valid] += 1
        fcs_windows.append(np.nanmean(corr_nan_diag, axis=1))

    recon = np.divide(
        corr_dist_sum,
        corr_dist_count,
        out=np.full(n_units, np.nan),
        where=corr_dist_count > 0,
    )
    fcs = np.nanmean(np.vstack(fcs_windows), axis=0)
    return pd.DataFrame({"EdgeStdFCV": fcv, "FCS": fcs, "ProfileCorrDistFCV": recon}), n_windows


def main() -> None:
    ensure_output_dirs()
    rows = []
    qc_rows = []
    window = int(CONFIG["fc_window_frames"])
    step = int(CONFIG["fc_step_frames"])

    for path in sorted((DERIVED_DIR / "functional_unit_traces").glob("*_raw_cluster_traces.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        rate = float(raw["sampling_rate_hz"])
        traces = highpass_filter(np.asarray(raw["traces"], dtype=float), rate, float(CONFIG["highpass_hz"]))
        traces = zscore_rows(traces)
        values, n_windows = compute_measures(traces, window, step)
        values.insert(0, "root_area_name", list(raw["root_area_names"]))
        values.insert(0, "root_area_id", list(raw["root_area_ids"]))
        values.insert(0, "cluster_id", list(raw["cluster_ids"]))
        values.insert(0, "recording_id", str(raw["recording_id"]))
        rows.append(values)
        qc_rows.append(
            {
                "recording_id": raw["recording_id"],
                "n_functional_units": traces.shape[0],
                "n_root_areas": len(set(raw["root_area_ids"])),
                "n_timepoints": traces.shape[1],
                "n_windows": n_windows,
                "sampling_rate_hz": rate,
                "highpass_hz": CONFIG["highpass_hz"],
                "window": window,
                "step": step,
            }
        )

    units = pd.concat(rows, ignore_index=True)
    regions = (
        units.groupby(["recording_id", "root_area_id", "root_area_name"], as_index=False)
        .agg(
            n_clusters=("cluster_id", "size"),
            EdgeStdFCV=("EdgeStdFCV", "mean"),
            FCS=("FCS", "mean"),
            ProfileCorrDistFCV=("ProfileCorrDistFCV", "mean"),
        )
        .rename(columns={"root_area_name": "node"})
    )
    units.to_csv(DERIVED_DIR / "figure9_functional_unit_fc_measures.csv", index=False)
    regions.to_csv(DERIVED_DIR / "figure9_fc_measures_recording_region.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(DERIVED_DIR / "figure9_fc_qc.csv", index=False)
    print(f"Saved {len(regions)} recording-region rows across {regions['node'].nunique()} regions")


if __name__ == "__main__":
    main()
