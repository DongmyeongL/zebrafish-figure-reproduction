#!/usr/bin/env python3
"""Validate zebrafish NetTE with whole-network circular-shift surrogates.

Each functional-unit trace is shifted independently. This preserves its value
distribution and circular autocorrelation while disrupting its temporal
alignment with all other units. The TE estimator is imported directly from the
Figure 9 processing pipeline so the validation and main analysis use identical
filtering, discretization, bin count, and lag.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PACK_ROOT = Path(__file__).resolve().parents[2]
DERIVED_ROOT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK_ROOT / "derived_data")).resolve()
FIGURE9_CONFIG = json.loads((PACK_ROOT / "config" / "figure9.json").read_text())
CONFIG = json.loads((PACK_ROOT / "config" / "figure_supply_13.json").read_text())
TRACE_DIR = DERIVED_ROOT / "figure9" / "functional_unit_traces"
REGION_FILE = (
    DERIVED_ROOT / "common" / "legacy_stimulus_forest_42_regions_no_rOB.csv"
)
FUNCTIONAL_FILE = (
    DERIVED_ROOT / "figure9" / "figure9_recording_region_measures.csv"
)
OUTPUT_DIR = DERIVED_ROOT / "figure_supply_13"
STATS_DIR = PACK_ROOT / "statistics"
TE_CODE = (
    PACK_ROOT / "data_processing_code" / "figure9" / "03_prepare_te_measures.py"
)


def load_te_module():
    spec = importlib.util.spec_from_file_location("figure9_te_estimator", TE_CODE)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TE_CODE.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def empirical_upper_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def process_recording(path_string: str) -> dict:
    path = Path(path_string)
    te_module = load_te_module()
    with path.open("rb") as handle:
        raw = pickle.load(handle)

    traces = np.asarray(raw["traces"], dtype=float)
    traces = te_module.highpass_filter(
        traces,
        float(raw["sampling_rate_hz"]),
        float(FIGURE9_CONFIG["highpass_hz"]),
    )
    traces = te_module.zscore_rows(traces)
    bins = int(FIGURE9_CONFIG["te_bins"])
    lag = int(FIGURE9_CONFIG["te_lag"])
    observed_net, observed_drive = te_module.net_te_from_traces(traces, bins, lag)

    n_units, n_time = traces.shape
    upper_i, upper_j = np.triu_indices(n_units, k=1)
    valid = np.isfinite(observed_net[upper_i, upper_j])
    upper_i, upper_j = upper_i[valid], upper_j[valid]
    observed_abs = np.abs(observed_net[upper_i, upper_j])

    n_surrogates = int(CONFIG["n_surrogates"])
    minimum_shift = int(CONFIG["minimum_circular_shift_frames"])
    if n_time <= 2 * minimum_shift:
        raise RuntimeError(f"Trace too short for circular shifts: {path}")
    subject_number = int(str(raw["recording_id"]).split("_")[-1])
    rng = np.random.default_rng(int(CONFIG["random_seed"]) + subject_number)
    null_abs = np.empty((n_surrogates, observed_abs.size), dtype=np.float32)

    for surrogate_index in range(n_surrogates):
        offsets = rng.integers(minimum_shift, n_time - minimum_shift + 1, size=n_units)
        shifted = np.vstack(
            [np.roll(traces[unit], int(offsets[unit])) for unit in range(n_units)]
        )
        surrogate_net, _ = te_module.net_te_from_traces(shifted, bins, lag)
        null_abs[surrogate_index] = np.abs(surrogate_net[upper_i, upper_j])

    observed_global = float(np.mean(observed_abs))
    null_global = np.mean(null_abs, axis=1).astype(float)
    # Cross-fit pairwise thresholds: each surrogate half is evaluated against
    # thresholds estimated from the other half, avoiding reuse of a surrogate
    # to define and assess its own null threshold.
    split = n_surrogates // 2
    threshold_first = np.quantile(null_abs[:split], 0.95, axis=0)
    threshold_second = np.quantile(null_abs[split:], 0.95, axis=0)
    observed_exceedance = float(
        0.5 * np.mean(observed_abs > threshold_first)
        + 0.5 * np.mean(observed_abs > threshold_second)
    )
    null_exceedance = np.concatenate(
        [
            np.mean(null_abs[:split] > threshold_second[None, :], axis=1),
            np.mean(null_abs[split:] > threshold_first[None, :], axis=1),
        ]
    ).astype(float)

    # The example is selected transparently as the largest-magnitude pair in
    # the prespecified example animal; it is illustrative, not an extra test.
    unit_nodes = np.asarray(raw["root_area_names"]).astype(str)
    interregional = unit_nodes[upper_i] != unit_nodes[upper_j]
    if not interregional.any():
        raise RuntimeError(f"No inter-regional functional-unit pairs in {path}")
    candidate_indices = np.flatnonzero(interregional)
    example_index = int(candidate_indices[np.argmax(observed_abs[interregional])])
    source = int(upper_i[example_index])
    target = int(upper_j[example_index])
    if observed_net[source, target] < 0:
        source, target = target, source

    unit_table = pd.DataFrame(
        {
            "recording_id": str(raw["recording_id"]),
            "cluster_id": np.asarray(raw["cluster_ids"]),
            "root_area_id": np.asarray(raw["root_area_ids"]),
            "node": np.asarray(raw["root_area_names"]).astype(str),
            "NetTE": observed_drive,
        }
    )
    region_table = (
        unit_table.groupby(["recording_id", "root_area_id", "node"], as_index=False)
        .agg(NetTE=("NetTE", "mean"), n_functional_units=("cluster_id", "size"))
    )

    return {
        "recording_id": str(raw["recording_id"]),
        "n_units": n_units,
        "n_timepoints": n_time,
        "n_pairs": observed_abs.size,
        "observed_global": observed_global,
        "null_global": null_global,
        "global_p": empirical_upper_p(observed_global, null_global),
        "observed_exceedance": observed_exceedance,
        "null_exceedance": null_exceedance,
        "exceedance_p": empirical_upper_p(observed_exceedance, null_exceedance),
        "region_table": region_table,
        "observed_matrix": observed_net.astype(np.float32),
        "unit_nodes": unit_nodes,
        "unit_clusters": np.asarray(raw["cluster_ids"]),
        "example_source": source,
        "example_target": target,
        "example_observed": float(observed_net[source, target]),
        "example_null": null_abs[:, example_index].astype(float),
    }


def zscore(values: pd.Series) -> pd.Series:
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - values.mean()) / sd


def build_loo_table(region_animal: pd.DataFrame) -> pd.DataFrame:
    functional = pd.read_csv(FUNCTIONAL_FILE)
    canonical = pd.read_csv(REGION_FILE)[["root_area_id", "node", "anatomy_group"]]
    functional = functional.merge(
        canonical, on=["root_area_id", "node", "anatomy_group"], how="inner"
    )
    data = functional[["recording_id", "node", "EdgeStdFCV"]].merge(
        region_animal[["recording_id", "node", "NetTE"]],
        on=["recording_id", "node"],
        how="inner",
        validate="one_to_one",
    )
    data["FCV_z"] = data.groupby("recording_id")["EdgeStdFCV"].transform(zscore)
    data["NetTE_z"] = data.groupby("recording_id")["NetTE"].transform(zscore)
    animals = sorted(data["recording_id"].unique())
    rows = []
    for omitted in [None] + animals:
        subset = data if omitted is None else data.loc[data["recording_id"] != omitted]
        summary = subset.groupby("node", as_index=False)[["FCV_z", "NetTE_z"]].mean()
        result = pearsonr(summary["FCV_z"], summary["NetTE_z"])
        rows.append(
            {
                "omitted_recording": "none_full_sample" if omitted is None else omitted,
                "n_animals": subset["recording_id"].nunique(),
                "n_regions": len(summary),
                "r": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(TRACE_DIR.glob("*_raw_cluster_traces.pkl"))
    if len(paths) != 7:
        raise RuntimeError(f"Expected seven trace files, found {len(paths)}")

    results = []
    with ProcessPoolExecutor(max_workers=min(7, len(paths))) as executor:
        futures = {executor.submit(process_recording, str(path)): path for path in paths}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['recording_id']}: global p={result['global_p']:.4f}, "
                f"extreme-pair fraction={result['observed_exceedance']:.3f}",
                flush=True,
            )
    results.sort(key=lambda item: item["recording_id"])

    animal_rows = []
    null_rows = []
    region_tables = []
    for result in results:
        animal_rows.append(
            {key: result[key] for key in [
                "recording_id", "n_units", "n_timepoints", "n_pairs",
                "observed_global", "global_p", "observed_exceedance", "exceedance_p",
            ]}
        )
        for index, (global_value, exceedance_value) in enumerate(
            zip(result["null_global"], result["null_exceedance"])
        ):
            null_rows.append(
                {
                    "recording_id": result["recording_id"],
                    "surrogate": index + 1,
                    "mean_abs_NetTE": global_value,
                    "extreme_pair_fraction": exceedance_value,
                }
            )
        region_tables.append(result["region_table"])

    animal_table = pd.DataFrame(animal_rows)
    null_table = pd.DataFrame(null_rows)
    region_animal = pd.concat(region_tables, ignore_index=True)
    canonical = pd.read_csv(REGION_FILE)
    region_animal = region_animal.merge(
        canonical, on=["root_area_id", "node"], how="inner", validate="many_to_one"
    )
    loo_table = build_loo_table(region_animal)

    animal_table.to_csv(OUTPUT_DIR / "te_surrogate_animal_summary.csv", index=False)
    null_table.to_csv(OUTPUT_DIR / "te_surrogate_network_null.csv", index=False)
    region_animal.to_csv(OUTPUT_DIR / "te_region_animal_values.csv", index=False)
    loo_table.to_csv(OUTPUT_DIR / "te_fcv_leave_one_animal_out.csv", index=False)

    example = next(
        result for result in results
        if result["recording_id"] == CONFIG["example_recording_id"]
    )
    pd.DataFrame(
        {
            "surrogate": np.arange(1, len(example["example_null"]) + 1),
            "abs_NetTE": example["example_null"],
        }
    ).to_csv(OUTPUT_DIR / "te_example_pair_null.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "te_example_observed_matrix.npz",
        recording_id=example["recording_id"],
        net_te_matrix=example["observed_matrix"],
        unit_nodes=example["unit_nodes"],
        unit_clusters=example["unit_clusters"],
        example_source=example["example_source"],
        example_target=example["example_target"],
        example_observed=example["example_observed"],
    )

    statistics = animal_table.copy()
    statistics["analysis"] = "whole-network circular-shift validation"
    statistics["n_surrogates"] = int(CONFIG["n_surrogates"])
    statistics.to_csv(STATS_DIR / "figure_supply_13_stats.csv", index=False)
    print(f"Saved validation tables to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
