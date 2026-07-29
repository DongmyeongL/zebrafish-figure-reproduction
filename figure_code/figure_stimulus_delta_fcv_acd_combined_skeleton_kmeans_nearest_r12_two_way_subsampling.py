"""Render t110-style stimulus panels with r12 two-way-subampling FU-SC data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import figure_stimulus_delta_fcv_acd_combined as source


PACK_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PACK_ROOT / "derived_data" / "common"
STRUCTURAL_INPUT = (
    INPUT_DIR
    / "figure_fcv_fcs_fu_sc_corr_forest_fcs_calibrated_skeleton_kmeans_nearest_r12_two_way_subsampling_input.csv"
)
SUBSAMPLING_CSV = (
    PACK_ROOT
    / "derived_data"
    / "figure12"
    / "validation"
    / "two_way_subsampling_r12"
    / "figure_stimulus_fcv_condition_sd_r12_two_way_subsampling_iterations.csv.gz"
)
VARIANT = "skeleton_kmeans_nearest_r12_two_way_subsampling"
NAME = f"figure_stimulus_delta_fcv_acd_combined_{VARIANT}"


def _install_two_way_subsampling_distribution(samples_path: Path) -> None:
    samples = pd.read_csv(samples_path)
    required = {"func_column", "sc_column", "pearson_r"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"{samples_path} is missing columns: {sorted(missing)}")

    order = [
        (func_col, sc_col)
        for func_col, _, _ in [
            ("EdgeStdFCV", "FCV", "#5B8DB8"),
            ("ConditionFCVSD", "FCV SD across OMR conditions", "#4C9A8A"),
        ]
        for sc_col, _ in source.FU_SC_MEASURES
    ]
    distributions = {}
    for func_col, sc_col in order:
        values = samples.loc[
            samples["func_column"].eq(func_col) & samples["sc_column"].eq(sc_col),
            "pearson_r",
        ].dropna().to_numpy(float)
        if len(values) < 10:
            raise ValueError(f"Too few subsampling values for {func_col} vs {sc_col}: {len(values)}")
        distributions[(func_col, sc_col)] = values

    call_index = {"value": 0}
    original_bootstrap = source.forest.bootstrap_pearson

    def two_way_subsampling_pearson(x, y):
        key = order[call_index["value"]]
        call_index["value"] += 1
        result = original_bootstrap(x, y)
        dist = distributions[key]
        result["boot"] = dist
        result["lo"], result["hi"] = [float(v) for v in np.percentile(dist, [2.5, 97.5])]
        return result

    source.forest.bootstrap_pearson = two_way_subsampling_pearson
    source.forest.N_BOOT = int(min(len(values) for values in distributions.values()))


def _build_base_frame_from_r12_two_way_input() -> pd.DataFrame:
    """Use the exact spontaneous r12 two-way input as the structural base."""
    sc_columns = [column for column, _ in source.FU_SC_MEASURES]
    base = pd.read_csv(STRUCTURAL_INPUT)[
        [
            "legacy_order",
            "root_area_id",
            "node",
            "anatomy_group",
            "EdgeStdFCV_raw",
            "FCS_raw",
            *sc_columns,
        ]
    ].rename(columns={"EdgeStdFCV_raw": "spont_FCV", "FCS_raw": "spont_FCS"})

    stimulus = pd.read_csv(source.STIMULUS)[["root_area_id", "node", "FCV", "FCS"]].rename(
        columns={"FCV": "stim_FCV", "FCS": "stim_FCS"}
    )
    condition_fcv = (
        pd.read_csv(source.STIMULUS_DETAIL)
        .groupby(["root_area_id", "node", "stimulus_index"], as_index=False)["FCV_z"]
        .mean()
        .groupby(["root_area_id", "node"], as_index=False)["FCV_z"]
        .agg(lambda values: float(np.std(values.to_numpy(float), ddof=0)))
        .rename(columns={"FCV_z": "stim_FCV_condition_sd"})
    )
    frame = (
        base.merge(stimulus, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .merge(condition_fcv, on=["root_area_id", "node"], how="left", validate="one_to_one")
        .replace([np.inf, -np.inf], np.nan)
        .sort_values("legacy_order")
        .reset_index(drop=True)
    )
    required = [
        "spont_FCV",
        "spont_FCS",
        "stim_FCV",
        "stim_FCS",
        "stim_FCV_condition_sd",
        *sc_columns,
    ]
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if len(frame) != len(base):
        raise RuntimeError(f"Stimulus matching dropped regions: base={len(base)}, complete={len(frame)}")
    return frame.reset_index(drop=True)


source._build_base_frame = _build_base_frame_from_r12_two_way_input
source.ARGS.sc_source = "fcs_calibrated_skeleton_kmeans_nearest_r12"
source.PANEL_AB_DISTRIBUTION = "two_way_subsampling"
source.PANEL_AB_P_MODE = "raw"
source.STIMULUS_INPUT = INPUT_DIR / f"{NAME}_input.csv"
source.OUT_PNG = PACK_ROOT / "figures" / f"{NAME}.png"
source.OUT_STATS = PACK_ROOT / "statistics" / f"{NAME}_stats.csv"


if __name__ == "__main__":
    _install_two_way_subsampling_distribution(SUBSAMPLING_CSV)
    source.main()
