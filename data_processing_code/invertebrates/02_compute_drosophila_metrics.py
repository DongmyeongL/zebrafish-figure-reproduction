"""Compute fly FCV and directed SC measures from Branson999 and FlyWire783."""

from __future__ import annotations

import pickle
import re
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve()
PACK = HERE.parents[2]
RAW = Path(os.environ.get("ZF_RAW_DATA_ROOT", PACK / "raw_data")).resolve() / "invertebrates" / "drosophila"
OUT = Path(os.environ.get("ZF_DERIVED_DATA_ROOT", PACK / "derived_data")).resolve() / "invertebrates"
sys.path.insert(0, str(HERE.parent))

from common_metrics import edge_std_fcv, preprocess_traces, rank1_dca_sparse, zscore_finite

WINDOW = 15
STEP = 5
HIGHPASS_HZ = 0.03
SYN_THRESHOLD = 5
CONFIDENCE_THRESHOLD = 0.30


def split_side(name: str) -> tuple[str, str]:
    text = str(name)
    if text.endswith("_L") or text.endswith("_R"):
        return text[:-2], text[-1]
    return text, "midline"


def canonical_label(name: str) -> str:
    base, side = split_side(name)
    base = {"SOG": "GNG", "LA": "LO", "MED": "ME", "LOB": "LO"}.get(base, base)
    return f"{base}_{side}" if side in {"L", "R"} else base


def collapse_region(region: str) -> str:
    base, side = split_side(region)
    if base.startswith("MB_"):
        base = "MB"
    if base == "AOTU":
        base = "OTU"
    return f"{base}_{side}" if side in {"L", "R"} else base


def load_order() -> pd.DataFrame:
    order = pd.read_csv(RAW / "ito_region_order.csv").sort_values("plot_order").copy()
    order["region"] = order["ito_label"].astype(str)
    order["base_block"] = order["ito_block"].astype(str)
    return order


def build_cell_map(order: pd.DataFrame) -> pd.DataFrame:
    valid = set(order["region"].astype(str))
    roots = pd.DataFrame({"root_id": np.load(RAW / "proofread_root_ids_783.npy").astype(np.int64)})
    pre = pd.read_feather(RAW / "per_neuron_neuropil_count_pre_783.feather").rename(
        columns={"pre_pt_root_id": "root_id", "count": "pre_count"})
    post = pd.read_feather(RAW / "per_neuron_neuropil_count_post_783.feather").rename(
        columns={"post_pt_root_id": "root_id", "count": "post_count"})
    pre["region"] = pre["neuropil"].map(canonical_label)
    post["region"] = post["neuropil"].map(canonical_label)
    pre = pre[pre["region"].isin(valid)].groupby(["root_id", "region"], as_index=False)["pre_count"].sum()
    post = post[post["region"].isin(valid)].groupby(["root_id", "region"], as_index=False)["post_count"].sum()
    counts = pre.merge(post, on=["root_id", "region"], how="outer").fillna(0)
    counts["localization_count"] = counts["pre_count"] + counts["post_count"]
    totals = counts.groupby("root_id", as_index=False)["localization_count"].sum().rename(
        columns={"localization_count": "total_localization_count"})
    dominant = counts.loc[counts.groupby("root_id")["localization_count"].idxmax(),
                          ["root_id", "region", "localization_count"]].rename(
        columns={"localization_count": "dominant_localization_count"})
    mapped = roots.merge(dominant, on="root_id", how="left").merge(totals, on="root_id", how="left")
    mapped["mapping_confidence"] = mapped["dominant_localization_count"] / mapped["total_localization_count"]
    mapped["is_confident"] = mapped["mapping_confidence"] >= CONFIDENCE_THRESHOLD
    return mapped.merge(order[["region", "big_group", "plot_order"]], on="region", how="left")


def structural_metrics() -> pd.DataFrame:
    order = load_order()
    mapped = build_cell_map(order)
    use = mapped[mapped["is_confident"] & mapped["region"].notna()].copy()
    conn = pd.read_feather(RAW / "proofread_connections_783.feather",
                           columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    conn = conn[conn["syn_count"] >= SYN_THRESHOLD].groupby(
        ["pre_pt_root_id", "post_pt_root_id"], as_index=False)["syn_count"].sum()
    pre_map = use[["root_id", "region"]].rename(columns={"root_id": "pre_pt_root_id", "region": "pre_region"})
    post_map = use[["root_id", "region"]].rename(columns={"root_id": "post_pt_root_id", "region": "post_region"})
    conn = conn.merge(pre_map, on="pre_pt_root_id", how="inner").merge(post_map, on="post_pt_root_id", how="inner")

    use = use.copy()
    use["DCA"] = np.nan
    for region, cells in use.groupby("region"):
        ids = cells["root_id"].to_numpy(np.int64)
        if len(ids) < 2:
            continue
        local = {rid: i for i, rid in enumerate(ids)}
        sub = conn[(conn["pre_region"] == region) & (conn["post_region"] == region)]
        rows = sub["pre_pt_root_id"].map(local).dropna().astype(int)
        cols = sub.loc[rows.index, "post_pt_root_id"].map(local).astype(int)
        weights = sub.loc[rows.index, "syn_count"].to_numpy(float)
        if not len(weights):
            continue
        matrix = sparse.coo_matrix((weights, (rows.to_numpy(), cols.to_numpy())), shape=(len(ids), len(ids))).tocsr()
        _, _, dca = rank1_dca_sparse(matrix)
        use.loc[cells.index, "DCA"] = dca

    src = use[["root_id", "region", "DCA"]].rename(
        columns={"root_id": "pre_pt_root_id", "region": "pre_region", "DCA": "source_DCA"})
    tgt = use[["root_id", "region", "DCA"]].rename(
        columns={"root_id": "post_pt_root_id", "region": "post_region", "DCA": "target_DCA"})
    inter = conn[["pre_pt_root_id", "post_pt_root_id", "syn_count"]].merge(src, on="pre_pt_root_id").merge(tgt, on="post_pt_root_id")
    inter = inter[(inter["pre_region"] != inter["post_region"]) & inter["source_DCA"].notna() & inter["target_DCA"].notna()].copy()
    inter["weighted_target"] = inter["syn_count"] * inter["target_DCA"]
    inter["weighted_source"] = inter["syn_count"] * inter["source_DCA"]
    inter["oo"] = (inter["source_DCA"] > 0) & (inter["target_DCA"] > 0)
    outgoing = inter.groupby("pre_pt_root_id", as_index=False).agg(
        post_num=("weighted_target", "sum"), post_den=("syn_count", "sum"),
        out_degree=("post_pt_root_id", "nunique"), OO_count=("oo", "sum"))
    outgoing["PostDCA"] = outgoing["post_num"] / outgoing["post_den"]
    outgoing["OO_fraction"] = outgoing["OO_count"] / outgoing["out_degree"]
    incoming = inter.groupby("post_pt_root_id", as_index=False).agg(
        pre_num=("weighted_source", "sum"), pre_den=("syn_count", "sum"))
    incoming["PreDCA"] = incoming["pre_num"] / incoming["pre_den"]
    cells = use.merge(outgoing, left_on="root_id", right_on="pre_pt_root_id", how="left").merge(
        incoming, left_on="root_id", right_on="post_pt_root_id", how="left")
    cells["node"] = cells["region"].map(collapse_region)
    summary = cells.groupby("node", as_index=False).agg(
        PostDCA=("PostDCA", "mean"), PreDCA=("PreDCA", "mean"), OO_fraction=("OO_fraction", "mean"),
        DCA=("DCA", "mean"), n_cells=("root_id", "size"), big_group=("big_group", lambda x: x.mode().iat[0]))
    cells.to_csv(OUT / "drosophila_cell_structural_metrics.csv", index=False)
    return summary


def functional_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((RAW / "recordings").glob("*_raw_traces.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        meta = pd.DataFrame({"roi": np.arange(len(raw["atlas_region"])), "node": raw["atlas_region"]})
        node_names, node_traces, n_rois = [], [], []
        traces = np.asarray(raw["traces"], dtype=float)
        for node, sub in meta.groupby("node", sort=False):
            idx = sub["roi"].to_numpy(int)
            node_names.append(str(node)); node_traces.append(np.nanmean(traces[idx], axis=0)); n_rois.append(len(idx))
        prepared = preprocess_traces(np.vstack(node_traces), float(raw["sampling_rate_hz"]), HIGHPASS_HZ)
        fcv = edge_std_fcv(prepared, WINDOW, STEP)
        fcv_z = zscore_finite(fcv)
        rows.extend({"recording_id": raw["recording_id"], "node": node, "FCV_raw": value,
                     "EdgeStdFCV": zvalue, "n_rois": count}
                    for node, value, zvalue, count in zip(node_names, fcv, fcv_z, n_rois))
    recording = pd.DataFrame(rows)
    summary = recording.groupby("node", as_index=False).agg(
        EdgeStdFCV=("EdgeStdFCV", "mean"), FCV_raw=("FCV_raw", "mean"),
        n_recordings=("recording_id", "nunique"), n_rois_mean=("n_rois", "mean"))
    return recording, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recording, functional = functional_metrics()
    structural = structural_metrics()
    merged = functional.merge(structural, on="node", how="inner")
    recording.to_csv(OUT / "drosophila_fcv_recording_region.csv", index=False)
    structural.to_csv(OUT / "drosophila_structural_region_metrics.csv", index=False)
    merged.to_csv(OUT / "drosophila_region_metrics.csv", index=False)
    print(f"Drosophila: {len(recording)} recording-region rows, {len(merged)} matched regions")


if __name__ == "__main__":
    main()
