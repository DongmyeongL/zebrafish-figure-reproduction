"""Draw anatomical-group summaries from the clean invertebrate tables.

Inputs are produced inside final_figure_pack_1 by the invertebrate processing
pipeline. No legacy plotting table or pre-rendered panel is used.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import figure_style as fs


PACK = Path(__file__).resolve().parents[1]
DATA_DIR = PACK / "derived_data" / "invertebrates"
CE_VALUES = DATA_DIR / "celegans_node_metrics.csv"
FLY_VALUES = DATA_DIR / "drosophila_region_metrics.csv"
CE_FCV_RECORDINGS = DATA_DIR / "celegans_fcv_recording_node.csv"
FLY_FCV_RECORDINGS = DATA_DIR / "drosophila_fcv_recording_region.csv"
OUT_FIG = PACK / "figures" / "figure_invertebrate_anatomical_group_summary.png"
OUT_SUMMARY = DATA_DIR / "invertebrate_anatomical_group_summary.csv"
OUT_FIG_WITH_ZERO = (
    PACK / "figures" / "figure_invertebrate_anatomical_group_summary_with_oo_zero.png"
)
OUT_SUMMARY_WITH_ZERO = (
    DATA_DIR / "invertebrate_anatomical_group_summary_with_oo_zero.csv"
)

METRICS = [
    ("EdgeStdFCV", "FCV"),
    ("PostDCA", r"$\mathrm{DCA}_{\mathrm{post}}$"),
    ("PreDCA", r"$\mathrm{DCA}_{\mathrm{pre}}$"),
    ("OO_fraction", "OO fraction"),
]

GROUP_ORDER = {
    "C. elegans": [
        "chemosensory",
        "mechanosensory / other sensory",
        "thermo/gas sensory",
        "interneuron / integrative",
        "associative interneuron",
        "state-modulatory interneuron",
        "head motor / premotor",
        "locomotor command interneuron",
    ],
    "Drosophila": [
        "olfactory-associative",
        "visual-association / integrative",
        "premotor / descending",
        "primary visual",
        "lateral/inferior protocerebrum",
        "other / central complex",
    ],
}

PALETTE = [
    "#6FA8C9", "#4E9DA6", "#DDA15E", "#6FAF6B",
    "#8B7CB6", "#E76F51", "#D48AB0", "#C9884B",
]
GROUP_COLORS = {
    species: dict(zip(groups, PALETTE))
    for species, groups in GROUP_ORDER.items()
}
GROUP_COLORS["Drosophila"].update({
    "olfactory-associative": "#6FA8C9",
    "visual-association / integrative": "#6FAF6B",
    "premotor / descending": "#D48AB0",
    "primary visual": "#DDA15E",
    "lateral/inferior protocerebrum": "#8B7CB6",
    "other / central complex": fs.MAIN_COLORS["neutral"],
})

SHORT_LABELS = {
    "chemosensory": "chemo",
    "mechanosensory / other sensory": "mechano",
    "thermo/gas sensory": "thermo/gas",
    "interneuron / integrative": "inter/integ",
    "associative interneuron": "assoc. inter",
    "state-modulatory interneuron": "state inter",
    "head motor / premotor": "head motor",
    "locomotor command interneuron": "loco. cmd",
    "olfactory-associative": "olf.-assoc.",
    "visual-association / integrative": "vis.-assoc./integ.",
    "premotor / descending": "premotor",
    "primary visual": "primary visual",
    "lateral/inferior protocerebrum": "lat/inf proto",
    "other / central complex": "other/CX",
}

DROS_GROUP_BY_BASE = {
    "AL": "olfactory-associative", "LH": "olfactory-associative",
    "MB": "olfactory-associative",
    "AVLP": "visual-association / integrative",
    "PVLP": "visual-association / integrative",
    "PLP": "visual-association / integrative",
    "WED": "visual-association / integrative",
    "SLP": "visual-association / integrative",
    "SMP": "visual-association / integrative",
    "SIP": "visual-association / integrative",
    "LAL": "premotor / descending", "VES": "premotor / descending",
    "SPS": "premotor / descending",
    "LO": "primary visual", "LOP": "primary visual",
    "ME": "primary visual", "AME": "primary visual",
    "CRE": "lateral/inferior protocerebrum",
    "IB": "lateral/inferior protocerebrum",
    "ICL": "lateral/inferior protocerebrum",
    "ATL": "lateral/inferior protocerebrum",
    "EPA": "lateral/inferior protocerebrum",
}


def _base_region(node: object) -> str:
    text = str(node)
    return text[:-2] if text.endswith(("_L", "_R")) else text


def load_values() -> dict[str, pd.DataFrame]:
    ce = pd.read_csv(CE_VALUES).replace([np.inf, -np.inf], np.nan)
    # Retain all matched neurons, including those with OO_fraction equal to 0.
    ce["group"] = ce["fine_class"]

    ce_fcv = pd.read_csv(CE_FCV_RECORDINGS).replace([np.inf, -np.inf], np.nan)
    ce_fcv = ce_fcv.loc[ce_fcv["node"].isin(ce["node"])].copy()
    ce_fcv = ce_fcv.merge(
        ce[["node", "group"]].drop_duplicates(), on="node", how="inner"
    )
    ce["EdgeStdFCV"] = np.nan
    ce = pd.concat([ce, ce_fcv], ignore_index=True, sort=False)

    fly = pd.read_csv(FLY_VALUES).replace([np.inf, -np.inf], np.nan)
    fly["group"] = (
        fly["node"].map(_base_region).map(DROS_GROUP_BY_BASE)
        .fillna("other / central complex")
    )
    fly_fcv = pd.read_csv(FLY_FCV_RECORDINGS).replace([np.inf, -np.inf], np.nan)
    fly_fcv = fly_fcv.loc[fly_fcv["node"].isin(fly["node"])].copy()
    fly_fcv["group"] = (
        fly_fcv["node"].map(_base_region).map(DROS_GROUP_BY_BASE)
        .fillna("other / central complex")
    )
    fly["EdgeStdFCV"] = np.nan
    fly = pd.concat([fly, fly_fcv], ignore_index=True, sort=False)
    return {"C. elegans": ce, "Drosophila": fly}


def summarize(values: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for species, df in values.items():
        for group in GROUP_ORDER[species]:
            for metric, _ in METRICS:
                x = df.loc[df["group"].eq(group), metric].dropna().to_numpy(float)
                rows.append({
                    "species": species,
                    "group": group,
                    "metric": metric,
                    "n": len(x),
                    "mean": np.mean(x) if len(x) else np.nan,
                    "sem": (
                        np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
                    ),
                    "median": np.median(x) if len(x) else np.nan,
                })
    return pd.DataFrame(rows)


def draw_panel(ax, df, species: str, metric: str, ylabel: str, seed: int) -> None:
    order = GROUP_ORDER[species]
    colors = GROUP_COLORS[species]
    rng = np.random.default_rng(seed)
    arrays = [
        df.loc[df["group"].eq(group), metric].dropna().to_numpy(float)
        for group in order
    ]
    point_size = 9 if metric == "EdgeStdFCV" else 18
    point_alpha = 0.18 if metric == "EdgeStdFCV" else 0.78

    for x_pos, (group, vals) in enumerate(zip(order, arrays, strict=True)):
        if not len(vals):
            continue
        jitter = rng.normal(0, 0.065, len(vals))
        ax.scatter(
            x_pos + jitter,
            vals,
            s=point_size,
            color=colors[group],
            edgecolor="black",
            linewidth=0.35,
            alpha=point_alpha,
            rasterized=True,
            zorder=3,
        )

    boxes = ax.boxplot(
        arrays,
        positions=np.arange(len(order)),
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 0.9},
        boxprops={"linewidth": 0.8, "edgecolor": "#222222"},
        whiskerprops={"linewidth": 0.7, "color": "#222222"},
        capprops={"linewidth": 0.7, "color": "#222222"},
    )
    for patch, group in zip(boxes["boxes"], order, strict=True):
        patch.set_facecolor(colors[group])
        patch.set_alpha(0.30)

    ax.axhline(0, color="#bdbdbd", linewidth=0.7, linestyle=":", zorder=0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        [SHORT_LABELS[group] for group in order], rotation=38, ha="right"
    )
    for tick, group in zip(ax.get_xticklabels(), order, strict=True):
        tick.set_color(colors[group])
        tick.set_fontweight("bold")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", direction="out", pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    fs.apply_supplement_figure_style()

    values = load_values()
    summary = summarize(values)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_SUMMARY_WITH_ZERO, index=False)

    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.16))
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.145, top=0.925,
        wspace=0.52, hspace=0.62,
    )
    panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for row, species in enumerate(("C. elegans", "Drosophila")):
        for col, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            draw_panel(ax, values[species], species, metric, ylabel, 9100 + row * 10 + col)
            ax.text(
                -0.27, 1.04, panel_labels[row * 4 + col],
                transform=ax.transAxes, fontweight="bold", fontsize=fs.SUPP_PANEL_LABEL_FS,
                ha="left", va="bottom",
            )
        row_positions = [axes[row, col].get_position() for col in range(4)]
        row_center = (row_positions[0].x0 + row_positions[-1].x1) / 2
        fig.text(
            row_center,
            max(position.y1 for position in row_positions) + 0.055,
            species,
            ha="center",
            va="bottom",
            fontsize=fs.SUPP_TITLE_FS,
            fontweight="bold",
            fontstyle="italic",
        )

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", pad_inches=0.03, transparent=False)
    fig.savefig(
        OUT_FIG_WITH_ZERO,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        transparent=False,
    )
    plt.close(fig)
    print(f"Saved {OUT_FIG}")
    print(f"Saved {OUT_FIG_WITH_ZERO}")
    print(f"Saved {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
