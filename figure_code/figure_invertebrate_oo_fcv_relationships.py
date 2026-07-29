import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, spearmanr

import figure_style as fs


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

fs.apply_main_figure_style()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Nimbus Sans",
    "mathtext.it": "Nimbus Sans:italic",
    "mathtext.bf": "Nimbus Sans:bold",
    "mathtext.cal": "Nimbus Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Match the typography hierarchy used by the stimulus FCV/FCS forest figure.
ANNOT_FS = fs.TICK_FS_2COL - 1
TICK_FS = fs.TICK_FS_2COL
AXIS_FS = fs.AXIS_LABEL_FS_2COL
PANEL_FS = fs.PANEL_LABEL_FS_2COL

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DERIVED_DIR = os.path.join(PROJECT_ROOT, "derived_data", "invertebrates")
STATS_DIR = os.path.join(PROJECT_ROOT, "statistics", "invertebrates")
OUT_PNG = os.path.join(PROJECT_ROOT, "figures", "figure_invertebrate_oo_fcv_relationships.png")
STATS_CSV = os.path.join(STATS_DIR, "figure_invertebrate_oo_fcv_relationships_stats.csv")
CE_VALUES = os.path.join(DERIVED_DIR, "celegans_node_metrics.csv")
FLY_VALUES = os.path.join(DERIVED_DIR, "drosophila_region_metrics.csv")

SPECIES_ORDER = ["C. elegans", "Drosophila"]
SPECIES_LEVEL = {
    "C. elegans": "fine-class mean",
    "Drosophila": "side-aware region",
}

# Each point is one fine_class group (C. elegans is already a fine-class mean;
# Drosophila regions are one-to-many with fine_class), so color identity is
# assigned per fine_class rather than the coarser anatomy_group. Palette is the
# validated 8-slot categorical set from the dataviz skill (fixed hue order).
# Muted / journal-style palette matched to the main figures' saturation
# (figure_style division colors: #6FA8C9 / #6FAF6B / #DDA15E / #E76F51).
# First entry is the highlighted group (olf.-assoc. / chemo); it keeps the
# same low saturation and is emphasized only via black edge + larger size,
# exactly like the Tel highlight in figure_sc_fc_final_overview_v2.
_CATEGORICAL_PALETTE = [
    "#6FA8C9",  # soft blue   (highlight)
    "#4E9DA6",  # muted teal
    "#DDA15E",  # tan / gold
    "#6FAF6B",  # sage green
    "#8B7CB6",  # muted violet
    "#E76F51",  # coral
    "#D48AB0",  # muted pink
    "#C9884B",  # ochre
]

FINE_CLASS_ORDER = {
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
FINE_CLASS_COLORS = {
    species: dict(zip(order, _CATEGORICAL_PALETTE))
    for species, order in FINE_CLASS_ORDER.items()
}
FINE_CLASS_COLORS["Drosophila"].update(
    {
        # Muted / journal tones matched to _CATEGORICAL_PALETTE (same hues,
        # lowered saturation) so Drosophila and C. elegans share one look.
        "olfactory-associative": "#6FA8C9",          # soft blue (highlight)
        "visual-association / integrative": "#6FAF6B",  # sage green
        "premotor / descending": "#D48AB0",          # muted pink
        "primary visual": "#DDA15E",                 # tan / gold
        "lateral/inferior protocerebrum": "#8B7CB6",  # muted violet
        "other / central complex": fs.MAIN_COLORS["neutral"],
    }
)

# Distinct marker per fine_class so groups are separable by shape as well as
# colour (helps colour-blind readers and print reproduction). Assigned by the
# fixed hue order above.
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]
FINE_CLASS_MARKERS = {
    species: dict(zip(order, _MARKER_CYCLE))
    for species, order in FINE_CLASS_ORDER.items()
}

# Key group per species to emphasise with larger markers.
HIGHLIGHT_GROUP = {
    "Drosophila": "olfactory-associative",
    "C. elegans": "chemosensory",
}

DROS_COARSE_BY_BASE = {
    "AL": "olfactory-associative",
    "LH": "olfactory-associative",
    "MB": "olfactory-associative",
    "AVLP": "visual-association / integrative",
    "PVLP": "visual-association / integrative",
    "PLP": "visual-association / integrative",
    "WED": "visual-association / integrative",
    "SLP": "visual-association / integrative",
    "SMP": "visual-association / integrative",
    "SIP": "visual-association / integrative",
    "LAL": "premotor / descending",
    "VES": "premotor / descending",
    "SPS": "premotor / descending",
    "LO": "primary visual",
    "LOP": "primary visual",
    "ME": "primary visual",
    "AME": "primary visual",
    "CRE": "lateral/inferior protocerebrum",
    "IB": "lateral/inferior protocerebrum",
    "ICL": "lateral/inferior protocerebrum",
    "ATL": "lateral/inferior protocerebrum",
    "EPA": "lateral/inferior protocerebrum",
}

# Compact legend text -- keeps the in-axes legend box narrow enough to stay
# inside the empty corner instead of spanning the whole panel width.
SHORT_FINE_CLASS_LABELS = {
    "chemosensory": "chemo",
    "mechanosensory / other sensory": "mechano",
    "thermo/gas sensory": "thermo/gas",
    "interneuron / integrative": "inter/integ",
    "associative interneuron": "assoc. inter",
    "state-modulatory interneuron": "state inter",
    "head motor / premotor": "head motor",
    "locomotor command interneuron": "loco. cmd",
    "olfactory system": "olfactory",
    "visual / optic": "visual",
    "mushroom body": "MB",
    "superior protocerebrum": "sup. proto",
    "lateral/inferior protocerebrum": "lat/inf proto",
    "visual-association protocerebrum": "vis-assoc proto",
    "premotor / descending interface": "premotor",
    "olfactory-associative": "olf.-assoc.",
    "visual-association / integrative": "vis.-assoc./integ.",
    "premotor / descending": "premotor",
    "primary visual": "primary visual",
    "other / central complex": "other/CX",
}


def _drosophila_base_node(node):
    text = str(node)
    if text.endswith("_L") or text.endswith("_R"):
        return text[:-2]
    return text


def _add_drosophila_coarse_group(df):
    out = df.copy()
    bases = out["node"].map(_drosophila_base_node)
    out["fine_class"] = bases.map(DROS_COARSE_BY_BASE).fillna("other / central complex")
    return out


def _load_data():
    ce = pd.read_csv(CE_VALUES).replace([np.inf, -np.inf], np.nan)
    ce = ce.loc[ce["OO_fraction"].fillna(0).ne(0)].copy()
    ce = ce.groupby("fine_class", as_index=False).agg(
        node=("fine_class", "first"), OO_fraction=("OO_fraction", "mean"),
        EdgeStdFCV=("EdgeStdFCV", "mean"), PostDCA=("PostDCA", "mean"),
        PreDCA=("PreDCA", "mean"), n_neurons=("node", "count"))
    ce["species"] = "C. elegans"
    ce["oo_level"] = SPECIES_LEVEL["C. elegans"]

    fly = pd.read_csv(FLY_VALUES).replace([np.inf, -np.inf], np.nan)
    fly = _add_drosophila_coarse_group(fly)
    fly["species"] = "Drosophila"
    fly["oo_level"] = SPECIES_LEVEL["Drosophila"]
    return pd.concat([ce, fly], ignore_index=True, sort=False)


def _format_species(species):
    if species == "C. elegans":
        return r"$\it{C.\ elegans}$"
    return r"$\it{Drosophila}$"


def _draw_scatter(ax, df, species, x_col, y_col, x_label, y_label,text_offset=[0.6, 0.2]):
    sub = df.loc[df["species"].eq(species), [x_col, y_col, "fine_class"]].dropna(subset=[x_col, y_col])
    colors = FINE_CLASS_COLORS[species]
    markers = FINE_CLASS_MARKERS[species]
    highlight = HIGHLIGHT_GROUP.get(species)
    for group in FINE_CLASS_ORDER[species]:
        gsub = sub.loc[sub["fine_class"].eq(group)]
        if gsub.empty:
            continue
        marker = markers[group]
        is_hi = group == highlight
        # Match overview_v2 scatter style: size / edgecolor / linewidth.
        ax.scatter(
            gsub[x_col],
            gsub[y_col],
            s=62,
            marker=marker,
            color=colors[group],
            alpha=0.90 if is_hi else 0.78,
            edgecolor="black",
            linewidth=0.4,
            zorder=4 if is_hi else 2,
            label=group,
        )
    if len(sub) >= 3 and sub[x_col].nunique() > 1:
        slope, intercept = np.polyfit(sub[x_col], sub[y_col], 1)
        xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#4d4d4d", lw=1.8, zorder=3)
        pear = pearsonr(sub[x_col], sub[y_col])
        spear = spearmanr(sub[x_col], sub[y_col])
        stat_text = (
            #f"n={len(sub)}\n"
            f"r = {pear.statistic:.3f}\np = {pear.pvalue:.3g}\n"
            #+ r"$\rho$"
            #+ f"={spear.statistic:.2f}"
        )
    else:
        pear = spear = None
        stat_text = f"n={len(sub)}"
    ax.text(
        text_offset[0],
        text_offset[1],
        stat_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOT_FS,
    )
    # Species is shown once per row (see _add_row_label), not on every panel.
    ax.set_xlabel(x_label, fontsize=AXIS_FS, labelpad=2)
    ax.set_ylabel(y_label, fontsize=AXIS_FS, labelpad=2)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    # Main-paper style: no grid; light zero reference lines + sparse ticks.
    ax.axhline(0, color="#dddddd", lw=1, zorder=0)
    ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12, y=0.14)
    if pear is None:
        return None
    return {
        "species": species,
        "level": SPECIES_LEVEL[species],
        "x": x_col,
        "y": y_col,
        "n": int(len(sub)),
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }


def _add_row_label(fig, row_axes, text, y_offset=0.035):
    x0 = min(ax.get_position().x0 for ax in row_axes)
    x1 = max(ax.get_position().x1 for ax in row_axes)
    y1 = max(ax.get_position().y1 for ax in row_axes)
    fig.text(
        (x0 + x1) / 2,
        y1 + y_offset+0.15,
        text,
        rotation=0,
        ha="center",
        va="center",
        fontsize=AXIS_FS,
        fontweight="bold",
        style="italic",
    )


def _add_row_legend(fig, row_axes, species, ncol, y_offset=0.072):
    """Place the fine_class legend above the row, entirely outside the axes.

    This guarantees zero overlap with data (unlike an in-axes corner legend,
    which for sparse panels like C. elegans's n=8 can cover real points) at
    the cost of a bit of extra vertical space, recovered automatically by
    bbox_inches="tight" on save.
    """
    from matplotlib.lines import Line2D

    colors = FINE_CLASS_COLORS[species]
    markers = FINE_CLASS_MARKERS[species]
    highlight = HIGHLIGHT_GROUP.get(species)
    order = FINE_CLASS_ORDER[species]
    handles = [
        Line2D(
            [],
            [],
            marker=markers[group],
            color=colors[group],
            linestyle="None",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=9.0,
        )
        for group in order
    ]
    labels = [SHORT_FINE_CLASS_LABELS.get(group, group) for group in order]
    x0 = min(ax.get_position().x0 for ax in row_axes)
    x1 = max(ax.get_position().x1 for ax in row_axes)
    y1 = max(ax.get_position().y1 for ax in row_axes)
    
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=((x0 + x1) / 2, y1 + y_offset),
        ncol=ncol,
        fontsize=TICK_FS,
        frameon=True,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.3,
    )


def main():
    data = _load_data()
    fig = plt.figure(figsize=(8,7))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.085,
        right=0.985,
        top=0.820,
        bottom=0.075,
        wspace=0.34,
        hspace=0.75,
    )
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[0, 2]),
        "D": fig.add_subplot(gs[1, 0]),
        "E": fig.add_subplot(gs[1, 1]),
        "F": fig.add_subplot(gs[1, 2]),
    }

    stats_rows = []
    stats_rows.append(_draw_scatter(axes["A"], data, "C. elegans", "OO_fraction", "EdgeStdFCV", "OO fraction", "FCV"))
    stats_rows.append(_draw_scatter(axes["B"], data, "C. elegans", "PostDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{post}}$", "FCV"))
    stats_rows.append(_draw_scatter(axes["C"], data, "C. elegans", "PreDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{pre}}$", "FCV", text_offset=[0.60, 0.95]))
    stats_rows.append(_draw_scatter(axes["D"], data, "Drosophila", "OO_fraction", "EdgeStdFCV", "OO fraction", "FCV"))
    stats_rows.append(_draw_scatter(axes["E"], data, "Drosophila", "PostDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{post}}$", "FCV", text_offset=[0.2, 0.2]))
    stats_rows.append(_draw_scatter(axes["F"], data, "Drosophila", "PreDCA", "EdgeStdFCV", r"$\mathrm{DCA}_{\mathrm{pre}}$", "FCV", text_offset=[0.5, 0.2]))

    axes['A'].set_xlim((0.2,0.8));
    axes['D'].set_xlim((0.0,0.6));

    fig.canvas.draw()
    for ax in (axes["D"], axes["E"], axes["F"]):
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            pos.y0 - 0.065,  # 아래로 이동
            pos.width,
            pos.height,
        ])

    top_axes = [axes["A"], axes["B"], axes["C"]]
    bottom_axes = [axes["D"], axes["E"], axes["F"]]
    _add_row_label(fig, top_axes, "C. elegans")
    _add_row_label(fig, bottom_axes, "Drosophila")
    # Two-row legends sit above their species labels and outside all data axes.
    _add_row_legend(fig, top_axes, "C. elegans", ncol=4)
    _add_row_legend(fig, bottom_axes, "Drosophila", ncol=3)
    
    # Journal house style: bold, upright panel labels placed just outside the
    # top-left of each axes (matching the main-figure placement).
    for label, ax in axes.items():
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.062,
            pos.y1 + 0.014,
            label,
            fontsize=PANEL_FS,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03, transparent=False)
    pd.DataFrame([row for row in stats_rows if row is not None]).to_csv(STATS_CSV, index=False)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {STATS_CSV}")


if __name__ == "__main__":
    main()
