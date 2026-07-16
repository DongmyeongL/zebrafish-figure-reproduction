
"""Draw supplementary Figure 1 from the frozen distance/FC input bundle."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import figure_style as fs
from figure_style import add_panel_label_fig

fs.apply_supplement_figure_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "raw_data" / "figure_supply_1"
OUT_PNG = PROJECT_ROOT / "figures" / "figure_supply_1.png"
OUT_STATS = PROJECT_ROOT / "statistics" / "figure_supply_1_stats.csv"


# ============================================================
# Utility functions
# ============================================================

def exp_model(d, A, lamb, C):
    """Exponential decay model."""
    return A * np.exp(-d / lamb) + C

def exp_model1(d, A, lamb):
    """Exponential decay model."""
    return A * np.exp(-d / lamb);


# ============================================================
# Data loading
# ============================================================

compact = np.load(DATA_DIR / "figure_supply_1_compact.npz")
d = compact["d"]
lognorm_shape = compact["lognorm_shape"]
lognorm_scale = compact["lognorm_scale"]
hist_counts = compact["hist_counts"]
hist_edges = compact["hist_edges"]

# ============================================================
# Analysis
# ============================================================

# Exponential fit to scale parameter
p0 = [lognorm_scale.max(), 100.0, lognorm_scale.min()]
popt, pcov = curve_fit(exp_model, d, lognorm_scale, p0=p0, maxfev=10000)

A_hat, lambda_hat, C_hat = popt
perr = np.sqrt(np.diag(pcov))

d_fit = np.linspace(d.min(), d.max(), 300)
scale_fit = exp_model(d_fit, A_hat, lambda_hat, C_hat)

print("Mean lognormal shape parameter:", np.mean(lognorm_shape))
fit_stats = pd.DataFrame([
    {
        "figure": "figure_supply_1",
        "panel": "B",
        "model": "lognormal FC by distance bin",
        "statistic": "mean_lognormal_shape",
        "value": float(np.mean(lognorm_shape)),
        "standard_error": np.nan,
        "n_bins": int(len(lognorm_shape)),
    },
    {
        "figure": "figure_supply_1",
        "panel": "B",
        "model": "scale(d)=A*exp(-d/lambda)+C",
        "statistic": "A",
        "value": float(A_hat),
        "standard_error": float(perr[0]),
        "n_bins": int(len(d)),
    },
    {
        "figure": "figure_supply_1",
        "panel": "B",
        "model": "scale(d)=A*exp(-d/lambda)+C",
        "statistic": "lambda",
        "value": float(lambda_hat),
        "standard_error": float(perr[1]),
        "n_bins": int(len(d)),
    },
    {
        "figure": "figure_supply_1",
        "panel": "B",
        "model": "scale(d)=A*exp(-d/lambda)+C",
        "statistic": "C",
        "value": float(C_hat),
        "standard_error": float(perr[2]),
        "n_bins": int(len(d)),
    },
])

# ============================================================
# Plotting
# ============================================================

fig = plt.figure(figsize=(16.0, 4.0))

axA = plt.subplot2grid((1, 2), (0, 0))

axB = plt.subplot2grid((1, 2), (0, 1))


axs = [axA, axB]



for tax in axs:
    tax.tick_params(axis='both', which='both', direction='out',
                    bottom=True, left=True, length=4, width=1.2)





'''
p0 = [lognorm_scale.max(), 100.0];
popt, pcov = curve_fit(exp_model1, dist_data, fc_data, p0=p0, maxfev=10000)

A_hat, lambda_hat = popt
d_fit = np.linspace( dist_data.min(),  dist_data.max(), 300)
scale_fit = exp_model1(d_fit, A_hat, lambda_hat)
'''
'''
axC.hexbin(dist_data+0.01 ,fc_data,bins='log',xscale='log');
axC.plot(d_fit ,scale_fit);

textstr = (
    r'$y(d)=A e^{-d/\lambda}+C$' '\n'
    rf'$A = {A_hat:.3f} \pm {perr[0]:.3f}$' '\n'
    rf'$\lambda = {lambda_hat:.1f} \pm {perr[1]:.1f}$' '\n'
)

axC.text(
    0.05, 0.45, textstr,
    fontsize=11,
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
'''

axA.axvline(
    x=220,
    linestyle='--',
    linewidth=2,
    color='k',
    alpha=0.8
)


# --- Panel A: Synapse distance distribution
axA.bar(
    hist_edges[:-1],
    hist_counts,
    width=np.diff(hist_edges),
    align="edge",
)
axA.set_xlabel('Distance')
axA.set_ylabel('Count')

shape_color = 'tab:blue'
scale_color = 'tab:red'

axB_shape = axB                      # left y-axis
axB_scale = axB.twinx()              # right y-axis

# Scatter plots
axB_shape.scatter(
    d, lognorm_shape,
    color=shape_color,
    label='Shape(FC)',
    alpha=0.8
)

axB_scale.scatter(
    d, lognorm_scale,
    color=scale_color,
    marker='s',
    label='Scale(FC)',
    alpha=0.8
)

# Exponential fit (Scale)
axB_scale.plot(
    d_fit, scale_fit,
    color=scale_color,
    linewidth=2
)

# Axis labels
axB_shape.set_xlabel('Distance')
axB_shape.set_ylabel('Lognormal shape(FC)', color=shape_color)
axB_scale.set_ylabel('Lognormal scale(FC)', color=scale_color)

axB_shape.set_ylim((1.2,1.4));

# Right y-axis spine visible for twinx
axB_scale.spines['right'].set_visible(True)
axB_scale.spines['right'].set_color(scale_color)

# Tick colors
axB_shape.tick_params(axis='y', colors=shape_color)
axB_scale.tick_params(axis='y', colors=scale_color, direction='out', length=4, width=1.2)

# Text box (keep neutral)
textstr = (
    r'$y(d)=A e^{-d/\lambda}+C$' '\n'
    rf'$A = {A_hat:.3f} \pm {perr[0]:.3f}$' '\n'
    rf'$\lambda = {lambda_hat:.1f} \pm {perr[1]:.1f}$' '\n'
    rf'$C = {C_hat:.3f} \pm {perr[2]:.3f}$'
)

axB_shape.text(
    0.65, 0.52, textstr,
    transform=axB_shape.transAxes,
    fontsize=fs.SUPP_SMALL_FS, fontstyle='italic',
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# Legend (merged)
handles1, labels1 = axB_shape.get_legend_handles_labels()
handles2, labels2 = axB_scale.get_legend_handles_labels()
axB_shape.legend(
    handles1 + handles2,
    labels1 + labels2,
    frameon=False,
    loc='upper right',
    bbox_to_anchor=(1.0, 1.15),
    fontsize=fs.SUPP_SMALL_FS,
)

axes=[axA, axB]
for ax in axes:  
    pos = ax.get_position()
    ax.set_position([pos.x0+0.015, pos.y0+0.11, pos.width*0.8, pos.height*0.8]) 
    x0=pos.width*0.20
    
add_panel_label_fig(
    fig, axA, 'A', dx=-0.09, dy=0.02, fontsize=fs.SUPP_PANEL_LABEL_FS
)
add_panel_label_fig(
    fig, axB, 'B', dx=-0.09, dy=0.02, fontsize=fs.SUPP_PANEL_LABEL_FS
)


OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
fit_stats.to_csv(OUT_STATS, index=False)
plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved {OUT_PNG}")
print(f"Saved {OUT_STATS}")
