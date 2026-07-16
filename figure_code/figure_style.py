# ============================================================
# Global figure style (paper-wide)
# ============================================================
import logging
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def shrink_axis_in_rect(ax, pad_left=0.15, pad_right=0.0,
                        pad_bottom=0.0, pad_top=0.0):
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + pos.width * pad_left,
        pos.y0 + pos.height * pad_bottom,
        pos.width * (1 - pad_left - pad_right),
        pos.height * (1 - pad_bottom - pad_top)
    ])
          
def add_panel_label(ax, label, x=-0.28, y=1.12):
    ax.text(
        x,y, label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        ha='left'
    )
    
    
## ── Layout constants ─────────────────────────────────────────────
SINGLE_COL_CM = 8.6
SINGLE_COL_IN = SINGLE_COL_CM / 2.54   # ≈ 3.39"
TWO_COL_CM    = 17.8
TWO_COL_IN    = TWO_COL_CM / 2.54      # ≈ 7.01"

FONT_SIZE = 15

FONT_SCALE_1COL = 0.65   # one-column figures
FONT_SCALE_2COL = 0.75   # two-column figures

def _r(x): return int(x + 0.5)   # round-half-up

# ── Derived font sizes – one-column ───────────────────────────
AXIS_LABEL_FS_1COL  = _r(FONT_SIZE * FONT_SCALE_1COL)           # 10
TICK_FS_1COL        = _r((FONT_SIZE - 2) * FONT_SCALE_1COL)      # 8
PANEL_LABEL_FS_1COL = _r((FONT_SIZE + 1) * FONT_SCALE_1COL)     # 10
STAR_FS_1COL        = _r((FONT_SIZE - 5) * FONT_SCALE_1COL)      # 7
STAT_FS_1COL        = _r((FONT_SIZE - 6) * FONT_SCALE_1COL)      # 6

# ── Derived font sizes – two-column ───────────────────────────
AXIS_LABEL_FS_2COL  = _r(FONT_SIZE * FONT_SCALE_2COL)           # 11
TICK_FS_2COL        = _r((FONT_SIZE - 2) * FONT_SCALE_2COL)      # 10
PANEL_LABEL_FS_2COL = _r((FONT_SIZE + 1) * FONT_SCALE_2COL)     # 12
STAR_FS_2COL        = _r((FONT_SIZE - 5) * FONT_SCALE_2COL)      # 8
STAT_FS_2COL        = _r((FONT_SIZE - 6) * FONT_SCALE_2COL)      # 7

# ── Standard GridSpec margins ──────────────────────────────────
MARGIN_1COL = dict(left=0.14, right=0.95, top=0.90, bottom=0.15)
MARGIN_2COL = dict(left=0.07, right=0.97, top=0.93, bottom=0.12)

# Main-figure defaults used across the final six figures. Figure-specific
# geometry stays in each figure script; shared visual style lives here.
MAIN_PANEL_LABEL_FS = PANEL_LABEL_FS_2COL
MAIN_TITLE_FS = AXIS_LABEL_FS_2COL
MAIN_AXIS_FS = AXIS_LABEL_FS_2COL
MAIN_TICK_FS = TICK_FS_2COL
MAIN_SMALL_FS = STAT_FS_2COL
MAIN_STAR_FS = STAR_FS_2COL
MAIN_LINE_W = 1.0
MAIN_AXIS_LW = 1.0
MAIN_MARKER_SIZE = 3.0
MAIN_SCATTER_SIZE = 24
MAIN_SCATTER_EDGE_LW = 0.35
MAIN_BOX_WIDTH = 0.45
MAIN_BOX_LW = 1.0
MAIN_STRIP_SIZE = 3.0
MAIN_STRIP_ALPHA = 0.20
MAIN_GRID_LW = 0.6
MAIN_REFERENCE_COLOR = "#AAAAAA"
MAIN_FIGURE_WIDTH = 8.0
MAIN_FIGURE_HEIGHT_SHORT = 5.0
MAIN_FIGURE_HEIGHT_MODEL = 6.4
MAIN_FIGURE_HEIGHT_TALL = 7.2

# Supplementary figures use the same typographic hierarchy as the main figures
# so that text has a consistent printed size at two-column width.
SUPP_PANEL_LABEL_FS = PANEL_LABEL_FS_2COL
SUPP_TITLE_FS = AXIS_LABEL_FS_2COL
SUPP_AXIS_FS = AXIS_LABEL_FS_2COL
SUPP_TICK_FS = TICK_FS_2COL
SUPP_SMALL_FS = STAT_FS_2COL
SUPP_STAR_FS = STAR_FS_2COL

MAIN_STYLE = {
    "panel_fs": MAIN_PANEL_LABEL_FS,
    "title_fs": MAIN_TITLE_FS,
    "axis_fs": MAIN_AXIS_FS,
    "tick_fs": MAIN_TICK_FS,
    "small_fs": MAIN_SMALL_FS,
    "star_fs": MAIN_STAR_FS,
    "line_w": MAIN_LINE_W,
    "axis_lw": MAIN_AXIS_LW,
    "marker_size": MAIN_MARKER_SIZE,
    "scatter_size": MAIN_SCATTER_SIZE,
    "scatter_edge_lw": MAIN_SCATTER_EDGE_LW,
    "box_width": MAIN_BOX_WIDTH,
    "box_lw": MAIN_BOX_LW,
    "strip_size": MAIN_STRIP_SIZE,
    "strip_alpha": MAIN_STRIP_ALPHA,
    "grid_lw": MAIN_GRID_LW,
    "reference_color": MAIN_REFERENCE_COLOR,
}

def set_paper_style():
    mpl.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FONT_SIZE,

        # Axes
        'axes.linewidth': 1.2,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Ticks
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.bottom': True,
        'ytick.left': True,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Legend
        'legend.frameon': False,
        'legend.fontsize': FONT_SIZE - 3,

        # Save
        'savefig.dpi': 600,
        'savefig.transparent': False,
    })

    # white background but keep ticks
    sns.set_style("white", {
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })
    # ensure ticks and despine are ON after seaborn overrides
    mpl.rcParams.update({
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def apply_main_figure_style():
    """Apply the shared compact style for final main figures."""
    set_paper_style()
    mpl.rcParams.update({
        "font.size": MAIN_AXIS_FS,
        "axes.labelsize": MAIN_AXIS_FS,
        "axes.titlesize": MAIN_TITLE_FS,
        "xtick.labelsize": MAIN_TICK_FS,
        "ytick.labelsize": MAIN_TICK_FS,
        "axes.linewidth": MAIN_AXIS_LW,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": MAIN_LINE_W,
        "lines.markersize": MAIN_MARKER_SIZE,
        "legend.fontsize": MAIN_SMALL_FS,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def apply_supplement_figure_style():
    """Apply the shared typography used by final supplementary figures."""
    apply_main_figure_style()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": SUPP_AXIS_FS,
        "axes.labelsize": SUPP_AXIS_FS,
        "axes.titlesize": SUPP_TITLE_FS,
        "xtick.labelsize": SUPP_TICK_FS,
        "ytick.labelsize": SUPP_TICK_FS,
        "legend.fontsize": SUPP_SMALL_FS,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Nimbus Sans",
        "mathtext.it": "Nimbus Sans:italic",
        "mathtext.bf": "Nimbus Sans:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_main_axis(ax, tick_fs=None, axis_lw=None, tick_len=3.0, tick_width=0.8):
    """Apply common tick and spine styling to one axes."""
    if tick_fs is None:
        tick_fs = MAIN_TICK_FS
    if axis_lw is None:
        axis_lw = MAIN_AXIS_LW
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        bottom=True,
        left=True,
        labelsize=tick_fs,
        length=tick_len,
        width=tick_width,
    )
    for side in ("left", "bottom"):
        if side in ax.spines:
            ax.spines[side].set_linewidth(axis_lw)
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)


def add_main_panel_label(ax, label, x=-0.105, y=1.075, fontsize=None, **kwargs):
    """Place a compact panel label in axes coordinates."""
    if fontsize is None:
        fontsize = MAIN_PANEL_LABEL_FS
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha=kwargs.pop("ha", "left"),
        va=kwargs.pop("va", "bottom"),
        color=kwargs.pop("color", "#111111"),
        clip_on=kwargs.pop("clip_on", False),
        **kwargs,
    )


def division_palette(order=None):
    """Return the shared zebrafish division palette in a requested order."""
    if order is None:
        order = ["Tel", "Di", "Mes", "Hind"]
    return [ZEBRAFISH_DIVISION_COLORS[d] for d in order]


def draw_main_box_strip(ax, labels, values, order, palette=None):
    """Draw a main-figure boxplot with matched strip points."""
    if palette is None:
        palette = division_palette(order)
    sns.boxplot(
        x=labels,
        y=values,
        order=order,
        hue=labels,
        hue_order=order,
        legend=False,
        palette=palette,
        saturation=1.0,
        showfliers=False,
        width=MAIN_BOX_WIDTH,
        linewidth=MAIN_BOX_LW,
        ax=ax,
    )
    sns.stripplot(
        x=labels,
        y=values,
        order=order,
        color="black",
        size=MAIN_STRIP_SIZE,
        alpha=MAIN_STRIP_ALPHA,
        jitter=True,
        ax=ax,
    )
    style_main_axis(ax)


def add_panel_label_left_of_ylabel(fig, ax, label, fontsize=None):
    """Place panel label just left of the y-axis title, above the axes."""
    if fontsize is None:
        fontsize = FONT_SIZE - 2
    fig.canvas.draw()
    ylabel = ax.yaxis.label
    bbox = ylabel.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    pos = ax.get_position()
    fig.text(bbox_fig.x0 - 0.005, pos.y1 + 0.02,
             label,
             fontsize=fontsize, fontweight='bold',
             va='bottom', ha='right')

def add_panel_label_fig(fig, ax, label, dx=-0.02, dy=0.01, fontsize=None):
    if fontsize is None:
        fontsize = FONT_SIZE - 2
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        fontsize=fontsize,
        fontweight='bold'
    )

def scatter_with_reg(ax, x, y, color,text_x,text_y,fontsize=12):
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    c = np.asarray(color, dtype=object)[mask]
    import seaborn as sns
    sns.regplot(x=x, y=y, ax=ax,
            ci=95, # 95% 신뢰 구간 (그림의 회색 영역)
            scatter_kws={'color': 'salmon', 'edgecolor': 'black', 'alpha': 0.4}, # 점 스타일
            line_kws={'color': 'black', 'lw': 2}) # 선 스타일

    ax.scatter(x, y, c=c, s=135, edgecolor='none')

    if len(x) > 1:
        r, p = pearsonr(x, y)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, lw=2,c='k')
        ax.text(
            text_x, text_y,
            f"r={r:.3f}\np={p:.3g}",
            transform=ax.transAxes,
            fontsize=fontsize,
            ha='left', va='top'
            #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    '''
    for i in range(len(sel_idx)):
        col = 'black';
        if(region[sel_idx[i]]=='P' or region[sel_idx[i]]=='rP' or region[sel_idx[i]]=='SP' or region[sel_idx[i]]=='rSP' or region[sel_idx[i]]=='Cb' or region[sel_idx[i]]=='rCb'):
            ax.annotate(
                region[sel_idx[i]],
                (x[i], y[i]),
                xycoords='data',
                xytext=(x[i] + 0.001, y[i] + 0.001),
                textcoords='data',
                fontsize=13,
                color=col
            )
    ''' 
    #ax.set_xlabel(xlabel)
    
    
def add_significance_bars(
    ax,
    data_by_division,
    divisions,
    division_names,
    priority_division=None,   # ← 핵심 division (예: Tel)
    y_padding=0.20,          # 축 범위 대비 padding (작게!)
    bar_spacing=0.08,         # bar 간격 (작게!)
    bar_height=0.01,
    fontsize=12
):
    """
    논문용 significance bars:
    - y축 스케일 절대 망가뜨리지 않음
    - 메시지 우선순위 기반 bar 순서
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import mannwhitneyu
    from scipy.stats import kruskal
    import itertools

    # 1. 유효 division
    active_divs = [d for d in divisions if len(data_by_division[d]) > 0]
    if len(active_divs) < 2:
        return

    pairs = list(itertools.combinations(active_divs, 2))

    # 2. Kruskal–Wallis (로그용)
    kw_stat, kw_p = kruskal(*[data_by_division[d] for d in active_divs])
    logger.info(f"Kruskal–Wallis: H={kw_stat:.3f}, p={kw_p:.4g}")
    try:
        kw_stat, kw_p = kruskal(*[data_by_division[d] for d in active_divs])
        logger.info(f"Kruskal–Wallis: H={kw_stat:.3f}, p={kw_p:.4g}")
    except Exception:
        return

    # 3. Pairwise Mann–Whitney
    pvals = []
    for d1, d2 in pairs:
        try:
            _, p = mannwhitneyu(
                data_by_division[d1],
                data_by_division[d2],
                alternative="two-sided"
            )
        except ValueError:
            p = 1.0
        pvals.append(p)

    reject, pvals_corr, _, _ = multipletests(pvals, method="holm")

    # 4. 유의한 pair만 정리
    sig_pairs = []
    for i, (d1, d2) in enumerate(pairs):
        if reject[i]:
            x1, x2 = divisions.index(d1), divisions.index(d2)
            sig_pairs.append({
                "d1": d1,
                "d2": d2,
                "x1": min(x1, x2),
                "x2": max(x1, x2),
                "p": pvals_corr[i],
                "dist": abs(x1 - x2),
                "priority": (
                    0 if priority_division in (d1, d2) else 1
                )
            })

    if not sig_pairs:
        return

    # 5. 정렬 규칙
    # (1) priority division 포함
    # (2) bar 길이 짧은 순
    sig_pairs.sort(key=lambda x: (x["priority"], x["dist"], x["x1"]))

    # 6. y 위치 계산 (현재 축 범위 기준!)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    y_start = y_max + y_range * y_padding
    spacing = y_range * bar_spacing
    bar_h = y_range * bar_height

    # 7. bar 그리기 (y축 확장 ❌)
    for i, pair in enumerate(sig_pairs):
        x1, x2 = pair["x1"], pair["x2"]
        p = pair["p"]
        
        y = y_start - i * spacing

        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bar_h, y + bar_h, y],
            lw=1.2,
            c="#333333",
            clip_on=False   # ← 중요!
        )

        if p < 0.001:
            label = "***"
        elif p < 0.01:
            label = "**"
        else:
            label = "*"

        ax.text(
            (x1 + x2) / 2,
            y - bar_h/10,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False
        )
        ax.set_ylim(y_min, y_max);
'''

def add_significance_bars(
    ax,
    data_by_division,
    divisions,
    division_names,
    y_padding=0.1,    # 데이터 최상단과 첫 번째 바 사이의 여백 (Y축 범위 비율)
    bar_spacing=0.08, # 바와 바 사이의 간격 (Y축 범위 비율)
    bar_height=0.02,  # 바 양끝 꺾이는 부분의 높이
    fontsize=15
):
    """
    개선된 Holm-corrected Mann-Whitney significance bars:
    - 바 길이에 따라 정렬하여 겹침 방지
    - Y축 스케일에 따라 간격 자동 조절
    """
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import mannwhitneyu
    import itertools

    # 1. 데이터가 있는 division만 필터링 (에러 방지)
    active_divs = [d for d in divisions if len(data_by_division[d]) > 0]
    if len(active_divs) < 2:
        logger.debug("Not enough non-empty divisions for significance testing.")
        return

    pairs = list(itertools.combinations(active_divs, 2))

    # Kruskal–Wallis using only active divisions
    try:
        kw_stat, kw_p = kruskal(*[data_by_division[d] for d in active_divs])
        logger.info(f"Kruskal–Wallis : H={kw_stat:.3f}, p={kw_p:.4g}")
    except Exception as e:
        logger.warning(f"Kruskal–Wallis test failed: {e}")
        return

    pvals = []
    pair_labels = []
    for d1, d2 in pairs:
        try:
            _, p = mannwhitneyu(data_by_division[d1], data_by_division[d2], alternative='two-sided')
        except ValueError:
            # If one of the groups is constant or empty, mannwhitneyu may fail
            p = 1.0
        pvals.append(p)
        pair_labels.append((d1, d2))

    reject, pvals_corr, _, _ = multipletests(pvals, method='holm')

    significant_pairs = [
        (pair_labels[i][0], pair_labels[i][1], pvals_corr[i])
        for i in range(len(pairs)) if reject[i]
    ]

    if significant_pairs:
        logger.info("Significant pairs (Holm corrected):")
        for d1, d2, p in significant_pairs:
            logger.info(f"{division_names[d1]} vs {division_names[d2]} : p={p:.4g}")

    # 2. 유의미한 쌍만 추출하고 '거리' 계산 (정렬용)
    sig_pairs = []
    for i in range(len(pairs)):
        if reject[i]:
            d1, d2 = pairs[i]
            x1, x2 = divisions.index(d1), divisions.index(d2)
            sig_pairs.append({
                'x1': min(x1, x2),
                'x2': max(x1, x2),
                'p': pvals_corr[i],
                'dist': abs(x1 - x2)
            })

    if not sig_pairs:
        return

    # 3. 중요: 바 길이가 짧은 순서대로 정렬 (아래서부터 쌓기 위해)
    sig_pairs = sorted(sig_pairs, key=lambda x: (x['dist'], x['x1']))

    # 4. Y축 범위를 기준으로 간격 계산 (자동 스케일링)
    y_min, y_max_actual = ax.get_ylim()
    # 실제 데이터의 최댓값 찾기 (안전하게 처리)
    y_max_data = max((max(data_by_division[d]) if len(data_by_division[d]) > 0 else float('-inf')) for d in active_divs)
    y_range = y_max_actual - y_min
    
    # 비율을 실제 값으로 변환
    actual_padding = y_range * y_padding
    actual_spacing = y_range * bar_spacing
    actual_bar_h = y_range * bar_height

    # 5. 바 그리기
    for level, pair in enumerate(sig_pairs):
        x1, x2 = pair['x1'], pair['x2']
        p = pair['p']
        
        # 레벨에 따라 높이 설정
        y_base = y_max_data + actual_padding + (level * actual_spacing)
        
        # 바 그리기
        ax.plot(
            [x1, x1, x2, x2],
            [y_base, y_base + actual_bar_h, y_base + actual_bar_h, y_base],
            lw=1.2, c='k'
        )

        # 텍스트 표시 (p값에 따라 별표 개수 조절 가능)
        if p < 0.001:
            p_text = "***"
        elif p < 0.01:
            p_text = "**"
        else:
            p_text = "*"
        
        # 혹은 요청하신 대로 (p=0.000) 형식 유지
        #p_label = f"{p_text}\n(p={p:.3f})" if p >= 0.001 else f"{p_text}\n(p<0.001)"
        p_label = f"{p_text}" if p >= 0.001 else f"{p_text}"

        ax.text(
            (x1 + x2) / 2,
            y_base + actual_bar_h * 0.0,
            p_label,
            ha='center', va='bottom',
            fontsize=fontsize,
            linespacing=0.8
        )

    # 6. 바가 그려진 후 Y축 범위 자동 확장
    new_y_max = y_max_data + actual_padding + (len(sig_pairs) * actual_spacing) + actual_spacing
    ax.set_ylim(y_min, new_y_max)
'''
'''               
division_colors = {
    0: '#1f77b4',  # Hindbrain, Rhombencephalon
    1: '#ff7f0e',  # Diencephalon
    2: '#2ca02c',  # Telencephalon
    3: '#d62728',  # Midbrain, Mesencephalon
}
'''
division_colors = {
    0: '#6FA8C9',  # Hindbrain / rhombencephalon
    1: '#DDA15E',  # Diencephalon
    2: '#6FAF6B',  # Telencephalon
    3: '#E76F51',  # Mesencephalon
}

# Shared semantic palette for main figures.  Keep nonessential categories
# neutral; use two accents for directed structure and two for key groups.
MAIN_COLORS = {
    "neutral_dark": "#2F3437",
    "neutral": "#7A7F83",
    "neutral_light": "#C8CDD0",
    "neutral_fill": "#ECEEEF",
    "post_out": "#D98C3A",
    "pre_in": "#4C78A8",
    "olfactory": "#4E9A9A",
    "associative": "#B65F73",
}

ZEBRAFISH_DIVISION_COLORS = {
    "Tel": division_colors[2],
    "Di": division_colors[1],
    "Mes": division_colors[3],
    "Hind": division_colors[0],
}

FUNCTIONAL_GROUP_COLORS = {
    0: MAIN_COLORS["olfactory"],
    1: MAIN_COLORS["neutral_light"],
    2: MAIN_COLORS["neutral_light"],
    3: MAIN_COLORS["neutral_light"],
    4: MAIN_COLORS["neutral_light"],
    5: MAIN_COLORS["associative"],
    6: MAIN_COLORS["neutral_light"],
    -1: MAIN_COLORS["neutral_light"],
}

SPECIES_COLORS = {
    "C. elegans": MAIN_COLORS["olfactory"],
    "Drosophila": MAIN_COLORS["associative"],
    "Zebrafish": division_colors[2],
}

METRIC_COLORS = {
    "FCV": MAIN_COLORS["neutral_dark"],
    "Post-DCA": MAIN_COLORS["post_out"],
    "Pre-DCA": MAIN_COLORS["pre_in"],
}



division_names = {
    2: 'Tel',
    1: 'Di',
    3: 'Mes',
    0: 'Hind'
}

# 🔑 desired order
valid_divisions = [2, 1, 3, 0]



def plot_division_box_with_stats(
    ax,
    division_data,
    ylabel,
    strip_size=6,
    strip_alpha=0.3,
):
    plot_data = []
    plot_labels = []

    for d in valid_divisions:
        plot_data.extend(division_data[d])
        plot_labels.extend([division_names[d]] * len(division_data[d]))

    for d in valid_divisions:
        print(division_names[d], np.mean(division_data[d]), np.std(division_data[d],ddof=1)/np.sqrt(len(division_data[d])), len(division_data[d]));
    sns.boxplot(
        x=plot_labels,
        y=plot_data,
        showfliers=False,
        width=0.4,
        linewidth=1.2,
        palette=[division_colors[d] for d in valid_divisions],
        ax=ax
    )

    sns.stripplot(
        x=plot_labels,
        y=plot_data,
        color='black',
        size=strip_size,
        alpha=strip_alpha,
        jitter=True,
        ax=ax
    )

    ax.set_ylabel(ylabel)

    # Kruskal–Wallis
    #kw_stat, kw_p = kruskal(*[division_data[d] for d in valid_divisions])
    #ax.set_title(f"{ylabel}\nKruskal–Wallis p = {kw_p:.3g}", fontsize=20)

    # significance bars
    add_significance_bars(
        ax,
        division_data,
        valid_divisions,
        division_names
    )

brain_division_list = [
    0,  # 1  Medial Octavolateral Nucleus
    0,  # 2  Cerebellum
    0,  # 3  Medulla Oblongata strip1
    0,  # 4  Medulla Oblongata strip2
    0,  # 5  Medulla Oblongata strip3
    0,  # 6  Medulla Oblongata strip4
    0,  # 7  Medulla Oblongata strip5
    0,  # 8  Interpeduncular Nucleus
    0,  # 9  Inferior Olive
    1,  # 10 Caudal Hypothalamus
    0,  # 11 Raphe Nucleus
    0,  # 12 Tegmentum
    0,  # 13 Anterior Reticular Formation
    0,  # 14 Intermediate Reticular Formation
    0,  # 15 Posterior Reticular Formation
    4,  # 16 Glossopharyngeal Ganglion
    1,  # 17 Habenula
    1,  # 18 Intermediate Hypothalamus
    1,  # 19 Rostral Hypothalamus
    4,  # 20 Octaval Ganglion
    2,  # 21 Olfactory Bulb
    2,  # 22 Olfactory Epithelium
    2,  # 23 Pallium
    1,  # 24 Pituitary
    1,  # 25 Posterior Tuberculum
    2,  # 26 Preoptic Region
    1,  # 27 Pretectum
    4,  # 28 Retina
    2,  # 29 Subpallium
    3,  # 30 Tectum
    1,  # 31 Thalamus
    3,  # 32 Torus Longitudinalis
    3,  # 33 Torus Semicircularis
    4,  # 34 Trigeminal Ganglion
    0,  # 35 Vagal Region
    0   # 36 Vagus Motor Neurons
]

brain_division_list=np.array(brain_division_list)
brain_division_list=np.concatenate((brain_division_list,brain_division_list));

def calculate_average_sliding_window_fc(data, window_size, overlap):
    n_areas, n_time = data.shape
    step = window_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than window_size.")

    corr_list = []
    for t in range(0, n_time - window_size + 1, step):
        corr_list.append(np.corrcoef(data[:, t:t + window_size]))

    corr_stack = np.array(corr_list)
    return (
        np.nanmean(corr_stack, axis=0),
        np.nanstd(corr_stack, axis=0),
        corr_stack
    )
    
    

division_colors = {
    0: '#6FA8C9',  # Hindbrain
    1: '#DDA15E',  # Diencephalon
    2: '#6FAF6B',  # Telencephalon
    3: '#E76F51',  # Mesencephalon
}

lregion = ['lMON','lCb','lMOS1','lMOS2','lMOS3','lMOS4','lMOS5','lIPN','lIO','lHc','lRa','lT',
           'laRF','limRF','lpRF','lGG','lHb','lHi','lHR','lOG','lOB','lOE','lP','lPi','lPT',
           'lPO','lPrT','lR','lSP','lTeO','lTh','lTL','lTS','lTG','lVR','lNX']

rregion = ['rMON','rCb','rMOS1','rMOS2','rMOS3','rMOS4','rMOS5','rIPN','rIO','rHc',
           'rRa','rT','raRF','rimRF','rpRF','rGG','rHb','rHi','rHR','rOG','rOB',
           'rOE','rP','rPi','rPT','rPO','rPrT','rR','rSP','rTeO','rTh','rTL',
           'rTS','rTG','rVR','rNX']

region = lregion + rregion

def darw_region_bar(ax,x_mean,sel_n):
    
    mean_x_mean=[];
    #mean_y_mean=[];
    
      
    err_x_mean=[];
    #err_y_mean=[];
    #p_values = []
    
    
    
    '''
    for i in range(len(sel_n)):
        ii = sel_n[i]

        xv = x_mean[ii]
        yv = y_mean[ii]
        
        print(region[ii],np.mean(xv),np.std(xv),np.mean(yv),np.std(yv),pvals_corr[i])
    '''
    #pvals_corr= p_values
    
    def p_to_star(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return None
    import random
    for i in range(len(sel_n)):
        ii=sel_n[i];
        mean_x_mean.append(np.mean(x_mean[ii]));
        #mean_y_mean.append(np.mean(y_mean[ii]));
        
        #mean_x_mean=np.array(mean_x_mean);
        #mean_y_mean=np.array(mean_y_mean);
        
        #random.shuffle(x_mean[ii])
        #random.shuffle(y_mean[ii])
                    
        #delta = np.array(x_mean[ii])-np.array(y_mean[ii]);
            
            

        #mean_delta = np.mean(delta)
        #std_delta  = np.std(delta)

        #mean_x_mean.append(mean_delta);
        #mean_y_mean.append(mean_delta);
        
        #err_x_mean.append(std_delta/np.sqrt(len(delta)-1));
        #err_y_mean.append(std_delta/np.sqrt(len(delta)-1));
        
        
        
        '''
        n_perm = 10000
        perm_means = []
        for _ in range(n_perm):
            signs = np.random.choice([-1,1], size=len(delta))
            perm_means.append(np.mean(delta * signs))

        p_value = np.mean(np.abs(perm_means) >= np.abs(mean_delta))
        
        print( len(x_mean[ii]),len(y_mean[ii]))
        '''
        err_x_mean.append(np.std(x_mean[ii])/np.sqrt(len(x_mean[ii])));
        #err_y_mean.append(np.std(y_mean[ii])/np.sqrt(len(y_mean[ii])));
        
        
    from matplotlib.colors import to_rgba

    def lighten_color(color, amount=0.5):
        """amount < 1 → 밝게, amount > 1 → 진하게"""
        r, g, b, a = to_rgba(color)
        r = 1 - (1 - r) * amount
        g = 1 - (1 - g) * amount
        b = 1 - (1 - b) * amount
        return (r, g, b, a)

    valid_divisions = [2, 1, 3, 0]

    err_mean_color = 'b'        # FC Mean errorbar
    err_var_color  = 'r'  # FC Var errorb
    ordered_region_names = []
    ordered_divisions = []
    ordered_x=np.zeros(len(sel_n));
    k=0;
    for d in valid_divisions:
        for i in range(len(sel_n)):
            if brain_division_list[sel_n[i]] == d:
                #axB.errorbar(
                #    k, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
                #    fmt='o', color=colors[i], alpha=0.6
                #)
                #axB.errorbar(
                #    k, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
                #    fmt='D', color=colors[i], alpha=1.0
                #)
                
                base_color = division_colors[d]

                color_mean = lighten_color(base_color, amount=1.0)  # FC Mean (연함)
                color_var  = lighten_color(base_color, amount=1.0)

                error_kw=dict(
                    ecolor=err_var_color,
                    elinewidth=2.8,
                    capthick=2.8
                )
                
                ax.bar(
                    k-0.12, mean_x_mean[i], yerr=err_x_mean[i],label='FC Mean' if i == 0 else "",
                    color=color_mean, alpha=0.8,width=0.6, error_kw=error_kw
                )
                
                #error_kw=dict(
                #    ecolor=err_var_color,
                #    elinewidth=2.2,
                #    capthick=2.2
                #)
                    
                #ax.bar(
                #    k+0.12,  mean_y_mean[i], yerr= err_y_mean[i],label='FC Var.' if i == 0 else "",
                #    color=color_var, alpha=0.6,width=0.4, error_kw=error_kw
                # )

                ordered_region_names.append(region[sel_n[i]])
                ordered_divisions.append(d)
                ordered_x[i]=k;
                k += 1
    ax.set_xticks(range(len(ordered_region_names)), ordered_region_names, rotation=45,fontsize=10)
    for lbl, d in zip(ax.get_xticklabels(), ordered_divisions):
        lbl.set_color(division_colors[d])
        if d == 2:
            lbl.set_fontweight('bold')
            lbl.set_fontsize(12)
            lbl.set_fontstyle('italic')
            
    prev_d = ordered_divisions[0]
    
    for i, d in enumerate(ordered_divisions):
        if d != prev_d:
            ax.axvline(i - 0.5, color='gray', lw=2, zorder=0)
        prev_d = d
    '''
    for i, p_corr in enumerate(pvals_corr):
        star = p_to_star(p_corr)
        if star is None:
            continue

        ymax = max(
            mean_x_mean[i] + err_x_mean[i],0
            #mean_y_mean[i] + err_y_mean[i]
        )

        ax.text(
            ordered_x[i], ymax * 1.08,
            star,
            ha='center', va='bottom',
            fontsize=14, fontweight='bold'
        )
    '''   

from matplotlib.colors import to_rgba
        
def lighten_color(color, amount=0.5):
    """amount < 1 → 밝게, amount > 1 → 진하게"""
    r, g, b, a = to_rgba(color)
    r = 1 - (1 - r) * amount
    g = 1 - (1 - g) * amount
    b = 1 - (1 - b) * amount
    return (r, g, b, a)


err_fcn_color = '#8C8C8C'   # light gray
err_fcv_color = '#1A1A1A'   # near black

#err_fcn_color = '#666666'        # FC Mean errorbar
#err_fcv_color  = '#333333'  # FC Var errorb

def draw_sig_bracket(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2, c='black')
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    
    
def draw_division_compare_bar_1(ax,mean_fc,sem_fc,mean_var,sem_var,new_region_array,colors,pvals_corr,x1,x2,legend_loc='center'):
    ordered_region_names = []
    ordered_divisions = []
    ordered_mean_fc = []
    ordered_mean_var = []
    k = 0       
    for d in valid_divisions:
        for i in range(len(new_region_array)):
            if brain_division_list[new_region_array[i]] == d:
                
                print(region[new_region_array[i]],mean_fc[i],sem_fc[i],mean_var[i],sem_var[i],pvals_corr[i])
                #axBt.errorbar(
                #    k, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
                #    fmt='o', color=colors[i], alpha=0.6
                #)
                #axBt.errorbar(
                #    k, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
                #    fmt='D', color=colors[i], alpha=1.0
                #)
                
                base_color = colors[i]

                color_mean = lighten_color(base_color, amount=0.6)  # FC Mean (연함)
                color_var  = lighten_color(base_color, amount=1.0)

                error_kw=dict(
                    ecolor=err_fcn_color,
                    elinewidth=2.2,
                    capthick=2.2
                )
                
                ax.bar(
                    k-0.12, mean_fc[i], yerr=sem_fc[i], label='FC Mean' if i == 0 else "",
                                        color=color_mean, alpha=0.6,width=0.4, error_kw=error_kw
                                    )

                                    # add significance star above the FC Mean bar based on pvals_corr[i]
                p = pvals_corr[i]
                
                
                if p is not None:
                    if p < 0.001:
                        star = '***'
                    elif p < 0.01:
                        star = '**'
                    elif p < 0.05:
                        star = '*'
                    else:
                        star = None

                    if star:
                        y0, y1 = ax.get_ylim()
                        y_text =0.7
                        #ax.text(k-0.12, y_text, star, ha='center', va='bottom', fontsize=24, fontweight='bold')
                        draw_sig_bracket(ax, k-0.12, k+0.12, y_text-0.05, 0.03, star)
                error_kw=dict(
                    ecolor=err_fcv_color,
                    elinewidth=2.2,
                    capthick=2.2
                )
                    
                ax.bar(
                    k+0.12, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
                    color=color_var, alpha=0.6,width=0.4, error_kw=error_kw

                )
                
                
                tregion_name=region[new_region_array[i]]
                ordered_region_names.append(tregion_name)
                ordered_divisions.append(d)
                ordered_mean_fc.append(mean_fc[i])
                ordered_mean_var.append(mean_var[i])
                k += 1

    #axBt.plot(ordered_mean_fc, color='black')
    #axBt.plot(ordered_mean_var, '--', color='black')
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_xticks(range(len(ordered_region_names)), ordered_region_names, rotation=45,fontsize=12)
    for lbl, d in zip(ax.get_xticklabels(), ordered_divisions):
        lbl.set_color(division_colors[d])
        #if d == 2:
        #    lbl.set_fontweight('bold')
        #    lbl.set_fontsize(14)
        #    lbl.set_fontstyle('italic')

    prev_d = ordered_divisions[0]
    for i, d in enumerate(ordered_divisions):
        if d != prev_d:
            ax.axvline(i - 0.5, color='black', lw=2, zorder=0)
        prev_d = d
        
    from matplotlib.lines import Line2D
    
    error_legend = [
        Line2D(
            [0], [0],
            color=err_fcn_color,
            lw=2.2,
            label=x1
        ),
        Line2D(
            [0], [0],
            color=err_fcv_color,
            lw=2.2,
            label=x2
        )
    ]

    leg2 = ax.legend(
        handles=error_legend,
        loc='lower center',
        frameon=False,
        bbox_to_anchor=(0.9, 0.7)
    )



def draw_division_compare_bar(ax,mean_fc,sem_fc,mean_var,sem_var,new_region_array,colors,pvals_corr,x1,x2,legend_loc='center'):
    ordered_region_names = []
    ordered_divisions = []
    ordered_mean_fc = []
    ordered_mean_var = []
    k = 0       
    for d in valid_divisions:
        for i in range(len(new_region_array)):
            if brain_division_list[new_region_array[i]] == d:
                
                #axBt.errorbar(
                #    k, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
                #    fmt='o', color=colors[i], alpha=0.6
                #)
                #axBt.errorbar(
                #    k, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
                #    fmt='D', color=colors[i], alpha=1.0
                #)
                
                base_color = colors[i]

                color_mean = lighten_color(base_color, amount=0.6)  # FC Mean (연함)
                color_var  = lighten_color(base_color, amount=1.0)

                error_kw=dict(
                    ecolor=err_fcn_color,
                    elinewidth=2.2,
                    capthick=2.2
                )
                
                ax.bar(
                    k-0.12, mean_fc[i], yerr=sem_fc[i], label='FC Mean' if i == 0 else "",
                                        color=color_mean, alpha=0.6,width=0.4, error_kw=error_kw
                                    )

                                    # add significance star above the FC Mean bar based on pvals_corr[i]
                p = pvals_corr[i]
                if p is not None:
                    if p < 0.001:
                        star = '***'
                    elif p < 0.01:
                        star = '**'
                    elif p < 0.05:
                        star = '*'
                    else:
                        star = None

                    if star:
                        y0, y1 = ax.get_ylim()
                        y_text =0.7
                        ax.text(k-0.12, y_text, star, ha='center', va='bottom', fontsize=24, fontweight='bold')
                
                error_kw=dict(
                    ecolor=err_fcv_color,
                    elinewidth=2.2,
                    capthick=2.2
                )
                    
                ax.bar(
                    k+0.12, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
                    color=color_var, alpha=0.6,width=0.4, error_kw=error_kw

                )
                
                
                tregion_name=region[new_region_array[i]]
                ordered_region_names.append(tregion_name)
                ordered_divisions.append(d)
                ordered_mean_fc.append(mean_fc[i])
                ordered_mean_var.append(mean_var[i])
                k += 1

    #axBt.plot(ordered_mean_fc, color='black')
    #axBt.plot(ordered_mean_var, '--', color='black')
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_xticks(range(len(ordered_region_names)), ordered_region_names, rotation=45,fontsize=12)
    for lbl, d in zip(ax.get_xticklabels(), ordered_divisions):
        lbl.set_color(division_colors[d])
        #if d == 2:
        #    lbl.set_fontweight('bold')
        #    lbl.set_fontsize(14)
        #    lbl.set_fontstyle('italic')

    prev_d = ordered_divisions[0]
    for i, d in enumerate(ordered_divisions):
        if d != prev_d:
            ax.axvline(i - 0.5, color='black', lw=2, zorder=0)
        prev_d = d
        
    from matplotlib.lines import Line2D
    
    error_legend = [
        Line2D(
            [0], [0],
            color=err_fcn_color,
            lw=2.2,
            label=x1
        ),
        Line2D(
            [0], [0],
            color=err_fcv_color,
            lw=2.2,
            label=x2
        )
    ]

    leg2 = ax.legend(
        handles=error_legend,
        loc='lower center',
        frameon=False,
        bbox_to_anchor=(0.9, 0.7)
    )

           
#set_paper_style()
