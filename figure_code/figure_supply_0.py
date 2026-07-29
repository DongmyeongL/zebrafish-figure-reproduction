import numpy as np
#import MONET_SNN_CUDA_PYTHON_BOOST as snn
import pickle
from pathlib import Path
import os
import pandas as pd
from scipy.stats import pearsonr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import figure_style as fig_help


"""figure5_proc.py

Light cleanup: added module docstring and removed duplicate definitions.
This script visualizes empirical and simulated calcium traces and FC metrics.
"""

fig_help.set_paper_style()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "raw_data" / "figure_supply_0"
OUT_PNG = PROJECT_ROOT / "figures" / "figure_supply_0.png"
OUT_STATS = PROJECT_ROOT / "statistics" / "figure_supply_0_stats.csv"


def get_brain_division_list() -> np.ndarray:
    """Return the brain division mapping for 36 regions (duplicated for left/right)."""
    base = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 1, 1, 1, 4, 2, 2,
        2, 1, 1, 2, 1, 4, 2, 3, 1, 3, 3, 4, 0, 0
    ])
    return np.concatenate((base, base))


def load_pickle_file(path: str) -> dict:
    """Load a pickle file and return its contents as a dict."""
    with open(path, 'rb') as pf:
        return pickle.load(pf)


load = np.load(DATA_DIR / 'figure1_emp_variation' / 'figure1_whole_brain_fc_mean_std_data.npz')
new_region_array = load['new_region_array']


set_path = DATA_DIR / 'figure5_simulation'
save_path = set_path / 'figure5_reprsesnt_data_data.pkl'
fig5 = load_pickle_file(save_path)

emp_ca_data: np.ndarray = fig5['emp_ca_data']
sim_ca_data: np.ndarray = fig5['sim_ca_data']
unique_indices = fig5['unique_indices']
root_area: np.ndarray = fig5['root_area']
save_fc = fig5['save_fc']
eave_fc = fig5['eave_fc']
estd_fc = fig5['estd_fc']
sstd_fc = fig5['sstd_fc']
sel_nn = fig5['sel_nn']
sel_region = fig5['sel_region']
color_region = fig5['color_region']

for i in range(len(sel_nn)):
    sel_region[i] = fig_help.region[root_area[sel_nn[i]]];


fig = plt.figure(figsize=(16, 19))

# =========================
# A) Empirical data
# =========================
axA = plt.subplot2grid((4, 24), (0, 0), colspan=16)
axB = plt.subplot2grid((4, 24), (0, 16), colspan=8)

axC1 = plt.subplot2grid((4, 24), (1, 0), colspan=6)
axC2 = plt.subplot2grid((4, 24), (1, 6), colspan=6)
axD1 = plt.subplot2grid((4, 24), (1, 12), colspan=6)
axD2 = plt.subplot2grid((4, 24), (1, 18), colspan=6)

axE = plt.subplot2grid((4, 24), (2, 18), colspan=6)
axF = plt.subplot2grid((4, 24), (3, 18), colspan=6)

axG = plt.subplot2grid((4, 24), (2, 0), colspan=18)
axH = plt.subplot2grid((4, 24), (3, 0), colspan=18)

axs = [axB,axC1, axC2, axD1, axD2, axE, axF, axG, axH]

for tax in axs:
    tax.tick_params(axis='both', which='both', direction='out',
                    bottom=True, left=True, length=4, width=1.2)
    
#axA.set_title("Empirical data")
axA.set_xlabel('time(s)')
# axA.plot(empirical_ts.T, lw=0.3)
#add_panel_label(axA, 'A')

k = 0
print(len(root_area),len(sel_nn));


x_tick_data = []
for i in range(len(sim_ca_data[0])):
    x_tick_data.append(i / 5)


dline=0.25;
for ii in unique_indices:
   
    
    bnn=fig_help.brain_division_list[root_area[ii]];
    
    if(bnn<3):
    
        
        xx=sim_ca_data[ii] ;
        xx = (xx - np.mean(xx));#/np.std(xx);
                
        axA.plot(x_tick_data, xx - k * dline, 'k', lw=1.0, color='black', alpha=1.0)
        
        print(fig_help.brain_division_list.shape, root_area[ii])
        
        bnn=fig_help.brain_division_list[root_area[ii]];
        cc=fig_help.division_colors[bnn];
        
        axA.text(-1, -k*dline, fig_help.region[root_area[ii]],color=cc, fontsize=6, ha='right', va='center')
        
        k = k + 1


scale_sec = 5          # 2 seconds
fs = 15                # font size

# x_tick_data는 0 ~ 300초 (0.5s step)이니까
x_end = 50

x0 = x_end - 1 - 0.1
y0 = -k*dline - 0.5            # 위로 이동
# scale bar 그리기
axA.plot([x0-6, x0 + scale_sec-6], [y0, y0], 'k', lw=2, clip_on=False)
axA.text(
    x0 + scale_sec/2-6, y0 - 0.3,
    f'{scale_sec} s',
    ha='center', va='top', fontsize=fs
)

axA.set_xlim((0,x_end));


#axA.set_xlim((0,600/2));
#add_panel_label(axA, 'A')
#axA.axis('off')             
            
# =========================
# B) Simulated data
# =========================

ave_fc, std_fc, corr_stack = fig_help.calculate_average_sliding_window_fc(
    sim_ca_data,
    window_size=20,
    overlap=15
)


for i, r in enumerate(root_area): 
    if r == 1:
        cb_n = i
    if r == 3:
        mos2_n = i
    if r == 22:
        p_n = i

xx = corr_stack[:, cb_n, p_n]
yy = corr_stack[:, cb_n, mos2_n]


xticks = np.arange(0, len(xx))    
print("Length of time series:", len(xx))
axB.plot(xticks,xx, lw=1.5, label='P–Cb',)
axB.plot(xticks,yy, lw=1.5, label='Cb–MOS2')
axB.set_ylabel("FC")
axB.set_xticks(np.arange(0, len(xx), 100))
axB.set_xlim(0, len(xx))

axB.set_ylim(-1.2, 1)
axB.set_yticks(np.arange(-1, 1.1, 0.5))

axB.set_xlabel("Time (s)")
axB.legend(loc='lower right', bbox_to_anchor=(0.98, -0.05))



# =========================
# C) Functional Connectivity
# =========================


axC1.set_title("Emp. FC")
axC2.set_title("Sim. FC")

 
im1 = axC1.imshow(eave_fc[np.ix_(sel_nn, sel_nn)], cmap='jet', vmin=-0.2, vmax=1, aspect='equal')
axC1.set_xticks(range(len(sel_region)))
axC1.set_xticklabels(sel_region, rotation=90, fontsize=6)
axC1.set_yticks(range(len(sel_region)))
axC1.set_yticklabels(sel_region, fontsize=6)

#cbar1 = plt.colorbar(im1, ax=axC1, fraction=0.046, pad=0.04)

# x축과 y축 레이블 색상 설정
for i, (label, color) in enumerate(zip(sel_region, color_region)):
        axC1.get_xticklabels()[i].set_color(color)
        axC1.get_yticklabels()[i].set_color(color)
    

im2 = axC2.imshow(save_fc[np.ix_(sel_nn, sel_nn)], cmap='jet', vmin=-0.2, vmax=1, aspect='equal')
axC2.set_xticks(range(len(sel_region)))
axC2.set_xticklabels(sel_region, rotation=90, fontsize=6)
axC2.set_yticks(range(len(sel_region)))
axC2.set_yticklabels(sel_region, fontsize=6)

#cbar2 = plt.colorbar(im2, ax=axC2, fraction=0.046, pad=0.04)

# x축과 y축 레이블 색상 설정
for i, (label, color) in enumerate(zip(sel_region, color_region)):
        axC2.get_xticklabels()[i].set_color(color)
        axC2.get_yticklabels()[i].set_color(color)



#cbar = fig.colorbar(im1, ax=[axC1,axC2], fraction=0.5, pad=0.01)

# im1 = axC1.imshow(empirical_fc, vmin=-1, vmax=1, cmap='jet')
# im2 = axC2.imshow(simulated_fc, vmin=-1, vmax=1, cmap='jet')

# =========================
# D) FC variation
# =========================


axD1.set_title("Emp. FCV")
axD2.set_title("Sim. FCV")

im3 = axD1.imshow(estd_fc[np.ix_(sel_nn, sel_nn)], cmap='jet', aspect='equal')
axD1.set_xticks(range(len(sel_region)))
axD1.set_xticklabels(sel_region, rotation=90, fontsize=6)
axD1.set_yticks(range(len(sel_region)))
axD1.set_yticklabels(sel_region, fontsize=6)

#cbar3= plt.colorbar(im3, ax=axD1, fraction=0.046, pad=0.04)
# x축과 y축 레이블 색상 설정
for i, (label, color) in enumerate(zip(sel_region, color_region)):
        axD1.get_xticklabels()[i].set_color(color)
        axD1.get_yticklabels()[i].set_color(color)
    
 
#im4=axD2.imshow(sstd_fc[np.ix_(sel_nn,sel_nn)],cmap='jet',vmin=0.2, aspect='equal');
im4 = axD2.imshow(sstd_fc[np.ix_(sel_nn, sel_nn)], cmap='jet', vmin=0.05, vmax=0.30,aspect='equal')
axD2.set_xticks(range(len(sel_region)))
axD2.set_xticklabels(sel_region, rotation=90, fontsize=6)
axD2.set_yticks(range(len(sel_region)))
axD2.set_yticklabels(sel_region, fontsize=6)

#cbar4= plt.colorbar(im4, ax=axD2, fraction=0.046, pad=0.04)

# x축과 y축 레이블 색상 설정
for i, (label, color) in enumerate(zip(sel_region, color_region)):
        axD2.get_xticklabels()[i].set_color(color)
        axD2.get_yticklabels()[i].set_color(color)
        
        








# =========================
# E) FC mean scatter
# =========================




fig1_data = np.load(DATA_DIR / 'figure1_emp_variation' / 'figure1_whole_brain_fc_mean_std_data.npz')

new_region_name = fig1_data['new_region_name']
mean_fc = fig1_data['mean_fc']
sem_fc = fig1_data['sem_fc']
mean_var = np.squeeze(fig1_data['mean_var'])
sem_var = fig1_data['sem_var']
new_brain_division_list = fig1_data['new_brain_division_list']
#colors = fig1_data['colors']
new_region_array = fig1_data['new_region_array']



save_path = set_path / 'figure5_compare_xy_scatter_data.pkl'
fig5 = load_pickle_file(save_path)

emp_ave_fc_list = fig5['emp_ave_fc_list']
emp_std_fc_list = fig5['emp_std_fc_list']

sim_ave_fc_list = fig5['sim_ave_fc_list']
sim_std_fc_list = fig5['sim_std_fc_list']
sel_region = fig5['sel_region']
sel_n = fig5['sel_n']
sel_n = list(set(sel_n))

emp_mean_ave_fc_list = []
emp_mean_std_fc_list = []
sim_mean_ave_fc_list = []
sim_mean_std_fc_list = []

emp_err_ave_fc_list = []
emp_err_std_fc_list = []
sim_err_ave_fc_list = []
sim_err_std_fc_list = []

nn = np.array([0, 1, 2, 4, 5, 6])
t_sel_n = []
for i in range(len(new_region_array)):
    ii = new_region_array[i]
    print(i, len(emp_ave_fc_list[ii]))
    e_ave_x = np.array(emp_ave_fc_list[ii])
    e_std_x = np.array(emp_std_fc_list[ii])

    s_ave_x = np.array(sim_ave_fc_list[ii])
    s_std_x=np.array(sim_std_fc_list[ii]);
    
    
    t_sel_n.append(ii);
    emp_mean_ave_fc_list.append(np.mean(e_ave_x));
    emp_mean_std_fc_list.append(np.mean(e_std_x));
    sim_mean_ave_fc_list.append(np.mean(s_ave_x));
    sim_mean_std_fc_list.append(np.mean(s_std_x)); 
    
    
    
    emp_err_ave_fc_list.append(np.std(e_ave_x,ddof=1)/np.sqrt(len(e_ave_x)));
    emp_err_std_fc_list.append(np.std(e_std_x,ddof=1)/np.sqrt(len(e_std_x)));
    sim_err_ave_fc_list.append(np.std(s_ave_x,ddof=1)/np.sqrt(len(s_ave_x)));
    sim_err_std_fc_list.append(np.std(s_std_x,ddof=1)/np.sqrt(len(s_std_x)));


    print(fig_help.region[ii],np.mean(e_std_x),np.mean(s_std_x));
    print(fig_help.region[ii],np.std(e_std_x),np.std(s_std_x));
        
sel_n=t_sel_n              
from matplotlib.colors import to_rgba

def lighten_color(color, amount=0.5):
    """amount < 1 → 밝게, amount > 1 → 진하게"""
    r, g, b, a = to_rgba(color)
    r = 1 - (1 - r) * amount
    g = 1 - (1 - g) * amount
    b = 1 - (1 - b) * amount
    return (r, g, b, a)

valid_divisions = [2, 1, 3, 0]








err_mean_color = '#8C8C8C'   # light gray
err_var_color = '#1A1A1A'   # near black
#err_mean_color = 'b'        # FC Mean errorbar
#err_var_color  = 'r'  # FC Var errorb
ordered_region_names = []
ordered_divisions = []
k=0;
for d in valid_divisions:
    for i in range(len(new_region_array)):
        if fig_help.brain_division_list[sel_n[i]] == d:
            #axB.errorbar(
            #    k, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
            #    fmt='o', color=colors[i], alpha=0.6
            #)
            #axB.errorbar(
            #    k, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
            #    fmt='D', color=colors[i], alpha=1.0
            #)
            
            base_color = fig_help.division_colors[d]

            color_mean = lighten_color(base_color, amount=0.6)  # FC Mean (연함)
            color_var  = lighten_color(base_color, amount=1.0)

            error_kw=dict(
                ecolor=err_mean_color,
                elinewidth=2.8,
                capthick=2.8
            )
            
            axG.bar(
                k-0.12, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
                 color=color_mean, alpha=0.6,width=0.4, error_kw=error_kw
            )
            
            error_kw=dict(
                ecolor=err_var_color,
                elinewidth=2.2,
                capthick=2.2
            )
                
            axG.bar(
                k+0.12,  sim_mean_ave_fc_list[i], yerr= sim_err_ave_fc_list[i],label='FC Var.' if i == 0 else "",
                 color=color_var, alpha=0.6,width=0.4, error_kw=error_kw

            )
            
            xv=emp_mean_ave_fc_list[i];
            yv=sim_mean_ave_fc_list[i];
        
            ordered_region_names.append(fig_help.region[sel_n[i]])
            ordered_divisions.append(d)
          
            k += 1
axG.set_xticks(range(len(ordered_region_names)), ordered_region_names, rotation=45,fontsize=10)

for lbl, d in zip(axG.get_xticklabels(), ordered_divisions):
    lbl.set_color(fig_help.division_colors[d])
#    if d == 2:
#        lbl.set_fontweight('bold')
#        lbl.set_fontsize(12)
#        lbl.set_fontstyle('italic')
        
prev_d = ordered_divisions[0]
for i, d in enumerate(ordered_divisions):
    if d != prev_d:
        axG.axvline(i - 0.5, color='gray', lw=2, zorder=0)
    prev_d = d
    
axG.set_ylabel('FCS (z-score)')
axG.set_ylim((-2.5,1.5));
axG.set_xlim((-1,k+0.1));

k=0;
for d in valid_divisions:
    for i in range(len(sel_n)):
        if fig_help.brain_division_list[sel_n[i]] == d:
            #axB.errorbar(
            #    k, mean_fc[i], yerr=sem_fc[i],label='FC Mean' if i == 0 else "",
            #    fmt='o', color=colors[i], alpha=0.6
            #)
            #axB.errorbar(
            #    k, mean_var[i], yerr=sem_var[i],label='FC Var.' if i == 0 else "",
            #    fmt='D', color=colors[i], alpha=1.0
            #)
            
            base_color = fig_help.division_colors[d]

            color_mean = lighten_color(base_color, amount=0.6)  # FC Mean (연함)
            color_var  = lighten_color(base_color, amount=1.0)

            error_kw=dict(
                ecolor=err_mean_color,
                elinewidth=2.8,
                capthick=2.8
            )
            
            axH.bar(
                k-0.12, mean_var[i], yerr=sem_var[i],label='FC Mean' if i == 0 else "",
                 color=color_mean, alpha=0.6,width=0.4, error_kw=error_kw
            )
            
            error_kw=dict(
                ecolor=err_var_color,
                elinewidth=2.2,
                capthick=2.2
            )
                
            axH.bar(
                k+0.12,  sim_mean_std_fc_list[i], yerr= sim_err_std_fc_list[i],label='FC Var.' if i == 0 else "",
                 color=color_var, alpha=0.6,width=0.4, error_kw=error_kw

            )

            xv=emp_mean_std_fc_list[i];
            yv=sim_mean_std_fc_list[i];
        
           
        
            #ordered_region_names.append(new_region_name[i])
            #ordered_divisions.append(d)
            #ordered_mean_fc.append(mean_fc[i])
            #ordered_mean_var.append(mean_var[i])
            k += 1


            
axH.set_xticks(range(len(ordered_region_names)), ordered_region_names, rotation=45,fontsize=10)
for lbl, d in zip(axH.get_xticklabels(), ordered_divisions):
    lbl.set_color(fig_help.division_colors[d])
    #if d == 2:
    #    lbl.set_fontweight('bold')
    #    lbl.set_fontsize(12)
    #    lbl.set_fontstyle('italic')
        
prev_d = ordered_divisions[0]
for i, d in enumerate(ordered_divisions):
    if d != prev_d:
        axH.axvline(i - 0.5, color='gray', lw=2, zorder=0)
    prev_d = d

axH.set_xlim((-1,k+0.1));
from matplotlib.lines import Line2D

error_legend = [
    Line2D(
        [0], [0],
        color=err_mean_color,
        lw=2.8,
        label='Emp.FCS'
    ),
    Line2D(
        [0], [0],
        color=err_var_color,
        lw=2.2,
        label='Sim.FCS'
    )
]

leg2 = axG.legend(
    handles=error_legend,
    loc='lower right',
    frameon=False,
    fontsize=11
)

error_legend_fcv = [
    Line2D(
        [0], [0],
        color=err_mean_color,
        lw=2.8,
        label='Emp.FCV'
    ),
    Line2D(
        [0], [0],
        color=err_var_color,
        lw=2.2,
        label='Sim.FCV'
    )
]

leg2 = axH.legend(
    handles=error_legend_fcv,
    loc='upper right',
    frameon=False,
    fontsize=11
)


axH.set_ylabel('FCV (z-score)')
axH.set_ylim((-2.3,2.1));
axH.set_yticks([-2,-1,0,1,2.0]);        
          


# axE.scatter(emp_fc_mean, sim_fc_mean, s=15)
# axE.plot(x, y_fit, 'r--')

# =========================
# F) FC variation scatter
# =========================

#axE.set_title("FCS")


node_color=[];

for i in sel_n:
    
    node_color.append(fig_help.division_colors[fig_help.brain_division_list[i]]);
    
x_data = emp_mean_ave_fc_list
y_data = sim_mean_ave_fc_list

x= np.array(x_data);
yy = np.array(y_data);
r, p = pearsonr(x, yy)
m, b = np.polyfit(x, yy, 1)
xs = np.linspace(x.min(), x.max(), 100)
supply0_stats = [
    {
        "figure": "figure_supply_0",
        "panel": "G",
        "metric": "FCS",
        "test": "Pearson correlation",
        "n_regions": int(np.isfinite(x).sum()),
        "pearson_r": float(r),
        "p_value": float(p),
        "linear_slope": float(m),
        "linear_intercept": float(b),
        "x_label": "Emp. FCS (z-score)",
        "y_label": "Sim. FCS (z-score)",
    }
]

import seaborn as sns
sns.regplot(x=x, y=yy, ax=axE,
            ci=95, # 95% 신뢰 구간 (그림의 회색 영역)
            scatter_kws={'color': 'salmon', 'edgecolor': 'black', 'alpha': 0.4}, # 점 스타일
            line_kws={'color': 'black', 'lw': 2}) # 선 스타일


axE.scatter(x, yy, color=node_color, s=90, edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
axE.plot(xs, m*xs + b, color='k', lw=2)
axE.set_xlabel('Emp. FCS (z-score)')
axE.set_ylabel('Sim. FCS (z-score)')
axE.yaxis.set_label_coords(-0.18, 0.5)
    #plt.title('Pre-DCA vs FC var')

axE.text(-1.8, 0.8, f"r = {r:.3f}\np = {p:.3g}", fontsize=10, fontstyle='italic')

#axE.text(0.05, 0.95, f"r={r:.3f}\np={p:.3g}", transform=plt.gca().transAxes,
#             ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

axE.axis('scaled')
axE.set_xlim((-2.1, 2.0));
axE.set_ylim((-2.2, 2.0));

'''
for i in range(len(sel_n)):
        col = 'black';
        axE.annotate(
            region[sel_n[i]],
            (x_data[i], y_data[i]),
            xycoords='data',
            xytext=(x_data[i] + 0.001, y_data[i] + 0.001),
            textcoords='data',
            fontsize=6,
            color=col
        )

'''      

#axF.set_title("FCV")


x_data = emp_mean_std_fc_list
y_data = sim_mean_std_fc_list

x= np.array(x_data);
yy = np.array(y_data);
r, p = pearsonr(x, yy)
m, b = np.polyfit(x, yy, 1)
xs = np.linspace(x.min(), x.max(), 100)
supply0_stats.append(
    {
        "figure": "figure_supply_0",
        "panel": "H",
        "metric": "FCV",
        "test": "Pearson correlation",
        "n_regions": int(np.isfinite(x).sum()),
        "pearson_r": float(r),
        "p_value": float(p),
        "linear_slope": float(m),
        "linear_intercept": float(b),
        "x_label": "Emp. FCV (z-score)",
        "y_label": "Sim. FCV (z-score)",
    }
)

import seaborn as sns
sns.regplot(x=x, y=yy, ax=axF,
            ci=95, # 95% 신뢰 구간 (그림의 회색 영역)
            scatter_kws={'color': 'salmon', 'edgecolor': 'black', 'alpha': 0.4}, # 점 스타일
            line_kws={'color': 'black', 'lw': 2}) # 선 스타일

axF.scatter(x, yy, color=node_color, s=90, edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
axF.plot(xs, m*xs + b, color='k', lw=2)
axF.axis('scaled')
axF.set_xlim((-2.0, 1.7));
axF.set_ylim((-2.0, 1.7));

axF.set_xlabel('Emp. FCV (z-score)')
axF.set_ylabel('Sim. FCV (z-score)')
axF.yaxis.set_label_coords(-0.18, 0.5)
    #plt.title('Pre-DCA vs FC var')
axF.text(-1.8, 0.95, f"r = {r:.3f}\np = {p:.3g}", fontsize=10, fontstyle='italic')
#axF.text(0.05, 0.95, f"r={r:.3f}\np={p:.3g}", transform=plt.gca().transAxes,
#             ha='left', va='top', bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'))

axE.set_xticks([-2,-1,0,1])
axE.set_yticks([-2,-1,0,1])

axF.set_xticks([-1.5,0,1.5])
axF.set_yticks([-1.5,0,1.5])



#axE.xaxis.set_label_coords(0.5, -0.12)
#axF.xaxis.set_label_coords(0.5, -0.12)


cbar1 = plt.colorbar(im1, ax=axC1, fraction=0.046, pad=0.04)
cbar2 = plt.colorbar(im2, ax=axC2, fraction=0.046, pad=0.04)
cbar3 = plt.colorbar(im3, ax=axD1, fraction=0.046, pad=0.04)
cbar4 = plt.colorbar(im4, ax=axD2, fraction=0.046, pad=0.04)
for cb in [cbar1, cbar2, cbar3, cbar4]:
    cb.ax.tick_params(labelsize=8)

# ── Layout adjustments ──

# A: Ca traces — shift down, shrink width
pos = axA.get_position()
axA.set_position([pos.x0, pos.y0 - pos.height * 0.4, pos.width * 0.90, pos.height])

# B: FC timeseries — vertically center-align with A
pos_a = axA.get_position()
pos_b = axB.get_position()
b_height = pos_a.height * 0.7
b_y0 = pos_a.y0 + (pos_a.height - b_height) / 2  # center vertically
axB.set_position([pos_b.x0 + 0.02, b_y0 + 0.02, pos_b.width * 0.85, b_height])

# C/D: FC matrices — shrink and space evenly
for tax in [axC1, axC2, axD1, axD2]:
    pos = tax.get_position()
    x_trade_off = pos.width * 0.2
    tax.set_position([pos.x0 - 0.015, pos.y0 - 0.02, pos.width * 0.8, pos.height * 0.8])

pos = axC2.get_position()
axC2.set_position([pos.x0 + x_trade_off / 2 - 0.013, pos.y0, pos.width, pos.height])

pos = axD1.get_position()
axD1.set_position([pos.x0 + x_trade_off / 2 + 0.013, pos.y0, pos.width, pos.height])

pos = axD2.get_position()
axD2.set_position([pos.x0 + x_trade_off, pos.y0, pos.width, pos.height])

# Adjust colorbar positions to match the adjusted axes
for cbar, ax in [(cbar1, axC1), (cbar2, axC2), (cbar3, axD1), (cbar4, axD2)]:
    pos = ax.get_position()
    cbar.ax.set_position([pos.x0 + pos.width + 0.005, pos.y0, 0.01, pos.height])

# E/F: bar charts — shrink height
for tax in [axG, axH]:
    pos = tax.get_position()
    tax.set_position([pos.x0, pos.y0, pos.width * 0.95, pos.height * 0.8])

# G/H: scatter panels — shift left, enlarge
for tax in [axE, axF]:
    pos = tax.get_position()
    tax.set_position([pos.x0 + pos.width * 0.1, pos.y0, pos.width * 0.9, pos.height * 0.8])

# ── Panel labels (B label before shifting axB down) ──
axA.axis('off')

# Align A, C, E, F labels on the same x position
label_x = axA.get_position().x0 - 0.06

def _add_label_at_x(fig, ax, label, x, dy=0.01):
    bbox = ax.get_position()
    fig.text(x, bbox.y1 + dy, label, fontsize=22, fontweight='bold')

_add_label_at_x(fig, axA, 'A', label_x, dy=-0.0075)
fig_help.add_panel_label_fig(fig, axB, 'B', dx=-0.06, dy=-0.0075)

# Shift axB plot area down (label stays in place)
pos_b = axB.get_position()
axB.set_position([pos_b.x0, pos_b.y0 - 0.015, pos_b.width, pos_b.height])

_add_label_at_x(fig, axC1, 'C', label_x)
fig_help.add_panel_label_fig(fig, axD1, 'D', dx=-0.03)
_add_label_at_x(fig, axG, 'E', label_x)
_add_label_at_x(fig, axH, 'F', label_x)
fig_help.add_panel_label_fig(fig, axE, 'G', dx=-0.06)
fig_help.add_panel_label_fig(fig, axF, 'H', dx=-0.06)




        
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(supply0_stats).to_csv(OUT_STATS, index=False)
plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
