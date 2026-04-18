"""
final_results.py — Full thesis plots for precision-switching scheduler comparison
                   Naive (FP32) vs Rule-Based vs RL — Jetson Orin NX 8 GB

Plots produced in results_plots/:

  Time-series:
    01_gpu_temp.png               GPU temperature over time
    02_cpu_temp.png               CPU temperature over time
    03_power.png                  Power consumption over time
    04_gpu_util.png               GPU utilization over time
    05_yolo_latency.png           YOLO latency over time
    06_bert_latency.png           BERT latency over time
    07_total_latency.png          Combined latency over time
    08_fps.png                    FPS over time
    09_throughput.png             Throughput (tasks/sec) over time
    10_energy.png                 Energy per inference over time
    11_cost_function.png          Thesis cost function C(m,t) over time
    12_ram.png                    RAM usage over time

  Statistical distributions:
    13_mean_bar_chart.png         Grouped bar chart mean ± std
    14_boxplot_latency.png        Box plots — latency
    15_boxplot_power_temp.png     Box plots — power and temperature
    16_violin_latency.png         Violin plots — latency
    17_cdf_latency.png            CDF — total latency
    18_cdf_power.png              CDF — power consumption
    19_cdf_energy.png             CDF — energy per inference

  Precision / decision analysis:
    20_precision_heatmap.png      Precision usage heatmap
    21_precision_pie_charts.png   Precision distribution pie charts
    22_precision_over_time.png    Precision decisions step plot
    23_system_state_over_time.png System state (Normal/High/Critical)

  Scatter / correlation:
    24_scatter_temp_vs_power.png
    25_scatter_gpu_vs_latency.png
    26_scatter_temp_vs_latency.png
    27_scatter_energy_vs_latency.png

  Thesis-specific:
    28_summary_4panel.png         4-panel thesis figure
    29_energy_efficiency_bar.png  Energy per inference bar chart
    30_latency_vs_accuracy.png    Latency vs accuracy trade-off
    31_scheduler_stability.png    Rolling std of latency
    32_scheduler_overhead.png     Scheduler overhead comparison
    33_model_size_comparison.png  Model size vs accuracy bar chart
    34_task_success_rate.png      Task success rate vs GPU load
    35_fps_vs_power.png           FPS vs power scatter
    36_throughput_bar.png         Mean throughput bar chart

  Data:
    summary_stats.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
LOG_FILES = {
    "Naive (FP32)": "naive_log.csv",
    "Rule-Based":   "rule_based_log.csv",
    "RL":           "rl_log.csv",
}
COLORS  = {"Naive (FP32)": "#e74c3c", "Rule-Based": "#3498db", "RL": "#2ecc71"}
MARKERS = {"Naive (FP32)": "o",       "Rule-Based": "s",       "RL": "^"}
OUTPUT_DIR = "results_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DPI = 150

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.labelsize": 11, "legend.fontsize": 10})

def out(f): return os.path.join(OUTPUT_DIR, f)
def rolling(s, w=5): return s.rolling(w, min_periods=1).mean()


# ─────────────────────────────────────────────
# Load & align
# ─────────────────────────────────────────────
dfs = {}
for label, path in LOG_FILES.items():
    if not os.path.exists(path):
        print(f"[WARN] {path} not found — skipping {label}")
        continue
    df = pd.read_csv(path)
    num_cols = ["gpu_temp","cpu_temp","gpu_util","power","ram",
                "yolo_latency_ms","bert_latency_ms","total_latency_ms",
                "fps","throughput_tasks_per_sec","energy_per_inf_j",
                "scheduler_overhead_ms","cost_function"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    dfs[label] = df

if len(dfs) < 2:
    raise RuntimeError("Need at least 2 scheduler logs.")

min_len    = min(len(df) for df in dfs.values())
dfs        = {k: df.iloc[:min_len].reset_index(drop=True) for k,df in dfs.items()}
iterations = np.arange(1, min_len + 1)
labels     = list(dfs.keys())
print(f"Aligned to {min_len} iterations, {len(dfs)} schedulers.\n")


# ─────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────
METRICS = {
    "gpu_temp":              "GPU Temp (°C)",
    "cpu_temp":              "CPU Temp (°C)",
    "gpu_util":              "GPU Util (%)",
    "power":                 "Power (W)",
    "ram":                   "RAM (MB)",
    "yolo_latency_ms":       "YOLO Latency (ms)",
    "bert_latency_ms":       "BERT Latency (ms)",
    "total_latency_ms":      "Total Latency (ms)",
    "fps":                   "FPS",
    "throughput_tasks_per_sec": "Throughput (tasks/s)",
    "energy_per_inf_j":      "Energy/Inf (J)",
    "scheduler_overhead_ms": "Sched Overhead (ms)",
    "cost_function":         "Cost Function",
}

rows = []
print(f"{'Metric':<26} {'Scheduler':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("="*76)
for col, lbl in METRICS.items():
    for label, df in dfs.items():
        if col not in df.columns: continue
        s = df[col].dropna()
        row = dict(metric=lbl, scheduler=label, mean=s.mean(),
                   std=s.std(), min=s.min(), max=s.max(), median=s.median())
        rows.append(row)
        print(f"{lbl:<26} {label:<16} {row['mean']:>8.3f} {row['std']:>8.3f} "
              f"{row['min']:>8.3f} {row['max']:>8.3f}")
    print("-"*76)

pd.DataFrame(rows).to_csv(out("summary_stats.csv"), index=False)
print(f"\nSummary saved.\n")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def style(ax, ylabel, title, hline=None):
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if hline:
        ax.axhline(hline[0], color="black", linestyle="--",
                   linewidth=1.2, label=hline[1])
        ax.legend()

def ts_plot(col, ylabel, title, fname, hline=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    for label, df in dfs.items():
        if col not in df.columns: continue
        raw = df[col].values
        mu  = rolling(df[col]).values
        std = rolling(df[col].rolling(5,min_periods=1).std().fillna(0)).values
        ax.plot(iterations, raw, color=COLORS[label], alpha=0.18, linewidth=0.8)
        ax.plot(iterations, mu,  color=COLORS[label], linewidth=2.0,
                label=label, marker=MARKERS[label], markevery=10, markersize=5)
        ax.fill_between(iterations, mu-std, mu+std,
                        color=COLORS[label], alpha=0.10)
    style(ax, ylabel, title, hline)
    fig.tight_layout()
    fig.savefig(out(fname), dpi=DPI)
    plt.close(fig)
    print(f"Saved: {fname}")


# ══════════════════════════════════════════════
# 01–12  Time-series
# ══════════════════════════════════════════════
ts_plot("gpu_temp",         "GPU Temp (°C)",    "GPU Temperature over Time",            "01_gpu_temp.png",    hline=(75,"Critical threshold"))
ts_plot("cpu_temp",         "CPU Temp (°C)",    "CPU Temperature over Time",            "02_cpu_temp.png",    hline=(75,"Critical threshold"))
ts_plot("power",            "Power (W)",         "Power Consumption over Time",          "03_power.png",       hline=(10,"Critical threshold"))
ts_plot("gpu_util",         "GPU Util (%)",      "GPU Utilization over Time",            "04_gpu_util.png",    hline=(85,"High load threshold"))
ts_plot("yolo_latency_ms",  "Latency (ms)",      "YOLO Inference Latency over Time",     "05_yolo_latency.png")
ts_plot("bert_latency_ms",  "Latency (ms)",      "BERT Inference Latency over Time",     "06_bert_latency.png")
ts_plot("total_latency_ms", "Total Latency (ms)","Combined Inference Latency over Time", "07_total_latency.png")
ts_plot("fps",              "FPS",               "Frames Per Second over Time",          "08_fps.png")
ts_plot("throughput_tasks_per_sec","Tasks/sec",  "Inference Throughput over Time",       "09_throughput.png")
ts_plot("energy_per_inf_j", "Energy (J)",        "Energy per Inference over Time",       "10_energy.png")
ts_plot("cost_function",    "Cost C(m,t)",       "Thesis Cost Function C(m,t) over Time\n"
                                                  "C = α(1−A) + βL + γE + δT",          "11_cost_function.png")
ts_plot("ram",              "RAM (MB)",           "RAM Usage over Time",                  "12_ram.png")


# ══════════════════════════════════════════════
# 13  Grouped bar chart
# ══════════════════════════════════════════════
bar_specs = [
    ("gpu_temp",         "GPU\nTemp(°C)"),
    ("power",            "Power\n(W)"),
    ("total_latency_ms", "Total\nLatency(ms)"),
    ("fps",              "FPS"),
    ("throughput_tasks_per_sec","Throughput\n(tasks/s)"),
    ("energy_per_inf_j", "Energy/Inf\n(J)"),
]
x = np.arange(len(bar_specs)); width = 0.25
fig, ax = plt.subplots(figsize=(14, 5))
for i, (label, df) in enumerate(dfs.items()):
    means = [df[c].mean() if c in df.columns else 0 for c,_ in bar_specs]
    stds  = [df[c].std()  if c in df.columns else 0 for c,_ in bar_specs]
    bars  = ax.bar(x+i*width, means, width, yerr=stds,
                   label=label, color=COLORS[label], alpha=0.85, capsize=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{m:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x+width); ax.set_xticklabels([bl for _,bl in bar_specs])
ax.set_ylabel("Mean Value"); ax.set_title("Mean Metrics per Scheduler (±1 std)", fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout(); fig.savefig(out("13_mean_bar_chart.png"), dpi=DPI); plt.close(fig)
print("Saved: 13_mean_bar_chart.png")


# ══════════════════════════════════════════════
# 14  Box plots — latency
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, title) in zip(axes, [
    ("yolo_latency_ms",  "YOLO Latency"),
    ("bert_latency_ms",  "BERT Latency"),
    ("total_latency_ms", "Total Latency"),
]):
    data = [df[col].dropna().values for df in dfs.values() if col in df.columns]
    bp   = ax.boxplot(data, patch_artist=True, notch=True,
                      medianprops=dict(color="black", linewidth=2))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[label]); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Latency (ms)"); ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout(); fig.savefig(out("14_boxplot_latency.png"), dpi=DPI); plt.close(fig)
print("Saved: 14_boxplot_latency.png")


# ══════════════════════════════════════════════
# 15  Box plots — power and temperature
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, ylabel, title) in zip(axes, [
    ("power",    "Power (W)",     "Power Distribution"),
    ("gpu_temp", "GPU Temp (°C)", "GPU Temp Distribution"),
    ("cpu_temp", "CPU Temp (°C)", "CPU Temp Distribution"),
]):
    data = [df[col].dropna().values for df in dfs.values() if col in df.columns]
    bp   = ax.boxplot(data, patch_artist=True, notch=True,
                      medianprops=dict(color="black", linewidth=2))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[label]); patch.set_alpha(0.7)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout(); fig.savefig(out("15_boxplot_power_temp.png"), dpi=DPI); plt.close(fig)
print("Saved: 15_boxplot_power_temp.png")


# ══════════════════════════════════════════════
# 16  Violin — latency
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title in zip(axes,
    ["yolo_latency_ms","bert_latency_ms","total_latency_ms"],
    ["YOLO Latency","BERT Latency","Total Latency"]
):
    data  = [df[col].dropna().values for df in dfs.values() if col in df.columns]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc, label in zip(parts["bodies"], labels):
        pc.set_facecolor(COLORS[label]); pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Latency (ms)"); ax.set_title(f"{title} (Violin)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout(); fig.savefig(out("16_violin_latency.png"), dpi=DPI); plt.close(fig)
print("Saved: 16_violin_latency.png")


# ══════════════════════════════════════════════
# 17–19  CDFs
# ══════════════════════════════════════════════
for col, xlabel, title, fname in [
    ("total_latency_ms", "Total Latency (ms)", "CDF — Total Latency",        "17_cdf_latency.png"),
    ("power",            "Power (W)",           "CDF — Power Consumption",     "18_cdf_power.png"),
    ("energy_per_inf_j", "Energy (J)",          "CDF — Energy per Inference",  "19_cdf_energy.png"),
]:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, df in dfs.items():
        if col not in df.columns: continue
        data = np.sort(df[col].dropna().values)
        cdf  = np.arange(1, len(data)+1) / len(data)
        ax.plot(data, cdf, color=COLORS[label], linewidth=2, label=label,
                marker=MARKERS[label], markevery=max(1,len(data)//15), markersize=5)
    ax.set_xlabel(xlabel); ax.set_ylabel("Cumulative Probability")
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--"); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(out(fname), dpi=DPI); plt.close(fig)
    print(f"Saved: {fname}")


# ══════════════════════════════════════════════
# 20  Precision heatmap
# ══════════════════════════════════════════════
COMBOS = ["NAIVE_FP32+FP32","YOLO_FP16+BERT_FP16",
          "YOLO_INT8+BERT_FP16","YOLO_FP16+BERT_INT8","YOLO_INT8+BERT_INT8"]

if all("decision" in df.columns for df in dfs.values()):
    heat = np.zeros((len(COMBOS), len(dfs)))
    for j, (label, df) in enumerate(dfs.items()):
        counts = df["decision"].value_counts()
        for i, combo in enumerate(COMBOS):
            heat[i,j] = counts.get(combo, 0)
    col_sums = heat.sum(axis=0, keepdims=True); col_sums[col_sums==0]=1
    heat_pct = heat / col_sums * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = LinearSegmentedColormap.from_list("g",["#ffffff","#2ecc71","#16a085"])
    im   = ax.imshow(heat_pct, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(dfs))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(COMBOS))); ax.set_yticklabels(COMBOS)
    ax.set_title("Precision Combination Usage (%)", fontweight="bold")
    for i in range(len(COMBOS)):
        for j in range(len(dfs)):
            ax.text(j, i, f"{heat_pct[i,j]:.1f}%", ha="center", va="center",
                    fontsize=10, color="black" if heat_pct[i,j]<60 else "white")
    plt.colorbar(im, ax=ax, label="% of iterations")
    fig.tight_layout(); fig.savefig(out("20_precision_heatmap.png"), dpi=DPI); plt.close(fig)
    print("Saved: 20_precision_heatmap.png")


# ══════════════════════════════════════════════
# 21  Precision pie charts
# ══════════════════════════════════════════════
if all("decision" in df.columns for df in dfs.values()):
    fig, axes = plt.subplots(1, len(dfs), figsize=(5*len(dfs), 5))
    if len(dfs)==1: axes=[axes]
    pie_colors = ["#e74c3c","#2ecc71","#3498db","#e67e22","#9b59b6"]
    for ax, (label, df) in zip(axes, dfs.items()):
        counts = df["decision"].value_counts()
        sizes  = [counts.get(c,0) for c in COMBOS]
        nonz   = [(s,c,pc) for s,c,pc in zip(sizes,COMBOS,pie_colors) if s>0]
        if nonz:
            s,c,pc = zip(*nonz)
            ax.pie(s,labels=c,colors=pc,autopct="%1.1f%%",
                   startangle=140,textprops={"fontsize":8})
        ax.set_title(f"{label}", fontweight="bold")
    fig.suptitle("Precision Distribution per Scheduler", fontsize=13)
    fig.tight_layout(); fig.savefig(out("21_precision_pie_charts.png"), dpi=DPI); plt.close(fig)
    print("Saved: 21_precision_pie_charts.png")


# ══════════════════════════════════════════════
# 22  Precision over time (step plot)
# ══════════════════════════════════════════════
if all("decision" in df.columns for df in dfs.values()):
    combo_idx = {c:i for i,c in enumerate(COMBOS)}
    fig, axes = plt.subplots(len(dfs),1,figsize=(13,3*len(dfs)),sharex=True)
    if len(dfs)==1: axes=[axes]
    for ax,(label,df) in zip(axes,dfs.items()):
        y = [combo_idx.get(d,-1) for d in df["decision"]]
        ax.step(iterations,y,color=COLORS[label],linewidth=1.5,where="post")
        ax.set_yticks(range(len(COMBOS))); ax.set_yticklabels(COMBOS,fontsize=8)
        ax.set_title(f"{label} — Precision Decisions over Time", fontweight="bold")
        ax.grid(True,alpha=0.2,linestyle="--")
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout(); fig.savefig(out("22_precision_over_time.png"), dpi=DPI); plt.close(fig)
    print("Saved: 22_precision_over_time.png")


# ══════════════════════════════════════════════
# 23  System state over time
# ══════════════════════════════════════════════
if all("system_state" in df.columns for df in dfs.values()):
    state_map = {"Normal":0,"High Load":1,"Critical":2,"Unknown":-1}
    state_colors = {0:"#2ecc71",1:"#f39c12",2:"#e74c3c",-1:"#bdc3c7"}
    fig, axes = plt.subplots(len(dfs),1,figsize=(13,2.5*len(dfs)),sharex=True)
    if len(dfs)==1: axes=[axes]
    for ax,(label,df) in zip(axes,dfs.items()):
        y = [state_map.get(s,-1) for s in df["system_state"]]
        colors_bar = [state_colors[v] for v in y]
        ax.bar(iterations,np.ones(len(y)),color=colors_bar,width=1.0,alpha=0.8)
        ax.set_yticks([0,1,2]); ax.set_yticklabels(["Normal","High Load","Critical"])
        ax.set_title(f"{label} — System State Classification", fontweight="bold")
    axes[-1].set_xlabel("Iteration")
    patches = [mpatches.Patch(color=state_colors[k],label=v)
               for v,k in state_map.items() if k>=0]
    fig.legend(handles=patches,loc="upper right")
    fig.tight_layout(); fig.savefig(out("23_system_state_over_time.png"), dpi=DPI); plt.close(fig)
    print("Saved: 23_system_state_over_time.png")


# ══════════════════════════════════════════════
# 24–27  Scatter plots
# ══════════════════════════════════════════════
scatter_specs = [
    ("gpu_temp","power",           "GPU Temp (°C)","Power (W)",
     "GPU Temperature vs Power",           "24_scatter_temp_vs_power.png"),
    ("gpu_util","total_latency_ms","GPU Util (%)","Total Latency (ms)",
     "GPU Utilization vs Total Latency",   "25_scatter_gpu_vs_latency.png"),
    ("gpu_temp","total_latency_ms","GPU Temp (°C)","Total Latency (ms)",
     "GPU Temperature vs Total Latency",   "26_scatter_temp_vs_latency.png"),
    ("energy_per_inf_j","total_latency_ms","Energy/Inf (J)","Total Latency (ms)",
     "Energy per Inference vs Latency",    "27_scatter_energy_vs_latency.png"),
]
for xc,yc,xl,yl,title,fname in scatter_specs:
    fig,ax = plt.subplots(figsize=(8,6))
    for label,df in dfs.items():
        if xc not in df.columns or yc not in df.columns: continue
        ax.scatter(df[xc],df[yc],color=COLORS[label],alpha=0.45,
                   s=30,label=label,marker=MARKERS[label])
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(title, fontweight="bold"); ax.legend()
    ax.grid(True,alpha=0.3,linestyle="--")
    fig.tight_layout(); fig.savefig(out(fname),dpi=DPI); plt.close(fig)
    print(f"Saved: {fname}")


# ══════════════════════════════════════════════
# 28  4-panel thesis summary
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(15,10))
gs  = GridSpec(2,2,figure=fig,hspace=0.35,wspace=0.3)
for spec,(col,ylabel,title) in zip(
    [gs[0,0],gs[0,1],gs[1,0],gs[1,1]],
    [("gpu_temp","GPU Temp (°C)","GPU Temperature"),
     ("power","Power (W)","Power Consumption"),
     ("total_latency_ms","Total Latency (ms)","Total Inference Latency"),
     ("energy_per_inf_j","Energy (J)","Energy per Inference")]):
    ax = fig.add_subplot(spec)
    for label,df in dfs.items():
        if col not in df.columns: continue
        mu = rolling(df[col]).values
        ax.plot(iterations,mu,color=COLORS[label],linewidth=2,label=label,
                marker=MARKERS[label],markevery=10,markersize=4)
    style(ax,ylabel,title)
handles = [mpatches.Patch(color=COLORS[l],label=l) for l in labels]
fig.legend(handles=handles,loc="upper center",ncol=len(labels),
           bbox_to_anchor=(0.5,1.01),fontsize=11)
fig.suptitle("Precision-Switching Scheduler Comparison — Jetson Orin NX 8 GB",
             fontsize=13,fontweight="bold",y=1.04)
fig.savefig(out("28_summary_4panel.png"),dpi=DPI,bbox_inches="tight"); plt.close(fig)
print("Saved: 28_summary_4panel.png")


# ══════════════════════════════════════════════
# 29  Energy efficiency bar
# ══════════════════════════════════════════════
fig,ax = plt.subplots(figsize=(8,5))
elabels,evals,ecolors = [],[],[]
for label,df in dfs.items():
    if "energy_per_inf_j" not in df.columns: continue
    e = df["energy_per_inf_j"].mean()
    elabels.append(label); evals.append(e); ecolors.append(COLORS[label])
bars = ax.bar(elabels,evals,color=ecolors,alpha=0.85,width=0.4)
for bar,val in zip(bars,evals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0001,
            f"{val:.4f} J", ha="center",va="bottom",fontsize=10)
ax.set_ylabel("Mean Energy per Inference (Joules)")
ax.set_title("Energy Efficiency Comparison\n(Lower is Better)", fontweight="bold")
ax.grid(axis="y",alpha=0.3,linestyle="--")
fig.tight_layout(); fig.savefig(out("29_energy_efficiency_bar.png"),dpi=DPI); plt.close(fig)
print("Saved: 29_energy_efficiency_bar.png")


# ══════════════════════════════════════════════
# 30  Latency vs accuracy trade-off
#     Update ACCURACY_MAP with real mAP values after COCO evaluation
# ══════════════════════════════════════════════
ACCURACY_MAP = {
    "Naive (FP32)": 1.00,
    "Rule-Based":   0.91,
    "RL":           0.93,
}
fig,ax = plt.subplots(figsize=(8,6))
for label,df in dfs.items():
    if "total_latency_ms" not in df.columns: continue
    lat = df["total_latency_ms"].mean()
    acc = ACCURACY_MAP.get(label,0.90)
    ax.scatter(lat,acc,color=COLORS[label],s=250,
               marker=MARKERS[label],zorder=5,label=label)
    ax.annotate(f"  {label}",(lat,acc),fontsize=10)
ax.set_xlabel("Mean Total Latency (ms)"); ax.set_ylabel("Relative Accuracy")
ax.set_title("Latency vs Accuracy Trade-off\n(Update accuracy values after COCO eval)",
             fontweight="bold")
ax.set_ylim(0.75,1.05); ax.legend()
ax.grid(True,alpha=0.3,linestyle="--")
fig.tight_layout(); fig.savefig(out("30_latency_vs_accuracy.png"),dpi=DPI); plt.close(fig)
print("Saved: 30_latency_vs_accuracy.png")


# ══════════════════════════════════════════════
# 31  Scheduler stability — rolling std latency
# ══════════════════════════════════════════════
fig,ax = plt.subplots(figsize=(11,4))
for label,df in dfs.items():
    if "total_latency_ms" not in df.columns: continue
    std_r = df["total_latency_ms"].rolling(10,min_periods=1).std().fillna(0)
    ax.plot(iterations,std_r.values,color=COLORS[label],linewidth=2,
            label=label,marker=MARKERS[label],markevery=10,markersize=4)
ax.set_xlabel("Iteration"); ax.set_ylabel("Rolling Std (ms)")
ax.set_title("Scheduler Stability — Rolling Std of Total Latency (window=10)\n"
             "Lower = more consistent inference times", fontweight="bold")
ax.legend(); ax.grid(True,alpha=0.3,linestyle="--")
fig.tight_layout(); fig.savefig(out("31_scheduler_stability.png"),dpi=DPI); plt.close(fig)
print("Saved: 31_scheduler_stability.png")


# ══════════════════════════════════════════════
# 32  Scheduler overhead comparison
# ══════════════════════════════════════════════
if any("scheduler_overhead_ms" in df.columns for df in dfs.values()):
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    # Bar — mean overhead
    oh_labels,oh_means,oh_stds,oh_colors = [],[],[],[]
    for label,df in dfs.items():
        if "scheduler_overhead_ms" not in df.columns: continue
        oh_labels.append(label)
        oh_means.append(df["scheduler_overhead_ms"].mean())
        oh_stds.append(df["scheduler_overhead_ms"].std())
        oh_colors.append(COLORS[label])
    axes[0].bar(oh_labels,oh_means,yerr=oh_stds,color=oh_colors,
                alpha=0.85,capsize=5,width=0.4)
    axes[0].set_ylabel("Overhead (ms)")
    axes[0].set_title("Mean Scheduler Overhead (ms)\n(Lower = less computation per decision)",
                      fontweight="bold")
    axes[0].grid(axis="y",alpha=0.3,linestyle="--")
    # Time-series
    for label,df in dfs.items():
        if "scheduler_overhead_ms" not in df.columns: continue
        mu = rolling(df["scheduler_overhead_ms"]).values
        axes[1].plot(iterations,mu,color=COLORS[label],linewidth=2,
                     label=label,marker=MARKERS[label],markevery=10,markersize=4)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Overhead (ms)")
    axes[1].set_title("Scheduler Overhead over Time", fontweight="bold")
    axes[1].legend(); axes[1].grid(True,alpha=0.3,linestyle="--")
    fig.tight_layout(); fig.savefig(out("32_scheduler_overhead.png"),dpi=DPI); plt.close(fig)
    print("Saved: 32_scheduler_overhead.png")


# ══════════════════════════════════════════════
# 33  Model size vs accuracy (from thesis table)
# ══════════════════════════════════════════════
model_data = {
    "Model":    ["BERT\n(Original)","DistilBERT\n(FP32)","DistilBERT\n(FP16 TRT)","DistilBERT\n(INT8 TRT)"],
    "Size_MB":  [420,               240,                  120,                      60],
    "Accuracy": [91.0,              90.2,                 89.8,                     88.5],
}
# Auto-detect actual engine sizes
import glob
engine_sizes = {}
for f in glob.glob("engines/*.engine"):
    engine_sizes[os.path.basename(f)] = os.path.getsize(f)/(1024*1024)

fig, ax1 = plt.subplots(figsize=(10,5))
x    = np.arange(len(model_data["Model"]))
bars = ax1.bar(x, model_data["Size_MB"], color=["#3498db","#2ecc71","#e67e22","#e74c3c"],
               alpha=0.8, width=0.5, label="Size (MB)")
ax1.set_ylabel("Model Size (MB)", color="#3498db")
ax1.set_xticks(x); ax1.set_xticklabels(model_data["Model"])
ax1.set_title("Model Size vs Accuracy\n(Update with actual engine sizes from engines/ directory)",
              fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(x, model_data["Accuracy"], "D--k", linewidth=2, markersize=8, label="Accuracy (%)")
ax2.set_ylabel("Accuracy (%)", color="black")
ax2.set_ylim(85, 93)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")
fig.tight_layout(); fig.savefig(out("33_model_size_comparison.png"),dpi=DPI); plt.close(fig)
print("Saved: 33_model_size_comparison.png")


# ══════════════════════════════════════════════
# 34  Task success rate vs GPU load
#     Success = iteration completed without Critical state
# ══════════════════════════════════════════════
if all("system_state" in df.columns and "gpu_util" in df.columns
       for df in dfs.values()):
    fig, ax = plt.subplots(figsize=(9,6))
    for label, df in dfs.items():
        # Bin by GPU util
        df2 = df.copy()
        df2["success"] = (df2["system_state"] != "Critical").astype(float)
        df2["gpu_bin"]  = pd.cut(df2["gpu_util"], bins=[0,20,40,60,80,100],
                                 labels=["0-20","20-40","40-60","60-80","80-100"])
        grouped = df2.groupby("gpu_bin")["success"].mean() * 100
        ax.plot(grouped.index, grouped.values, color=COLORS[label],
                linewidth=2, marker=MARKERS[label], markersize=7, label=label)
    ax.set_xlabel("GPU Utilization Range (%)")
    ax.set_ylabel("Task Success Rate (%)")
    ax.set_title("Task Success Rate vs GPU Load\n"
                 "(Success = iteration without Critical thermal state)",
                 fontweight="bold")
    ax.legend(); ax.grid(True,alpha=0.3,linestyle="--"); ax.set_ylim(0,105)
    fig.tight_layout(); fig.savefig(out("34_task_success_rate.png"),dpi=DPI); plt.close(fig)
    print("Saved: 34_task_success_rate.png")


# ══════════════════════════════════════════════
# 35  FPS vs Power scatter
# ══════════════════════════════════════════════
fig,ax = plt.subplots(figsize=(8,6))
for label,df in dfs.items():
    if "fps" not in df.columns or "power" not in df.columns: continue
    ax.scatter(df["power"],df["fps"],color=COLORS[label],alpha=0.45,
               s=30,label=label,marker=MARKERS[label])
ax.set_xlabel("Power (W)"); ax.set_ylabel("FPS")
ax.set_title("FPS vs Power Consumption\n(Higher FPS + Lower Power = better)",
             fontweight="bold")
ax.legend(); ax.grid(True,alpha=0.3,linestyle="--")
fig.tight_layout(); fig.savefig(out("35_fps_vs_power.png"),dpi=DPI); plt.close(fig)
print("Saved: 35_fps_vs_power.png")


# ══════════════════════════════════════════════
# 36  Throughput bar chart
# ══════════════════════════════════════════════
fig,ax = plt.subplots(figsize=(8,5))
tp_labels,tp_means,tp_stds,tp_colors = [],[],[],[]
for label,df in dfs.items():
    if "throughput_tasks_per_sec" not in df.columns: continue
    tp_labels.append(label)
    tp_means.append(df["throughput_tasks_per_sec"].mean())
    tp_stds.append(df["throughput_tasks_per_sec"].std())
    tp_colors.append(COLORS[label])
bars = ax.bar(tp_labels,tp_means,yerr=tp_stds,color=tp_colors,
              alpha=0.85,capsize=5,width=0.4)
for bar,val in zip(bars,tp_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{val:.1f}", ha="center",va="bottom",fontsize=10)
ax.set_ylabel("Mean Throughput (tasks/sec)")
ax.set_title("Inference Throughput Comparison\n(Higher is Better)", fontweight="bold")
ax.grid(axis="y",alpha=0.3,linestyle="--")
fig.tight_layout(); fig.savefig(out("36_throughput_bar.png"),dpi=DPI); plt.close(fig)
print("Saved: 36_throughput_bar.png")


print(f"\nAll 36 plots saved to: {OUTPUT_DIR}/")
print("Transfer with: scp nvidia@<ip>:~/thesis_rl/jetson/src/results_plots/* ./")
