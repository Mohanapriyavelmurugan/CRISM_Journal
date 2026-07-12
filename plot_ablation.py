import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

OUTPUT_DIR = "results/ablation"
df = pd.read_csv(os.path.join(OUTPUT_DIR, "ablation_metrics.csv"))

# Compute Aggregates
agg = df.groupby("Variant").agg({
    "SNR": ["mean", "std"],
    "Pearson": ["mean", "std"],
    "SAM": ["mean", "std"],
    "Runtime": ["mean", "std"],
    "Memory": ["mean", "std"]
}).reset_index()
agg.columns = ["Variant", "SNR_mean", "SNR_std", "Pearson_mean", "Pearson_std", "SAM_mean", "SAM_std", "Runtime_mean", "Runtime_std", "Mem_mean", "Mem_std"]

# Setup IEEE Matplotlib Params
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white'
})
colors = ['#cccccc', '#1f77b4', '#ff7f0e', '#2ca02c']
variants = agg["Variant"].values

def plot_bar(metric, filename, title, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    means = agg[f"{metric}_mean"].values
    stds = agg[f"{metric}_std"].values
    x = np.arange(len(variants))
    
    ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    labels = ["Physics\n(Variant A)", "Spatial CNN\n(Variant B)", "Spectral DAE\n(Variant C)", "Hybrid\n(Variant D)"]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim: ax.set_ylim(ylim)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=600, bbox_inches='tight')
    plt.close()

plot_bar("SNR", "ablation_snr.png", "Ablation: Spatial Signal-to-Noise Ratio", "SNR (Mean ± SD)")
plot_bar("Pearson", "ablation_pearson.png", "Ablation: Spectral Pearson Correlation", "Pearson r (Mean ± SD)", ylim=(0.95, 1.0))
plot_bar("SAM", "ablation_sam.png", "Ablation: Spectral Angle Mapper (SAM)", "SAM (radians, Mean ± SD)")
plot_bar("Runtime", "ablation_runtime.png", "Ablation: Computation Time", "Runtime (Seconds, Mean ± SD)")
plot_bar("Mem", "ablation_memory.png", "Ablation: Peak Memory Usage", "Memory (MB, Mean ± SD)")

# Statistical Tests against Variant D
with open(os.path.join(OUTPUT_DIR, "ablation_statistics.txt"), "w") as f:
    f.write("=== ABLATION STATISTICAL ANALYSIS ===\n")
    f.write("Reference: Variant D (Full Hybrid Pipeline)\n")
    
    var_d = df[df["Variant"] == "Variant D"]
    for v in ["Variant A", "Variant B", "Variant C"]:
        var_x = df[df["Variant"] == v]
        f.write(f"\n--- {v} vs Variant D ---\n")
        
        for metric in ["SNR", "Pearson", "SAM"]:
            x = var_x[metric].values
            d = var_d[metric].values
            
            t_stat, p_t = ttest_rel(x, d)
            w_stat, p_w = wilcoxon(x, d)
            
            diff = x - d
            cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
            
            ci_lower = np.mean(diff) - 1.96 * (np.std(diff, ddof=1) / np.sqrt(len(diff)))
            ci_upper = np.mean(diff) + 1.96 * (np.std(diff, ddof=1) / np.sqrt(len(diff)))
            
            f.write(f"{metric}:\n")
            f.write(f"  Mean diff (X - D): {np.mean(diff):.4f}\n")
            f.write(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            f.write(f"  Paired t-test p-value: {p_t:.3e}\n")
            f.write(f"  Wilcoxon p-value: {p_w:.3e}\n")
            f.write(f"  Cohen's d: {cohens_d:.3f}\n")

# LaTeX Component Contribution Table
var_a = agg[agg["Variant"] == "Variant A"].iloc[0]
with open(os.path.join(OUTPUT_DIR, "component_contribution_table.tex"), "w") as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\centering\n")
    f.write("\\caption{Component Contribution Analysis (Relative to Physics Baseline)}\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\hline\n")
    f.write("Variant & $\\Delta$ SNR & $\\Delta$ Pearson $r$ & $\\Delta$ SAM (rad) \\\\\n")
    f.write("\\hline\n")
    for _, row in agg.iterrows():
        if row["Variant"] == "Variant A":
            f.write(f"Physics Baseline (Variant A) & - & - & - \\\\\n")
        else:
            d_snr = row["SNR_mean"] - var_a["SNR_mean"]
            d_pear = row["Pearson_mean"] - var_a["Pearson_mean"]
            d_sam = row["SAM_mean"] - var_a["SAM_mean"]
            name = row["Variant"].replace("Variant B", "Spatial CNN (Variant B)").replace("Variant C", "Spectral DAE (Variant C)").replace("Variant D", "Full Pipeline (Variant D)")
            f.write(f"{name} & {d_snr:+.2f} & {d_pear:+.4f} & {d_sam:+.4f} \\\\\n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

# LaTeX and MD Ablation Table
def write_table(file_path, is_tex=False):
    with open(file_path, "w") as f:
        if is_tex:
            f.write("\\begin{table*}[ht]\n\\centering\n\\caption{Ablation Study Metrics}\n")
            f.write("\\begin{tabular}{lccccc}\n\\hline\n")
            f.write("Variant & SNR & Pearson $r$ & SAM (rad) & Runtime (s) & Memory (MB) \\\\\n\\hline\n")
        else:
            f.write("| Variant | SNR | Pearson $r$ | SAM (rad) | Runtime (s) | Memory (MB) |\n")
            f.write("|---------|-----|-------------|-----------|-------------|-------------|\n")
            
        for _, row in agg.iterrows():
            snr = f"{row['SNR_mean']:.2f} $\\pm$ {row['SNR_std']:.2f}"
            pear = f"{row['Pearson_mean']:.4f} $\\pm$ {row['Pearson_std']:.4f}"
            sam = f"{row['SAM_mean']:.4f} $\\pm$ {row['SAM_std']:.4f}"
            rt = f"{row['Runtime_mean']:.2f} $\\pm$ {row['Runtime_std']:.2f}"
            mem = f"{row['Mem_mean']:.0f} $\\pm$ {row['Mem_std']:.0f}"
            
            if is_tex:
                f.write(f"{row['Variant']} & {snr} & {pear} & {sam} & {rt} & {mem} \\\\\n")
            else:
                snr = snr.replace('$\\pm$', '±')
                pear = pear.replace('$\\pm$', '±')
                sam = sam.replace('$\\pm$', '±')
                rt = rt.replace('$\\pm$', '±')
                mem = mem.replace('$\\pm$', '±')
                f.write(f"| {row['Variant']} | {snr} | {pear} | {sam} | {rt} | {mem} |\n")
                
        if is_tex:
            f.write("\\hline\n\\end{tabular}\n\\end{table*}\n")

write_table(os.path.join(OUTPUT_DIR, "ablation_table.tex"), is_tex=True)
write_table(os.path.join(OUTPUT_DIR, "ablation_table.md"), is_tex=False)

# Dump summary csv
agg.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)
