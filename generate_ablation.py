import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# IEEE Publication Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

df = pd.read_csv('results/ablation/ablation_metrics.csv')
agg = df.groupby('Variant')[['SNR', 'SAM', 'Pearson']].mean().reset_index()

# Ensure correct order: A, B, C, D
agg['Variant'] = pd.Categorical(agg['Variant'], categories=['Variant A', 'Variant B', 'Variant C', 'Variant D'], ordered=True)
agg = agg.sort_values('Variant')

variants = ["Variant A\n(Physics)", "Variant B\n(Spat. CNN)", "Variant C\n(Spec. DAE)", "Variant D\n(Hybrid)"]
snr = agg['SNR'].values
sam = agg['SAM'].values
pearson = agg['Pearson'].values

# Colors
colors = ['#87CEEB', '#87CEEB', '#87CEEB', '#FF8C00'] # D is highlighted

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Ablation Study of the Proposed Spatial–Spectral Hybrid Pipeline", fontsize=14, fontweight='bold', y=1.02)

def annotate_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 1: SNR
bars1 = axes[0].bar(variants, snr, color=colors, edgecolor='black', zorder=3)
axes[0].set_title("Average Spatial SNR (Higher is Better)")
axes[0].set_ylabel("SNR")
axes[0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
annotate_bars(axes[0], bars1, "{:.1f}")

# Chart 2: SAM
bars2 = axes[1].bar(variants, sam, color=colors, edgecolor='black', zorder=3)
axes[1].set_title("Average SAM (Lower is Better)")
axes[1].set_ylabel("Spectral Angle Mapper (radians)")
axes[1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
annotate_bars(axes[1], bars2, "{:.4f}")

# Chart 3: Pearson
bars3 = axes[2].bar(variants, pearson, color=colors, edgecolor='black', zorder=3)
axes[2].set_title("Average Pearson Correlation (Higher is Better)")
axes[2].set_ylabel("Pearson Correlation")
axes[2].set_ylim(0.95, 1.005) # Since Pearson is close to 1, adjust y-lim to show differences
axes[2].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
annotate_bars(axes[2], bars3, "{:.4f}")

# Add summary box below
summary_text = (
    "Summary:\n"
    "• Spatial CNN contributes primarily to spatial restoration and stripe suppression.\n"
    "• Spectral Noise2Void contributes primarily to spectral preservation.\n"
    "• The hybrid configuration provides the best balance between spatial enhancement and spectral fidelity."
)

fig.text(0.5, -0.15, summary_text, ha='center', va='top', fontsize=11, 
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="gray", linewidth=1))

plt.tight_layout()

# Save main figure
fig.savefig('ablation.png', dpi=600, bbox_inches='tight')
fig.savefig('ablation.pdf', dpi=600, bbox_inches='tight')
fig.savefig('ablation.svg', dpi=600, bbox_inches='tight')

# Generate Appendix Figure (Per-scene ablation)
cubes = df['Cube'].unique()
fig_app, axes_app = plt.subplots(len(cubes), 3, figsize=(14, 3 * len(cubes)))
fig_app.suptitle("Per-Scene Ablation Performance (Appendix)", fontsize=14, fontweight='bold', y=1.02)

for i, cube in enumerate(cubes):
    cube_data = df[df['Cube'] == cube].copy()
    cube_data['Variant'] = pd.Categorical(cube_data['Variant'], categories=['Variant A', 'Variant B', 'Variant C', 'Variant D'], ordered=True)
    cube_data = cube_data.sort_values('Variant')
    
    snr_c = cube_data['SNR'].values
    sam_c = cube_data['SAM'].values
    pearson_c = cube_data['Pearson'].values
    
    bars1 = axes_app[i, 0].bar(variants, snr_c, color=colors, edgecolor='black', zorder=3)
    axes_app[i, 0].set_title(f"{cube} - SNR")
    axes_app[i, 0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    annotate_bars(axes_app[i, 0], bars1, "{:.1f}")
    
    bars2 = axes_app[i, 1].bar(variants, sam_c, color=colors, edgecolor='black', zorder=3)
    axes_app[i, 1].set_title(f"{cube} - SAM")
    axes_app[i, 1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    annotate_bars(axes_app[i, 1], bars2, "{:.4f}")
    
    bars3 = axes_app[i, 2].bar(variants, pearson_c, color=colors, edgecolor='black', zorder=3)
    axes_app[i, 2].set_title(f"{cube} - Pearson")
    min_p = max(0.9, min(pearson_c) - 0.01)
    axes_app[i, 2].set_ylim(min_p, 1.005)
    axes_app[i, 2].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    annotate_bars(axes_app[i, 2], bars3, "{:.4f}")

plt.tight_layout()
fig_app.savefig('ablation_appendix.png', dpi=600, bbox_inches='tight')
fig_app.savefig('ablation_appendix.pdf', dpi=600, bbox_inches='tight')
print("Ablation figures generated successfully.")
