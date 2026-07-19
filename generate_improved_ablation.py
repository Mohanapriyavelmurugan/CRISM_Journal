import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# IEEE Publication Style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
})

variants = ["Variant A\n(Physics)", "Variant B\n(Spatial CNN)", "Variant C\n(Spectral DAE)", "Variant D\n(Proposed Hybrid)"]
snr = [55.82, 98.54, 55.70, 97.01]
sam = [0.0000, 0.0315, 0.0076, 0.0272]
pearson = [1.0000, 0.9876, 0.9998, 0.9915]

# Colors: Muted gray for A-C, Dark Blue for D
colors = ['#B0B0B0', '#B0B0B0', '#B0B0B0', '#004080']
edge_colors = ['#333333'] * 4

# Create figure and axes
# Increase figure height to accommodate findings and footnote
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("Ablation Analysis of the Proposed Spatial–Spectral Hybrid Denoising Pipeline", 
             fontsize=16, fontweight='bold', y=1.0)

def annotate_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')

# Subplot 1: SNR
bars1 = axes[0].bar(variants, snr, color=colors, edgecolor=edge_colors, linewidth=1.2, zorder=3, width=0.6)
axes[0].set_title("Average Spatial SNR\n(Higher is Better)")
axes[0].set_ylabel("SNR")
axes[0].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)
axes[0].set_ylim(0, max(snr) * 1.15)
annotate_bars(axes[0], bars1, "{:.2f}")

# Subplot 2: SAM
bars2 = axes[1].bar(variants, sam, color=colors, edgecolor=edge_colors, linewidth=1.2, zorder=3, width=0.6)
axes[1].set_title("Average Spectral Angle Mapper\n(Lower is Better)")
axes[1].set_ylabel("SAM (radians)")
axes[1].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)
axes[1].set_ylim(0, max(sam) * 1.15)
annotate_bars(axes[1], bars2, "{:.4f}")

# Subplot 3: Pearson
bars3 = axes[2].bar(variants, pearson, color=colors, edgecolor=edge_colors, linewidth=1.2, zorder=3, width=0.6)
axes[2].set_title("Average Pearson Correlation\n(Higher is Better)")
axes[2].set_ylabel("Pearson Correlation")
axes[2].set_ylim(0.97, 1.002) # To visually differentiate the correlation values since they are all close to 1
axes[2].grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)
annotate_bars(axes[2], bars3, "{:.4f}")

plt.tight_layout()
fig.subplots_adjust(bottom=0.42, top=0.85)

# Add Boxed Annotation for Key Findings
key_findings_text = (
    "Key Findings\n"
    "• Variant A represents the physics-based correction stage and serves as the reference.\n"
    "• Variant B demonstrates that the Spatial CNN is primarily responsible for spatial stripe suppression and SNR improvement.\n"
    "• Variant C shows that the Spectral Noise2Void autoencoder primarily preserves spectral fidelity with minimal spatial enhancement.\n"
    "• Variant D combines both modules, maintaining nearly all spatial enhancement while improving spectral fidelity through reduced SAM and higher Pearson correlation."
)

fig.text(0.5, 0.28, key_findings_text, ha='center', va='top', fontsize=11, linespacing=1.6,
         bbox=dict(boxstyle="square,pad=0.6", facecolor="#FDFDFD", edgecolor="#333333", linewidth=1.2))

# Add Footnote
footnote_text = "* Variant A is the reference stage; therefore, SAM = 0 and Pearson correlation = 1 by definition because the physics-corrected image is compared with itself."
fig.text(0.5, 0.05, footnote_text, ha='center', va='bottom', fontsize=10, fontstyle='italic', color='#444444')

# Save figures
fig.savefig('ablation_improved.png', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('ablation_improved.pdf', dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig('ablation_improved.svg', dpi=600, bbox_inches='tight', facecolor='white')

print("Improved ablation figures generated successfully.")
