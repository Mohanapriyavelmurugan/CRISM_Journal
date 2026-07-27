import matplotlib.pyplot as plt
import numpy as np

# Set publication quality styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Illustrative Experimental Data
variants_labels = ["(a) Physics-based\ncorrection only", 
                   "(b) Physics-based\n+ Spatial CNN", 
                   "(c) Physics-based\n+ Spectral DAE", 
                   "(d) Proposed\nHybrid Framework"]

# Data
snr_values = [56.1, 82.4, 74.8, 97.3]
sam_values = [0.031, 0.026, 0.018, 0.008]
pearson_values = [0.988, 0.992, 0.996, 0.999]

# Color Scheme: Grey/Blue palette only
colors = ['#E0E0E0', '#B0B0B0', '#808080', '#1F497D']

x = np.arange(len(variants_labels))
width = 0.55

# Figure setup: Make it tall enough to fit the text box at the bottom
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
plt.subplots_adjust(top=0.9, bottom=0.35, wspace=0.25)

fig.suptitle("Ablation Study of the Proposed Hybrid Denoising Framework", fontsize=16, fontweight='bold', y=0.98)

def add_value_labels(ax, rects, is_sam=False):
    for rect in rects:
        height = rect.get_height()
        if is_sam:
            label = f'{height:.3f}'
        else:
            label = f'{height:.1f}' if height > 1 else f'{height:.3f}'
        
        ax.text(rect.get_x() + rect.get_width()/2., height + (ax.get_ylim()[1] * 0.015),
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 1: SNR
rects1 = axes[0].bar(x, snr_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('Average Spatial SNR (Higher is Better)', fontsize=14, pad=15)
axes[0].set_ylabel('SNR (dB)', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max(snr_values) * 1.15)
add_value_labels(axes[0], rects1, is_sam=False)

# Plot 2: SAM
rects2 = axes[1].bar(x, sam_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_title('Average Spectral Angle Mapper (Lower is Better)', fontsize=14, pad=15)
axes[1].set_ylabel('Spectral Angle Mapper (SAM)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, max(sam_values) * 1.15)
add_value_labels(axes[1], rects2, is_sam=True)

# Plot 3: Pearson Correlation
rects3 = axes[2].bar(x, pearson_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[2].set_title('Average Pearson Correlation (Higher is Better)', fontsize=14, pad=15)
axes[2].set_ylabel('Pearson Correlation (r)', fontsize=14, fontweight='bold')
axes[2].set_ylim(0, 1.05)
add_value_labels(axes[2], rects3, is_sam=False)

# Apply common styling
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(variants_labels, fontsize=12, linespacing=1.3)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#D3D3D3', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Key Observations Text Box ---
key_findings_text = (
    "Key Observations\n\n"
    "• Physics-based correction provides the baseline removal of deterministic sensor artefacts.\n"
    "• Adding the Spatial CNN substantially improves spatial restoration by suppressing residual stripe noise.\n"
    "• Adding the Spectral Noise2Void DAE better preserves spectral signatures while reducing random Gaussian noise.\n"
    "• Combining both modules produces the best overall balance between spatial enhancement and spectral fidelity."
)

bbox_props = dict(boxstyle="square,pad=1.0", fc="#F8F8F8", ec="black", lw=1.0)
fig.text(0.5, 0.22, key_findings_text, ha='center', va='top', fontsize=12, linespacing=1.5, bbox=bbox_props)

# Save figure
plt.savefig('illustrative_ablation_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('illustrative_ablation_study.svg', dpi=300, bbox_inches='tight')
plt.savefig('illustrative_ablation_study.png', dpi=300, bbox_inches='tight')

print("Illustrative ablation study figure successfully generated.")
