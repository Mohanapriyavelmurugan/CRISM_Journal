import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set publication quality styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Actual Experimental Data from ablation_table.md
variants_labels = ["Variant A\n(Physics-Based\nCorrection)", 
                   "Variant B\n(Spatial CNN\nOnly)", 
                   "Variant C\n(Spectral Noise2Void\nDAE Only)", 
                   "Variant D\n(Proposed Hybrid\nFramework)"]

# Data
snr_values = [55.82, 98.54, 55.70, 97.01]
sam_values = [0.0000, 0.0315, 0.0076, 0.0272]
pearson_values = [1.0000, 0.9876, 0.9998, 0.9915]

# Color Scheme
# Variant A: Light Gray, B: Medium Gray, C: Gray, D: Dark Blue
colors = ['#E0E0E0', '#A0A0A0', '#808080', '#1F497D']

x = np.arange(len(variants_labels))
width = 0.55

# Figure setup: Make it tall enough to fit the text box at the bottom
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
plt.subplots_adjust(top=0.9, bottom=0.35, wspace=0.25)

fig.suptitle("Ablation Analysis of the Proposed Spatial–Spectral Hybrid Denoising Framework", fontsize=16, fontweight='bold', y=0.98)

def add_value_labels(ax, rects, is_four_decimal=False):
    for rect in rects:
        height = rect.get_height()
        if is_four_decimal:
            label = f'{height:.4f}'
        else:
            label = f'{height:.2f}'
        
        ax.text(rect.get_x() + rect.get_width()/2., height + (ax.get_ylim()[1] * 0.015),
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 1: SNR
rects1 = axes[0].bar(x, snr_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('(a) Average Spatial SNR (Higher is Better)', fontsize=14, pad=15)
axes[0].set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=14, fontweight='bold')
axes[0].set_ylim(0, max(snr_values) * 1.15)
add_value_labels(axes[0], rects1, is_four_decimal=False)

# Plot 2: SAM
rects2 = axes[1].bar(x, sam_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_title('(b) Average Spectral Angle Mapper (Lower is Better)', fontsize=14, pad=15)
axes[1].set_ylabel('Spectral Angle Mapper (SAM, radians)', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, max(sam_values) * 1.15)
add_value_labels(axes[1], rects2, is_four_decimal=True)

# Plot 3: Pearson Correlation
rects3 = axes[2].bar(x, pearson_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[2].set_title('(c) Average Pearson Correlation (Higher is Better)', fontsize=14, pad=15)
axes[2].set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
axes[2].set_ylim(0, 1.05)
add_value_labels(axes[2], rects3, is_four_decimal=True)

# Apply common styling
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(variants_labels, fontsize=12, linespacing=1.3)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#D3D3D3', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Key Findings Text Box ---
key_findings_text = (
    "Key Findings\n\n"
    "• Variant A (Physics-Based Correction) serves as the reference baseline and provides initial removal of deterministic sensor artifacts.\n"
    "• Variant B (Spatial CNN) achieves the highest spatial enhancement by effectively suppressing residual stripe noise and improving SNR.\n"
    "• Variant C (Spectral Noise2Void DAE) preserves spectral characteristics, producing lower SAM and higher Pearson correlation than the spatial module alone.\n"
    "• Variant D (Proposed Hybrid Framework) combines the strengths of both learning modules, achieving the best balance between spatial restoration and spectral fidelity."
)

bbox_props = dict(boxstyle="square,pad=1.0", fc="#F8F8F8", ec="black", lw=1.0)
fig.text(0.5, 0.22, key_findings_text, ha='center', va='top', fontsize=12, linespacing=1.5, bbox=bbox_props)

# --- Footnote ---
footnote_text = "* Variant A is the reference stage; therefore, SAM = 0 and Pearson correlation = 1 when the physics-corrected cube is compared with itself."
fig.text(0.5, 0.04, footnote_text, ha='center', va='top', fontsize=11, style='italic', color='#555555')


# Save figure
plt.savefig('ablation_study_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('ablation_study_comparison.svg', dpi=300, bbox_inches='tight')
plt.savefig('ablation_study_comparison.png', dpi=300, bbox_inches='tight')

print("Ablation study figure successfully generated.")
