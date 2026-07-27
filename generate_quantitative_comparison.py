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

# ==============================================================================
# PLACEHOLDER DATA: REPLACE THESE WITH ACTUAL EXPERIMENTAL RESULTS
# Currently populated with dummy values for layout drafting purposes.
# ==============================================================================
methods = ["Physics-Based\nCorrection", "HSID-CNN", "N2N4M", "Proposed\nHybrid\nFramework"]

snr_values = [25.4, 31.2, 33.8, 52.6]       # Replace with actual SNR values
sam_values = [0.125, 0.084, 0.076, 0.042]   # Replace with actual SAM values
pearson_values = [0.72, 0.84, 0.87, 0.96]   # Replace with actual Pearson values
# ==============================================================================

x = np.arange(len(methods))
width = 0.55

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

fig.suptitle("Quantitative Comparison of CRISM Hyperspectral Image Denoising Methods", fontsize=16, fontweight='bold', y=1.05)

# Define colors (Light gray for others, Dark blue for Proposed)
colors = ['#C0C0C0', '#C0C0C0', '#C0C0C0', '#1F497D']

def add_value_labels(ax, rects, is_sam=False):
    for rect in rects:
        height = rect.get_height()
        if is_sam:
            label = f'{height:.3f}'
        else:
            label = f'{height:.1f}' if height > 1 else f'{height:.2f}'
        
        ax.text(rect.get_x() + rect.get_width()/2., height + (height * 0.02),
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 1: SNR
rects1 = axes[0].bar(x, snr_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_title('(a) Average Spatial SNR (Higher is Better)', fontsize=14, pad=15)
axes[0].set_ylabel('Signal-to-Noise Ratio (SNR ↑)', fontsize=14, fontweight='bold')
add_value_labels(axes[0], rects1)
# Adjust ylim to give room for labels
axes[0].set_ylim(0, max(snr_values) * 1.15)

# Plot 2: SAM
rects2 = axes[1].bar(x, sam_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[1].set_title('(b) Average Spectral Angle Mapper (Lower is Better)', fontsize=14, pad=15)
axes[1].set_ylabel('SAM (radians ↓)', fontsize=14, fontweight='bold')
add_value_labels(axes[1], rects2, is_sam=True)
axes[1].set_ylim(0, max(sam_values) * 1.15)

# Plot 3: Pearson Correlation
rects3 = axes[2].bar(x, pearson_values, width, color=colors, edgecolor='black', linewidth=0.5)
axes[2].set_title('(c) Average Pearson Correlation (Higher is Better)', fontsize=14, pad=15)
axes[2].set_ylabel('Pearson Correlation ↑', fontsize=14, fontweight='bold')
add_value_labels(axes[2], rects3)
axes[2].set_ylim(0, 1.05) # Pearson is typically up to 1.0, adjust top slightly for labels

# Apply common styling to all subplots
for i, ax in enumerate(axes):
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12, linespacing=1.3)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#D3D3D3', alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure in publication quality formats
plt.savefig('quantitative_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('quantitative_comparison.svg', dpi=300, bbox_inches='tight')
plt.savefig('quantitative_comparison.png', dpi=300, bbox_inches='tight')

print("Quantitative comparison figure successfully generated.")
