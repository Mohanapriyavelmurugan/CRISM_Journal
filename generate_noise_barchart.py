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

# Data definition
categories = ["N1 Stripe", "N2 Gaussian", "N3 Spike", "N4 Thermal", "N5 Saturation"]
before_correction = [0.027, 0.045, 0.150, 0.080, 1.0]
after_correction = [0.003, 0.012, 0.00795, 0.00496, 0.12]
percentages = ["−88.9%", "−73.3%", "−94.7%", "−93.8%", "−88.0%"]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors
color_before = '#8B0000' # Dark red
color_after = '#006400'  # Dark green
color_percent = '#006400'

# Plot bars
rects1 = ax.bar(x - width/2, before_correction, width, label='Before Correction', color=color_before, alpha=0.85, edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x + width/2, after_correction, width, label='After Correction', color=color_after, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add y-axis gridlines
ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#D3D3D3', alpha=0.7)
ax.set_axisbelow(True) # Put grid behind bars

# Labels and Title
ax.set_xlabel('Identified Noise Components', fontsize=14, fontweight='bold')
ax.set_ylabel('Residual Noise Magnitude (log I/F)', fontsize=14, fontweight='bold')
ax.set_title('Average Noise Magnitude Before and After Physics-Based Correction', fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)

# Set logarithmic scale for y-axis
ax.set_yscale('log')

# Add legend
ax.legend(fontsize=12, frameon=True, edgecolor='black', loc='upper left')

# Add percentage labels above pairs
for i in range(len(categories)):
    # Calculate height for label (slightly above the taller bar, which is always 'before')
    height = before_correction[i]
    # Place text directly above the center of the pair of bars
    ax.text(x[i], height * 1.25, percentages[i], ha='center', va='bottom', fontsize=11, color=color_percent, fontweight='bold')

# Adjust y-limits to give room for the labels
ax.set_ylim(bottom=1e-3, top=3.0)

# Clean up top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save figure in publication quality formats
plt.savefig('noise_magnitude_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('noise_magnitude_comparison.svg', dpi=300, bbox_inches='tight')
plt.savefig('noise_magnitude_comparison.png', dpi=300, bbox_inches='tight')

print("Bar chart successfully generated in publication quality.")
