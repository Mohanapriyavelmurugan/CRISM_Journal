import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Data
components = ['N1 Stripe', 'N2 Gaussian', 'N3 Spike', 'N4 Thermal', 'N5 Saturation']
before = np.array([0.027, 0.045, 0.15, 0.08, 1.0])
percents = [-88.9, -73.3, -94.7, -93.8, -88.0]
after = before * (1 + np.array(percents) / 100.0)

# Settings for HD quality and larger text size
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'font.family': 'serif',
    'axes.axisbelow': True
})

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(components))
width = 0.35

color_before = '#972728' # Maroon
color_after = '#256F2B'  # Green

rects1 = ax.bar(x - width/2, before, width, label='Before Correction', color=color_before, edgecolor='none')
rects2 = ax.bar(x + width/2, after, width, label='After Correction', color=color_after, edgecolor='none')

ax.set_ylabel('Residual Noise Magnitude (log I/F)', fontweight='bold')
ax.set_xlabel('Identified Noise Components', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(components)

legend = ax.legend(loc='upper left', framealpha=1.0, edgecolor='black', borderpad=0.8)
legend.get_frame().set_linewidth(1.0)

ax.set_yscale('log')
ax.set_ylim(1e-3, 4e0)

# Set major ticks only for grid lines
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
ax.grid(axis='y', which='major', linestyle='-', alpha=0.4, color='lightgray')

# Add percentage text on top of the Before Correction bars
for i, rect in enumerate(rects1):
    height = rect.get_height()
    # Adjust multiplier for log scale visually
    y_pos = height * 1.25 
    ax.text(rect.get_x() + rect.get_width()/2., y_pos, 
            f'{percents[i]:.1f}%',
            ha='center', va='bottom', color=color_after, fontweight='bold', fontsize=18)

# Despine
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('noise_correction_hd.png', dpi=1000, bbox_inches='tight')
plt.savefig('noise_correction_hd.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('noise_correction_hd.svg', dpi=1000, bbox_inches='tight')

print("HD Plot generated successfully!")
