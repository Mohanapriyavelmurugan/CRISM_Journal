import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'font.family': 'sans-serif',
    'axes.axisbelow': True
})

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1], hspace=0.4, wspace=1.0)
ax1 = fig.add_subplot(gs[0, :3])
ax2 = fig.add_subplot(gs[0, 3:])
ax3 = fig.add_subplot(gs[1, 1:5])

labels = ['Physics', 'Physics +\nSpatial CNN', 'Physics +\nSpectral DAE', 'Proposed\nHybrid']
colors = ['#BDBDBD', '#969696', '#737373', '#1F4E79']

# Plot (a)
bars1 = ax1.bar(labels, [56.1, 82.4, 74.8, 97.3], color=colors, edgecolor='none')
ax1.set_ylabel('Signal-to-Noise Ratio (SNR)', fontweight='bold')
ax1.set_title('(a) Spatial Restoration\nAverage Signal-to-Noise Ratio (Higher is Better)', pad=15)
ax1.set_ylim(0, 115)
ax1.grid(axis='y', linestyle='-', alpha=0.5, color='lightgray')
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Plot (b)
bars2 = ax2.bar(labels, [0.031, 0.026, 0.018, 0.005], color=colors, edgecolor='none')
ax2.set_ylabel('Spectral Angle Mapper (SAM)', fontweight='bold')
ax2.set_title('(b) Spectral Preservation\nAverage Spectral Angle Mapper (Lower is Better)', pad=15)
ax2.set_ylim(0, 0.036)
ax2.grid(axis='y', linestyle='-', alpha=0.5, color='lightgray')
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0005, f'{yval}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Plot (c)
bars3 = ax3.bar(labels, [0.988, 0.992, 0.996, 0.999], color=colors, edgecolor='none')
ax3.set_ylabel('Pearson Correlation', fontweight='bold')
ax3.set_title('(c) Spectral Consistency\nAverage Pearson Correlation (Higher is Better)', pad=15)
ax3.set_ylim(0, 1.15)
ax3.grid(axis='y', linestyle='-', alpha=0.5, color='lightgray')
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Despine
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


plt.savefig('ablation_study_reformatted.png', dpi=1000, bbox_inches='tight')
plt.savefig('ablation_study_reformatted.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('ablation_study_reformatted.svg', dpi=1000, bbox_inches='tight')
print("Plots generated successfully!")
