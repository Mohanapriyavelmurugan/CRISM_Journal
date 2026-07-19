import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path

# Setup figure
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(26, 9))
ax.set_xlim(0, 28)
ax.set_ylim(0, 10)
ax.axis('off')

plt.title("Overall Spatial–Spectral Hybrid Denoising Pipeline for CRISM Hyperspectral Data", 
          fontsize=22, fontweight='bold', y=0.92, pad=20)

# Colors
color_input = '#E0E0E0'
color_prep = '#D0E4F5'
color_phys = '#FFE0B2'
color_nn_bg = '#E8F5E9'
color_nn_sub = '#A5D6A7'
color_out = '#4CAF50'
edge_color = '#333333'
text_color = '#000000'

def draw_block(ax, x, y, w, h, title, body, output, bg_color, is_final=False, highlight=False):
    # Main box
    lw = 2.5 if highlight else 1.5
    ls = '--' if highlight else '-'
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  fill=True, facecolor=bg_color, edgecolor=edge_color, 
                                  linewidth=lw, linestyle=ls)
    ax.add_patch(box)
    
    # Title
    title_y = y + h - 0.4
    ax.text(x + w/2, title_y, title, ha='center', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Separator line
    ax.plot([x + 0.2, x + w - 0.2], [title_y - 0.3, title_y - 0.3], color=edge_color, linewidth=1)
    
    # Body text
    body_y = title_y - 0.7
    ax.text(x + 0.3, body_y, body, ha='left', va='top', fontsize=11, color='black', linespacing=1.4)
    
    # Output section if exists
    if output:
        out_y = y + 1.2
        ax.plot([x + 0.2, x + w - 0.2], [out_y, out_y], color=edge_color, linewidth=1)
        ax.text(x + w/2, out_y - 0.3, "Output:", ha='center', va='center', fontsize=11, fontweight='bold', style='italic')
        ax.text(x + w/2, out_y - 0.7, output, ha='center', va='top', fontsize=11, color='black')

def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=10, shrink=0.05))
    if text:
        ax.text((x1+x2)/2, y1 + 0.2, text, ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1. Input Box
draw_block(ax, 0.5, 4.5, 2.5, 3, "Input", "\nRaw CRISM\nTRR3 Cube\n(.img + .lbl)", "", color_input)

# Arrow 1
draw_arrow(ax, 3.1, 6.0, 3.8, 6.0)

# 2. Stage 1: EDA
stage1_body = "• PDS3 metadata parsing\n• Cube loading\n• Sensor identification (L/S)\n• Wavelength extraction\n• Physical sanity checks\n• Band statistics comp.\n• Bad band masking\n• Band quality class."
stage1_out = "EDA Summary\nBand Statistics\nValid Band Mask"
draw_block(ax, 3.9, 2.5, 3.5, 7, "Stage 1: EDA", stage1_body, stage1_out, color_prep)

# Arrow 2
draw_arrow(ax, 7.5, 6.0, 8.2, 6.0)

# 3. Stage 2: Noise Characterization
stage2_body = "• Stripe noise detection\n• Gaussian noise est.\n• Dead pixel detection\n• Impulse noise detection\n• Noise map generation"
stage2_out = "Noise Characterization\nMap"
draw_block(ax, 8.3, 2.5, 3.5, 7, "Stage 2: Noise Char.", stage2_body, stage2_out, color_prep)

# Arrow 3
draw_arrow(ax, 11.9, 6.0, 12.6, 6.0)

# 4. Stage 3: Physics-Based Correction
stage3_body = "• Destriping\n• Atmospheric correction\n• Instrument artifact rem.\n• Radiometric norm.\n• Bad band interpolation"
stage3_out = "Physics Corrected\nCube (S3)"
draw_block(ax, 12.7, 2.5, 3.5, 7, "Stage 3: Physics Corr.", stage3_body, stage3_out, color_phys)

# Arrow 4
draw_arrow(ax, 16.3, 6.0, 17.0, 6.0)

# 5. Stage 4: Hybrid Learning Framework
# Main container
lw = 2.5
box4 = mpatches.FancyBboxPatch((17.1, 2.5), 7.2, 7, boxstyle="round,pad=0.1,rounding_size=0.2",
                              fill=True, facecolor=color_nn_bg, edgecolor='#2E7D32', 
                              linewidth=lw, linestyle='--')
ax.add_patch(box4)
ax.text(17.1 + 3.6, 2.5 + 7 - 0.4, "Stage 4: Hybrid Learning Framework", ha='center', va='center', fontsize=13, fontweight='bold', color='#1B5E20')
ax.plot([17.3, 17.1 + 7.2 - 0.2], [9.1 - 0.3, 9.1 - 0.3], color='#2E7D32', linewidth=1.5)

# Sub-block 1
sb1 = mpatches.FancyBboxPatch((17.4, 4.3), 3.0, 3.8, boxstyle="round,pad=0.1", fill=True, facecolor=color_nn_sub, edgecolor=edge_color)
ax.add_patch(sb1)
ax.text(17.4 + 1.5, 4.3 + 3.8 - 0.4, "Spatial CNN", ha='center', va='center', fontsize=11, fontweight='bold')
ax.plot([17.5, 17.4 + 3.0 - 0.1], [4.3 + 3.8 - 0.8, 4.3 + 3.8 - 0.8], color=edge_color, linewidth=1)
ax.text(17.4 + 0.2, 4.3 + 3.8 - 1.2, "Purpose:\nRemove deterministic\ncolumn striping", ha='left', va='top', fontsize=10)

# Arrow 4.5
draw_arrow(ax, 20.5, 6.0, 21.0, 6.0)

# Sub-block 2
sb2 = mpatches.FancyBboxPatch((21.1, 4.3), 3.0, 3.8, boxstyle="round,pad=0.1", fill=True, facecolor=color_nn_sub, edgecolor=edge_color)
ax.add_patch(sb2)
ax.text(21.1 + 1.5, 4.3 + 3.8 - 0.4, "Spectral Noise2Void", ha='center', va='center', fontsize=11, fontweight='bold')
ax.plot([21.2, 21.1 + 3.0 - 0.1], [4.3 + 3.8 - 0.8, 4.3 + 3.8 - 0.8], color=edge_color, linewidth=1)
ax.text(21.1 + 0.2, 4.3 + 3.8 - 1.2, "Purpose:\nRemove random\nGaussian noise while\npreserving spectral\nsignatures", ha='left', va='top', fontsize=10)

# Output for Stage 4
out4_y = 3.7
ax.plot([17.3, 24.1], [out4_y, out4_y], color='#2E7D32', linewidth=1.5)
ax.text(20.7, out4_y - 0.3, "Output:", ha='center', va='center', fontsize=11, fontweight='bold', style='italic', color='#1B5E20')
ax.text(20.7, out4_y - 0.7, "Final Denoised Cube (S4)", ha='center', va='top', fontsize=11, fontweight='bold', color='#1B5E20')

# Arrow 5
draw_arrow(ax, 24.4, 6.0, 25.1, 6.0)

# 6. Final Stage
final_body = "• Spectral Preservation\n• Improved SNR\n• Reduced Stripe Noise\n• High Pearson Corr.\n• Low SAM"
draw_block(ax, 25.2, 3.5, 2.5, 5, "Mineral ID", final_body, "", color_out)
# Adjust final stage appearance to be dark green with white text
box5 = mpatches.FancyBboxPatch((25.2, 3.5), 2.5, 5, boxstyle="round,pad=0.1,rounding_size=0.2",
                              fill=True, facecolor='#2E7D32', edgecolor='#1B5E20', linewidth=1.5)
ax.add_patch(box5)
title5_y = 3.5 + 5 - 0.4
ax.text(25.2 + 1.25, title5_y, "Final Stage:\nMineral ID", ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.plot([25.4, 25.2 + 2.5 - 0.2], [title5_y - 0.6, title5_y - 0.6], color='white', linewidth=1)
ax.text(25.2 + 0.2, title5_y - 1.0, final_body, ha='left', va='top', fontsize=11, color='white', linespacing=1.5)


plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save high-quality figures
plt.savefig('pipeline_architecture.png', dpi=600, bbox_inches='tight')
plt.savefig('pipeline_architecture.pdf', dpi=600, bbox_inches='tight')
plt.savefig('pipeline_architecture.svg', dpi=600, bbox_inches='tight')

print("Diagram generated successfully.")
