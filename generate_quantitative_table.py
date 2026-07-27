import matplotlib.pyplot as plt

# Set publication quality styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['font.size'] = 11

fig, ax = plt.subplots(figsize=(13, 3.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ==============================================================================
# PLACEHOLDER DATA: REPLACE THESE WITH ACTUAL EXPERIMENTAL RESULTS
# Populated with placeholder values from the previous quantitative comparison
# ==============================================================================
cols = ["Method", "Spatial\nSNR ↑", "SAM ↓", "Pearson\nCorrelation ↑", "Processing Strategy", "Generalization"]
rows = [
    ["Physics-Based Correction", "25.4", "0.125", "0.72", "Physics-based", "CRISM only"],
    ["HSID-CNN", "31.2", "0.084", "0.84", "Supervised Deep Learning", "CRISM only"],
    ["N2N4M", "33.8", "0.076", "0.87", "Self-supervised Deep Learning", "CRISM only"],
    ["Proposed Hybrid Framework", "52.6", "0.042", "0.96", "Hybrid Physics +\nSelf-supervised Learning", "CRISM + M³"]
]
# ==============================================================================

# Column positions and alignments
x_pos = [0.01, 0.23, 0.33, 0.44, 0.65, 0.90]
aligns = ['left', 'center', 'center', 'center', 'center', 'center']

# Draw top rule
ax.plot([0, 1], [0.90, 0.90], color='black', lw=1.5)

# Draw headers
y_header = 0.75
for i, col in enumerate(cols):
    ax.text(x_pos[i], y_header, col, ha=aligns[i], va='center', fontweight='bold', linespacing=1.2)

# Draw mid rule
ax.plot([0, 1], [0.60, 0.60], color='black', lw=1.0)

# Draw rows
y_start = 0.48
y_step = 0.12
for r, row in enumerate(rows):
    y_curr = y_start - r * y_step
    fw = 'bold' if r == 3 else 'normal'
    for c, val in enumerate(row):
        ax.text(x_pos[c], y_curr, val, ha=aligns[c], va='center', fontweight=fw, linespacing=1.2)

# Draw bottom rule
y_bottom = y_start - 3 * y_step - 0.08
ax.plot([0, 1], [y_bottom, y_bottom], color='black', lw=1.5)

# Add footnote
footnote = "Higher Spatial SNR and Pearson Correlation indicate better spatial restoration and spectral consistency, whereas a lower SAM indicates improved preservation of spectral signatures."
ax.text(0.01, y_bottom - 0.08, footnote, ha='left', va='top', fontsize=9.5, style='italic', wrap=True)

plt.tight_layout()

# Save figure
plt.savefig('quantitative_table.pdf', dpi=300, bbox_inches='tight')
plt.savefig('quantitative_table.png', dpi=300, bbox_inches='tight')
plt.savefig('quantitative_table.svg', dpi=300, bbox_inches='tight')

# Also generate a LaTeX version for convenience
tex_content = r"""\begin{table*}[htbp]
\centering
\caption{Quantitative comparison between the proposed hybrid framework and representative state-of-the-art denoising methods. The proposed method combines physics-based preprocessing with self-supervised spatial and spectral learning, achieving improved spatial restoration while preserving spectral fidelity across both CRISM and M$^3$ datasets.}
\label{tab:quant_comparison}
\begin{tabular}{l c c c c c}
\hline\hline
\textbf{Method} & \textbf{Spatial SNR $\uparrow$} & \textbf{SAM $\downarrow$} & \textbf{Pearson Correlation $\uparrow$} & \textbf{Processing Strategy} & \textbf{Generalization} \\
\hline
Physics-Based Correction & 25.4 & 0.125 & 0.72 & Physics-based & CRISM only \\
HSID-CNN & 31.2 & 0.084 & 0.84 & Supervised Deep Learning & CRISM only \\
N2N4M & 33.8 & 0.076 & 0.87 & Self-supervised Deep Learning & CRISM only \\
\textbf{Proposed Hybrid Framework} & \textbf{52.6} & \textbf{0.042} & \textbf{0.96} & \textbf{Hybrid Physics + Self-supervised Learning} & \textbf{CRISM + M$^3$} \\
\hline\hline
\end{tabular}
\\[1ex]
\raggedright \footnotesize \textit{Higher Spatial SNR and Pearson Correlation indicate better spatial restoration and spectral consistency, whereas a lower SAM indicates improved preservation of spectral signatures.}
\end{table*}
"""

with open('quantitative_table.tex', 'w') as f:
    f.write(tex_content)

print("Table image and LaTeX source successfully generated.")
