import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup figure
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Font sizes
FONT_TITLE = 22
FONT_LARGE = 18
FONT_MED = 16
FONT_SMALL = 14

fig, ax = plt.subplots(figsize=(12, 21))
ax.set_xlim(0, 10)
ax.set_ylim(0, 36)
ax.axis('off')

# Colors for borders
C_IN = '#1f77b4'     # Blue
C_S1 = '#ff7f0e'     # Orange
C_S2 = '#2ca02c'     # Green
C_S3 = '#d62728'     # Red
C_S4 = '#9467bd'     # Purple
C_VAL = '#8c564b'    # Brown
C_OUT = '#00897B'    # Teal

def draw_container(ax, x, y, w, h, title, border_color, title_color=None):
    if title_color is None:
        title_color = border_color
        
    box = mpatches.FancyBboxPatch((x-w/2, y-h), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                  fill=True, facecolor='white', edgecolor=border_color, 
                                  linewidth=2.0, zorder=1)
    ax.add_patch(box)
    
    # Optional subtitle support (if title has \n)
    parts = title.split('\n')
    ax.text(x, y - 0.55, parts[0], ha='center', va='center', fontsize=FONT_LARGE, fontweight='bold', color=title_color, zorder=3)
    if len(parts) > 1:
        ax.text(x, y - 1.1, parts[1], ha='center', va='center', fontsize=FONT_MED, color='black', fontweight='bold', zorder=3)

def draw_inner_box(ax, x, y, w, h, text, dashed=False, bold=False, text_color='black', border_color='#CCCCCC', face_color='#FAFAFA'):
    ls = '--' if dashed else '-'
    box = mpatches.FancyBboxPatch((x-w/2, y-h), w, h, boxstyle="round,pad=0.05,rounding_size=0.1",
                                  fill=True, facecolor=face_color, edgecolor=border_color, 
                                  linewidth=1.2, linestyle=ls, zorder=2)
    ax.add_patch(box)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y - h/2, text, ha='center', va='center', fontsize=FONT_MED, fontweight=fw, color=text_color, linespacing=1.4, zorder=3)

cx = 5.0
cw = 8.8
col1_x = cx - 2.1
col2_x = cx + 2.1
bw_half = 4.0
bw_full = 8.2
bh = 0.95
v_gap = 1.15

# ----------------- 0. INPUT -----------------
y_curr = 35.0
ch_in = 1.8
draw_container(ax, cx, y_curr, cw, ch_in, "RAW INPUT\nInput CRISM / M³ raw hyperspectral data", C_IN)
y_curr -= (ch_in + 0.7)

# ----------------- 1. STAGE 1 (EDA) -----------------
ch_s1 = 5.8
draw_container(ax, cx, y_curr, cw, ch_s1, "STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)", C_S1)

y_box = y_curr - 1.5
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Metadata Parsing")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Cube Loading")

y_box -= v_gap
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Physical Validation")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Wavelength Assignment")

y_box -= v_gap
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Band Statistics")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Bad Band Detection")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Output: Validated Noise-Aware Cube", face_color='white')

y_curr -= (ch_s1 + 0.7)

# ----------------- 2. STAGE 2 (NOISE) -----------------
ch_s2 = 5.8
draw_container(ax, cx, y_curr, cw, ch_s2, "STAGE 2: NOISE CHARACTERIZATION", C_S2)

y_box = y_curr - 1.5
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Stripe Noise")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Gaussian Noise")

y_box -= v_gap
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Impulse Noise")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Dead Bands")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Thermal / Saturation Artifacts")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Output: Noise Map", face_color='white')

y_curr -= (ch_s2 + 0.7)

# ----------------- 3. STAGE 3 (PHYSICS) -----------------
ch_s3 = 5.8
draw_container(ax, cx, y_curr, cw, ch_s3, "STAGE 3: PHYSICS-BASED CORRECTION", C_S3)

y_box = y_curr - 1.5
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Illumination Correction")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Atmospheric Correction")

y_box -= v_gap
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "Spike Removal")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "Thermal Emission Correction")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Saturation Recovery (Interpolation)")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Output: Physics-Corrected Cube", face_color='white')

y_curr -= (ch_s3 + 0.7)

# ----------------- 4. STAGE 4 (ML) -----------------
ch_s4 = 4.7
draw_container(ax, cx, y_curr, cw, ch_s4, "STAGE 4: HYBRID SELF-SUPERVISED ML", C_S4)

y_box = y_curr - 1.5
t1 = "Module 1: 1D Spatial CNN\n\nStripe Residual Learning"
t2 = "Module 2: Spectral Noise2Void (DAE)\n\nEncoder → Bottleneck → Decoder"
draw_inner_box(ax, col1_x, y_box, bw_half, 1.9, t1, dashed=True, bold=True, border_color=C_S4, face_color='white')
draw_inner_box(ax, col2_x, y_box, bw_half, 1.9, t2, dashed=True, bold=True, border_color=C_S4, face_color='white')

y_box -= 2.2
draw_inner_box(ax, cx, y_box, bw_full, bh, "Adaptive Blending & Composite Loss", bold=True, face_color='white')

y_curr -= (ch_s4 + 0.7)

# ----------------- 5. VALIDATION -----------------
ch_val = 3.6
draw_container(ax, cx, y_curr, cw, ch_val, "QUALITY ASSESSMENT (VALIDATION)", C_VAL)

y_box = y_curr - 1.5
draw_inner_box(ax, col1_x, y_box, bw_half, bh, "SNR & Pearson Correlation")
draw_inner_box(ax, col2_x, y_box, bw_half, bh, "SAM & Band Depth")

y_box -= v_gap
draw_inner_box(ax, cx, y_box, bw_full, bh, "Ensures Spatial Restoration while Preserving Spectra", face_color='white')

y_curr -= (ch_val + 0.7)

# ----------------- 6. FINAL OUTPUT -----------------
ch_out = 1.8
draw_container(ax, cx, y_curr, cw, ch_out, "FINAL DENOISED HYPERSPECTRAL CUBE\nImproved SNR + Feature Preservation", C_OUT)

plt.tight_layout()

# Save in high-quality vector formats
plt.savefig('ieee_pipeline_architecture.pdf', dpi=1000, bbox_inches='tight')
plt.savefig('ieee_pipeline_architecture.svg', dpi=1000, bbox_inches='tight')
plt.savefig('ieee_pipeline_architecture.png', dpi=1000, bbox_inches='tight')

print("Reference-style IEEE architecture diagram successfully generated.")
