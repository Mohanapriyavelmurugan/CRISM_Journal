"""
config.py  —  Central configuration for HSID-CNN project
=========================================================
All paths, hyperparameters, and sensor constants in one place.
Override per-run using environment variables or by passing a dict
to any script's main() function.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "test")
PHYSICS_DIR   = os.path.join(DATA_DIR, "physics_output")   # *_S3.npy  (noisy)
ML_DIR        = os.path.join(DATA_DIR, "ml_output")        # *_S4.npy  (clean)
HSID_DIR      = os.path.join(BASE_DIR, "hsid_cnn")
CKPT_DIR      = os.path.join(HSID_DIR, "checkpoints")
RESULTS_DIR   = os.path.join(HSID_DIR, "results")
FIGURES_DIR   = os.path.join(HSID_DIR, "figures")

for d in (CKPT_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SENSOR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CRISM_L_BANDS = 438
CRISM_S_BANDS = 107
IF_MIN        = -0.05
IF_MAX        =  1.30

# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
K_ADJACENT    = 24      # number of adjacent spectral bands (paper default)
PATCH_H       = 15      # spatial height for CRISM strip (full height = 15)
PATCH_W       = 32      # spatial width for CRISM strip (half of 64)
PATCH_STRIDE  = 16      # stride for patch extraction
N_FEAT_SCALE  = 20      # channels per kernel size in multiscale block
N_BACKBONE_CH = 64      # channels in deep backbone
N_SKIP_LAYERS = [3, 5, 7, 9]   # backbone layer indices whose outputs are concatenated

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 64
LR            = 1e-3           # Adam initial learning rate
ADAM_BETAS    = (0.9, 0.999)
ADAM_EPS      = 1e-8
WEIGHT_DECAY  = 1e-5
LR_STEP_SIZE  = 30             # ReduceLROnPlateau patience
LR_FACTOR     = 0.5
EARLY_STOP_PATIENCE = 20

# Noise injection parameters (for synthetic training augmentation)
NOISE_SIGMA_RANGE = (5, 75)   # min/max sigma for injected Gaussian noise

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SPLIT
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_VAL_SPLIT = 0.85         # 85% train, 15% validation

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS THRESHOLDS (from Stage 4 pipeline)
# ─────────────────────────────────────────────────────────────────────────────
SAM_PASS_THRESHOLD   = 0.05    # rad — PASS
SAM_REVIEW_THRESHOLD = 0.10    # rad — REVIEW, else FAIL

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
FIG_DPI   = 600
FIG_FONT  = "Times New Roman"
TARGET_WL = [0.750, 1.010, 1.509, 2.018]   # µm  (for publication figures)
