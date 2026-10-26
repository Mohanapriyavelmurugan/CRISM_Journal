"""
Shared multi-scene transfer settings for Stage4 denoise + Stage5 encoder.

Modes
-----
  full  — train from scratch (current default budgets as *max* epochs)
  adapt — load multi-scene prior, mandatory short fine-tune on this FRT

AbundanceNetwork is never transferred (always fresh per scene).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

SCENES_ROOT = os.environ.get(
    "CRISM_SCENES_ROOT",
    r"D:\CRISM scenes\scenes",
)
SHARED_PRIOR_DIR = os.environ.get(
    "CRISM_SHARED_PRIOR",
    r"D:\CRISM scenes\models\shared",
)

# Epoch budgets: static numbers become MAX; early-stop may finish sooner.
FULL_EPOCHS: Dict[str, int] = {
    "stripe": 150,
    "dae_L": 150,
    "dae_S": 80,
    "encoder": 80,
    "unmix_a": 40,
    "unmix_b": 8,
}
ADAPT_EPOCHS: Dict[str, int] = {
    "stripe": 30,
    "dae_L": 30,
    "dae_S": 20,
    "encoder": 25,
    "unmix_a": 25,
    "unmix_b": 8,
}

FULL_PATIENCE: Dict[str, int] = {
    "stripe": 20,
    "dae": 25,
    "encoder": 12,
    "unmix": 10,
}
ADAPT_PATIENCE: Dict[str, int] = {
    "stripe": 8,
    "dae": 8,
    "encoder": 6,
    "unmix": 6,
}

PRETRAIN_PIXELS_PER_SCENE = int(os.environ.get("CRISM_PRETRAIN_PIXELS", "60000"))
PRETRAIN_STRIPE_PROFILES_PER_SCENE = int(
    os.environ.get("CRISM_PRETRAIN_STRIPE", "8000")
)
ENCODER_VAL_FRAC = 0.10

STRIPE_CKPT = "stripe_cnn_W640.pth"
DAE_CKPT = "spectral_dae_L_B438.pth"
ENCODER_CKPT = "encoder_B438.pth"
PRIOR_META = "prior_meta.json"


def resolve_mode(mode: Optional[str] = None) -> str:
    m = (mode or os.environ.get("CRISM_TRAIN_MODE", "full")).strip().lower()
    if m not in ("adapt", "full"):
        raise ValueError(f"CRISM_TRAIN_MODE must be adapt|full, got {m!r}")
    return m


def resolve_prior_dir(prior_dir: Optional[str] = None) -> str:
    return prior_dir or os.environ.get("CRISM_SHARED_PRIOR", SHARED_PRIOR_DIR)


def epoch_budget(mode: str, key: str) -> int:
    table = ADAPT_EPOCHS if mode == "adapt" else FULL_EPOCHS
    return int(table[key])


def patience_budget(mode: str, key: str) -> int:
    table = ADAPT_PATIENCE if mode == "adapt" else FULL_PATIENCE
    return int(table[key])


def scene_dir(scene: int) -> str:
    return os.path.join(SCENES_ROOT, f"Scene {int(scene)}")


def prior_paths(prior_dir: Optional[str] = None) -> Dict[str, str]:
    root = resolve_prior_dir(prior_dir)
    return {
        "root": root,
        "stripe": os.path.join(root, STRIPE_CKPT),
        "dae": os.path.join(root, DAE_CKPT),
        "encoder": os.path.join(root, ENCODER_CKPT),
        "meta": os.path.join(root, PRIOR_META),
    }
