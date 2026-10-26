#!/usr/bin/env python3
"""
CRISM journal figures for denoising / noise / claim-status.

NO OVERLAP with mineral proof pack (R0–R3, P1):
  C0  categorical claim-status matrix (PRESENT / WEAK / ABSENT)
  D0  cross-scene S3→S4 SAM + spatial-SNR proxy (spectral preservation)
  D1  Raw / physics(S3) / ML(S4) spatial before–after (one band)
  N0  cross-scene N1–N5 affected-band counts
  N1  per-scene noise-identification dashboard

Outputs:
  D:\\CRISM scenes\\journal_results\\Paper_Figures\\Pipeline\\
  D:\\CRISM scenes\\journal_results\\Paper_Figures\\Scene_N\\
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Allow importing crism_utils from this folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crism_utils import load_crism_cube, parse_pds3_label  # noqa: E402

SCENES_ROOT = Path(r"D:\CRISM scenes\scenes")
PAPER = Path(r"D:\CRISM scenes\journal_results\Paper_Figures")
PIPE_OUT = PAPER / "Pipeline"

BEST_RUNS = {
    1: "mineral_output_depth_v1",
    2: "mineral_output_depth_v2",
    3: "mineral_output_adapt",
    4: "mineral_output_depth_v1",
    5: "mineral_output_adapt_v4",
}

# Mineral groups shown in claim matrix (fixed order)
CLAIM_GROUPS = [
    "Hydrated Minerals",
    "Al-OH Clays",
    "Fe/Mg-OH Phyllosilicates",
    "Fe/Mg Phyllosilicates",
    "Monohydrated Sulfates",
    "Sulfates",
    "Carbonates",
    "Olivine",
    "Low-Ca Pyroxene",
    "High-Ca Pyroxene",
    "Water Ice",
    "CO2 Ice",
]

STATUS_PRESENT = 2
STATUS_WEAK = 1
STATUS_ABSENT = 0
STATUS_LABEL = {2: "PRESENT", 1: "WEAK", 0: "ABSENT"}

RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "#000000",
    "axes.titlecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "text.color": "#000000",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
}


def _configure_fonts() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    elif "Times" in available:
        plt.rcParams["font.serif"] = ["Times", "DejaVu Serif"]
    else:
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", edgecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _scene_stem(scene: int) -> str:
    imgs = sorted((SCENES_ROOT / f"Scene {scene}").glob("*trr3.img"))
    if not imgs:
        raise FileNotFoundError(f"No TRR3 img for Scene {scene}")
    return imgs[0].stem


def _paths(scene: int) -> dict:
    sd = SCENES_ROOT / f"Scene {scene}"
    stem = _scene_stem(scene)
    return {
        "scene_dir": sd,
        "stem": stem,
        "img": sd / f"{stem}.img",
        "lbl": sd / f"{stem}.lbl",
        "s3": sd / "physics_output" / f"{stem}_S3.npy",
        "s4": sd / "ml_output" / f"{stem}_S4.npy",
        "noise": sd / "noise_output" / "noise_map.json",
        "val": sd / "ml_output" / "stage4_validation.json",
        "proof": sd / BEST_RUNS[scene] / "mineral_proof.json",
        "waves": sd / "wavelengths_L438.npy",
        "paper_scene": PAPER / f"Scene_{scene}",
    }


def _claim_status(ent: dict, summary_lines: List[str], group: str) -> int:
    if ent.get("present") is True:
        return STATUS_PRESENT
    # WEAK / CANDIDATE from summary text for this group
    for line in summary_lines:
        if group in line and ("CANDIDATE" in line or "WEAK" in line):
            return STATUS_WEAK
    if ent and ent.get("present") is False and (ent.get("evidence") or ent.get("n_hot_pixels")):
        # evidence listed but not PRESENT → treat as WEAK if summary says so else ABSENT
        ev = " ".join(ent.get("evidence") or [])
        if "CANDIDATE" in ev or "WEAK" in ev:
            return STATUS_WEAK
    return STATUS_ABSENT


def plot_c0_claim_matrix(scenes: List[int]) -> None:
    """Categorical PRESENT/WEAK/ABSENT matrix — not a depth heatmap (R0)."""
    mat = np.zeros((len(CLAIM_GROUPS), len(scenes)), dtype=int)
    for j, sc in enumerate(scenes):
        p = _paths(sc)["proof"]
        proof = _load_json(p)
        minerals = proof.get("minerals") or {}
        summary = proof.get("summary") or []
        for i, g in enumerate(CLAIM_GROUPS):
            mat[i, j] = _claim_status(minerals.get(g) or {}, summary, g)

    cmap = ListedColormap(["#e8e8e8", "#f0c674", "#2d6a4f"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 8.5), facecolor="white")
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([f"Scene {s}" for s in scenes])
    ax.set_yticks(range(len(CLAIM_GROUPS)))
    ax.set_yticklabels(CLAIM_GROUPS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i, STATUS_LABEL[int(mat[i, j])],
                ha="center", va="center", fontsize=10,
                color="#000000" if mat[i, j] < 2 else "#ffffff", fontweight="bold",
            )
    ax.set_title(
        "C0.  Mineral claim-status matrix (PRESENT / WEAK / ABSENT)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Scene")
    legend = [
        Patch(facecolor="#2d6a4f", edgecolor="black", label="PRESENT"),
        Patch(facecolor="#f0c674", edgecolor="black", label="WEAK / CANDIDATE"),
        Patch(facecolor="#e8e8e8", edgecolor="black", label="ABSENT"),
    ]
    ax.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, frameon=False,
    )
    fig.text(
        0.5, -0.02,
        "Distinct from R0 (which plots PRESENT diagnostic depths only).",
        ha="center", fontsize=11, color="#444444",
    )
    _save(fig, PIPE_OUT / "C0_claim_status_matrix.png")
    # drop unused im warning
    del im


def plot_d0_snr_sam(scenes: List[int]) -> None:
    """Cross-scene Stage-4 SAM + spatial-SNR proxy — not mineral metrics (R3)."""
    snr_b, snr_a, sams, labels = [], [], [], []
    for sc in scenes:
        val = _load_json(_paths(sc)["val"])
        key = list(val.keys())[0]
        m = val[key]
        snr_b.append(float(m["snr_before"]))
        snr_a.append(float(m["snr_after"]))
        sams.append(float(m["sam_rad"]))
        labels.append(f"S{sc}")

    x = np.arange(len(scenes))
    w = 0.36
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.2), facecolor="white",
        gridspec_kw={"wspace": 0.35},
    )

    axes[0].bar(x - w / 2, snr_b, w, label="S3 (physics)", color="#7a8b99", edgecolor="black", lw=0.5)
    axes[0].bar(x + w / 2, snr_a, w, label="S4 (ML)", color="#1f6f8b", edgecolor="black", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Spatial SNR proxy (|mean|/std)")
    axes[0].set_title("A.  Spatial SNR proxy (S3 → S4)", fontweight="bold", pad=10)
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].text(
        0.0, -0.22,
        "Note: scene contrast proxy (geology-dominated); not instrument noise floor.",
        transform=axes[0].transAxes, fontsize=11, color="#444444",
    )

    axes[1].bar(x, sams, color="#c45c26", edgecolor="black", lw=0.5, width=0.55)
    axes[1].axhline(0.05, color="#666666", ls="--", lw=1.0, label="SAM PASS < 0.05 rad")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean SAM (rad)")
    axes[1].set_title("B.  Spectral preservation S3→S4 (SAM)", fontweight="bold", pad=10)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    for xi, v in zip(x, sams):
        axes[1].text(xi, v + max(sams) * 0.03, f"{v:.4f}", ha="center", fontsize=11)

    fig.suptitle(
        "D0.  Denoising validation across scenes (Stage 4)",
        fontweight="bold", y=1.02,
    )
    _save(fig, PIPE_OUT / "D0_cross_scene_snr_sam.png")


def _band_near(waves: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(waves - target)))


def _show_band(ax, img: np.ndarray, title: str, vmin=None, vmax=None):
    fin = img[np.isfinite(img)]
    if fin.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(fin, [2, 98])
        if vmax <= vmin:
            vmax = vmin + 1e-6
    im = ax.imshow(img, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_d1_spatial_before_after(scene: int) -> None:
    """Raw / S3 / S4 spatial panel — denoising only (no SPM/hotspots)."""
    p = _paths(scene)
    waves = np.load(p["waves"]) if p["waves"].is_file() else None
    bi = _band_near(waves, 2.13) if waves is not None else 220
    wl = float(waves[bi]) if waves is not None else float("nan")

    # S3 / S4 via memmap — single band
    s3 = np.load(p["s3"], mmap_mode="r")
    s4 = np.load(p["s4"], mmap_mode="r")
    # shape (H,W,B)
    band_s3 = np.array(s3[:, :, bi], dtype=np.float64)
    band_s4 = np.array(s4[:, :, bi], dtype=np.float64)

    # Raw from IMG
    meta = parse_pds3_label(str(p["lbl"]))
    raw = load_crism_cube(str(p["img"]), meta)
    if raw.ndim != 3:
        raise ValueError(f"Unexpected raw shape {raw.shape}")
    # match band axis: often (H,W,B)
    if raw.shape[2] == s3.shape[2]:
        band_raw = raw[:, :, bi].astype(np.float64)
    elif raw.shape[0] == s3.shape[2]:
        band_raw = raw[bi].astype(np.float64)
    else:
        # try last axis match
        band_raw = np.array(raw[..., bi], dtype=np.float64)

    # Shared stretch from S3 percentiles
    fin = band_s3[np.isfinite(band_s3) & (band_s3 > 0) & (band_s3 < 1.5)]
    if fin.size:
        vmin, vmax = np.percentile(fin, [2, 98])
    else:
        vmin, vmax = 0.0, 0.3

    residual = np.abs(band_s4 - band_s3)

    fig, axes = plt.subplots(
        1, 4, figsize=(16.0, 4.6), facecolor="white",
        gridspec_kw={"wspace": 0.22},
    )
    _show_band(axes[0], band_raw, "Raw TRR3", vmin, vmax)
    _show_band(axes[1], band_s3, "After physics (S3)", vmin, vmax)
    im = _show_band(axes[2], band_s4, "After ML (S4)", vmin, vmax)
    _show_band(axes[3], residual, "|S4 − S3| residual")
    if im is not None:
        cbar = fig.colorbar(im, ax=list(axes[0:3]), fraction=0.025, pad=0.02)
        cbar.set_label("I/F", fontsize=12)

    fig.suptitle(
        f"D1.  Scene {scene}  |  {_paths(scene)['stem']}  |  "
        f"spatial denoising at {wl:.3f} µm (band {bi})",
        fontweight="bold", y=1.05,
    )
    fig.text(
        0.5, -0.02,
        "Pipeline stages only — not mineral SPM/hotspot maps (see R1 / P1).",
        ha="center", fontsize=11, color="#444444",
    )
    out = p["paper_scene"] / f"D1_spatial_raw_S3_S4.png"
    _save(fig, out)
    del raw, s3, s4


def plot_n0_noise_counts(scenes: List[int]) -> None:
    """Cross-scene N1–N5 affected-band counts."""
    types = [
        ("N1 stripe", "striped_bands"),
        ("N2 high-noise", "high_noise_bands"),
        ("N3 spikes", "spike_bands"),
        ("N4 thermal", "thermal_bands"),
        ("N5 saturation", "saturated_bands"),
    ]
    data = {name: [] for name, _ in types}
    for sc in scenes:
        noise = _load_json(_paths(sc)["noise"])
        entry = noise[list(noise.keys())[0]]
        for name, key in types:
            data[name].append(len(entry.get(key) or []))

    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(11.5, 5.5), facecolor="white")
    width = 0.15
    colors = ["#1f6f8b", "#c45c26", "#6a4c93", "#8b6914", "#555555"]
    for i, ((name, _), col) in enumerate(zip(types, colors)):
        ax.bar(x + (i - 2) * width, data[name], width, label=name, color=col, edgecolor="black", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Scene {s}" for s in scenes])
    ax.set_ylabel("Affected bands (count)")
    ax.set_title("N0.  Noise-type band counts by scene (Stage 2)", fontweight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=5, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, PIPE_OUT / "N0_cross_scene_noise_counts.png")


def plot_n1_noise_dashboard(scene: int) -> None:
    """Per-scene N1–N5 diagnostic board from noise_map.json."""
    p = _paths(scene)
    noise = _load_json(p["noise"])
    entry = noise[list(noise.keys())[0]]
    waves = np.load(p["waves"]) if p["waves"].is_file() else np.arange(438)

    n1 = entry.get("n1_stripes") or {}
    n2 = entry.get("n2_gaussian") or {}
    n3 = entry.get("n3_spikes") or {}
    n4 = entry.get("n4_thermal") or {}
    n5 = entry.get("n5_saturation") or {}

    fig = plt.figure(figsize=(13.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35, left=0.08, right=0.96, top=0.90, bottom=0.08)

    # Summary text
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    lines = [
        "Detected noise summary",
        "",
        f"N1 striped bands:     {len(entry.get('striped_bands') or [])}",
        f"N2 high-noise bands:  {len(entry.get('high_noise_bands') or [])}",
        f"N2 medium-noise:      {len(entry.get('medium_noise_bands') or [])}",
        f"N3 spike bands:       {len(entry.get('spike_bands') or [])}",
        f"N4 thermal present:   {bool(n4.get('thermal_present'))}",
        f"N5 saturated bands:   {len(entry.get('saturated_bands') or [])}",
        f"Stripe amplitude:     {float(n1.get('stripe_amplitude') or 0):.4f}",
    ]
    ax0.text(
        0.02, 0.98, "\n".join(lines), transform=ax0.transAxes,
        va="top", ha="left", fontsize=13, linespacing=1.35,
        bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#888888"),
    )

    # N2 SNR per band
    ax1 = fig.add_subplot(gs[0, 1])
    snr = np.array(n2.get("snr_per_band") or [], dtype=float)
    if snr.size and waves.size >= snr.size:
        ax1.plot(waves[: snr.size], snr, color="#1f6f8b", lw=1.2)
        ax1.set_xlabel("Wavelength (µm)")
        ax1.set_ylabel("SNR / band")
    else:
        ax1.plot(snr, color="#1f6f8b", lw=1.2)
        ax1.set_xlabel("Band index")
        ax1.set_ylabel("SNR / band")
    ax1.set_title("N2 Gaussian — SNR per band", fontweight="bold", pad=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # N1 stripe scores
    ax2 = fig.add_subplot(gs[0, 2])
    scores = np.array(n1.get("stripe_scores") or [], dtype=float)
    if scores.size and waves.size >= scores.size:
        ax2.plot(waves[: scores.size], scores, color="#c45c26", lw=1.2)
        ax2.set_xlabel("Wavelength (µm)")
    else:
        ax2.plot(scores, color="#c45c26", lw=1.2)
        ax2.set_xlabel("Band index")
    ax2.set_ylabel("Stripe severity")
    ax2.set_title("N1 Stripe severity", fontweight="bold", pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # N3 spike counts
    ax3 = fig.add_subplot(gs[1, 0])
    spikes = np.array(n3.get("spike_counts") or [], dtype=float)
    if spikes.size and waves.size >= spikes.size:
        ax3.bar(waves[: spikes.size], spikes, width=0.01, color="#6a4c93", edgecolor="none")
        ax3.set_xlabel("Wavelength (µm)")
    else:
        ax3.bar(np.arange(spikes.size), spikes, color="#6a4c93", edgecolor="none")
        ax3.set_xlabel("Band index")
    ax3.set_ylabel("Spike count")
    ax3.set_title("N3 Spectral spikes", fontweight="bold", pad=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # N4 / N5 flags
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    t_bands = entry.get("thermal_bands") or []
    s_bands = entry.get("saturated_bands") or []
    txt = [
        "N4 Thermal",
        f"  present = {bool(n4.get('thermal_present'))}",
        f"  n_bands = {len(t_bands)}",
        "",
        "N5 Saturation",
        f"  n_bands = {len(s_bands)}",
    ]
    ax4.text(
        0.02, 0.98, "\n".join(txt), transform=ax4.transAxes,
        va="top", fontsize=13, linespacing=1.35,
        bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="#888888"),
    )
    ax4.set_title("N4 / N5 flags", fontweight="bold", pad=8)

    # Affected-band count bars
    ax5 = fig.add_subplot(gs[1, 2])
    counts = [
        len(entry.get("striped_bands") or []),
        len(entry.get("high_noise_bands") or []),
        len(entry.get("spike_bands") or []),
        len(entry.get("thermal_bands") or []),
        len(entry.get("saturated_bands") or []),
    ]
    labs = ["N1", "N2", "N3", "N4", "N5"]
    cols = ["#1f6f8b", "#c45c26", "#6a4c93", "#8b6914", "#555555"]
    ax5.bar(labs, counts, color=cols, edgecolor="black", lw=0.5)
    ax5.set_ylabel("Band count")
    ax5.set_title("Affected bands", fontweight="bold", pad=8)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    fig.suptitle(
        f"N1.  Scene {scene}  |  {p['stem']}  |  noise identification diagnostics",
        fontweight="bold", y=0.97,
    )
    _save(fig, p["paper_scene"] / "N1_noise_identification.png")


def write_manifest() -> None:
    manifest = {
        "note": "Pipeline / claim-status figures — no overlap with R0–R3 or P1 mineral proof pack.",
        "avoid_overlap_with": [
            "R0_cross_scene_present_summary.png (PRESENT depths)",
            "R1_present_evidence_panel.png (SPM hotspot + CR)",
            "R2_diagnostic_depth_maps.png",
            "R3_present_metrics.png (n_hot / depth / hmb)",
            "P1_*_proof.png (per-mineral multi-evidence)",
        ],
        "figures": {
            "C0_claim_status_matrix.png": "Categorical PRESENT/WEAK/ABSENT matrix",
            "D0_cross_scene_snr_sam.png": "Stage4 spatial-SNR proxy + SAM S3→S4",
            "D1_spatial_raw_S3_S4.png": "Per-scene Raw/S3/S4 spatial + residual",
            "N0_cross_scene_noise_counts.png": "N1–N5 affected-band counts",
            "N1_noise_identification.png": "Per-scene Stage2 noise dashboard",
        },
    }
    path = PIPE_OUT / "figure_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    plt.rcParams.update(RC)
    _configure_fonts()
    PIPE_OUT.mkdir(parents=True, exist_ok=True)

    print("C0 claim matrix")
    plot_c0_claim_matrix(args.scenes)

    print("D0 SNR / SAM")
    plot_d0_snr_sam(args.scenes)

    print("N0 noise counts")
    plot_n0_noise_counts(args.scenes)

    for sc in args.scenes:
        print(f"Scene {sc}: N1 dashboard + D1 spatial")
        plot_n1_noise_dashboard(sc)
        plot_d1_spatial_before_after(sc)

    write_manifest()
    print("Done.")


if __name__ == "__main__":
    main()
