#!/usr/bin/env python3
"""
Journal Results figures (NEW — does not regenerate Stage5 fig1–fig7).

Per scene (into <mineral_dir>/journal_results/ and Paper_Figures/Scene_N/):
  R1_present_evidence_panel.png  — PRESENT minerals: hotspot map + hot/bland CR
  R2_diagnostic_depth_maps.png   — continuum depth maps for key diagnostics
  R3_present_metrics.png         — n_hot / depth / hot−bland bars (PRESENT only)
  P1_<Mineral>_proof.png         — one multi-panel proof page per PRESENT mineral

Cross-scene (into D:\\CRISM scenes\\journal_results\\ and Paper_Figures/):
  R0_cross_scene_present_summary.png — clay triad + Scene4 specials

Usage:
  python plot_journal_results.py
  python plot_journal_results.py --scenes 1 5
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SCENES_ROOT = Path(r"D:\CRISM scenes\scenes")
CROSS_OUT = Path(r"D:\CRISM scenes\journal_results")
PAPER_FIGURES = CROSS_OUT / "Paper_Figures"

BEST_RUNS = {
    1: "mineral_output_depth_v1",
    2: "mineral_output_depth_v2",
    3: "mineral_output_adapt",
    4: "mineral_output_depth_v1",
    5: "mineral_output_adapt_v4",
}

# Times Roman, 14 pt, white, journal-ready
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
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.6,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "axes.titlepad": 10,
    "axes.labelpad": 6,
}

SPM_DEPTH_WL = {
    "d1.05": 1.054,
    "d1.33": 1.330,
    "d1.43": 1.435,
    "d1.50": 1.510,
    "d1.75": 1.750,
    "d1.93": 1.930,
    "d2.10": 2.100,
    "d2.12": 2.120,
    "d2.21": 2.210,
    "d2.26": 2.265,
    "d2.29": 2.290,
    "d2.30": 2.300,
    "d2.40": 2.400,
    "d2.50": 2.500,
}

GROUP_COLOR = {
    "Hydrated Minerals": "#1f6f8b",
    "Al-OH Clays": "#c45c26",
    "Fe/Mg-OH Phyllosilicates": "#2d6a4f",
    "High-Ca Pyroxene": "#6a4c93",
    "Sulfates": "#8b6914",
    "Water Ice": "#2a6f7f",
    "CO2 Ice": "#5a6a7a",
}

# Short axis labels to avoid overlap at 14 pt
GROUP_SHORT = {
    "Hydrated Minerals": "Hydrated",
    "Al-OH Clays": "Al-OH",
    "Fe/Mg-OH Phyllosilicates": "Fe/Mg-OH",
    "High-Ca Pyroxene": "HCP",
    "Sulfates": "Sulfates",
    "Water Ice": "Water Ice",
    "CO2 Ice": "CO2 Ice",
}


def _configure_fonts() -> None:
    """Prefer Times New Roman when installed."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Times New Roman" in available:
        plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    elif "Times" in available:
        plt.rcParams["font.serif"] = ["Times", "DejaVu Serif"]
    else:
        plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _nearest_band(waves: np.ndarray, wl: float) -> int:
    return int(np.argmin(np.abs(waves - wl)))


def _simple_depth_map(
    cube_cr: np.ndarray, waves: np.ndarray, center: float, left: float, right: float
) -> np.ndarray:
    ci = _nearest_band(waves, center)
    li = _nearest_band(waves, left)
    ri = _nearest_band(waves, right)
    rc = cube_cr[:, :, ci].astype(np.float64)
    rl = cube_cr[:, :, li].astype(np.float64)
    rr = cube_cr[:, :, ri].astype(np.float64)
    ok = np.isfinite(rc) & np.isfinite(rl) & np.isfinite(rr)
    denom = (waves[ri] - waves[li]) + 1e-9
    t = (waves[ci] - waves[li]) / denom
    cont = rl + (rr - rl) * t
    out = np.full(rc.shape, np.nan, dtype=np.float64)
    good = ok & np.isfinite(cont) & (np.abs(cont) >= 1e-8)
    out[good] = 1.0 - rc[good] / cont[good]
    return out


DEPTH_TRIPLES = {
    "d1.05": (1.054, 1.030, 1.250),
    "d1.50": (1.510, 1.367, 1.810),
    "d1.93": (1.930, 1.857, 2.100),
    "d2.12": (2.120, 1.810, 2.530),
    "d2.21": (2.210, 2.165, 2.250),
    "d2.29": (2.290, 2.250, 2.350),
    "d2.40": (2.400, 2.300, 2.500),
}


def _scene_paths(scene: int) -> Tuple[Path, Path]:
    scene_dir = SCENES_ROOT / f"Scene {scene}"
    mineral_dir = scene_dir / BEST_RUNS[scene]
    return scene_dir, mineral_dir


def _load_scene(scene: int) -> Optional[dict]:
    scene_dir, mineral_dir = _scene_paths(scene)
    proof_p = mineral_dir / "mineral_proof.json"
    if not proof_p.is_file():
        print(f"  [skip] Scene {scene}: missing {proof_p}")
        return None
    proof = _load_json(proof_p)
    report_p = mineral_dir / "mineral_identification_report.json"
    report = _load_json(report_p) if report_p.is_file() else {}

    spm_p = mineral_dir / "spm_map.npy"
    cr_p = mineral_dir / "cube_cr.npy"
    if not cr_p.is_file():
        cr_p = mineral_dir / "cube_cr_l2.npy"
    waves_p = scene_dir / "wavelengths_L438.npy"
    if not waves_p.is_file():
        cands = list(scene_dir.glob("**/wavelengths*.npy"))
        waves_p = cands[0] if cands else None

    if not spm_p.is_file() or not cr_p.is_file() or waves_p is None:
        print(f"  [skip] Scene {scene}: missing spm/cr/waves")
        return None

    spm_names = [
        "BD1900_2", "BD2100_2", "BD2210_2", "BD2290", "BD2300", "BD2500_2",
        "OLINDEX3", "LCPINDEX2", "HCPINDEX2", "SINDEX2", "BD1500_2", "BD1435",
    ]
    return {
        "scene": scene,
        "scene_dir": scene_dir,
        "mineral_dir": mineral_dir,
        "out_dir": mineral_dir / "journal_results",
        "proof": proof,
        "report": report,
        "spm_map": np.load(spm_p),
        "cube_cr": np.load(cr_p, mmap_mode="r"),
        "waves": np.load(waves_p),
        "spm_names": spm_names,
        "frt": report.get("scene") or proof.get("scene") or f"Scene{scene}",
    }


def _present_minerals(proof: dict) -> List[Tuple[str, dict]]:
    out = []
    for g, e in (proof.get("minerals") or {}).items():
        if e.get("present"):
            out.append((g, e))
    order = [
        "Hydrated Minerals", "Al-OH Clays", "Fe/Mg-OH Phyllosilicates",
        "High-Ca Pyroxene", "Sulfates", "Water Ice", "CO2 Ice",
    ]
    rank = {n: i for i, n in enumerate(order)}
    out.sort(key=lambda x: rank.get(x[0], 50))
    return out


def _mean_cr_from_mask(cube: np.ndarray, mask: np.ndarray, nmax: int = 4000) -> Optional[np.ndarray]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    specs = cube[ys, xs, :]
    ok = np.isfinite(specs).sum(axis=1) >= max(8, specs.shape[1] // 4)
    if not ok.any():
        return None
    specs = specs[ok]
    if len(specs) > nmax:
        rng = np.random.default_rng(42)
        specs = specs[rng.choice(len(specs), nmax, replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(specs, axis=0)
    if mean is None or not np.isfinite(mean).any():
        return None
    return mean


def _barh_labels(ax, y, values, fmt: str = "{:.3f}", inside_frac: float = 0.22) -> None:
    """Place value labels inside long bars; outside short ones — never into next axes."""
    vmax = max(values) if values else 0.0
    xmax = ax.get_xlim()[1]
    for yi, v in zip(y, values):
        label = fmt.format(v)
        if vmax > 0 and v >= inside_frac * vmax:
            ax.text(v * 0.97, yi, label, va="center", ha="right", fontsize=12, color="white")
        else:
            # keep inside axes bounds
            x = min(v + 0.02 * max(xmax, 1e-6), xmax * 0.98)
            ax.text(x, yi, label, va="center", ha="left", fontsize=12, color="#000000")


def _depth_metrics(ent: dict) -> Tuple[str, float, float, float]:
    dk = ent.get("proof_depth_key") or ""
    use = "d2.40" if dk == "sulfate_multi" else dk
    d = float((ent.get("band_depths_hot") or {}).get(use, float("nan")))
    db = float((ent.get("band_depths_bland") or {}).get(use, float("nan")))
    hmb = (d - db) if np.isfinite(d) and np.isfinite(db) else float("nan")
    return use, d, db, hmb


def _fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}" if np.isfinite(v) else "n/a"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white", edgecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def _paper_scene_dir(scene: int) -> Path:
    return PAPER_FIGURES / f"Scene_{scene}"


def _mirror_to_paper(src: Path, scene: Optional[int] = None) -> None:
    """Copy a saved figure into the Paper_Figures tree."""
    if scene is None:
        dest = PAPER_FIGURES / src.name
    else:
        dest = _paper_scene_dir(scene) / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  copied {dest}")


def _mineral_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "Mineral"


def _hot_mask_for_ent(
    data: dict, ent: dict
) -> Tuple[Optional[np.ndarray], np.ndarray, float, int]:
    """Return (spm_layer_or_None, hot_mask, thr, n_hot)."""
    spm = data["spm_map"]
    names = data["spm_names"]
    valid = np.any(np.isfinite(spm), axis=2)
    spm_name = ent.get("spm_index") or ""
    thr = float(ent.get("thr_adaptive") or 0.01)
    n_hot = int(ent.get("n_hot_pixels") or 0)
    j = names.index(spm_name) if spm_name in names else None
    if j is None:
        return None, np.zeros(valid.shape, dtype=bool), thr, n_hot
    layer = spm[:, :, j].astype(np.float64)
    hot = valid & np.isfinite(layer) & (layer >= thr)
    if n_hot == 0:
        n_hot = int(hot.sum())
    return layer, hot, thr, n_hot


def _bland_mask(data: dict, valid: np.ndarray) -> np.ndarray:
    spm = data["spm_map"]
    bland_proxy = valid & np.isfinite(np.nanmax(np.nan_to_num(spm, nan=-np.inf), axis=2))
    if not bland_proxy.any():
        return np.zeros_like(valid)
    mx = np.nanmax(np.nan_to_num(spm, nan=-np.inf), axis=2)
    thr_b = float(np.percentile(mx[bland_proxy], 10))
    return bland_proxy & (mx <= thr_b)


def plot_p1_single_mineral_proof(data: dict) -> None:
    """One 2x2 proof page per PRESENT mineral (SPM, depth, spectrum, evidence)."""
    present = _present_minerals(data["proof"])
    if not present:
        print(f"  Scene {data['scene']}: no PRESENT — skip P1")
        return

    waves = data["waves"]
    cube = np.asarray(data["cube_cr"])
    spm = data["spm_map"]
    valid = np.any(np.isfinite(spm), axis=2)
    bland = _bland_mask(data, valid)
    frt = data["frt"]
    scene = data["scene"]

    for group, ent in present:
        color = GROUP_COLOR.get(group, "#333333")
        layer, hot, thr, n_hot = _hot_mask_for_ent(data, ent)
        spm_name = ent.get("spm_index") or ""
        dkey, dval, dbland, hmb = _depth_metrics(ent)
        sam = ent.get("best_sam")
        lib = ent.get("best_library") or ""
        evidence = ent.get("evidence") or []

        fig = plt.figure(figsize=(13.5, 10.5), facecolor="white")
        gs = fig.add_gridspec(
            2, 2, hspace=0.38, wspace=0.32,
            left=0.07, right=0.96, top=0.90, bottom=0.07,
        )
        ax_spm = fig.add_subplot(gs[0, 0])
        ax_dep = fig.add_subplot(gs[0, 1])
        ax_spec = fig.add_subplot(gs[1, 0])
        ax_txt = fig.add_subplot(gs[1, 1])

        # --- A. SPM hotspot ---
        if layer is not None:
            show = np.where(np.isfinite(layer), layer, np.nan)
            vmin = float(np.nanpercentile(show, 5)) if np.isfinite(show).any() else 0.0
            vmax = float(np.nanpercentile(show, 99)) if np.isfinite(show).any() else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-3
            im = ax_spm.imshow(show, origin="upper", cmap="cividis", vmin=vmin, vmax=vmax)
            if hot.any():
                ax_spm.contour(
                    hot.astype(float), levels=[0.5], colors=["white"],
                    linewidths=0.8, origin="upper",
                )
            cbar = fig.colorbar(im, ax=ax_spm, fraction=0.046, pad=0.04)
            cbar.set_label("SPM", fontsize=12)
            cbar.ax.tick_params(labelsize=12)
        else:
            ax_spm.text(0.5, 0.5, "SPM missing", ha="center", va="center",
                        transform=ax_spm.transAxes)
        ax_spm.set_xticks([])
        ax_spm.set_yticks([])
        ax_spm.set_title(f"A.  SPM hotspot  ({spm_name})", fontweight="bold", pad=10, color=color)
        ax_spm.set_xlabel(f"n_hot = {n_hot:,}    thr = {thr:.4f}", fontsize=12, labelpad=8)

        # --- B. Diagnostic depth map ---
        use_key = dkey if dkey in DEPTH_TRIPLES else None
        if use_key is None and dkey == "sulfate_multi":
            use_key = "d2.40"
        if use_key in DEPTH_TRIPLES:
            c, l, r = DEPTH_TRIPLES[use_key]
            dmap = _simple_depth_map(cube, waves, c, l, r)
            fin = dmap[np.isfinite(dmap)]
            if fin.size:
                vmin, vmax = np.percentile(fin, [2, 98])
                if vmax <= vmin:
                    vmax = vmin + 1e-3
            else:
                vmin, vmax = 0.0, 0.05
            im2 = ax_dep.imshow(
                dmap, origin="upper", cmap="magma",
                vmin=max(vmin, -0.02), vmax=vmax,
            )
            cbar2 = fig.colorbar(im2, ax=ax_dep, fraction=0.046, pad=0.04)
            cbar2.set_label("CR depth", fontsize=12)
            cbar2.ax.tick_params(labelsize=12)
            ax_dep.set_title(
                f"B.  Diagnostic depth  ({use_key})",
                fontweight="bold", pad=10, color=color,
            )
        else:
            ax_dep.text(0.5, 0.5, "No diagnostic depth key", ha="center", va="center",
                        transform=ax_dep.transAxes)
            ax_dep.set_title("B.  Diagnostic depth", fontweight="bold", pad=10)
        ax_dep.set_xticks([])
        ax_dep.set_yticks([])

        # --- C. Hot vs bland CR spectrum ---
        spec_h = _mean_cr_from_mask(cube, hot) if hot.any() else None
        spec_b = _mean_cr_from_mask(cube, bland) if bland.any() else None
        if spec_h is not None:
            ax_spec.plot(waves, spec_h, color=color, lw=1.8, label="Hot mean CR")
        if spec_b is not None:
            ax_spec.plot(waves, spec_b, color="#666666", lw=1.4, ls="--", label="Bland mean CR")
        if dkey in SPM_DEPTH_WL and np.isfinite(dval):
            wl = SPM_DEPTH_WL[dkey]
            ax_spec.axvline(
                wl, color=color, ls=":", lw=1.2, alpha=0.9,
                label=f"{dkey} = {_fmt(dval)}",
            )
        ax_spec.set_xlim(1.0, 2.6)
        ax_spec.set_xlabel("Wavelength (µm)")
        ax_spec.set_ylabel("CR reflectance")
        ax_spec.set_title("C.  Hot vs bland continuum-removed spectra", fontweight="bold", pad=10)
        for spine in ("top", "right"):
            ax_spec.spines[spine].set_visible(False)
        ax_spec.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
            frameon=False, fontsize=11, columnspacing=1.2, handlelength=2.0,
        )

        # --- D. Evidence panel (from mineral_proof.json) ---
        ax_txt.axis("off")
        sam_s = _fmt(float(sam), 4) if sam is not None else "n/a"
        lines = [
            f"D.  Multi-evidence PRESENT proof",
            "",
            f"Mineral group:  {group}",
            f"SPM index:      {spm_name}",
            f"Hotspot pixels: {n_hot:,}",
            f"Adaptive thr:   {thr:.4f}",
            f"Proof depth:    {dkey or 'n/a'} = {_fmt(dval, 4)} (hot)",
            f"Bland depth:    {_fmt(dbland, 4)}",
            f"Hot − bland:    {_fmt(hmb, 4)}",
            f"Library match:  {lib or 'n/a'}  (SAM = {sam_s})",
            "",
            "Evidence lines (from mineral_proof.json):",
        ]
        for e in evidence[:8]:
            lines.append(f"  • {e}")
        if ent.get("ice_proof_only"):
            lines.append("")
            lines.append("Note: ice_proof_only — cite with ice-context caveats.")
        ax_txt.text(
            0.02, 0.98, "\n".join(lines),
            transform=ax_txt.transAxes, va="top", ha="left",
            fontsize=13, linespacing=1.35, family="serif",
            bbox=dict(boxstyle="square,pad=0.55", facecolor="white", edgecolor="#888888"),
        )

        fig.suptitle(
            f"Scene {scene}  |  {frt}  |  {group} — PRESENT proof",
            fontweight="bold", y=0.97, color=color,
        )
        out_name = f"P1_{_mineral_slug(group)}_proof.png"
        out_path = data["out_dir"] / out_name
        _save(fig, out_path)
        _mirror_to_paper(out_path, scene=scene)


def plot_r1_present_evidence(data: dict) -> None:
    """Hotspot map + hot/bland CR for each PRESENT mineral."""
    present = _present_minerals(data["proof"])
    if not present:
        print(f"  Scene {data['scene']}: no PRESENT — skip R1")
        return

    n = len(present)
    # Extra room for 14 pt titles + below-axes legends; wide gap vs colorbars
    fig, axes = plt.subplots(
        n, 2, figsize=(13.5, 5.0 * n),
        gridspec_kw={"width_ratios": [1.0, 1.4], "wspace": 0.52, "hspace": 0.78},
        squeeze=False,
    )
    waves = data["waves"]
    cube = data["cube_cr"]
    spm = data["spm_map"]
    names = data["spm_names"]
    valid = np.any(np.isfinite(spm), axis=2)

    for row, (group, ent) in enumerate(present):
        axm, axs = axes[row, 0], axes[row, 1]
        spm_name = ent.get("spm_index") or ""
        thr = float(ent.get("thr_adaptive") or 0.01)
        j = names.index(spm_name) if spm_name in names else None
        color = GROUP_COLOR.get(group, "#333333")
        dkey, dval, _db, hmb = _depth_metrics(ent)
        sam = ent.get("best_sam")
        lib = ent.get("best_library") or ""
        n_hot = int(ent.get("n_hot_pixels") or 0)

        if j is not None:
            layer = spm[:, :, j].astype(np.float64)
            hot = valid & np.isfinite(layer) & (layer >= thr)
            show = np.where(np.isfinite(layer), layer, np.nan)
            vmin = float(np.nanpercentile(show, 5)) if np.isfinite(show).any() else 0.0
            vmax = float(np.nanpercentile(show, 99)) if np.isfinite(show).any() else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-3
            im = axm.imshow(show, origin="upper", cmap="cividis", vmin=vmin, vmax=vmax)
            if hot.any():
                axm.contour(
                    hot.astype(float), levels=[0.5], colors=["white"],
                    linewidths=0.7, origin="upper",
                )
            cbar = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
            cbar.set_label("SPM", fontsize=12)
            cbar.ax.tick_params(labelsize=12)
            if n_hot == 0:
                n_hot = int(hot.sum())
        else:
            axm.text(0.5, 0.5, "SPM missing", ha="center", va="center", transform=axm.transAxes)
            hot = np.zeros(valid.shape, dtype=bool)

        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_title(f"{group}\n{spm_name}", color=color, fontweight="bold", pad=10)
        axm.set_xlabel(f"n_hot = {n_hot:,}    thr = {thr:.3f}", fontsize=12, labelpad=8)

        # Spectra
        bland_proxy = valid & np.isfinite(np.nanmax(np.nan_to_num(spm, nan=-np.inf), axis=2))
        if bland_proxy.any():
            mx = np.nanmax(np.nan_to_num(spm, nan=-np.inf), axis=2)
            thr_b = float(np.percentile(mx[bland_proxy], 10))
            bland = bland_proxy & (mx <= thr_b)
        else:
            bland = np.zeros_like(valid)

        spec_h = _mean_cr_from_mask(np.asarray(cube), hot) if hot.any() else None
        spec_b = _mean_cr_from_mask(np.asarray(cube), bland) if bland.any() else None
        if spec_h is not None:
            axs.plot(waves, spec_h, color=color, lw=1.8, label="Hot mean CR")
        if spec_b is not None:
            axs.plot(waves, spec_b, color="#666666", lw=1.4, ls="--", label="Bland mean CR")

        axs.set_xlim(1.0, 2.6)
        axs.set_xlabel("Wavelength (µm)")
        axs.set_ylabel("CR reflectance")

        # Diagnostic marker line; depth value goes in subtitle (no in-plot text)
        if dkey in SPM_DEPTH_WL and np.isfinite(dval):
            wl = SPM_DEPTH_WL[dkey]
            axs.axvline(wl, color=color, ls=":", lw=1.2, alpha=0.9, label=f"{dkey} = {_fmt(dval)}")

        for spine in ("top", "right"):
            axs.spines[spine].set_visible(False)

        sam_s = _fmt(float(sam), 3) if sam is not None else "n/a"
        lib_s = f" ({lib})" if lib else ""
        axs.set_title(
            f"CR evidence\n"
            f"{dkey or 'depth'} hot−bland = {_fmt(hmb, 4)}    SAM = {sam_s}{lib_s}",
            fontweight="bold", pad=10,
        )
        # Legend below axes — clear of spectra and titles
        axs.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
            frameon=False, fontsize=11, columnspacing=1.4, handlelength=2.2,
        )

    frt = data["frt"]
    fig.suptitle(
        f"Scene {data['scene']}  |  {frt}  |  PRESENT mineral evidence",
        fontweight="bold", y=0.995,
    )
    _save(fig, data["out_dir"] / "R1_present_evidence_panel.png")


def plot_r2_depth_maps(data: dict) -> None:
    scene = data["scene"]
    if scene == 4:
        keys = ["d1.50", "d2.12", "d2.40", "d1.93"]
        titles = ["H$_2$O ice\n(d1.50)", "High-Ca pyroxene\n(d2.12)",
                  "Sulfates\n(d2.40)", "Hydrated minerals\n(d1.93)"]
    else:
        keys = ["d1.93", "d2.21", "d2.29"]
        titles = ["Hydrated minerals\n(d1.93)", "Al-OH clays\n(d2.21)",
                  "Fe/Mg-OH phyllosilicates\n(d2.29)"]

    cube = np.asarray(data["cube_cr"])
    waves = data["waves"]
    n = len(keys)
    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n, 5.2), squeeze=False,
        gridspec_kw={"wspace": 0.28},
    )
    for i, (k, title) in enumerate(zip(keys, titles)):
        ax = axes[0, i]
        c, l, r = DEPTH_TRIPLES[k]
        dmap = _simple_depth_map(cube, waves, c, l, r)
        fin = dmap[np.isfinite(dmap)]
        if fin.size:
            vmin, vmax = np.percentile(fin, [2, 98])
            if vmax <= vmin:
                vmax = vmin + 1e-3
        else:
            vmin, vmax = 0.0, 0.05
        im = ax.imshow(dmap, origin="upper", cmap="magma", vmin=max(vmin, -0.02), vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("CR depth", fontsize=12)
        cbar.ax.tick_params(labelsize=12)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Scene {scene}  |  Diagnostic continuum band-depth maps",
        fontweight="bold", y=1.02,
    )
    _save(fig, data["out_dir"] / "R2_diagnostic_depth_maps.png")


def plot_r3_present_metrics(data: dict) -> None:
    present = _present_minerals(data["proof"])
    if not present:
        return

    groups = [g for g, _ in present]
    shorts = [GROUP_SHORT.get(g, g) for g in groups]
    n_hot = [int(e.get("n_hot_pixels") or 0) for _, e in present]
    depths, hmbs, dkeys = [], [], []
    for _, e in present:
        dk, d, _db, hmb = _depth_metrics(e)
        depths.append(d if np.isfinite(d) else 0.0)
        hmbs.append(hmb if np.isfinite(hmb) else 0.0)
        dkeys.append(dk or "?")

    colors = [GROUP_COLOR.get(g, "#555555") for g in groups]
    # Horizontal bars + generous gaps so 14 pt labels never collide
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, max(4.2, 1.25 * len(groups) + 2.6)),
        gridspec_kw={"wspace": 0.62},
    )
    y = np.arange(len(groups))

    axes[0].barh(y, n_hot, color=colors, edgecolor="black", linewidth=0.5, height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(shorts)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Hotspot pixels")
    axes[0].set_title(r"$n_{\mathrm{hot}}$ (SPM ≥ thr)", fontweight="bold", pad=10)
    axes[0].set_xlim(0, max(n_hot) * 1.08 if n_hot else 1)
    _barh_labels(axes[0], y, n_hot, fmt="{:,.0f}")

    axes[1].barh(y, depths, color=colors, edgecolor="black", linewidth=0.5, height=0.62)
    axes[1].axvline(0.008, color="#666666", ls="--", lw=1.0, label="BD floor 0.008")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([f"{s} ({k})" for s, k in zip(shorts, dkeys)])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("CR depth (hot)")
    axes[1].set_title("Diagnostic depth", fontweight="bold", pad=10)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=11,
    )
    axes[1].set_xlim(0, max(depths) * 1.15 if depths else 0.02)
    _barh_labels(axes[1], y, depths, fmt="{:.3f}")

    axes[2].barh(y, hmbs, color=colors, edgecolor="black", linewidth=0.5, height=0.62)
    axes[2].axvline(0.005, color="#666666", ls="--", lw=1.0, label="hot−bland floor 0.005")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(shorts)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("hot − bland depth")
    axes[2].set_title("Hotspot contrast", fontweight="bold", pad=10)
    axes[2].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=11,
    )
    axes[2].set_xlim(0, max(hmbs) * 1.15 if hmbs else 0.02)
    _barh_labels(axes[2], y, hmbs, fmt="{:.3f}")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Scene {data['scene']}  |  PRESENT metrics (multi-evidence)",
        fontweight="bold", y=1.03,
    )
    legend_bits = [f"{GROUP_SHORT.get(g, g)} = {g}" for g in groups]
    fig.text(0.5, -0.06, "   |   ".join(legend_bits), ha="center", va="top", fontsize=12)

    _save(fig, data["out_dir"] / "R3_present_metrics.png")


def plot_r0_cross_scene(all_data: List[dict]) -> None:
    CROSS_OUT.mkdir(parents=True, exist_ok=True)
    triad = ["Hydrated Minerals", "Al-OH Clays", "Fe/Mg-OH Phyllosilicates"]
    mid = [d for d in all_data if d["scene"] in (1, 2, 3, 5)]
    if not mid:
        return

    fig = plt.figure(figsize=(13.5, 9.0), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.2, 1.05],
        hspace=0.48, wspace=0.35,
        left=0.10, right=0.92, top=0.90, bottom=0.08,
    )

    # Panel A
    ax = fig.add_subplot(gs[0, :])
    scenes = [d["scene"] for d in mid]
    mat = np.full((len(triad), len(scenes)), np.nan)
    annot = [[""] * len(scenes) for _ in triad]
    for j, d in enumerate(mid):
        minerals = d["proof"].get("minerals") or {}
        for i, g in enumerate(triad):
            e = minerals.get(g) or {}
            if not e.get("present"):
                continue
            dk, val, _db, _hmb = _depth_metrics(e)
            mat[i, j] = val
            nh = int(e.get("n_hot_pixels") or 0)
            annot[i][j] = f"{val:.3f}\nn = {nh // 1000}k" if np.isfinite(val) else ""

    im = ax.imshow(mat, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=0.025)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([f"Scene {s}" for s in scenes])
    ax.set_yticks(range(len(triad)))
    ax.set_yticklabels([GROUP_SHORT.get(g, g) for g in triad])
    for i in range(len(triad)):
        for j in range(len(scenes)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=12, color="#000000")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("CR depth (hot)", fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    ax.set_title(
        "A.  Mid-latitude PRESENT suite — diagnostic CR depth",
        loc="left", fontweight="bold", pad=12,
    )
    # Key under panel A
    ax.text(
        0.0, -0.18,
        "Hydrated = Hydrated Minerals    |    Al-OH = Al-OH Clays    |    "
        "Fe/Mg-OH = Fe/Mg-OH Phyllosilicates",
        transform=ax.transAxes, ha="left", va="top", fontsize=11,
    )

    # Panel B
    ax2 = fig.add_subplot(gs[1, 0])
    s4 = next((d for d in all_data if d["scene"] == 4), None)
    if s4:
        present = _present_minerals(s4["proof"])
        labs = [GROUP_SHORT.get(g, g) for g, _ in present]
        vals = []
        for _, e in present:
            _dk, d, _db, _hmb = _depth_metrics(e)
            vals.append(d if np.isfinite(d) else 0.0)
        cols = [GROUP_COLOR.get(g, "#666666") for g, _ in present]
        ax2.barh(range(len(labs)), vals, color=cols, edgecolor="black", linewidth=0.5, height=0.6)
        ax2.set_yticks(range(len(labs)))
        ax2.set_yticklabels(labs)
        ax2.set_xlabel("CR depth (hot)")
        ax2.set_title("B.  Scene 4 PRESENT (ice-aware)", loc="left", fontweight="bold", pad=10)
        ax2.invert_yaxis()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for yi, v in zip(range(len(vals)), vals):
            ax2.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=12)
        xmax = max(vals) if vals else 0.05
        ax2.set_xlim(0, xmax * 1.25)
        ax2.text(
            0.0, -0.22,
            "Note: Water Ice is proof-only; cite with ice-context caveats.",
            transform=ax2.transAxes, ha="left", va="top", fontsize=11, color="#333333",
        )
    else:
        ax2.axis("off")

    # Panel C
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = [
        "C.  What these figures support",
        "",
        "• Scenes 1–3 & 5: same mineral GROUPS PRESENT",
        "  (hydrate + Al-OH + Fe/Mg-OH) under one rule.",
        "• Scene 5: adapt transfer from Scenes 1+2 prior",
        "  recovers the same PRESENT triad.",
        "• Cluster specimen names are NOT confirmation.",
        "• Scene 4 Water Ice / Sulfates: methods/context,",
        "  not unqualified surface-ice claims.",
        "",
        "Cite: n_hot, CR depth, hot−bland, library SAM",
        "from mineral_proof.json — not fig1 cluster maps.",
    ]
    ax3.text(
        0.0, 1.0, "\n".join(lines), transform=ax3.transAxes,
        va="top", ha="left", fontsize=12, linespacing=1.35,
        bbox=dict(boxstyle="square,pad=0.6", facecolor="white", edgecolor="#888888"),
    )

    fig.suptitle(
        "CRISM multi-scene mineral detection — Results summary",
        fontweight="bold", y=0.97,
    )
    _save(fig, CROSS_OUT / "R0_cross_scene_present_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    plt.rcParams.update(RC)
    _configure_fonts()
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    loaded: List[dict] = []
    for s in args.scenes:
        print(f"Scene {s} ({BEST_RUNS.get(s)})")
        data = _load_scene(s)
        if data is None:
            continue
        plot_r1_present_evidence(data)
        plot_r2_depth_maps(data)
        plot_r3_present_metrics(data)
        plot_p1_single_mineral_proof(data)
        for name in (
            "R1_present_evidence_panel.png",
            "R2_diagnostic_depth_maps.png",
            "R3_present_metrics.png",
        ):
            src = data["out_dir"] / name
            if src.is_file():
                _mirror_to_paper(src, scene=s)
        loaded.append(data)

    if loaded:
        print("Cross-scene summary")
        plot_r0_cross_scene(loaded)
        r0 = CROSS_OUT / "R0_cross_scene_present_summary.png"
        if r0.is_file():
            _mirror_to_paper(r0, scene=None)
    print("Done.")


if __name__ == "__main__":
    main()
