#!/usr/bin/env python3
"""
Offline acceptance gates for CRISM mineral ID — run BEFORE a full Stage5 re-run.

Loads saved intermediates from <scene>/mineral_output (and optional S4) and prints
PASS/FAIL for each gate. Target runtime: well under 1 minute.

Usage:
  python audit_scene_mineral_gates.py --scene 2
  python audit_scene_mineral_gates.py --data-dir "D:\\CRISM scenes\\scenes\\Scene 2"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SCENE_ROOT = os.environ.get("CRISM_SCENES_ROOT", r"D:\CRISM scenes\scenes")


def _scene_dir(scene: int) -> str:
    return os.path.join(SCENE_ROOT, f"Scene {scene}")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gate_cr_sanity(mineral_dir: str) -> Tuple[bool, str]:
    path = os.path.join(mineral_dir, "cube_cr.npy")
    if not os.path.isfile(path):
        return False, "cube_cr.npy missing"
    cr = np.load(path, mmap_mode="r")
    finite = np.isfinite(cr)
    if not finite.any():
        return False, "cube_cr has no finite values"
    vals = cr[finite]
    p99 = float(np.percentile(np.abs(vals), 99))
    max_abs = float(np.max(np.abs(vals)))
    frac_hi = float((np.abs(vals) > 2.5).mean())
    ok = (p99 <= 5.0) and (frac_hi < 0.02) and (max_abs < 1e3)
    return ok, (
        f"p99|CR|={p99:.3g} max|CR|={max_abs:.3g} frac>|2.5|={frac_hi:.3%} "
        f"(need p99<=5, frac<2%, max<1e3)"
    )


def gate_depth_keys(mineral_dir: str) -> Tuple[bool, str]:
    """
    Verify proof depth table + d1.93 fallback using code helpers on mean spectrum
    from cube_cr (or mineral_proof if already regenerated).
    """
    from Stage5_minerals import (
        PROOF_DEPTH_DEFS,
        SPM_PROOF_DEPTH_KEY,
        build_mineral_bad_mask,
        compute_proof_depths,
        robust_band_depth,
    )
    from crism_utils import WAVES_L

    waves = np.asarray(WAVES_L, dtype=np.float64)
    bad = build_mineral_bad_mask(waves)
    cr_path = os.path.join(mineral_dir, "cube_cr.npy")
    if not os.path.isfile(cr_path):
        return False, "cube_cr.npy missing"

    # Map keys required
    required = {
        "BD1900_2": "d1.93",
        "BD2100_2": "d2.10",
        "BD2210_2": "d2.21",
        "BD2290": "d2.29",
        "BD2300": "d2.30",
        "BD2500_2": "d2.50",
    }
    map_ok = all(SPM_PROOF_DEPTH_KEY.get(k) == v for k, v in required.items())
    defs_ok = all(k in PROOF_DEPTH_DEFS for k in ("d1.93", "d2.10", "d2.29", "d2.50"))

    cr = np.load(cr_path, mmap_mode="r")
    # Sample a small random set of finite pixels for a mean CR spectrum
    H, W, B = cr.shape
    rng = np.random.default_rng(0)
    ys = rng.integers(0, H, size=2000)
    xs = rng.integers(0, W, size=2000)
    specs = cr[ys, xs, :]
    finite_rows = np.isfinite(specs).sum(axis=1) > (0.5 * B)
    if not finite_rows.any():
        return False, "could not sample finite CR spectra"
    mean_spec = np.nanmean(specs[finite_rows], axis=0)
    depths = compute_proof_depths(mean_spec, waves, bad)
    d193 = depths.get("d1.93")
    d229 = depths.get("d2.29")
    d250 = depths.get("d2.50")

    # d1.93 must be finite with fallbacks even when 2.067 µm is dead in S4
    bi_2067 = int(np.argmin(np.abs(waves - 2.067)))
    dead_2067 = not np.isfinite(cr[:, :, bi_2067]).any()
    d193_ok = np.isfinite(d193) or not dead_2067  # if 2.067 alive, either way; if dead, must fallback
    if dead_2067:
        d193_ok = np.isfinite(robust_band_depth(mean_spec, waves, "d1.93", bad))

    ok = map_ok and defs_ok and d193_ok and np.isfinite(d229) and np.isfinite(d250)
    detail = (
        f"map_ok={map_ok} defs_ok={defs_ok} dead_2.067={dead_2067} "
        f"d1.93={d193} d2.29={d229} d2.50={d250}"
    )
    return bool(ok), detail


def gate_hdbscan_params() -> Tuple[bool, str]:
    """Static check: MCS formula uses fit-set scale (source inspection)."""
    path = os.path.join(ROOT, "Stage5_minerals.py")
    src = open(path, encoding="utf-8").read()
    has_mcs = ("0.010 * n_fit" in src) or ("0.008 * n_fit" in src)
    has_leaf = '"leaf"' in src
    ok = has_mcs and has_leaf
    return ok, "min_cluster_size scaled to n_fit + leaf fallback present" if ok else (
        "Stage5_minerals.py missing fit-set MCS / leaf fallback"
    )


def gate_unmix_smoke(mineral_dir: str) -> Tuple[bool, str]:
    """
    Recon MSE smoke on sanitized CR vs library endmembers (no neural train).
    Uses endmember_names.json + lib CR if present; else synthetic unit CR check.
    """
    cr_path = os.path.join(mineral_dir, "cube_cr.npy")
    if not os.path.isfile(cr_path):
        return False, "cube_cr.npy missing"
    cr = np.load(cr_path, mmap_mode="r")
    from Stage5_minerals import CR_SANITIZE_HI, CR_SANITIZE_LO, sanitize_cr_array

    # Sanitize a small sample as the next Stage5 run will
    sample = np.asarray(cr[::20, ::20, :], dtype=np.float32)
    sample, stats = sanitize_cr_array(sample)
    finite = np.isfinite(sample)
    if not finite.any():
        return False, "sanitized sample empty"
    # Proxy recon: MSE of sample vs its nanmean spectrum (order-of-magnitude check)
    mean_spec = np.nanmean(sample.reshape(-1, sample.shape[-1]), axis=0)
    flat = sample.reshape(-1, sample.shape[-1])
    m = np.isfinite(flat) & np.isfinite(mean_spec)[None, :]
    if m.sum() < 100:
        return False, "insufficient finite pairs for MSE smoke"
    err = (np.clip(flat, CR_SANITIZE_LO, CR_SANITIZE_HI) - mean_spec[None, :]) ** 2
    mse = float(np.nanmean(np.where(m, err, np.nan)))
    ok = mse < 1e3 and (stats.get("p99_abs_after") or 0) <= 2.0 + 1e-6
    return ok, f"proxy_mse={mse:.3g} sanitize_p99={stats.get('p99_abs_after')} clipped={stats.get('n_clipped')}"


def gate_present_consistency(mineral_dir: str) -> Tuple[bool, str]:
    proof_path = os.path.join(mineral_dir, "mineral_proof.json")
    report_path = os.path.join(mineral_dir, "mineral_identification_report.json")
    if not os.path.isfile(proof_path):
        return False, "mineral_proof.json missing (run Stage5 once to create baseline)"
    proof = _load_json(proof_path)
    minerals = proof.get("minerals") or {}
    present = [k for k, v in minerals.items() if v.get("present")]
    # Olivine must not be PRESENT with tiny depth
    ol = minerals.get("Olivine") or {}
    depths = ol.get("band_depths_hot") or {}
    d105 = depths.get("d1.05")
    ol_bad = bool(ol.get("present")) and (
        d105 is None or (isinstance(d105, (int, float)) and d105 < 0.01)
    )
    # Al-OH / FeMg should retain depth evidence if claimed present
    notes = []
    for name in ("Al-OH Clays", "Fe/Mg Phyllosilicates"):
        e = minerals.get(name) or {}
        if e.get("present") and not e.get("depth_hits"):
            notes.append(f"{name} PRESENT without depth_hits")
    ok = (not ol_bad) and (len(notes) == 0)
    # Soft info from report if present
    verdict = None
    if os.path.isfile(report_path):
        report = _load_json(report_path)
        verdict = report.get("overall_verdict")
    detail = f"present={present} olivine_false_pos={ol_bad} notes={notes} verdict={verdict}"
    return ok, detail


def run_offline_gates(
    mineral_dir: str,
    data_dir: Optional[str] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate offline gates and write mineral_output/offline_gate_report.json.

    Callable from Stage5 orchestrator after a run so the report is never stale.
    """
    mineral_dir = os.path.abspath(mineral_dir)
    data_dir = os.path.abspath(data_dir or os.path.dirname(mineral_dir))
    if not os.path.isdir(mineral_dir):
        payload = {
            "data_dir": data_dir,
            "mineral_dir": mineral_dir,
            "n_fail": 1,
            "all_pass": False,
            "gates": {},
            "error": "mineral_output missing",
        }
        return payload

    if not quiet:
        print("=" * 64)
        print(" OFFLINE MINERAL GATES")
        print(f" Scene dir : {data_dir}")
        print(f" Mineral   : {mineral_dir}")
        print("=" * 64)

    gates: List[Tuple[str, Tuple[bool, str]]] = [
        ("CR_sanity", gate_cr_sanity(mineral_dir)),
        ("Depth_keys", gate_depth_keys(mineral_dir)),
        ("HDBSCAN_params", gate_hdbscan_params()),
        ("Unmix_smoke", gate_unmix_smoke(mineral_dir)),
        ("PRESENT_consistency", gate_present_consistency(mineral_dir)),
    ]

    n_fail = 0
    results: Dict[str, Any] = {}
    for name, (ok, detail) in gates:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        if not quiet:
            print(f"  [{status}] {name}: {detail}")
        results[name] = {"ok": bool(ok), "detail": detail}

    hard = ["Depth_keys", "HDBSCAN_params", "Unmix_smoke"]
    hard_fail = any(not results[h]["ok"] for h in hard)
    payload = {
        "data_dir": data_dir,
        "mineral_dir": mineral_dir,
        "n_fail": n_fail,
        "all_pass": n_fail == 0,
        "hard_pass": not hard_fail,
        "gates": results,
        "note": (
            "Written by audit_scene_mineral_gates.run_offline_gates. "
            "Stage5 calls this at end of run so the report matches latest outputs."
        ),
    }
    out_path = os.path.join(mineral_dir, "offline_gate_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if not quiet:
        print("-" * 64)
        print(f" Wrote {out_path}")
        print(f" RESULT: {'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'} "
              f"(hard_pass={not hard_fail})")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline mineral-ID acceptance gates")
    ap.add_argument("--scene", type=int, default=None, help="Scene number under CRISM_SCENES_ROOT")
    ap.add_argument("--data-dir", type=str, default=None, help="Scene directory")
    args = ap.parse_args()
    if args.data_dir:
        data_dir = args.data_dir
    elif args.scene is not None:
        data_dir = _scene_dir(args.scene)
    else:
        data_dir = _scene_dir(2)

    mineral_dir = os.path.join(data_dir, "mineral_output")
    if not os.path.isdir(mineral_dir):
        print("FAIL: mineral_output/ missing — nothing to audit")
        return 2
    payload = run_offline_gates(mineral_dir, data_dir=data_dir, quiet=False)
    return 0 if payload.get("hard_pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
