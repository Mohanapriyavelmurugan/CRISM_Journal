#!/usr/bin/env python3
"""
Compare adapt vs full mineral outputs (held-scene A/B gates).

Usage
-----
  # After adapt run into mineral_output_adapt, compare to existing full:
  python compare_adapt_vs_full.py ^
    --full-dir "D:\\CRISM scenes\\scenes\\Scene 3\\mineral_output" ^
    --adapt-dir "D:\\CRISM scenes\\scenes\\Scene 3\\mineral_output_adapt"

Exit code 0 if hard gates pass; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Set


def _load_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _present_set(proof: Optional[dict], report: Optional[dict]) -> Set[str]:
    if proof and "minerals" in proof:
        return {k for k, v in proof["minerals"].items() if v.get("present")}
    if report and report.get("minerals_present"):
        return set(report["minerals_present"])
    return set()


def _depth_map(proof: Optional[dict]) -> Dict[str, float]:
    out = {}
    if not proof:
        return out
    for name, body in (proof.get("minerals") or {}).items():
        if not body.get("present"):
            continue
        depths = body.get("diagnostic_depths") or body.get("depths") or {}
        # Prefer hot_minus_bland style keys if present
        for k, v in depths.items():
            if isinstance(v, dict):
                if "hot_minus_bland" in v:
                    out[f"{name}:{k}"] = float(v["hot_minus_bland"])
                elif "hot" in v:
                    out[f"{name}:{k}"] = float(v["hot"])
            elif isinstance(v, (int, float)):
                out[f"{name}:{k}"] = float(v)
    return out


def compare(full_dir: str, adapt_dir: str,
            depth_rel_tol: float = 0.35) -> Dict[str, Any]:
    full_rep = _load_json(os.path.join(full_dir, "mineral_identification_report.json"))
    adapt_rep = _load_json(os.path.join(adapt_dir, "mineral_identification_report.json"))
    full_proof = _load_json(os.path.join(full_dir, "mineral_proof.json"))
    adapt_proof = _load_json(os.path.join(adapt_dir, "mineral_proof.json"))
    full_gates = _load_json(os.path.join(full_dir, "offline_gates.json"))
    adapt_gates = _load_json(os.path.join(adapt_dir, "offline_gates.json"))

    # offline_gates may live only inside report
    if full_gates is None and full_rep:
        full_gates = full_rep.get("offline_gates")
    if adapt_gates is None and adapt_rep:
        adapt_gates = adapt_rep.get("offline_gates")

    present_f = _present_set(full_proof, full_rep)
    present_a = _present_set(adapt_proof, adapt_rep)
    new_false = sorted(present_a - present_f)
    missed = sorted(present_f - present_a)

    depths_f = _depth_map(full_proof)
    depths_a = _depth_map(adapt_proof)
    depth_fails: List[str] = []
    for key, vf in depths_f.items():
        if key not in depths_a:
            continue
        va = depths_a[key]
        denom = max(abs(vf), 1e-4)
        rel = abs(va - vf) / denom
        if rel > depth_rel_tol and abs(va - vf) > 0.005:
            depth_fails.append(f"{key}: full={vf:.4f} adapt={va:.4f} rel={rel:.2f}")

    # Unmix recon scale from report if present
    def _recon(rep):
        if not rep:
            return None
        pv = rep.get("physical_validity") or {}
        return pv.get("mean_recon_mse") or rep.get("mean_recon_mse")

    recon_f, recon_a = _recon(full_rep), _recon(adapt_rep)
    recon_ok = True
    recon_note = "n/a"
    if recon_f is not None and recon_a is not None:
        try:
            rf, ra = float(recon_f), float(recon_a)
            # Same order of magnitude; reject 1e12-scale blowups
            recon_ok = (ra < 1e6) and (rf <= 0 or ra / max(rf, 1e-12) < 1e4)
            recon_note = f"full={rf:.4g} adapt={ra:.4g}"
        except (TypeError, ValueError):
            recon_ok = True

    adapt_gates_ok = True
    if isinstance(adapt_gates, dict):
        if "hard_pass" in adapt_gates:
            adapt_gates_ok = bool(adapt_gates["hard_pass"])
        elif "all_pass" in adapt_gates:
            adapt_gates_ok = bool(adapt_gates["all_pass"])

    hard = {
        "adapt_offline_gates": adapt_gates_ok,
        "no_new_false_present": len(new_false) == 0,
        "recon_scale_ok": recon_ok,
    }
    soft = {
        "present_equal_or_stricter": present_a.issubset(present_f),
        "depth_tolerance_ok": len(depth_fails) == 0,
        "missed_present": missed,
    }
    hard_pass = all(hard.values())
    payload = {
        "hard_pass": hard_pass,
        "hard": hard,
        "soft": soft,
        "present_full": sorted(present_f),
        "present_adapt": sorted(present_a),
        "new_false_present": new_false,
        "depth_fails": depth_fails,
        "recon": recon_note,
        "full_verdict": (full_rep or {}).get("overall_verdict"),
        "adapt_verdict": (adapt_rep or {}).get("overall_verdict"),
        "adapt_train_mode": (adapt_rep or {}).get("train_mode"),
        "adapt_encoder_epochs": (adapt_rep or {}).get("encoder_epochs_ran"),
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="A/B adapt vs full mineral gates")
    parser.add_argument("--full-dir", required=True)
    parser.add_argument("--adapt-dir", required=True)
    parser.add_argument("--depth-rel-tol", type=float, default=0.35)
    parser.add_argument(
        "--out-json", type=str, default=None,
        help="Write comparison JSON (default: <adapt-dir>/adapt_vs_full.json)",
    )
    args = parser.parse_args()

    payload = compare(args.full_dir, args.adapt_dir, depth_rel_tol=args.depth_rel_tol)
    out = args.out_json or os.path.join(args.adapt_dir, "adapt_vs_full.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 60)
    print(" ADAPT vs FULL")
    print("=" * 60)
    print(f"  hard_pass: {payload['hard_pass']}")
    for k, v in payload["hard"].items():
        print(f"    {k}: {v}")
    print(f"  present full : {payload['present_full']}")
    print(f"  present adapt: {payload['present_adapt']}")
    print(f"  new false PRESENT: {payload['new_false_present']}")
    print(f"  missed vs full   : {payload['soft']['missed_present']}")
    print(f"  recon: {payload['recon']}")
    if payload["depth_fails"]:
        print("  depth fails:")
        for line in payload["depth_fails"][:12]:
            print(f"    {line}")
    print(f"  wrote: {out}")
    print("=" * 60)
    return 0 if payload["hard_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
