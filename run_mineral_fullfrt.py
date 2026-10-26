#!/usr/bin/env python3
"""
Run mineral identification on the new full-height FRT scenes.

Usage
-----
  python run_mineral_fullfrt.py --scene 1
  python run_mineral_fullfrt.py --scene 3 --mode adapt --prior-dir "D:\\CRISM scenes\\models\\shared"
  python run_mineral_fullfrt.py --scene 3 --mode adapt --output-subdir mineral_output_adapt
  python run_mineral_fullfrt.py --all
"""

from __future__ import annotations

import argparse
import os
import sys


SCENES_ROOT = r"D:\CRISM scenes\scenes"


def main():
    parser = argparse.ArgumentParser(description="Run Stage5 mineral ID on full-height FRT scenes")
    parser.add_argument("--scene", type=int, default=1, help="Scene number 1-5")
    parser.add_argument("--all", action="store_true", help="Run all Scene 1-5 sequentially")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override encoder max epochs (default from mode budget)",
    )
    parser.add_argument(
        "--mode", type=str, choices=("full", "adapt"), default=None,
        help="full=from-scratch; adapt=warm-start from shared prior + fine-tune",
    )
    parser.add_argument(
        "--prior-dir", type=str, default=None,
        help="Shared prior directory (for --mode adapt)",
    )
    parser.add_argument(
        "--output-subdir", type=str, default="mineral_output",
        help="Subfolder under the scene dir (use mineral_output_adapt for A/B)",
    )
    args = parser.parse_args()

    os.environ["PYTHONIOENCODING"] = "utf-8"
    if args.mode:
        os.environ["CRISM_TRAIN_MODE"] = args.mode
    if args.prior_dir:
        os.environ["CRISM_SHARED_PRIOR"] = args.prior_dir

    scenes = list(range(1, 6)) if args.all else [args.scene]

    import Stage5_minerals as m

    if args.epochs is not None:
        m.ENCODER_EPOCHS = int(args.epochs)

    results = {}
    for s in scenes:
        data_dir = os.path.join(SCENES_ROOT, f"Scene {s}")
        if not os.path.isdir(data_dir):
            print(f"Missing {data_dir}")
            results[s] = None
            continue
        output_dir = os.path.join(data_dir, args.output_subdir)
        print("\n" + "#" * 70)
        print(f"# FULL-HEIGHT MINERAL ID — Scene {s}  mode={args.mode or 'full'}")
        print("#" * 70)
        os.environ["CRISM_DATA_DIR"] = data_dir
        summary = m.run_mineral_identification(
            data_dir=data_dir,
            output_dir=output_dir,
            mode=args.mode,
            prior_dir=args.prior_dir,
        )
        results[s] = summary

    print("\n" + "=" * 70)
    print(" SUMMARY")
    for s, summary in results.items():
        if not summary:
            print(f"  Scene {s}: FAILED")
        else:
            print(f"  Scene {s}: {summary.get('overall_verdict')} | "
                  f"mode={summary.get('train_mode')} | "
                  f"clusters={summary.get('n_clusters')} | "
                  f"present={summary.get('minerals_present')} | "
                  f"shape={summary.get('cube_shape')}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
