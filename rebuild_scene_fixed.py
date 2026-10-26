#!/usr/bin/env python3
"""Rebuild Stages 1–5 for a chosen scene with fixed wavelengths + S4 mineral ID."""
from __future__ import annotations

import argparse
import os
import sys
import time

SCENES_ROOT = r"D:\CRISM scenes\scenes"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=2, help="Scene number 1-5")
    args = parser.parse_args()

    scene = os.path.join(SCENES_ROOT, f"Scene {args.scene}")
    if not os.path.isdir(scene):
        print(f"Missing {scene}")
        return 1

    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["CRISM_DATA_DIR"] = scene
    os.environ["CRISM_USE_UMAP"] = "0"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    print("=" * 70)
    print(f" REBUILD Scene {args.scene} — Stages 1→5 (fixed wavelengths + S4 mineral ID)")
    print("=" * 70)

    import Stage1_eda as s1
    print("\n>>> STAGE 1 EDA")
    s1.run_eda(data_dir=scene, output_dir=os.path.join(scene, "eda_output"))

    import Stage2_noise as s2
    print("\n>>> STAGE 2 NOISE")
    s2.run_noise_characterisation(
        data_dir=scene, output_dir=os.path.join(scene, "noise_output")
    )

    import Stage3_physics as s3
    print("\n>>> STAGE 3 PHYSICS")
    s3.run_physics_correction(
        data_dir=scene,
        eda_dir=os.path.join(scene, "eda_output"),
        noise_dir=os.path.join(scene, "noise_output"),
        output_dir=os.path.join(scene, "physics_output"),
    )

    import Stage4_denoising as s4
    print("\n>>> STAGE 4 DENOISING (mode=full)")
    s4.run_ml_denoising(
        data_dir=scene,
        physics_dir=os.path.join(scene, "physics_output"),
        noise_dir=os.path.join(scene, "noise_output"),
        output_dir=os.path.join(scene, "ml_output"),
        mode="full",
    )

    import Stage5_minerals as s5
    print("\n>>> STAGE 5 MINERAL ID (mode=full)")
    report = s5.run_mineral_identification(
        data_dir=scene,
        output_dir=os.path.join(scene, "mineral_output"),
        mode="full",
    )

    print("\n" + "=" * 70)
    print(f" DONE Scene {args.scene} in {(time.time() - t0) / 60.0:.1f} min")
    if not report:
        print(" FAILED")
        return 1
    print(f" verdict={report.get('overall_verdict')} clusters={report.get('n_clusters')}")
    print(f" minerals_present={report.get('minerals_present')}")
    meth = report.get("methodology") or {}
    print(f" cr_source={meth.get('cr_source')} cube_source={report.get('cube_source')}")
    print(f" shape={report.get('cube_shape')} latent_dim={meth.get('latent_dim')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
