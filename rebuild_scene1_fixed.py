#!/usr/bin/env python3
"""Re-run Stages 1–5 on Scene 1 after wavelength / denoise / mineral fixes."""
from __future__ import annotations

import os
import sys
import time

SCENE = r"D:\CRISM scenes\scenes\Scene 1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CRISM_DATA_DIR"] = SCENE
os.environ["CRISM_USE_UMAP"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print("=" * 70)
    print(" REBUILD Scene 1 — Stages 1→5 (fixed wavelengths + S4 mineral ID)")
    print("=" * 70)

    import Stage1_eda as s1
    print("\n>>> STAGE 1 EDA")
    s1.run_eda(
        data_dir=SCENE,
        output_dir=os.path.join(SCENE, "eda_output"),
    )

    import Stage2_noise as s2
    print("\n>>> STAGE 2 NOISE")
    s2.run_noise_characterisation(
        data_dir=SCENE,
        output_dir=os.path.join(SCENE, "noise_output"),
    )

    import Stage3_physics as s3
    print("\n>>> STAGE 3 PHYSICS")
    s3.run_physics_correction(
        data_dir=SCENE,
        eda_dir=os.path.join(SCENE, "eda_output"),
        noise_dir=os.path.join(SCENE, "noise_output"),
        output_dir=os.path.join(SCENE, "physics_output"),
    )

    import Stage4_denoising as s4
    print("\n>>> STAGE 4 DENOISING")
    s4.run_ml_denoising(
        data_dir=SCENE,
        physics_dir=os.path.join(SCENE, "physics_output"),
        noise_dir=os.path.join(SCENE, "noise_output"),
        output_dir=os.path.join(SCENE, "ml_output"),
    )

    import Stage5_minerals as s5
    print("\n>>> STAGE 5 MINERAL ID")
    report = s5.run_mineral_identification(
        data_dir=SCENE,
        output_dir=os.path.join(SCENE, "mineral_output"),
    )

    print("\n" + "=" * 70)
    print(f" DONE in {(time.time()-t0)/60:.1f} min")
    if report:
        print(f" verdict={report.get('overall_verdict')} clusters={report.get('n_clusters')}")
        print(f" minerals_present={report.get('minerals_present')}")
        print(f" cr_source={report.get('methodology', {}).get('cr_source')}")
    else:
        print(" FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
