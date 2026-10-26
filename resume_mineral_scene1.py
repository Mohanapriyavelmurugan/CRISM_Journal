#!/usr/bin/env python3
"""Resume Stage 5 mineral ID only (S3/S4 already rebuilt with fixed wavelengths)."""
from __future__ import annotations

import os
import sys
import time

SCENE = r"D:\CRISM scenes\scenes\Scene 1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CRISM_DATA_DIR"] = SCENE
os.environ["CRISM_USE_UMAP"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    t0 = time.time()
    print("=" * 70)
    print(" RESUME Stage 5 — mineral ID on fixed-wavelength S4")
    print("=" * 70)
    import Stage5_minerals as s5
    report = s5.run_mineral_identification(
        data_dir=SCENE,
        output_dir=os.path.join(SCENE, "mineral_output"),
    )
    print("\n" + "=" * 70)
    print(f" DONE in {(time.time() - t0) / 60.0:.1f} min")
    if not report:
        print(" FAILED")
        return 1
    print(f" verdict={report.get('overall_verdict')} clusters={report.get('n_clusters')}")
    print(f" clustering_quality={report.get('clustering_quality')}")
    print(f" mineral_confidence={report.get('mineral_detection_confidence')}")
    print(f" dbcv={report.get('dbcv_score')} ({report.get('dbcv_status')})")
    print(f" minerals_present={report.get('minerals_present')}")
    meth = report.get("methodology") or {}
    print(f" cr_source={meth.get('cr_source')} cube_source={meth.get('cube_source')}")
    print(f" latent_dim={meth.get('latent_dim')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
