#!/usr/bin/env python3
"""
Build multi-scene shared prior for Stage4 denoise + Stage5 encoder.

Usage
-----
  # A/B prior (Scenes 1+2 only; hold Scene 3 out)
  python pretrain_shared_models.py --scenes 1 2

  # Production prior after A/B passes (Scenes 1+2+3)
  python pretrain_shared_models.py --scenes 1 2 3

Writes to CRISM_SHARED_PRIOR (default D:\\CRISM scenes\\models\\shared):
  stripe_cnn_W640.pth
  spectral_dae_L_B438.pth
  encoder_B438.pth
  prior_meta.json

Does NOT train AbundanceNetwork (always fresh per scene).
Does NOT put Scene 4 (polar) in the prior by default — pass it only if you intend to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from crism_transfer_config import (
    PRETRAIN_PIXELS_PER_SCENE,
    PRETRAIN_STRIPE_PROFILES_PER_SCENE,
    SHARED_PRIOR_DIR,
    ENCODER_CKPT,
    DAE_CKPT,
    STRIPE_CKPT,
    PRIOR_META,
    prior_paths,
    scene_dir,
    FULL_EPOCHS,
    FULL_PATIENCE,
)


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pool_stripe_from_scenes(scenes: List[int], W: int = 640):
    """Pool stripe CNN training profiles with equal per-scene caps."""
    import Stage4_denoising as s4
    from crism_utils import find_label, parse_pds3_label, get_sensor_waves, build_bad_mask

    all_n, all_c = [], []
    meta = {"scenes": [], "profiles_per_scene": {}}
    for s in scenes:
        d = scene_dir(s)
        physics = os.path.join(d, "physics_output")
        noise_json = os.path.join(d, "noise_output", "noise_map.json")
        if not os.path.isfile(noise_json):
            print(f"  [stripe] Scene {s}: missing noise_map — skip")
            continue
        with open(noise_json) as f:
            noise_map = json.load(f)
        npy_files = [
            os.path.join(physics, x) for x in os.listdir(physics)
            if x.endswith("_S3.npy")
        ] if os.path.isdir(physics) else []
        if not npy_files:
            print(f"  [stripe] Scene {s}: no S3 — skip")
            continue
        npy_path = npy_files[0]
        sname = os.path.basename(npy_path).replace("_S3.npy", "")
        if sname not in noise_map:
            print(f"  [stripe] Scene {s}: {sname} not in noise map — skip")
            continue
        cube_s3 = np.load(npy_path).astype(np.float32, copy=False)
        H, Ww, B = cube_s3.shape
        noise = noise_map[sname]
        stripe_bands = noise.get("striped_bands", [])
        if not isinstance(stripe_bands, (list, tuple, np.ndarray)):
            stripe_bands = (
                noise.get("n1_stripes", {}) or {}
            ).get("striped_bands", []) or []
        stripe_bands = [
            int(b) for b in stripe_bands
            if isinstance(b, (int, np.integer)) and 0 <= int(b) < B
        ]
        img_path = None
        for cand in os.listdir(d):
            if cand.lower().endswith(".img") and sname.lower() in cand.lower():
                img_path = os.path.join(d, cand)
                break
        meta_lbl = {}
        if img_path:
            lbl = find_label(img_path)
            if lbl:
                meta_lbl = parse_pds3_label(lbl)
        waves = get_sensor_waves(
            B, meta_lbl.get("wavelengths"), data_dir=d, cube=cube_s3
        )
        bad_mask = build_bad_mask(waves)

        if not stripe_bands:
            print(f"  [stripe] Scene {s}: no stripe bands — skip")
            continue

        cube_classical = s4.classical_destripe(
            cube_s3, bad_mask, only_bands=stripe_bands
        )
        prof_n, prof_c = s4.build_stripe_training_data(
            cube_classical, cube_s3, stripe_bands, bad_mask, waves, Ww
        )
        if prof_n is None or len(prof_n) < 50:
            print(f"  [stripe] Scene {s}: insufficient profiles — skip")
            continue
        cap = PRETRAIN_STRIPE_PROFILES_PER_SCENE
        if len(prof_n) > cap:
            sel = np.random.choice(len(prof_n), cap, replace=False)
            prof_n, prof_c = prof_n[sel], prof_c[sel]

        def _pad(arr):
            arr = np.nan_to_num(arr, nan=0.0).astype(np.float32)
            if arr.shape[1] == W:
                return arr
            if arr.shape[1] > W:
                return arr[:, :W]
            return np.pad(arr, ((0, 0), (0, W - arr.shape[1])))

        all_n.append(_pad(prof_n))
        all_c.append(_pad(prof_c))
        meta["scenes"].append(s)
        meta["profiles_per_scene"][str(s)] = int(len(prof_n))
        print(f"  [stripe] Scene {s}: {len(prof_n)} profiles")

    if not all_n:
        return None, None, meta
    return np.concatenate(all_n, axis=0), np.concatenate(all_c, axis=0), meta


def _pool_dae_train_one_scene(scene: int, init_path: Optional[str],
                              out_tmp: str, epochs: int, patience: int):
    """Fine-tune / train DAE on one scene's normalized cube; return state path."""
    import Stage4_denoising as s4
    from crism_utils import find_label, parse_pds3_label, get_sensor_waves, build_bad_mask

    d = scene_dir(scene)
    physics = os.path.join(d, "physics_output")
    noise_json = os.path.join(d, "noise_output", "noise_map.json")
    if not os.path.isfile(noise_json) or not os.path.isdir(physics):
        return None
    with open(noise_json) as f:
        noise_map = json.load(f)
    npy_files = [os.path.join(physics, x) for x in os.listdir(physics)
                 if x.endswith("_S3.npy")]
    if not npy_files:
        return None
    npy_path = npy_files[0]
    sname = os.path.basename(npy_path).replace("_S3.npy", "")
    if sname not in noise_map:
        return None
    cube = np.load(npy_path).astype(np.float32, copy=False)
    H, W, B = cube.shape
    noise = noise_map[sname]
    img_path = None
    for cand in os.listdir(d):
        if cand.lower().endswith(".img") and sname.lower() in cand.lower():
            img_path = os.path.join(d, cand)
            break
    meta_lbl = {}
    if img_path:
        lbl = find_label(img_path)
        if lbl:
            meta_lbl = parse_pds3_label(lbl)
    waves = get_sensor_waves(B, meta_lbl.get("wavelengths"), data_dir=d, cube=cube)
    bad_mask = build_bad_mask(waves)

    stripe_bands = noise.get("striped_bands", [])
    if not isinstance(stripe_bands, (list, tuple, np.ndarray)):
        stripe_bands = (
            noise.get("n1_stripes", {}) or {}
        ).get("striped_bands", []) or []
    stripe_bands = [
        int(b) for b in stripe_bands
        if isinstance(b, (int, np.integer)) and 0 <= int(b) < B
    ]
    cube_work = cube
    if stripe_bands:
        cube_work = s4.classical_destripe(cube, bad_mask, only_bands=stripe_bands)

    cube_n, means, stds = s4.per_band_norm(cube_work)
    os.makedirs(out_tmp, exist_ok=True)
    model = s4.train_spectral_dae(
        cube_n, bad_mask, B, out_tmp,
        epochs_L=epochs, epochs_S=min(epochs, 80),
        patience=patience, init_state_path=init_path,
        save_name=DAE_CKPT,
    )
    if model is None:
        return None
    return os.path.join(out_tmp, DAE_CKPT)

def _train_encoder_pooled(scenes: List[int], out_dir: str,
                          epochs: int, patience: int) -> Tuple[bool, dict]:
    """Train encoder on equal-capped pools from each scene's mineral_output CR products."""
    import Stage5_minerals as m5

    device = m5._device()
    parts_l2, parts_cr, parts_spm = [], [], []
    meta = {"scenes": [], "pixels_per_scene": {}}
    lib_l2 = lib_cr = lib_spm = None

    for s in scenes:
        d = scene_dir(s)
        mo = os.path.join(d, "mineral_output")
        need = [
            "cube_cr_l2.npy", "cube_cr.npy", "spm_map.npy",
            "training_coords.npy", "lib_spectra_438_cr_l2.npy",
            "lib_spectra_438_cr.npy", "lib_spm.npy",
        ]
        if not all(os.path.isfile(os.path.join(mo, f)) for f in need):
            print(f"  [encoder] Scene {s}: missing mineral_output CR products — skip")
            print(f"            Run Stage5 full on Scene {s} first, or ensure "
                  f"cube_cr*.npy / training_coords.npy exist.")
            continue
        cube_l2 = np.load(os.path.join(mo, "cube_cr_l2.npy"), mmap_mode="r")
        cube_cr = np.load(os.path.join(mo, "cube_cr.npy"), mmap_mode="r")
        spm_map = np.load(os.path.join(mo, "spm_map.npy"), mmap_mode="r")
        coords = np.load(os.path.join(mo, "training_coords.npy"))
        if lib_l2 is None:
            lib_l2 = np.load(os.path.join(mo, "lib_spectra_438_cr_l2.npy"))
            lib_cr = np.load(os.path.join(mo, "lib_spectra_438_cr.npy"))
            lib_spm = np.load(os.path.join(mo, "lib_spm.npy"))

        cap = min(PRETRAIN_PIXELS_PER_SCENE, len(coords))
        if len(coords) > cap:
            # Prefer SPM-active if mask exists
            mask_path = os.path.join(mo, "spm_active_mask.npy")
            if os.path.isfile(mask_path):
                active = np.load(mask_path)
                act_coords = []
                for r, c in coords:
                    if active[int(r), int(c)]:
                        act_coords.append([r, c])
                act_coords = np.asarray(act_coords, dtype=coords.dtype)
                if len(act_coords) >= cap // 2:
                    n_act = min(len(act_coords), int(0.7 * cap))
                    sel_a = np.random.choice(len(act_coords), n_act, replace=False)
                    rem = cap - n_act
                    # Fill from all coords
                    all_sel = np.random.choice(len(coords), min(rem, len(coords)), replace=False)
                    coords_use = np.concatenate(
                        [act_coords[sel_a], coords[all_sel]], axis=0
                    )[:cap]
                else:
                    sel = np.random.choice(len(coords), cap, replace=False)
                    coords_use = coords[sel]
            else:
                sel = np.random.choice(len(coords), cap, replace=False)
                coords_use = coords[sel]
        else:
            coords_use = coords

        scene_l2 = np.asarray(
            cube_l2[coords_use[:, 0], coords_use[:, 1], :], dtype=np.float32
        )
        scene_cr = np.asarray(
            cube_cr[coords_use[:, 0], coords_use[:, 1], :], dtype=np.float32
        )
        scene_spm = np.asarray(
            spm_map[coords_use[:, 0], coords_use[:, 1], :], dtype=np.float32
        )
        parts_l2.append(np.nan_to_num(scene_l2, nan=0.0))
        parts_cr.append(np.nan_to_num(scene_cr, nan=0.0))
        parts_spm.append(np.nan_to_num(scene_spm, nan=0.0))
        meta["scenes"].append(s)
        meta["pixels_per_scene"][str(s)] = int(len(coords_use))
        print(f"  [encoder] Scene {s}: {len(coords_use)} pixels")

    if not parts_l2 or lib_l2 is None:
        return False, meta

    # Library + synth mixes (shared across scenes)
    n_lib = min(m5.LIB_SAMPLE_N, max(1, len(lib_l2) * 3))
    lib_idx = np.random.choice(len(lib_l2), n_lib, replace=True)
    parts_l2.append(lib_l2[lib_idx].astype(np.float32))
    parts_cr.append(lib_cr[lib_idx].astype(np.float32))
    parts_spm.append(np.clip(lib_spm[lib_idx], 0, 1).astype(np.float32))

    synth_l2, synth_cr, synth_spm = [], [], []
    n_synth = m5.SYNTH_MIXES if len(lib_l2) >= 2 else 0
    for _ in range(n_synth):
        i1, i2 = np.random.choice(len(lib_l2), 2, replace=False)
        alpha = np.random.uniform(0.1, 0.9)
        mixed_l2 = m5.l2_normalize(alpha * lib_l2[i1] + (1 - alpha) * lib_l2[i2])
        mixed_cr = alpha * lib_cr[i1] + (1 - alpha) * lib_cr[i2]
        mixed_spm = alpha * lib_spm[i1] + (1 - alpha) * lib_spm[i2]
        synth_l2.append(mixed_l2)
        synth_cr.append(mixed_cr.astype(np.float32))
        synth_spm.append(mixed_spm.astype(np.float32))
    if synth_l2:
        parts_l2.append(np.asarray(synth_l2, dtype=np.float32))
        parts_cr.append(np.asarray(synth_cr, dtype=np.float32))
        parts_spm.append(np.asarray(synth_spm, dtype=np.float32))

    pool = np.nan_to_num(np.concatenate(parts_l2, axis=0), nan=0.0).astype(np.float32)
    pool_cr = np.concatenate(parts_cr, axis=0).astype(np.float32)
    pool_cr, _ = m5.sanitize_cr_array(pool_cr)
    pool_cr = np.clip(np.nan_to_num(pool_cr, nan=0.0), m5.CR_SANITIZE_LO, m5.CR_SANITIZE_HI)
    pool_spm = np.clip(np.nan_to_num(np.concatenate(parts_spm, axis=0), nan=0.0), 0, 1)
    n_pool = len(pool)
    B = pool.shape[1]
    print(f"  [encoder] Pooled {n_pool} spectra, B={B}")

    # Protected diagnostic bands from first usable scene wavelengths
    waves = None
    for s in meta["scenes"]:
        mo = os.path.join(scene_dir(s), "mineral_output")
        # Prefer eda / physics waves via Stage5 context if available
        try:
            from crism_utils import get_sensor_waves, find_label, parse_pds3_label
            d = scene_dir(s)
            imgs = [x for x in os.listdir(d) if x.lower().endswith(".img")]
            if imgs:
                ip = os.path.join(d, imgs[0])
                lbl = find_label(ip)
                ml = parse_pds3_label(lbl) if lbl else {}
                waves = get_sensor_waves(B, ml.get("wavelengths"), data_dir=d)
                break
        except Exception:
            continue
    if waves is None or len(waves) != B:
        waves = np.linspace(1.021, 3.920, B)
    protected = set()
    for wl in m5.SPM_DIAGNOSTIC_WLS:
        bi = int(np.argmin(np.abs(waves - wl)))
        for offset in range(-3, 4):
            if 0 <= bi + offset < B:
                protected.add(bi + offset)

    encoder = m5.SpectralEncoder(
        n_bands=B, latent_dim=m5.LATENT_DIM, n_spm=m5.N_SPM
    ).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    idx = np.arange(n_pool)
    np.random.shuffle(idx)
    n_val = max(64, int(0.10 * n_pool))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    best_val = float("inf")
    patience_ctr = 0
    save_path = os.path.join(out_dir, ENCODER_CKPT)
    t0 = time.time()

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        encoder.train()
        tr_losses = []
        for start in range(0, len(train_idx), m5.ENCODER_BATCH):
            bi = train_idx[start:start + m5.ENCODER_BATCH]
            if len(bi) < 4:
                continue
            specs = pool[bi]
            spm_b = pool_spm[bi]
            v1 = np.stack([m5.augment_spectrum(s, protected) for s in specs], 0)
            v2 = np.stack([m5.augment_spectrum(s, protected) for s in specs], 0)
            t1 = torch.from_numpy(v1).float().unsqueeze(1).to(device)
            t2 = torch.from_numpy(v2).float().unsqueeze(1).to(device)
            spm_t = torch.from_numpy(spm_b).float().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z1, z2 = encoder(t1), encoder(t2)
                loss_c = m5.nt_xent_loss(z1, z2, temperature=0.07)
                loss_s = F.mse_loss(encoder.predict_spm(z1), spm_t)
                recon = encoder.reconstruct(z1)
                target = torch.from_numpy(specs).float().to(device)
                prot = sorted(protected) if protected else list(range(0, B, 8))
                prot_t = torch.tensor(prot, device=device, dtype=torch.long)
                loss_r = F.mse_loss(
                    recon.index_select(1, prot_t), target.index_select(1, prot_t)
                )
                loss = loss_c + 0.1 * loss_s + 0.2 * loss_r
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(loss.detach().cpu()))
        sched.step()

        encoder.eval()
        va = []
        with torch.no_grad():
            for start in range(0, len(val_idx), m5.ENCODER_BATCH):
                bi = val_idx[start:start + m5.ENCODER_BATCH]
                if len(bi) < 2:
                    continue
                specs = pool[bi]
                spm_b = pool_spm[bi]
                v1 = np.stack([m5.augment_spectrum(s, protected) for s in specs], 0)
                v2 = np.stack([m5.augment_spectrum(s, protected) for s in specs], 0)
                t1 = torch.from_numpy(v1).float().unsqueeze(1).to(device)
                t2 = torch.from_numpy(v2).float().unsqueeze(1).to(device)
                spm_t = torch.from_numpy(spm_b).float().to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    z1, z2 = encoder(t1), encoder(t2)
                    loss_c = m5.nt_xent_loss(z1, z2, temperature=0.07)
                    loss_s = F.mse_loss(encoder.predict_spm(z1), spm_t)
                    recon = encoder.reconstruct(z1)
                    target = torch.from_numpy(specs).float().to(device)
                    prot = sorted(protected) if protected else list(range(0, B, 8))
                    prot_t = torch.tensor(prot, device=device, dtype=torch.long)
                    loss_r = F.mse_loss(
                        recon.index_select(1, prot_t), target.index_select(1, prot_t)
                    )
                    va.append(float((loss_c + 0.1 * loss_s + 0.2 * loss_r).cpu()))
        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        mean_va = float(np.mean(va)) if va else mean_tr
        if mean_va < best_val - 1e-5:
            best_val = mean_va
            patience_ctr = 0
            torch.save(encoder.state_dict(), save_path)
        else:
            patience_ctr += 1
        if epoch % 10 == 0 or patience_ctr == 0:
            print(f"  [encoder] ep {epoch:03d}/{epochs-1}  "
                  f"train={mean_tr:.4f} val={mean_va:.4f} best={best_val:.4f}")
        if patience_ctr >= patience:
            print(f"  [encoder] early stop at ep {epoch}")
            break

    meta["encoder_best_val"] = best_val
    meta["encoder_seconds"] = time.time() - t0
    print(f"  [encoder] done in {meta['encoder_seconds']:.1f}s → {save_path}")
    return os.path.isfile(save_path), meta


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scene pretrain for CRISM denoise + encoder prior"
    )
    parser.add_argument(
        "--scenes", type=int, nargs="+", default=[1, 2],
        help="Scene numbers to pool (default: 1 2 for A/B; exclude held-out)",
    )
    parser.add_argument(
        "--prior-dir", type=str, default=None,
        help=f"Output directory (default: {SHARED_PRIOR_DIR})",
    )
    parser.add_argument(
        "--skip-stripe", action="store_true",
        help="Skip stripe CNN pretrain",
    )
    parser.add_argument(
        "--skip-dae", action="store_true",
        help="Skip spectral DAE pretrain",
    )
    parser.add_argument(
        "--skip-encoder", action="store_true",
        help="Skip encoder pretrain",
    )
    parser.add_argument(
        "--encoder-only-from-checkpoints", action="store_true",
        help="If mineral CR products missing, average existing encoder_B438.pth "
             "weights from scene mineral_output (fallback, weaker)",
    )
    args = parser.parse_args()
    _set_seed(42)

    out_dir = args.prior_dir or SHARED_PRIOR_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = prior_paths(out_dir)
    print("=" * 65)
    print("  MULTI-SCENE SHARED PRIOR")
    print(f"  Scenes : {args.scenes}")
    print(f"  Output : {out_dir}")
    print("=" * 65)

    if 4 in args.scenes:
        print("  [WARN] Scene 4 is polar (+82N). Prefer excluding it from the "
              "shared prior unless you have run a polar-specific A/B.")

    meta = {
        "scenes": list(args.scenes),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "components": {},
    }
    t_all = time.time()

    # --- Stripe ---
    if not args.skip_stripe:
        print("\n--- Stripe CNN multi-scene pretrain ---")
        import Stage4_denoising as s4
        pn, pc, sm = _pool_stripe_from_scenes(args.scenes, W=640)
        meta["components"]["stripe_pool"] = sm
        if pn is not None and len(pn) >= 200:
            model = s4.train_stripe_cnn(
                pn, pc, 640, out_dir,
                epochs=FULL_EPOCHS["stripe"],
                patience=FULL_PATIENCE["stripe"],
                save_name=STRIPE_CKPT,
            )
            meta["components"]["stripe"] = {
                "ok": model is not None,
                "path": paths["stripe"],
                "n_profiles": int(len(pn)),
            }
        else:
            print("  [stripe] Insufficient pooled data — copying Scene 1 checkpoint if present")
            src = os.path.join(scene_dir(args.scenes[0]), "ml_output", STRIPE_CKPT)
            if os.path.isfile(src):
                import shutil
                shutil.copy2(src, paths["stripe"])
                meta["components"]["stripe"] = {"ok": True, "copied_from": src}
            else:
                meta["components"]["stripe"] = {"ok": False}

    # --- DAE: sequential multi-scene (warm-start chain) ---
    if not args.skip_dae:
        print("\n--- Spectral DAE multi-scene pretrain (sequential) ---")
        import shutil
        init = None
        tmp = os.path.join(out_dir, "_dae_tmp")
        os.makedirs(tmp, exist_ok=True)
        ok_any = False
        for s in args.scenes:
            print(f"  DAE pass Scene {s}...")
            p = _pool_dae_train_one_scene(
                s, init, tmp,
                epochs=FULL_EPOCHS["dae_L"],
                patience=FULL_PATIENCE["dae"],
            )
            if p and os.path.isfile(p):
                init = p
                ok_any = True
                print(f"  DAE Scene {s} OK → continue")
            else:
                print(f"  DAE Scene {s} skipped/failed")
        if ok_any and init:
            shutil.copy2(init, paths["dae"])
            meta["components"]["dae"] = {"ok": True, "path": paths["dae"]}
        else:
            src = os.path.join(scene_dir(args.scenes[0]), "ml_output", DAE_CKPT)
            if os.path.isfile(src):
                shutil.copy2(src, paths["dae"])
                meta["components"]["dae"] = {"ok": True, "copied_from": src}
            else:
                meta["components"]["dae"] = {"ok": False}

    # --- Encoder ---
    if not args.skip_encoder:
        print("\n--- Encoder multi-scene pretrain ---")
        ok, em = _train_encoder_pooled(
            args.scenes, out_dir,
            epochs=FULL_EPOCHS["encoder"],
            patience=FULL_PATIENCE["encoder"],
        )
        meta["components"]["encoder"] = {"ok": ok, **em}
        if not ok and args.encoder_only_from_checkpoints:
            # Fallback: copy Scene 1 encoder (not a true multi-scene train)
            import shutil
            src = os.path.join(scene_dir(args.scenes[0]), "mineral_output", ENCODER_CKPT)
            if os.path.isfile(src):
                shutil.copy2(src, paths["encoder"])
                meta["components"]["encoder"] = {
                    "ok": True,
                    "fallback_copy": src,
                    "note": "Not multi-scene trained — mineral_output CR products missing",
                }
                print(f"  [encoder] Fallback copied {src}")

    meta["elapsed_seconds"] = time.time() - t_all
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    print("\n" + "=" * 65)
    print(f"  PRIOR COMPLETE  ({meta['elapsed_seconds']/60:.1f} min)")
    print(f"  Meta: {paths['meta']}")
    for k, v in meta["components"].items():
        if isinstance(v, dict) and "ok" in v:
            print(f"  {k}: {'OK' if v.get('ok') else 'FAIL'}")
    print("=" * 65)
    # Exit 0 if encoder OK (critical for adapt Stage5); warn on denoise fails
    enc_ok = meta["components"].get("encoder", {}).get("ok", False)
    return 0 if enc_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
