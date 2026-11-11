#!/usr/bin/env python3
r"""
fase2_calibration.py
====================
FASE 2 -- Kalibrasi LiDAR-kamera: SUSUN otomatis `calib_config.json` untuk bag yang sedang diproses.
(Menggantikan fase2_calib_check.py. Tidak ada kalibrasi ICP baru; hasil ICP lama dipakai ulang.)

INPUT   : <out>/camera_info.json            (dibuat Fase 1 dari rosbag: K, D, ukuran gambar)
          rig_extrinsics.json   (T_lidar_to_cam hasil ICP, disimpan SEKALI di folder kode)
PROSES  : calib_config.json = K, D, IMG_W, IMG_H dari camera_info bag ini  +  T_lidar_to_cam dari rig_extrinsics.
          T adalah sifat pemasangan fisik sensor (bukan data bag) -> tidak berubah antar rekaman
          selama sensor tidak dipasang ulang. calib_config_koreksi.json (Fase 2b) DIABAIKAN.
OUTPUT  : <out>/calib/calib_config.json   (skema sama dengan file lama) + checkpoints/fase2_report.json
CHECKPOINT: T 4x4 rigid; K cocok camera_info; dan -- pengganti Fase 2b -- kedalaman titik LiDAR yang
          diproyeksikan cocok dengan depth ZED pada ~N frame (peringatan bila T tampak tidak cocok bag ini).

Bila <out>/calib/calib_config.json SUDAH ada, dipakai apa adanya (tidak ditimpa) kecuali --force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import Checker, Paths, resolve_by_name

DEFAULT_RIG = Path(__file__).resolve().parent / "rig_extrinsics.json"

# cek konsistensi depth (sama dengan ide Fase 2b: rasio depth ZED / z-kamera LiDAR ~ 1)
N_FRAMES = 20
Z_RANGE = (3.0, 16.0)
MIN_VALID_PTS = 800
MAX_ABS_LOG_RATIO = 0.15       # median |log(depth/z)| antar-frame; ~ selisih 16%  (TEBAKAN awal, sesuaikan dgn data)
MAX_LOSS = 0.15                # rata-rata min(|log rasio|, 0.3)
PITCH_GRID_DEG = np.arange(-8.0, 8.01, 0.5)   # pencarian koreksi pitch kamera (seperti Fase 2b)
MAX_ABS_PITCH_DEG = 4.0        # |koreksi pitch optimal| > ini -> T tampak tidak cocok bag ini (Fase 2b nyata: ~2 deg, tak berpengaruh)
MAX_PTS_PER_FRAME = 8000


def build_calib_config(camera_info, rig):
    K = np.array(camera_info["K"], float).reshape(3, 3)
    return {
        "T_lidar_to_cam": rig["T_lidar_to_cam"],
        "K": K.tolist(),
        "D": list(camera_info.get("D", [0, 0, 0, 0, 0])),
        "IMG_W": int(camera_info["width"]),
        "IMG_H": int(camera_info["height"]),
        "reference_frame_idx": rig.get("reference_frame_idx"),
        "note": ("DISUSUN OTOMATIS Fase 2: K, D, ukuran gambar dari camera_info.json bag ini; "
                 "T_lidar_to_cam dari rig_extrinsics.json (hasil ICP, dipakai langsung tanpa tuning). "
                 + rig.get("note", "")),
    }


def _loss_at(pc, depth, K, W, H, r_med=False):
    z = pc[:, 2]
    m = (z > Z_RANGE[0]) & (z < Z_RANGE[1])
    uv = (K @ pc[m].T).T
    u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
    inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    d = depth[v[inb].astype(int), u[inb].astype(int)]
    good = np.isfinite(d) & (d > 0.3)
    r = np.log(d[good] / z[m][inb][good])
    if len(r) == 0:
        return 1.0, 0, 0.0
    return float(np.mean(np.minimum(np.abs(r), 0.3))), int(len(r)), float(np.median(r))


def depth_consistency(paths, T, K, W, H, n_frames=N_FRAMES):
    """Per frame: loss/median log(depth_ZED / z_LiDAR) pada T apa adanya, DAN koreksi pitch kamera optimal
    (rx, derajat) yang meminimalkan loss -- ide Fase 2b. rx ~ 0 berarti T cocok dengan depth bag ini."""
    from scipy.spatial.transform import Rotation as Rot
    df = pd.read_csv(paths.sync_csv)
    df = df[df["lidar_path"].notna() & df["depth_data"].notna()].reset_index(drop=True)
    if len(df) == 0:
        return []
    pick = df.iloc[np.linspace(0, len(df) - 1, min(n_frames, len(df))).astype(int)]
    rng = np.random.default_rng(0)
    Rs = [(rx, Rot.from_euler("xyz", [rx, 0, 0], degrees=True).as_matrix()) for rx in PITCH_GRID_DEG]
    out = []
    for _, row in pick.iterrows():
        lp = resolve_by_name(paths.out, row["lidar_path"], "lidar")
        dp = resolve_by_name(paths.out, row["depth_data"], "depth")
        if lp is None or dp is None or not lp.exists() or not dp.exists():
            continue
        xyz = np.load(lp)[:, :3].astype(float)
        pc0 = (T[:3, :3] @ xyz.T).T + T[:3, 3]
        pc0 = pc0[(pc0[:, 2] > 1.0) & (pc0[:, 2] < 40.0)]
        if len(pc0) > MAX_PTS_PER_FRAME:
            pc0 = pc0[rng.choice(len(pc0), MAX_PTS_PER_FRAME, replace=False)]
        depth = np.load(dp)
        loss0, n0, med0 = _loss_at(pc0, depth, K, W, H)
        if n0 < MIN_VALID_PTS * MAX_PTS_PER_FRAME / 25000:      # skala dgn subsampling
            continue
        losses = [_loss_at((R @ pc0.T).T, depth, K, W, H)[0] for _, R in Rs]
        out.append({"idx": int(row["idx"]), "n": n0, "median": med0, "loss": loss0,
                    "rx": float(Rs[int(np.argmin(losses))][0]), "loss_best": float(min(losses))})
    return out


def run(out, rig_path=None, force=False, strict=False):
    paths = Paths(out)
    rig_path = Path(rig_path) if rig_path else DEFAULT_RIG
    paths.ensure("calib", "checkpoints")
    ck = Checker("FASE 2 (kalibrasi)", paths.checkpoints / "fase2_report.json", strict=strict)

    # 1) susun calib_config.json bila belum ada
    if paths.calib_json.exists() and not force:
        ck.info("calib_config.json", f"sudah ada -> dipakai apa adanya ({paths.calib_json}); --force untuk disusun ulang")
    else:
        if not ck.ok("camera_info.json ada (dari Fase 1)", paths.camera_info.exists(), str(paths.camera_info)):
            ck.finish()
        if not ck.ok("rig_extrinsics.json ada", rig_path.exists(), str(rig_path)):
            ck.finish()
        with open(paths.camera_info) as f:
            ci = json.load(f)
        with open(rig_path) as f:
            rig = json.load(f)
        cfg = build_calib_config(ci, rig)
        with open(paths.calib_json, "w") as f:
            json.dump(cfg, f, indent=2)
        ck.info("calib_config.json", f"DISUSUN dari {paths.camera_info.name} + {rig_path.name} -> {paths.calib_json}")

    # 2) validasi struktur
    with open(paths.calib_json) as f:
        cal = json.load(f)
    T, K = np.array(cal["T_lidar_to_cam"], float), np.array(cal["K"], float)
    W, H = int(cal.get("IMG_W", 640)), int(cal.get("IMG_H", 360))
    ck.ok("T_lidar_to_cam 4x4, baris terakhir [0,0,0,1]", T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1]))
    if T.shape == (4, 4):
        R = T[:3, :3]
        ck.ok("rotasi ortonormal (det=+1)", np.allclose(R @ R.T, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1) < 1e-3)
    ck.ok("K 3x3", K.shape == (3, 3))
    if paths.camera_info.exists() and K.shape == (3, 3):
        with open(paths.camera_info) as f:
            Kc = np.array(json.load(f)["K"], float).reshape(3, 3)
        idx = ([0, 1, 0, 1], [0, 1, 2, 2])
        rel = np.abs(K[idx] - Kc[idx]) / np.abs(Kc[idx])
        ck.ok("K sesuai camera_info.json bag ini (fx,fy,cx,cy dalam 1%)", bool((rel < 0.01).all()),
              f"selisih maks {rel.max() * 100:.2f}%", fatal=False)
    ck.info("calib_config_koreksi.json", "diabaikan (keputusan Fase 2b)")

    # 3) konsistensi dengan depth ZED bag ini (pengganti Fase 2b)
    if paths.sync_csv.exists() and T.shape == (4, 4) and K.shape == (3, 3):
        res = depth_consistency(paths, T, K, W, H)
        if len(res) < 3:
            ck.ok("Konsistensi LiDAR vs depth ZED", False, f"hanya {len(res)} frame cukup titik -> tak bisa diverifikasi", fatal=False)
        else:
            med = float(np.median([r["median"] for r in res]))
            loss = float(np.median([r["loss"] for r in res]))
            rng = float(np.ptp([r["median"] for r in res]))
            ck.ok("T cocok dengan depth ZED bag ini: |median log(depth/LiDAR)| kecil", abs(med) <= MAX_ABS_LOG_RATIO,
                  f"median {med:+.3f} (rasio ~{np.exp(med):.2f}), {len(res)} frame; ambang {MAX_ABS_LOG_RATIO}", fatal=False)
            ck.ok("T cocok dengan depth ZED bag ini: loss rendah", loss <= MAX_LOSS,
                  f"loss {loss:.3f}; ambang {MAX_LOSS}", fatal=False)
            rx = float(np.median([r["rx"] for r in res]))
            iqr = float(np.percentile([r["rx"] for r in res], 75) - np.percentile([r["rx"] for r in res], 25))
            ck.ok("Koreksi pitch optimal terhadap depth ZED kecil", abs(rx) <= MAX_ABS_PITCH_DEG,
                  f"median {rx:+.1f} deg (IQR {iqr:.1f}); ambang {MAX_ABS_PITCH_DEG} deg. Jauh di atas ambang = "
                  f"sensor mungkin dipasang ulang -> perbarui rig_extrinsics.json", fatal=False)
            ck.info("Sebaran median log-rasio antar-frame", f"{rng:.3f}")
    ck.finish()
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig-calib", help="default: rig_extrinsics.json di folder kode")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    run(a.out, a.rig_calib, a.force, a.strict)
