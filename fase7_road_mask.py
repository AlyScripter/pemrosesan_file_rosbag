#!/usr/bin/env python3
r"""
fase7_road_mask.py
==================
Fase 7 - Fusi mask jalan BEV (LiDAR + kamera). Sisipan BEV ROAD. Menggantikan `build_road_masks.py`.
Algoritma fusi = `road_lib.py` + `camera_fusion_lib.py` (kode asli yang sudah tervalidasi, tidak diubah; hanya pesan peringatan
label di road_lib.Sequence dibuang karena label kini dibaca dari Fase 4).

INPUT : fase4_segmentasi/points/<idx>.npy (x,y,z,r,g,b,class_id)  |  fase6_seg_cam/<idx>.png  |  meta/odom.csv (Fase 1)
        sync_table.csv + depth/ (Fase 1)  |  calib/calib_config.json (Fase 2)  |  lidar/ (hanya timestamp)
OUTPUT: fase7_road_mask/masks/<idx:06d>.npz  (mask_fused/lidar/cam, lidar_hits, cam_road_hits, cam_non_hits, obstacle_hits,
        ground_plane, x_range, y_range, res)  +  fase7_road_mask/report.csv, fase7_done.json, checkpoints/fase7_report.json
CHECKPOINT (otomatis): idx sync_table == urutan LiDAR; semua frame ber-pose & ber-label punya npz (gagal <= 2%); tak ada mask kosong
  (> 2%); kalibrasi cocok dengan depth ZED; cakupan segmentasi kamera; bidang tanah (sumber fit vs cadangan); recall jalur ego
  lampau/depan >= 0,90; konflik dengan rintangan vertikal <= 0,05 (target kode asli).
GOAL: mask jalan mentah per frame + seluruh bukti dasar (hits LiDAR/kamera/rintangan) untuk Fase 8 (polygon berbasis bukti).

Keputusan penyatuan dibanding build_road_masks.py:
  * Kalibrasi = calib_config.json Fase 2 LANGSUNG (T_lidar_to_cam dipakai apa adanya, sama dengan Fase 3). Tebak-4-arah dibuang;
    sebagai gantinya kecocokan LiDAR vs depth ZED diukur & dilaporkan sebagai checkpoint.
  * Label LiDAR dibaca langsung dari Fase 4 (kolom 6); tanpa monkey-patch.
  * Frame diproses paralel (potongan frame berurutan per proses). Hasil per frame sama dengan serial kecuali kasus langka
    "bidang tanah cadangan" (urutan cadangan bergantung frame sebelumnya).
  * Frame tanpa pose valid (awal rekaman sebelum ZED tracking) tidak dihitung -> tidak punya mask/scene; streaming menahan jalan terakhir.
  * Tanpa PNG/plot (hanya --debug-plots di Fase 8).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import camera_fusion_lib as cf
import road_lib as rl
from common import Checker, Paths, Timer

try:
    from tqdm import tqdm
except ImportError:                             # pragma: no cover
    def tqdm(x, **kw):
        return x

LABEL_COL = 6
MAX_FAIL_FRAC = 0.02
MAX_EMPTY_FRAC = 0.02
MIN_CALIB_AGREE = 0.50          # target_calib_agree kode asli
MIN_CHUNK = 40                  # frame minimum per potongan (pemanasan cache jendela ~60 frame)


# ---------------------------------------------------------------------------------------------------------------------
# Sequence dengan label Fase 4
# ---------------------------------------------------------------------------------------------------------------------
class Fase4Sequence(rl.Sequence):
    """Scan = fase4_segmentasi/points/<idx>.npy (x,y,z,r,g,b,class_id). Frame tanpa berkas = kosong (bukan error)."""

    def get_scan(self, i):
        p = self.cfg.label_dir / f"{i:06d}.npy"
        if not p.exists():
            return np.zeros((0, 3), np.float32), np.zeros((0,), np.int16)
        arr = np.load(p, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] <= LABEL_COL:
            raise ValueError(f"{p.name}: bentuk {arr.shape}, butuh (N, >={LABEL_COL + 1}) [x,y,z,r,g,b,class_id]")
        xyz = arr[:, :3].astype(np.float32)
        lab = np.rint(arr[:, LABEL_COL]).astype(np.int16)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        ok = np.isfinite(xyz).all(1) & (r >= self.cfg.min_range) & (r <= self.cfg.max_range)
        return xyz[ok], lab[ok]


def build_context(out):
    """Semua objek fusi dari folder output Fase 1-6. Dipakai di proses utama dan tiap worker."""
    out = Path(out)
    d7 = out / "fase7_road_mask"
    cfg = rl.Config(root=out, label_dir=out / "fase4_segmentasi" / "points", out_dir=d7)
    f2 = cf.Fase2Config(root=out, seg_dir=out / "fase6_seg_cam", out_dir=d7)
    seq = Fase4Sequence(cfg)
    odom = rl.load_odom(cfg.odom_csv, cfg.drop_identity_prefix)
    poses, valid = rl.build_lidar_poses(seq.t, odom, cfg)
    sync = cf.load_sync(f2)
    calib = cf.Calibration.from_json(f2.calib_json)
    calib.M, calib.M_name = calib.T, "T_lidar_to_cam Fase 2 (apa adanya)"
    return cfg, f2, seq, poses, valid, sync, calib, cf.FusionContext(cfg, f2, seq, poses, valid, sync, calib)


# ---------------------------------------------------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------------------------------------------------
_CTX = None


def _init_worker(out):
    global _CTX
    _CTX = build_context(out)[-1]


def _run_chunk(frames):
    ctx = _CTX
    rows, failed = [], []
    mdir = ctx.cfg.out_dir / "masks"
    for i in frames:
        try:
            res = ctx.compute(int(i))
            cf.save_masks(ctx, res, mdir / f"{i:06d}.npz")
            rows.append(res["row"])
        except Exception as e:                          # noqa: BLE001
            failed.append((int(i), repr(e)))
    return rows, failed


def make_chunks(frames, workers):
    n = len(frames)
    k = max(1, min(workers * 2, n // MIN_CHUNK))
    bounds = np.linspace(0, n, k + 1).round().astype(int)
    return [frames[bounds[a]:bounds[a + 1]] for a in range(k) if bounds[a + 1] > bounds[a]]


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------------------------------------
def calibration_agreement(seq, sync, calib, n=12):
    """Kecocokan jarak LiDAR (via T Fase 2) vs depth ZED pada n frame tersebar. Sama dengan ukuran kode asli."""
    from PIL import Image
    have = sync.index[sync["rgb_ok"] & sync["depth_ok"]]
    if len(have) == 0:
        return float("nan"), 0
    pick = [int(i) for i in have[np.unique(np.linspace(0, len(have) - 1, min(n, len(have))).round().astype(int))]]
    W, H = Image.open(sync.at[pick[0], "rgb_file"]).size
    K = calib.K_for(W, H)
    n_in = n_ag = 0
    for j in pick:
        pts, _ = seq.get_scan(j)
        if len(pts) == 0:
            continue
        s = cf.score_candidate(K, calib.M, pts, np.load(sync.at[j, "depth_file"]).astype(np.float32))
        n_in += s["n_in"]
        n_ag += s["n_agree"]
    return (n_ag / n_in if n_in >= 200 else float("nan")), n_in


def run_checkpoints(out, S, failed, n_expected, seq, poses, valid, sync, calib, strict=False):
    d7 = Path(out) / "fase7_road_mask"
    f2 = cf.Fase2Config(root=out)
    chk = Checker("Fase 7", report_path=Paths(out).checkpoints / "fase7_report.json", strict=strict)
    n_npz = len(list((d7 / "masks").glob("??????.npz")))
    chk.ok("jumlah npz == frame ber-pose & ber-label", n_npz == n_expected - len(failed), f"{n_npz} dari {n_expected} (gagal {len(failed)})")
    chk.ok(f"frame gagal <= {MAX_FAIL_FRAC:.0%}", len(failed) <= MAX_FAIL_FRAC * max(n_expected, 1),
           f"{len(failed)}; contoh: {failed[:2]}")
    if len(S) == 0:
        chk.ok("ada hasil", False, "tidak ada frame berhasil")
        chk.finish()
        return
    empty = float((S["cells_fused"] == 0).mean())
    chk.ok(f"mask kosong <= {MAX_EMPTY_FRAC:.0%}", empty <= MAX_EMPTY_FRAC, f"{empty:.1%}")
    # alignment idx sync_table <-> urutan LiDAR (kunci seluruh pipeline)
    if "lidar_t" in sync.columns:
        idxs = sync.index.to_numpy()
        ok_idx = idxs[idxs < len(seq)]
        dt = np.abs(sync.loc[ok_idx, "lidar_t"].to_numpy(float) - seq.t[ok_idx])
        chk.ok("idx sync_table == urutan timestamp LiDAR", float(dt.max()) < 1e-3 and len(ok_idx) == len(idxs),
               f"selisih maks {dt.max():.2e} s")
    agree, n_in = calibration_agreement(seq, sync, calib)
    chk.ok(f"kalibrasi cocok dgn depth ZED >= {MIN_CALIB_AGREE:.0%}", bool(agree >= MIN_CALIB_AGREE), f"{agree:.1%} dari {n_in} titik", fatal=False)
    cam_ok = float((S["frames_cam"] > 0).mean())
    ver = cf.verdict(S, f2, calib_agree=agree if np.isfinite(agree) else 0.0, cam_frames_ok=cam_ok)
    for _, r in ver.iterrows():
        if r["hasil"] in ("LULUS", "PERIKSA"):
            chk.ok(r["cek"], r["hasil"] == "LULUS", f"{r['nilai']} (target {r['target']})", fatal=False)
        elif r["hasil"] == "INFO":
            chk.info(r["cek"], f"{r['nilai']}" + ("" if pd.isna(r["target"]) else f" / {r['target']}"))
    src = S["plane_src"].astype(str).str.split(":").str[0].value_counts().to_dict()
    chk.ok("bidang tanah dari fit sendiri >= 50% frame", S["plane_ok"].mean() >= 0.5,
           f"plane_ok {S['plane_ok'].mean():.1%}; sumber {src}; ground_z median {S['ground_z'].median():.2f} m", fatal=False)
    chk.info("frame: LiDAR/kamera dipakai per jendela (median)", f"{S['frames_lidar'].median():.0f} / {S['frames_cam'].median():.0f}")
    chk.finish()


# ---------------------------------------------------------------------------------------------------------------------
def run(out, force=False, strict=False, workers=None, max_frames=None):
    t_all = time.time()
    out = Path(out)
    d7 = out / "fase7_road_mask"
    marker = d7 / "fase7_done.json"
    if marker.exists() and not force:
        print(f"[Fase 7] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    for need in (out / "fase4_segmentasi" / "fase4_done.json", out / "fase6_seg_cam" / "fase6_done.json",
                 out / "meta" / "odom.csv", out / "sync_table.csv", out / "calib" / "calib_config.json"):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    cfg, f2, seq, poses, valid, sync, calib, _ = build_context(out)
    pts_dir = cfg.label_dir
    frames = [i for i in range(len(seq)) if valid[i] and (pts_dir / f"{i:06d}.npy").exists()]
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise SystemExit("Tidak ada frame dengan pose valid & berkas Fase 4")
    st = rl.pose_valid_stats(valid)
    print(f"[Fase 7] {len(seq)} scan | pose valid {int(valid.sum())} (valid pertama idx {st['first_valid']}) | "
          f"diproses {len(frames)} frame | kalibrasi: {calib.M_name}")
    if d7.exists():
        for p in (d7 / "masks").glob("*.npz"):
            p.unlink()
    (d7 / "masks").mkdir(parents=True, exist_ok=True)

    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    chunks = make_chunks(frames, workers)
    rows, failed = [], []
    with Timer("Fase 7 fusi mask"):
        if workers == 1 or len(chunks) == 1:
            _init_worker(out)
            for ch in tqdm(chunks, desc="Fase 7"):
                r, f = _run_chunk(ch)
                rows += r
                failed += f
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(out),)) as ex:
                for r, f in tqdm(ex.map(_run_chunk, chunks), total=len(chunks), desc="Fase 7"):
                    rows += r
                    failed += f
    S = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    S.to_csv(d7 / "report.csv", index=False)
    run_checkpoints(out, S, sorted(failed), len(frames), seq, poses, valid, sync, calib, strict=strict)
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"frames": int(len(S)), "failed": len(failed), "seconds": round(time.time() - t_all, 1)}, fh)
    print(f"[Fase 7] SELESAI dalam {time.time() - t_all:.0f} s -> {d7}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.workers, a.max_frames)
    sys.exit(0)
