#!/usr/bin/env python3
r"""
fase5_payload.py
================
Fase 5 - Payload builder BEV. Menggantikan notebook `fase5_payload_builder_v29.ipynb`.

INPUT : fase4_segmentasi/points/*.npy (x,y,z,R,G,B,class_id)  |  fase3_deteksi/objects/*.json
        waypoints_per_frame.json (Fase 1b, TIDAK dihitung ulang)  |  sync_table_with_waypoints.csv (lidar_t)
        calib/calib_config.json (hanya utk menurunkan sektor pandang kamera -> meta.json)
OUTPUT: fase5_payload/<frame>.json   {"idx","t","pts":[[x,y,z,class],...],"obj":[...],"wp":[[x,y],...]}
        fase5_payload/meta.json      (sekali: tabel kelas+warna, sektor FOV kamera, parameter downsampling, skema)
CHECKPOINT (otomatis): payload ada utk semua frame Fase 4; jumlah objek payload == Fase 3; semua titik Vehicle/Person
  dipertahankan utuh; t naik tegas; waypoint mengarah ke depan; class_id 0..5; ukuran payload/frame.
GOAL: satu payload ringkas per frame siap stream 10 Hz ke Flutter (Fase 6).

PERUBAHAN dibanding notebook (semuanya diberi catatan di CATATAN_PERUBAHAN.md):
  1. Voxel downsampling DIVEKTORISASI (np.unique pada kunci int64, bukan loop Python per titik). Hasil sama, ~100x lebih cepat.
     Perbaikan kecil: id band jarak ikut dalam kunci voxel (di notebook voxel band dekat & jauh bisa bertabrakan kunci -> titik hilang).
  2. Titik objek (Vehicle/Person) TIDAK ikut downsampling: semuanya dikirim (rata-rata ~9 titik/frame; murah). Hanya latar
     (Unknown/Road/Vegetation/Building) yang dirampingkan secara adaptif.
  3. Latar 360 derajat vs objek 3D di FOV depan dibedakan EKSPLISIT: `pts` = seluruh titik 360 derajat; `obj` = kotak 3D yang HANYA
     ada di sektor pandang kamera depan (YOLO butuh kamera). Sektor itu (pusat azimut & lebar, dari K dan T_lidar_to_cam)
     ditulis di meta.json supaya Flutter bisa menggambar kerucut FOV dan tidak menganggap "tidak ada kotak" = "tidak ada objek"
     di luar sektor tsb.
  4. Waypoint dibaca dari waypoints_per_frame.json (Fase 1b), bukan dihitung ulang. Frame tanpa waypoint -> "wp": [].
  5. Field objek tambahan: "st" (L = lidar_ok, D = depth_only) dan "yc" (1 = yaw terverifikasi, 0 = tebakan prior).
  6. Frame diproses paralel (--workers); keluaran identik dengan serial.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from common import Checker, Timer
from fase4_segmentation import CLASS_COLORS, CLASS_NAMES

try:
    from tqdm import tqdm
except ImportError:                             # pragma: no cover
    def tqdm(x, **kw):
        return x

# ---------------------------------------------------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------------------------------------------------
DISTANCE_BANDS = ((5.0, 0.08), (15.0, 0.20), (float("inf"), 0.45))   # (batas_jarak_m, ukuran_voxel_m); nilai notebook
OBJECT_CLASS_IDS = (4, 5)               # Vehicle, Person: tidak di-downsample
ROUND_DECIMALS = 2                      # cm
MAX_MEAN_KB = 250.0                     # target rata-rata ukuran payload/frame (peringatan bila lewat)
WP_FWD_MIN_FRAC = 0.95
SCHEMA_VERSION = 1
_OFF = 1 << 14                          # offset indeks voxel agar non-negatif (|indeks| < 16384)


# ---------------------------------------------------------------------------------------------------------------------
# DOWNSAMPLING VOXEL ADAPTIF (vektorisasi)
# ---------------------------------------------------------------------------------------------------------------------
def adaptive_voxel_downsample(xyz, class_id, distance_bands=DISTANCE_BANDS, keep_all_classes=OBJECT_CLASS_IDS):
    """Return indeks titik yang dipertahankan (terurut naik): 1 titik per (voxel, kelas, band); titik berkelas
    `keep_all_classes` selalu dipertahankan. Titik pertama dalam voxel menang (sama dengan notebook)."""
    n = len(xyz)
    if n == 0:
        return np.zeros(0, np.int64)
    dist = np.hypot(xyz[:, 0], xyz[:, 1])
    vsize = np.zeros(n)
    band = np.zeros(n, np.int64)
    lower = 0.0
    for b, (upper, vs) in enumerate(distance_bands):
        m = (dist >= lower) & (dist < upper)
        vsize[m] = vs
        band[m] = b
        lower = upper
    vox = np.floor(xyz / vsize[:, None]).astype(np.int64) + _OFF
    if vox.min() < 0 or vox.max() >= 2 * _OFF:
        raise ValueError("indeks voxel di luar rentang (koordinat tidak wajar)")
    key = (((vox[:, 0] << 15 | vox[:, 1]) << 15 | vox[:, 2]) << 2 | band) << 3 | class_id.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    keep = np.zeros(n, bool)
    keep[first] = True
    if keep_all_classes:
        keep |= np.isin(class_id, keep_all_classes)
    return np.nonzero(keep)[0]


# ---------------------------------------------------------------------------------------------------------------------
# PER FRAME
# ---------------------------------------------------------------------------------------------------------------------
def build_payload(idx, t, points7, objects, wp):
    if points7 is None or len(points7) == 0:
        pts = []
        n_before = n_after = n_obj_pts = 0
    else:
        xyz = points7[:, :3]
        cls = points7[:, 6].astype(np.int64)
        keep = adaptive_voxel_downsample(xyz, cls)
        n_before, n_after = len(points7), len(keep)
        n_obj_pts = int(np.isin(cls[keep], OBJECT_CLASS_IDS).sum())
        q = np.round(xyz[keep].astype(np.float64), ROUND_DECIMALS)      # float64 dulu: float32.tolist() mencetak 3.140000104904175
        pts = [list(r) for r in zip(q[:, 0].tolist(), q[:, 1].tolist(), q[:, 2].tolist(), cls[keep].tolist())]
    obj = [{
        "id": o.get("track_id"), "cls": o["class"],
        "cx": round(o["center_xyz"][0], ROUND_DECIMALS), "cy": round(o["center_xyz"][1], ROUND_DECIMALS),
        "cz": round(o["center_xyz"][2], ROUND_DECIMALS),
        "l": o["length_m"], "w": o["width_m"], "h": o["height_m"],
        "yaw": round(float(o.get("yaw_rad", 0.0)), 4),
        "st": "L" if o.get("status") == "lidar_ok" else "D",
        "yc": int(bool(o.get("yaw_confident", False))),
    } for o in objects]
    wp_out = [[round(x, ROUND_DECIMALS), round(y, ROUND_DECIMALS)] for x, y in (wp or [])]
    payload = {"idx": idx, "t": t, "pts": pts, "obj": obj, "wp": wp_out}
    return payload, n_before, n_after, n_obj_pts


def process_frame(args):
    f3_dir, f4_dir, out_dir, idx, t, wp = args
    p4 = Path(f4_dir) / "points" / f"{idx:06d}.npy"
    points7 = np.load(p4) if p4.exists() else None
    jp = Path(f3_dir) / "objects" / f"{idx:06d}.json"
    objects = []
    if jp.exists():
        with open(jp, encoding="utf-8") as fh:
            objects = json.load(fh).get("objects", [])
    payload, nb, na, nop = build_payload(idx, t, points7, objects, wp)
    path = Path(out_dir) / f"{idx:06d}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    n_obj4 = int(np.isin(points7[:, 6].astype(int), OBJECT_CLASS_IDS).sum()) if points7 is not None and len(points7) else 0
    pts_cls = [p[3] for p in payload["pts"]]
    wp_fwd = float(np.mean([p[0] > 0 for p in payload["wp"]])) if payload["wp"] else None
    yaw_bad = sum(1 for o in payload["obj"] if o["cls"] == "Car" and not (-math.pi - 1e-6 <= o["yaw"] <= math.pi + 1e-6))
    return {"idx": idx, "t": t, "kb": path.stat().st_size / 1024, "n_before": nb, "n_after": na,
            "n_obj_json": len(objects), "n_obj_payload": len(payload["obj"]),
            "n_objpts_f4": n_obj4, "n_objpts_payload": nop,
            "cls_ok": int(all(0 <= c <= 5 for c in pts_cls)), "n_wp": len(payload["wp"]), "wp_fwd": wp_fwd, "yaw_bad": yaw_bad}


# ---------------------------------------------------------------------------------------------------------------------
# META (sektor FOV kamera, kelas, parameter)
# ---------------------------------------------------------------------------------------------------------------------
def camera_fov_sector(calib_path: Path):
    """Pusat azimut sumbu optik kamera di frame LiDAR (derajat, 0 = +x, positif = kiri/CCW) dan lebar FOV horizontal."""
    cfg = json.load(open(calib_path, encoding="utf-8"))
    T = np.asarray(cfg["T_lidar_to_cam"], float)
    K = np.asarray(cfg["K"], float)
    fwd = T[:3, :3][2, :]                                   # sumbu z kamera dinyatakan di frame LiDAR
    az = math.degrees(math.atan2(fwd[1], fwd[0]))
    hfov = math.degrees(2 * math.atan(float(cfg["IMG_W"]) / (2 * K[0, 0])))
    return round(az, 2), round(hfov, 2)


def build_meta(out: Path):
    az, hfov = camera_fov_sector(out / "calib" / "calib_config.json")
    return {
        "schema": SCHEMA_VERSION,
        "frame": "LiDAR/ego: x depan, y kiri, z atas (meter)",
        "pts": "[x, y, z, class_id] - latar 360 derajat (titik objek Vehicle/Person utuh, latar didownsample adaptif)",
        "obj": "kotak 3D objek - HANYA ada di sektor pandang kamera depan (fov); di luar sektor tidak ada deteksi",
        "wp": "[x, y] jalur 20 m ke depan tiap 0,5 m (ego frame); [] bila tak ada",
        "obj_fields": {"id": "track_id", "cls": "kelas", "cx/cy/cz": "pusat", "l/w/h": "ukuran kanonik", "yaw": "rad, arah hidung (-pi..pi)",
                      "st": "L = lidar_ok, D = depth_only (posisi dari depth ZED)", "yc": "1 = yaw terverifikasi, 0 = tebakan prior"},
        "fov": {"az_center_deg": az, "hfov_deg": hfov, "az_min_deg": round(az - hfov / 2, 2), "az_max_deg": round(az + hfov / 2, 2)},
        "classes": {str(c): {"name": CLASS_NAMES[c], "color": CLASS_COLORS[c]} for c in range(6)},
        "downsample": {"bands": [[None if math.isinf(u) else u, v] for u, v in DISTANCE_BANDS], "keep_all_classes": list(OBJECT_CLASS_IDS),
                       "decimals": ROUND_DECIMALS},
    }


# ---------------------------------------------------------------------------------------------------------------------
# ORKESTRASI
# ---------------------------------------------------------------------------------------------------------------------
def load_sync(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): float(r["lidar_t"]) for r in csv.DictReader(f)}


def run_checkpoints(out: Path, S, n_in: int, strict=False):
    ck = Checker("FASE 5", out / "checkpoints" / "fase5_report.json", strict=strict)
    pdir = out / "fase5_payload"
    n_files = len(list(pdir.glob("[0-9]*.json")))
    ck.ok("Payload ada untuk semua frame Fase 4", n_files == n_in == len(S), f"{n_files} berkas dari {n_in} frame")
    ck.ok("meta.json ada (tabel kelas, sektor FOV)", (pdir / "meta.json").exists())
    mism = S[S.n_obj_json != S.n_obj_payload]
    ck.ok("Jumlah objek payload == Fase 3 (tidak ada yang hilang)", len(mism) == 0, f"{len(mism)} frame beda; total {int(S.n_obj_payload.sum())} objek")
    ck.ok("Semua titik Vehicle/Person Fase 4 dipertahankan utuh", int(S.n_objpts_f4.sum()) == int(S.n_objpts_payload.sum()),
          f"{int(S.n_objpts_payload.sum())} dari {int(S.n_objpts_f4.sum())}")
    ck.ok("class_id titik dalam 0..5", int(S.cls_ok.min()) == 1)
    St = S.sort_values("idx")
    ck.ok("Timestamp t naik tegas menurut idx", bool((np.diff(St.t.values) > 0).all()), f"t {St.t.iloc[0]:.2f} .. {St.t.iloc[-1]:.2f}")
    w = S.dropna(subset=["wp_fwd"])
    wf = float(w.wp_fwd.mean()) if len(w) else 0.0
    ck.ok(f"Waypoint mengarah ke depan (x>0) >= {100 * WP_FWD_MIN_FRAC:.0f}%", wf >= WP_FWD_MIN_FRAC, f"{100 * wf:.1f}% dari {len(w)} frame ber-waypoint")
    ck.info("Frame tanpa waypoint", f"{int((S.n_wp == 0).sum())} frame")
    ck.ok("yaw Car dalam [-pi, pi]", int(S.yaw_bad.sum()) == 0, f"{int(S.yaw_bad.sum())} di luar rentang")
    ck.info("Titik/frame", f"{S.n_before.mean():.0f} -> {S.n_after.mean():.0f} ({100 * S.n_after.mean() / max(S.n_before.mean(), 1):.1f}%)")
    ck.ok(f"Ukuran payload rata-rata <= {MAX_MEAN_KB:.0f} KB/frame", S.kb.mean() <= MAX_MEAN_KB,
          f"rata-rata {S.kb.mean():.1f} KB, p95 {S.kb.quantile(.95):.1f} KB, maks {S.kb.max():.1f} KB; total {S.kb.sum() / 1024:.0f} MB", fatal=False)
    ck.finish()


def run(out, force=False, strict=False, workers=None, max_frames=None):
    import pandas as pd
    t_all = time.time()
    out = Path(out)
    f3, f4, pdir = out / "fase3_deteksi", out / "fase4_segmentasi", out / "fase5_payload"
    marker = pdir / "fase5_done.json"
    if marker.exists() and not force:
        print(f"[Fase 5] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    for need in (f4 / "fase4_done.json", out / "waypoints_per_frame.json", out / "sync_table_with_waypoints.csv", out / "calib" / "calib_config.json"):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    frames = sorted(int(p.stem) for p in (f4 / "points").glob("*.npy"))
    if max_frames:
        frames = frames[:max_frames]
    tmap = load_sync(out / "sync_table_with_waypoints.csv")
    wpj = json.load(open(out / "waypoints_per_frame.json", encoding="utf-8"))
    miss = [f for f in frames if f not in tmap]
    if miss:
        raise SystemExit(f"{len(miss)} frame Fase 4 tidak ada di sync_table (mis. {miss[:3]})")
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True)
    meta = build_meta(out)
    with open(pdir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    print(f"[Fase 5] {len(frames)} frame | FOV kamera: pusat {meta['fov']['az_center_deg']} deg, lebar {meta['fov']['hfov_deg']} deg (di frame LiDAR)")

    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    jobs = [(str(f3), str(f4), str(pdir), f, tmap[f], wpj.get(str(f), [])) for f in frames]
    with Timer("Fase 5 payload"):
        if workers == 1:
            recs = [process_frame(j) for j in tqdm(jobs, desc="Fase 5")]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                recs = list(tqdm(ex.map(process_frame, jobs, chunksize=8), total=len(jobs), desc="Fase 5"))
    S = pd.DataFrame(recs)
    S.to_csv(pdir / "summary_frames.csv", index=False)
    run_checkpoints(out, S, len(frames), strict=strict)
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"frames": len(frames), "seconds": round(time.time() - t_all, 1)}, fh)
    print(f"[Fase 5] SELESAI dalam {time.time() - t_all:.0f} s -> {pdir}")


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
