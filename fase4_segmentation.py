#!/usr/bin/env python3
r"""
fase4_segmentation.py
=====================
Fase 4 - Segmentasi semantik LiDAR (SETELAH Fase 3). Menggantikan notebook `fase4_segmentasi_lidar_rev.ipynb`.

INPUT  (dari Fase 3): fase3_deteksi/points/*.npy (N,7) | heights/*.npy (N,) | objects/*.json | object_point_idx/*.npy
OUTPUT fase4_segmentasi/points/<frame>.npy  = (M,7) float32: x, y, z, R, G, B (0-255), class_id
       fase4_segmentasi/summary_frames.csv, fase4_segmentasi/fase4_report.json, checkpoints/fase4_report.json
CHECKPOINT (otomatis): semua frame ada & tanpa NaN; class_id 0..5; SEMUA titik objek Fase 3 jadi Vehicle/Person;
  titik tanah berlabel Unknown <= 5%; Road setinggi > 0,5 m ~ 0; palet kontras.
GOAL: tiap titik 360 derajat berlabel Unknown/Road/Vegetation/Building, titik objek Fase 3 = Vehicle/Person, dengan warna tetap.

Algoritma = notebook (range image -> normal -> kekasaran -> RANSAC tanah terbatas -> aturan geometris), TIDAK diubah.
Perubahan dibanding notebook (semuanya mengurangi pekerjaan, bukan mengubah hasil):
  * Mode segmentasi dikunci ke "A" (cloud penuh lalu timpa objek). A/B test dibuang (beda A vs B di notebook ~0,02%).
  * Render BEV & preview dibuang (hanya tampilan).
  * Frame diproses paralel (--workers); hasil per-frame identik dengan serial.
  * Titik yatim Fase 3 (berlabel 4/5 tapi di luar semua kotak) TIDAK ditimpa -> pakai label model (perilaku notebook; hanya dihitung).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import Checker, Timer

try:                                            # tqdm opsional
    from tqdm import tqdm
except ImportError:                             # pragma: no cover
    def tqdm(x, **kw):
        return x

# ---------------------------------------------------------------------------------------------------------------------
# KONFIGURASI (nilai lama dari notebook, tervalidasi; tidak diubah)
# ---------------------------------------------------------------------------------------------------------------------
Z_CAP_MIN_M = -0.20           # tinggi di atas BIDANG TANAH Fase 3; isi 0.0 untuk persis [0,2]
Z_CAP_MAX_M = 2.00
N_COLS = 1800
NORMAL_GAP_COL_MAX = 3
NORMAL_GAP_ROW_MAX = 2
DIST_THRESHOLD, RANSAC_N, NUM_ITER = 0.2, 3, 1000
HEIGHT_THRESH = 0.3
ROUGHNESS_THRESH = 0.03
WALL_NZ_THRESH = 0.4
LEDGE_HEIGHT_MIN = 1.6
GROUND_HEIGHT_TOL = 0.15
GROUND_NZ_THRESH = 0.85
GROUND_FROM_HEIGHT = True
RANSAC_CAND_H = 0.5
LEVEL_NORMALS = False

GROUND_H_M = 0.15             # |h| <= ini (bukan objek) = "tanah" utk checkpoint
GROUND_UNKNOWN_FLAG = 0.05    # > 5% tanah berlabel Unknown -> peringatan
ROAD_HIGH_MAX_FRAC = 0.005    # Road dengan h > 0,5 m maksimal 0,5% dari semua Road

CLASS_NAMES = {0: "Unknown", 1: "Road", 2: "Vegetation", 3: "Building", 4: "Vehicle", 5: "Person"}
# palet FINAL (kontras >= 3:1 di atas putih). Fase 5/7 harus memakai palet yang sama.
CLASS_COLORS = {0: "#e6007e", 1: "#1f4e9c", 2: "#2e8b2e", 3: "#b5651d", 4: "#7b2cbf", 5: "#e00000"}
OBJECT_CLASS_ID = {"Car": 4, "Motorcycle": 4, "Bicycle": 4, "Person": 5}


def _hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _lum(rgb):
    c = rgb / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


PALETTE_RGB = np.stack([_hex2rgb(CLASS_COLORS[i]) for i in range(6)])


def check_palette(min_contrast=3.0, min_dist=90.0):
    cr = [1.05 / (_lum(PALETTE_RGB[i]) + 0.05) for i in range(6)]
    d = np.linalg.norm(PALETTE_RGB[:, None] - PALETTE_RGB[None], axis=2) + np.eye(6) * 1e9
    return bool(min(cr) >= min_contrast and d.min() >= min_dist), min(cr), float(d.min())


# ---------------------------------------------------------------------------------------------------------------------
# SEGMENTASI GEOMETRIS (verbatim dari notebook)
# ---------------------------------------------------------------------------------------------------------------------
def build_range_image(points_xyz, ring, n_cols=N_COLS):
    # baris = ring, kolom = bin azimut; >1 titik per sel -> yang paling DEKAT menang
    ring = ring.astype(int)
    n_rows = int(ring.max()) + 1
    az = (np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0])) + 360.0) % 360.0
    col = np.clip(np.floor(az / 360.0 * n_cols).astype(int), 0, n_cols - 1)
    row = ring
    rng = np.linalg.norm(points_xyz, axis=1)
    range_img = np.zeros((n_rows, n_cols), np.float32)
    xyz_img = np.zeros((n_rows, n_cols, 3), np.float32)
    order = np.argsort(-rng)
    range_img[row[order], col[order]] = rng[order]
    xyz_img[row[order], col[order]] = points_xyz[order]
    return range_img, xyz_img, row, col


def _gap_tolerant_neighbor(xyz_img, valid_img, axis, max_gap, circular):
    H, W, _ = xyz_img.shape
    found_xyz = np.zeros_like(xyz_img)
    found_valid = np.zeros_like(valid_img)
    remaining = np.ones_like(valid_img)
    for k in range(1, max_gap + 1):
        sx = np.roll(xyz_img, -k, axis=axis)
        sv = np.roll(valid_img, -k, axis=axis)
        if not circular:
            if axis == 0:
                sv[H - k:, :] = False
            else:
                sv[:, W - k:] = False
        take = remaining & sv
        found_xyz[take] = sx[take]
        found_valid[take] = True
        remaining &= ~take
    return found_xyz, found_valid


def compute_normals(xyz_img, valid_img):
    H, W, _ = xyz_img.shape
    xr, vr = _gap_tolerant_neighbor(xyz_img, valid_img, 1, NORMAL_GAP_COL_MAX, True)
    xu, vu = _gap_tolerant_neighbor(xyz_img, valid_img, 0, NORMAL_GAP_ROW_MAX, False)
    n = np.cross(xr - xyz_img, xu - xyz_img)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    norm[norm < 1e-6] = 1.0
    n = n / norm
    ok = valid_img & vr & vu
    normal_img = np.zeros((H, W, 3), np.float32)
    normal_img[ok] = n[ok]
    return normal_img, ok


def compute_roughness(normal_img, normal_valid, win=1):
    H, W, _ = normal_img.shape
    pad = win
    n_pad = np.pad(normal_img, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    n_pad = np.pad(n_pad, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
    v_pad = np.pad(normal_valid.astype(np.float32), ((pad, pad), (0, 0)), mode="edge")
    v_pad = np.pad(v_pad, ((0, 0), (pad, pad)), mode="wrap")
    acc_sum = np.zeros((H, W, 3), np.float32)
    acc_cnt = np.zeros((H, W), np.float32)
    offs = range(-win, win + 1)
    for dr in offs:
        for dc in offs:
            sn = n_pad[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
            sv = v_pad[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
            acc_sum += sn * sv[..., None]
            acc_cnt += sv
    cnt = np.clip(acc_cnt, 1, None)
    mean_n = acc_sum / cnt[..., None]
    var = np.zeros((H, W), np.float32)
    for dr in offs:
        for dc in offs:
            sn = n_pad[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
            sv = v_pad[pad + dr: pad + dr + H, pad + dc: pad + dc + W]
            var += np.sum((sn - mean_n) ** 2, axis=2) * sv
    return var / cnt


def remove_ground_plane(points, h, distance_threshold=DIST_THRESHOLD, ransac_n=RANSAC_N, num_iterations=NUM_ITER):
    # RANSAC satu bidang per frame -> (ground_mask, normal bidang menghadap +z).
    # Bidang dicari hanya dari kandidat tanah (|h| < RANSAC_CAND_H, h dari Fase 3), inlier hanya |h| < HEIGHT_THRESH
    # (mencegah RANSAC memilih bidang DINDING di lorong).
    n = len(points)
    ground = np.zeros(n, bool)
    up = np.array([0.0, 0.0, 1.0])
    cand = np.nonzero(np.abs(h) < RANSAC_CAND_H)[0]
    if len(cand) < 50:
        return ground, up
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[cand])
        model, _ = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                     num_iterations=num_iterations)
        nv = np.array(model[:3])
        d0 = model[3]
    except ImportError:
        rng = np.random.default_rng(0)
        best = -1
        nv = np.array([0, 0, 1.0])
        d0 = 0.0
        pc = points[cand]
        for _ in range(min(num_iterations, 400)):
            a, b, c = pc[rng.choice(len(pc), 3, replace=False)]
            v = np.cross(b - a, c - a)
            L = np.linalg.norm(v)
            if L < 1e-9:
                continue
            v /= L
            cnt = int((np.abs((pc - a) @ v) < distance_threshold).sum())
            if cnt > best:
                best, nv, d0 = cnt, v, -float(v @ a)
    L = np.linalg.norm(nv)
    nv, d0 = nv / L, d0 / L
    if abs(nv[2]) < 0.7:                       # bukan bidang mendatar -> jangan dipercaya
        return ground, up
    ground = (np.abs(points @ nv + d0) < distance_threshold) & (np.abs(h) < HEIGHT_THRESH)
    if nv[2] < 0:
        nv = -nv
    return ground, nv


def classify_points_heuristic(height, ground_mask, normal_per_point, roughness_per_point, up):
    n = len(height)
    class_id = np.zeros(n, np.uint8)
    class_id[ground_mask] = 1
    has_normal = np.linalg.norm(normal_per_point, axis=1) > 1e-3
    nz = np.abs(normal_per_point @ up) if LEVEL_NORMALS else np.abs(normal_per_point[:, 2])
    rough = roughness_per_point > ROUGHNESS_THRESH
    quasi = (~ground_mask) & has_normal & (np.abs(height) < GROUND_HEIGHT_TOL) & (nz > GROUND_NZ_THRESH) & (~rough)
    class_id[quasi] = 1
    if GROUND_FROM_HEIGHT:
        wall_foot = has_normal & (nz < WALL_NZ_THRESH) & (~rough)
        by_h = (class_id == 0) & (np.abs(height) < HEIGHT_THRESH) & (~wall_foot)
        class_id[by_h] = 1
        quasi = quasi | by_h
    non_ground = (~ground_mask) & (~quasi) & has_normal & (height > HEIGHT_THRESH)
    wall = (non_ground & ~rough & (nz < WALL_NZ_THRESH)) | \
           (non_ground & ~rough & (nz >= WALL_NZ_THRESH) & (height > LEDGE_HEIGHT_MIN))
    veg = non_ground & rough & ~wall
    class_id[wall] = 3
    class_id[veg] = 2
    return class_id


def segment_cloud(P, h):
    xyz = P[:, :3].astype(np.float64)
    range_img, xyz_img, row, col = build_range_image(xyz, P[:, 4])
    normal_img, normal_valid = compute_normals(xyz_img, range_img > 0)
    rough_img = compute_roughness(normal_img, normal_valid)
    ground_mask, up = remove_ground_plane(xyz, h)
    return classify_points_heuristic(h, ground_mask, normal_img[row, col], rough_img[row, col], up)


# ---------------------------------------------------------------------------------------------------------------------
# PER FRAME
# ---------------------------------------------------------------------------------------------------------------------
def fallback_heights(P):
    # Cadangan kalau heights/*.npy tidak ada: bidang tanah dari titik terendah per sel 2 m (<25 m), LSQ + trimming.
    xyz = P[:, :3]
    m = np.hypot(xyz[:, 0], xyz[:, 1]) < 25
    cell = np.floor(xyz[m, :2] / 2).astype(np.int64)
    key = cell[:, 0] * 100003 + cell[:, 1]
    order = np.lexsort((xyz[m, 2], key))
    k = key[order]
    g = xyz[m][order][np.r_[True, k[1:] != k[:-1]]]
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    keep = np.ones(len(g), bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(A[keep], g[keep, 2], rcond=None)
        res = g[:, 2] - A @ coef
        keep = np.abs(res) < 2 * max(np.std(res[keep]), 0.05)
    return (xyz[:, 2] - (xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2])).astype(np.float32)


def load_frame(f3_dir: Path, f: int):
    # P (N,7) | h (N,) | obj_cls (N,) 0/4/5 dari object_point_idx (HASIL AKHIR Fase 3) | orphan | fallback_h
    P = np.load(f3_dir / "points" / f"{f:06d}.npy").astype(np.float32)
    hp = f3_dir / "heights" / f"{f:06d}.npy"
    h = np.load(hp).astype(np.float32) if hp.exists() else None
    fb = False
    if h is None or len(h) != len(P):
        h, fb = fallback_heights(P), True
    rec = {"objects": []}
    jp = f3_dir / "objects" / f"{f:06d}.json"
    if jp.exists():
        with open(jp, encoding="utf-8") as fh:
            rec = json.load(fh)
    obj_cls = np.zeros(len(P), np.int8)
    ip = f3_dir / "object_point_idx" / f"{f:06d}.npy"
    if ip.exists() and rec["objects"]:
        pairs = np.load(ip).reshape(-1, 2)
        if len(pairs):
            per_obj = np.array([OBJECT_CLASS_ID.get(o.get("class"), 4) for o in rec["objects"]], np.int8)
            ok = (pairs[:, 0] < len(per_obj)) & (pairs[:, 1] < len(P))
            obj_cls[pairs[ok, 1]] = per_obj[pairs[ok, 0]]
    orphan = np.isin(P[:, 6].astype(int), (4, 5)) & (obj_cls == 0)
    return P, h, obj_cls, orphan, fb


def process_frame(args):
    """Satu frame -> tulis points/<f>.npy, kembalikan satu baris ringkasan. Modul-level agar bisa dipakai ProcessPool."""
    f3_dir, out_points, f = args
    P, h, oc, orphan, fb = load_frame(Path(f3_dir), f)
    model_lab = segment_cloud(P, h)                       # mode A: cloud penuh
    final = model_lab.copy()
    m = oc > 0
    final[m] = oc[m]                                      # timpa titik objek Fase 3
    keep = ((h >= Z_CAP_MIN_M) & (h <= Z_CAP_MAX_M)) | m
    out = np.zeros((int(keep.sum()), 7), np.float32)
    out[:, :3] = P[keep, :3]
    out[:, 3:6] = PALETTE_RGB[final[keep]]
    out[:, 6] = final[keep]
    np.save(Path(out_points) / f"{f:06d}.npy", out)

    free = oc == 0
    gm = free & (np.abs(h) <= GROUND_H_M)
    dist = np.hypot(P[:, 0], P[:, 1])
    rec = {"frame": f, "n_full": len(P), "n_out": len(out), "n_obj_pts": int(m.sum()),
           "n_orphan_f3": int(orphan.sum()), "n_ground": int(gm.sum()),
           "n_ground_unknown": int((final[gm] == 0).sum()),
           "n_ground_unknown_far": int(((final == 0) & gm & (dist > 20)).sum()),
           "n_ground_far": int((gm & (dist > 20)).sum()),
           "n_unknown_nonobj": int(((final == 0) & free).sum()),
           "n_road_high": int(((final == 1) & (h > 0.5)).sum()),
           "h_fallback": int(fb), "bad_values": int(not np.isfinite(out).all())}
    for c in range(6):
        rec[f"n_{CLASS_NAMES[c]}"] = int((out[:, 6] == c).sum())
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# ORKESTRASI
# ---------------------------------------------------------------------------------------------------------------------
def run_checkpoints(out: Path, summary: pd.DataFrame, n_in: int, strict=False):
    ck = Checker("FASE 4", out / "checkpoints" / "fase4_report.json", strict=strict)
    o4 = out / "fase4_segmentasi"
    n_files = len(list((o4 / "points").glob("*.npy")))
    ck.ok("Semua frame Fase 3 tersegmentasi", n_files == n_in == len(summary), f"{n_files} berkas dari {n_in} frame Fase 3")
    ck.ok("Tidak ada NaN/Inf di output", int(summary.bad_values.sum()) == 0, f"{int(summary.bad_values.sum())} frame bermasalah")

    cls_cols = [f"n_{CLASS_NAMES[c]}" for c in range(6)]
    tot = summary[cls_cols].sum()
    ck.ok("Jumlah titik per kelas = jumlah titik output", int(tot.sum()) == int(summary.n_out.sum()), f"{int(tot.sum())} vs {int(summary.n_out.sum())}")
    ck.info("Distribusi kelas", ", ".join(f"{CLASS_NAMES[c]} {100 * tot[cls_cols[c]] / max(tot.sum(), 1):.1f}%" for c in range(6)))

    n_obj = int(summary.n_obj_pts.sum())
    n_vp = int(tot["n_Vehicle"] + tot["n_Person"])
    ck.ok("Semua titik objek Fase 3 jadi Vehicle/Person (tidak ada yang hilang/ganda)", n_vp == n_obj, f"{n_vp} berlabel 4/5, {n_obj} titik objek Fase 3")

    pal_ok, cr, dmin = check_palette()
    ck.ok("Palet kontras di atas putih", pal_ok, f"kontras min {cr:.1f}:1, jarak RGB min {dmin:.0f}")

    gu = summary.n_ground_unknown.sum() / max(summary.n_ground.sum(), 1)
    far = summary.n_ground_unknown_far.sum() / max(summary.n_ground_unknown.sum(), 1)
    worst = summary.assign(rate=summary.n_ground_unknown / summary.n_ground.clip(lower=1)).nlargest(3, "rate")
    ck.ok(f"Tanah (|h|<={GROUND_H_M} m) berlabel Unknown <= {100 * GROUND_UNKNOWN_FLAG:.0f}%", gu <= GROUND_UNKNOWN_FLAG,
          f"{100 * gu:.2f}% ({100 * far:.0f}% dari itu > 20 m); terburuk: " +
          ", ".join(f"{int(a)} ({100 * b:.1f}%)" for a, b in zip(worst.frame, worst.rate)), fatal=False)
    rh = int(summary.n_road_high.sum())
    frac = rh / max(int(tot["n_Road"]), 1)
    ck.ok("Titik Road dengan h > 0,5 m (dinding salah jadi Road) ~ 0", frac <= ROAD_HIGH_MAX_FRAC, f"{rh} titik = {100 * frac:.3f}% dari Road", fatal=False)
    orph = int(summary.n_orphan_f3.sum())
    ck.info("Titik yatim Fase 3 (4/5 di luar kotak) -> TIDAK ditimpa, pakai label model", f"{orph} titik ({orph / max(n_obj + orph, 1) * 100:.1f}% dari titik berlabel 4/5 Fase 3)")
    if int(summary.h_fallback.sum()):
        ck.ok("heights/*.npy tersedia (tanpa tinggi cadangan)", False, f"{int(summary.h_fallback.sum())} frame memakai tinggi cadangan", fatal=False)

    report = {"mode": "A", "palette_ok": pal_ok, "ground_unknown_rate": float(gu),
              "class_totals": {CLASS_NAMES[c]: int(tot[cls_cols[c]]) for c in range(6)},
              "orphans_f3": orph, "road_high_pts": rh, "z_cap": [Z_CAP_MIN_M, Z_CAP_MAX_M]}
    with open(o4 / "fase4_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    ck.finish()


def run(out, force=False, strict=False, workers=None, max_frames=None):
    t_all = time.time()
    out = Path(out)
    f3 = out / "fase3_deteksi"
    o4 = out / "fase4_segmentasi"
    marker = o4 / "fase4_done.json"
    if marker.exists() and not force:
        print(f"[Fase 4] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    if not (f3 / "fase3_done.json").exists():
        raise SystemExit(f"Fase 3 belum selesai ({f3 / 'fase3_done.json'} tidak ada). Jalankan Fase 3 dulu.")
    frames = sorted(int(p.stem) for p in (f3 / "points").glob("*.npy"))
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise SystemExit(f"Tidak ada frame di {f3 / 'points'}")
    if o4.exists():
        shutil.rmtree(o4)                                  # run baru = hasil baru
    (o4 / "points").mkdir(parents=True)

    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    jobs = [(str(f3), str(o4 / "points"), f) for f in frames]
    print(f"[Fase 4] {len(frames)} frame | mode A (cloud penuh lalu timpa objek) | {workers} proses")
    with Timer("Fase 4 segmentasi"):
        if workers == 1:
            recs = [process_frame(j) for j in tqdm(jobs, desc="Fase 4")]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                recs = list(tqdm(ex.map(process_frame, jobs, chunksize=8), total=len(jobs), desc="Fase 4"))
    summary = pd.DataFrame(recs)
    summary.to_csv(o4 / "summary_frames.csv", index=False)
    run_checkpoints(out, summary, len(frames), strict=strict)
    with open(marker, "w", encoding="utf-8") as fh:        # penanda selesai: ditulis SETELAH checkpoint lulus
        json.dump({"frames": len(frames), "seconds": round(time.time() - t_all, 1)}, fh)
    print(f"[Fase 4] SELESAI dalam {time.time() - t_all:.0f} s -> {o4}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--workers", type=int, help="jumlah proses paralel (default: min(6, CPU-1))")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.workers, a.max_frames)
    sys.exit(0)
