#!/usr/bin/env python3
r"""
fase1_extract_sync.py
=====================
FASE 1 -- Ekstraksi & sinkronisasi rosbag ROS1 (.bag).  Menggantikan DUA skrip lama
(`extract_sync_rosbag.py` + `extract_lidar_v2.py`) dengan satu skrip.

INPUT   : rosbag .bag
PROSES  : ekstrak lidar, rgb, depth, camera_info, lat, lon, odom (px..pz, qx..qw), imu,
          velocity, yaw; interpolasi lat/lon ke timestamp LiDAR; sinkronkan semua ke timestamp LiDAR.
OUTPUT  : <out>/sync_table.csv, <out>/sync_table_with_gps.csv, <out>/lidar/lidar_<t>.npy
          (Nx6 float32: x, y, z, intensity, ring, time), rgb/, depth/, camera_info.json, meta/*.csv,
          checkpoints/fase1_report.json
CHECKPOINT (otomatis, menghentikan pipeline bila FAIL):
          - field ring/time LiDAR ada
          - tabel lat/lon tanpa NaN (dan celah GPS dilaporkan)
          - px, py, qz odom benar-benar dinamis (bukan konstan 0)

Cara kerja (lebih cepat dari versi lama, hasil sinkronisasi identik):
  Pass 1  baca bag, simpan HANYA timestamp lidar/rgb/depth + decode topic kecil (imu/odom/gps/...).
  Sync    hitung pasangan terdekat (vektor numpy) -> tahu persis rgb/depth mana yang dipakai.
  Pass 2  decode & tulis lidar (semua), rgb/depth (hanya yang terpilih sync), tulis di thread paralel.
  (Versi lama menyimpan SEMUA gambar rgb/depth; yang tak masuk toleransi sync tidak pernah dipakai.
   Pakai --save-unmatched untuk perilaku lama.)

PAKAI:
    python fase1_extract_sync.py --bag /path/Rosbag_Rute_02.bag --out /path/output_rute02
    python fase1_extract_sync.py --bag ... --list          # lihat topic saja
Install: pip install rosbags numpy pillow pandas
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import (GPS_MAX_GAP_S, ODOM_MIN_STD, ODOM_QZ_MIN_STD, SYNC_ORDER, SYNC_TOL_S, TOPICS,
                    Checker, Paths, Timer)

try:
    from PIL import Image as PILImage
except ImportError:                                  # pragma: no cover
    PILImage = None

try:
    import pandas as pd
except ImportError:                                  # pragma: no cover
    raise SystemExit("pip install pandas")

try:
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:                                  # pragma: no cover
    raise SystemExit("pip install rosbags numpy pillow pandas")

FIELD_DTYPE_MAP = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                   5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}

IMU_COLS = ["t", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"]
ODOM_COLS = ["t", "px", "py", "pz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"]
META_HEADERS = {"imu": IMU_COLS, "odom": ODOM_COLS, "lat": ["t", "latitude"], "lon": ["t", "longitude"],
                "velocity": ["t", "velocity"], "yaw": ["t", "yaw"]}


def norm(topic: str) -> str:
    return topic.rstrip("/")


def list_topics(bag_path):
    with Reader(bag_path) as reader:
        print(f"{'Topic':45s} {'Type':35s} {'Count':>8s}")
        print("-" * 90)
        for c in sorted(reader.connections, key=lambda c: c.topic):
            print(f"{c.topic:45s} {c.msgtype:35s} {c.msgcount:8d}")


# --------------------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------------------
def decode_image(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("bgr8", "rgb8"):
        arr = buf.reshape(h, w, 3)
        return np.ascontiguousarray(arr[:, :, ::-1] if enc == "bgr8" else arr)
    if enc in ("bgra8", "rgba8"):
        arr = buf.reshape(h, w, 4)[:, :, :3]
        return np.ascontiguousarray(arr[:, :, ::-1] if enc == "bgra8" else arr)
    if enc == "mono8":
        return buf.reshape(h, w)
    if enc == "32fc1":
        return buf.view(np.float32).reshape(h, w)
    if enc == "16uc1":                                # ZED biasanya 32FC1 (meter); jaga-jaga 16UC1 (mm)
        return buf.view(np.uint16).reshape(h, w).astype(np.float32) / 1000.0
    raise ValueError(f"Encoding gambar belum didukung: {msg.encoding}")


def decode_pointcloud2(msg) -> np.ndarray:
    """PointCloud2 -> Nx6 float32 [x, y, z, intensity, ring, time]. Titik x/y/z/intensity non-finite dibuang
    (sama seperti extractor lama)."""
    dtype = np.dtype({"names": [f.name for f in msg.fields],
                      "formats": [FIELD_DTYPE_MAP[f.datatype] for f in msg.fields],
                      "offsets": [f.offset for f in msg.fields],
                      "itemsize": msg.point_step})
    pts = np.frombuffer(msg.data, dtype=dtype)
    missing = [k for k in ("x", "y", "z", "intensity", "ring", "time") if k not in pts.dtype.names]
    if missing:
        raise RuntimeError(f"Field PointCloud2 hilang: {missing}; ada: {pts.dtype.names}")
    x, y, z = (np.asarray(pts[k], dtype=np.float32) for k in ("x", "y", "z"))
    inten = np.asarray(pts["intensity"], dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(inten)
    return np.stack([x[valid], y[valid], z[valid], inten[valid],
                     np.asarray(pts["ring"])[valid].astype(np.float32),
                     np.asarray(pts["time"], dtype=np.float32)[valid]], axis=1).astype(np.float32)


def get_field(msg, *names):
    for n in names:
        if hasattr(msg, n):
            return getattr(msg, n)
    raise AttributeError(f"None of {names} found on message")


# --------------------------------------------------------------------------------------
# Penulisan file paralel (zlib/PNG dan np.save melepas GIL)
# --------------------------------------------------------------------------------------
class AsyncWriter:
    def __init__(self, workers=4, max_pending=32):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.sem = threading.BoundedSemaphore(max_pending)
        self.futs = []

    def submit(self, fn, *a, **k):
        self.sem.acquire()

        def run():
            try:
                return fn(*a, **k)
            finally:
                self.sem.release()

        self.futs.append(self.pool.submit(run))

    def close(self):
        try:
            for f in self.futs:
                f.result()                          # lempar exception penulisan bila ada
        finally:
            self.pool.shutdown()


def _save_png(path, arr):
    PILImage.fromarray(arr).save(path, compress_level=1)    # lossless, ~3-5x lebih cepat dari default (6)


# --------------------------------------------------------------------------------------
# Sinkronisasi
# --------------------------------------------------------------------------------------
def nearest_match(anchor_t, other_t, tol):
    """Untuk tiap anchor: indeks terdekat di `other_t` (terurut), selisih waktu, dan mask 'masuk toleransi'.
    Ekuivalen dengan loop `_nearest` lama (kandidat idx-1 dan idx, ambil |dt| minimum), tapi vektor."""
    n = len(other_t)
    if n == 0:
        return np.zeros(len(anchor_t), int), np.full(len(anchor_t), np.nan), np.zeros(len(anchor_t), bool)
    pos = np.searchsorted(other_t, anchor_t)
    lo = np.clip(pos - 1, 0, n - 1)
    hi = np.clip(pos, 0, n - 1)
    pick_hi = np.abs(other_t[hi] - anchor_t) < np.abs(other_t[lo] - anchor_t)   # seri -> lo (sama dgn min() lama)
    best = np.where(pick_hi, hi, lo)
    dt = other_t[best] - anchor_t
    return best, dt, np.abs(dt) <= tol


# --------------------------------------------------------------------------------------
# Pass 1 : scan
# --------------------------------------------------------------------------------------
def scan_bag(bag_path, paths):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = {norm(t): k for k, t in TOPICS.items()}
    ts_ns = {k: [] for k in ("lidar", "rgb", "depth")}
    records = {k: [] for k in ("imu", "odom", "lat", "lon", "velocity", "yaw")}
    counts = {k: 0 for k in TOPICS}
    cam_saved = False

    with Reader(bag_path) as reader:
        conns = [c for c in reader.connections if norm(c.topic) in wanted]
        if not conns:
            raise SystemExit("Topic tidak ditemukan. Jalankan dulu dengan --list.")
        missing = [k for k, t in TOPICS.items() if norm(t) not in {norm(c.topic) for c in conns}]
        if missing:
            print(f"PERINGATAN: topic tidak ada di bag: {missing}")
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            key = wanted[norm(connection.topic)]
            counts[key] += 1
            if key in ts_ns:                                   # data besar: cukup timestamp di pass 1
                ts_ns[key].append(timestamp)
                continue
            if key == "camera_info" and cam_saved:
                continue
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            t = timestamp / 1e9
            if key == "camera_info":
                info = {"width": int(msg.width), "height": int(msg.height),
                        "distortion_model": msg.distortion_model,
                        "K": [float(v) for v in get_field(msg, "k", "K")],
                        "D": [float(v) for v in get_field(msg, "d", "D")],
                        "P": [float(v) for v in get_field(msg, "p", "P")]}
                with open(paths.camera_info, "w") as f:
                    json.dump(info, f, indent=2)
                cam_saved = True
            elif key == "imu":
                o, av, la = msg.orientation, msg.angular_velocity, msg.linear_acceleration
                records[key].append((t, *[float(v) for v in (o.x, o.y, o.z, o.w, av.x, av.y, av.z, la.x, la.y, la.z)]))
            elif key == "odom":
                p, o = msg.pose.pose.position, msg.pose.pose.orientation
                lv, av = msg.twist.twist.linear, msg.twist.twist.angular
                records[key].append((t, *[float(v) for v in (p.x, p.y, p.z, o.x, o.y, o.z, o.w,
                                                              lv.x, lv.y, lv.z, av.x, av.y, av.z)]))
            else:                                              # lat, lon, velocity, yaw
                records[key].append((t, float(msg.data)))

    for k in records:
        records[k].sort(key=lambda r: r[0])
    ns = {k: np.array(sorted(v), dtype=np.int64) for k, v in ts_ns.items()}
    print("Jumlah pesan per topic:")
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")
    return ns, records, counts


def dump_meta_csv(paths, records):
    for key, cols in META_HEADERS.items():
        rows = records.get(key, [])
        if rows:
            with open(paths.meta / f"{key}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                w.writerows(rows)


# --------------------------------------------------------------------------------------
# Sync table
# --------------------------------------------------------------------------------------
def build_sync_table(paths, ns, records, tol, save_unmatched):
    if len(ns["lidar"]) == 0:
        raise SystemExit("Tidak ada data LiDAR.")
    anchor_t = ns["lidar"] / 1e9
    n = len(anchor_t)
    cols = {"idx": np.arange(n), "lidar_t": anchor_t,
            "lidar_path": [str(paths.lidar / f"lidar_{t:.6f}.npy") for t in anchor_t]}
    needed_ns = {"rgb": set(), "depth": set()}
    match_report = {}

    for key in SYNC_ORDER:
        if key in ("rgb", "depth"):
            other_t = ns[key] / 1e9
            best, dt, ok = nearest_match(anchor_t, other_t, tol)
            tcol = np.where(ok, other_t[best] if len(other_t) else np.nan, np.nan)
            ext, sub = ("png", "rgb") if key == "rgb" else ("npy", "depth")
            data = [str(paths.out / sub / f"{key}_{other_t[b]:.6f}.{ext}") if o else "" for b, o in zip(best, ok)]
            needed_ns[key] = {int(ns[key][b]) for b, o in zip(best, ok) if o}
        else:
            rec = records.get(key, [])
            other_t = np.array([r[0] for r in rec])
            best, dt, ok = nearest_match(anchor_t, other_t, tol)
            tcol = np.where(ok, other_t[best] if len(other_t) else np.nan, np.nan)
            data = []
            for b, o in zip(best, ok):
                if not o:
                    data.append("")
                    continue
                payload = rec[b][1:]
                data.append(payload[0] if len(payload) == 1 else str(tuple(payload)))
        cols[f"{key}_t"] = tcol
        cols[f"{key}_dt"] = np.where(ok, dt, np.nan)
        cols[f"{key}_data"] = data
        match_report[key] = (int(ok.sum()), int((~ok).sum()))

    df = pd.DataFrame(cols)
    df.to_csv(paths.sync_csv, index=False)
    print(f"\nsync_table.csv: {n} frame (anchor=lidar, tol={tol}s)")
    for k, (m, miss) in match_report.items():
        print(f"  {k:10s}: {m} match, {miss} kosong")
    if save_unmatched:
        needed_ns = {"rgb": set(ns["rgb"].tolist()), "depth": set(ns["depth"].tolist())}
    return df, needed_ns, match_report


def add_gps_interp(paths, df, records):
    """Interpolasi linear lat/lon mentah ke timestamp LiDAR. Di luar rentang -> nilai ujung (tanpa NaN)."""
    t = df["lidar_t"].to_numpy()
    n_outside = {}
    for key, col in (("lat", "lat_interp"), ("lon", "lon_interp")):
        rec = records[key]
        if not rec:
            raise SystemExit(f"Data {key} kosong, tidak bisa interpolasi GPS.")
        rt = np.array([r[0] for r in rec])
        rv = np.array([r[1] for r in rec])
        df[col] = np.interp(t, rt, rv)
        n_outside[key] = int(((t < rt[0]) | (t > rt[-1])).sum())
    return df, n_outside          # CSV baru ditulis di run() SETELAH checkpoint lulus (penanda 'fase selesai')


# --------------------------------------------------------------------------------------
# Pass 2 : tulis lidar / rgb / depth
# --------------------------------------------------------------------------------------
def write_sensor_files(bag_path, paths, needed_ns, save_workers=4):
    if PILImage is None:
        raise SystemExit("pip install pillow")
    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = {norm(TOPICS[k]): k for k in ("lidar", "rgb", "depth")}
    stats = {"frames": 0, "points": 0, "ring_min": None, "ring_max": None, "time_const_frames": 0,
             "empty_frames": 0, "n_rgb": 0, "n_depth": 0}
    writer = AsyncWriter(workers=save_workers)
    try:
        with Reader(bag_path) as reader:
            conns = [c for c in reader.connections if norm(c.topic) in wanted]
            for connection, timestamp, rawdata in reader.messages(connections=conns):
                key = wanted[norm(connection.topic)]
                if key != "lidar" and int(timestamp) not in needed_ns[key]:
                    continue                                   # tidak dipilih sync -> lewati (tanpa decode)
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                t = timestamp / 1e9
                if key == "lidar":
                    arr = decode_pointcloud2(msg)
                    writer.submit(np.save, paths.lidar / f"lidar_{t:.6f}.npy", arr)
                    stats["frames"] += 1
                    stats["points"] += len(arr)
                    if len(arr) == 0:
                        stats["empty_frames"] += 1
                        continue
                    r = arr[:, 4]
                    rmin, rmax = float(r.min()), float(r.max())
                    stats["ring_min"] = rmin if stats["ring_min"] is None else min(stats["ring_min"], rmin)
                    stats["ring_max"] = rmax if stats["ring_max"] is None else max(stats["ring_max"], rmax)
                    if arr[:, 5].max() == arr[:, 5].min():
                        stats["time_const_frames"] += 1
                elif key == "rgb":
                    writer.submit(_save_png, paths.rgb / f"rgb_{t:.6f}.png", decode_image(msg))
                    stats["n_rgb"] += 1
                else:
                    writer.submit(np.save, paths.depth / f"depth_{t:.6f}.npy", decode_image(msg).astype(np.float32))
                    stats["n_depth"] += 1
    finally:
        writer.close()
    return stats


# --------------------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------------------
def run_checkpoints(paths, stats, records, df, n_outside, strict=False):
    ck = Checker("FASE 1", paths.checkpoints / "fase1_report.json", strict=strict)

    # 1) field ring/time LiDAR
    ring_ok = stats["ring_min"] is not None and stats["ring_max"] > stats["ring_min"]
    ck.ok("LiDAR punya field ring (bervariasi) dan time", ring_ok,
          f"ring global {stats['ring_min']}-{stats['ring_max']}, {stats['frames']} frame, "
          f"rata-rata {stats['points'] / max(stats['frames'], 1):.0f} titik")
    ck.ok("Field time LiDAR bervariasi dalam frame", stats["time_const_frames"] == 0,
          f"{stats['time_const_frames']} frame dengan time konstan", fatal=False)
    ck.ok("Tidak ada frame LiDAR kosong", stats["empty_frames"] == 0, f"{stats['empty_frames']} kosong")

    # 2) lat/lon tanpa gap / NaN
    for k, col in (("lat", "lat_interp"), ("lon", "lon_interp")):
        v = np.array([r[1] for r in records[k]])
        t = np.array([r[0] for r in records[k]])
        finite = bool(len(v) and np.isfinite(v).all())
        ck.ok(f"{k}: sampel mentah ada & finite", finite, f"{len(v)} sampel, NaN={int((~np.isfinite(v)).sum())}")
        ck.ok(f"{k}: tabel interpolasi tanpa NaN", int(df[col].isna().sum()) == 0)
        gap = float(np.max(np.diff(t))) if len(t) > 1 else float("inf")
        ck.ok(f"{k}: celah GPS terbesar <= {GPS_MAX_GAP_S}s", gap <= GPS_MAX_GAP_S, f"{gap:.2f}s", fatal=False)
        ck.ok(f"{k}: frame di luar rentang GPS (nilai ujung dipakai)", n_outside[k] == 0,
              f"{n_outside[k]} frame", fatal=False)

    # 3) odom dinamis
    odom = np.array([r[1:] for r in records["odom"]]) if records["odom"] else None
    if odom is None or len(odom) < 2:
        ck.ok("odom ada", False, "kosong")
    else:
        px, py, qz = odom[:, 0], odom[:, 1], odom[:, 5]
        ck.ok("odom px, py dinamis", px.std() > ODOM_MIN_STD or py.std() > ODOM_MIN_STD,
              f"std px={px.std():.3f} py={py.std():.3f}")
        ck.ok("odom qz dinamis (bukan konstan 0)", qz.std() > ODOM_QZ_MIN_STD and not np.allclose(qz, 0),
              f"std qz={qz.std():.4f}")
    ck.finish()


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------
def run(bag, out, tol=SYNC_TOL_S, force=False, save_unmatched=False, strict=False, workers=4):
    paths = Paths(out)
    if paths.gps_csv.exists() and not force:
        print(f"[Fase 1] sudah ada: {paths.gps_csv.name} -> dilewati (pakai --force untuk ulang)")
        return paths
    if not Path(bag).exists():
        raise SystemExit(f"Bag tidak ditemukan: {bag}")
    paths.ensure("lidar", "rgb", "depth", "meta", "checkpoints")

    with Timer("Fase 1 pass 1 (scan bag)"):
        ns, records, _ = scan_bag(bag, paths)
    dump_meta_csv(paths, records)
    df, needed_ns, _ = build_sync_table(paths, ns, records, tol, save_unmatched)
    with Timer("Fase 1 pass 2 (tulis lidar/rgb/depth)"):
        stats = write_sensor_files(bag, paths, needed_ns, workers)
    print(f"Ditulis: {stats['frames']} lidar, {stats['n_rgb']} rgb, {stats['n_depth']} depth")
    df, n_outside = add_gps_interp(paths, df, records)
    print(f"sync_table_with_gps.csv: lat/lon NaN = {int(df['lat_interp'].isna().sum() + df['lon_interp'].isna().sum())}")
    run_checkpoints(paths, stats, records, df, n_outside, strict=strict)
    df.to_csv(paths.gps_csv, index=False)                 # penanda fase selesai; hanya ada bila checkpoint LULUS
    return paths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bag", required=True)
    p.add_argument("--out", default="./output")
    p.add_argument("--list", action="store_true", help="Hanya list topic & jumlah pesan")
    p.add_argument("--tol", type=float, default=SYNC_TOL_S)
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-unmatched", action="store_true", help="Simpan semua rgb/depth (perilaku lama)")
    p.add_argument("--strict", action="store_true", help="Peringatan checkpoint dianggap gagal")
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.list:
        list_topics(a.bag)
    else:
        run(a.bag, a.out, a.tol, a.force, a.save_unmatched, a.strict, a.workers)
