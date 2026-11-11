#!/usr/bin/env python3
"""
common.py
=========
Konfigurasi, path, dan utilitas checkpoint yang dipakai bersama oleh semua fase
pipeline BEV (rosbag -> payload Flutter).

Satu sumber kebenaran untuk: nama topic, toleransi sinkronisasi, parameter waypoint,
struktur folder output, dan mekanisme checkpoint otomatis (pengganti plot manual).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path, PurePosixPath

# --------------------------------------------------------------------------------------
# Rosbag
# --------------------------------------------------------------------------------------
TOPICS = {
    "lidar": "/velodyne_points",
    "rgb": "/zed2i/zed_node/left/image_rect_color",
    "depth": "/zed2i/zed_node/depth/depth_registered",
    "camera_info": "/zed2i/zed_node/left/camera_info",
    "imu": "/imu",
    "odom": "/zed2i/zed_node/odom",
    "lat": "/latitude",
    "lon": "/longitude",
    "velocity": "/velocity",
    "yaw": "/yaw_displacement_for_dataset",
}

# urutan kolom di sync_table.csv (anchor = lidar). 3 + 8*3 = 27 kolom, sama seperti versi lama.
SYNC_ORDER = ("rgb", "depth", "imu", "odom", "lat", "lon", "velocity", "yaw")
SYNC_TOL_S = 0.05

# Checkpoint Fase 1
GPS_MAX_GAP_S = 2.0          # celah antar sampel GPS mentah lebih dari ini -> peringatan
ODOM_MIN_STD = 1e-3          # std px/py minimum agar odom dianggap "dinamis"
ODOM_QZ_MIN_STD = 1e-4       # std qz minimum (bukan konstan / 0)

# --------------------------------------------------------------------------------------
# Waypoint (Fase 1b) -- algoritma Fase 5 v29: horizon dalam METER, spasi rata, tanpa ekstrapolasi
# --------------------------------------------------------------------------------------
R_EARTH = 6371000.0
HEADING_LOOKAHEAD_FRAMES = 5     # bearing GPS = posisi sekarang vs 5 frame ke depan (~0.5 s @10 Hz)
HEADING_MIN_DIST_M = 0.5         # di bawah ini bearing GPS dianggap terlalu noisy
WAYPOINT_HORIZON_M = 20.0
WAYPOINT_SPACING_M = 0.5
SMOOTH_SIGMA_FRAMES = 3.0        # haluskan jitter GPS (~0.1 m rms), 3 frame ~ 0.3 s
N_WAYPOINTS_FULL = int(round(WAYPOINT_HORIZON_M / WAYPOINT_SPACING_M))   # 40


# --------------------------------------------------------------------------------------
# Path
# --------------------------------------------------------------------------------------
class Paths:
    """Struktur folder output (<out> = folder output_rute02)."""

    def __init__(self, out):
        self.out = Path(out)
        self.lidar = self.out / "lidar"
        self.rgb = self.out / "rgb"
        self.depth = self.out / "depth"
        self.meta = self.out / "meta"
        self.waypoints = self.out / "waypoints"
        self.checkpoints = self.out / "checkpoints"
        self.calib = self.out / "calib"

        self.camera_info = self.out / "camera_info.json"
        self.sync_csv = self.out / "sync_table.csv"
        self.gps_csv = self.out / "sync_table_with_gps.csv"
        self.wp_csv = self.out / "sync_table_with_waypoints.csv"
        self.wp_json = self.out / "waypoints_per_frame.json"
        self.calib_json = self.calib / "calib_config.json"

    def ensure(self, *names):
        for n in names:
            getattr(self, n).mkdir(parents=True, exist_ok=True)


def resolve_by_name(base_dir, stored_path, subfolder):
    """Path di sync_table.csv bisa path Linux (WSL). Ambil nama file saja lalu gabung ke base_dir.
    Dipakai fase 2-5 supaya pipeline jalan di Windows maupun WSL."""
    if not stored_path or (isinstance(stored_path, float)):   # '' / NaN
        return None
    return Path(base_dir) / subfolder / PurePosixPath(str(stored_path).replace("\\", "/")).name


# --------------------------------------------------------------------------------------
# Checkpoint otomatis
# --------------------------------------------------------------------------------------
class CheckpointError(RuntimeError):
    """Checkpoint fatal gagal -> pipeline berhenti (supaya fase berikutnya tidak memproses data rusak)."""


class Checker:
    """Kumpulkan hasil checkpoint satu fase, cetak ringkas, simpan JSON, dan hentikan pipeline bila gagal.

    ok(..., fatal=True)  : gagal -> FAIL (menghentikan pipeline di finish()).
    ok(..., fatal=False) : gagal -> WARN (tidak menghentikan, kecuali strict=True).
    """

    def __init__(self, phase, report_path=None, strict=False):
        self.phase, self.report_path, self.strict = phase, report_path, strict
        self.items = []

    def ok(self, name, cond, detail="", fatal=True):
        status = "OK" if cond else ("FAIL" if fatal else "WARN")
        self.items.append({"name": name, "status": status, "detail": str(detail)})
        return bool(cond)

    def info(self, name, detail=""):
        self.items.append({"name": name, "status": "INFO", "detail": str(detail)})

    def finish(self):
        print(f"\n=== CHECKPOINT {self.phase} ===")
        for it in self.items:
            print(f"[{it['status']:<4}] {it['name']}" + (f": {it['detail']}" if it["detail"] else ""))
        fails = [i for i in self.items if i["status"] == "FAIL"]
        warns = [i for i in self.items if i["status"] == "WARN"]
        if self.report_path:
            Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.report_path, "w", encoding="utf-8") as f:
                json.dump({"phase": self.phase, "items": self.items,
                           "n_fail": len(fails), "n_warn": len(warns)}, f, indent=2, ensure_ascii=False)
        if fails or (self.strict and warns):
            bad = fails + (warns if self.strict else [])
            raise CheckpointError(f"{self.phase}: {len(bad)} checkpoint gagal -> " +
                                  "; ".join(i["name"] for i in bad))
        print(f"-> {self.phase}: LULUS ({len(warns)} peringatan)")


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        print(f"[waktu] {self.label}: {time.time() - self.t0:.1f} s")
        sys.stdout.flush()
