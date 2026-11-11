#!/usr/bin/env python3
r"""
fase1b_waypoints.py
===================
FASE 1b -- Waypoint ke ego-frame (x = depan, y = kiri; y negatif = belok kanan).

INPUT   : <out>/sync_table_with_gps.csv  (lat_interp, lon_interp, odom_data, lidar_t, lidar_path)
PROSES  : lat/lon -> meter lokal (ENU) ; heading dari odom (yaw = 2*atan2(qz,qw)) dikalibrasi ke ENU lewat
          bearing GPS ; titik jalur yang BENAR-BENAR dilewati mobil 20 m ke depan, tiap 0,5 m, diputar &
          digeser ke ego-frame tiap frame. Tanpa ekstrapolasi (di ujung rekaman titik dipotong).
OUTPUT  : <out>/waypoints/waypoint_<t>.npy        (Nx2 float32 [x_ego, y_ego])
          <out>/waypoints_per_frame.json          {idx: [[x,y],...]}  <- dibaca langsung oleh Fase 5
          <out>/sync_table_with_waypoints.csv     (+ global_x, global_y, yaw_odom_raw, yaw_enu,
                                                   waypoint_path, n_waypoints)
          <out>/checkpoints/fase1b_report.json   (+ PNG bila --debug-plots)
CHECKPOINT (otomatis, pengganti cek visual):
          - offset heading konsisten (sebaran bearing GPS - yaw odom kecil)
          - waypoint berada di depan mobil
          - waypoint melengkung ke arah yang sama dengan perubahan heading saat mobil belok
          - trajectory global masuk akal (panjang, tanpa loncatan)

PAKAI:  python fase1b_waypoints.py --out /path/output_rute02 [--debug-plots]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HEADING_LOOKAHEAD_FRAMES, HEADING_MIN_DIST_M, N_WAYPOINTS_FULL, R_EARTH,
                    SMOOTH_SIGMA_FRAMES, WAYPOINT_HORIZON_M, WAYPOINT_SPACING_M, Checker, Paths, Timer)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


_NP_WRAP = re.compile(r"np\.float\d+\(([^)]*)\)")


def parse_odom_qz_qw(s):
    """odom_data = str(tuple(px,py,pz,qx,qy,qz,qw,...)). Tahan terhadap np.float64(...)/nan/inf."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return np.nan, np.nan
    parts = _NP_WRAP.sub(r"\1", str(s)).strip().strip("()[]").split(",")
    v = [float(p.strip()) for p in parts if p.strip()]
    return v[5], v[6]


# --------------------------------------------------------------------------------------
# Inti perhitungan (fungsi murni -> mudah diuji)
# --------------------------------------------------------------------------------------
def latlon_to_local_xy(lat, lon):
    """Equirectangular terhadap titik pertama (cukup akurat untuk rute ratusan meter)."""
    lat0, lon0 = lat[0], lon[0]
    x = R_EARTH * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = R_EARTH * np.radians(lat - lat0)
    return x, y


def odom_yaw(df):
    """yaw odom per frame; frame tanpa odom diinterpolasi (setelah unwrap supaya tak loncat di +-180)."""
    qq = df["odom_data"].apply(parse_odom_qz_qw)
    qz = np.array([a for a, _ in qq], float)
    qw = np.array([b for _, b in qq], float)
    yaw_raw = 2.0 * np.arctan2(qz, qw)
    valid = np.isfinite(yaw_raw)
    if valid.sum() < 2:
        raise SystemExit("Data odom terlalu sedikit untuk heading.")
    lt = df["lidar_t"].to_numpy()
    return np.interp(lt, lt[valid], np.unwrap(yaw_raw[valid])), int((~valid).sum())


def calibrate_heading_offset(gx, gy, yaw_odom, k=HEADING_LOOKAHEAD_FRAMES, min_dist=HEADING_MIN_DIST_M):
    """offset = median sirkular (bearing GPS - yaw odom) pada frame yang benar-benar bergerak."""
    n = len(gx)
    bearing = np.full(n, np.nan)
    if n > k:
        dx, dy = gx[k:] - gx[:-k], gy[k:] - gy[:-k]
        moving = np.hypot(dx, dy) >= min_dist
        bearing[:n - k][moving] = np.arctan2(dy, dx)[moving]
    m = ~np.isnan(bearing)
    if m.sum() < 10:
        raise SystemExit("Terlalu sedikit frame bergerak untuk kalibrasi heading (mobil diam / GPS rusak).")
    cand = wrap_angle(bearing[m] - yaw_odom[m])
    offset = float(np.arctan2(np.median(np.sin(cand)), np.median(np.cos(cand))))
    spread = float(np.std(wrap_angle(cand - offset)))
    return offset, spread, int(m.sum())


def build_all_waypoints(gx, gy, yaw, horizon_m=WAYPOINT_HORIZON_M, spacing_m=WAYPOINT_SPACING_M,
                        sigma=SMOOTH_SIGMA_FRAMES, return_extra=False):
    """Waypoint ego-frame per frame (list ndarray Nx2). Algoritma identik dengan Fase 5 v29:
    posisi dihaluskan (Gaussian), diparameterkan panjang busur s, titik tiap `spacing_m` sampai `horizon_m`,
    diputar ke ego-frame frame tsb. Sisa jalur < horizon -> titik dipotong (tidak ditebak)."""
    from scipy.ndimage import gaussian_filter1d
    xs = gaussian_filter1d(gx, sigma, mode="nearest")
    ys = gaussian_filter1d(gy, sigma, mode="nearest")
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]
    s = s + np.arange(len(s)) * 1e-6                     # s naik tegas (frame diam -> jarak 0)
    offsets = np.arange(spacing_m, horizon_m + 1e-9, spacing_m)
    out, last_s = [], []
    for k in range(len(gx)):
        s_t = s[k] + offsets
        s_t = s_t[s_t <= s[-1]]
        if len(s_t) == 0:
            out.append(np.zeros((0, 2), np.float32))
            last_s.append(None)
            continue
        dx, dy = np.interp(s_t, s, xs) - xs[k], np.interp(s_t, s, ys) - ys[k]
        c, sn = np.cos(yaw[k]), np.sin(yaw[k])
        out.append(np.column_stack([c * dx + sn * dy, -sn * dx + c * dy]).astype(np.float32))
        last_s.append(s_t[-1])
    if return_extra:
        return out, s, last_s
    return out


# --------------------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------------------
def run_checkpoints(paths, df, gx, gy, lt, yaw_enu, wps, s, last_s, off_deg, spread_deg, n_used, strict=False):
    n = len(df)
    ck = Checker("FASE 1b", paths.checkpoints / "fase1b_report.json", strict=strict)

    step = np.hypot(np.diff(gx), np.diff(gy))
    ck.ok("Trajectory: panjang total > 5 m", step.sum() > 5.0, f"{step.sum():.1f} m")
    ck.ok("Trajectory: tanpa loncatan GPS > 3 m antar frame", float(step.max()) <= 3.0,
          f"loncatan maks {step.max():.2f} m", fatal=False)
    ck.ok("Heading: frame bergerak cukup untuk kalibrasi", n_used >= 0.1 * n, f"{n_used}/{n} frame")
    ck.ok("Heading: sebaran offset kecil", spread_deg <= 20.0,
          f"offset {off_deg:.1f} deg, std {spread_deg:.2f} deg (target <= 8)", fatal=True)
    ck.ok("Heading: sebaran offset ideal (<= 8 deg)", spread_deg <= 8.0, f"{spread_deg:.2f} deg", fatal=False)

    nonempty = [w for w in wps if len(w)]
    allp = np.vstack(nonempty) if nonempty else np.zeros((0, 2))
    fwd = float((allp[:, 0] > 0).mean()) if len(allp) else 0.0
    ck.ok("Waypoint ada & >= 95% titik di depan mobil (x_ego > 0)", fwd >= 0.95, f"{100 * fwd:.1f}%")
    empties = [i for i, w in enumerate(wps) if len(w) == 0]
    # kosong boleh berupa SATU blok menyambung sampai ujung (mobil berhenti di akhir rekaman); kosong di tengah = masalah
    tail_only = (empties == list(range(empties[0], n))) if empties else True
    ck.ok("Frame tanpa waypoint hanya di ujung rekaman", tail_only, f"{len(empties)} frame kosong", fatal=False)

    # melengkung searah belok: bandingkan tanda y_ego titik terjauh dgn perubahan heading sepanjang jalur itu
    agree = tot = 0
    for k, w in enumerate(wps):
        if len(w) < N_WAYPOINTS_FULL or last_s[k] is None:
            continue
        j = min(int(np.searchsorted(s, last_s[k])), n - 1)
        dpsi = float(wrap_angle(yaw_enu[j] - yaw_enu[k]))
        if abs(dpsi) > np.radians(20):
            tot += 1
            agree += int(np.sign(w[-1][1]) == np.sign(dpsi) and abs(w[-1][1]) > 0.5)
    if tot:
        ck.ok("Waypoint melengkung searah belok (heading berubah > 20 deg)", agree / tot >= 0.9,
              f"{agree}/{tot} frame", fatal=False)
    else:
        ck.info("Waypoint vs belok", "tidak ada tikungan > 20 deg di frame berhorizon penuh -> cek tidak berlaku")
    ck.finish()


def debug_plots(paths, gx, gy, lt, yaw_enu, wps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths.ensure("checkpoints")
    n = len(gx)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(gx, gy, c=lt - lt[0], s=6, cmap="viridis")
    plt.colorbar(sc, label="waktu (s)")
    plt.axis("equal"); plt.grid(alpha=.3); plt.title("Trajectory global GPS")
    plt.savefig(paths.checkpoints / "trajectory_global.png", dpi=110); plt.close()
    step = 40
    plt.figure(figsize=(7, 7))
    plt.scatter(gx, gy, c="lightgray", s=4)
    plt.quiver(gx[::step], gy[::step], np.cos(yaw_enu[::step]), np.sin(yaw_enu[::step]),
               color="crimson", scale=25, width=0.004)
    plt.axis("equal"); plt.grid(alpha=.3); plt.title("Heading terkalibrasi vs lintasan")
    plt.savefig(paths.checkpoints / "heading_check.png", dpi=110); plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, i in zip(axes.flat, np.linspace(min(50, n - 1), max(n - 50, 0), 6).astype(int)):
        p = wps[i]
        ax.scatter(0, 0, c="k", marker="s"); ax.arrow(0, 0, 1.5, 0, head_width=0.3, color="k")
        if len(p):
            ax.plot(p[:, 0], p[:, 1], "-o", c="crimson", ms=3)
        ax.set_title(f"frame {i}"); ax.axis("equal"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(paths.checkpoints / "waypoints_samples.png", dpi=100); plt.close()


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------
def run(out, force=False, plots=False, strict=False):
    paths = Paths(out)
    if paths.wp_csv.exists() and paths.wp_json.exists() and not force:
        print(f"[Fase 1b] sudah ada: {paths.wp_csv.name} -> dilewati (pakai --force untuk ulang)")
        return paths
    if not paths.gps_csv.exists():
        raise SystemExit(f"Tidak ketemu {paths.gps_csv} -- jalankan Fase 1 dulu.")
    paths.ensure("waypoints", "checkpoints")

    df = pd.read_csv(paths.gps_csv)
    n = len(df)
    print(f"Total frame: {n}")
    lt = df["lidar_t"].to_numpy()

    with Timer("Fase 1b"):
        gx, gy = latlon_to_local_xy(df["lat_interp"].to_numpy(), df["lon_interp"].to_numpy())
        df["global_x"], df["global_y"] = gx, gy
        yaw_odom, n_no_odom = odom_yaw(df)
        df["yaw_odom_raw"] = yaw_odom
        print(f"Frame tanpa odom (diinterpolasi): {n_no_odom}")
        offset, spread, n_used = calibrate_heading_offset(gx, gy, yaw_odom)
        print(f"Offset heading: {np.degrees(offset):.2f} deg | frame terpakai {n_used}/{n} | std {np.degrees(spread):.2f} deg")
        yaw_enu = wrap_angle(yaw_odom + offset)
        df["yaw_enu"] = yaw_enu

        wps, s, last_s = build_all_waypoints(gx, gy, yaw_enu, return_extra=True)

        wp_paths, wp_json = [], {}
        for i, w in enumerate(wps):
            stem = Path(str(df["lidar_path"].iloc[i]).replace("\\", "/")).stem.replace("lidar_", "waypoint_", 1)
            f = paths.waypoints / f"{stem}.npy"
            np.save(f, w)
            wp_paths.append(str(f))
            wp_json[int(df["idx"].iloc[i])] = np.round(w, 3).tolist()
        df["waypoint_path"] = wp_paths
        df["n_waypoints"] = [len(w) for w in wps]
        df.to_csv(paths.wp_csv, index=False)
        with open(paths.wp_json, "w") as f:
            json.dump(wp_json, f, separators=(",", ":"))
    print("Tersimpan: waypoints/*.npy, waypoints_per_frame.json, sync_table_with_waypoints.csv")

    if plots:
        debug_plots(paths, gx, gy, lt, yaw_enu, wps)
        print("Plot debug ->", paths.checkpoints)
    run_checkpoints(paths, df, gx, gy, lt, yaw_enu, wps, s, last_s, np.degrees(offset), np.degrees(spread), n_used, strict)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--debug-plots", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    run(a.out, a.force, a.debug_plots, a.strict)
