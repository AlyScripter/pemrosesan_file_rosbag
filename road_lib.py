"""
road_lib.py -- Fase 1: masker jalan BEV dari LiDAR semantik + akumulasi scan.

Alur inti
---------
1. Baca scan LiDAR (.npy) + label semantik per titik (Fase 3 lama).
   Struktur data (BEV_ROAD):  lidar/lidar_<detik>.<pecahan>.npy      -> (N,6) x,y,z,intensity,ring,time
                              labels/points/<idx:06d>.npy            -> (N,7) = 6 kolom di atas + label (kol. 7)
                              meta/odom.csv                          -> t,px,py,pz,qx,qy,qz,qw,...
2. Ambil titik kelas Road (opsional: + titik Unknown yang menempel di ketinggian tanah).
3. Akumulasikan scan-scan sebelumnya ke frame LiDAR saat ini memakai gerak ego (odom, SE(2)).
4. Rasterisasi ke grid BEV -> masker jalan (closing + buang komponen kecil).
5. Evaluasi tanpa ground truth: "path recall" (jalur yang benar-benar dilewati mobil
   pasti jalan) + validasi odom dengan ICP.

Konvensi sumbu (frame LiDAR): x = depan, y = kiri, z = atas.
Grid disimpan sebagai array [ix, iy] (nx, ny).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

_STEM_RE = re.compile(r"(\d{9,})[._](\d+)$")   # lidar_1765606289.870195 / lidar_1765606289_870195


# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
@dataclass
class Config:
    root: Path
    lidar_dir: Optional[Path] = None
    label_dir: Optional[Path] = None
    odom_csv: Optional[Path] = None
    out_dir: Optional[Path] = None

    # ID kelas -- WAJIB dicocokkan dengan class_color_map.json / label Fase 3 kamu.
    # Di data kamu (frame 0) hanya muncul ID 0..3; ID 4/5 (Vehicle/Person) mungkin muncul di frame lain.
    class_ids: Dict[str, int] = field(default_factory=lambda: {
        "Unknown": 0, "Road": 1, "Vegetation": 2, "Building": 3, "Vehicle": 4, "Person": 5,
    })
    # Pola nama file label (relatif ke label_dir). {idx} = urutan scan (terurut waktu, sama dengan
    # kolom idx di sync_table.csv), {stem} = nama file lidar tanpa .npy.
    label_patterns: Tuple[str, ...] = ("{idx:06d}.npy", "{stem}.npy", "{stem}_labels.npy", "{stem}_label.npy")
    # Jika file label 2D (N,>=4): kolom mana yang berisi label (default -1 = kolom terakhir).
    label_col: int = -1

    # Grid BEV (meter, frame LiDAR: x depan, y kiri)
    x_range: Tuple[float, float] = (-15.0, 40.0)
    y_range: Tuple[float, float] = (-20.0, 20.0)
    res: float = 0.2
    min_range: float = 1.5
    max_range: float = 60.0

    # Pose ego (SE(2) dari odom)
    lidar_in_base: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x [m], y [m], yaw [deg]
    odom_time_offset: float = 0.0   # detik, ditambahkan ke timestamp LiDAR sebelum interpolasi odom
    drop_identity_prefix: bool = True  # buang baris awal odom berpose identitas (ZED belum tracking)
    max_pose_gap_s: float = 0.5     # frame ditandai tidak valid jika sampel odom terdekat > ini

    # Akumulasi
    window_past_s: float = 4.0      # cukup panjang agar blind zone VLP-16 (h/tan15deg) terisi saat mobil bergerak
    window_future_s: float = 0.0    # 0 = hanya masa lalu (jujur untuk klaim real-time)
    min_frames: int = 2             # sel jalan harus terlihat >= N frame berbeda (akumulasi)
    unknown_max_xy_dist: float = 2.0
    unknown_z_tol: float = 0.15

    # Pasca-proses masker
    close_radius_m: float = 0.8
    min_component_m2: float = 4.0
    max_hole_m2: float = 2.0

    # Evaluasi jalur ego
    ego_half_width_m: float = 0.8
    eval_past_s: float = 2.0
    eval_future_s: float = 3.0

    # Target checkpoint (awal -- boleh disesuaikan)
    target_recall_past: float = 0.90
    target_recall_future: float = 0.60
    target_gain: float = 1.5
    target_icp_median_shift_m: float = 0.20
    target_icp_median_yaw_deg: float = 1.0

    def __post_init__(self):
        self.root = Path(self.root)
        self.lidar_dir = Path(self.lidar_dir) if self.lidar_dir else self.root / "lidar"
        self.label_dir = Path(self.label_dir) if self.label_dir else self.root / "labels" / "points"
        self.odom_csv = Path(self.odom_csv) if self.odom_csv else self.root / "meta" / "odom.csv"
        self.out_dir = Path(self.out_dir) if self.out_dir else self.root / "output_fase1"

    @property
    def nx(self) -> int:
        return int(round((self.x_range[1] - self.x_range[0]) / self.res))

    @property
    def ny(self) -> int:
        return int(round((self.y_range[1] - self.y_range[0]) / self.res))


# ----------------------------------------------------------------------------
# SE(2)
# ----------------------------------------------------------------------------
def se2(x, y, yaw) -> np.ndarray:
    """Matriks homogen 3x3 (batch). Input skalar atau array 1D -> (N,3,3)."""
    x, y, yaw = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float), np.atleast_1d(yaw).astype(float)
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.zeros((len(x), 3, 3))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 2] = c, -s, x
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2] = s, c, y
    T[:, 2, 2] = 1.0
    return T


def apply_se2(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Terapkan SE(2) ke titik (N,2) atau (N,3); z tidak diubah."""
    out = pts.copy()
    out[:, :2] = pts[:, :2] @ T[:2, :2].T + T[:2, 2]
    return out


def se2_to_se3(T: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:2, :2] = T[:2, :2]
    M[:2, 3] = T[:2, 2]
    return M


# ----------------------------------------------------------------------------
# Odom & pose
# ----------------------------------------------------------------------------
@dataclass
class Odom:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray  # rad, sudah di-unwrap
    n_dropped: int = 0  # baris awal berpose identitas yang dibuang


_ODOM_ALIASES = {"x": ("x", "px"), "y": ("y", "py"), "z": ("z", "pz")}


def load_odom(csv_path: Path, drop_identity_prefix: bool = True) -> Odom:
    """
    CSV odom ZED. Kolom yang diterima: t, (x|px), (y|py), qx, qy, qz, qw  (kolom lain diabaikan).
    Baris AWAL berpose identitas persis (xyz = 0 dan quaternion identitas) = ZED belum mulai tracking;
    pose itu bukan pengukuran, jadi dibuang bila drop_identity_prefix=True.
    """
    df = pd.read_csv(csv_path, float_precision="round_trip")
    col = {}
    for k, cands in _ODOM_ALIASES.items():
        hit = [c for c in cands if c in df.columns]
        if hit:
            col[k] = hit[0]
    need_q = {"t", "qx", "qy", "qz", "qw"}
    if "x" not in col or "y" not in col or not need_q.issubset(df.columns):
        raise ValueError(f"odom.csv butuh kolom t, x|px, y|py, qx,qy,qz,qw; ditemukan {list(df.columns)}")
    df = df.sort_values("t").drop_duplicates("t").reset_index(drop=True)
    n_drop = 0
    if drop_identity_prefix:
        ident = ((df[[col["x"], col["y"]] + ([col["z"]] if "z" in col else [])] == 0).all(axis=1)
                 & (df[["qx", "qy", "qz"]] == 0).all(axis=1) & (df["qw"] == 1))
        if ident.iloc[0]:
            n_drop = int(np.argmax(~ident.to_numpy())) if (~ident).any() else len(df)
            df = df.iloc[n_drop:].reset_index(drop=True)
    qx, qy, qz, qw = (df[c].to_numpy() for c in ("qx", "qy", "qz", "qw"))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return Odom(df["t"].to_numpy(), df[col["x"]].to_numpy(), df[col["y"]].to_numpy(), np.unwrap(yaw), n_drop)


def build_lidar_poses(t_lidar: np.ndarray, odom: Odom, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Pose T_world_lidar (N,3,3) per frame LiDAR + flag validitas."""
    tq = t_lidar + cfg.odom_time_offset
    x = np.interp(tq, odom.t, odom.x)
    y = np.interp(tq, odom.t, odom.y)
    yaw = np.interp(tq, odom.t, odom.yaw)
    T_w_b = se2(x, y, yaw)
    bx, by, byaw = cfg.lidar_in_base
    T_b_l = se2(bx, by, np.deg2rad(byaw))[0]
    poses = T_w_b @ T_b_l

    k = np.searchsorted(odom.t, tq).clip(1, len(odom.t) - 1)
    gap = np.minimum(np.abs(tq - odom.t[k]), np.abs(tq - odom.t[k - 1]))
    valid = (tq >= odom.t[0]) & (tq <= odom.t[-1]) & (gap <= cfg.max_pose_gap_s)
    return poses, valid


def pose_valid_stats(valid: np.ndarray) -> dict:
    """Frame sebelum pose valid pertama (rekaman mulai sebelum ZED tracking) dipisahkan dari celah di tengah."""
    if not valid.any():
        return {"first_valid": -1, "n_before": len(valid), "n_gaps_after": 0, "frac_after_first": 0.0}
    f = int(np.argmax(valid))
    return {"first_valid": f, "n_before": f, "n_gaps_after": int((~valid[f:]).sum()),
            "frac_after_first": float(valid[f:].mean())}


# ----------------------------------------------------------------------------
# Data scan
# ----------------------------------------------------------------------------
class Sequence:
    """Daftar scan LiDAR terurut waktu + loader label + cache titik jalan."""

    def __init__(self, cfg: Config, cache_cap: int = 512):
        self.cfg = cfg
        files = sorted(cfg.lidar_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"Tidak ada *.npy di {cfg.lidar_dir}")
        stamped, skipped = [], 0
        for f in files:
            m = _STEM_RE.search(f.stem)
            if not m:
                skipped += 1
                continue
            sec, frac = m.groups()
            stamped.append((int(sec) + int(frac) / 10 ** len(frac), f))
        if not stamped:
            raise ValueError("Nama file lidar harus berakhiran <detik>.<pecahan>, mis. lidar_1765606289.870195.npy")
        if skipped:
            print(f"[peringatan] {skipped} file dilewati (nama tidak berpola <detik>.<pecahan>)")
        stamped.sort(key=lambda a: a[0])
        self.t = np.array([a[0] for a in stamped], dtype=np.float64)
        self.files: List[Path] = [a[1] for a in stamped]
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_cap = cache_cap

    def __len__(self) -> int:
        return len(self.files)

    def index_from_time(self, t: float) -> int:
        """Indeks frame terdekat dengan timestamp t (detik epoch)."""
        return int(np.abs(self.t - t).argmin())

    # -- loader mentah ----------------------------------------------------
    def load_xyz(self, i: int) -> np.ndarray:
        arr = np.load(self.files[i], allow_pickle=False)
        if arr.dtype.names:
            return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"{self.files[i].name}: shape {arr.shape} tidak dikenali (butuh (N, >=3))")
        return arr[:, :3].astype(np.float32)

    def label_path(self, i: int) -> Optional[Path]:
        stem = self.files[i].stem
        for pat in self.cfg.label_patterns:
            p = self.cfg.label_dir / pat.format(stem=stem, idx=i)
            if p.exists():
                return p
        return None

    def load_labels(self, i: int, xyz: np.ndarray) -> np.ndarray:
        """
        Label per titik. Mendukung file 1D (N,) atau 2D (N,>=4) [xyz... + label di kolom cfg.label_col].
        Untuk file 2D, xyz di file label dibandingkan dengan xyz scan -> menangkap label yang salah indeks.
        """
        p = self.label_path(i)
        if p is None:
            raise FileNotFoundError(
                f"Label untuk {self.files[i].name} (idx {i}) tidak ditemukan di {self.cfg.label_dir} "
                f"(pola: {self.cfg.label_patterns})")
        arr = np.load(p, allow_pickle=False)
        n = len(xyz)
        if arr.ndim == 2:
            if arr.shape[0] != n or arr.shape[1] < 4:
                raise ValueError(f"{p.name}: shape {arr.shape} tidak cocok dengan {n} titik di {self.files[i].name}")
            if not np.allclose(arr[:, :3], xyz, atol=1e-4, equal_nan=True):
                raise ValueError(f"{p.name}: xyz di file label != xyz {self.files[i].name}. "
                                 "Pasangan label-scan salah (indeks geser?).")
            lab = arr[:, self.cfg.label_col]
        else:
            lab = arr.reshape(-1)
            if lab.shape[0] != n:
                raise ValueError(f"{p.name}: {lab.shape[0]} label != {n} titik di {self.files[i].name}. "
                                 "Label harus 1:1 dengan titik scan (urutan sama).")
        return np.rint(lab).astype(np.int16)

    def get_scan(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        xyz = self.load_xyz(i)
        lab = self.load_labels(i, xyz)
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        ok = np.isfinite(xyz).all(1) & (r >= self.cfg.min_range) & (r <= self.cfg.max_range)
        return xyz[ok], lab[ok]

    # -- titik jalan (cache) ----------------------------------------------
    def road_points(self, i: int, include_unknown: bool) -> np.ndarray:
        if i not in self._cache:
            xyz, lab = self.get_scan(i)
            pair = (extract_road_points(xyz, lab, self.cfg, False),
                    extract_road_points(xyz, lab, self.cfg, True))
            if len(self._cache) >= self._cache_cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = pair
        return self._cache[i][1 if include_unknown else 0]


def extract_road_points(xyz: np.ndarray, lab: np.ndarray, cfg: Config, include_unknown: bool) -> np.ndarray:
    """Titik Road; opsional + titik Unknown yang berdekatan (xy) dan setinggi titik Road."""
    rid, uid = cfg.class_ids["Road"], cfg.class_ids["Unknown"]
    road = xyz[lab == rid]
    if not include_unknown or len(road) < 20:
        return road
    unk = xyz[lab == uid]
    if len(unk) == 0:
        return road
    tree = cKDTree(road[:, :2])
    d, k = tree.query(unk[:, :2], k=1, distance_upper_bound=cfg.unknown_max_xy_dist)
    ok = np.isfinite(d)
    if not ok.any():
        return road
    zok = np.abs(unk[ok, 2] - road[k[ok], 2]) <= cfg.unknown_z_tol
    return np.vstack([road, unk[ok][zok]])


# ----------------------------------------------------------------------------
# Grid & akumulasi
# ----------------------------------------------------------------------------
def _cells(xy: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    ix = np.floor((xy[:, 0] - cfg.x_range[0]) / cfg.res).astype(np.int64)
    iy = np.floor((xy[:, 1] - cfg.y_range[0]) / cfg.res).astype(np.int64)
    ok = (ix >= 0) & (ix < cfg.nx) & (iy >= 0) & (iy < cfg.ny)
    return ix[ok], iy[ok]


def window_indices(t: np.ndarray, i: int, past_s: float, future_s: float, valid: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(t, t[i] - past_s, side="left")
    hi = np.searchsorted(t, t[i] + future_s, side="right")
    idx = np.arange(lo, hi)
    return idx[valid[idx]]


def accumulate(seq: Sequence, poses: np.ndarray, valid: np.ndarray, i: int, cfg: Config,
               include_unknown: bool, single: bool = False) -> Tuple[np.ndarray, int, int]:
    """
    Grid 'hits' (nx,ny): jumlah FRAME berbeda yang mengenai tiap sel (bukan jumlah titik,
    supaya near-field yang padat tidak mendominasi dan mobil diam tidak memalsukan keyakinan).
    Return (hits, jumlah_titik_dipakai, jumlah_frame_dipakai).
    """
    if single or not valid[i]:
        idx = np.array([i])
    else:
        idx = window_indices(seq.t, i, cfg.window_past_s, cfg.window_future_s, valid)
        if i not in idx:
            idx = np.append(idx, i)
    Ti_inv = np.linalg.inv(poses[i])
    hits = np.zeros((cfg.nx, cfg.ny), np.uint16)
    n_pts = 0
    for j in idx:
        pts = seq.road_points(int(j), include_unknown)
        if len(pts) == 0:
            continue
        p = apply_se2(Ti_inv @ poses[j], pts[:, :2])
        ix, iy = _cells(p, cfg)
        if len(ix) == 0:
            continue
        hits.flat[np.unique(ix * cfg.ny + iy)] += 1
        n_pts += len(ix)
    return hits, n_pts, len(idx)


def _disk(r: int) -> np.ndarray:
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _remove_small(mask: np.ndarray, min_cells: float) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.concatenate([[False], sizes >= min_cells])
    return keep[lab]


def _fill_small_holes(mask: np.ndarray, max_cells: float) -> np.ndarray:
    inv = ~mask
    lab, n = ndi.label(inv)
    if n == 0:
        return mask
    sizes = ndi.sum(inv, lab, index=np.arange(1, n + 1))
    border = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    fill = sizes <= max_cells
    fill[border[border > 0] - 1] = False
    return mask | np.concatenate([[False], fill])[lab]


def make_mask(hits: np.ndarray, min_frames: int, cfg: Config) -> np.ndarray:
    m = hits >= min_frames
    if not m.any():
        return m
    r = int(round(cfg.close_radius_m / cfg.res))
    if r >= 1:
        pad = r + 1
        mp = ndi.binary_closing(np.pad(m, pad), structure=_disk(r))
        m = mp[pad:-pad, pad:-pad]
    m = _fill_small_holes(m, cfg.max_hole_m2 / cfg.res ** 2)
    return _remove_small(m, cfg.min_component_m2 / cfg.res ** 2)


# ----------------------------------------------------------------------------
# Jalur ego (evaluasi tanpa ground truth)
# ----------------------------------------------------------------------------
def ego_path_xy(poses, valid, t, i, past_s, future_s) -> np.ndarray:
    """Posisi ego (frame LiDAR-i) untuk frame di jendela [t_i - past, t_i + future]. (K,2)"""
    idx = window_indices(t, i, past_s, future_s, valid)
    Ti_inv = np.linalg.inv(poses[i])
    rel = Ti_inv[None] @ poses[idx]
    return rel[:, :2, 2]


def _swath(path_xy: np.ndarray, cfg: Config) -> np.ndarray:
    g = np.zeros((cfg.nx, cfg.ny), bool)
    if len(path_xy) == 0:
        return g
    pts = [path_xy[:1]]
    for a, b in zip(path_xy[:-1], path_xy[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / (cfg.res / 2))), 1)
        pts.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    pts = np.vstack(pts)
    ix, iy = _cells(pts, cfg)
    g[ix, iy] = True
    r = max(int(round(cfg.ego_half_width_m / cfg.res)), 1)
    return ndi.binary_dilation(g, structure=_disk(r))


def _recall(mask: np.ndarray, swath: np.ndarray) -> float:
    n = int(swath.sum())
    return float(mask[swath].mean()) if n else float("nan")


# ----------------------------------------------------------------------------
# Satu frame lengkap
# ----------------------------------------------------------------------------
VARIANTS = ("single", "accum", "accum_unk")
VARIANT_TITLES = {
    "single": "Satu scan (Road)",
    "accum": "Akumulasi (Road)",
    "accum_unk": "Akumulasi (Road + Unknown setinggi tanah)",
}


def compute_frame(seq: Sequence, poses: np.ndarray, valid: np.ndarray, i: int, cfg: Config) -> dict:
    hits, masks, npts, nfr = {}, {}, {}, {}
    hits["single"], npts["single"], nfr["single"] = accumulate(seq, poses, valid, i, cfg, False, single=True)
    hits["accum"], npts["accum"], nfr["accum"] = accumulate(seq, poses, valid, i, cfg, False)
    hits["accum_unk"], npts["accum_unk"], nfr["accum_unk"] = accumulate(seq, poses, valid, i, cfg, True)
    # Jika pose frame ini tidak valid, akumulasi jatuh ke 1 scan -> ambang tidak boleh > jumlah frame yang dipakai.
    thr = {"single": 1, "accum": min(cfg.min_frames, max(nfr["accum"], 1)),
           "accum_unk": min(cfg.min_frames, max(nfr["accum_unk"], 1))}
    for k in VARIANTS:
        masks[k] = make_mask(hits[k], thr[k], cfg)

    past_xy = ego_path_xy(poses, valid, seq.t, i, cfg.eval_past_s, 0.0)
    fut_xy = ego_path_xy(poses, valid, seq.t, i, 0.0, cfg.eval_future_s)
    sw_past, sw_fut = _swath(past_xy, cfg), _swath(fut_xy, cfg)

    row = {"idx": i, "t": float(seq.t[i]), "pose_ok": bool(valid[i]), "frames_used": nfr["accum"]}
    for k in VARIANTS:
        row[f"cells_{k}"] = int(masks[k].sum())
        row[f"pts_{k}"] = int(npts[k])
        row[f"recall_past_{k}"] = _recall(masks[k], sw_past)
        row[f"recall_future_{k}"] = _recall(masks[k], sw_fut)
    row["gain_accum"] = row["cells_accum"] / max(row["cells_single"], 1)
    row["gain_accum_unk"] = row["cells_accum_unk"] / max(row["cells_single"], 1)
    return {"row": row, "hits": hits, "masks": masks, "thr": thr, "past_xy": past_xy, "fut_xy": fut_xy}


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def _bev_image(hits: np.ndarray, mask: np.ndarray, thr: int) -> np.ndarray:
    img = np.full(mask.shape + (3,), 0.965, np.float32)
    img[mask] = (0.72, 0.82, 0.96)      # masker (setelah closing)
    img[hits >= thr] = (0.16, 0.36, 0.78)  # sel teramati langsung
    return img[:, ::-1, :]              # y kiri = kiri layar


def plot_frame(res: dict, cfg: Config, title_extra: str = ""):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    row = res["row"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.4))
    extent = (cfg.y_range[1], cfg.y_range[0], cfg.x_range[0], cfg.x_range[1])
    for ax, k in zip(axes, VARIANTS):
        ax.imshow(_bev_image(res["hits"][k], res["masks"][k], res["thr"][k]),
                  origin="lower", extent=extent, interpolation="nearest")
        for rr in (10, 20, 30):
            ax.add_patch(Circle((0, 0), rr, fill=False, ec="#999999", lw=0.6, ls=":"))
        if len(res["past_xy"]):
            ax.plot(res["past_xy"][:, 1], res["past_xy"][:, 0], color="#1b9e3e", lw=2, label="jalur lampau ego")
        if len(res["fut_xy"]):
            ax.plot(res["fut_xy"][:, 1], res["fut_xy"][:, 0], color="#e67e00", lw=2, ls="--", label="jalur ke depan ego")
        ax.add_patch(Polygon([(0, 1.2), (-0.6, -0.6), (0.6, -0.6)], closed=True, fc="#d62728", ec="k", zorder=5))
        ax.set_xlim(cfg.y_range[1], cfg.y_range[0])
        ax.set_ylim(cfg.x_range[0], cfg.x_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("y [m] (kiri = kiri layar)")
        ax.set_ylabel("x [m] (depan)")
        rp, rf = row[f"recall_past_{k}"], row[f"recall_future_{k}"]
        ax.set_title(f"{VARIANT_TITLES[k]}\nsel={row[f'cells_{k}']}  recall lampau={rp:.2f}  ke depan={rf:.2f}", fontsize=10)
    if len(res["past_xy"]) or len(res["fut_xy"]):
        axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Frame idx={row['idx']}  t={row['t']:.3f}  pose_ok={row['pose_ok']}  frame dipakai (akumulasi)={row['frames_used']}  {title_extra}\n"
                 "biru tua = sel teramati langsung, biru muda = hasil closing", fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Pemilihan frame uji, batch, verdict
# ----------------------------------------------------------------------------
def motion_profile(seq: Sequence, poses, valid, half_s: float = 0.5):
    """Kecepatan [m/s] dan laju yaw [deg/s] per frame (beda hingga simetris +-half_s)."""
    t = seq.t
    pos = poses[:, :2, 2]
    yaw = np.unwrap(np.arctan2(poses[:, 1, 0], poses[:, 0, 0]))
    j0 = np.searchsorted(t, t - half_s).clip(0, len(t) - 1)
    j1 = np.searchsorted(t, t + half_s).clip(0, len(t) - 1)
    dt = np.maximum(t[j1] - t[j0], 1e-6)
    speed = np.linalg.norm(pos[j1] - pos[j0], axis=1) / dt
    yrate = np.degrees(np.abs(yaw[j1] - yaw[j0])) / dt
    return speed, yrate


def pick_test_indices(seq: Sequence, poses, valid, cfg: Config, n_straight: int = 3, n_turn: int = 3,
                      n_slow: int = 2, turn_deg_s: float = 8.0, straight_deg_s: float = 2.0,
                      min_speed: float = 2.0, slow_range: Tuple[float, float] = (0.3, 1.5),
                      return_categories: bool = False):
    """
    Pilih frame uji yang mewakili: lurus, belokan, dan pelan (kandidat persimpangan / stop-and-go).
    Persimpangan tidak bisa dideteksi otomatis di Fase 1 -> cek PNG dan/atau isi TEST_INDICES manual.
    """
    t = seq.t
    speed, yrate = motion_profile(seq, poses, valid)
    base = valid & (t - t[0] >= cfg.window_past_s) & (t[-1] - t >= cfg.eval_future_s)
    cats = {
        "lurus": base & (speed >= min_speed) & (yrate <= straight_deg_s),
        "belok": base & (speed >= 1.0) & (yrate >= turn_deg_s),
        "pelan": base & (speed >= slow_range[0]) & (speed <= slow_range[1]),
    }
    want = {"lurus": n_straight, "belok": n_turn, "pelan": n_slow}
    picked, label = [], {}
    for name, m in cats.items():
        cand = np.where(m)[0]
        if len(cand) == 0:
            print(f"[peringatan] tidak ada kandidat frame '{name}'")
            continue
        sel = cand[np.unique(np.linspace(0, len(cand) - 1, min(want[name], len(cand))).round().astype(int))]
        for k in sel:
            if int(k) not in label:
                picked.append(int(k))
                label[int(k)] = name
    if not picked:
        raise RuntimeError("Tidak ada frame yang memenuhi syarat (cek odom / sinkronisasi waktu).")
    picked.sort()
    return (picked, label) if return_categories else picked


def run_batch(seq: Sequence, poses, valid, cfg: Config, indices, save_png=True, save_npz=True) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    png_dir, npz_dir = cfg.out_dir / "png", cfg.out_dir / "masks"
    png_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in tqdm(list(indices), desc="frame"):
        res = compute_frame(seq, poses, valid, int(i), cfg)
        rows.append(res["row"])
        if save_png:
            fig = plot_frame(res, cfg)
            fig.savefig(png_dir / f"{i:06d}.png", dpi=110)
            plt.close(fig)
        if save_npz:
            np.savez_compressed(
                npz_dir / f"{i:06d}.npz",
                idx=i, t=seq.t[i], lidar_file=seq.files[i].name,
                x_range=np.array(cfg.x_range), y_range=np.array(cfg.y_range), res=cfg.res,
                mask_single=res["masks"]["single"], mask_accum=res["masks"]["accum"],
                mask_accum_unk=res["masks"]["accum_unk"], hits_accum_unk=res["hits"]["accum_unk"].astype(np.uint8))
    df = pd.DataFrame(rows)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out_dir / "phase1_report.csv", index=False)
    return df


# ----------------------------------------------------------------------------
# Validasi odom dengan ICP (opsional, butuh open3d)
# ----------------------------------------------------------------------------
def icp_odom_check(seq: Sequence, poses, valid, cfg: Config, n_pairs: int = 30, gap_s: float = 0.5,
                   min_motion_m: float = 0.3, voxel: float = 0.3, max_corr: float = 1.0) -> pd.DataFrame:
    """
    Untuk pasangan frame (i, j~i+gap_s): prediksi odom dijadikan tebakan awal ICP point-to-plane;
    yang dilaporkan adalah seberapa jauh ICP menggeser tebakan itu. Koreksi kecil = odom konsisten.
    Catatan: ICP bisa 'longgar' di sepanjang lorong lurus tanpa fitur -> baca hasilnya sebagai
    indikator, bukan kebenaran mutlak.
    """
    import open3d as o3d
    t = seq.t
    car = cfg.class_ids
    dyn = [car["Vehicle"], car["Person"]]
    j_of = np.searchsorted(t, t + gap_s).clip(0, len(t) - 1)
    cand = [i for i in range(len(t))
            if valid[i] and valid[j_of[i]] and j_of[i] > i
            and np.linalg.norm((np.linalg.inv(poses[i]) @ poses[j_of[i]])[:2, 2]) >= min_motion_m]
    if not cand:
        raise RuntimeError("Tidak ada pasangan frame dengan gerak cukup untuk uji ICP.")
    pick = np.array(cand)[np.unique(np.linspace(0, len(cand) - 1, min(n_pairs, len(cand))).round().astype(int))]

    def pcd_of(k, normals):
        xyz, lab = seq.get_scan(k)
        xyz = xyz[~np.isin(lab, dyn)]
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        p = p.voxel_down_sample(voxel)
        if normals:
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        return p

    rows = []
    for i in pick:
        j = int(j_of[i])
        T_pred = np.linalg.inv(poses[i]) @ poses[j]
        init = se2_to_se3(T_pred)
        tgt, src = pcd_of(int(i), True), pcd_of(j, False)
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_corr, init, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        D = np.linalg.inv(init) @ r.transformation
        rows.append({
            "i": int(i), "j": j,
            "odom_move_m": float(np.linalg.norm(T_pred[:2, 2])),
            "icp_move_m": float(np.linalg.norm(r.transformation[:2, 3])),
            "shift_m": float(np.linalg.norm(D[:2, 3])),
            "shift_fwd_m": float(D[0, 3]), "shift_lat_m": float(D[1, 3]),
            "yaw_deg": float(np.degrees(np.arctan2(D[1, 0], D[0, 0]))),
            "fitness": float(r.fitness), "rmse": float(r.inlier_rmse),
        })
    return pd.DataFrame(rows)


def summarize_icp(df: pd.DataFrame) -> dict:
    ratio = df["icp_move_m"] / df["odom_move_m"].clip(lower=1e-6)
    return {
        "pairs": int(len(df)),
        "median_shift_m": float(df["shift_m"].median()),
        "p90_shift_m": float(df["shift_m"].quantile(0.9)),
        "median_abs_yaw_deg": float(df["yaw_deg"].abs().median()),
        "median_abs_fwd_m": float(df["shift_fwd_m"].abs().median()),
        "median_abs_lat_m": float(df["shift_lat_m"].abs().median()),
        "median_move_ratio_icp_over_odom": float(ratio.median()),
        "median_fitness": float(df["fitness"].median()),
    }


def verdict(df: pd.DataFrame, cfg: Config, icp: Optional[dict] = None) -> pd.DataFrame:
    """Checkpoint Fase 1 -> tabel LULUS / PERIKSA (target awal, boleh disesuaikan di Config)."""
    checks = [
        ("Recall jalur lampau (akumulasi+Unknown) >= target",
         float(df["recall_past_accum_unk"].median()), cfg.target_recall_past, ">="),
        ("Recall jalur ke depan (akumulasi+Unknown) >= target",
         float(df["recall_future_accum_unk"].median()), cfg.target_recall_future, ">="),
        ("Kenaikan sel jalan vs satu scan >= target (median)",
         float(df["gain_accum_unk"].median()), cfg.target_gain, ">="),
    ]
    if icp is not None:
        # Komponen lateral & yaw terobservasi baik oleh ICP; komponen maju/mundur lemah di jalan
        # lurus tanpa fitur (degenerate), jadi hanya dilaporkan sebagai informasi.
        checks += [
            ("ICP: median |koreksi lateral| <= target [m]", icp["median_abs_lat_m"], cfg.target_icp_median_shift_m, "<="),
            ("ICP: median |koreksi yaw| <= target [deg]", icp["median_abs_yaw_deg"], cfg.target_icp_median_yaw_deg, "<="),
        ]
    rows = []
    for name, val, tgt, op in checks:
        ok = (val >= tgt) if op == ">=" else (val <= tgt)
        rows.append({"cek": name, "nilai": round(val, 3), "target": tgt, "hasil": "LULUS" if ok else "PERIKSA"})
    if icp is not None:
        rows.append({"cek": "ICP (info): median |koreksi maju| [m] -- lemah di jalan lurus",
                     "nilai": round(icp["median_abs_fwd_m"], 3), "target": float("nan"), "hasil": "INFO"})
        rows.append({"cek": "ICP (info): rasio gerak ICP/odom (ideal ~1.0)",
                     "nilai": round(icp["median_move_ratio_icp_over_odom"], 3), "target": float("nan"), "hasil": "INFO"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Peta global (diagnostik drift odom)
# ----------------------------------------------------------------------------
def global_road_map(seq: Sequence, poses, valid, cfg: Config, step: int = 3, res: float = 0.5,
                    include_unknown: bool = False):
    pts = []
    for j in range(0, len(seq), step):
        if not valid[j]:
            continue
        p = seq.road_points(j, include_unknown)
        if len(p):
            pts.append(apply_se2(poses[j], p[:, :2]))
    if not pts:
        raise RuntimeError("Tidak ada titik Road untuk peta global.")
    P = np.vstack(pts)
    xe = np.arange(P[:, 0].min() - 1, P[:, 0].max() + 1 + res, res)
    ye = np.arange(P[:, 1].min() - 1, P[:, 1].max() + 1 + res, res)
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[xe, ye])
    traj = poses[valid][:, :2, 2]
    return H, xe, ye, traj


def plot_global_map(H, xe, ye, traj):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow((H.T > 0).astype(float), origin="lower", cmap="Blues", vmin=0, vmax=1.4,
              extent=(xe[0], xe[-1], ye[0], ye[-1]), interpolation="nearest")
    ax.plot(traj[:, 0], traj[:, 1], color="#e67e00", lw=1.2, label="lintasan ego (odom)")
    ax.set_aspect("equal")
    ax.set_xlabel("x dunia [m]")
    ax.set_ylabel("y dunia [m]")
    ax.set_title("Titik Road semua frame di kerangka odom -- jalan yang sama tidak boleh muncul dobel/bergeser")
    ax.legend()
    fig.tight_layout()
    return fig
