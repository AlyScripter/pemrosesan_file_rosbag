"""
camera_fusion_lib.py -- Fase 2: fusi segmentasi jalan kamera + LiDAR -> masker jalan BEV.

Bergantung pada road_lib.py (Fase 1, versi yang sudah kamu sesuaikan dengan struktur BEV_ROAD).

Alur
----
1. Segmentasi kamera (02a_segment_road.py, sekali jalan) -> seg_cam/<idx:06d>.png
   kode: 0 = abaikan (mobil, orang, langit, dst.), 1 = jalan, 2 = bukan-jalan (trotoar, bangunan, vegetasi, ...)
2. Proyeksi "maju": tiap sel grid BEV (dengan tinggi dari bidang tanah hasil LiDAR) diproyeksikan ke gambar
   memakai kalibrasi LiDAR-kamera, lalu kode segmentasi di piksel itu dibaca. Tidak ada lubang seperti
   proyeksi piksel->tanah, dan tidak bergantung pada depth stereo yang buruk di jarak jauh.
   Depth ZED hanya dipakai untuk "gerbang oklusi" (sel di belakang objek yang lebih dekat -> tidak diketahui).
3. Akumulasi bukti kamera + LiDAR + rintangan vertikal lintas frame lampau (jendela berbasis JARAK tempuh).
4. Fusi: jalan = (LiDAR jalan | kamera jalan) tanpa rintangan vertikal; kamera bisa memveto LiDAR yang lemah.

Konvensi sumbu frame LiDAR: x depan, y kiri, z atas. Grid BEV [ix, iy] sama dengan Fase 1.
Kamera: bingkai optik (x kanan, y bawah, z depan), K piksel.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence as Seq, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import road_lib as rl

SEG_IGNORE, SEG_ROAD, SEG_NONROAD = 0, 1, 2


# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------
@dataclass
class Fase2Config:
    root: Path
    sync_csv: Optional[Path] = None
    rgb_dir: Optional[Path] = None
    depth_dir: Optional[Path] = None
    calib_json: Optional[Path] = None
    seg_dir: Optional[Path] = None
    out_dir: Optional[Path] = None

    # kamera -> BEV
    cam_min_range: float = 1.0
    cam_max_range: float = 25.0        # di atas ini stereo ZED & sensitivitas pitch membuat proyeksi tak andal
    image_margin_px: int = 2
    use_depth_gate: bool = True        # sel di belakang objek yang lebih dekat -> tidak diketahui
    depth_gate_abs_tol: float = 1.0
    depth_gate_rel_tol: float = 0.20
    depth_valid_range: Tuple[float, float] = (0.3, 60.0)
    depth_gate_max_range: float = 15.0
    # ^ di atas jarak ini depth stereo ZED sendiri sudah terbukti tak andal (lihat scatter plot
    #   kalibrasi: melengkung/"hockey-stick" di atas ~15 m, dan toleransi linear di bawah tidak
    #   cukup menutupinya). Di luar jarak ini gerbang oklusi DIMATIKAN dan kode segmentasi kamera
    #   dipakai apa adanya -- lebih baik sesekali salah karena oklusi sungguhan tak tertangkap,
    #   drpd rutin membuang SEMUA sel jalan jauh yg sebenarnya benar (itulah yg terjadi sebelumnya:
    #   far_cam = 0 di seluruh 8 frame uji karena setiap sel jauh selalu dianggap "ada yg lebih
    #   dekat" akibat depth ZED yg meleset, bukan krn oklusi nyata).

    # bidang tanah dari titik Road LiDAR (per frame)
    plane_max_range: float = 15.0
    plane_min_pts: int = 60
    plane_max_slope: float = 0.45      # diukur di data nyata: median 0,307 (p5-p95 0,264-0,357) krn LiDAR terpasang miring ~17 deg;
                                       # bidang z=a*x+b*y+c dipakai LANGSUNG utk proyeksi kamera & tinggi rintangan, jadi bidang miring itu BENAR
    plane_inlier_tol: float = 0.08

    # jendela akumulasi (berbasis jarak tempuh; waktu hanya batas bawah/atas)
    window_dist_m: float = 12.0
    window_min_s: float = 2.0
    window_max_s: float = 10.0
    window_max_frames: int = 60
    obstacle_window_s: float = 2.0
    obstacle_max_frames: int = 20

    # fusi
    lidar_min_frames: int = 2
    lidar_strong_frames: int = 4
    cam_min_obs: int = 2
    cam_road_frac: float = 0.6
    cam_non_frac: float = 0.2
    obstacle_h_min: float = 0.4        # tinggi rintangan di atas tanah [m]; kanopi pohon (> h_max) tidak ikut memveto
    obstacle_h_max: float = 1.8
    obstacle_min_frames: int = 2
    policy: str = "veto"               # "veto" | "union"
    veto_max_range: float = 12.0       # kamera hanya boleh MEMBATALKAN LiDAR dlm jarak ini; di luar itu kamera cuma boleh menambah
    veto_min_obs: int = 4              # ambang observasi KHUSUS utk veto, > cam_min_obs -- veto = tindakan lebih berisiko
    veto_non_frac: float = 0.10        # dan mayoritas "bukan-jalan" yg lbh kuat drpd cam_non_frac (0.2) sebelum boleh membatalkan
    # ^ cam_min_obs=2 & cam_non_frac=0.2 cukup utk MENAMBAH bukti kamera (aman: worst-case cuma
    #   false-positive kecil yg tersaring cek lain). Tapi kalau dipakai jg utk MEMBATALKAN LiDAR yg
    #   sudah dikonfirmasi, cukup 2 frame kamera salah baca (mis. aspal di batas bayangan keras
    #   disangka bukan-jalan) utk menghapus jalan yg sebenarnya benar (lihat frame 744: recall jalur
    #   lampau turun 0.45->0.12 justru karena veto ini, bukan krn far-range gate). Veto perlu bukti
    #   yg lebih kuat & konsisten drpd sekadar menambah.

    # target checkpoint (awal; boleh disesuaikan)
    target_recall_past: float = 0.90
    target_recall_future: float = 0.90
    target_conflict_max: float = 0.05
    target_far_gain: float = 1.30
    target_calib_agree: float = 0.50
    target_cam_frames_ok: float = 0.90

    def __post_init__(self):
        self.root = Path(self.root)
        self.sync_csv = Path(self.sync_csv) if self.sync_csv else self.root / "sync_table.csv"
        self.rgb_dir = Path(self.rgb_dir) if self.rgb_dir else self.root / "rgb"
        self.depth_dir = Path(self.depth_dir) if self.depth_dir else self.root / "depth"
        self.calib_json = Path(self.calib_json) if self.calib_json else self.root / "calib" / "calib_config.json"
        self.seg_dir = Path(self.seg_dir) if self.seg_dir else self.root / "seg_cam"
        self.out_dir = Path(self.out_dir) if self.out_dir else self.root / "output_fase2"


# ----------------------------------------------------------------------------
# sync_table: frame LiDAR -> berkas RGB / depth lokal
# ----------------------------------------------------------------------------
def _basename(p) -> Optional[str]:
    if p is None or (isinstance(p, float) and np.isnan(p)) or (isinstance(p, str) and not p.strip()):
        return None
    return str(p).replace("\\", "/").rsplit("/", 1)[-1]


def load_sync(f2: Fase2Config) -> pd.DataFrame:
    """
    Baca sync_table.csv. Kolom rgb_data/depth_data berisi path WSL (/home/kali/...); yang dipakai hanya NAMA BERKAS,
    dicari di rgb_dir / depth_dir lokal. Index = idx frame LiDAR.
    """
    s = pd.read_csv(f2.sync_csv)
    need = {"idx", "rgb_data", "depth_data"}
    if not need.issubset(s.columns):
        raise ValueError(f"sync_table.csv harus punya kolom {sorted(need)}; ada: {list(s.columns)}")
    rgb_b = [_basename(p) for p in s["rgb_data"]]
    dep_b = [_basename(p) for p in s["depth_data"]]
    s["rgb_file"] = [f2.rgb_dir / b if b else None for b in rgb_b]
    s["depth_file"] = [f2.depth_dir / b if b else None for b in dep_b]
    s["rgb_ok"] = [bool(p is not None and p.exists()) for p in s["rgb_file"]]
    s["depth_ok"] = [bool(p is not None and p.exists()) for p in s["depth_file"]]
    return s.set_index("idx", drop=False)


def sync_report(sync: pd.DataFrame, f2: Fase2Config) -> dict:
    has = sync["rgb_data"].notna()
    seg_n = len(list(f2.seg_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].png"))) if f2.seg_dir.exists() else 0
    return {
        "frames_total": int(len(sync)),
        "frames_dengan_rgb": int(has.sum()),
        "rgb_ada_di_disk": int(sync["rgb_ok"].sum()),
        "depth_ada_di_disk": int(sync["depth_ok"].sum()),
        "idx_rgb_pertama": int(sync.index[has][0]) if has.any() else -1,
        "seg_cam_terisi": seg_n,
        "rgb_dt_maks_s": float(sync.loc[has, "rgb_dt"].abs().max()) if has.any() and "rgb_dt" in sync else float("nan"),
    }


# ----------------------------------------------------------------------------
# Peta kelas segmentasi -> kode 0/1/2
# ----------------------------------------------------------------------------
_ROAD_NAMES = {"road", "service-lane", "parking"}
_NONROAD_NAMES = {"sidewalk", "building", "wall", "fence", "guard rail", "guard-rail", "pole", "polegroup",
                  "traffic light", "traffic sign", "traffic-sign-frame", "vegetation", "terrain", "bridge",
                  "tunnel", "other-barrier", "barrier", "curb", "rail track", "ground"}


def name_to_code(name: str) -> int:
    n = str(name).lower().strip()
    last = n.split("--")[-1]                     # dukung label bergaya Mapillary: construction--flat--road
    if last in _ROAD_NAMES:
        return SEG_ROAD
    if last in _NONROAD_NAMES:
        return SEG_NONROAD
    return SEG_IGNORE


def build_lut(id2label: Dict) -> np.ndarray:
    """LUT id kelas model -> kode 0/1/2. Gagal keras bila model tak punya kelas 'road'."""
    ids = [int(k) for k in id2label.keys()]
    lut = np.zeros(max(ids) + 1, np.uint8)
    for k, v in id2label.items():
        lut[int(k)] = name_to_code(v)
    if not (lut == SEG_ROAD).any():
        raise ValueError(f"Model tidak punya kelas 'road': {sorted(set(map(str, id2label.values())))[:12]} ...")
    return lut


def codes_from_pred(pred_ids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[np.clip(pred_ids.astype(np.int64), 0, len(lut) - 1)]


# ----------------------------------------------------------------------------
# Kalibrasi LiDAR -> kamera
# ----------------------------------------------------------------------------
class CalibrationError(RuntimeError):
    pass


_ROS_TO_OPT = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.0]])


def _norm_key(k: str) -> str:
    return "".join(ch for ch in str(k).lower() if ch.isalnum())


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield f"{path}/{k}", k, v
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def _mat(v, rows, cols):
    try:
        a = np.array(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.size == rows * cols:
        return a.reshape(rows, cols)
    return None


def _key_tree(obj, depth=0, maxd=3) -> str:
    lines = []
    if isinstance(obj, dict) and depth <= maxd:
        for k, v in obj.items():
            if isinstance(v, dict):
                lines.append("  " * depth + f"{k}: {{...}}")
                lines.append(_key_tree(v, depth + 1, maxd))
            else:
                a = np.array(v, dtype=object) if isinstance(v, list) else None
                shp = f" list{a.shape}" if a is not None else f" {type(v).__name__}"
                lines.append("  " * depth + f"{k}:{shp}")
    return "\n".join(l for l in lines if l)


@dataclass
class Calibration:
    K: np.ndarray                       # 3x3 pada ukuran gambar kalibrasi
    size: Optional[Tuple[int, int]]     # (W, H) gambar kalibrasi, bila tercatat
    T: np.ndarray                       # 4x4 apa adanya dari berkas (arah/konvensi ditentukan resolve())
    T_key: str = ""
    notes: List[str] = field(default_factory=list)
    M: Optional[np.ndarray] = None      # LiDAR -> kamera optik, setelah resolve()
    M_name: str = ""

    @classmethod
    def from_json(cls, path: Path) -> "Calibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        notes: List[str] = []
        K = None
        size = None
        T = None
        T_key = ""
        # --- intrinsik
        for p, k, v in _walk(data):
            nk = _norm_key(k)
            if K is None and nk in ("k", "cameramatrix", "intrinsic", "intrinsics", "intrinsicmatrix",
                                    "cameraintrinsics", "kmatrix", "kmat"):
                m = _mat(v, 3, 3)
                if m is not None:
                    K, _ = m, notes.append(f"K dari {p}")
            if K is None and nk == "p":
                m = _mat(v, 3, 4)
                if m is not None:
                    K, _ = m[:, :3], notes.append(f"K dari P (3x4) di {p}")
            if isinstance(v, dict) and {"fx", "fy", "cx", "cy"} <= {_norm_key(x) for x in v.keys()} and K is None:
                d = {_norm_key(x): float(y) for x, y in v.items() if isinstance(y, (int, float))}
                K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]])
                notes.append(f"K dari fx,fy,cx,cy di {p}")
        if K is None and isinstance(data, dict) and {"fx", "fy", "cx", "cy"} <= {_norm_key(x) for x in data.keys()}:
            d = {_norm_key(x): float(y) for x, y in data.items() if isinstance(y, (int, float))}
            K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1.0]])
            notes.append("K dari fx,fy,cx,cy di level atas")
        # --- ukuran gambar
        w = h = None
        for p, k, v in _walk(data):
            nk = _norm_key(k)
            if nk in ("width", "imagewidth", "w") and isinstance(v, (int, float)):
                w = int(v)
            if nk in ("height", "imageheight", "h") and isinstance(v, (int, float)):
                h = int(v)
            if nk in ("imagesize", "size", "resolution") and isinstance(v, list) and len(v) == 2:
                w, h = int(v[0]), int(v[1])
        if isinstance(data, dict):
            for k, v in data.items():
                nk = _norm_key(k)
                if nk in ("width", "imagewidth") and isinstance(v, (int, float)):
                    w = int(v)
                if nk in ("height", "imageheight") and isinstance(v, (int, float)):
                    h = int(v)
        if w and h:
            size = (w, h)
            notes.append(f"ukuran gambar kalibrasi {w}x{h}")
        # --- ekstrinsik
        def is_ext_key(nk: str) -> bool:
            return (("lidar" in nk and ("cam" in nk or "2" in nk)) or "extrinsic" in nk or nk in ("t", "transform", "tcl", "tlc"))
        cands = []
        for p, k, v in list(_walk(data)) + ([(f"/{k}", k, v) for k, v in data.items()] if isinstance(data, dict) else []):
            nk = _norm_key(k)
            if not is_ext_key(nk):
                continue
            m = _mat(v, 4, 4)
            if m is None and _mat(v, 3, 4) is not None:
                m = np.vstack([_mat(v, 3, 4), [0, 0, 0, 1]])
            if m is None and isinstance(v, dict):
                vk = {_norm_key(x): y for x, y in v.items()}
                R = _mat(vk.get("r", vk.get("rotation", vk.get("rot"))), 3, 3) if any(x in vk for x in ("r", "rotation", "rot")) else None
                t = _mat(vk.get("t", vk.get("translation", vk.get("trans"))), 3, 1) if any(x in vk for x in ("t", "translation", "trans")) else None
                if R is not None and t is not None:
                    m = np.eye(4)
                    m[:3, :3], m[:3, 3] = R, t.ravel()
                else:
                    for kk in ("tlidartocam", "tcamlidar", "tlidartocamera", "matrix", "t"):
                        if kk in vk and _mat(vk[kk], 4, 4) is not None:
                            m = _mat(vk[kk], 4, 4)
                            break
            if m is not None:
                cands.append((p, m))
        if cands:
            pri = ("tlidartocam", "tcamlidar", "tlidartocamera", "lidartocam", "lidar2cam", "tlidarcam", "extrinsic")
            cands.sort(key=lambda pm: min([i for i, s in enumerate(pri) if s in _norm_key(pm[0])] or [99]))
            T_key, T = cands[0]
            notes.append(f"T dari {T_key}")
        if K is None or T is None:
            raise CalibrationError(
                f"Tidak menemukan {'K ' if K is None else ''}{'T ' if T is None else ''}di {path}.\n"
                f"Struktur berkas:\n{_key_tree(data)}\nKirim struktur ini supaya loader disesuaikan.")
        return cls(K=K, size=size, T=T, T_key=T_key, notes=notes)

    def K_for(self, W: int, H: int) -> np.ndarray:
        """K diskalakan bila gambar berukuran lain dari gambar kalibrasi."""
        if self.size is None or (W, H) == tuple(self.size):
            return self.K.copy()
        sx, sy = W / self.size[0], H / self.size[1]
        K = self.K.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def candidates(self) -> List[Tuple[str, np.ndarray]]:
        Ti = np.linalg.inv(self.T)
        return [("T apa adanya (LiDAR->kamera optik)", self.T),
                ("inv(T)", Ti),
                ("ROS->optik o T", _ROS_TO_OPT @ self.T),
                ("ROS->optik o inv(T)", _ROS_TO_OPT @ Ti)]


def project_points(K: np.ndarray, M: np.ndarray, pts: np.ndarray):
    Pc = pts[:, :3] @ M[:3, :3].T + M[:3, 3]
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    return u, v, z


def score_candidate(K: np.ndarray, M: np.ndarray, pts: np.ndarray, depth: Optional[np.ndarray],
                    rng_lo=3.0, rng_hi=30.0) -> dict:
    """Berapa banyak titik LiDAR yang jatuh di gambar dan (bila ada depth) cocok dengan depth kamera."""
    H, W = depth.shape if depth is not None else (0, 0)
    r = np.linalg.norm(pts[:, :2], axis=1)
    pts = pts[(r >= rng_lo) & (r <= rng_hi)]
    u, v, z = project_points(K, M, pts)
    if depth is None:
        return {"n_in": 0, "n_agree": 0, "agree": 0.0, "median_err": float("nan")}
    ok = (z > 0.5) & np.isfinite(u) & np.isfinite(v)
    iu, iv = np.rint(u[ok]).astype(int), np.rint(v[ok]).astype(int)
    inside = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    zc = z[ok][inside]
    d = depth[iv[inside], iu[inside]]
    valid = np.isfinite(d) & (d > 0.3) & (d < 60)
    if valid.sum() == 0:
        return {"n_in": int(inside.sum()), "n_agree": 0, "agree": 0.0, "median_err": float("nan")}
    err = np.abs(zc[valid] - d[valid])
    agree = err <= np.maximum(0.5, 0.15 * d[valid])
    return {"n_in": int(valid.sum()), "n_agree": int(agree.sum()), "agree": float(agree.mean()),
            "median_err": float(np.median(err))}


def resolve_calibration(calib: Calibration, samples: Seq[Tuple[np.ndarray, np.ndarray]],
                        W: int, H: int) -> pd.DataFrame:
    """
    samples: [(xyz_lidar (N,3+), depth (H,W))]. Menguji 4 kandidat arah/konvensi, memilih yang paling cocok
    dengan depth kamera, menyimpannya ke calib.M. Return tabel skor per kandidat.
    """
    K = calib.K_for(W, H)
    rows = []
    for name, M in calib.candidates():
        agg = {"n_in": 0, "n_agree": 0}
        errs = []
        for pts, dep in samples:
            s = score_candidate(K, M, pts, dep)
            agg["n_in"] += s["n_in"]
            agg["n_agree"] += s["n_agree"]
            if np.isfinite(s["median_err"]):
                errs.append(s["median_err"])
        rows.append({"kandidat": name, "titik_valid": agg["n_in"], "cocok": agg["n_agree"],
                     "tingkat_cocok": agg["n_agree"] / max(agg["n_in"], 1) if agg["n_in"] >= 200 else 0.0,
                     "median_err_m": float(np.median(errs)) if errs else float("nan")})
    df = pd.DataFrame(rows)
    best = int(df["tingkat_cocok"].to_numpy().argmax())
    calib.M = calib.candidates()[best][1]
    calib.M_name = calib.candidates()[best][0]
    df["dipilih"] = ["<--" if i == best else "" for i in range(len(df))]
    return df


# ----------------------------------------------------------------------------
# Bidang tanah dari titik Road LiDAR
# ----------------------------------------------------------------------------
def fit_ground_plane(pts: np.ndarray, f2: Fase2Config,
                     fallback: Optional[Tuple[float, float, float]] = None) -> Tuple[Tuple[float, float, float], bool]:
    """z = a*x + b*y + c. Return (plane, ok). ok=False bila jatuh ke cadangan."""
    if len(pts):
        P = pts[np.hypot(pts[:, 0], pts[:, 1]) <= f2.plane_max_range]
    else:
        P = pts
    if len(P) < f2.plane_min_pts:
        if len(P) >= 10:
            return (0.0, 0.0, float(np.median(P[:, 2]))), False
        return (fallback if fallback is not None else (0.0, 0.0, float("nan"))), False
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    w = np.ones(len(P), bool)
    coef = np.array([0.0, 0.0, float(np.median(P[:, 2]))])
    for _ in range(4):
        if w.sum() < f2.plane_min_pts // 2:
            break
        coef, *_ = np.linalg.lstsq(A[w], P[w, 2], rcond=None)
        w = np.abs(A @ coef - P[:, 2]) <= f2.plane_inlier_tol
    a, b, c = (float(x) for x in coef)
    if np.hypot(a, b) > f2.plane_max_slope:
        return (0.0, 0.0, float(np.median(P[:, 2]))), False
    return (a, b, c), True


# ----------------------------------------------------------------------------
# Konteks fusi
# ----------------------------------------------------------------------------
class FusionContext:
    def __init__(self, cfg: rl.Config, f2: Fase2Config, seq: rl.Sequence, poses: np.ndarray, valid: np.ndarray,
                 sync: pd.DataFrame, calib: Calibration, cache_cap: int = 160):
        if calib.M is None:
            raise CalibrationError("Panggil resolve_calibration() dulu.")
        self.cfg, self.f2, self.seq, self.poses, self.valid = cfg, f2, seq, poses, valid
        self.sync, self.calib = sync, calib
        self.nx, self.ny = cfg.nx, cfg.ny
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        self.gx = cfg.x_range[0] + (ix + 0.5) * cfg.res
        self.gy = cfg.y_range[0] + (iy + 0.5) * cfg.res
        self.centers = np.stack([self.gx.ravel(), self.gy.ravel()], 1)
        self._planes: Dict[int, Tuple[float, float, float]] = {}
        self._plane_ok: Dict[int, bool] = {}
        self._last_plane: Optional[Tuple[float, float, float]] = None
        self._last_any: Optional[Tuple[float, float, float]] = None
        self._plane_src: Dict[int, str] = {}
        self.neighbor_search = 40          # frame (+-) utk mencari tetangga bertitik Road bila frame ini kosong
        self._cam: "OrderedDict[int, Optional[np.ndarray]]" = OrderedDict()
        self._obs: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cap = cache_cap

    # ---- bidang tanah
    def plane(self, j: int) -> Tuple[float, float, float]:
        """Bidang tanah frame j. Urutan cadangan (sebelumnya hanya 'bidang valid terakhir', yang praktis tak pernah terisi
        karena plane_ok ~0% -> frame tanpa titik Road GAGAL dan jalan di viewer membeku):
          1. fit frame j sendiri;  2. bidang terakhir yang valid;  3. bidang terakhir yang dihitung (valid atau datar);
          4. frame TETANGGA terdekat (+-1..neighbor_search) yang punya titik Road (tinggi tanah antar-frame berdekatan ~sama).
        Sumber dicatat di self._plane_src[j]: fit | cadangan_valid | cadangan_terakhir | tetangga:<k>."""
        if j not in self._planes:
            pts = self.seq.road_points(j, True)
            pl, ok = fit_ground_plane(pts, self.f2, None)          # tanpa cadangan di sini: cascade eksplisit di bawah
            src = "fit" if np.isfinite(pl[2]) and ok else ("datar" if np.isfinite(pl[2]) else None)
            if src is None and self._last_plane is not None:
                pl, src = self._last_plane, "cadangan_valid"
            if src is None and self._last_any is not None:
                pl, src = self._last_any, "cadangan_terakhir"
            if src is None:
                for k in range(1, self.neighbor_search + 1):
                    for jj in (j - k, j + k):
                        if 0 <= jj < len(self.seq):
                            pj, okj = fit_ground_plane(self.seq.road_points(jj, True), self.f2, None)
                            if np.isfinite(pj[2]):
                                pl, ok, src = pj, False, f"tetangga:{jj - j:+d}"
                                break
                    if src is not None:
                        break
            if src is None or not np.isfinite(pl[2]):
                raise RuntimeError(f"frame {j}: tak ada titik Road untuk bidang tanah (cadangan & tetangga +-{self.neighbor_search} frame kosong)")
            self._planes[j], self._plane_ok[j], self._plane_src[j] = pl, ok, src
            if np.isfinite(pl[2]):
                self._last_any = pl
            if ok:
                self._last_plane = pl
        return self._planes[j]

    # ---- kamera
    def _load_codes(self, j: int) -> Optional[np.ndarray]:
        from PIL import Image
        p = self.f2.seg_dir / f"{j:06d}.png"
        if not p.exists():
            return None
        a = np.array(Image.open(p))
        return a[..., 0] if a.ndim == 3 else a

    def _load_depth(self, j: int) -> Optional[np.ndarray]:
        if not self.f2.use_depth_gate or j not in self.sync.index:
            return None
        p = self.sync.at[j, "depth_file"]
        if p is None or not Path(p).exists():
            return None
        return np.load(p).astype(np.float32)

    def project_state(self, codes: np.ndarray, depth: Optional[np.ndarray], plane) -> np.ndarray:
        """Grid (nx,ny) uint8: 0 tak diketahui, 1 jalan, 2 bukan-jalan (proyeksi maju sel BEV -> gambar)."""
        f2 = self.f2
        H, W = codes.shape
        K = self.calib.K_for(W, H)
        a, b, c = plane
        Z = a * self.gx + b * self.gy + c
        P = np.stack([self.gx.ravel(), self.gy.ravel(), Z.ravel()], 1)
        u, v, z = project_points(K, self.calib.M, P)
        good = (z > 0.3) & np.isfinite(u) & np.isfinite(v)
        iu = np.full(len(z), -10, np.int64)
        iv = np.full(len(z), -10, np.int64)
        iu[good] = np.rint(u[good]).astype(np.int64)
        iv[good] = np.rint(v[good]).astype(np.int64)
        m = f2.image_margin_px
        rng = np.hypot(self.gx, self.gy).ravel()
        inside = good & (iu >= m) & (iu < W - m) & (iv >= m) & (iv < H - m) & (rng >= f2.cam_min_range) & (rng <= f2.cam_max_range)
        st = np.zeros(len(z), np.uint8)
        idx = np.where(inside)[0]
        s = codes[iv[idx], iu[idx]].copy()
        if depth is not None:
            if depth.shape != codes.shape:                              # depth beda resolusi -> nearest
                ry = (iv[idx] * depth.shape[0] / H).astype(int).clip(0, depth.shape[0] - 1)
                rx = (iu[idx] * depth.shape[1] / W).astype(int).clip(0, depth.shape[1] - 1)
                d = depth[ry, rx]
            else:
                d = depth[iv[idx], iu[idx]]
            lo, hi = f2.depth_valid_range
            dv = np.isfinite(d) & (d >= lo) & (d <= hi)
            dv &= rng[idx] <= f2.depth_gate_max_range   # di luar ini depth ZED sendiri tak andal, gate dimatikan
            tol = np.maximum(f2.depth_gate_abs_tol, f2.depth_gate_rel_tol * z[idx])
            s = np.where(dv & (d < z[idx] - tol), SEG_IGNORE, s)     # ada sesuatu yang lebih dekat -> tak terlihat
        st[idx] = s
        return st.reshape(self.nx, self.ny)

    def diagnose_projection(self, j: int, min_range: float = 15.0) -> dict:
        """
        Untuk 1 frame kamera sumber j: pecah sel grid BEV dgn jarak >= min_range menjadi kategori
        nasib (mana yang gagal proyeksi vs digugurkan gerbang oklusi vs memang diberi label
        non-road/ignore oleh model). Dipakai untuk mencari kenapa cam_road jarang muncul jauh
        (rendahnya far_gain), TANPA perlu menjalankan ulang segmentasi.
        """
        f2 = self.f2
        codes = self._load_codes(j)
        if codes is None:
            return {"error": f"seg_cam/{j:06d}.png tidak ada"}
        depth = self._load_depth(j)
        a, b, c = self.plane(j)
        H, W = codes.shape
        K = self.calib.K_for(W, H)
        Z = a * self.gx + b * self.gy + c
        P = np.stack([self.gx.ravel(), self.gy.ravel(), Z.ravel()], 1)
        u, v, z = project_points(K, self.calib.M, P)
        rng = np.hypot(self.gx, self.gy).ravel()
        far = rng >= min_range
        n_far = int(far.sum())
        good = (z > 0.3) & np.isfinite(u) & np.isfinite(v)
        m = f2.image_margin_px
        iu = np.where(good, np.rint(np.nan_to_num(u)).astype(np.int64), -10)
        iv = np.where(good, np.rint(np.nan_to_num(v)).astype(np.int64), -10)
        in_img = good & (iu >= m) & (iu < W - m) & (iv >= m) & (iv < H - m)
        in_rng = rng <= f2.cam_max_range
        out = {"n_far_cells": n_far, "min_range": min_range, "cam_max_range": f2.cam_max_range}
        out["di_belakang_kamera_z<=0.3"] = int((far & ~good).sum())
        out["di_luar_gambar_atau_margin"] = int((far & good & ~in_img).sum())
        out["di_luar_cam_max_range"] = int((far & good & in_img & ~in_rng).sum())
        vis = far & good & in_img & in_rng
        out["terlihat_di_gambar"] = int(vis.sum())
        idx = np.where(vis)[0]
        if len(idx) == 0:
            return out
        s_code = codes[iv[idx], iu[idx]]
        out["seg_road"] = int((s_code == SEG_ROAD).sum())
        out["seg_nonroad"] = int((s_code == SEG_NONROAD).sum())
        out["seg_ignore"] = int((s_code == SEG_IGNORE).sum())
        if depth is not None:
            if depth.shape != codes.shape:
                ry = (iv[idx] * depth.shape[0] / H).astype(int).clip(0, depth.shape[0] - 1)
                rx = (iu[idx] * depth.shape[1] / W).astype(int).clip(0, depth.shape[1] - 1)
                d = depth[ry, rx]
            else:
                d = depth[iv[idx], iu[idx]]
            lo, hi = f2.depth_valid_range
            dv = np.isfinite(d) & (d >= lo) & (d <= hi)
            dv_gate = dv & (rng[idx] <= f2.depth_gate_max_range)   # sama dengan project_state
            tol = np.maximum(f2.depth_gate_abs_tol, f2.depth_gate_rel_tol * z[idx])
            gated = dv_gate & (d < z[idx] - tol)
            out["di_luar_depth_gate_max_range"] = int((rng[idx] > f2.depth_gate_max_range).sum())
            out["digugurkan_gerbang_oklusi"] = int((gated & (s_code == SEG_ROAD)).sum())
            out["road_lolos_gerbang"] = int((~gated & (s_code == SEG_ROAD)).sum())
        else:
            out["depth_tidak_ada"] = True
        return out

    def cam_state(self, j: int) -> Optional[np.ndarray]:
        if j in self._cam:
            self._cam.move_to_end(j)
            return self._cam[j]
        st = None
        codes = self._load_codes(j)
        if codes is not None:
            st = self.project_state(codes, self._load_depth(j), self.plane(j))
        self._cam[j] = st
        if len(self._cam) > self._cap:
            self._cam.popitem(last=False)
        return st

    # ---- rintangan vertikal (Building/Vegetation setinggi kendaraan)
    def obstacle_xy(self, j: int) -> np.ndarray:
        if j in self._obs:
            return self._obs[j]
        xyz, lab = self.seq.get_scan(j)
        a, b, c = self.plane(j)
        hgt = xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c)
        ids = [self.cfg.class_ids["Building"], self.cfg.class_ids["Vegetation"]]
        m = np.isin(lab, ids) & (hgt >= self.f2.obstacle_h_min) & (hgt <= self.f2.obstacle_h_max)
        out = xyz[m, :2].astype(np.float32)
        self._obs[j] = out
        if len(self._obs) > self._cap:
            self._obs.popitem(last=False)
        return out

    # ---- jendela
    def window(self, i: int) -> np.ndarray:
        """Frame lampau sampai jarak tempuh >= window_dist_m (dan >= window_min_s), dibatasi window_max_s."""
        f2, t = self.f2, self.seq.t
        if not self.valid[i]:
            return np.array([i])
        j0 = int(np.searchsorted(t, t[i] - f2.window_max_s, side="left"))
        idx = np.arange(j0, i + 1)
        idx = idx[self.valid[idx]]
        rev = idx[::-1]
        pos = self.poses[rev, :2, 2]
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
        need = (d >= f2.window_dist_m) & ((t[i] - t[rev]) >= f2.window_min_s)
        stop = int(np.argmax(need)) if need.any() else len(rev) - 1
        keep = rev[: stop + 1][::-1]
        if len(keep) > f2.window_max_frames:
            sel = np.unique(np.linspace(0, len(keep) - 1, f2.window_max_frames).round().astype(int))
            keep = keep[sel]
            if keep[-1] != i:
                keep = np.append(keep, i)
        return keep

    def short_window(self, i: int) -> np.ndarray:
        f2, t = self.f2, self.seq.t
        if not self.valid[i]:
            return np.array([i])
        j0 = int(np.searchsorted(t, t[i] - f2.obstacle_window_s, side="left"))
        idx = np.arange(j0, i + 1)
        idx = idx[self.valid[idx]]
        if len(idx) > f2.obstacle_max_frames:
            idx = idx[np.unique(np.linspace(0, len(idx) - 1, f2.obstacle_max_frames).round().astype(int))]
        return idx

    # ---- akumulasi
    def lidar_hits(self, i: int, J: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        hits = np.zeros((self.nx, self.ny), np.uint16)
        Ti_inv = np.linalg.inv(self.poses[i])
        for j in J:
            pts = self.seq.road_points(int(j), True)
            if len(pts) == 0:
                continue
            p = rl.apply_se2(Ti_inv @ self.poses[j], pts[:, :2])
            ix, iy = rl._cells(p, cfg)
            if len(ix):
                hits.flat[np.unique(ix * self.ny + iy)] += 1
        return hits

    def obstacle_hits(self, i: int, J: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        hits = np.zeros((self.nx, self.ny), np.uint16)
        Ti_inv = np.linalg.inv(self.poses[i])
        for j in J:
            xy = self.obstacle_xy(int(j))
            if len(xy) == 0:
                continue
            p = rl.apply_se2(Ti_inv @ self.poses[j], xy)
            ix, iy = rl._cells(p, cfg)
            if len(ix):
                hits.flat[np.unique(ix * self.ny + iy)] += 1
        return hits

    def cam_hits(self, i: int, J: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Pemetaan MUNDUR: tiap sel frame-i dicari padanannya di grid kamera frame-j (nearest) -> tanpa lubang."""
        cfg = self.cfg
        road = np.zeros((self.nx, self.ny), np.uint16)
        non = np.zeros((self.nx, self.ny), np.uint16)
        n_used = 0
        Ti = self.poses[i]
        for j in J:
            st = self.cam_state(int(j))
            if st is None:
                continue
            n_used += 1
            Tji = np.linalg.inv(self.poses[j]) @ Ti
            Pj = self.centers @ Tji[:2, :2].T + Tji[:2, 2]
            ix = np.floor((Pj[:, 0] - cfg.x_range[0]) / cfg.res).astype(np.int64)
            iy = np.floor((Pj[:, 1] - cfg.y_range[0]) / cfg.res).astype(np.int64)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            s = np.zeros(len(Pj), np.uint8)
            s[ok] = st[ix[ok], iy[ok]]
            s = s.reshape(self.nx, self.ny)
            road += (s == SEG_ROAD)
            non += (s == SEG_NONROAD)
        return road, non, n_used

    # ---- fusi
    def fuse(self, L, Cr, Cn, O) -> dict:
        f2, cfg = self.f2, self.cfg
        lidar_road = L >= f2.lidar_min_frames
        tot = Cr + Cn
        frac = Cr / np.maximum(tot, 1)
        seen = tot >= f2.cam_min_obs
        cam_road = seen & (frac >= f2.cam_road_frac)
        cam_non = seen & (frac <= f2.cam_non_frac)
        obstacle = O >= f2.obstacle_min_frames
        raw = lidar_road | cam_road
        if f2.policy == "veto":
            near = np.hypot(self.gx, self.gy) <= f2.veto_max_range
            veto_seen = tot >= f2.veto_min_obs               # ambang lebih tinggi drpd cam_min_obs
            veto_non = veto_seen & (frac <= f2.veto_non_frac)  # mayoritas non-jalan lbh kuat drpd cam_non_frac
            vetoable = veto_non & (L < f2.lidar_strong_frames) & near
            raw = raw & ~vetoable
        raw = raw & ~obstacle
        masks = {
            "lidar": rl.make_mask(L, f2.lidar_min_frames, cfg),
            "cam": rl.make_mask(cam_road.astype(np.uint16), 1, cfg),
            "fused": rl.make_mask(raw.astype(np.uint16), 1, cfg) & ~obstacle,
        }
        return {"masks": masks, "raw": raw, "cam_road": cam_road, "cam_non": cam_non, "cam_seen": seen,
                "obstacle": obstacle}

    # ---- satu frame lengkap
    def compute(self, i: int) -> dict:
        cfg, f2 = self.cfg, self.f2
        J = self.window(i)
        L = self.lidar_hits(i, J)
        Cr, Cn, n_cam = self.cam_hits(i, J)
        Jo = self.short_window(i)
        O = self.obstacle_hits(i, Jo)
        fu = self.fuse(L, Cr, Cn, O)
        masks = fu["masks"]

        past_xy = rl.ego_path_xy(self.poses, self.valid, self.seq.t, i, cfg.eval_past_s, 0.0)
        fut_xy = rl.ego_path_xy(self.poses, self.valid, self.seq.t, i, 0.0, cfg.eval_future_s)
        sw_p, sw_f = rl._swath(past_xy, cfg), rl._swath(fut_xy, cfg)
        far = (self.gx >= 15.0)

        row = {"idx": i, "t": float(self.seq.t[i]), "pose_ok": bool(self.valid[i]),
               "frames_lidar": int(len(J)), "frames_cam": int(n_cam),
               "plane_ok": bool(self._plane_ok.get(i, False)),
               "ground_z": float(self.plane(i)[2]), "plane_src": self._plane_src.get(i, "")}
        for k, m in masks.items():
            row[f"cells_{k}"] = int(m.sum())
            row[f"far_{k}"] = int((m & far).sum())
            row[f"recall_past_{k}"] = rl._recall(m, sw_p)
            row[f"recall_future_{k}"] = rl._recall(m, sw_f)
            row[f"conflict_{k}"] = float((m & fu["obstacle"]).sum() / max(m.sum(), 1))
        # Konsistensi LiDAR-jalan vs kamera pada sel yang KEDUANYA punya bukti (bukan IoU: IoU bercampur dgn
        # cakupan, padahal LiDAR VLP-16 memang lebih pendek jangkauannya daripada kamera)
        l_seen = masks["lidar"] & fu["cam_seen"]
        n_ls = int(l_seen.sum())
        row["lidar_cam_agree"] = float((l_seen & fu["cam_road"]).sum() / n_ls) if n_ls else float("nan")
        row["lidar_cam_conflict"] = float((l_seen & fu["cam_non"]).sum() / n_ls) if n_ls else float("nan")
        row["cam_region_cells"] = int(n_ls)
        row["far_gain"] = min(row["far_fused"] / max(row["far_lidar"], 1), 10.0)   # dibatasi 10 (LiDAR sering 0 di > 15 m)
        row["far_added_abs"] = int(row["far_fused"] - row["far_lidar"])
        # ^ far_gain jenuh/mentok 10.00 begitu far_lidar kecil (mis. 0-131), jadi "LULUS" di situ tak
        #   banyak bermakna secara kuantitatif -- pakai far_added_abs sbg pelengkap: jumlah sel jalan
        #   jauh BARU (bersih) yg disumbang kamera, tak peduli seberapa kecil basis LiDAR-nya.
        return {"row": row, "L": L, "Cr": Cr, "Cn": Cn, "O": O, "fu": fu, "past_xy": past_xy, "fut_xy": fut_xy}


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def _bev_from(hits, mask, thr):
    return rl._bev_image(hits.astype(np.uint16), mask, thr)


def plot_fusion_frame(ctx: FusionContext, res: dict, show_rgb: bool = True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from PIL import Image

    cfg = ctx.cfg
    row, fu = res["row"], res["fu"]
    i = row["idx"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.8))
    ax = axes[0]
    ax.axis("off")
    try:
        rgb_p = ctx.sync.at[i, "rgb_file"]
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        codes = ctx._load_codes(i)
        ov = rgb.copy()
        if codes is not None and codes.shape == rgb.shape[:2]:
            col = np.array([[0, 0, 0], [40, 200, 60], [220, 60, 60]], np.uint8)
            m = codes > 0
            ov[m] = (0.55 * rgb[m] + 0.45 * col[codes[m]]).astype(np.uint8)
        ax.imshow(ov)
        ax.set_title("RGB + segmentasi (hijau=jalan, merah=bukan jalan)", fontsize=10)
    except Exception as e:  # noqa: BLE001
        ax.text(0.5, 0.5, f"RGB tidak tersedia\n{e}", ha="center", va="center")

    extent = (cfg.y_range[1], cfg.y_range[0], cfg.x_range[0], cfg.x_range[1])
    panels = [
        ("LiDAR saja (jendela jarak)", res["L"], fu["masks"]["lidar"], ctx.f2.lidar_min_frames, "lidar"),
        ("Kamera saja (proyeksi maju)", fu["cam_road"].astype(np.uint16), fu["masks"]["cam"], 1, "cam"),
        (f"FUSI (kebijakan: {ctx.f2.policy})", fu["raw"].astype(np.uint16), fu["masks"]["fused"], 1, "fused"),
    ]
    for ax, (title, hits, mask, thr, key) in zip(axes[1:], panels):
        ax.imshow(_bev_from(hits, mask, thr), origin="lower", extent=extent, interpolation="nearest")
        obs = fu["obstacle"]
        if obs.any():
            oy, ox = np.where(obs.T)
            ax.scatter(cfg.y_range[1] - (oy + 0.5) * cfg.res, cfg.x_range[0] + (ox + 0.5) * cfg.res,
                       s=1.5, c="#ff8c00", marker="s", linewidths=0, label="rintangan vertikal")
        for rr in (10, 20, 30):
            ax.add_patch(Circle((0, 0), rr, fill=False, ec="#999999", lw=0.6, ls=":"))
        if len(res["past_xy"]):
            ax.plot(res["past_xy"][:, 1], res["past_xy"][:, 0], color="#1b9e3e", lw=2)
        if len(res["fut_xy"]):
            ax.plot(res["fut_xy"][:, 1], res["fut_xy"][:, 0], color="#e67e00", lw=2, ls="--")
        ax.add_patch(Polygon([(0, 1.2), (-0.6, -0.6), (0.6, -0.6)], closed=True, fc="#d62728", ec="k", zorder=5))
        ax.set_xlim(cfg.y_range[1], cfg.y_range[0])
        ax.set_ylim(cfg.x_range[0], cfg.x_range[1])
        ax.set_aspect("equal")
        ax.set_xlabel("y [m]")
        ax.set_title(f"{title}\nsel={row[f'cells_{key}']}  recall lampau={row[f'recall_past_{key}']:.2f}  "
                     f"ke depan={row[f'recall_future_{key}']:.2f}  konflik={row[f'conflict_{key}']:.3f}", fontsize=10)
    axes[1].set_ylabel("x [m] (depan)")
    fig.suptitle(f"Frame idx={i}  frame LiDAR={row['frames_lidar']}  frame kamera={row['frames_cam']}  "
                 f"LiDAR dibenarkan kamera={row['lidar_cam_agree']:.2f}  dibantah={row['lidar_cam_conflict']:.2f}  far_gain={row['far_gain']:.2f}\n"
                 "biru tua = teramati langsung, biru muda = hasil closing, oranye = rintangan vertikal", fontsize=11)
    fig.tight_layout()
    return fig


def plot_calib_check(ctx: FusionContext, j: int):
    """Titik LiDAR diproyeksikan ke gambar (warna = jarak dari kalibrasi) + perbandingan dengan depth kamera."""
    import matplotlib.pyplot as plt
    from PIL import Image
    rgb = np.array(Image.open(ctx.sync.at[j, "rgb_file"]).convert("RGB"))
    depth = np.load(ctx.sync.at[j, "depth_file"]).astype(np.float32)
    xyz, _ = ctx.seq.get_scan(j)
    H, W = rgb.shape[:2]
    K = ctx.calib.K_for(W, H)
    u, v, z = project_points(K, ctx.calib.M, xyz)
    ok = (z > 0.5) & (z < 40) & np.isfinite(u) & np.isfinite(v)
    iu0, iv0 = np.zeros(len(z), np.int64), np.zeros(len(z), np.int64)
    iu0[ok], iv0[ok] = np.rint(u[ok]).astype(np.int64), np.rint(v[ok]).astype(np.int64)
    ok &= (iu0 >= 0) & (iu0 < W) & (iv0 >= 0) & (iv0 < H)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6))
    ax[0].imshow(rgb)
    sc = ax[0].scatter(u[ok], v[ok], c=z[ok], s=2, cmap="turbo", vmin=0, vmax=30)
    fig.colorbar(sc, ax=ax[0], fraction=0.03, label="jarak z kamera [m]")
    ax[0].set_title(f"LiDAR terproyeksi (kalibrasi: {ctx.calib.M_name})")
    ax[0].axis("off")
    d = depth[iv0[ok], iu0[ok]]
    g = np.isfinite(d)
    ax[1].scatter(z[ok][g], d[g], s=2, alpha=0.4)
    lim = [0, 40]
    ax[1].plot(lim, lim, "r--", lw=1, label="ideal")
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel("z dari LiDAR+kalibrasi [m]")
    ax[1].set_ylabel("depth kamera [m]")
    ax[1].set_title("Kesesuaian jarak LiDAR vs depth ZED (di dekat garis merah = baik)")
    ax[1].legend()
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Batch, verdict
# ----------------------------------------------------------------------------
def run_batch(ctx: FusionContext, indices, save_png=True, save_npz=True) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # noqa: E731
    out = ctx.f2.out_dir
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in tqdm(list(indices), desc="frame"):
        res = ctx.compute(int(i))
        rows.append(res["row"])
        if save_png:
            fig = plot_fusion_frame(ctx, res)
            fig.savefig(out / "png" / f"{i:06d}.png", dpi=100)
            plt.close(fig)
        if save_npz:
            save_masks(ctx, res, out / "masks" / f"{i:06d}.npz")
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase2_report.csv", index=False)
    return df


def save_masks(ctx: FusionContext, res: dict, path: Path):
    cfg, fu = ctx.cfg, res["fu"]
    np.savez_compressed(
        path, idx=res["row"]["idx"], t=res["row"]["t"],
        x_range=np.array(cfg.x_range), y_range=np.array(cfg.y_range), res=cfg.res,
        mask_fused=fu["masks"]["fused"], mask_lidar=fu["masks"]["lidar"], mask_cam=fu["masks"]["cam"],
        lidar_hits=np.minimum(res["L"], 255).astype(np.uint8),
        cam_road_hits=np.minimum(res["Cr"], 255).astype(np.uint8),
        cam_non_hits=np.minimum(res["Cn"], 255).astype(np.uint8),
        obstacle_hits=np.minimum(res["O"], 255).astype(np.uint8),
        ground_plane=np.array(ctx.plane(res["row"]["idx"])))


def verdict(df: pd.DataFrame, f2: Fase2Config, calib_agree: Optional[float] = None,
            cam_frames_ok: Optional[float] = None) -> pd.DataFrame:
    med = lambda c: float(df[c].median())  # noqa: E731
    rows = []

    def add(name, val, tgt, op):
        ok = (val >= tgt) if op == ">=" else (val <= tgt)
        rows.append({"cek": name, "nilai": round(val, 3), "target": tgt, "hasil": "LULUS" if ok else "PERIKSA"})

    if calib_agree is not None:
        add("Kalibrasi: kecocokan LiDAR vs depth kamera", calib_agree, f2.target_calib_agree, ">=")
    if cam_frames_ok is not None:
        add("Cakupan segmentasi kamera pada frame uji", cam_frames_ok, f2.target_cam_frames_ok, ">=")
    add("FUSI: recall jalur lampau (median)", med("recall_past_fused"), f2.target_recall_past, ">=")
    add("FUSI: recall jalur ke depan (median)", med("recall_future_fused"), f2.target_recall_future, ">=")
    add("FUSI: konflik dgn rintangan vertikal (median, makin kecil makin baik)", med("conflict_fused"),
        f2.target_conflict_max, "<=")
    add("FUSI: sel jalan >= 15 m vs LiDAR saja (median rasio)", med("far_gain"), f2.target_far_gain, ">=")
    rows.append({"cek": "info: sel jalan jauh (>=15m) BARU bersih dari kamera (median, absolut)",
                 "nilai": round(med("far_added_abs"), 1), "target": float("nan"), "hasil": "INFO"})
    rows.append({"cek": "info: recall lampau LiDAR saja (jendela jarak)", "nilai": round(med("recall_past_lidar"), 3),
                 "target": float("nan"), "hasil": "INFO"})
    rows.append({"cek": "info: recall ke depan LiDAR saja", "nilai": round(med("recall_future_lidar"), 3),
                 "target": float("nan"), "hasil": "INFO"})
    rows.append({"cek": "info: recall lampau kamera saja", "nilai": round(med("recall_past_cam"), 3),
                 "target": float("nan"), "hasil": "INFO"})
    rows.append({"cek": "info: konflik LiDAR saja / kamera saja", "nilai": round(med("conflict_lidar"), 3),
                 "target": round(med("conflict_cam"), 3), "hasil": "INFO"})
    rows.append({"cek": "info: LiDAR-jalan yang DIBENARKAN kamera (median)", "nilai": round(med("lidar_cam_agree"), 3),
                 "target": float("nan"), "hasil": "INFO"})
    rows.append({"cek": "info: LiDAR-jalan yang DIBANTAH kamera (median) -- tinggi = plaza/trotoar berlabel Road", "nilai":
                 round(med("lidar_cam_conflict"), 3), "target": float("nan"), "hasil": "INFO"})
    return pd.DataFrame(rows)
