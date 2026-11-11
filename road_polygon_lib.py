#!/usr/bin/env python3
"""
road_polygon_lib.py -- BEV ROAD Fase 1: masker jalan final -> poligon jalan + jarak ego ke tepi.

Pengganti fase3_lib.py (lama). Diganti nama supaya tidak bentrok dengan "Fase 3 deteksi objek"
di pipeline utama (folder fase3_deteksi). Semua fungsi lama yang sudah divalidasi (smooth_mask,
ring_sample_arms, classify_road_type) dipertahankan; yang berubah ada di daftar di bawah.

INPUT  (per frame): <masks_dir>/<idx:06d>.npz  -- keluaran Fase 2 (camera_fusion_lib.save_masks)
        wajib: mask_fused (nx,ny bool), x_range (2,), y_range (2,), res (skalar)
        grid ego-relative, frame LiDAR: x = depan (+), y = kiri (+), ego (LiDAR) di (0,0).
        array diindeks [ix, iy] (baris = x, kolom = y).
OUTPUT (per frame): <out_dir>/<idx:06d>.json
        {
          "idx": int, "t": float,
          "road": {
            "polygons": [[[x,y],...], ...],            # cincin LUAR, CCW (x depan, y kiri), meter
            "holes":    [{"parent": i, "ring": [[x,y],...]}, ...],   # cincin LUBANG (CW), parent = indeks di polygons
            "res_equiv_m": float
          },
          "ego_edge_m": {
            "front": m|null, "back": m|null, "left": m|null, "right": m|null,
            "censored": ["front", ...],               # sinar yang sampai batas grid tanpa menemukan tepi (nilai = batas bawah)
            "ego_on_road": bool,
            "lane_center_offset_m": m|null            # (left-right)/2; > 0 = pusat jalan ada di KIRI ego
          },
          "road_type": null                           # null kecuali PolygonConfig.compute_road_type=True
        }
        Konsumen lama (fase4_lib.load_road_polygons) hanya membaca road.polygons -> tetap kompatibel.

PERUBAHAN dibanding fase3_lib.py lama
  1. smooth_mask: closing/opening dijalankan pada array yang di-pad (mode 'edge'). Versi lama mengikis
     r sel (0.6 m) di tepi grid -> jalan terpotong di jangkauan terjauh (terbukti dgn uji sintetis).
  2. Kontur subpiksel (marching squares via contourpy pada mask yang di-blur ringan, level 0.5),
     bukan kontur piksel cv2 yang menyusut setengah sel (0.1 m). Cadangan: cv2 (piksel) bila contourpy tak ada.
  3. Lubang (pulau jalan / blok) disimpan di road.holes; versi lama hanya kontur luar.
  4. BARU ego_edge_distances() -> "ego_edge_m".
  5. road_type default null (compute_road_type=False); fungsi lengan/tipe jalan tetap ada utk Fase 5 opsional.
  6. BARU temporal_stability(): IoU poligon antar-frame setelah kompensasi gerak ego (SE(2) dari odom).
  7. BARU verdict() + run_batch() mencatat ukuran payload vs baseline grid mentah.
  9. (rev 2) temporal_stability: hasil kosong kini DataFrame BERKOLOM (sebelumnya DataFrame tanpa kolom -> KeyError('lag') di
     notebook) + peringatan jelas; verdict() menandai IoU "BELUM DIUKUR" = PERIKSA (sebelumnya baris IoU hilang diam-diam
     sehingga "SEMUA LULUS" bisa keluar tanpa IoU pernah dihitung).
 10. (rev 2) verdict: cek ego_on_road dihitung di luar frame pemanasan (t < t_track_start + warmup_s; akumulasi belum penuh
     setelah ZED mulai tracking) + baris info untuk semua frame dan daftar idx yang gagal.
  8. plot_frame_diagnostics DIPERBAIKI: versi lama menggambar masker terbalik kiri-kanan
     (jalan di y=+7.5 tergambar di y=-7.5) sementara poligon/ego benar -> membingungkan saat cek "ego di tengah jalan?".
 11. (rev 3) Fase R1: PolygonConfig.r1 (road_evidence_lib.R1Config). None = perilaku rev 2 PERSIS (diuji regresi). Bila diisi,
     process_frame memakai bukti tri-state + perbaikan lubang + penghalusan tepi, JSON +road.vq +road.inf; run_batch menerima
     poses/valid (prior temporal kausal opsional) dan mencatat kolom inferred_frac.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:  # pragma: no cover
    _HAVE_CV2 = False

try:
    import contourpy
    _HAVE_CONTOURPY = True
except ImportError:  # pragma: no cover
    _HAVE_CONTOURPY = False


# ----------------------------------------------------------------------------
# konfigurasi
# ----------------------------------------------------------------------------
@dataclass
class PolygonConfig:
    root: Path
    masks_dir: Optional[Path] = None     # default: <root>/road_mask/masks bila ada, else <root>/output_fase2/masks
    out_dir: Optional[Path] = None       # default <root>/road_polygon
    mask_key: str = "mask_fused"
    r1: Optional[Any] = None             # (rev 3) road_evidence_lib.R1Config() -> mode R1; None = perilaku rev 2 persis

    # penghalusan morfologi (meter -> piksel via res)
    close_radius_m: float = 0.6
    open_radius_m: float = 0.3
    max_fill_hole_area_m2: float = 8.0   # lubang tertutup <= ini diisi (noise segmentasi)
    keep_only_ego_component: bool = True

    # poligon
    subpixel: bool = True
    subpixel_sigma_px: float = 0.8       # blur ringan sebelum marching squares (0 = tanpa blur)
    simplify_tol_m: float = 0.25         # Douglas-Peucker (meter)
    min_polygon_area_m2: float = 1.0
    keep_holes: bool = True
    min_hole_area_m2: float = 8.0        # lubang tersisa (> max_fill_hole_area_m2) = pulau nyata
    coord_decimals: int = 2

    # jarak ego ke tepi
    edge_step_m: float = 0.1
    edge_gap_tol_m: float = 0.3          # celah non-jalan <= ini sepanjang sinar tidak dianggap tepi
    ego_snap_m: float = 0.6              # bila sel ego bukan jalan, cari jalan dlm radius ini

    # tipe jalan (opsional, Fase 5)
    compute_road_type: bool = False
    ring_radii_m: Tuple[float, ...] = (6.0, 10.0)
    ring_angle_step_deg: float = 2.0
    min_arm_width_deg: float = 14.0
    max_continuous_gap_m: float = 2.0
    radial_probe_step_m: float = 0.2
    max_gap_close_deg: float = 8.0
    ring_vote_frac: float = 0.6
    straight_angle_tol_deg: float = 30.0

    # checkpoint (USULAN awal -- sepakati dengan pembimbing)
    stability_eval_range_m: float = 20.0
    target_iou: float = 0.9
    target_ego_on_road: float = 0.95
    warmup_s: float = 10.0               # = Fase2Config.window_max_s: setelah tracking mulai, jendela akumulasi belum penuh

    def __post_init__(self):
        self.root = Path(self.root)
        if self.masks_dir:
            self.masks_dir = Path(self.masks_dir)
        else:   # prefer masker baru (build_road_masks.py), lalu masker lama
            cands = [self.root / "road_mask" / "masks", self.root / "output_fase2" / "masks"]
            self.masks_dir = next((c for c in cands if c.exists() and any(c.glob("*.npz"))), cands[0])
        self.out_dir = Path(self.out_dir) if self.out_dir else self.root / "road_polygon"


# ----------------------------------------------------------------------------
# I/O masker
# ----------------------------------------------------------------------------
def list_mask_frames(cfg: PolygonConfig) -> List[int]:
    return sorted(int(p.stem) for p in cfg.masks_dir.glob("*.npz") if p.stem.isdigit())


def load_mask_frame(cfg: PolygonConfig, idx: int) -> dict:
    p = cfg.masks_dir / f"{idx:06d}.npz"
    d = np.load(p)
    if cfg.mask_key not in d.files:
        raise KeyError(f"{p.name}: tidak ada '{cfg.mask_key}'. Isi berkas: {d.files}")
    x_range, y_range, res = d["x_range"].astype(float), d["y_range"].astype(float), float(d["res"])
    mask = d[cfg.mask_key].astype(bool)
    exp = (int(round((x_range[1] - x_range[0]) / res)), int(round((y_range[1] - y_range[0]) / res)))
    if mask.shape != exp:
        raise ValueError(f"{p.name}: bentuk mask {mask.shape} != grid dari x_range/y_range/res {exp}")
    return {"idx": int(d["idx"]) if "idx" in d.files else idx,
            "t": float(d["t"]) if "t" in d.files else None,
            "mask": mask, "x_range": x_range, "y_range": y_range, "res": res}


def _grid_to_world(ix, iy, x_range, y_range, res):
    """Indeks (boleh pecahan) -> meter; pusat sel [i] ada di x_range[0] + (i+0.5)*res."""
    return x_range[0] + (np.asarray(ix) + 0.5) * res, y_range[0] + (np.asarray(iy) + 0.5) * res


def _world_to_grid(x, y, x_range, y_range, res):
    return (np.asarray(x) - x_range[0]) / res - 0.5, (np.asarray(y) - y_range[0]) / res - 0.5


def ego_cell(x_range, y_range, res) -> Tuple[int, int]:
    return int(np.floor((0.0 - x_range[0]) / res)), int(np.floor((0.0 - y_range[0]) / res))


# ----------------------------------------------------------------------------
# A. penghalusan
# ----------------------------------------------------------------------------
def smooth_mask(mask: np.ndarray, res: float, x_range, y_range, cfg: PolygonConfig) -> np.ndarray:
    """Seperti versi lama, tetapi morfologi dijalankan pada array ber-pad 'edge' (lihat PERUBAHAN no. 1)."""
    if not mask.any():
        return mask.copy()
    close_r = max(1, int(round(cfg.close_radius_m / res)))
    open_r = max(1, int(round(cfg.open_radius_m / res)))
    st = ndi.generate_binary_structure(2, 1)
    pad = max(close_r, open_r) + 2
    mp = np.pad(mask, pad, mode="edge")
    mp = ndi.binary_closing(mp, structure=st, iterations=close_r)
    mp = ndi.binary_opening(mp, structure=st, iterations=open_r)
    m = mp[pad:-pad, pad:-pad]

    if cfg.max_fill_hole_area_m2 > 0:
        filled = ndi.binary_fill_holes(m)
        newly = filled & ~m
        if newly.any():
            lbl, n = ndi.label(newly, structure=np.ones((3, 3), bool))
            for i in range(1, n + 1):
                comp = lbl == i
                if comp.sum() * res * res <= cfg.max_fill_hole_area_m2:
                    m = m | comp

    if cfg.keep_only_ego_component:
        lbl, n = ndi.label(m, structure=np.ones((3, 3), bool))
        if n > 0:
            eix, eiy = ego_cell(x_range, y_range, res)
            H, W = m.shape
            if 0 <= eix < H and 0 <= eiy < W and lbl[eix, eiy] > 0:
                keep_id = lbl[eix, eiy]
            else:
                sizes = ndi.sum(m, lbl, index=range(1, n + 1))
                keep_id = int(np.argmax(sizes)) + 1
            m = lbl == keep_id
    return m


# ----------------------------------------------------------------------------
# B. mask -> cincin (poligon luar + lubang)
# ----------------------------------------------------------------------------
def _signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _rings_subpixel(mask: np.ndarray, x_range, y_range, res: float, cfg: PolygonConfig):
    """Marching squares (contourpy) -> [(outer_ring, [hole_rings])] dalam meter [x,y]."""
    Z = mask.astype(np.float32)
    if cfg.subpixel_sigma_px > 0 and _HAVE_CV2:
        Z = cv2.GaussianBlur(Z, (0, 0), cfg.subpixel_sigma_px)
    Zp = np.pad(Z, 1, mode="constant", constant_values=0.0)   # tutup kontur di batas grid
    gen = contourpy.contour_generator(z=Zp, fill_type=contourpy.FillType.OuterOffset)
    pts_list, off_list = gen.filled(0.5, np.inf)
    out = []
    for pts, off in zip(pts_list, off_list):
        rings = []
        for a, b in zip(off[:-1], off[1:]):
            r = np.asarray(pts[a:b], float)
            if len(r) > 1 and np.allclose(r[0], r[-1]):
                r = r[:-1]
            if len(r) < 3:
                continue
            col, row = r[:, 0] - 1.0, r[:, 1] - 1.0           # contourpy: x = kolom (iy), y = baris (ix); -1 utk pad
            wx, wy = _grid_to_world(row, col, x_range, y_range, res)
            rings.append(np.stack([wx, wy], 1))
        if rings:
            out.append((rings[0], rings[1:]))
    return out


def _rings_pixel(mask: np.ndarray, x_range, y_range, res: float, cfg: PolygonConfig):
    """Cadangan: kontur piksel cv2 (menyusut ~setengah sel). RETR_CCOMP -> luar + lubang."""
    u8 = mask.astype(np.uint8) * 255
    contours, hier = cv2.findContours(u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    conv = {}
    for i, c in enumerate(contours):
        p = c.reshape(-1, 2)
        if len(p) < 3:
            continue
        wx, wy = _grid_to_world(p[:, 1].astype(float), p[:, 0].astype(float), x_range, y_range, res)
        conv[i] = np.stack([wx, wy], 1)
    out = []
    for i, r in conv.items():
        if hier[i][3] == -1:
            holes = [conv[j] for j in conv if hier[j][3] == i]
            out.append((r, holes))
    return out


def extract_rings(mask: np.ndarray, x_range, y_range, res: float, cfg: PolygonConfig):
    """Return (polygons: List[ndarray(N,2)], holes: List[(parent_idx, ndarray)]) dalam meter, sudah disederhanakan,
    luar CCW, lubang CW."""
    if not _HAVE_CV2:
        raise RuntimeError("opencv-python diperlukan: pip install opencv-python-headless")
    if not mask.any():
        return [], []
    if cfg.subpixel and _HAVE_CONTOURPY:
        raw = _rings_subpixel(mask, x_range, y_range, res, cfg)
    else:
        raw = _rings_pixel(mask, x_range, y_range, res, cfg)

    def simp(r):
        a = cv2.approxPolyDP(r.astype(np.float32).reshape(-1, 1, 2), cfg.simplify_tol_m, True)
        return a.reshape(-1, 2).astype(float)

    polys, holes = [], []
    for outer, hs in raw:
        o = simp(outer)
        if len(o) < 3 or abs(_signed_area(o)) < cfg.min_polygon_area_m2:
            continue
        if _signed_area(o) < 0:
            o = o[::-1]
        pid = len(polys)
        polys.append(o)
        if cfg.keep_holes:
            for h in hs:
                h = simp(h)
                if len(h) < 3 or abs(_signed_area(h)) < cfg.min_hole_area_m2:
                    continue
                if _signed_area(h) > 0:
                    h = h[::-1]
                holes.append((pid, h))
    return polys, holes


def _ring_json(r: np.ndarray, dec: int) -> List[List[float]]:
    return [[round(float(x), dec), round(float(y), dec)] for x, y in r]


# ----------------------------------------------------------------------------
# C. jarak ego ke tepi jalan
# ----------------------------------------------------------------------------
_DIRS = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


def _cast_edge(mask, x_range, y_range, res, ux, uy, cfg: PolygonConfig):
    """Return (jarak|None, censored). Jarak = dari ego ke batas jalan->bukan-jalan pertama sepanjang sinar."""
    x0, x1, y0, y1 = x_range[0], x_range[1], y_range[0], y_range[1]
    dmax = min((x1 / ux) if ux > 0 else (x0 / ux) if ux < 0 else np.inf,
               (y1 / uy) if uy > 0 else (y0 / uy) if uy < 0 else np.inf)
    step = cfg.edge_step_m
    ds = np.arange(0.0, dmax - 1e-9, step)
    ix = np.floor((ux * ds - x0) / res).astype(int)
    iy = np.floor((uy * ds - y0) / res).astype(int)
    H, W = mask.shape
    ok = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    vals = np.zeros(len(ds), bool)
    vals[ok] = mask[ix[ok], iy[ok]]
    if not vals.any():
        return None, False
    first = int(np.argmax(vals))
    if ds[first] > cfg.ego_snap_m:                 # ego tidak berada di/dekat jalan
        return None, False
    last = first
    for j in range(first + 1, len(ds)):
        if vals[j]:
            last = j
        elif ds[j] - ds[last] > cfg.edge_gap_tol_m:
            return float(ds[last] + step / 2), False
    return float(dmax), True                         # sinar habis di batas grid: nilai = batas bawah


def ego_edge_distances(mask: np.ndarray, x_range, y_range, res: float, cfg: PolygonConfig) -> dict:
    """mask = masker yang SUDAH dihaluskan. Sinar dari ego (0,0) ke depan/belakang/kiri/kanan (kiri = +y)."""
    eix, eiy = ego_cell(x_range, y_range, res)
    H, W = mask.shape
    on_road = bool(0 <= eix < H and 0 <= eiy < W and mask[eix, eiy])
    out, cens = {}, []
    for name, (ux, uy) in _DIRS.items():
        d, c = _cast_edge(mask, x_range, y_range, res, ux, uy, cfg)
        out[name] = None if d is None else round(d, 2)
        if c and d is not None:
            cens.append(name)
    lo = None
    if out["left"] is not None and out["right"] is not None and "left" not in cens and "right" not in cens:
        lo = round((out["left"] - out["right"]) / 2.0, 2)
    # ego_on_road: sel ego jalan ATAU sinar bisa dimulai (dalam ego_snap_m) -> semua nilai tersedia
    out.update({"censored": cens, "ego_on_road": on_road, "lane_center_offset_m": lo})
    return out


# ----------------------------------------------------------------------------
# G. tipe jalan via lengan radial (DIPERTAHANKAN dari fase3_lib.py lama; opsional, Fase 5)
# ----------------------------------------------------------------------------
def _sample_ring(mask: np.ndarray, x_range, y_range, res: float, radius_m: float, angle_step_deg: float) -> np.ndarray:
    angles = np.deg2rad(np.arange(0, 360, angle_step_deg))
    x = radius_m * np.cos(angles)
    y = radius_m * np.sin(angles)
    ix, iy = _world_to_grid(x, y, x_range, y_range, res)
    ix = np.round(ix).astype(int)
    iy = np.round(iy).astype(int)
    H, W = mask.shape
    inb = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    hit = np.zeros(len(angles), bool)
    hit[inb] = mask[ix[inb], iy[inb]]
    return hit  # True = sel ini jalan, indeks sejajar dgn `angles`


def _circular_runs(hit: np.ndarray, angle_step_deg: float) -> List[Tuple[float, float]]:
    """hit: array bool melingkar -> list (center_deg, width_deg) tiap run True yg kontigu,
    menangani wraparound 360->0 dgn benar."""
    n = len(hit)
    if not hit.any():
        return []
    if hit.all():
        return [(0.0, 360.0)]
    # putar array supaya mulai dari sebuah False, biar run gak kepotong di indeks 0/n
    start = int(np.argmax(~hit))
    rolled = np.roll(hit, -start)
    runs = []
    i = 0
    while i < n:
        if rolled[i]:
            j = i
            while j < n and rolled[j]:
                j += 1
            lo = (start + i) * angle_step_deg
            hi = (start + j) * angle_step_deg
            center = ((lo + hi) / 2.0) % 360.0
            width = hi - lo
            runs.append((center, width))
            i = j
        else:
            i += 1
    return runs


def _circular_close_gaps(hit: np.ndarray, angle_step_deg: float, max_gap_deg: float) -> np.ndarray:
    """Jembatani celah SUDUT pendek (<= max_gap_deg) di array boolean melingkar. Perlu krn
    sampling radial dgn indeks grid dibulatkan (nearest-neighbor) bisa bikin dip 1-2 bin
    akibat aliasing piksel, bukan celah nyata -- tanpa ini satu lengan lebar bisa kepecah
    jadi beberapa lengan sempit palsu (diverifikasi nyata di frame 744)."""
    max_gap_bins = int(round(max_gap_deg / angle_step_deg))
    if max_gap_bins <= 0 or not hit.any() or hit.all():
        return hit
    gaps = _circular_runs(~hit, angle_step_deg)
    out = hit.copy()
    n = len(hit)
    for center, width in gaps:
        if width <= max_gap_deg:
            lo_deg = (center - width / 2) % 360
            lo_bin = int(round(lo_deg / angle_step_deg))
            n_bins = int(round(width / angle_step_deg))
            for b in range(n_bins):
                out[(lo_bin + b) % n] = True
    return out


def ring_sample_arms(mask: np.ndarray, x_range, y_range, res: float, f3: PolygonConfig) -> List[Dict[str, float]]:
    """Deteksi lengan via celah-terpanjang di sepanjang sinar radial (radial gap test).

    RIWAYAT PERBAIKAN DI FUNGSI INI (ditulis krn parameter di sini TERBUKTI sensitif &
    perlu verifikasi visual, bukan cuma tebak angka):
    1) Versi awal: voting antar-cincin ring_vote_frac=0.6 dgn 2 radius scr matematis jadi
       AND (harus cocok di KEDUANYA) -> jalan lurus yg jangkauan depannya kebetulan cuma
       ~8m (sangat umum) salah dilabel 'buntu'. Terbukti dari distribusi batch nyata: buntu
       33% (459/1371), jauh lebih tinggi dari wajar utk rute yg terus bergerak.
    2) Fix pertama: syarat *fraksi* sampel road sepanjang sinar dari dekat ego s.d radius
       terdekat >= 0.85. Ternyata match_fused di dekat ego (~1.5-6m) SERING punya cakupan
       rendah scr WAJAR (bukan noise) krn ego kerap berada di ruang sempit/dikelilingi
       objek (lihat frame 744: frac_road di cincin tunggal r=3m cuma 0.26) -- syarat fraksi
       rata-rata sepanjang seluruh rentang jarak jadi kelewat ketat, hasilnya malah lebih
       parah: buntu naik jadi 64% (877/1371) di re-run batch nyata user.
    3) Fix ini: ganti ke *celah terpanjang* (meter) sepanjang sinar, bukan fraksi rata-rata.
       Lebih tahan thd noise sporadis (celah pendek² tersebar tak bikin gap panjang) tapi
       tetap menolak blokade nyata (tembok dsb, gap panjang & kontinu). TAPI: nilai
       `max_continuous_gap_m` yg pas ternyata SANGAT sensitif (diuji di frame 744: 1.0m/1.5m
       -> 1 lengan, 2.0m -> 2 lengan, 2.5m -> 3 lengan, 3.0m -> 1 lengan lagi -- tak stabil
       tanpa referensi visual). Makanya disediakan plot_frame_diagnostics() -- JANGAN percaya
       default di bawah begitu saja, cocokkan dulu ke overlay visual di beberapa frame nyata
       sebelum lanjut batch, drpd nebak parameter lagi tanpa bukti (itu yg dua kali salah).
    """
    n_ang = int(round(360 / f3.ring_angle_step_deg))
    r_far = max(f3.ring_radii_m)
    r_near = min(f3.ring_radii_m)
    step = f3.radial_probe_step_m
    r_samples = np.arange(step, r_far + 1e-9, step)
    H, W = mask.shape

    longest_gap = np.zeros(n_ang)
    reach = np.zeros(n_ang)
    for k in range(n_ang):
        ang = math.radians(k * f3.ring_angle_step_deg)
        xs = r_samples * math.cos(ang)
        ys = r_samples * math.sin(ang)
        ix, iy = _world_to_grid(xs, ys, x_range, y_range, res)
        ix = np.round(ix).astype(int)
        iy = np.round(iy).astype(int)
        inb = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
        vals = np.zeros(len(r_samples), bool)
        vals[inb] = mask[ix[inb], iy[inb]]
        best = cur = 0.0
        last_true_r = 0.0
        for rr, v in zip(r_samples, vals):
            if v:
                cur = 0.0
                last_true_r = rr
            else:
                cur += step
                best = max(best, cur)
        longest_gap[k] = best
        reach[k] = last_true_r

    reachable = longest_gap <= f3.max_continuous_gap_m
    reachable = _circular_close_gaps(reachable, f3.ring_angle_step_deg, f3.max_gap_close_deg)
    runs = _circular_runs(reachable, f3.ring_angle_step_deg)
    arms = []
    for c, w in runs:
        if w < f3.min_arm_width_deg:
            continue
        lo = int(round((c - w / 2) / f3.ring_angle_step_deg))
        hi = int(round((c + w / 2) / f3.ring_angle_step_deg))
        idxs = [i % n_ang for i in range(lo, hi + 1)]
        arms.append({"angle_deg": round(c, 1), "width_deg": round(w, 1),
                     "reach_m": round(float(np.max(reach[idxs])), 1)})
    arms.sort(key=lambda a: a["angle_deg"])
    return arms



def classify_road_type(arms: List[Dict[str, float]], f3: PolygonConfig) -> str:
    n = len(arms)
    if n == 0:
        return "tidak_jelas"
    if n == 1:
        return "buntu"
    if n == 2:
        d = abs(arms[0]["angle_deg"] - arms[1]["angle_deg"])
        d = min(d, 360 - d)
        if abs(d - 180) <= f3.straight_angle_tol_deg:
            return "lurus"
        return "belok"
    if n == 3:
        return "pertigaan"
    if n == 4:
        return "perempatan"
    return "kompleks"


# ----------------------------------------------------------------------------
# D. gabungan per-frame + batch
# ----------------------------------------------------------------------------
def process_frame(cfg: PolygonConfig, idx: int, return_internal: bool = False, state=None):
    if cfg.r1 is not None:      # (rev 3) Fase R1: bukti tri-state + perbaikan lubang + penghalusan tepi
        import road_evidence_lib as rel
        return rel.process_frame_r1(cfg, idx, state=state, return_internal=return_internal)
    fr = load_mask_frame(cfg, idx)
    mask, x_range, y_range, res = fr["mask"], fr["x_range"], fr["y_range"], fr["res"]
    sm = smooth_mask(mask, res, x_range, y_range, cfg)
    polys, holes = extract_rings(sm, x_range, y_range, res, cfg)
    dec = cfg.coord_decimals
    edges = ego_edge_distances(sm, x_range, y_range, res, cfg)
    road_type = None
    arms = []
    if cfg.compute_road_type:
        arms = ring_sample_arms(sm, x_range, y_range, res, cfg)
        road_type = {"n_arms": len(arms), "arms": arms, "label": classify_road_type(arms, cfg)}
    result = {
        "idx": fr["idx"], "t": fr["t"],
        "road": {
            "polygons": [_ring_json(p, dec) for p in polys],
            "holes": [{"parent": pid, "ring": _ring_json(h, dec)} for pid, h in holes],
            "res_equiv_m": res,
        },
        "ego_edge_m": edges,
        "road_type": road_type,
    }
    if return_internal:
        return result, {"mask": mask, "smooth": sm, "x_range": x_range, "y_range": y_range, "res": res, "arms": arms}
    return result


def save_frame_json(cfg: PolygonConfig, result: dict) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    p = cfg.out_dir / f"{result['idx']:06d}.json"
    p.write_text(json.dumps(result, separators=(",", ":")))
    return p


def naive_grid_json_bytes(mask: np.ndarray) -> int:
    """Ukuran JSON bila mask disimpan mentah sbg array 0/1 datar (baseline). Tiap elemen 1 karakter:
    N elemen + (N-1) koma + 2 kurung = 2N+1 byte -- sama persis dgn json.dumps(list, separators=(',',':'))."""
    return 2 * int(mask.size) + 1


def run_batch(cfg: PolygonConfig, indices: Optional[Sequence[int]] = None, log=print, poses=None, valid=None):
    import pandas as pd
    if indices is None:
        indices = list_mask_frames(cfg)
    if not len(indices):
        log(f"TIDAK ADA masker (*.npz) di {cfg.masks_dir} -> jalankan build_road_masks.py dulu atau arahkan masks_dir ke folder masker.")
        return pd.DataFrame()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    state = None
    if cfg.r1 is not None and cfg.r1.temporal and poses is not None:      # prior temporal kausal: frame harus diproses berurutan
        import road_evidence_lib as rel
        state = rel.TemporalState(cfg.r1, poses, valid)
    for n_done, idx in enumerate(indices, 1):
        try:
            res, internal = process_frame(cfg, int(idx), return_internal=True, state=state)
        except Exception as e:  # noqa: BLE001
            log(f"  [{int(idx):06d}] GAGAL: {e}")
            continue
        p = save_frame_json(cfg, res)
        e = res["ego_edge_m"]
        naive = naive_grid_json_bytes(internal["mask"])
        rows.append({
            "idx": res["idx"], "t": res["t"],
            "n_polygons": len(res["road"]["polygons"]), "n_holes": len(res["road"]["holes"]),
            "n_verts_total": sum(len(pl) for pl in res["road"]["polygons"]),
            "json_bytes": p.stat().st_size, "naive_grid_bytes": naive,
            "size_ratio": p.stat().st_size / naive,
            "cells_raw": int(internal["mask"].sum()), "cells_smooth": int(internal["smooth"].sum()),
            "ego_on_road": e["ego_on_road"],
            "ego_near_road": any(e[k] is not None for k in ("front", "back", "left", "right")),
            "edge_front": e["front"], "edge_back": e["back"],
            "edge_left": e["left"], "edge_right": e["right"], "n_censored": len(e["censored"]),
            "lane_center_offset_m": e["lane_center_offset_m"],
            "road_label": res["road_type"]["label"] if res["road_type"] else None,
            "inferred_frac": res["road"].get("inf"),
        })
        if n_done % 100 == 0:
            log(f"  {n_done}/{len(indices)}")
    df = pd.DataFrame(rows)
    if len(df):
        log(f"selesai: {len(df)} frame -> {cfg.out_dir}")
        log(f"json/frame: median {df['json_bytes'].median()/1024:.1f} KB (maks {df['json_bytes'].max()/1024:.1f} KB) | "
            f"baseline grid mentah {df['naive_grid_bytes'].median()/1024:.0f} KB | rasio median {df['size_ratio'].median():.3f}")
        log(f"ego_on_road: {df['ego_on_road'].mean():.1%} frame | poligon kosong: {(df['n_polygons'] == 0).sum()} frame")
    return df


# ----------------------------------------------------------------------------
# E. stabilitas antar-frame (IoU setelah kompensasi gerak ego)
# ----------------------------------------------------------------------------
def load_ego_poses(root, lidar_dir=None, odom_csv=None, odom_time_offset: float = 0.0,
                   lidar_in_base: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Pose SE(2) T_world_lidar per scan, idx = urutan waktu nama berkas lidar (sama dgn road_lib.Sequence & sync_table.idx).
    Memakai road_lib.py (Fase 1 lama) apa adanya. Return (t, poses (N,3,3), valid (N,))."""
    import road_lib as rl
    c = rl.Config(root=root, lidar_dir=lidar_dir, odom_csv=odom_csv,
                  odom_time_offset=odom_time_offset, lidar_in_base=lidar_in_base)
    stamped = []
    for f in sorted(c.lidar_dir.glob("*.npy")):
        m = rl._STEM_RE.search(f.stem)
        if m:
            sec, frac = m.groups()
            stamped.append(int(sec) + int(frac) / 10 ** len(frac))
    if not stamped:
        raise FileNotFoundError(f"tidak ada lidar_<detik>.<pecahan>.npy di {c.lidar_dir}")
    t = np.array(sorted(stamped), dtype=np.float64)
    odom = rl.load_odom(c.odom_csv, c.drop_identity_prefix)
    poses, valid = rl.build_lidar_poses(t, odom, c)
    return t, poses, valid


def load_waypoints_ego(csv_path) -> Dict[int, np.ndarray]:
    """sync_table_with_waypoints.csv, kolom 'waypoints_ego' (string '[[x, y], ...]') -> {idx: array (K,2)}.
    ASUMSI konvensi [x depan, y kiri] di frame ego -- VERIFIKASI di 1 frame belokan yang rutenya kamu kenal
    (belok kiri harus menghasilkan y > 0)."""
    import ast
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["idx", "waypoints_ego"])
    out = {}
    for i, w in zip(df["idx"], df["waypoints_ego"]):
        try:
            arr = np.asarray(ast.literal_eval(w) if isinstance(w, str) else [], float).reshape(-1, 2)
        except (ValueError, SyntaxError):
            continue
        if len(arr):
            out[int(i)] = arr
    return out


def polygons_to_mask(road: dict, x_range, y_range, res: float) -> np.ndarray:
    """Rasterisasi ulang poligon (+lubang) ke grid [ix, iy] -- utk mengukur IoU apa yang BENAR-BENAR dikirim ke viewer."""
    nx = int(round((x_range[1] - x_range[0]) / res))
    ny = int(round((y_range[1] - y_range[0]) / res))
    img = np.zeros((nx, ny), np.uint8)

    def px(ring):
        a = np.asarray(ring, float)
        ix, iy = _world_to_grid(a[:, 0], a[:, 1], x_range, y_range, res)
        return np.round(np.stack([iy, ix], 1) * 16).astype(np.int32)     # cv2: (kolom, baris), shift=4

    for p in road.get("polygons", []):
        cv2.fillPoly(img, [px(p)], 1, shift=4)
    for h in road.get("holes", []):
        cv2.fillPoly(img, [px(h["ring"])], 0, shift=4)
    return img.astype(bool)


def _iou_pair(ma, mb, Ta, Tb, gx, gy, x_range, y_range, res, rng_m):
    rel = np.linalg.inv(Tb) @ Ta                       # titik frame-a -> frame-b
    px = rel[0, 0] * gx + rel[0, 1] * gy + rel[0, 2]
    py = rel[1, 0] * gx + rel[1, 1] * gy + rel[1, 2]
    jx = np.floor((px - x_range[0]) / res).astype(int)
    jy = np.floor((py - y_range[0]) / res).astype(int)
    nx, ny = ma.shape
    inb = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
    sb = np.zeros_like(ma)
    sb[inb] = mb[jx[inb], jy[inb]]
    region = inb & (np.hypot(gx, gy) <= rng_m)
    inter = int((ma & sb & region).sum())
    union = int(((ma | sb) & region).sum())
    return (inter / union if union else float("nan")), union


def temporal_stability(cfg: PolygonConfig, poses: np.ndarray, valid: np.ndarray,
                       indices: Optional[Sequence[int]] = None, lags: Sequence[int] = (1, 5), log=print):
    """IoU antara poligon frame a dan frame a+lag, frame b di-warp ke frame a dgn pose odom (SE(2)).
    Hanya dihitung dlm radius cfg.stability_eval_range_m dari ego & di area yg tumpang tindih kedua grid.
    Catatan: sampling nearest-neighbour menyumbang ~+-0.1 m jitter -> IoU sedikit lebih rendah dari aslinya."""
    import pandas as pd
    avail = sorted(int(p.stem) for p in cfg.out_dir.glob("*.json") if p.stem.isdigit())
    if indices is None:
        indices = avail
    avail_set = set(avail)
    cache: Dict[int, np.ndarray] = {}
    meta = {}

    def get_mask(i):
        if i not in cache:
            d = json.loads((cfg.out_dir / f"{i:06d}.json").read_text())
            fr = load_mask_frame(cfg, i)                    # hanya utk x_range/y_range/res
            meta.setdefault("g", (fr["x_range"], fr["y_range"], fr["res"]))
            cache[i] = polygons_to_mask(d["road"], fr["x_range"], fr["y_range"], fr["res"])
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[i]

    rows = []
    gx = gy = None
    for a in indices:
        for lag in lags:
            b = a + lag
            if b not in avail_set or a not in avail_set or a >= len(poses) or b >= len(poses):
                continue
            if not (valid[a] and valid[b]):
                continue
            ma, mb = get_mask(a), get_mask(b)
            xr, yr, res = meta["g"]
            if gx is None:
                ix, iy = np.meshgrid(np.arange(ma.shape[0]), np.arange(ma.shape[1]), indexing="ij")
                gx, gy = _grid_to_world(ix, iy, xr, yr, res)
            iou, un = _iou_pair(ma, mb, poses[a], poses[b], gx, gy, xr, yr, res, cfg.stability_eval_range_m)
            rows.append({"idx_a": a, "idx_b": b, "lag": lag, "iou": iou, "n_union": un})
    df = pd.DataFrame(rows, columns=["idx_a", "idx_b", "lag", "iou", "n_union"])   # kolom selalu ada, walau kosong
    if not len(df):
        log(f"[PERINGATAN] 0 pasangan frame berurutan (lag {tuple(lags)}) -> IoU TIDAK terukur. Masker hanya ada utk idx "
            f"{sorted(avail_set)[:8]}{' ...' if len(avail_set) > 8 else ''}. Bangun masker frame BERURUTAN, mis.: "
            f"python build_road_masks.py --root . --frames test --span {max(lags) + 5}")
    if len(df):
        for lag in lags:
            s = df[df["lag"] == lag]["iou"].dropna()
            if len(s):
                log(f"lag {lag}: n={len(s)} IoU median {s.median():.3f} | p10 {s.quantile(0.1):.3f} | min {s.min():.3f} | "
                    f">= {cfg.target_iou}: {(s >= cfg.target_iou).mean():.1%}")
    return df


def verdict(df_batch, df_stab=None, cfg: Optional[PolygonConfig] = None, t_track_start: Optional[float] = None):
    """Checkpoint Fase 1. t_track_start = timestamp LiDAR pertama dgn pose valid (ZED mulai tracking); frame dgn
    t < t_track_start + cfg.warmup_s dikeluarkan dari cek ego_on_road (akumulasi belum penuh). Tanpa ini semua frame dihitung."""
    import pandas as pd
    assert cfg is not None
    rows = []

    def add(name, val, tgt, op):
        ok = (val >= tgt) if op == ">=" else (val <= tgt)
        rows.append({"cek": name, "nilai": round(float(val), 3), "target": tgt, "hasil": "LULUS" if ok else "PERIKSA"})

    def info(name, val, tgt=float("nan")):
        rows.append({"cek": name, "nilai": val, "target": tgt, "hasil": "INFO"})

    add("Payload poligon / baseline grid mentah (median rasio)", float(df_batch["size_ratio"].median()), 1.0, "<=")
    add("Frame dengan poligon kosong (jumlah)", float((df_batch["n_polygons"] == 0).sum()), 0, "<=")

    if t_track_start is not None and "t" in df_batch and df_batch["t"].notna().all():
        warm = df_batch["t"] < (t_track_start + cfg.warmup_s)
    else:
        warm = pd.Series(False, index=df_batch.index)
    steady = df_batch[~warm]
    if len(steady):
        add(f"Ego berada di jalan, di luar pemanasan (n={len(steady)} frame; fraksi)", float(steady["ego_on_road"].mean()),
            cfg.target_ego_on_road, ">=")
    else:
        rows.append({"cek": "Ego di jalan: SEMUA frame sampel ada di masa pemanasan -> tak bisa dinilai", "nilai": float("nan"),
                     "target": cfg.target_ego_on_road, "hasil": "PERIKSA"})
    info("info: ego_on_road, SEMUA frame (termasuk pemanasan)", round(float(df_batch["ego_on_road"].mean()), 3), cfg.target_ego_on_road)
    if "ego_near_road" in df_batch:
        info("info: ego di/dekat jalan (sinar ego valid dlm ego_snap_m)", round(float(df_batch["ego_near_road"].mean()), 3))
    bad = df_batch.loc[~df_batch["ego_on_road"], "idx"].astype(int).tolist()
    if bad:
        info(f"info: idx ego TIDAK di jalan (pemanasan: {[i for i in bad if warm[df_batch['idx'] == i].any()]})", str(bad[:12]))

    if df_stab is not None and len(df_stab) and "lag" in df_stab:
        lag0 = int(df_stab["lag"].min())
        s1 = df_stab[df_stab["lag"] == lag0]["iou"].dropna()
        if len(s1):
            add(f"IoU poligon antar-frame (lag {lag0}, median) -- target USULAN, sepakati dgn pembimbing",
                float(s1.median()), cfg.target_iou, ">=")
            info("info: IoU p10 (frame terburuk 10%)", round(float(s1.quantile(0.1)), 3))
            info("info: fraksi pasangan frame dgn IoU >= target", round(float((s1 >= cfg.target_iou).mean()), 3))
    if not any(r["cek"].startswith("IoU poligon antar-frame") for r in rows):
        rows.append({"cek": "IoU poligon antar-frame: BELUM DIUKUR (butuh masker frame berurutan)", "nilai": float("nan"),
                     "target": cfg.target_iou, "hasil": "PERIKSA"})
    info("info: json/frame median [KB]", round(float(df_batch["json_bytes"].median()) / 1024, 2))
    info("info: sinar ego->tepi yang 'censored' (rata-rata per frame, maks 4)", round(float(df_batch["n_censored"].mean()), 2))
    lc = df_batch["lane_center_offset_m"].dropna()
    if len(lc):
        info("info: |offset pusat jalan thd ego| median [m] (>0 = ego tidak di tengah)", round(float(lc.abs().median()), 2))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# F. plot diagnostik (DIPERBAIKI: masker tidak lagi terbalik kiri-kanan)
# ----------------------------------------------------------------------------
def _show_mask(ax, m, x_range, y_range, cmap="Greens", alpha=0.85):
    # sumbu horizontal = y dengan +y (kiri mobil) di KIRI layar -> extent (y1, y0); kolom iy harus dibalik agar
    # iy=0 (y = y_range[0], paling kanan) jatuh di sisi kanan layar. Versi lama lupa membalik.
    extent = (y_range[1], y_range[0], x_range[0], x_range[1])
    ax.imshow(m[:, ::-1], extent=extent, origin="lower", cmap=cmap, vmin=0, vmax=1, alpha=alpha, interpolation="nearest")
    ax.set_xlim(y_range[1], y_range[0])
    ax.set_ylim(x_range[0], x_range[1])
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]  (kiri mobil = kiri layar)")
    ax.set_ylabel("x [m]  (depan)")


def plot_frame_diagnostics(cfg: PolygonConfig, idx: int, save_path: Optional[Path] = None,
                           waypoints_xy: Optional[np.ndarray] = None):
    """Kiri: mask_fused mentah. Kanan: setelah smooth + poligon (biru) + lubang (merah) + ego (segitiga merah di 0,0)
    + jarak ego ke tepi (oranye) + waypoint (hijau; waypoints_xy = array (K,2) [x depan, y kiri] dlm frame LiDAR ego)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    res_, it = process_frame(cfg, idx, return_internal=True)
    mask, sm, xr, yr = it["mask"], it["smooth"], it["x_range"], it["y_range"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    for ax, m, title in zip(axes, [mask, sm], ["mask_fused (mentah)", "setelah smooth + poligon"]):
        _show_mask(ax, m, xr, yr)
        ax.set_title(title)
        ax.add_patch(MplPoly([(0, 1.2), (-0.6, -0.6), (0.6, -0.6)], closed=True, fc="#d62728", ec="k", zorder=6))
    ax = axes[1]
    for p in res_["road"]["polygons"]:
        a = np.array(p + [p[0]])
        ax.plot(a[:, 1], a[:, 0], color="#1f4ed8", lw=1.4)
    for h in res_["road"]["holes"]:
        a = np.array(h["ring"] + [h["ring"][0]])
        ax.plot(a[:, 1], a[:, 0], color="#d62728", lw=1.2, ls="--")
    e = res_["ego_edge_m"]
    for name, (ux, uy) in _DIRS.items():
        d = e[name]
        if d is None:
            continue
        ex, ey = ux * d, uy * d
        ax.plot([0, ey], [0, ex], color="#ff8c00", lw=2, zorder=5)
        ax.annotate(("≥" if name in e["censored"] else "") + f"{d:.1f} m", (ey, ex), color="#b45f00", fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    if waypoints_xy is not None and len(waypoints_xy):
        w = np.asarray(waypoints_xy, float)
        ax.plot(w[:, 1], w[:, 0], color="#1b9e3e", lw=2, marker="o", ms=3, zorder=5, label="waypoint")
        ax.legend(loc="lower left", fontsize=8)
    lo = e["lane_center_offset_m"]
    fig.suptitle(f"idx={idx}  poligon={len(res_['road']['polygons'])}  lubang={len(res_['road']['holes'])}  "
                 f"ego_on_road={e['ego_on_road']}  offset pusat jalan="
                 f"{'-' if lo is None else f'{lo:+.2f} m (+ = pusat di kiri ego)'}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return save_path
    return fig


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="BEV ROAD Fase 1: mask_fused -> poligon jalan JSON")
    ap.add_argument("--root", default=".")
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--frames", default="all", help="all | 150,191,744 | awal:akhir[:langkah]")
    a = ap.parse_args()
    cfg = PolygonConfig(root=a.root, masks_dir=a.masks_dir, out_dir=a.out_dir)
    avail = list_mask_frames(cfg)
    if not avail:
        print(f"TIDAK ADA masker (*.npz) di {cfg.masks_dir}. Beri --masks-dir <folder> atau jalankan build_road_masks.py dulu.")
        return 1
    if a.frames == "all":
        idxs = avail
    elif ":" in a.frames:
        p = [int(x) for x in a.frames.split(":")]
        lo, hi, st = (p + [1])[:3] if len(p) == 2 else p[:3]
        idxs = [i for i in avail if lo <= i <= hi and (i - lo) % st == 0]
    else:
        want = {int(x) for x in a.frames.split(",") if x.strip()}
        idxs = [i for i in avail if i in want]
    run_batch(cfg, idxs)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
