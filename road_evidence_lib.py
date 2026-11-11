#!/usr/bin/env python3
"""
road_evidence_lib.py -- BEV ROAD, FASE R1: jalan TIDAK bolong & tepi TIDAK bergerigi, tanpa mengubah bentuk nyata.

Dipakai oleh road_polygon_lib.py (rev 3) lewat  cfg.r1 = road_evidence_lib.R1Config()  (None = perilaku lama rev 2, persis).
Tidak memakai kamera/LiDAR mentah: hanya berkas masker <idx>.npz hasil build_road_masks.py (hit-count per sel sudah ada di
sana) + objek Fase 3 (fase3_deteksi/objects/<idx>.json, hanya DIBACA).

Tiga langkah (urutan = urutan eksekusi):
  R1a  BUKTI tri-state per sel:  JALAN (+1) | BUKAN-JALAN (-1) | TAK-TAHU (0).
       "Tidak ada bukti jalan" tidak lagi dianggap "bukan jalan": bayangan/oklusi objek, area di luar jangkauan sensor,
       dan sel tanpa pengamatan = TAK-TAHU. Hanya bukti aktif (kamera: vegetasi/trotoar/bangunan; LiDAR: rintangan
       di luar jejak objek) yang menjadi BUKAN-JALAN.
  R1b  PERBAIKAN LUBANG (hanya menyentuh sel TAK-TAHU, tidak pernah sel BUKAN-JALAN):
         1. (opsional) prior temporal kausal: sel yang jalan di masa LALU (di-warp dgn pose SE(2)) & kini TAK-TAHU.
         2. closing dgn radius ADAPTIF terhadap jarak (dekat kecil, jauh besar; VLP-16 makin jarang di kejauhan).
         3. lubang tertutup: diisi bila (kecil) ATAU (tanpa bukti BUKAN-JALAN & >=60% kelilingnya jalan & <= luas maks).
            Lubang dgn bukti BUKAN-JALAN = pulau NYATA -> dipertahankan (hanya inti bukti yg dijaga).
         4. opening (buang duri) + buang sliver sempit (< sliver_width_m) di luar zona ego, lalu komponen ego.
       Sel hasil tebakan ditandai  `inferred`  (dilaporkan; fraksi harus kecil).
  R1c  PENGHALUSAN TEPI: kontur subpiksel dire-sampling per panjang busur, dihaluskan Gaussian sepanjang busur, lalu
       perpindahan tiap titik DIBATASI <= refine_max_disp_m dari kontur asli (bentuk nyata tidak berubah). Tiap titik
       poligon diberi tipe  vq: '0' = tepi didukung bukti BUKAN-JALAN (tepi nyata), '1' = tepi tanpa bukti
       (batas sensor/tak diketahui -> viewer boleh memudarkan / jangan dipercaya).

JSON per frame: sama dgn rev 2 + road.vq (list string per poligon) + road.inf (fraksi sel tebakan) + road.ir (kontur gumpalan
sel tebakan, dipakai viewer R2 utk memudarkan area yg ditebak).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

import cv2

NOT_ROAD, UNKNOWN, ROAD = -1, 0, 1


# ----------------------------------------------------------------------------
# konfigurasi
# ----------------------------------------------------------------------------
@dataclass
class R1Config:
    # --- R1a: bukti ---
    min_hits: int = 2                    # ambang hit minimum (sama dgn camera_fusion_lib: >=2)
    cam_non_frac: float = 0.2            # frac jalan kamera <= ini (dgn hit>=min_hits) => BUKAN-JALAN (kamera)
    obstacle_min_hits: int = 2           # hit rintangan LiDAR >= ini => BUKAN-JALAN (kecuali di jejak objek)
    object_margin_m: float = 0.5         # jejak objek diperlebar sekian (bayangan/oklusi)
    objects_dir: Optional[Path] = None   # default <root>/fase3_deteksi/objects (hanya dibaca)
    use_objects: bool = True

    # --- R1b: perbaikan lubang ---
    close_near_m: float = 0.6            # radius closing dekat ego
    close_far_m: float = 2.0             # radius closing di jarak close_far_range_m
    close_near_range_m: float = 5.0
    close_far_range_m: float = 20.0
    close_levels: int = 4                # jumlah tingkat radius antara dekat & jauh
    island_min_area_m2: float = 4.0      # lubang tertutup < ini SELALU diisi (noise); >= ini ikut aturan bukti
    fill_max_area_m2: float = 80.0       # lubang tanpa bukti BUKAN-JALAN diisi hanya bila luas <= ini
    fill_min_perimeter_road: float = 0.6 # fraksi keliling lubang yg berbatasan dgn jalan
    island_core_radius_m: float = 0.6    # sel bukti BUKAN-JALAN di dalam lubang dijaga + radius ini (pulau nyata)
    open_radius_m: float = 0.3
    sliver_width_m: float = 1.4          # bagian jalan yg lebih sempit dari ini dibuang (0 = mati)
    sliver_protect_ego_m: float = 3.0    # sliver di dekat ego tidak dibuang
    keep_only_ego_component: bool = True

    # --- prior temporal kausal (OPSIONAL, default mati; hanya memakai frame LAMPAU) ---
    temporal: bool = False
    t_decay: float = 0.85                # peluruhan log-odds per frame
    t_road_gain: float = 0.8
    t_not_gain: float = 1.5
    t_thr: float = 0.5                   # log-odds > ini = "dulu jalan"
    t_max_range_m: float = 25.0
    t_max_gap_frames: int = 5            # loncatan idx > ini / pose tak valid => state di-reset

    # --- R1c: penghalusan tepi ---
    refine: bool = True
    resample_step_m: float = 0.1
    refine_sigma_m: float = 0.5
    refine_max_disp_m: float = 0.3       # batas TOTAL pergeseran titik poligon final dari kontur asli (haluskan + sederhanakan)
    simplify_tol_m: float = 0.08         # Douglas-Peucker setelah dihaluskan (rev 2: 0,25); refine_max_disp_m sudah memperhitungkannya
    min_ring_len_for_refine_m: float = 1.5
    edge_type_radius_m: float = 0.6      # titik poligon dekat bukti BUKAN-JALAN dlm radius ini => vq '0'
    min_limit_run_m: float = 3.0         # potongan tepi 'tanpa bukti' lebih pendek dari ini (diapit tepi berbukti) dianggap berbukti
    min_obs_run_m: float = 1.5           # potongan tepi 'berbukti' lebih pendek dari ini dianggap tanpa bukti (tak perlu garis)

    # --- R2: geometri sel tebakan utk viewer (JSON road.ir) ---
    inferred_rings: bool = True
    inferred_min_area_m2: float = 2.0
    inferred_simplify_m: float = 0.3
    inferred_max_rings: int = 20

    def __post_init__(self):
        if self.objects_dir is not None:
            self.objects_dir = Path(self.objects_dir)


# ----------------------------------------------------------------------------
# util geometri / morfologi
# ----------------------------------------------------------------------------
def _disk(r_cells: int) -> np.ndarray:
    r = max(1, int(r_cells))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _morph(mask: np.ndarray, op: int, r_cells: int) -> np.ndarray:
    """Morfologi dgn padding 'edge' (jalan yang menyentuh batas grid tidak terkikis)."""
    pad = int(r_cells) + 2
    mp = np.pad(mask, pad, mode="edge").astype(np.uint8)
    out = cv2.morphologyEx(mp, op, _disk(r_cells))
    return out[pad:-pad, pad:-pad].astype(bool)


def _cells(m_meter: float, res: float) -> int:
    return max(1, int(round(m_meter / res)))


def range_map(x_range, y_range, res) -> np.ndarray:
    nx = int(round((x_range[1] - x_range[0]) / res))
    ny = int(round((y_range[1] - y_range[0]) / res))
    xs = x_range[0] + (np.arange(nx) + 0.5) * res
    ys = y_range[0] + (np.arange(ny) + 0.5) * res
    return np.hypot(xs[:, None], ys[None, :])


# ----------------------------------------------------------------------------
# R1a: bukti tri-state
# ----------------------------------------------------------------------------
def load_object_boxes(objects_dir: Optional[Path], idx: int) -> List[Tuple[float, float, float, float, float]]:
    """fase3_deteksi/objects/<idx:06d>.json -> [(x, y, panjang, lebar, yaw)] di frame ego. Berkas hilang/rusak -> []."""
    if objects_dir is None:
        return []
    p = Path(objects_dir) / f"{idx:06d}.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    objs = d.get("objects", []) if isinstance(d, dict) else d
    out = []
    for o in objs or []:
        c = o.get("center_xyz") if isinstance(o, dict) else None
        if not c or o.get("length_m") is None or o.get("width_m") is None:
            continue
        out.append((float(c[0]), float(c[1]), float(o["length_m"]), float(o["width_m"]), float(o.get("yaw_rad") or 0.0)))
    return out


def rasterize_boxes(boxes, x_range, y_range, res, margin_m: float, shape) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    for (cx, cy, l, w, yaw) in boxes:
        hl, hw = l / 2.0 + margin_m, w / 2.0 + margin_m
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], float)
        wx = cx + corners[:, 0] * c - corners[:, 1] * s
        wy = cy + corners[:, 0] * s + corners[:, 1] * c
        ix = (wx - x_range[0]) / res - 0.5
        iy = (wy - y_range[0]) / res - 0.5
        pts = np.round(np.stack([iy, ix], 1) * 16).astype(np.int32)      # cv2: (kolom, baris), shift=4
        cv2.fillPoly(img, [pts], 1, shift=4)
    return img.astype(bool)


def build_evidence(npz, road: np.ndarray, boxes, x_range, y_range, res, r1: R1Config) -> Tuple[np.ndarray, dict]:
    """Return (ev int8 [-1,0,+1], aux). `road` = mask_fused (JALAN). Hit-count dibaca dari npz bila ada.
    Tanpa hit-count (masker lama) -> BUKAN-JALAN kosong dan semua sel non-jalan TAK-TAHU (perbaikan menjadi lebih agresif;
    dilaporkan lewat aux['has_hits']=False)."""
    shape = road.shape
    zeros = np.zeros(shape, np.uint8)

    def get(k):
        if npz is not None and k in npz.files:
            a = np.asarray(npz[k])
            if a.shape == shape:
                return a.astype(np.int32)
        return None

    cr, cn, ob, li = get("cam_road_hits"), get("cam_non_hits"), get("obstacle_hits"), get("lidar_hits")
    has_hits = cr is not None and cn is not None
    cr = cr if cr is not None else zeros.astype(np.int32)
    cn = cn if cn is not None else zeros.astype(np.int32)
    ob = ob if ob is not None else zeros.astype(np.int32)
    li = li if li is not None else zeros.astype(np.int32)

    tot = cr + cn
    frac = np.where(tot > 0, cr / np.maximum(tot, 1), 1.0)
    cam_non = (tot >= r1.min_hits) & (frac <= r1.cam_non_frac)
    obst = ob >= r1.obstacle_min_hits
    footprint = rasterize_boxes(boxes, x_range, y_range, res, r1.object_margin_m, shape) if (r1.use_objects and boxes) \
        else np.zeros(shape, bool)

    not_road = (cam_non | obst) & ~road & ~footprint        # rintangan = objek yg sudah ditangani Fase 3 -> TAK-TAHU, bukan tepi
    ev = np.zeros(shape, np.int8)
    ev[not_road] = NOT_ROAD
    ev[road] = ROAD
    observed = (li > 0) | (tot > 0) | (ob > 0)
    return ev, {"footprint": footprint, "observed": observed, "obstacle": ob >= r1.obstacle_min_hits,
                "has_hits": bool(has_hits), "cam_non": cam_non}


# ----------------------------------------------------------------------------
# R1b: perbaikan lubang
# ----------------------------------------------------------------------------
def adaptive_close(base: np.ndarray, allowed: np.ndarray, rng: np.ndarray, res: float, r1: R1Config) -> np.ndarray:
    """Closing dgn radius naik terhadap jarak; sel baru HANYA di `allowed` (= bukan BUKAN-JALAN)."""
    L = max(1, int(r1.close_levels))
    radii = np.linspace(r1.close_near_m, r1.close_far_m, L)
    edges = np.linspace(r1.close_near_range_m, r1.close_far_range_m, L)
    band = np.clip(np.searchsorted(edges, rng, side="right") - 1, 0, L - 1)
    out = base.copy()
    for k, rm in enumerate(radii):
        closed = _morph(base, cv2.MORPH_CLOSE, _cells(rm, res))
        out |= closed & (band == k) & allowed
    return out


def fill_enclosed_holes(m: np.ndarray, notm: np.ndarray, res: float, r1: R1Config) -> np.ndarray:
    """Isi lubang tertutup menurut BUKTI. Return mask baru. Lubang dgn bukti BUKAN-JALAN: hanya inti bukti (+radius) dijaga."""
    filled = ndi.binary_fill_holes(m)
    holes = filled & ~m
    if not holes.any():
        return m
    lbl, n = ndi.label(holes, structure=np.ones((3, 3), bool))
    out = m.copy()
    cell_area = res * res
    core_r = _cells(r1.island_core_radius_m, res)
    cross = ndi.generate_binary_structure(2, 1)
    for i, sl in enumerate(ndi.find_objects(lbl), 1):
        if sl is None:
            continue
        # jendela lokal (+1 sel) supaya murah
        s0 = slice(max(sl[0].start - core_r - 2, 0), sl[0].stop + core_r + 2)
        s1 = slice(max(sl[1].start - core_r - 2, 0), sl[1].stop + core_r + 2)
        comp = lbl[s0, s1] == i
        area = comp.sum() * cell_area
        if area < r1.island_min_area_m2:
            out[s0, s1] |= comp                                # noise kecil: selalu diisi
            continue
        if area > r1.fill_max_area_m2:
            continue                                          # terlalu besar utk ditebak: biarkan sbg lubang/pulau
        ring = ndi.binary_dilation(comp, cross) & ~comp
        pr = float((m[s0, s1] & ring).sum()) / max(int(ring.sum()), 1)
        if pr < r1.fill_min_perimeter_road:
            continue
        nots = notm[s0, s1] & comp
        if nots.any():
            core = (cv2.dilate(nots.astype(np.uint8), _disk(core_r)) > 0) & comp
            out[s0, s1] |= comp & ~core                        # sisa (TAK-TAHU) diisi, inti bukti tetap pulau
        else:
            out[s0, s1] |= comp
    return out


def remove_slivers(m: np.ndarray, res: float, r1: R1Config, rng: np.ndarray) -> np.ndarray:
    if r1.sliver_width_m <= 0 or not m.any():
        return m
    r = _cells(r1.sliver_width_m / 2.0, res)
    opened = _morph(m, cv2.MORPH_OPEN, r)
    keep = cv2.dilate(opened.astype(np.uint8), _disk(r)) > 0
    return m & (keep | (rng <= r1.sliver_protect_ego_m))


def _ego_component(m: np.ndarray, x_range, y_range, res) -> np.ndarray:
    lbl, n = ndi.label(m, structure=np.ones((3, 3), bool))
    if n == 0:
        return m
    eix, eiy = int(np.floor((0.0 - x_range[0]) / res)), int(np.floor((0.0 - y_range[0]) / res))
    if 0 <= eix < m.shape[0] and 0 <= eiy < m.shape[1] and lbl[eix, eiy] > 0:
        return lbl == lbl[eix, eiy]
    sizes = ndi.sum(m, lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def repair_mask(ev: np.ndarray, x_range, y_range, res, r1: R1Config, prior_road: Optional[np.ndarray] = None):
    """Return (mask_bersih, inferred). inferred = sel jalan hasil tebakan (bukan bukti jalan langsung)."""
    road = ev == ROAD
    notm = ev == NOT_ROAD
    rng = range_map(x_range, y_range, res)
    m = road.copy()
    if prior_road is not None:
        m |= prior_road & (ev == UNKNOWN)
    m = adaptive_close(m, ~notm, rng, res, r1)
    m = fill_enclosed_holes(m, notm, res, r1)
    m = _morph(m, cv2.MORPH_OPEN, _cells(r1.open_radius_m, res)) if r1.open_radius_m > 0 else m
    m = remove_slivers(m, res, r1, rng)
    if r1.keep_only_ego_component:
        m = _ego_component(m, x_range, y_range, res)
    m = fill_enclosed_holes(m, notm, res, r1)
    return m, (m & ~road)


# ----------------------------------------------------------------------------
# prior temporal kausal (opsional)
# ----------------------------------------------------------------------------
def warp_grid(a: np.ndarray, T_rel: np.ndarray, x_range, y_range, res) -> np.ndarray:
    """Warp array grid [ix,iy] dari frame ego SEBELUMNYA ke frame ego SEKARANG. T_rel = inv(T_prev) @ T_cur
    (memetakan koordinat ego-sekarang ke ego-sebelumnya)."""
    S = np.array([[res, 0, x_range[0] + 0.5 * res], [0, res, y_range[0] + 0.5 * res], [0, 0, 1.0]])
    M = np.linalg.inv(S) @ T_rel @ S                      # idx_sekarang -> idx_sebelumnya (ix,iy)
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])     # cv2 memakai (kolom=iy, baris=ix)
    Mcv = P @ M @ P
    nx, ny = a.shape
    return cv2.warpAffine(a.astype(np.float32), Mcv[:2], (ny, nx), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


class TemporalState:
    """Log-odds jalan di frame ego (di-warp tiap frame). KAUSAL: hanya memakai frame sebelumnya untuk prior frame ini."""

    def __init__(self, r1: R1Config, poses: np.ndarray, valid: np.ndarray):
        self.r1, self.poses, self.valid = r1, poses, valid
        self.L: Optional[np.ndarray] = None
        self.last_idx: Optional[int] = None

    def _pose_ok(self, idx: int) -> bool:
        return 0 <= idx < len(self.valid) and bool(self.valid[idx])

    def step(self, idx: int, ev: np.ndarray, x_range, y_range, res) -> Optional[np.ndarray]:
        r1 = self.r1
        prior = None
        gap = None if self.last_idx is None else idx - self.last_idx
        warped = None
        if (self.L is not None and gap is not None and 1 <= gap <= r1.t_max_gap_frames
                and self._pose_ok(idx) and self._pose_ok(self.last_idx)):
            T_rel = np.linalg.inv(self.poses[self.last_idx]) @ self.poses[idx]
            warped = warp_grid(self.L, T_rel, x_range, y_range, res) * (r1.t_decay ** gap)
            rng = range_map(x_range, y_range, res)
            prior = (warped > r1.t_thr) & (rng <= r1.t_max_range_m)
        gain = np.zeros(ev.shape, np.float32)
        gain[ev == ROAD] = r1.t_road_gain
        gain[ev == NOT_ROAD] = -r1.t_not_gain
        base = warped if warped is not None else np.zeros(ev.shape, np.float32)
        self.L = np.clip(base + gain, -4.0, 4.0).astype(np.float32)
        self.last_idx = idx
        return prior


# ----------------------------------------------------------------------------
# R1c: penghalusan tepi
# ----------------------------------------------------------------------------
def _ring_length(r: np.ndarray) -> float:
    d = np.diff(np.vstack([r, r[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def resample_closed(r: np.ndarray, step: float) -> np.ndarray:
    P = np.vstack([r, r[:1]])
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L <= 1e-9:
        return r.copy()
    n = max(int(round(L / step)), 8)
    u = np.linspace(0.0, L, n, endpoint=False)
    return np.stack([np.interp(u, s, P[:, 0]), np.interp(u, s, P[:, 1])], 1)


def smooth_ring(r: np.ndarray, r1: R1Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return (asli_rapat, halus_rapat) -- keduanya titik rapat bersampel seragam; halus dibatasi <= refine_max_disp_m."""
    dense = resample_closed(r, r1.resample_step_m)
    if len(dense) < 12 or _ring_length(r) < r1.min_ring_len_for_refine_m:
        return dense, dense.copy()
    ds = _ring_length(dense) / len(dense)
    sm = ndi.gaussian_filter1d(dense, sigma=max(r1.refine_sigma_m / ds, 0.5), axis=0, mode="wrap")
    d = sm - dense
    n = np.hypot(d[:, 0], d[:, 1])
    clamp = max(r1.refine_max_disp_m - r1.simplify_tol_m, 0.05)      # + Douglas-Peucker (<= simplify_tol_m) => total <= refine_max_disp_m
    scale = np.minimum(1.0, clamp / np.maximum(n, 1e-12))
    return dense, dense + d * scale[:, None]


def _simplify(r: np.ndarray, tol: float) -> np.ndarray:
    a = cv2.approxPolyDP(r.astype(np.float32).reshape(-1, 1, 2), tol, True)
    return a.reshape(-1, 2).astype(float)


def _signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ring_displacement(final: np.ndarray, original_dense: np.ndarray) -> np.ndarray:
    """Jarak tiap titik rapat `final` (diresampling 0,1 m) ke titik rapat kontur asli."""
    from scipy.spatial import cKDTree
    f = resample_closed(final, 0.1)
    return cKDTree(original_dense).query(f)[0]


def roughness(rings: List[np.ndarray], step: float = 0.2) -> float:
    """Total |sudut belok| per meter (rad/m) pada cincin yang di-resample seragam; makin kecil makin halus."""
    tot_turn, tot_len = 0.0, 0.0
    for r in rings:
        if len(r) < 3:
            continue
        L = _ring_length(np.asarray(r, float))
        if L < 2 * step:
            continue
        d = resample_closed(np.asarray(r, float), step)
        v = np.diff(np.vstack([d, d[:1]]), axis=0)
        th = np.arctan2(v[:, 1], v[:, 0])
        dth = np.angle(np.exp(1j * (np.roll(th, -1) - th)))
        tot_turn += float(np.abs(dth).sum())
        tot_len += L
    return tot_turn / tot_len if tot_len > 0 else 0.0


def _smooth_runs(q: List[str], ring: np.ndarray, min_limit_m: float, min_obs_m: float) -> List[str]:
    """Pembersihan tipe tepi (sirkular): potongan '1' pendek (< min_limit_m) di antara '0' -> '0'; potongan '0' pendek
    (< min_obs_m) -> '1'. Mencegah garis tepi/pemudaran yang tambal-sulam akibat bukti kamera yang bolong sesaat."""
    n = len(q)
    if n < 4 or len(set(q)) == 1:
        return q
    seg = np.hypot(*(np.roll(ring, -1, axis=0) - ring).T)             # panjang sisi i -> i+1
    start = next(i for i in range(n) if q[i] != q[i - 1])               # awal sebuah potongan
    order = [(start + k) % n for k in range(n)]
    runs, cur = [], [order[0]]
    for i in order[1:]:
        if q[i] == q[cur[-1]]:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    out = list(q)
    for r in runs:
        L = float(sum(seg[i] for i in r))
        t = q[r[0]]
        if t == "1" and L < min_limit_m:
            for i in r: out[i] = "0"
        elif t == "0" and L < min_obs_m:
            for i in r: out[i] = "1"
    return out


def edge_types(ring: np.ndarray, near_not: np.ndarray, x_range, y_range, res, r1: Optional[R1Config] = None) -> str:
    ix = np.clip(np.round((ring[:, 0] - x_range[0]) / res - 0.5).astype(int), 0, near_not.shape[0] - 1)
    iy = np.clip(np.round((ring[:, 1] - y_range[0]) / res - 0.5).astype(int), 0, near_not.shape[1] - 1)
    q = ["0" if near_not[a, b] else "1" for a, b in zip(ix, iy)]
    if r1 is not None:
        q = _smooth_runs(q, ring, r1.min_limit_run_m, r1.min_obs_run_m)
        q = _smooth_runs(q, ring, r1.min_limit_run_m, r1.min_obs_run_m)      # dua lintasan: koreksi satu bisa memicu yang lain
    return "".join(q)


def build_rings(raw_rings, ev: np.ndarray, x_range, y_range, res, r1: R1Config, min_poly_area: float):
    """raw_rings = [(luar, [lubang...])] kontur subpiksel (meter) -> (polys, holes[(pid, ring)], vq, stats)."""
    notm = ev == NOT_ROAD
    near_not = cv2.dilate(notm.astype(np.uint8), _disk(_cells(r1.edge_type_radius_m, res))) > 0
    polys, holes, vq = [], [], []
    disp_all, rough_raw = [], []
    xlo, xhi, ylo, yhi = x_range[0], x_range[1], y_range[0], y_range[1]

    def proc(raw):
        dense, sm = smooth_ring(raw, r1) if r1.refine else (resample_closed(raw, r1.resample_step_m),) * 2
        sm = np.stack([np.clip(sm[:, 0], xlo, xhi), np.clip(sm[:, 1], ylo, yhi)], 1)
        out = _simplify(sm, r1.simplify_tol_m)
        return dense, out

    for outer, hs in raw_rings:
        dense, o = proc(np.asarray(outer, float))
        if len(o) < 3 or abs(_signed_area(o)) < 1.0:
            continue
        if _signed_area(o) < 0:
            o = o[::-1]
        pid = len(polys)
        polys.append(o)
        vq.append(edge_types(o, near_not, x_range, y_range, res, r1))
        disp_all.append(ring_displacement(o, dense))
        rough_raw.append(dense)
        for h in hs:
            dh, hh = proc(np.asarray(h, float))
            if len(hh) < 3 or abs(_signed_area(hh)) < r1.island_min_area_m2:
                continue
            if _signed_area(hh) > 0:
                hh = hh[::-1]
            holes.append((pid, hh))
            disp_all.append(ring_displacement(hh, dh))
    disp = np.concatenate(disp_all) if disp_all else np.zeros(0)
    return polys, holes, vq, {"disp": disp, "dense_raw": rough_raw}


def inferred_rings(inferred: np.ndarray, x_range, y_range, res, r1: R1Config) -> List[np.ndarray]:
    """Kontur luar gumpalan sel `inferred` (meter, [x,y]) -- hanya utk memudarkan area tebakan di viewer (R2)."""
    if not inferred.any():
        return []
    cnts, _ = cv2.findContours(inferred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        p = c.reshape(-1, 2).astype(float)                               # (kolom=iy, baris=ix)
        if len(p) < 3:
            continue
        area = abs(cv2.contourArea(c.astype(np.float32))) * res * res
        # kontur piksel menyusuri PUSAT sel tepi -> area piksel kurang ~setengah sel di keliling; cukup utk ambang kasar
        if area < r1.inferred_min_area_m2:
            continue
        wx = x_range[0] + (p[:, 1] + 0.5) * res
        wy = y_range[0] + (p[:, 0] + 0.5) * res
        ring = _simplify(np.stack([wx, wy], 1), r1.inferred_simplify_m)
        if len(ring) >= 3:
            out.append((area, ring))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out[:r1.inferred_max_rings]]


# ----------------------------------------------------------------------------
# gabungan per-frame (dipanggil dari road_polygon_lib.process_frame bila cfg.r1 != None)
# ----------------------------------------------------------------------------
def process_frame_r1(cfg, idx: int, state: Optional[TemporalState] = None, return_internal: bool = False):
    import road_polygon_lib as rp
    r1: R1Config = cfg.r1
    fr = rp.load_mask_frame(cfg, idx)
    road, x_range, y_range, res = fr["mask"], fr["x_range"], fr["y_range"], fr["res"]
    with np.load(cfg.masks_dir / f"{idx:06d}.npz") as npz:
        obj_dir = r1.objects_dir if r1.objects_dir is not None else Path(cfg.root) / "fase3_deteksi" / "objects"
        boxes = load_object_boxes(obj_dir, idx) if r1.use_objects else []
        ev, aux = build_evidence(npz, road, boxes, x_range, y_range, res, r1)
    prior = state.step(idx, ev, x_range, y_range, res) if (state is not None and r1.temporal) else None
    sm, inferred = repair_mask(ev, x_range, y_range, res, r1, prior)

    if cfg.subpixel and rp._HAVE_CONTOURPY and sm.any():
        raw = rp._rings_subpixel(sm, x_range, y_range, res, cfg)
    elif sm.any():
        raw = rp._rings_pixel(sm, x_range, y_range, res, cfg)
    else:
        raw = []
    polys, holes, vq, rstats = build_rings(raw, ev, x_range, y_range, res, r1, cfg.min_polygon_area_m2)

    dec = cfg.coord_decimals
    edges = rp.ego_edge_distances(sm, x_range, y_range, res, cfg)
    road_type, arms = None, []
    if cfg.compute_road_type:
        arms = rp.ring_sample_arms(sm, x_range, y_range, res, cfg)
        road_type = {"n_arms": len(arms), "arms": arms, "label": rp.classify_road_type(arms, cfg)}
    n_final = max(int(sm.sum()), 1)
    result = {
        "idx": fr["idx"], "t": fr["t"],
        "road": {
            "polygons": [rp._ring_json(p, dec) for p in polys],
            "holes": [{"parent": pid, "ring": rp._ring_json(h, dec)} for pid, h in holes],
            "res_equiv_m": res,
            "vq": vq,
            "inf": round(float(inferred.sum()) / n_final, 4),
            "ir": [rp._ring_json(r, dec) for r in inferred_rings(inferred, x_range, y_range, res, r1)] if r1.inferred_rings else [],
        },
        "ego_edge_m": edges,
        "road_type": road_type,
    }
    if return_internal:
        return result, {"mask": road, "smooth": sm, "x_range": x_range, "y_range": y_range, "res": res, "arms": arms,
                        "ev": ev, "inferred": inferred, "aux": aux, "r1": rstats, "n_boxes": len(boxes)}
    return result


# ----------------------------------------------------------------------------
# diagnosis penyebab lubang (mode R0-ringan): kenapa mask_fused bolong?
# ----------------------------------------------------------------------------
def classify_holes(ev: np.ndarray, aux: dict, x_range, y_range, res, min_area_m2: float = 0.2,
                   far_range_m: float = 15.0) -> Dict[str, int]:
    """Lubang tertutup pada mask_fused (komponen latar yang terkurung jalan) -> hitung per penyebab.
    objek: >=30% di jejak objek | bukti_bukan_jalan: ada sel BUKAN-JALAN | tanpa_pengamatan: <30% sel teramati |
    jauh: pusat > far_range_m | ambigu: sisanya (teramati tapi label bukan jalan/tak cukup suara)."""
    road = ev == ROAD
    holes = ndi.binary_fill_holes(road) & ~road
    out = {"objek": 0, "bukti_bukan_jalan": 0, "tanpa_pengamatan": 0, "jauh": 0, "ambigu": 0, "total": 0}
    if not holes.any():
        return out
    lbl, n = ndi.label(holes, structure=np.ones((3, 3), bool))
    rng = range_map(x_range, y_range, res)
    for i in range(1, n + 1):
        comp = lbl == i
        a = comp.sum() * res * res
        if a < min_area_m2:
            continue
        out["total"] += 1
        if aux["footprint"][comp].mean() >= 0.3:
            out["objek"] += 1
        elif (ev[comp] == NOT_ROAD).any():
            out["bukti_bukan_jalan"] += 1
        elif aux["observed"][comp].mean() < 0.3:
            out["tanpa_pengamatan"] += 1 if rng[comp].mean() <= far_range_m else 0
            out["jauh"] += 0 if rng[comp].mean() <= far_range_m else 1
        else:
            out["ambigu"] += 1
    return out
