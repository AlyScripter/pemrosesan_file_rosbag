#!/usr/bin/env python3
r"""
fase3_detection.py
==================
FASE 3 -- Deteksi objek & ekstraksi titik objek dari LiDAR MENTAH (dijalankan SEBELUM segmentasi).

INPUT   : <out>/lidar/*.npy (mentah, tanpa label), rgb/, depth/, calib/calib_config.json, sync_table.csv
PROSES  : YOLO11l-seg (+ByteTrack) -> titik LiDAR di dalam mask (tinggi di atas BIDANG TANAH, bukan z mentah)
          -> buang tanah & dinding (RANSAC) -> DBSCAN, pilih klaster yang cocok depth ZED -> fit box 3D
          (L-shape, prior ukuran per kelas) -> pass offline per-track: sambung track, konfirmasi track, yaw dari
          permukaan mask+depth, kunci yaw per-track, konsistensi ukuran/jarak, isi celah, arah hidung.
          Status per objek: lidar_ok / depth_only; yaw berbukti ditandai (yaw_evidence / yaw_confident).
OUTPUT  : <out>/fase3_deteksi/objects/*.json, object_point_idx/*.npy, points/*.npy (N,7: x,y,z,intensity,ring,
          time,class_id; class_id 0 kecuali titik objek 4 Vehicle / 5 Person), heights/*.npy (tinggi di atas tanah)
          + fase3_deteksi/track_yaw_lock.json, ego_yaw.json, surface/ (cache), checkpoints/fase3_report.json

KODE ALGORITMA diambil VERBATIM dari notebook fase3_deteksi_objek_baru_final_v29.ipynb (lihat penanda [SEL n]).
Yang berubah hanya cara orkestrasi:
  * semua pass offline (8a-0 ... 8d) bekerja pada "disk virtual" di memori (JSON per frame disimpan sebagai string
    di RAM, perilaku baca/tulis identik dgn file) lalu ditulis SEKALI ke disk di akhir;
  * tanpa plot / viewer / sel diagnostik (diagnosa skala depth, A/B mask-vs-bbox, diagnose_selection, ringkasan plot);
  * checkpoint otomatis (fase3_report.json); DEPTH_SCALE dikunci 1.0 (keputusan notebook).
"""
import builtins
import csv
import gzip
import io
import math
import os
import shutil
import sys
import time
import json as _json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import cv2
from tqdm import tqdm

from common import Checker, Paths, Timer, resolve_by_name


# ----------------------------------------------------------------------------------------------------------------
# "Disk virtual" di memori untuk objects/*.json  (kode pass membaca/menulis lewat open()/json seperti biasa)
# ----------------------------------------------------------------------------------------------------------------
_builtin_open = builtins.open


class MemPath:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def exists(self):
        return self.name in self.store

    def read(self):
        if self.name not in self.store:
            raise FileNotFoundError(self.name)
        return self.store[self.name]

    def __str__(self):
        return f"<mem>/{self.name}"


class MemDir:
    def __init__(self):
        self.store = {}

    def __truediv__(self, name):
        return MemPath(self.store, str(name))

    def exists(self):
        return True

    def mkdir(self, *a, **k):
        pass

    def flush(self, real_dir):
        real_dir = Path(real_dir)
        real_dir.mkdir(parents=True, exist_ok=True)
        for old in real_dir.glob("*.json"):
            old.unlink()
        for name, txt in self.store.items():
            with _builtin_open(real_dir / name, "w", encoding="utf-8") as f:
                f.write(txt)


class _MemWriter:
    def __init__(self, p):
        self.p, self.buf = p, []

    def write(self, s):
        self.buf.append(s)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.p.store[self.p.name] = "".join(self.buf)
        return False


def open(file, mode="r", *a, **k):          # noqa: A001  (sengaja menimpa open untuk disk virtual)
    if isinstance(file, MemPath):
        if "w" in mode:
            return _MemWriter(file)
        return io.StringIO(file.read())
    return _builtin_open(file, mode, *a, **k)


class _JsonShim:
    """json biasa, kecuali dump() ke disk virtual: pakai encoder C (tanpa indent) -> jauh lebih cepat."""

    def __getattr__(self, n):
        return getattr(_json, n)

    def dump(self, obj, fp, **kw):
        if isinstance(fp, _MemWriter):
            fp.write(_json.dumps(obj))
        else:
            _json.dump(obj, fp, **kw)


json = _JsonShim()

# ----------------------------------------------------------------------------------------------------------------
# Konfigurasi (nilai dari sel config notebook) & global yang diisi configure()
# ----------------------------------------------------------------------------------------------------------------
VERBOSE_SKIP = False   # True = cetak tiap deteksi lemah yang dilewati (default mati: mengotori progress bar)
DEPTH_SCALE = 1.0                # dikunci (diagnosa notebook: bukan bias skala murni)
CANDIDATE_LABEL_FILTER = False
FUSION_STRONG_OVERRIDE = True
FUSION_STRONG_MIN_PTS = 20
FUSION_STRONG_MAX_MAD_M = 0.5
FUSION_STRONG_MIN_IOU = 0.60
FUSION_STRONG_MAX_SHIFT_M = 2.5
FUSION_STRONG_SHIFT_FRAC = 0.15

BASE_DIR = OUT_DIR = POINTS_DIR = HEIGHT_DIR = PTIDX_DIR = PREVIEW_DIR = SURF_DIR = HEADING_OVERRIDE_FILE = None
OBJ_DIR = MemDir()
T_lidar_to_cam = T_cam_to_lidar = K = D = None
IMG_W, IMG_H = 640, 360
rows, candidates = [], []
YOLO_MODEL = None


def resolve(base_dir, stored_path, subfolder):
    """Path di sync_table.csv bisa path Linux (WSL) atau Windows -> ambil nama file saja (tahan backslash)."""
    return resolve_by_name(base_dir, stored_path, subfolder)


# ====================================================================================================
# [SEL 3 NOTEBOOK] loader & kalibrasi (tanpa eksekusi)
# ====================================================================================================

def load_calib(path):
    with open(path) as f:
        c = json.load(f)
    T = np.array(c["T_lidar_to_cam"], dtype=np.float64)
    T_inv = np.linalg.inv(T)
    K = np.array(c["K"], dtype=np.float64)
    D = np.array(c.get("D", [0, 0, 0, 0, 0]), dtype=np.float64)
    w = c.get("IMG_W", 640)
    h = c.get("IMG_H", 360)
    return T, T_inv, K, D, w, h


def load_sync_rows(sync_csv):
    with open(sync_csv, newline="") as f:
        return list(csv.DictReader(f))


def load_lidar_points(row):
    '''LiDAR MENTAH -> (N,7) = x,y,z,intensity,ring,time,class_id(=0 semua). Baris non-finite dibuang.
    Output Fase 3 menyimpan cloud HASIL pembuangan ini, jadi Fase 4 baru membaca points/ kita, bukan file mentah.'''
    p = resolve(BASE_DIR, row.get("lidar_path"), "lidar")
    if p is None or not p.exists():
        return None
    a = np.load(p)
    assert a.ndim == 2 and a.shape[1] >= 3, f"{p.name}: bentuk {a.shape} tidak dikenali"
    out = np.zeros((len(a), 7), dtype=np.float32)
    k = min(a.shape[1], 6)
    out[:, :k] = a[:, :k]
    out = out[np.isfinite(out[:, :3]).all(1)]
    return out


# ====================================================================================================
# [SEL 4 NOTEBOOK] kelas & warna
# ====================================================================================================
CLASS_NAMES = {0: "Unknown", 1: "Road", 2: "Vegetation", 3: "Building", 4: "Vehicle", 5: "Person"}
CLASS_COLORS = {0: "#e6007e", 1: "#1f4e9c", 2: "#2e8b2e", 3: "#c97b2e", 4: "#00e5ff", 5: "#ff4d4d"}
GROUND_CLASS_ID = 1   # hanya dipakai default-arg fungsi lama; tidak ada titik berlabel Road di fase ini
OBJECT_CLASS_ID_NOTE = "class_id titik objek: 4 Vehicle / 5 Person; semua titik lain 0 (belum disegmentasi)"


# ====================================================================================================
# [SEL 5 NOTEBOOK] YOLO-seg + ByteTrack
# ====================================================================================================
try:
    from ultralytics import YOLO
except ImportError:      # diuji tanpa ultralytics (detektor di-stub)
    YOLO = None
import cv2

# Fase 3 baru: YOLO-SEG (instance mask). Mask dipakai utk memilih titik LiDAR milik objek; bbox tetap dipakai utk lifting 3D.
YOLO_MODEL_NAME = "yolo11l-seg.pt"

# kelas COCO -> kelas lokal. Sepeda & motor ikut dideteksi (permintaan); truck/bus dipetakan ke Car seperti sebelumnya.
COCO_TO_LOCAL = {"car": "Car", "truck": "Car", "bus": "Car", "person": "Person", "bicycle": "Bicycle", "motorcycle": "Motorcycle"}
COCO_CLASS_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
CONF_THRESH_BY_CLASS = {"person": 0.70, "car": 0.45, "truck": 0.55, "bus": 0.55, "bicycle": 0.60, "motorcycle": 0.60}
PERSON_ENABLED = True    # Person tetap aktif di semua rute; hidran/tiang ditolak oleh gerbang tinggi fisik + konfirmasi track
CONF_THRESH_DEFAULT  = 0.40
OBJECT_CLASS_ID = {"Car": 4, "Motorcycle": 4, "Bicycle": 4, "Person": 5}   # 4 Vehicle (roda 2/4), 5 Person


def track_objects_2d(rgb_path, model=None, conf_by_class=CONF_THRESH_BY_CLASS):
    '''YOLO-seg + ByteTrack. Tiap deteksi: class, confidence, bbox2d, track_id (None = belum dikonfirmasi tracker),
    mask (bool H x W, ukuran gambar asli). retina_masks=True supaya mask sudah di resolusi gambar.'''
    model = model or YOLO_MODEL
    active = [k for k in conf_by_class if PERSON_ENABLED or k != "person"]
    conf_min = min(conf_by_class[k] for k in active)
    result = model.track(str(rgb_path), persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_min,
                         agnostic_nms=True, classes=[COCO_CLASS_IDS[k] for k in active], retina_masks=True)[0]
    names = result.names
    boxes = result.boxes
    ids = boxes.id
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    dets = []
    for i in range(len(boxes)):
        cls_name = names[int(boxes.cls[i])]
        if cls_name not in COCO_TO_LOCAL:
            continue
        conf = float(boxes.conf[i])
        if conf < conf_by_class.get(cls_name, CONF_THRESH_DEFAULT):
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        mk = None
        if masks is not None and i < len(masks):
            mk = masks[i] > 0.5
            if mk.shape != (IMG_H, IMG_W):
                mk = cv2.resize(mk.astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 0
        dets.append({
            "class":      COCO_TO_LOCAL[cls_name],
            "confidence": conf,
            "bbox2d":     [x1, y1, x2, y2],
            "track_id":   int(ids[i]) if ids is not None else None,
            "mask":       mk,
        })
    return dets


def reset_tracker():
    '''Panggil sebelum batch resmi supaya ID tracking mulai bersih.'''
    global YOLO_MODEL
    YOLO_MODEL = YOLO(YOLO_MODEL_NAME)
    if "reset_ground_state" in globals():
        reset_ground_state()
    print("Tracker direset (model instance baru).")


# ====================================================================================================
# [SEL 7 NOTEBOOK] Bagian 4: deteksi, bidang tanah, lift 3D, fusi LiDAR (Fase 4 revisi 23 + saklar Fase 3)
# ====================================================================================================
import cv2  # Revisi ke-12: dipakai utk cv2.minAreaRect (fit bentuk & orientasi dari titik LiDAR
            # yang cukup banyak) -- belum ada di Bagian 0, ditambahkan di sini spy Bagian 0
            # (imports) tidak perlu diubah. Kalau belum terpasang: %pip install -q opencv-python

CLASS_CANONICAL_SIZE_LWH = {"Car": (4.2, 1.8, 1.5), "Person": (0.5, 0.5, 1.7),
                            "Motorcycle": (2.0, 0.8, 1.3), "Bicycle": (1.8, 0.6, 1.2)}   # Fase 3 baru: sepeda & motor ikut dideteksi

# =============================================================================
# REVISI KE-12 -- REDESAIN INTI (bukan tambal lagi di atas 11 revisi sebelumnya)
# =============================================================================
# RINGKASAN TEMUAN (dari uji langsung ke data mentah frame 1090: 001090.npy +
# calib_config.json + rgb_1765606399_778865.png -- lihat laporan lengkap yang menyertai
# revisi ini):
#
# 1) AKAR MASALAH #1 -- kontaminasi titik latar (BUKAN cuma "salah gate", tapi memang
#    fisik kejadiannya begitu): utk bbox 2D mobil terdekat di frame 1090, filter LAMA
#    (tol_above tinggi = 1.5, plafon jadi z_ref+3.0m) meloloskan >100 titik KANOPI POHON
#    & TEMBOK yang PERSIS berada di kedalaman kamera yang sama dgn mobil (14-20m), pada
#    ketinggian 1.8m-4.6m di atas tanah -- membanjiri & mengalahkan segelintir titik mobil
#    ASLI (yang ternyata cuma 3 titik, kelas "Unknown", tinggi 1.47-1.70m -- pas atap
#    mobil). PCA/bin-histogram apa pun yg dijalankan di atas campuran ini pasti salah,
#    krn secara STATISTIK titik latar itu memang lebih banyak & lebih "meyakinkan".
# 2) AKAR MASALAH #2: bahkan SETELAH difilter benar, titik LiDAR asli yang tersisa per
#    mobil di jarak 15-30m pada dataset ini SANGAT SEDIKIT (0-8 titik) -- bukan krn bug,
#    tapi kepadatan sudut LiDAR yang dipakai memang serendah itu di jarak segitu. PCA
#    (Revisi ke-7/9) pada <10 titik SANGAT TIDAK STABIL -- 2 mobil yang sebenarnya
#    terparkir SEJAJAR (harusnya arahnya mirip) dilaporkan yaw 64 derajat & 137 derajat
#    (beda >70 derajat!) oleh pipeline lama, PERSIS krn ini.
# 3) Konsekuensi langsung: ketergantungan pd "jumlah titik LiDAR + PCA" sbg sumber UTAMA
#    orientasi/bentuk (Revisi ke-7 s.d. ke-11) itu SALAH ARSITEKTUR utk dataset ini, bukan
#    soal kurang tuning ambang.
#
# PERBAIKAN (menggantikan _density_cluster_mask, select_object_points_mask,
# estimate_yaw_from_points, get_object_yaw, fit_oriented_box_and_filter LAMA):
#   A. Filter tinggi jauh lebih KETAT (akar masalah #1) -- 1 baris ini menghilangkan
#      >90% kontaminasi tanpa mesin tambahan apa pun.
#   B. Seleksi titik jadi SATU alur linear (bukan iterasi/gate berlapis): tinggi -> jendela
#      kedalaman -> prioritas 'Unknown' -> klaster kepadatan terbesar. Lebih murah & lebih
#      mudah dijelaskan drpd bin-histogram-mode + cross-check 3x tolerance versi lama.
#   C. ORIENTASI: PCA titik CUMA dipakai kalau titik BANYAK & bentuknya memang memanjang
#      (>=12 titik, elongasi >=1.6x) -- baru bisa dipercaya scr statistik. Kalau tidak,
#      dipakai metode BARU: "pencarian orientasi lewat proyeksi" -- box ukuran kanonik
#      disintesis pd jarak yang SUDAH DIPERCAYA dari sensor (LiDAR kalau ada, kalau tidak
#      depth map), diputar ke berbagai sudut kandidat, tiap sudut DIPROYEKSIKAN ke gambar
#      (persis konsep "3D Viewport -> proyeksi 2D" dari seminar rujukan), dan sudut yang
#      kotak-pembungkus proyeksinya PALING MIRIP (IoU) dgn bbox 2D asli YOLO itulah yang
#      dipakai. Ini TIDAK butuh titik LiDAR sama sekali utk cari arah -- jadi tetap jalan
#      walau LiDAR nyaris kosong (kasus paling umum di dataset ini).
#   D. PENTING -- jarak (depth) TIDAK ikut "dicari bebas" bersama sudut. Percobaan pertama
#      pendekatan ini (membiarkan jarak digeser 0.4x-2x) TERBUKTI memicu solusi
#      "degenerate": box didekatkan drastis SEKALIGUS diputar nyaris sejajar sinar
#      pandang bisa kebetulan menghasilkan siluet 2D yang SAMA SEMPIT dgn box yang
#      sebenarnya JAUH & menyamping -- ambiguitas klasik monokular (persis yang
#      dijelaskan di seminar rujukan soal kenapa distance estimation monokular itu
#      ill-posed). Jarak WAJIB dijangkar ke pengukuran sensor; pencarian proyeksi cuma
#      boleh menggeser TIPIS (+-15%) utk menghaluskan, dan HANYA MENCARI SUDUT.
#   E. Ukuran box: tetap kanonik per kelas (proporsi realistis & KONSISTEN antar frame,
#      sesuai targetmu), cuma di-override oleh ukuran hasil fit titik kalau titiknya
#      benar-benar banyak (>=12) DAN hasilnya masuk akal (0.6x-1.6x kanonik).
#   F. Semua bagian LAIN (SPLIT objek gabungan, COASTING, status gerak/diam, smoothing
#      EMA per track, filter plausibilitas tinggi, depth_fallback) TIDAK diubah -- itu
#      levelnya sudah benar, gak ada bukti bug di situ, dan mengubahnya tanpa perlu cuma
#      menambah risiko regresi (lihat instruksi: jangan ubah yang tidak perlu).
MIN_LIDAR_PTS_IN_BBOX = 2        # instansi filter tahap seleksi titik (bukan lagi 5) --
                                  # 2 titik SUDAH cukup dipakai sbg BUKTI POSISI (bukan
                                  # orientasi -- itu dijaga terpisah oleh MIN_PTS_FOR_SHAPE_FIT)

# --- Revisi ke-6 (TETAP dipertahankan): anti-false-positive khusus depth_fallback ---
MIN_CONF_FOR_DEPTH_FALLBACK = {"Car": 0.65, "Person": 0.75}
DEFAULT_MIN_CONF_FOR_DEPTH_FALLBACK = 0.70
MIN_BBOX_AREA_PX_FOR_DEPTH_FALLBACK = 1200
# Revisi ke-13: TURUN dari 3000 (Revisi ke-6). Ambang 3000 awalnya dipasang utk buang
# false-positive kecil di pojok gambar, TAPI itu jg membuang objek JAUH & KECIL yang
# NYATA & yakin terdeteksi (mis. frame 1090: mobil jauh, bbox 61x30px=1830px, conf=0.85,
# depth map valid setelah dilasi -- tapi dulu ke-skip krn cuma soal luas piksel, BUKAN krn
# datanya gak valid). 1200px (kira2 35x35px) masih cukup ketat utk nolak noise super kecil,
# tapi gak lagi membuang objek jauh yg genuin. Kalau masih ada false-positive kecil yg lolos,
# naikkan lagi nilainya di sini (bukan di Bagian 4 manapun lain -- 1 titik tuning).

# --- Revisi ke-12: filter tinggi (perbaikan #1, AKAR MASALAH UTAMA) ---
OBJECT_HEIGHT_TOL_BELOW_M = 0.3   # toleransi kecil di bawah tanah (noise z_ref)
OBJECT_HEIGHT_TOL_ABOVE_M = 0.5   # <-- TURUN DARI 1.5 (versi Revisi ke-11). Plafon Car jadi
                                  # z_ref + 1.5 + 0.5 = ~2.0m di atas tanah, BUKAN 3.0m lagi.
                                  # Inilah 1 baris yang paling banyak berkontribusi membuang
                                  # kontaminasi kanopi pohon/tembok (lihat laporan analisis).

# --- Revisi ke-12: seleksi titik & klaster (ganti select_object_points_mask lama) ---
DEPTH_WINDOW_FACTOR = 1.3         # jendela kedalaman = +- factor * max(L,W) sekitar ref_depth
DENSITY_CLUSTER_EPS_M = 1.3       # radius union-find utk klaster kepadatan titik kandidat
UNKNOWN_CLASS_ID = 0

# --- Revisi ke-12: kapan boleh percaya BENTUK (L/W/yaw) dari titik LiDAR langsung ---
MIN_PTS_FOR_SHAPE_FIT = 12        # baru percaya minAreaRect kalau titik CUKUP BANYAK
MIN_ECC_FOR_SHAPE_FIT = 1.6       # rasio sisi panjang/pendek min spy dianggap "memanjang"
SIZE_CLIP_MIN_FACTOR = 0.6        # klem ukuran hasil fit thd ukuran kanonik (jangan sampai
SIZE_CLIP_MAX_FACTOR = 1.6        # box "meledak"/"menciut" krn beberapa titik nyasar)

# --- Revisi ke-12: pencarian ORIENTASI lewat proyeksi (ganti PCA-pada-titik-sedikit) ---
YAW_SEARCH_STEP_DEG = 5
YAW_SEARCH_RANGE_DEG = 90                              # box simetris -> cukup +-90 (mod pi)
DEPTH_SEARCH_FACTORS_NARROW = np.arange(0.85, 1.151, 0.03)  # jarak CUMA digeser +-15%
MIN_IOU_FOR_YAW_SEARCH = 0.15     # kalau kecocokan terbaik masih jelek, JANGAN dipakai --
                                  # lebih jujur fallback ke default/riwayat track drpd
                                  # memaksakan sudut yg sebenarnya gak match sama sekali

YAW_EMA_ALPHA = 0.3               # smoothing sudut antar-frame per track_id (TETAP, Revisi ke-7)
POS_EMA_ALPHA = 0.2               # smoothing posisi antar-frame per track_id (TETAP, Revisi ke-9)

# --- Revisi ke-8a (TETAP): pisahkan objek yg "menggabung" dlm 1 bbox 2D ---
CLUSTER_EPS_M = {"Car": 1.2, "Person": 0.8}
DEFAULT_CLUSTER_EPS_M = 1.0
MIN_CLUSTER_GAP_FRACTION = 0.75

# --- Revisi ke-8c/ke-9 (TETAP): status gerak/diam + smoothing posisi ---
REL_SPEED_STATIC_THRESHOLD_MPS = 0.3
SPEED_WINDOW_S = 1.0
BOX_FIT_MARGIN = 0.35   # tidak lagi dipakai fit_oriented_box_and_filter (dihapus), disimpan
                        # cuma spy referensi versi lama di markdown atas tetap konsisten

# --- Revisi ke-10 (TETAP): jarak konsistensi depth kamera vs LiDAR, ambang confidence, coasting ---
DEPTH_CONSISTENCY_TOL_M = {"Car": 2.5, "Person": 1.5}
DEFAULT_DEPTH_TOL_M = 2.0
MAX_COAST_FRAMES = 5


def project_lidar_to_image(points_xyz, T, K):
    pts_h = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])
    cam_pts = (T @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    valid = z > 0.1
    idx_valid = np.nonzero(valid)[0]
    cam_pts_v, z_v = cam_pts[valid], z[valid]
    uv = (K @ cam_pts_v.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, z_v, idx_valid


def backproject_pixel_to_lidar(u, v, depth_val, K, T_cam_to_lidar):
    """(TETAP dari Revisi ke-10.) Balik-proyeksikan 1 piksel + jarak sepanjang sumbu-depan
    kamera jadi 1 titik 3D di frame LiDAR -- dipakai spy posisi LATERAL objek SELALU
    konsisten dgn tempat objek itu terlihat di gambar (RGB jadi validator posisi)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u - cx) * depth_val / fx
    y_cam = (v - cy) * depth_val / fy
    pt_cam = np.array([x_cam, y_cam, depth_val, 1.0])
    return (T_cam_to_lidar @ pt_cam)[:3]


def robust_camera_depth(xyz_lidar, T_lidar_to_cam):
    pts_h = np.hstack([xyz_lidar, np.ones((len(xyz_lidar), 1))])
    cam_pts = (T_lidar_to_cam @ pts_h.T).T[:, :3]
    return float(np.median(cam_pts[:, 2]))


def bbox_depth_fallback(depth, bbox2d, K, T_cam_to_lidar):
    """Revisi ke-13: sekarang pakai get_reference_depth_with_dilation (didefinisikan di bawah)
    -- kalau bbox asli 100% kosong di peta depth, coba perlebar area sampling dulu sebelum
    nyerah (lihat catatan lengkap di atas get_reference_depth_with_dilation)."""
    x1, y1, x2, y2 = bbox2d
    z_med, factor_used = get_reference_depth_with_dilation(depth, bbox2d)
    if z_med is None:
        return None, None
    u_c, v_c = (x1 + x2) / 2, (y1 + y2) / 2
    return backproject_pixel_to_lidar(u_c, v_c, z_med, K, T_cam_to_lidar), factor_used


def get_reference_depth(depth, bbox2d):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox2d]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, depth.shape[1]), min(y2, depth.shape[0])
    patch = depth[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0.1) & (patch < 60.0)
    if valid.sum() == 0:
        return None
    return float(np.median(patch[valid]))


# --- Revisi ke-13: pelebaran bbox bertahap kalau patch depth kosong sama sekali ---
# TEMUAN NYATA (diuji ke depth_1765606399_779261.npy, frame 1090, mobil jauh): bbox 2D
# aslinya (61x30px) 100% inf/nan di peta depth -- BUKAN cuma sedikit noise, tapi benar2
# kosong (mobil gelap/kurang tekstur, kemungkinan di luar jangkauan percaya diri estimator
# depth). Padahal YOLO yakin ada objek di situ. Sebelum nyerah total ke fallback temporal
# (Bagian 8b, perlu frame lain), coba dulu perbesar area sampling bbox scr bertahap dgn
# PUSAT TETAP SAMA -- kalau ketemu piksel valid di cincin terdekat, biasanya masih cukup
# representatif drpd tidak dapat apa2 sama sekali. BERHENTI di cincin PERTAMA yang berhasil
# (jangan terus melebar) supaya tidak ketarik depth objek lain di sekitarnya.
DEPTH_DILATION_MAX_FACTOR = 1.5     # bbox boleh membesar sampai 2.5x luas aslinya (1+1.5)
DEPTH_DILATION_STEPS = 6


def get_reference_depth_with_dilation(depth, bbox2d, max_factor=DEPTH_DILATION_MAX_FACTOR,
                                       steps=DEPTH_DILATION_STEPS):
    """Return (depth_val, dilation_factor_terpakai) atau (None, None) kalau tetap gagal
    sampai batas pelebaran. dilation_factor=1.0 berarti bbox asli (tidak perlu diperlebar)."""
    x1, y1, x2, y2 = bbox2d
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    for step in range(steps + 1):
        factor = 1.0 + max_factor * step / steps
        hw, hh = w * factor / 2.0, h * factor / 2.0
        bx = (cx - hw, cy - hh, cx + hw, cy + hh)
        val = get_reference_depth(depth, bx)
        if val is not None:
            return val, factor
    return None, None


def cluster_points_xy(xyz, eps_m, min_pts):
    """Union-Find sederhana di bidang x-y (TETAP dari Revisi ke-8a, dipakai ulang di
    seleksi titik yang baru & split_merged_clusters)."""
    n = len(xyz)
    if n == 0:
        return []
    xy = xyz[:, :2]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n - 1):
        d = np.linalg.norm(xy[i + 1:] - xy[i], axis=1)
        for j in np.nonzero(d <= eps_m)[0]:
            union(i, i + 1 + int(j))

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    clusters = [np.array(v) for v in groups.values() if len(v) >= min_pts]
    clusters.sort(key=len, reverse=True)
    return clusters


def split_merged_clusters(pts_in_bbox, cls_name):
    """(TETAP dari Revisi ke-8a -- tidak ada bukti bug di sini, tidak diubah.)"""
    eps = CLUSTER_EPS_M.get(cls_name, DEFAULT_CLUSTER_EPS_M)
    raw_clusters = cluster_points_xy(pts_in_bbox[:, :3], eps_m=eps, min_pts=MIN_LIDAR_PTS_IN_BBOX)
    if len(raw_clusters) <= 1:
        return [np.arange(len(pts_in_bbox))]
    length_std, width_std, _ = CLASS_CANONICAL_SIZE_LWH.get(cls_name, (2.0, 2.0, 1.5))
    gap_threshold = MIN_CLUSTER_GAP_FRACTION * max(length_std, width_std)
    accepted = [raw_clusters[0]]
    for cand in raw_clusters[1:]:
        cand_centroid = pts_in_bbox[cand, :2].mean(axis=0)
        dists = [np.linalg.norm(cand_centroid - pts_in_bbox[a, :2].mean(axis=0)) for a in accepted]
        nearest_i = int(np.argmin(dists))
        if dists[nearest_i] >= gap_threshold:
            accepted.append(cand)
        else:
            accepted[nearest_i] = np.concatenate([accepted[nearest_i], cand])
    if len(accepted) <= 1:
        return [np.arange(len(pts_in_bbox))]
    accepted.sort(key=lambda idxs: pts_in_bbox[idxs, 1].mean())
    return accepted


def adjust_center_for_near_face(center_xyz, yaw_rad, length_m, width_m):
    """(TETAP dari Revisi ke-8b -- ide bagus, tidak ada bukti bug di sini.) LiDAR/depth cuma
    'melihat' permukaan objek yang menghadap ego -- di sini center digeser MENJAUHI ego
    sejauh setengah dimensi box yang paling searah garis pandang."""
    x, y, z = center_xyz
    dist_xy = float(np.hypot(x, y))
    if dist_xy < 1e-6:
        return np.array([x, y, z])
    direction = np.array([x, y]) / dist_xy
    length_axis = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
    width_axis = np.array([-np.sin(yaw_rad), np.cos(yaw_rad)])
    align_length = abs(float(np.dot(direction, length_axis)))
    align_width = abs(float(np.dot(direction, width_axis)))
    shift_dim = length_m if align_length >= align_width else width_m
    new_xy = np.array([x, y]) + direction * (shift_dim / 2.0)
    return np.array([new_xy[0], new_xy[1], z])


def circular_ema_mod_pi(prev_yaw, new_yaw, alpha):
    prev_vec = np.array([np.cos(2 * prev_yaw), np.sin(2 * prev_yaw)])
    new_vec = np.array([np.cos(2 * new_yaw), np.sin(2 * new_yaw)])
    blended = alpha * new_vec + (1 - alpha) * prev_vec
    return (np.arctan2(blended[1], blended[0]) / 2) % np.pi


def smooth_track_position(track_key, new_center_xyz, history, alpha=POS_EMA_ALPHA):
    new_center_xyz = np.asarray(new_center_xyz, dtype=float)
    if track_key is None:
        return new_center_xyz
    prev = history.get(track_key)
    if prev is None:
        history[track_key] = new_center_xyz.copy()
        return new_center_xyz
    smoothed = alpha * new_center_xyz + (1 - alpha) * prev
    history[track_key] = smoothed
    return smoothed


def compute_frame_ground_z_ref(points_full7, ground_class_id):
    ground_pts = points_full7[points_full7[:, 6].astype(int) == ground_class_id]
    if len(ground_pts) == 0:
        return 0.0
    return float(np.median(ground_pts[:, 2]))


def filter_plausible_height_mask(z_lidar_raw, z_ref, height_m,
                                  tol_below=OBJECT_HEIGHT_TOL_BELOW_M, tol_above=OBJECT_HEIGHT_TOL_ABOVE_M):
    h = z_lidar_raw - z_ref
    return (h >= -tol_below) & (h <= height_m + tol_above)


# =============================================================================
# Revisi ke-12 -- fungsi INTI baru: seleksi titik & estimasi bentuk/orientasi
# =============================================================================
def select_object_points_v2(uv, z_cam, idx_valid, points_full7, bbox2d, ref_depth,
                             length_std, width_std, height_std, frame_z_ref):
    """Ganti select_object_points_mask + _density_cluster_mask lama. SATU alur linear:
      1) titik di dalam bbox 2D
      2) filter tinggi KETAT (perbaikan #1, lihat catatan revisi di atas)
      3) jendela kedalaman +- DEPTH_WINDOW_FACTOR*max(L,W) sekitar ref_depth (kalau ada)
      4) prioritas titik 'Unknown', fallback ke semua kelas yang lolos di atas
      5) klaster kepadatan (union-find) -- ambil klaster TERBESAR
    Return idx (indeks ke points_full7) dari titik yang lolos (bisa kosong)."""
    x1, y1, x2, y2 = bbox2d
    inb = (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)
    sel = idx_valid[inb]
    zc = z_cam[inb]
    if len(sel) == 0:
        return np.zeros(0, dtype=int)

    h = points_full7[sel, 2] - frame_z_ref
    hmask = (h >= -OBJECT_HEIGHT_TOL_BELOW_M) & (h <= height_std + OBJECT_HEIGHT_TOL_ABOVE_M)
    sel, zc = sel[hmask], zc[hmask]
    if len(sel) == 0:
        return np.zeros(0, dtype=int)

    if ref_depth is not None:
        margin = DEPTH_WINDOW_FACTOR * max(length_std, width_std)
        dmask = np.abs(zc - ref_depth) <= margin
        sel, zc = sel[dmask], zc[dmask]
        if len(sel) == 0:
            return np.zeros(0, dtype=int)

    def best_cluster(idx_pool):
        if len(idx_pool) == 0:
            return idx_pool
        xyz_pool = points_full7[idx_pool, :3]
        clusters = cluster_points_xy(xyz_pool, eps_m=DENSITY_CLUSTER_EPS_M, min_pts=1)
        if not clusters:
            return idx_pool
        return idx_pool[clusters[0]]

    cls_sel = points_full7[sel, 6].astype(int)
    unk = sel[cls_sel == UNKNOWN_CLASS_ID]
    if len(unk) >= MIN_LIDAR_PTS_IN_BBOX:
        out = best_cluster(unk)
        if len(out) >= MIN_LIDAR_PTS_IN_BBOX:
            return out
    return best_cluster(sel)


def _compute_box_corners_3d_local(center_xyz, yaw_rad, length_m, width_m, height_m):
    """Salinan kecil compute_box_corners_3d (didefinisikan lengkap lagi di Bagian 6 utk
    menggambar) -- dibutuhkan DI SINI juga karena Bagian 4 jalan LEBIH DULU drpd Bagian 6,
    dan pencarian orientasi baru (di bawah) perlu memproyeksikan box kandidat."""
    cx, cy, cz = center_xyz
    l2, w2, h2 = length_m / 2.0, width_m / 2.0, height_m / 2.0
    local = np.array([
        [ l2,  w2, -h2], [ l2, -w2, -h2], [-l2, -w2, -h2], [-l2,  w2, -h2],
        [ l2,  w2,  h2], [ l2, -w2,  h2], [-l2, -w2,  h2], [-l2,  w2,  h2],
    ])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.array([cx, cy, cz])


def _project_box_corners_local(corners_xyz, T_lidar_to_cam, K):
    pts_h = np.hstack([corners_xyz, np.ones((8, 1))])
    cam_pts = (T_lidar_to_cam @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    if np.any(z <= 0.1):
        return None
    uv = (K @ cam_pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def _bbox_iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0


def search_yaw_by_bbox_projection(u_c, v_c, depth_guess, L, W, H, bbox2d, T, K, T_cam_to_lidar,
                                   z_clip=None, step_deg=YAW_SEARCH_STEP_DEG,
                                   rng_deg=YAW_SEARCH_RANGE_DEG,
                                   depth_factors=DEPTH_SEARCH_FACTORS_NARROW):
    """INTI BARU Revisi ke-12 utk kasus titik LiDAR terlalu sedikit/kosong (KASUS PALING
    SERING di dataset ini -- lihat catatan revisi di atas sel). Alih-alih menebak yaw dari
    statistik beberapa titik yang gampang meleset jauh (PCA versi lama), sudut dicari dgn
    men-SINTESIS box kanonik di sepanjang sinar pandang piksel (u_c,v_c) pd jarak yang SUDAH
    DIPERCAYA dari sensor (`depth_guess`, cuma digeser tipis +-15% lewat `depth_factors`),
    MEMPROYEKSIKANNYA ke gambar (persis konsep '3D Viewport -> proyeksi 2D' dari seminar
    rujukan), lalu memilih sudut yang kotak-pembungkus proyeksinya PALING MIRIP (IoU) dgn
    bbox 2D asli YOLO. PENTING: jarak SENGAJA tidak dicari bebas jauh -- itu terbukti memicu
    solusi 'degenerate' (box didekatkan drastis + diputar nyaris sejajar sinar pandang bisa
    kebetulan menyamai siluet box yg sebenarnya jauh & menyamping -- ambiguitas klasik
    monokular). Kalau sensor sendiri sudah meleset jauh, hasil metode ini ikut meleset --
    itu jujur & sesuai batas kemampuan geometri klasik tanpa jaringan terlatih."""
    best = None
    for factor in depth_factors:
        depth_val = depth_guess * factor
        if depth_val <= 0.5:
            continue
        center_xyz = backproject_pixel_to_lidar(u_c, v_c, depth_val, K, T_cam_to_lidar)
        if z_clip is not None and not (z_clip[0] <= center_xyz[2] <= z_clip[1]):
            continue
        for yaw_deg in range(-rng_deg, rng_deg + 1, step_deg):
            yaw = np.radians(yaw_deg)
            corners = _compute_box_corners_3d_local(center_xyz, yaw, L, W, H)
            uv = _project_box_corners_local(corners, T, K)
            if uv is None:
                continue
            b = (float(uv[:, 0].min()), float(uv[:, 1].min()),
                 float(uv[:, 0].max()), float(uv[:, 1].max()))
            score = _bbox_iou(b, bbox2d)
            if best is None or score > best[0]:
                best = (score, center_xyz.copy(), yaw % np.pi)
    return best


def fit_shape_pose(xyz_valid, canon_L, canon_W, canon_H, bbox2d, T_lidar_to_cam, K, T_cam_to_lidar,
                    u_c, v_c, depth_guess, center_hint, yaw_hint, frame_z_ref):
    """Ganti fit_oriented_box_and_filter + estimate_yaw_from_points/get_object_yaw lama.
    xyz_valid = titik yang SUDAH lolos select_object_points_v2 (bisa sangat sedikit/kosong).
    Return: center_xyz, yaw_rad, length_m, width_m, height_m, yaw_source(str, utk diagnostik)"""
    n = len(xyz_valid)
    length_m, width_m, height_m = canon_L, canon_W, canon_H
    center_xyz = np.asarray(center_hint, dtype=float).copy()
    # Klem (BUKAN timpa) sumbu-Z ke rentang wajar utk objek yg menapak tanah -- mencegah
    # nilai "melayang" (mis. >3m di atas tanah) saat backprojeksi 1 piksel jauh nyaris tanpa
    # dukungan LiDAR (lihat kasus mobil jauh di laporan analisis: z mentah = 6.5m).
    z_lo, z_hi = frame_z_ref - 0.5, frame_z_ref + canon_H
    center_xyz[2] = float(np.clip(center_xyz[2], z_lo, z_hi))

    yaw_source = "default"
    yaw_rad = yaw_hint if yaw_hint is not None else 0.0

    if n >= MIN_PTS_FOR_SHAPE_FIT:
        rect = cv2.minAreaRect(xyz_valid[:, :2].astype(np.float32))
        (rcx, rcy), (w, h), ang = rect
        long_side, short_side = (w, h) if w >= h else (h, w)
        if short_side > 1e-6 and (long_side / short_side) >= MIN_ECC_FOR_SHAPE_FIT:
            yaw_rad = np.radians(ang if w >= h else ang + 90.0) % np.pi
            yaw_source = "points"
            length_m = float(np.clip(long_side, SIZE_CLIP_MIN_FACTOR * canon_L,
                                      SIZE_CLIP_MAX_FACTOR * canon_L))
            width_m = float(np.clip(short_side, SIZE_CLIP_MIN_FACTOR * canon_W,
                                     SIZE_CLIP_MAX_FACTOR * canon_W))
            center_xyz[0], center_xyz[1] = rcx, rcy

    if yaw_source == "default":
        z_clip = (frame_z_ref - 0.5, frame_z_ref + canon_H)
        best = search_yaw_by_bbox_projection(u_c, v_c, depth_guess, length_m, width_m, height_m,
                                              bbox2d, T_lidar_to_cam, K, T_cam_to_lidar,
                                              z_clip=z_clip)
        if best is not None and best[0] >= MIN_IOU_FOR_YAW_SEARCH:
            _, center_xyz, yaw_rad = best
            yaw_source = f"bbox_search(iou={best[0]:.2f})"
        elif yaw_hint is not None:
            yaw_rad = yaw_hint
            yaw_source = "track_history"
        # else: tetap default yaw=0.0 (axis-aligned), center_xyz dari depth_guess sensor --
        # LEBIH JUJUR "gak yakin arahnya" drpd maksa cocokkan siluet & berisiko salah jarak.

    return center_xyz, yaw_rad, length_m, width_m, height_m, yaw_source


# =============================================================================
# Fungsi bantu tracking/coasting/motion (TETAP dari Revisi ke-9/10, tidak diubah)
# =============================================================================
def try_coast_from_history(track_key, history, max_coast=MAX_COAST_FRAMES):
    if track_key is None:
        return None
    info = history.get(track_key)
    if info is None or info["coast_count"] >= max_coast:
        return None
    return info


def record_valid_track_info(track_key, center_xyz, yaw_rad, length_m, width_m, height_m, history):
    if track_key is None:
        return
    history[track_key] = {
        "center_xyz": np.asarray(center_xyz, dtype=float).copy(),
        "yaw_rad": float(yaw_rad),
        "length_m": length_m, "width_m": width_m, "height_m": height_m,
        "coast_count": 0,
    }


def update_track_motion(track_key, center_xy, frame_t, history, window_s=SPEED_WINDOW_S):
    t_now = float(frame_t)
    xy_now = np.array(center_xy, dtype=float)
    if track_key is None:
        return None, "belum_diketahui"
    buf = history.setdefault(track_key, [])
    buf.append((t_now, xy_now))
    cutoff = t_now - window_s
    while len(buf) > 1 and buf[0][0] < cutoff:
        buf.pop(0)
    if len(buf) < 2:
        return None, "belum_diketahui"
    t_old, xy_old = buf[0]
    dt = t_now - t_old
    if dt <= 1e-3:
        return None, "belum_diketahui"
    dist = float(np.linalg.norm(xy_now - xy_old))
    speed = dist / dt
    status = "diam_relatif_ego" if speed < REL_SPEED_STATIC_THRESHOLD_MPS else "bergerak_relatif_ego"
    return round(speed, 3), status


def get_frame_time(row, frame_idx, fallback_fps=10.0):
    if row.get("lidar_t") not in (None, ""):
        return float(row["lidar_t"])
    print(f"  [PERINGATAN] kolom 'lidar_t' gak ada di sync_table utk frame {frame_idx} -- "
          f"relative_speed_mps jadi PERKIRAAN kasar (asumsi {fallback_fps} fps).")
    return frame_idx / fallback_fps


# =============================================================================
# =============================================================================
# REVISI KE-14 -- ARSITEKTUR BARU BAGIAN 4: "RGB-first 3D box -> transformasi ke LiDAR -> fusi"
# =============================================================================
# Alur (menggantikan alur LiDAR-first Revisi ke-4..13):
#   YOLO bbox 2D  ->  angkat ke box 3D di FRAME KAMERA (bidang tanah + ukuran kelas + kecocokan
#   tepi bbox + konsistensi depth-map)  ->  8 corner 3D  ->  T_cam_to_lidar  ->  8 corner di
#   frame LiDAR (dari sini BEV/yaw/center diturunkan, TIDAK dihitung ulang dari titik LiDAR)  ->
#   fusi: titik LiDAR di dalam box hanya untuk validasi/koreksi jarak kecil + relabel kelas.
from scipy.optimize import minimize

# ---- konstanta lifting ----
GROUND_PLANE_HISTORY = 15          # jumlah frame utk median bidang tanah (kestabilan antar-frame)
GROUND_MAX_NORMAL_DEV_DEG = 6.0    # estimasi frame ini ditolak kalau normal menyimpang > ini dari median riwayat
GROUND_MAX_H_DEV_M = 0.25
GROUND_MIN_DEPTH_INLIERS = 3000
LIFT_YAW_STEP_DEG = 6.0
LIFT_Z_STEP_M = 0.4
LIFT_W_DEPTH = 0.35                # bobot suku konsistensi depth-map di fungsi biaya
LIFT_YAW_HISTORY_W = 0.05          # bobot prior yaw temporal (per track)
LIFT_YAW_ALIGN_W = 0.01            # tie-breaker lemah: cenderung sejajar arah ego bila bukti seimbang
LIFT_YAW_RELIABLE_GAP = 0.03       # selisih biaya solusi terbaik vs solusi alternatif berbeda-yaw
MIN_FUSION_PTS = 6
FUSION_CAND_MARGIN_M = 0.4
FUSION_LABEL_MARGIN_M = 0.15
FUSION_MAX_SHIFT_M = 1.5
FUSION_MAX_SHIFT_FRAC = 0.12
FUSION_MAX_MAD_M = 0.6
INSIDE_VOL_TOL_M = 0.40      # dilebarkan dari 0.25: 1 titik Unknown asli lolos hanya 0.37m dari tmax (near-face + box sedikit lebih kecil dari body asli krn size_scale<1)       # toleransi kecil (kalibrasi/noise) di luar [tmin,tmax] box yg msh dihitung 'di dalam'
FUSION_MIN_IOU_AFTER = 0.75
GROUND_CLEARANCE_M = 0.12          # titik lebih rendah dari ini di atas bidang tanah dianggap jalan, bukan badan objek
SIZE_SCALE_RANGE = (0.85, 1.12)    # skala seragam (L,W,H) yg boleh dipilih per objek -- tetap dlm rentang mobil nyata
SIZE_SCALE_PRIOR_W = 0.6           # penalti (ln s - ln s_prior)^2 : ukuran default = kanonik, bergeser hanya kalau bukti kuat
NEIGHBOR_YAW_W = 0.04              # prior yaw dari tetangga SEJENIS yg yaw-nya sudah andal (mobil parkir berjajar)
NEIGHBOR_MAX_DIST_M = 10.0

# ---- Revisi ke-17: bbox 2D TERPOTONG OBJEK LAIN (bukan cuma tepi gambar) -> yaw/box jadi salah ----
# Root cause: _edge_loss lama HANYA tahu "trunc" di tepi GAMBAR (bbox[0]<=2 dst). Kalau sebuah mobil
# ditutupi SEBAGIAN oleh mobil/objek lain yg lebih dekat kamera (mis. hanya 1/4 badan terlihat), bbox
# YOLO-nya memang kecil/tak lengkap -- tapi bukan krn mobil itu kecil, melainkan krn sisanya tertutup.
# Kalau edge-fit dipaksa mencocokkan SEMUA 4 sisi box 3D ke bbox parsial ini seolah itu siluet asli,
# hasilnya box collapse ke orientasi yg memberi footprint 2D terkecil/paling 'pas' dgn bbox sempit itu
# -- biasanya yaw hampir 0 derajat (nyaris lurus searah pandang kamera), BUKAN orientasi fisik asli
# mobil (yg bisa saja parkir miring). Ini persis gejala yg dilaporkan user: "mobil parkir miring tapi
# di LiDAR/BEV jadi lurus" -- khusus utk frame di mana bbox YOLO-nya memang terpotong objek lain.
OCC_AREA_FRAC = 0.12          # overlap/luas-bbox-sendiri di atas ini -> tetangga dianggap menutupi sebagian
OCC_EDGE_FRAC = 0.6           # overlap menutupi >=60% tinggi/lebar bbox pd satu sisi -> sisi itu BUKAN tepi asli mobil
OCC_DEPTH_MARGIN_M = 0.5      # tetangga baru dianggap 'di depan'/menghalangi kalau depth_prior-nya lbh dekat >= ini
PARTIAL_VIEW_FRAC = 0.55      # visible_frac di bawah ini (atau ada sisi tertutup) -> ditandai is_partial_view
PARTIAL_VIEW_GAP_MULT = 2.5   # objek partial_view butuh selisih biaya yaw yg JAUH lebih jelas baru dicap 'yaw_reliable'
NEIGHBOR_YAW_W_PARTIAL = 0.12      # prior tetangga diperkuat khusus utk objek partial_view (vs NEIGHBOR_YAW_W biasa)
LIFT_YAW_HISTORY_W_PARTIAL = 0.22  # prior yaw riwayat track sendiri jg diperkuat kalau frame ini partial_view

# ---- Revisi ke-18: riwayat yaw per track TERLALU LEMAH (bobot cuma 0.05) utk menahan noise fit per-
# frame -- akibatnya objek YANG SAMA (mis. mobil parkir yg sama, dilihat di beberapa frame saat ego
# lewat) bisa dapat yaw yg jauh berbeda antar-frame (kadang 'lurus' ~0/180 derajat krn LIFT_YAW_ALIGN_W
# menarik ke sana saat bukti lemah/ambigu), MESKIPUN box RGB tunggal per-frame itu sendiri sudah valid
# geometris (reproyeksi konsisten ke LiDAR). Ini BUKAN bug transformasi RGB->LiDAR (sudah diverifikasi
# <0.01 deg), tapi bug "percaya diri berlebihan pd estimasi SATU frame" -- padahal tersedia timestamp &
# track_id utk tiap objek yg sama sepanjang waktu. Fix: prior riwayat yaw diperkuat SECARA ADAPTIF
# sesuai 'streak' (jumlah bacaan reliable berturut-turut yg SALING KONSISTEN) -- makin lama suatu arah
# konsisten, makin kuat ia menahan frame baru yg menyimpang krn noise. Kalau memang berubah arah BESAR
# & terus-menerus (bukan sekali doang), streak di-reset & bacaan baru dipercaya lagi (tetap bisa
# mengikuti belokan sungguhan, tidak dikunci permanen).
YAW_STREAK_HIST_W_GROWTH = 0.09    # tiap bacaan reliable & konsisten menambah bobot prior riwayat
YAW_STREAK_HIST_W_MAX = 0.55       # plafon bobot -- tetap PRIOR lembut, bukan fix_yaw keras
YAW_STREAK_AGREE_DEG = 12.0        # bacaan baru dianggap 'konsisten' dgn riwayat kalau selisih < ini -> streak naik
YAW_STREAK_BREAK_DEG = 35.0        # selisih besar yg dianggap 'benar2 berubah arah' (bukan noise) -> streak reset ke 0
# Revisi ke-18b: prior lembut di atas TERNYATA tidak selalu cukup KUAT -- kalau fit bebas frame ini
# punya minimum lokal yg tajam/dalam (raw edge/depth cost jauh lebih rendah di psi lain), prior cuma
# menggeser psi sedikit, tidak mengalahkannya. Fix TEGAS: kalau fit bebas frame ini AMBIGU (tidak
# reliable) & track sudah punya riwayat mapan (streak cukup), UJI TANDING scr adil -- hitung ULANG box
# dgn yaw DIKUNCI ke nilai riwayat (posisi & skala dioptimasi ulang scr normal, fix_yaw= riwayat, pakai
# mekanisme yg sama dgn Person upright), lalu PILIH yg biayanya lebih rendah. Bukan tebakan/patch, tapi
# perbandingan biaya yg adil antara 'percaya bacaan baru yg ambigu' vs 'pertahankan arah yg sudah
# terbukti konsisten'.
TEMPORAL_LOCK_MIN_STREAK = 2


def _occlusion_sides(me_bbox, me_depth, others):
    """Revisi ke-17. `others` = list (bbox, depth_prior) deteksi LAIN di frame ini. Kembalikan
    (occ_left, occ_top, occ_right, occ_bottom) -- sisi mana dari me_bbox yg kemungkinan BUKAN tepi
    asli objek (tertutup tetangga yg lebih dekat kamera) -- dan visible_frac (perkiraan proporsi
    bbox yg masih memperlihatkan objek aslinya, bukan tetangga)."""
    x1, y1, x2, y2 = me_bbox; w = max(x2 - x1, 1.0); h = max(y2 - y1, 1.0); me_area = w * h
    occ = [False, False, False, False]; worst_frac = 0.0
    for ob_bbox, ob_depth in others:
        if ob_depth is None or me_depth is None or ob_depth >= me_depth - OCC_DEPTH_MARGIN_M:
            continue  # tetangga tidak terbukti lebih dekat kamera -> tidak dianggap menghalangi
        ox1, oy1, ox2, oy2 = ob_bbox
        ix1, iy1 = max(x1, ox1), max(y1, oy1); ix2, iy2 = min(x2, ox2), min(y2, oy2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        if iw <= 0 or ih <= 0:
            continue
        frac = (iw * ih) / me_area
        if frac < OCC_AREA_FRAC:
            continue
        worst_frac = max(worst_frac, frac)
        if ih / h >= OCC_EDGE_FRAC:                   # overlap menutupi hampir seluruh TINGGI -> sisi kiri/kanan
            if ix1 <= x1 + 0.1 * w: occ[0] = True
            if ix2 >= x2 - 0.1 * w: occ[2] = True
        if iw / w >= OCC_EDGE_FRAC:                   # overlap menutupi hampir seluruh LEBAR -> sisi atas/bawah
            if iy1 <= y1 + 0.1 * h: occ[1] = True
            if iy2 >= y2 - 0.1 * h: occ[3] = True
        if frac >= 0.5:                                # overlap sangat besar & sisi yg tertutup tak jelas -> amankan semua
            occ = [True, True, True, True]
    visible_frac = float(max(0.0, 1.0 - worst_frac))
    return tuple(occ), visible_frac


_GROUND_STATE = {"hist": [], "last_lidar_plane": None}


def reset_ground_state():
    _GROUND_STATE["hist"] = []
    _GROUND_STATE["last_lidar_plane"] = None


GROUND_MIN_VERTICALITY = 0.75      # |n[1]| minimum -- bidang tanah HARUS dekat horizontal (bukan
                                    # tembok/dinding). Dari data asli: normal tanah sungguhan ~(0.03,
                                    # -0.999, 0.04); sebuah tembok yg salah tertangkap di sini punya
                                    # normal (-0.996, -0.0005, -0.09) -- |n[1]| cuma 0.0005, jelas gagal.


def _ransac_plane(P, iters=400, tol=0.08, seed=0, require_horizontal=True):
    rng = np.random.default_rng(seed); best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn
        if require_horizontal and abs(n[1]) < GROUND_MIN_VERTICALITY:
            continue          # buang kandidat 3-titik yg bukan bidang mendekati horizontal (mis. tembok)
        if n[1] > 0:
            n = -n                           # 'atas' di frame kamera = -y
        dd = -n @ a
        inl = np.abs(P @ n + dd) < tol
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), n, dd)
    if best is None:
        return None
    inl = np.abs(P @ best[1] + best[2]) < tol
    Pc = P[inl]; cen = Pc.mean(0)
    n = np.linalg.svd(Pc - cen, full_matrices=False)[2][2]
    if n[1] > 0:
        n = -n
    if require_horizontal and abs(n[1]) < GROUND_MIN_VERTICALITY:
        return None        # hasil refine ternyata tetap bukan bidang horizontal -- tolak, jangan dipakai
    return {"n": n, "h": float(-n @ cen), "inliers": int(inl.sum()), "n_pts": int(len(P))}


def fit_ground_plane_from_depth(depth, K, v_min_frac=0.68, max_depth=30.0):
    H, W = depth.shape; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[0:H, 0:W]
    m = np.isfinite(depth) & (depth > 0.3) & (depth < max_depth) & (v > v_min_frac * H)
    if m.sum() < 500:
        return None
    P = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], -1)[m]
    if len(P) > 20000:
        P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
    return _ransac_plane(P)


def fit_ground_plane_from_lidar(points7, T_lidar_to_cam, ground_class=GROUND_CLASS_ID, max_range=25.0):
    g = points7[points7[:, 6].astype(int) == ground_class][:, :3]
    if len(g) < 200:
        return None
    pc = (T_lidar_to_cam[:3, :3] @ g.T).T + T_lidar_to_cam[:3, 3]
    pc = pc[(pc[:, 2] > 1) & (pc[:, 2] < max_range)]
    if len(pc) < 200:
        return None
    return _ransac_plane(pc, iters=300, tol=0.10)


def get_ground_plane(depth, points_full7, T_lidar_to_cam, K):
    """Bidang tanah di frame KAMERA. Prioritas: depth-map (padat, native kamera) -> titik Road
    Fase 3 -> riwayat. Distabilkan median antar-frame (jalan tidak berubah kemiringan tiap frame)."""
    cand = None; src = None
    if depth is not None:
        g = fit_ground_plane_from_depth(depth, K)
        if g is not None and g["inliers"] >= GROUND_MIN_DEPTH_INLIERS:
            cand, src = g, "depth"
    if cand is None and points_full7 is not None:
        g = fit_ground_plane_from_lidar(points_full7, T_lidar_to_cam)
        if g is not None:
            cand, src = g, "lidar_road"
    hist = _GROUND_STATE["hist"]
    if hist:
        n_med = np.median([x[0] for x in hist], axis=0); n_med /= np.linalg.norm(n_med)
        h_med = float(np.median([x[1] for x in hist]))
    if cand is not None and hist:
        dev = np.degrees(np.arccos(np.clip(cand["n"] @ n_med, -1, 1)))
        if dev > GROUND_MAX_NORMAL_DEV_DEG or abs(cand["h"] - h_med) > GROUND_MAX_H_DEV_M:
            cand, src = None, "history(outlier ditolak)"
    if cand is not None:
        hist.append((cand["n"], cand["h"])); del hist[:-GROUND_PLANE_HISTORY]
        n_med = np.median([x[0] for x in hist], axis=0); n_med /= np.linalg.norm(n_med)
        h_med = float(np.median([x[1] for x in hist]))
    elif not hist:
        return None
    n_lidar = T_lidar_to_cam[:3, :3].T @ n_med
    return {"n": n_med, "h": h_med, "source": src,
            "tilt_lidar_z_deg": float(np.degrees(np.arccos(np.clip(n_lidar[2], -1, 1))))}


def ground_frame(n, h):
    """Basis (kanan, maju, atas) + titik tanah tepat di bawah kamera, semuanya di frame kamera."""
    up = n / np.linalg.norm(n)
    zc = np.array([0., 0., 1.])
    f = zc - (zc @ up) * up; f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if r[0] < 0:
        r = -r
    return r, f, up, -h * up


def box_corners_cam(a, b, psi, L, W, H, gf):
    """8 corner box di frame kamera. (a,b)= posisi pusat alas di bidang tanah (kanan,maju);
    psi = arah panjang mobil diukur dari 'maju' ke 'kanan'. Urutan: 0-3 alas, 4-7 atap (sejajar)."""
    r, f, up, O = gf
    G = O + a * r + b * f
    hd = np.cos(psi) * f + np.sin(psi) * r
    sd = np.cross(up, hd)
    sx = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    bot = G + (L / 2) * sx[:, [0]] * hd + (W / 2) * sx[:, [1]] * sd
    return np.vstack([bot, bot + H * up])


def project_cam(P, K):
    uv = (K @ P.T).T
    return uv[:, :2] / uv[:, 2:3]


def _box_frame(a, b, psi, gf):
    r, f, up, O = gf
    G = O + a * r + b * f
    hd = np.cos(psi) * f + np.sin(psi) * r
    return G, np.stack([hd, np.cross(up, hd), up], 0)


def ray_box_depth(pix_uv, K, a, b, psi, L, W, H, gf):
    """z-kamera permukaan pertama box yg kena sinar tiap piksel; NaN kalau meleset."""
    G, Rm = _box_frame(a, b, psi, gf)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = np.stack([(pix_uv[:, 0] - cx) / fx, (pix_uv[:, 1] - cy) / fy, np.ones(len(pix_uv))], 1)
    d_b = d @ Rm.T; o_b = Rm @ (-G)
    lo = np.array([-L / 2, -W / 2, 0.0]); hi = np.array([L / 2, W / 2, H])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o_b) / d_b; t2 = (hi - o_b) / d_b
        tmin = np.nanmax(np.minimum(t1, t2), 1); tmax = np.nanmin(np.maximum(t1, t2), 1)
    return np.where(tmax > np.maximum(tmin, 0.1), tmin, np.nan)


def ray_box_depth_range(pix_uv, K, a, b, psi, L, W, H, gf):
    """Revisi ke-15: SAMA dgn ray_box_depth tapi kembalikan (tmin, tmax) sekaligus -- tmin = jarak
    sinar MASUK volume box, tmax = jarak sinar KELUAR box. Dipakai utk tes yg lebih ketat & benar
    drpd ray_box_depth saja: 'apakah TITIK LIDAR YG TERUKUR benar2 berada di antara tmin & tmax'
    (di DALAM volume box sepanjang sinar itu), bukan cuma 'apakah sinarnya menyentuh box di suatu
    titik'. Tanpa tes ini, tembok/pohon jauh di belakang objek yg kebetulan segaris sinar dgn box
    (terutama kalau box miring, spt yaw=52 derajat) ikut lolos jadi 'dukungan LiDAR' padahal titik
    itu ada JAUH di luar box (di belakangnya) -- persis kasus nyata: cross-check kelas Fase3 pada
    titik yg lolos ray_box_depth SAJA didominasi Vegetation/Building, bukan Unknown."""
    G, Rm = _box_frame(a, b, psi, gf)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = np.stack([(pix_uv[:, 0] - cx) / fx, (pix_uv[:, 1] - cy) / fy, np.ones(len(pix_uv))], 1)
    d_b = d @ Rm.T; o_b = Rm @ (-G)
    lo = np.array([-L / 2, -W / 2, 0.0]); hi = np.array([L / 2, W / 2, H])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - o_b) / d_b; t2 = (hi - o_b) / d_b
        tmin = np.nanmax(np.minimum(t1, t2), 1); tmax = np.nanmin(np.maximum(t1, t2), 1)
    ok = tmax > np.maximum(tmin, 0.1)
    return np.where(ok, tmin, np.nan), np.where(ok, tmax, np.nan)


def _edge_loss(uv, bbox, trunc):
    x1, y1, x2, y2 = bbox; w = max(x2 - x1, 8.); h = max(y2 - y1, 8.)
    e = np.array([(uv[:, 0].min() - x1) / w, (uv[:, 1].min() - y1) / h,
                  (uv[:, 0].max() - x2) / w, (uv[:, 1].max() - y2) / h])
    if trunc[0]: e[0] = max(e[0], 0.0)
    if trunc[1]: e[1] = max(e[1], 0.0)
    if trunc[2]: e[2] = min(e[2], 0.0)
    if trunc[3]: e[3] = min(e[3], 0.0)
    return float(np.sum(np.sqrt(e ** 2 + 1e-4) - 1e-2))


def _make_probe(bbox, depth, n=14, m=8):
    x1, y1, x2, y2 = bbox
    us = np.linspace(x1, x2, n + 2)[1:-1]; vs = np.linspace(y1, y2, m + 2)[1:-1]
    uu, vv = np.meshgrid(us, vs); uv = np.stack([uu.ravel(), vv.ravel()], 1)
    dv = depth[np.clip(vv.ravel().astype(int), 0, depth.shape[0] - 1),
               np.clip(uu.ravel().astype(int), 0, depth.shape[1] - 1)]
    ok = np.isfinite(dv) & (dv > 0.3)
    return uv[ok], dv[ok]



# =============================================================================
# REVISI KE-19 -- AKAR MASALAH "mobil miring jadi lurus / yaw loncat-loncat": AMBIGUITAS CERMIN
# =============================================================================
# Siluet 2D (bbox YOLO) sebuah kotak 3D SIMETRIS CERMIN terhadap garis pandang kamera ke pusat kotak:
# yaw psi dan psi' = 2*beta - psi (beta = bearing objek) menghasilkan bbox yang HAMPIR SAMA. Itulah "loncat
# 50 <-> 160 / jadi lurus". Solusi: (1) hitung KEDUA hipotesis eksplisit, (2) pilih pakai bukti yg TIDAK simetris
# (titik LiDAR: permukaan + free-space; lalu tetangga sejenis & riwayat), (3) tandai `yaw_mirror_ambiguous` kalau
# tetap seri supaya pass konsensus per-track (Bagian 8a-2) bisa memutuskan.
LIFT_W_IOU = 0.35                  # bobot suku (1 - IoU) proyeksi box vs bbox YOLO (viewport matching, sadar-oklusi)
MIRROR_MIN_SEP_DEG = 12.0          # hipotesis cermin dianggap beda kalau yaw-nya terpisah >= ini
MIRROR_AMBIG_GAP_NO_LIDAR = 0.10   # tanpa validasi LiDAR (n_hit < MIN_FUSION_PTS) siluet saja tidak cukup: ambang ambigu lebih longgar
NEIGHBOR_AGREE_DEG = 25.0          # hipotesis terpilih dianggap 'dikonfirmasi tetangga' kalau yaw-nya dlm selisih ini
MIRROR_AMBIG_GAP = 0.03            # selisih biaya total antar hipotesis < ini -> masih ambigu (x PARTIAL_VIEW_GAP_MULT utk partial)
LIDAR_HYP_W = 0.45                 # bobot suku konsistensi LiDAR (permukaan + free-space) per hipotesis
LIDAR_HYP_GROUND_MARGIN_M = 0.12   # titik harus >= ini di atas garis tanah lokal (cincin jalan jatuh < 0.06)
LIDAR_HYP_MIN_PTS = 4              # titik kandidat minimum utk suku LiDAR dipakai
LIDAR_HYP_FULL_CONF_PTS = 25       # jumlah titik utk bobot penuh (kurang dari ini bobot dikecilkan)
NEIGHBOR_ROW_MAX_DIST_M = 30.0     # jarak maks tetangga sejenis utk 'barisan parkir' (sebelumnya 10 m, terlalu pendek)


def _bbox_iou_trunc(uv, bbox, trunc):
    '''IoU bbox proyeksi box 3D vs bbox YOLO. Sisi yg terpotong (tepi gambar / tertutup objek lain)
    tidak dihukum kalau box melewatinya (sadar-oklusi, Revisi ke-17).'''
    pb = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    if trunc[0]: pb[0] = max(pb[0], bbox[0])
    if trunc[1]: pb[1] = max(pb[1], bbox[1])
    if trunc[2]: pb[2] = min(pb[2], bbox[2])
    if trunc[3]: pb[3] = min(pb[3], bbox[3])
    return _bbox_iou(pb, bbox)


def _lift_cost(p, ctx):
    a, b, psi = p[0], p[1], p[2]
    sc = float(np.clip(p[3], *SIZE_SCALE_RANGE)) if len(p) > 3 else ctx["s_prior"]
    Ls, Ws, Hs = ctx["L"] * sc, ctx["W"] * sc, ctx["H"] * sc
    C = box_corners_cam(a, b, psi, Ls, Ws, Hs, ctx["gf"])
    if np.any(C[:, 2] < 0.5):
        return 1e3
    uv_p = project_cam(C, ctx["K"])
    c = _edge_loss(uv_p, ctx["bbox"], ctx["trunc"])
    if ctx.get("w_iou", 0.0) > 0:
        c += ctx["w_iou"] * (1.0 - _bbox_iou_trunc(uv_p, ctx["bbox"], ctx["trunc"]))
    pr = ctx["probe"]
    if ctx["w_depth"] > 0 and pr is not None:
        td = ray_box_depth(pr[0], ctx["K"], a, b, psi, Ls, Ws, Hs, ctx["gf"]); hit = np.isfinite(td)
        if hit.sum() >= 3:
            err = np.abs(td[hit] - pr[1][hit]) / np.maximum(2.0, 0.08 * pr[1][hit])
            c += ctx["w_depth"] * (np.mean(np.minimum(err, 1.5)) + 0.5 * (1 - hit.mean()))
        else:
            c += ctx["w_depth"] * 1.5
    if ctx.get("lid") is not None:
        c += lidar_point_cost(ctx["lid"], ctx["K"], ctx["gf"], a, b, psi, Ls, Ws, Hs)
    for yp in ctx["yaw_priors"]:
        c += yp["w"] * (1 - np.cos(2 * (psi - yp["psi"])))
    if len(p) > 3:
        c += SIZE_SCALE_PRIOR_W * (np.log(sc) - np.log(ctx["s_prior"])) ** 2
    return c


def _ang_diff_mod_pi(x, y):
    return abs(((x - y) + np.pi / 2) % np.pi - np.pi / 2)


def lift_bbox_to_3d(bbox, depth, K, gf, dims, img_wh, depth_prior=None, yaw_priors=(), fix_yaw=None,
                    w_depth=LIFT_W_DEPTH, s_prior=1.0, allow_scale=True,
                    occ_sides=(False, False, False, False), w_iou=LIFT_W_IOU, lid=None):
    """INTI Revisi ke-14. bbox 2D -> box 3D (frame kamera). Tiga derajat kebebasan (a,b,psi) + skala
    dicari dgn kendala: (1) box menapak bidang tanah, (2) ukuran kanonik kelas, (3) proyeksi 8 corner
    harus pas dgn 4 sisi bbox YOLO (prinsip 'tight-fit' Deep3DBox), (4) permukaan box harus konsisten
    dgn depth-map di dalam bbox (gradien kedalaman memberi petunjuk yaw), dan (5) prior yaw
    temporal/sejajar-ego yg SANGAT lemah utk memutus imbang.
    Revisi ke-17: `occ_sides` (left,top,right,bottom) -- sisi bbox yg tertutup OBJEK LAIN yg lebih
    dekat (dari `_occlusion_sides`). Digabung (OR) dgn trunc tepi-gambar lama: kedua kasus sama secara
    matematis -- sisi itu BUKAN tepi asli objek, jadi edge-loss hanya menghukum box kalau MELEWATI
    bbox di sisi itu, tidak dipaksa menyentuhnya. Ini mencegah box 'collapse' ke yaw hampir 0 derajat
    hanya krn sebagian badannya tertutup objek lain di depan (mis. mobil parkir berjajar)."""
    L, W, H = dims
    r, f, up, O = gf
    Wimg, Himg = img_wh
    trunc_border = (bbox[0] <= 2, bbox[1] <= 2, bbox[2] >= Wimg - 2, bbox[3] >= Himg - 2)
    trunc = tuple(bool(tb or oc) for tb, oc in zip(trunc_border, occ_sides))
    probe = _make_probe(bbox, depth) if depth is not None else None
    use_depth = probe is not None and len(probe[0]) >= 6
    ctx = dict(L=L, W=W, H=H, K=K, gf=gf, bbox=bbox, trunc=trunc, probe=probe if use_depth else None,
               w_depth=w_depth if use_depth else 0.0, yaw_priors=list(yaw_priors), s_prior=s_prior, w_iou=w_iou, lid=lid)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xc = ((bbox[0] + bbox[2]) / 2 - cx) / fx
    ray = np.array([xc, (bbox[3] - cy) / fy, 1.0])
    denom = up @ ray
    z_contact = float((up @ O) / denom) if denom < -1e-6 else None
    zs = [z for z in (z_contact if (z_contact and 1 < z_contact < 80) else None, depth_prior) if z]
    z_lo = max(1.5, (min(zs) if zs else 5.0) - 1.5); z_hi = (max(zs) if zs else 40.0) + 1.5 + 0.5 * L
    yaws = np.radians(np.arange(-90, 90, LIFT_YAW_STEP_DEG)) if fix_yaw is None else np.array([fix_yaw])
    cand = []
    for z in np.arange(z_lo, z_hi, LIFT_Z_STEP_M):
        a0 = xc * z
        b0 = (z - O[2] - a0 * r[2]) / max(f[2], 1e-6)
        for psi in yaws:
            cand.append((_lift_cost((a0, b0, psi), ctx), a0, b0, psi))
    cand.sort(key=lambda t: t[0])
    tops, seen = [], []
    for c in cand:
        if not seen or all(_ang_diff_mod_pi(c[3], s) > np.radians(20) or abs(c[2] - t) > 2.0 for s, t in seen):
            tops.append(c); seen.append((c[3], c[2]))
        if len(tops) >= 6:
            break
    sols = []
    for c in tops:
        if fix_yaw is None:
            x0 = np.array(list(c[1:]) + ([s_prior] if allow_scale else []))
            res = minimize(lambda p: _lift_cost(p, ctx), x0, method="Nelder-Mead",
                           options=dict(xatol=1e-3, fatol=1e-5, maxiter=600))
            x = res.x if allow_scale else np.r_[res.x, s_prior]
            sols.append((res.fun, x))
        else:
            x0 = np.array(list(c[1:3]) + ([s_prior] if allow_scale else []))
            res = minimize(lambda p: _lift_cost((p[0], p[1], fix_yaw) + tuple(p[2:]), ctx), x0, method="Nelder-Mead")
            x = np.array([res.x[0], res.x[1], fix_yaw, res.x[2] if allow_scale else s_prior])
            sols.append((res.fun, x))
    sols.sort(key=lambda t: t[0])
    best = sols[0]
    alt = next((s_ for s_ in sols[1:] if _ang_diff_mod_pi(s_[1][2], best[1][2]) > np.radians(25)
                or np.linalg.norm(s_[1][:2] - best[1][:2]) > 2.0), None)
    a, b, psi = best[1][:3]
    sc = float(np.clip(best[1][3], *SIZE_SCALE_RANGE))
    psi = float(((psi + np.pi / 2) % np.pi) - np.pi / 2)
    L0, W0, H0 = L, W, H
    L, W, H = L * sc, W * sc, H * sc
    C = box_corners_cam(a, b, psi, L, W, H, gf)
    # ---- Revisi ke-19: hipotesis CERMIN eksplisit (psi' = 2*beta - psi), dioptimasi penuh (a,b,skala) ----
    mirror = None
    if fix_yaw is None:
        beta = float(np.arctan2(a, b))
        psi_m = float(((2 * beta - psi) + np.pi / 2) % np.pi - np.pi / 2)
        if _ang_diff_mod_pi(psi_m, psi) >= np.radians(MIRROR_MIN_SEP_DEG):
            best_m = None
            for z_try in (0.0, -1.5, 1.5):          # sedikit variasi titik awal jarak (sepanjang sinar)
                ray_len = np.hypot(a, b)
                k = (ray_len + z_try) / max(ray_len, 1e-6)
                x0 = np.array([a * k, b * k] + ([sc] if allow_scale else []))
                res = minimize(lambda p: _lift_cost((p[0], p[1], psi_m) + tuple(p[2:]), ctx), x0, method="Nelder-Mead",
                               options=dict(xatol=1e-3, fatol=1e-5, maxiter=600))
                if best_m is None or res.fun < best_m.fun:
                    best_m = res
            scm = float(np.clip(best_m.x[2], *SIZE_SCALE_RANGE)) if allow_scale else s_prior
            Cm = box_corners_cam(best_m.x[0], best_m.x[1], psi_m, L0 * scm, W0 * scm, H0 * scm, gf)
            if np.all(Cm[:, 2] > 0.5):
                mirror = dict(a=float(best_m.x[0]), b=float(best_m.x[1]), psi=psi_m, size_scale=scm,
                              dims=(L0 * scm, W0 * scm, H0 * scm), corners_cam=Cm, uv=project_cam(Cm, K),
                              loss=float(best_m.fun))
    return dict(mirror=mirror, a=float(a), b=float(b), psi=psi, size_scale=sc, dims=(L, W, H), corners_cam=C, uv=project_cam(C, K), loss=float(best[0]),
                alt_loss=(float(alt[0]) if alt else None), alt_psi=(float(alt[1][2]) if alt else None),
                ground_contact_depth=z_contact, used_depth_probe=bool(use_depth), n_probe=int(len(probe[0])) if probe is not None else 0)


def _bbox_of(uv):
    return (float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max()))


FUSE_EXCLUDE_GROUND_WALL = True    # Fase 3 baru: saklar (False = perilaku revisi 23)


def fuse_lidar_with_box(pc_cam, pts7, fit, dims, gf, K, bbox):
    """FUSI/VALIDASI (Revisi ke-15: pisahkan 'berapa titik LiDAR sungguh melihat objek ini' dari
    'apakah box boleh digeser'). Box RGB sudah final utk POSE (posisi/yaw/ukuran); titik LiDAR
    (pc_cam, frame kamera) dipakai utk:
    (a) DUKUNGAN/LABEL: `n_hit` = titik kandidat (dlm margin toleransi box) yang garis pandangnya
        dari kamera benar-benar menembus volume box (`ray_box_depth` valid) -- ini murni tes
        GEOMETRIS (sudut pandang), TIDAK disyaratkan sudah persis di jarak yg sama dgn box.
        Titik-titik inilah yg dipakai sbg num_lidar_points/relabel Vehicle/Person, TERLEPAS dari
        apakah koreksi posisi di bawah diterapkan atau ditolak -- karena "LiDAR melihat objek di
        situ" adalah fakta terpisah dari "apakah box perlu/boleh digeser mengikuti mereka".
    (b) KOREKSI JARAK (opsional, KECIL, sepanjang garis pandang, yaw & ukuran TIDAK diubah): dihitung
        dari median selisih jarak titik n_hit vs permukaan box, diterapkan HANYA kalau titik-titik
        itu konsisten satu sama lain (MAD kecil) DAN geserannya tidak merusak kecocokan bbox YOLO
        (IoU). Kalau ditolak, box RGB dipertahankan APA ADANYA -- tapi titik n_hit TETAP dihitung
        sbg dukungan LiDAR yg sah (lihat (a)), bukan dibuang begitu saja.
    Return (fit_baru, info, idx_hit -- index ke pc_cam/pts7)."""
    L, W, H = dims
    a, b, psi = fit["a"], fit["b"], fit["psi"]
    info = {"n_candidates": 0, "n_hit": 0, "delta_median_m": None, "mad_m": None, "applied": False,
            "rejected_reason": None, "iou_before": None, "iou_after": None}

    def _inside(a_, b_, margin_xy, top_margin, valid):
        G, Rm = _box_frame(a_, b_, psi, gf)
        pb = (pc_cam - G) @ Rm.T
        m = (np.abs(pb[:, 0]) < L / 2 + margin_xy) & (np.abs(pb[:, 1]) < W / 2 + margin_xy) & \
            (pb[:, 2] > GROUND_CLEARANCE_M) & (pb[:, 2] < H + top_margin) & valid
        return m
    valid = pc_cam[:, 2] > 0.5
    # FASE 3 BARU: titik hit TIDAK boleh tanah (cincin jalan 0.26-0.5 m di atas bidang depth-map lolos GROUND_CLEARANCE_M)
    # dan TIDAK boleh dinding. Tanah = di bawah garis tanah lokal + margin; dinding = hasil build_frame_context.
    if FUSE_EXCLUDE_GROUND_WALL:
        with np.errstate(divide="ignore", invalid="ignore"):
            _uvf = project_cam(pc_cam, K)
        _gl = local_ground_line(pc_cam, _uvf, bbox, gf)
        if _gl is not None:
            _r, _f, _up, _O = gf
            valid &= ((pc_cam - _O) @ _up) > (_gl[0] + _gl[1] * pc_cam[:, 2] + OBJSEL_GROUND_MARGIN_M)
        _w = _FRAME_CTX.get("wall")
        if _w is not None and len(_w) == len(valid):
            valid &= ~_w
    cand_idx = np.nonzero(_inside(a, b, FUSION_CAND_MARGIN_M, 0.3, valid))[0]
    info["n_candidates"] = int(len(cand_idx))
    iou0 = _bbox_iou(_bbox_of(fit["uv"]), bbox); info["iou_before"] = round(iou0, 3)
    hit_idx = np.zeros(0, dtype=int)
    if len(cand_idx) >= MIN_FUSION_PTS:
        uvp = project_cam(pc_cam[cand_idx], K)
        # Revisi ke-15: tes VOLUMETRIK penuh, bukan cuma "sinar menyentuh box di suatu titik".
        # tmin/tmax = jarak sinar masuk/keluar box; titik LiDAR yg TERUKUR (z_cam-nya) harus benar2
        # berada di antara keduanya (+ toleransi kecil) utk dianggap "di dalam box" -- kalau tidak,
        # itu artinya sinarnya kebetulan LEWAT box (krn box miring) tapi titik aslinya adalah
        # permukaan LAIN yg jauh di depan/belakang box (mis. tembok/pohon), BUKAN bagian objek ini.
        tmin, tmax = ray_box_depth_range(uvp, K, a, b, psi, L, W, H, gf)
        z_obs = pc_cam[cand_idx, 2]
        inside_vol = np.isfinite(tmin) & (z_obs >= tmin - INSIDE_VOL_TOL_M) & (z_obs <= tmax + INSIDE_VOL_TOL_M)
        hit_idx = cand_idx[inside_vol]
        info["n_hit"] = int(len(hit_idx))
        if len(hit_idx) >= MIN_FUSION_PTS:
            delta = z_obs[inside_vol] - tmin[inside_vol]
            med = float(np.median(delta)); mad = float(np.median(np.abs(delta - med)))
            info["delta_median_m"], info["mad_m"] = round(med, 3), round(mad, 3)
            if mad > FUSION_MAX_MAD_M:
                # Titik n_hit sendiri TIDAK konsisten satu sama lain (kemungkinan bukan 1 permukaan
                # tunggal, mis. tercampur objek lain yg kebetulan sejajar sinar pandang) -- jangan
                # dipercaya sbg dukungan objek ini sama sekali (bukan cuma "jangan dikoreksi").
                info["rejected_reason"] = f"titik LiDAR tidak konsisten satu sama lain (MAD {mad:.2f}m) -- diabaikan sbg dukungan"
                hit_idx = np.zeros(0, dtype=int); info["n_hit"] = 0
            elif abs(med) < 0.15:
                info["rejected_reason"] = "sudah konsisten (<0.15m), tidak perlu koreksi"
            else:
                zc = float(fit["corners_cam"][:4, 2].mean())
                strong = (FUSION_STRONG_OVERRIDE and len(hit_idx) >= FUSION_STRONG_MIN_PTS and mad <= FUSION_STRONG_MAX_MAD_M)
                lim = min(FUSION_STRONG_MAX_SHIFT_M, FUSION_STRONG_SHIFT_FRAC * zc) if strong else min(FUSION_MAX_SHIFT_M, FUSION_MAX_SHIFT_FRAC * zc)
                s = float(np.clip(med, -lim, lim))
                r_, f_, up_, O_ = gf
                G = O_ + a * r_ + b * f_
                dg = (G - O_); dg = dg - (dg @ up_) * up_; dg /= np.linalg.norm(dg)
                new = G + s * dg
                a2, b2 = float((new - O_) @ r_), float((new - O_) @ f_)
                C2 = box_corners_cam(a2, b2, psi, L, W, H, gf)
                if np.all(C2[:, 2] > 0.5):
                    uv2 = project_cam(C2, K); iou1 = _bbox_iou(_bbox_of(uv2), bbox); info["iou_after"] = round(iou1, 3)
                    if iou1 >= max(FUSION_MIN_IOU_AFTER, iou0 - 0.05) or (strong and iou1 >= FUSION_STRONG_MIN_IOU):
                        if iou1 < max(FUSION_MIN_IOU_AFTER, iou0 - 0.05):
                            info["strong_override"] = True   # diterima karena bukti LiDAR kuat, bukan karena IoU
                        fit = dict(fit, a=a2, b=b2, corners_cam=C2, uv=uv2)
                        info["applied"] = True; info["shift_m"] = round(s, 3)
                        # Box baru digeser tepat sejauh median delta n_hit -> secara konstruksi titik
                        # n_hit sekarang TEPAT di permukaan box (tidak perlu dihitung ulang).
                    else:
                        info["rejected_reason"] = (f"koreksi merusak kecocokan bbox (IoU {iou0:.2f}->{iou1:.2f}) "
                                                    f"-- box RGB dipertahankan, TAPI {len(hit_idx)} titik n_hit "
                                                    f"tetap dihitung sbg dukungan LiDAR yg sah")
    else:
        info["rejected_reason"] = f"LiDAR sparse ({int(len(cand_idx))}<{MIN_FUSION_PTS} titik) -> box RGB dipertahankan"
    return fit, info, hit_idx


def cam_box_to_lidar(corners_cam, T_cam_to_lidar):
    return (T_cam_to_lidar[:3, :3] @ corners_cam.T).T + T_cam_to_lidar[:3, 3]


def pose_from_lidar_corners(cl):
    """Center/yaw/dimensi DITURUNKAN dari 8 corner LiDAR hasil transformasi (bukan dihitung dari titik).
    yaw = arah sumbu panjang box diproyeksikan ke bidang x-y LiDAR (mod pi)."""
    center = cl.mean(0)
    e_len = ((cl[0] - cl[3]) + (cl[1] - cl[2])) / 2.0
    yaw = float(np.arctan2(e_len[1], e_len[0]) % np.pi)
    return center, yaw


def wrap_pi(x):
    return float(x % np.pi)


# Fungsi utama Bagian 4 -- SAMA nama & signature dgn versi lama (Bagian 5/6/7/8/9/10
# TIDAK PERLU diubah), tapi isinya memakai fungsi seleksi/estimasi BARU di atas.
# =============================================================================
def associate_and_classify_rev13_legacy(points_full7, uv, z_cam, idx_valid, detections, depth, K, T_cam_to_lidar,
                            track_yaw_history, track_motion_history, track_position_history,
                            track_last_valid_info, frame_t):
    objects = []
    class_col = points_full7[:, 6].copy()
    frame_z_ref = compute_frame_ground_z_ref(points_full7, GROUND_CLASS_ID)

    for det in detections:
        x1, y1, x2, y2 = det["bbox2d"]
        u_c, v_c = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        length_std, width_std, height_std = CLASS_CANONICAL_SIZE_LWH.get(det["class"], (2.0, 2.0, 1.5))

        ref_depth, ref_depth_dilation = get_reference_depth_with_dilation(depth, det["bbox2d"])
        sel_idx = select_object_points_v2(uv, z_cam, idx_valid, points_full7, det["bbox2d"],
                                           ref_depth, length_std, width_std, height_std, frame_z_ref)
        pts_in_bbox = points_full7[sel_idx]
        n_pts = len(pts_in_bbox)

        crosscheck = {}
        if n_pts > 0:
            orig_cls, orig_cnt = np.unique(pts_in_bbox[:, 6].astype(int), return_counts=True)
            crosscheck = {CLASS_NAMES.get(int(c), str(c)): int(n) for c, n in zip(orig_cls, orig_cnt)}

        if n_pts >= MIN_LIDAR_PTS_IN_BBOX:
            sub_groups = split_merged_clusters(pts_in_bbox, det["class"])
            is_split = len(sub_groups) > 1
            if is_split:
                print(f'  [SPLIT] {det["class"]} id={det.get("track_id")} bbox={det["bbox2d"]} -- '
                      f'{n_pts} titik terpecah jadi {len(sub_groups)} objek')

            for sub_i, local_idx in enumerate(sub_groups):
                sub_sel_idx = sel_idx[local_idx]
                sub_pts = pts_in_bbox[local_idx]
                xyz = sub_pts[:, :3]
                mins, maxs = xyz.min(axis=0), xyz.max(axis=0)
                dims_raw = maxs - mins
                length_raw, width_raw, height_raw = float(dims_raw[0]), float(dims_raw[1]), float(dims_raw[2])

                base_track_id = det.get("track_id")
                track_key = (f"{base_track_id}_{sub_i}" if (is_split and base_track_id is not None)
                             else base_track_id)

                yaw_guess = track_yaw_history.get(track_key) if track_key is not None else None
                depth_guess = robust_camera_depth(xyz, T_lidar_to_cam) if len(xyz) else ref_depth
                if depth_guess is None:
                    depth_guess = ref_depth if ref_depth is not None else 20.0
                if is_split:
                    uv_sub, _, _ = project_lidar_to_image(xyz, T_lidar_to_cam, K)
                    anchor_u, anchor_v = float(uv_sub[:, 0].mean()), float(uv_sub[:, 1].mean())
                    anchor_bbox = det["bbox2d"]   # bbox asli (gabungan) dipakai sbg acuan proyeksi
                else:
                    anchor_u, anchor_v = u_c, v_c
                    anchor_bbox = det["bbox2d"]
                center_hint = backproject_pixel_to_lidar(anchor_u, anchor_v, depth_guess, K, T_cam_to_lidar)

                center, yaw_rad, length_std_i, width_std_i, height_std_i, yaw_source = fit_shape_pose(
                    xyz, length_std, width_std, height_std, anchor_bbox, T_lidar_to_cam, K,
                    T_cam_to_lidar, anchor_u, anchor_v, depth_guess, center_hint, yaw_guess, frame_z_ref)
                yaw_from_estimate = yaw_source.startswith(("points", "bbox_search"))

                if track_key is not None and yaw_from_estimate:
                    if track_key in track_yaw_history:
                        yaw_rad = circular_ema_mod_pi(track_yaw_history[track_key], yaw_rad, YAW_EMA_ALPHA)
                    track_yaw_history[track_key] = yaw_rad

                # --- Revisi ke-8b (TETAP): geser center menjauhi ego, kompensasi bias "near face" ---
                center = adjust_center_for_near_face(center, yaw_rad, length_std_i, width_std_i)
                # --- Revisi ke-9 (TETAP): smoothing posisi antar-frame per track ---
                center = smooth_track_position(track_key, center, track_position_history)
                # --- Revisi ke-8c (TETAP): status gerak/diam relatif ego per track ---
                rel_speed, move_status = update_track_motion(track_key, center[:2], frame_t,
                                                               track_motion_history)
                record_valid_track_info(track_key, center, yaw_rad, length_std_i, width_std_i,
                                         height_std_i, track_last_valid_info)

                objects.append({
                    "class":              det["class"],
                    "track_id":           track_key,
                    "confidence":         det["confidence"],
                    "bbox2d":             det["bbox2d"],
                    "center_xyz":         center.tolist(),
                    "distance_m":         float(np.linalg.norm(np.asarray(center)[:2])),
                    "length_m":           length_std_i, "width_m": width_std_i, "height_m": height_std_i,
                    "length_m_raw":       length_raw,  "width_m_raw": width_raw,  "height_m_raw": height_raw,
                    "num_lidar_points":   int(len(xyz)),
                    "num_rejected_depth_outliers": None,
                    "num_rejected_height_outliers": None,
                    "num_box_outliers_removed":    None,
                    "box_validated":      yaw_source == "points",
                    "source":             "lidar",
                    "yaw_rad":            round(float(yaw_rad), 4),
                    "yaw_from_estimate":  bool(yaw_from_estimate),
                    "yaw_source":         yaw_source,
                    "fase3_class_crosscheck": crosscheck if sub_i == 0 else {},
                    "is_split_from_single_detection": is_split,
                    "relative_speed_mps": rel_speed,
                    "movement_status":    move_status,
                    "is_coasted":         False,
                    "coast_count":        0,
                    "depth_dilation_factor": None,
                })
                # Override class_id HANYA utk titik yg BENERAN dipakai (sel_idx/sub_sel_idx) --
                # tidak ada lagi "inlier_mask" terpisah krn seleksi sudah final di select_object_points_v2.
                class_col[sub_sel_idx] = OBJECT_CLASS_ID[det["class"]]
            continue

        # --- n_pts < MIN_LIDAR_PTS_IN_BBOX -> jalur depth_fallback (TETAP dari Revisi ke-6/10) ---
        track_key = det.get("track_id")
        fb, fb_dilation = bbox_depth_fallback(depth, det["bbox2d"], K, T_cam_to_lidar)
        if fb is None:
            coast_info = try_coast_from_history(track_key, track_last_valid_info)
            if coast_info is not None:
                center = smooth_track_position(track_key, coast_info["center_xyz"], track_position_history)
                rel_speed, move_status = update_track_motion(track_key, center[:2], frame_t, track_motion_history)
                new_coast_count = coast_info["coast_count"] + 1
                track_last_valid_info[track_key] = {**coast_info, "coast_count": new_coast_count}
                print(f'  [COAST] {det["class"]} id={track_key} -- LiDAR & depth kamera gagal '
                      f'frame ini, pinjam posisi valid terakhir (coasting ke-{new_coast_count}/{MAX_COAST_FRAMES})')
                objects.append({
                    "class": det["class"], "track_id": track_key, "confidence": det["confidence"],
                    "bbox2d": det["bbox2d"], "center_xyz": center.tolist(),
                    "distance_m": float(np.linalg.norm(np.asarray(center)[:2])),
                    "length_m": coast_info["length_m"], "width_m": coast_info["width_m"],
                    "height_m": coast_info["height_m"],
                    "length_m_raw": None, "width_m_raw": None, "height_m_raw": None,
                    "num_lidar_points": int(n_pts), "num_rejected_depth_outliers": None,
                    "num_rejected_height_outliers": None, "num_box_outliers_removed": None,
                    "box_validated": False, "source": "coasted",
                    "yaw_rad": round(float(coast_info["yaw_rad"]), 4), "yaw_from_estimate": False,
                    "yaw_source": "coasted",
                    "fase3_class_crosscheck": crosscheck, "is_split_from_single_detection": False,
                    "relative_speed_mps": rel_speed, "movement_status": move_status,
                    "is_coasted": True, "coast_count": new_coast_count,
                    "depth_dilation_factor": None,
                })
                continue
            print(f'  [SKIP] {det["class"]} id={det.get("track_id")} bbox={det["bbox2d"]} -- '
                  f'LiDAR: {n_pts} titik (kurang dari {MIN_LIDAR_PTS_IN_BBOX}); depth kamera '
                  f'juga tidak valid; tidak ada posisi sebelumnya utk coasting -- dilewati')
            continue

        center = fb
        source = "depth_fallback"
        yaw_rad, yaw_from_estimate = 0.0, False

        bbox_area = (x2 - x1) * (y2 - y1)
        min_conf = MIN_CONF_FOR_DEPTH_FALLBACK.get(det["class"], DEFAULT_MIN_CONF_FOR_DEPTH_FALLBACK)
        if det["confidence"] < min_conf or bbox_area < MIN_BBOX_AREA_PX_FOR_DEPTH_FALLBACK:
            print(f'  [SKIP-depth_fallback] {det["class"]} id={det.get("track_id")} '
                  f'conf={det["confidence"]:.2f} (butuh >={min_conf}) bbox_area={bbox_area:.0f}px '
                  f'(butuh >={MIN_BBOX_AREA_PX_FOR_DEPTH_FALLBACK}px) -- kemungkinan false positive, dilewati')
            continue

        # Revisi ke-12: WALAU LiDAR nyaris kosong, tetap coba cari orientasi lewat proyeksi
        # (bukan cuma pasrah yaw=0) -- pakai depth dari depth map (`ref_depth`, sudah dihitung
        # di awal loop) sbg jangkar jarak, sama seperti jalur LiDAR di atas.
        yaw_guess = track_yaw_history.get(track_key) if track_key is not None else None
        depth_guess_fb = ref_depth if ref_depth is not None else float(np.linalg.norm(center[:2]))
        center, yaw_rad, length_std, width_std, height_std, yaw_source = fit_shape_pose(
            np.zeros((0, 3)), length_std, width_std, height_std, det["bbox2d"], T_lidar_to_cam,
            K, T_cam_to_lidar, u_c, v_c, depth_guess_fb, center, yaw_guess, frame_z_ref)
        yaw_from_estimate = yaw_source.startswith("bbox_search")
        if track_key is not None and yaw_from_estimate:
            if track_key in track_yaw_history:
                yaw_rad = circular_ema_mod_pi(track_yaw_history[track_key], yaw_rad, YAW_EMA_ALPHA)
            track_yaw_history[track_key] = yaw_rad

        center = adjust_center_for_near_face(np.asarray(center, dtype=float), yaw_rad, length_std, width_std)
        center = smooth_track_position(track_key, center, track_position_history)
        rel_speed, move_status = update_track_motion(track_key, center[:2], frame_t, track_motion_history)
        record_valid_track_info(track_key, center, yaw_rad, length_std, width_std, height_std,
                                 track_last_valid_info)

        objects.append({
            "class": det["class"], "track_id": track_key, "confidence": det["confidence"],
            "bbox2d": det["bbox2d"], "center_xyz": center.tolist() if hasattr(center, "tolist") else list(center),
            "distance_m": float(np.linalg.norm(np.asarray(center)[:2])),
            "length_m": length_std, "width_m": width_std, "height_m": height_std,
            "length_m_raw": None, "width_m_raw": None, "height_m_raw": None,
            "num_lidar_points": int(n_pts), "num_rejected_depth_outliers": None,
            "num_rejected_height_outliers": None, "num_box_outliers_removed": None,
            "box_validated": False, "source": source,
            "yaw_rad": round(float(yaw_rad), 4), "yaw_from_estimate": bool(yaw_from_estimate),
            "yaw_source": yaw_source,
            "fase3_class_crosscheck": crosscheck, "is_split_from_single_detection": False,
            "relative_speed_mps": rel_speed, "movement_status": move_status,
            "is_coasted": False, "coast_count": 0,
            "depth_dilation_factor": fb_dilation,   # Revisi ke-13: >1.0 kalau bbox depth diperlebar
        })

    return objects, class_col



# =============================================================================
# Fungsi utama Bagian 4 (REVISI KE-14) -- nama & signature SAMA dgn versi sebelumnya, jadi Bagian
# 5/6/7/8 tetap memanggilnya persis seperti biasa. Versi lama disimpan sbg
# `associate_and_classify_rev13_legacy` (tidak dipakai lagi; hanya utk pembanding).
# Tahap: A) tiap deteksi -> box 3D RGB + fusi | B) prior yaw tetangga utk yaw yg ambigu |
#        C) transformasi ke LiDAR + tulis state track & objek.
# =============================================================================

def local_ground_line(pc_cam, uv_all, bbox, gf):
    """Revisi ke-19. Tinggi 'tanah' menurut LiDAR sbg fungsi jarak, dari titik yg PASTI tanah: titik tepat di BAWAH
    bbox di gambar. Data nyata: cincin jalan terukur 0.26-0.5 m DI ATAS bidang depth-map & naik seiring jarak
    (kalibrasi/pitch), jadi ambang tinggi tetap (0.12 m) meloloskan cincin jalan sbg 'badan mobil'.
    Return (alpha, beta) utk tinggi_tanah(z) = alpha + beta*z, atau None kalau titik tanah kurang."""
    r, f, up, O = gf
    x1, y1, x2, y2 = bbox
    m = (pc_cam[:, 2] > 2.0) & (uv_all[:, 0] >= x1 - 60) & (uv_all[:, 0] <= x2 + 60) & \
        (uv_all[:, 1] > y2 + 3) & (uv_all[:, 1] < y2 + 90)
    if m.sum() < 25:
        return None
    z = pc_cam[m, 2]; h = ((pc_cam[m] - O) @ up)
    keep = h < np.median(h) + 0.5
    if keep.sum() < 25:
        return None
    beta, alpha = np.polyfit(z[keep], h[keep], 1)
    beta = float(np.clip(beta, 0.0, 0.08))
    alpha = float(np.median(h[keep] - beta * z[keep]))
    return alpha, beta


def select_lidar_candidates(pc_cam, uv_all, bbox, gf, H_obj, point_ok=None, L_obj=4.2):
    """Revisi ke-19. Pilih titik LiDAR yg MUNGKIN badan objek: di dalam bbox (dikecilkan 4%), DI ATAS tanah lokal
    (bukan cincin jalan), kelas Fase3 yg mungkin badan mobil, lalu klaster kedalaman terpadat (jendela 1.5 m)
    dilebarkan ke [z_mode-1, z_mode+L+1]. Himpunan ini SAMA utk semua hipotesis -> perbandingan antar-hipotesis adil.
    Return dict(idx, uv, z, z_mode, n) atau None kalau < LIDAR_HYP_MIN_PTS titik."""
    r, f, up, O = gf
    x1, y1, x2, y2 = bbox
    sh_x, sh_y = 0.04 * (x2 - x1), 0.04 * (y2 - y1)
    m = (pc_cam[:, 2] > 1.0) & (uv_all[:, 0] >= x1 + sh_x) & (uv_all[:, 0] <= x2 - sh_x) & \
        (uv_all[:, 1] >= y1 + sh_y) & (uv_all[:, 1] <= y2 - sh_y)
    hgt = (pc_cam - O) @ up
    gl = local_ground_line(pc_cam, uv_all, bbox, gf)
    if gl is not None:
        h_ground = gl[0] + gl[1] * pc_cam[:, 2]
        m &= (hgt > h_ground + LIDAR_HYP_GROUND_MARGIN_M) & (hgt < h_ground + H_obj + 0.5)
    else:
        m &= (hgt > GROUND_CLEARANCE_M + 0.35) & (hgt < H_obj + 1.0)
    if point_ok is not None:
        m &= point_ok
    idx = np.nonzero(m)[0]
    if len(idx) < LIDAR_HYP_MIN_PTS:
        return None
    z = pc_cam[idx, 2]
    grid = np.arange(z.min(), z.max() + 0.5, 0.5)
    cnt = np.array([((z >= g0) & (z < g0 + 1.5)).sum() for g0 in grid])
    g_best = grid[int(np.argmax(cnt))]
    z_mode = float(np.median(z[(z >= g_best) & (z < g_best + 1.5)]))
    keep = (z >= z_mode - 1.0) & (z <= z_mode + L_obj + 1.0)
    idx = idx[keep]
    if len(idx) < LIDAR_HYP_MIN_PTS:
        return None
    return dict(idx=idx, uv=uv_all[idx], z=pc_cam[idx, 2], z_mode=z_mode, n=int(len(idx)))


def lidar_point_cost(lid, K, gf, a, b, psi, L, W, H):
    """Revisi ke-19. Biaya konsistensi LiDAR utk SATU box thd himpunan kandidat `lid`. Per titik: sinar harus
    MENEMBUS box (meleset -> 1); titik di dalam volume -> 0; titik di DEPAN permukaan box (free-space violation) ->
    naik s/d 1 pd 1 m; titik di belakang box (latar) -> 0.3. Tidak simetris-cermin."""
    if lid is None:
        return 0.0
    tmin, tmax = ray_box_depth_range(lid["uv"], K, a, b, psi, L, W, H, gf)
    z_obs = lid["z"]
    e = np.ones(len(z_obs))
    hit = np.isfinite(tmin)
    front = hit & (z_obs < tmin - 0.15)
    inside = hit & ~front & (z_obs <= tmax + 0.3)
    behind = hit & (z_obs > tmax + 0.3)
    e[inside] = 0.0
    e[front] = np.minimum((tmin[front] - z_obs[front]) / 1.0, 1.0)
    e[behind] = 0.3
    conf = min(1.0, len(z_obs) / LIDAR_HYP_FULL_CONF_PTS)
    return float(LIDAR_HYP_W * lid.get("w_mult", 1.0) * conf * e.mean())


def _hyp_lidar_yaw(hyp, T_cam_to_lidar_):
    cl = cam_box_to_lidar(hyp["corners_cam"], T_cam_to_lidar_)
    c, yw = pose_from_lidar_corners(cl)
    return float(yw), c


def _fit_from_hyp(base_fit, hyp):
    """Bangun dict `fit` (format sama dgn keluaran lift_bbox_to_3d) dari satu hipotesis (primer atau cermin)."""
    out = dict(base_fit)
    out.update(a=hyp["a"], b=hyp["b"], psi=hyp["psi"], size_scale=hyp["size_scale"], dims=hyp["dims"],
               corners_cam=hyp["corners_cam"], uv=hyp["uv"], loss=hyp["loss"])
    return out


def _lidar_depth_prior(pc_cam, uv_all, bbox, gf, H_obj, ref_depth, point_ok=None):
    """Prior jarak dari LiDAR utk MEMBANTU pencarian (bukan menentukan bentuk/yaw). Revisi ke-19: pakai
    `select_lidar_candidates` (di atas TANAH LOKAL, kelas Unknown/Vehicle) -- versi lama menyaring tinggi thd bidang
    depth-map & menggating dgn ref_depth dari dilasi bbox, sehingga utk mobil jauh yg depth-nya kosong
    ia mengambil cincin JALAN di 11 m (padahal mobil di 25 m)."""
    cand = select_lidar_candidates(pc_cam, uv_all, bbox, gf, H_obj, point_ok=point_ok)
    if cand is None:
        return None, 0
    return cand["z_mode"], cand["n"]


def _eval_hypotheses(rec, fit, gf, K, pc_cam, uv_all, pts7):
    """Revisi ke-19: nilai hipotesis primer & cermin. Biaya TOTAL = loss lift (sudah memuat suku LiDAR krn suku itu
    ikut dioptimasi). Disimpan di rec['hyps'] (fit, total, lidar_cost, lidar_yaw, center)."""
    base = dict(fit); base.pop("mirror", None)
    cands = [("primary", fit)] + ([("mirror", fit["mirror"])] if fit.get("mirror") is not None else [])
    lid = rec.get("lid")
    hyps = []
    for name, h in cands:
        Lh, Wh, Hh = h["dims"]
        lc = lidar_point_cost(lid, K, gf, h["a"], h["b"], h["psi"], Lh, Wh, Hh)
        yw, ctr = _hyp_lidar_yaw(h, T_cam_to_lidar)
        hyps.append(dict(name=name, fit=_fit_from_hyp(base, h), lidar_cost=lc, n_cand=(lid["n"] if lid else 0),
                         total=float(h["loss"]), lidar_yaw=yw, center=ctr))
    return hyps



# =============================================================================
# FASE 3 BARU (perbaikan yaw "lurus") -- FIT BENTUK-L LANGSUNG DARI TITIK LiDAR
# =============================================================================
# AKAR MASALAH (frame 1098, cold-start): box RGB-lift ditempatkan di z=10.1 m (depth ZED) & yaw 6 deg (tebakan dari bbox 2D, ambigu cermin).
# Titik LiDAR terpilih (24 titik, membentuk GARIS/L sepanjang ~3 m miring ~110 deg) ada di z=11.5-13 m -> jauh dari box -> 0 titik tervalidasi
# -> yaw tidak pernah dikoreksi oleh bentuk titik (suku LiDAR di lift hanya menggeser, tidak punya cukup tarikan melawan suku depth/bbox).
# PERBAIKAN: bila klaster terpilih cukup kuat dan bentuknya jelas (garis/L, ukuran sesuai mobil), yaw & posisi dihitung LANGSUNG dari titik:
# cari yaw (1 deg) yg meminimalkan jarak titik ke 2 sisi terdekat-sensor persegi panjang berukuran kanonik; pusat = sisi terdekat digeser
# setengah panjang/lebar menjauhi sensor. Tidak ada ambiguitas cermin (arah sisi nyata, box simetris mod pi).
LFIT_ENABLED = True
LFIT_MIN_PTS = 12
LFIT_MIN_EXTENT_M = 1.5
LFIT_MAX_RESID_M = 0.15
LFIT_EXT_TOL_M = 0.35


def lidar_L_fit(lid, pc_cam, gf, dims, T_l2c, pts7=None):
    """Return dict(a,b,psi,res,extent) atau None. Koordinat (a=kanan, b=maju) pada bidang tanah, sensor LiDAR sbg titik pandang."""
    if lid is None or lid.get("idx") is None or len(lid["idx"]) < LFIT_MIN_PTS:
        return None
    if pts7 is not None and _n_rings_ge3(pts7, lid["idx"]) < YAW_EV_MIN_RINGS:
        return None
    r, f, up, O = gf
    Lc, Wc, _ = dims
    Q = pc_cam[lid["idx"]] - O
    ab = np.stack([Q @ r, Q @ f], 1)
    s0 = np.asarray(T_l2c)[:3, 3] - O
    sen = np.array([s0 @ r, s0 @ f])
    ab = ab - sen                                   # sensor di (0,0)
    best = None
    for th in np.radians(np.arange(0.0, 180.0, 1.0)):
        hd = np.array([np.sin(th), np.cos(th)]); sd = np.array([hd[1], -hd[0]])
        u = ab @ hd; v = ab @ sd
        eu = np.percentile(u, 98) - np.percentile(u, 2); ev = np.percentile(v, 98) - np.percentile(v, 2)
        pen = max(0.0, eu - (Lc + LFIT_EXT_TOL_M)) + max(0.0, ev - (Wc + LFIT_EXT_TOL_M))
        upos = u.mean() > 0; vpos = v.mean() > 0
        un = u.min() if upos else u.max(); vn = v.min() if vpos else v.max()
        res = float(np.median(np.minimum(np.abs(u - un), np.abs(v - vn))))
        sc = res + 0.5 * pen
        if best is None or sc < best[0]:
            best = (sc, th, un, vn, upos, vpos, res, pen, eu, ev)
    sc, th, un, vn, upos, vpos, res, pen, eu, ev = best
    ext = float(max(eu, ev))
    if pen > 0 or res > LFIT_MAX_RESID_M or ext < LFIT_MIN_EXTENT_M:
        return None
    hd = np.array([np.sin(th), np.cos(th)]); sd = np.array([hd[1], -hd[0]])
    c = (un + (Lc / 2 if upos else -Lc / 2)) * hd + (vn + (Wc / 2 if vpos else -Wc / 2)) * sd + sen
    return dict(a=float(c[0]), b=float(c[1]), psi=float(th), res=res, extent=ext)


def _run_lift_and_fuse(rec, gf, K, depth, pc_cam, pts7, img_wh, extra_priors=(), force_hyp=None):
    det = rec["det"]; L, W, H = rec["canon"]; cls = det["class"]
    fit = lift_bbox_to_3d(det["bbox2d"], depth, K, gf, (L, W, H), img_wh, depth_prior=rec["depth_prior"],
                          yaw_priors=list(rec["yaw_priors"]) + list(extra_priors),
                          w_depth=(LIFT_W_DEPTH * OBJSEL_DEPTH_W_MULT if (rec.get("lid") or {}).get("anchored") else LIFT_W_DEPTH),
                          fix_yaw=(0.0 if cls == "Person" else None), s_prior=rec["s_prior"],
                          occ_sides=rec.get("occ_sides", (False, False, False, False)), lid=rec.get("lid"))
    rec["lfit"] = None
    if LFIT_ENABLED and cls != "Person" and (rec.get("lid") or {}).get("anchored"):
        _lf = lidar_L_fit(rec["lid"], pc_cam, gf, fit["dims"], T_lidar_to_cam, pts7)
        if _lf is not None:
            _C = box_corners_cam(_lf["a"], _lf["b"], _lf["psi"], *fit["dims"], gf)
            if np.all(_C[:4, 2] > 0.5):
                fit = dict(fit, a=_lf["a"], b=_lf["b"], psi=_lf["psi"], corners_cam=_C, uv=project_cam(_C, K),
                           mirror=None, alt_loss=None, alt_psi=None)
                rec["lfit"] = _lf
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_all = (K @ pc_cam.T).T; uv_all = uv_all[:, :2] / uv_all[:, 2:3]
    hyps = _eval_hypotheses(rec, fit, gf, K, pc_cam, uv_all, pts7)
    rec["hyps"] = hyps
    # pilih hipotesis biaya total terendah (kecuali dipaksa oleh tahap keputusan: tetangga/riwayat)
    chosen = min(hyps, key=lambda h: h["total"]) if force_hyp is None else next(h for h in hyps if h["name"] == force_hyp)
    other = next((h for h in hyps if h is not chosen), None)
    fit0 = dict(chosen["fit"], alt_loss=fit["alt_loss"], alt_psi=fit["alt_psi"])
    fit_f, fus, lab_idx = fuse_lidar_with_box(pc_cam, pts7, fit0, fit0["dims"], gf, K, det["bbox2d"])
    rec.update(fit=fit_f, fus=fus, lab_idx=lab_idx, chosen_hyp=chosen["name"])
    # gap = ketidakpastian TERBESAR dari dua sumber: (a) minimum lokal lain (alt_loss), (b) hipotesis cermin
    gap_alt = (fit["alt_loss"] - fit["loss"]) if fit["alt_loss"] is not None else 1.0
    gap_mir = (other["total"] - chosen["total"]) if other is not None else 1.0
    rec["gap"] = float(min(gap_alt, gap_mir)) if force_hyp is None else float(max(gap_mir, 0.0))
    rec["mirror_gap"] = float(gap_mir)
    rec["alt_hyp"] = other
    mult = PARTIAL_VIEW_GAP_MULT if rec.get("is_partial_view") else 1.0
    gap_req = LIFT_YAW_RELIABLE_GAP * mult
    lidar_validated = fus["n_hit"] >= MIN_FUSION_PTS
    amb_thr = (MIRROR_AMBIG_GAP if lidar_validated else MIRROR_AMBIG_GAP_NO_LIDAR) * mult
    rec["mirror_ambiguous"] = bool(other is not None and gap_mir < amb_thr and cls != "Person")
    rec["yaw_reliable"] = bool((rec["gap"] >= gap_req and not rec["mirror_ambiguous"]) or cls == "Person")



# =============================================================================
# REVISI KE-21 -- GERBANG VALIDASI: "YOLO halu" (Person di rute tanpa orang, "mobil" yang bukan mobil)
# =============================================================================
# AKAR MASALAH (dari screenshot viewer frame 696/1353/1267/845/397 + pembacaan kode):
#  1. Gerbang bukti lemah lama (`MIN_CONF_FOR_DEPTH_FALLBACK`, area minimum) HANYA aktif kalau `depth_prior is None`.
#     Begitu depth-map memberi angka (walau di piksel kecil/derau), deteksi conf 0.44-0.69 lolos tanpa syarat apa pun.
#  2. Semua objek yang halu bertanda `track_id=None`: ByteTrack belum mengonfirmasi (deteksi sekali lewat). Kode lama
#     memperlakukannya sama dgn track terkonfirmasi.
#  3. Tidak ada uji "apakah box 3D yang dihasilkan masih COCOK dgn bbox YOLO?". Contoh frame 1353: bbox Person cuma
#     ~18 px tinggi (artinya jauh), tapi depth-map bilang 14.5 m -> box 3D setinggi 60 px nyangkut di atas bbox
#     (IoU ~ 0.0-0.3). Mobil asli: IoU 0.66-0.99 (diukur di idx 1095, bbox digoyang +-4 px: min 0.78).
#  4. Tidak ada batas jarak utk objek tanpa bukti LiDAR (frame 1267: 87 m, frame 845: 57 m -- depth/LiDAR di jarak itu
#     tidak ada, jadi posisi & yaw cuma tebakan).
# PERBAIKAN: gerbang berbasis BUKTI (bukan potong jarak buta). Setiap penolakan dicatat di JSON (`rejected`, + alasan)
# dan bisa dilihat di viewer Fase 4b (tombol r) -- jadi ambang di bawah bisa disetel dari data nyata.
PERSON_ENABLED_DEFAULT = True          # override dgn variabel notebook PERSON_ENABLED (sel YOLO). Rute tanpa orang: set False.
MAX_RANGE_ANY_M = 70.0                 # di atas ini apapun buktinya dianggap tidak terverifikasi
MAX_RANGE_NO_LIDAR_M = 40.0            # tanpa titik LiDAR tervalidasi (n_hit < MIN_FUSION_PTS) & lebih jauh dari ini -> tolak
CAR_MIN_CONF_NO_LIDAR = 0.60           # conf minimum Car tanpa dukungan LiDAR (track terkonfirmasi)
CAR_MIN_CONF_NO_LIDAR_UNCONFIRMED = 0.80   # idem tapi track_id=None (belum dikonfirmasi tracker)
CAR_MIN_CONF_UNCONFIRMED_LIDAR = 0.55  # Car track_id=None walau ada LiDAR
MIN_BBOX_AREA_NO_LIDAR_PX = 500        # bbox terlalu kecil & tanpa LiDAR -> bentuknya tak bisa diverifikasi
MIN_FIT_IOU_CAR = 0.40                 # IoU(proyeksi box 3D, bbox YOLO) minimum (sadar-oklusi/tepi gambar)
PERSON_MIN_CONF = 0.70
PERSON_UNCONFIRMED_MIN_CONF = 0.90
PERSON_NO_LIDAR_MIN_CONF = 0.85        # Person tanpa >= PERSON_MIN_LIDAR_CAND titik kandidat harus sangat yakin
PERSON_MIN_LIDAR_CAND = 3
PERSON_MIN_ASPECT = 1.3                # tinggi/lebar bbox minimum (orang berdiri ~2-3)
PERSON_MIN_HEIGHT_PX = 28
PERSON_MAX_RANGE_M = 30.0
MIN_FIT_IOU_PERSON = 0.50

# ---- Gerbang kelas rentan halusinasi (Person/Motorcycle/Bicycle) -- universal, bukan per-rute ----
ROUTE_ALLOWED_CLASSES = None           # None = semua kelas. Isi mis. ("Car",) utk rute yg dipastikan hanya ada mobil.
GROUNDED_CLASSES = ("Person", "Motorcycle", "Bicycle")
GROUNDED_REL_TOL = 0.35                # |jarak kontak-tanah bbox - jarak depth| / jarak depth maksimum
SMALL_VEH_MIN_CONF = 0.65
SMALL_VEH_MIN_CONF_UNCONFIRMED = 0.85
SMALL_VEH_H_MIN_M = 0.7
SMALL_VEH_H_MAX_M = 2.4
SMALL_VEH_MIN_LIDAR_CAND = 3
SMALL_VEH_MAX_RANGE_M = 30.0

REJECTED_LAST_FRAME = []               # diisi tiap panggilan associate_and_classify; dibaca process_frame_fase4


# =============================================================================
# REVISI KE-22 -- (a) BUKTI YAW BERBASIS BENTUK TITIK LiDAR (bukan sekadar jumlah titik / selisih loss RGB)
#                 (b) FILTER TINGGI FISIK Person (hidran/tiang merah bukan orang) -- Person tetap AKTIF di semua rute
# =============================================================================
# AKAR MASALAH yaw "lurus" yg menyebar ke ~20 frame (screenshot frame 1111/1119/1124/1134, mobil #67):
#  - Frame dianggap 'andal/kuat' hanya karena selisih loss RGB cukup besar (yaw_reliable) + >=6 titik DI DALAM box. Titik di
#    dalam box TIDAK memberi info arah: 12-36 titik yg mengumpul di satu sudut/ruas pendek cocok dgn yaw apa pun.
#  - Frame 'andal' palsu itu lalu (1) mengisi riwayat yaw & streak -> temporal lock memaksa frame berikutnya, dan
#    (2) jadi jangkar konsensus 8a-2 -> disalin ke frame tanpa LiDAR (depth_fallback, 0 titik) sbg 'consensus_snap'.
# PERBAIKAN: yaw hanya disebut BERBUKTI (`yaw_evidence=True`) bila titik LiDAR kandidat badan mobil membentuk GARIS/permukaan
# yg cukup panjang DAN arahnya cocok dgn sumbu box terpilih tapi TIDAK cocok dgn hipotesis cermin (= bukti asimetris).
# Hanya frame berbukti yg boleh mengisi riwayat/streak, memicu temporal lock, dan jadi jangkar konsensus.
YAW_EV_MIN_PTS = 10                # titik kandidat minimum utk menilai bentuk
YAW_EV_MIN_EXTENT_M = 1.4          # sebaran terpanjang titik (persentil 2.5-97.5) minimum
YAW_EV_MAX_RESID_M = 0.15          # median jarak titik ke 2 sisi (bentuk-L) box berorientasi yaw terpilih
YAW_EV_ALT_MARGIN_M = 0.06         # ...dan hipotesis cermin harus lebih buruk sebesar ini (bukti asimetris)
PERSON_H_MIN_M = 1.15              # tinggi fisik (dari bbox & jarak LiDAR/depth) Person: 1.15-2.30 m (anak/duduk s.d. tinggi)
PERSON_H_MAX_M = 2.30


YAW_EV_ACROSS_TOL_M = 0.6          # sebaran melintang maks = lebar kanonik + ini
YAW_EV_MIN_RINGS = 2               # jumlah ring LiDAR (masing2 >= 3 titik) minimum utk bukti yaw / fit bentuk-L


def _n_rings_ge3(pts7, idx):
    if pts7.shape[1] < 5 or len(idx) == 0:
        return 99                                  # tanpa kolom ring: tidak bisa menilai -> jangan menolak
    _, cn = np.unique(pts7[idx, 4].astype(int), return_counts=True)
    return int((cn >= 3).sum())


def _L_residual(xy, yaw):
    """Median jarak titik ke DUA sisi terdekat-sensor dari persegi panjang berorientasi `yaw` (bentuk-L; garis lurus = kasus
    khusus). Sensor di (0,0). Kecil = titik membentuk permukaan yg searah yaw; besar = tidak cocok."""
    c, s = np.cos(yaw), np.sin(yaw)
    u = xy[:, 0] * c + xy[:, 1] * s; v = -xy[:, 0] * s + xy[:, 1] * c
    ua = u.min() if 0.0 < u.mean() else u.max()      # sisi yg menghadap sensor (sensor di u=0)
    va = v.min() if 0.0 < v.mean() else v.max()
    return float(np.median(np.minimum(np.abs(u - ua), np.abs(v - va))))


YAW_EV_AXIS_TOL_DEG = 15.0         # v25: sumbu utama titik harus sejajar sisi PANJANG box (yaw) dlm toleransi ini (mod 180)


def _lidar_yaw_evidence(xy, yaw_c, yaw_alt, canon_w=None):
    """xy: (n,2) titik kandidat badan mobil (kerangka LiDAR). yaw_c/yaw_alt: yaw box terpilih/cermin (rad, kerangka LiDAR).
    Bukti ada bila titik membentuk permukaan (garis atau L) yg searah yaw terpilih DAN jelas lebih buruk utk hipotesis cermin."""
    out = dict(ok=False, n=int(len(xy)), extent=None, axis_deg=None, err_deg=None, alt_err_deg=None, axis_dev_deg=None, along_m=None, reason=None)
    if len(xy) < YAW_EV_MIN_PTS:
        out["reason"] = "titik_kurang"
        return out
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T)); ax = v[:, 1]
    p1 = c @ ax
    e1 = float(np.percentile(p1, 97.5) - np.percentile(p1, 2.5))
    rc = _L_residual(xy, yaw_c); ra = _L_residual(xy, yaw_alt) if yaw_alt is not None else None
    out.update(extent=round(e1, 2), axis_deg=round(float(np.degrees(np.arctan2(ax[1], ax[0]) % np.pi)), 1),
               err_deg=round(rc, 3), alt_err_deg=(round(ra, 3) if ra is not None else None))   # (nama field lama; satuan kini meter)
    out["ok"] = bool(e1 >= YAW_EV_MIN_EXTENT_M and rc <= YAW_EV_MAX_RESID_M and (ra is None or ra - rc >= YAW_EV_ALT_MARGIN_M))
    # v25 -- AKAR MASALAH track 44 (frame 979/983) & track 30 (frame 509): bentuk-L/garis bisa 'cocok' dgn yaw yg SEBENARNYA
    # tegak lurus atau miring ke garis titik (jarak ke 2 sisi kecil tapi sumbu titik tidak searah sisi mana pun) -> jadi jangkar -> disalin
    # ke puluhan frame. Dua syarat tambahan, keduanya geometri murni (tanpa tuning per frame/rute):
    #  (a) sumbu utama titik harus SEJAJAR sisi panjang box (yaw) -- garis titik = sisi samping mobil;
    #  (b) sebaran titik sepanjang yaw harus > lebar mobil + toleransi. Garis sependek lebar mobil bisa saja bagian BELAKANG mobil
    #      (sisi pendek) dan yaw-nya tertukar 90 deg; hanya sisi yg lebih panjang dari lebar yg membuktikan arah sisi panjang.
    axis_dev = abs((out["axis_deg"] - float(np.degrees(yaw_c)) + 90.0) % 180.0 - 90.0)
    ux = c @ np.array([np.cos(yaw_c), np.sin(yaw_c)])
    along = float(np.percentile(ux, 97.5) - np.percentile(ux, 2.5))
    out["axis_dev_deg"] = round(float(axis_dev), 1); out["along_m"] = round(along, 2)
    if out["ok"] and axis_dev > YAW_EV_AXIS_TOL_DEG:
        out["ok"] = False; out["reason"] = "sumbu_titik_tidak_sejajar_yaw"
    if out["ok"] and canon_w is not None and along < canon_w + YAW_EV_ACROSS_TOL_M:
        out["ok"] = False; out["reason"] = "sisi_panjang_tidak_terlihat"
    return out


# REVISI KE-23 -- kelas titik yg boleh jadi kandidat BADAN MOBIL. Cek nyata frame 1098 (mobil 12 m): dari ~100 titik yg benar2 di mobil,
# hanya 21 berlabel Unknown, 81 berlabel Vegetation (Fase 3 sering melabeli badan mobil sbg Vegetation); dinding berlabel Building
# (std lateral 0.01 m = bidang tegak lurus) TIDAK ikut. Kandidat lama (Unknown, Vehicle) membuang ~80% titik mobil -> n_hit kecil
# -> depth_fallback & yaw tanpa bukti. Vegetation ikut, tapi tetap disaring: di atas tanah lokal, di dalam bbox, jendela kedalaman
# terpadat 1.5 m, dan uji volumetrik fusion. Building TETAP dikecualikan (dinding).
CAR_CAND_CLASS_IDS = (UNKNOWN_CLASS_ID, 2, 4)      # Unknown, Vegetation, Vehicle (ID Fase 3: 0 Unknown,1 Road,2 Vegetation,3 Building)


def _gate_reason(rec, img_wh):
    """Kembalikan alasan penolakan (str) atau None kalau deteksi lolos. Juga mengisi rec['fit_iou']."""
    det = rec["det"]; cls = det["class"]; conf = float(det["confidence"])
    x1, y1, x2, y2 = det["bbox2d"]; w = max(x2 - x1, 1.0); h = max(y2 - y1, 1.0)
    fit, fus = rec["fit"], rec["fus"]
    rng = float(fit["corners_cam"][:4, 2].mean())
    unconfirmed = rec["track_key"] is None
    n_hit, n_cand = int(fus.get("n_hit", 0)), int(fus.get("n_candidates", 0))
    lidar_ok = n_hit >= MIN_FUSION_PTS
    Wimg, Himg = img_wh
    trunc_border = (x1 <= 2, y1 <= 2, x2 >= Wimg - 2, y2 >= Himg - 2)
    trunc = tuple(bool(a or b) for a, b in zip(trunc_border, rec.get("occ_sides", (False,) * 4)))
    fit_iou = _bbox_iou_trunc(fit["uv"], det["bbox2d"], trunc)
    rec["fit_iou"] = float(fit_iou)
    if rng > MAX_RANGE_ANY_M:
        return "terlalu_jauh"
    _allowed = globals().get("ROUTE_ALLOWED_CLASSES", None)
    if _allowed is not None and cls not in _allowed:
        return "kelas_dimatikan_rute"
    if cls in GROUNDED_CLASSES:
        # Objek yg berdiri di tanah: titik dasar bbox harus jatuh di jarak yg SAMA dgn jarak depth objek. Rambu/papan/poster/pantulan
        # (mis. rambu parkir motor di tiang) bentuknya mirip Person/Motorcycle tapi TIDAK menapak tanah di jaraknya -> ditolak.
        gc = fit.get("ground_contact_depth"); dp = (rec.get("objsel") or {}).get("zed_mask_med") or rec.get("depth_prior")
        if gc and dp and dp > 0 and not trunc[3] and abs(gc - dp) / dp > GROUNDED_REL_TOL:
            return "tidak_menapak_tanah(%s: dasar bbox %.1fm vs depth %.1fm)" % (cls, gc, dp)
    if cls in ("Motorcycle", "Bicycle"):
        if conf < (SMALL_VEH_MIN_CONF_UNCONFIRMED if unconfirmed else SMALL_VEH_MIN_CONF):
            return "kendaraan_kecil_conf_rendah" + ("_belum_dikonfirmasi" if unconfirmed else "")
        mh = rec.get("meas_height_m")
        if mh is not None and not (trunc[1] or trunc[3]) and not (SMALL_VEH_H_MIN_M <= mh <= SMALL_VEH_H_MAX_M):
            return "kendaraan_kecil_tinggi_fisik_tidak_wajar(%.2fm)" % mh
        if n_cand < SMALL_VEH_MIN_LIDAR_CAND and rec.get("depth_prior") is None:
            return "kendaraan_kecil_tanpa_dukungan_lidar_depth"
        if rng > SMALL_VEH_MAX_RANGE_M:
            return "kendaraan_kecil_terlalu_jauh"
    if cls == "Person":
        if not globals().get("PERSON_ENABLED", PERSON_ENABLED_DEFAULT):
            return "person_dimatikan"
        if unconfirmed and conf < PERSON_UNCONFIRMED_MIN_CONF:
            return "person_belum_dikonfirmasi_tracker"
        if conf < PERSON_MIN_CONF:
            return "person_conf_rendah"
        if h / w < PERSON_MIN_ASPECT or h < PERSON_MIN_HEIGHT_PX:
            return "person_bentuk_bbox_tidak_wajar"
        mh = rec.get("meas_height_m")
        if mh is not None and not (trunc[1] or trunc[3]) and not (PERSON_H_MIN_M <= mh <= PERSON_H_MAX_M):
            return "person_tinggi_fisik_tidak_wajar(%.2fm)" % mh      # hidran/tiang/tong: terlalu pendek utk orang
        if n_cand < PERSON_MIN_LIDAR_CAND and conf < PERSON_NO_LIDAR_MIN_CONF:
            return "person_tanpa_dukungan_lidar"
        if rng > PERSON_MAX_RANGE_M:
            return "person_terlalu_jauh"
        if fit_iou < MIN_FIT_IOU_PERSON:
            return "person_fit_buruk"
        return None
    if not lidar_ok:
        if rng > MAX_RANGE_NO_LIDAR_M:
            return "jauh_tanpa_lidar"
        if conf < (CAR_MIN_CONF_NO_LIDAR_UNCONFIRMED if unconfirmed else CAR_MIN_CONF_NO_LIDAR):
            return "conf_rendah_tanpa_lidar" + ("_belum_dikonfirmasi" if unconfirmed else "")
        if w * h < MIN_BBOX_AREA_NO_LIDAR_PX:
            return "bbox_terlalu_kecil_tanpa_lidar"
    elif unconfirmed and conf < CAR_MIN_CONF_UNCONFIRMED_LIDAR:
        return "conf_rendah_belum_dikonfirmasi"
    if fit_iou < MIN_FIT_IOU_CAR:
        return "fit_buruk"
    return None


def _rejected_record(rec, reason):
    det = rec["det"]; fit = rec.get("fit"); fus = rec.get("fus") or {}
    return {"class": det["class"], "track_id": rec.get("track_key"), "confidence": round(float(det["confidence"]), 3),
            "bbox2d": [round(float(v), 1) for v in det["bbox2d"]], "reason": reason,
            "range_cam_m": (round(float(fit["corners_cam"][:4, 2].mean()), 1) if fit is not None else None),
            "n_hit": int(fus.get("n_hit", 0)), "n_candidates": int(fus.get("n_candidates", 0)),
            "fit_iou": (round(float(rec["fit_iou"]), 3) if rec.get("fit_iou") is not None else None)}


# =============================================================================
# FASE 3 BARU -- PEMILIH TITIK OBJEK BERBASIS MASK YOLO-seg (menggantikan select_lidar_candidates utk deteksi yg punya mask)
# =============================================================================
# Urutan (sesuai rencana Fase 3 baru):
#   1. titik LiDAR MENTAH (tanpa label kelas) yg terproyeksi ke dalam MASK instance (di-erode OBJSEL_MASK_ERODE_PX piksel)
#   2. buang TANAH: harus >= OBJSEL_GROUND_MARGIN_M di atas garis tanah lokal (sama dgn Revisi 19) dan < tinggi objek + 0.5 m.
#      (Memakai TINGGI di atas tanah, bukan z mentah: sensor miring ~17-20 derajat, z mentah mobil jauh = 3.5-6.5 m.)
#   3. buang DINDING: bidang tegak (garis lurus di BEV) yg panjang, padat, dan menerus. Dideteksi sekali per frame
#      (`build_frame_context`) dgn RANSAC garis 2D pada SEMUA titik non-tanah, lalu titik dalam OBJSEL_WALL_REMOVE_M dari garis dibuang.
#   4. DBSCAN pd titik sisa; pilih klaster yg kedalamannya paling cocok dgn MEDIAN depth ZED di dalam mask
#      (kalau depth kosong: klaster terpadat yg bentuknya bukan garis satu-ring/bidang tipis). Klaster lain dalam
#      OBJSEL_MERGE_DIST_M dari klaster terpilih digabung (badan mobil sering terputus antar-ring).
# Keluaran = format sama dgn select_lidar_candidates (dict idx, uv, z, z_mode, n) + `objsel` (statistik utk diagnosa).
OBJSEL_ENABLED = True
OBJSEL_MASK_ERODE_PX = 3
OBJSEL_GROUND_MARGIN_M = 0.12
OBJSEL_WALL_ENABLED = True
OBJSEL_WALL_MIN_EXTENT_M = 6.0      # garis lurus non-tanah sepanjang ini = dinding (mobil <= ~5.5 m)
OBJSEL_WALL_MIN_INLIERS = 40
OBJSEL_WALL_LINE_THR_M = 0.12
OBJSEL_WALL_REMOVE_M = 0.20
OBJSEL_WALL_MIN_CONTINUITY = 0.65   # fraksi sel 0.5 m sepanjang garis yg terisi titik (dinding menerus; barisan mobil berlubang)
OBJSEL_WALL_MIN_HEIGHT_M = 0.35     # hanya titik yg cukup tinggi di atas bidang tanah dipakai mendeteksi dinding
OBJSEL_WALL_MAX_LINES = 4
OBJSEL_WALL_RANSAC_ITERS = 400
OBJSEL_DBSCAN_EPS_M = {"Car": 0.8, "Motorcycle": 0.6, "Bicycle": 0.6, "Person": 0.5}
OBJSEL_DBSCAN_MIN_SAMPLES = 3
OBJSEL_ZED_TOL_NEAR_M = 2.5         # klaster boleh lebih DEKAT dari ZED paling banyak ini
OBJSEL_ZED_TOL_FAR_ABS_M = 6.0      # ...dan lebih JAUH paling banyak max(ini, frac*zed)
OBJSEL_ZED_TOL_FAR_FRAC = 0.5
OBJSEL_ZED_LIDAR_OFFSET_M = 1.5     # selisih sistematis LiDAR - ZED pada badan mobil (terukur)
OBJSEL_ZED_SIGMA_M = 3.5
OBJSEL_MERGE_DIST_M = 2.0
OBJSEL_LINE_MAX_WIDTH_M = 0.10      # klaster tipis di dua arah = satu garis scan (cincin jalan/pangkal tembok)
OBJSEL_PLANAR_THICK_M = 0.05
OBJSEL_PLANAR_MIN_WIDTH_M = 0.35
OBJSEL_MIN_CLUSTER_PTS = 4
OBJSEL_ANCHOR_MIN_PTS = 8           # klaster >= ini = 'jangkar': bobot suku LiDAR naik, bobot suku depth ZED turun saat lifting box
OBJSEL_LIDAR_W_MULT = 1.0
OBJSEL_DEPTH_W_MULT = 0.3
OBJSEL_NOZED_MIN_PTS = 8            # tanpa depth ZED: klaster minimal segini + harus cocok dgn jarak kontak-tanah bbox
OBJSEL_CONTACT_LO = 0.6
OBJSEL_CONTACT_HI = 1.5
_SEL_LOG = []

_FRAME_CTX = {"wall": None, "wall_lines": [], "nonground": None}
_LAST_OBJSEL_INFO = {}


def _ransac_wall_lines(xy, rng):
    """Garis 2D (BEV) panjang, padat, menerus. Return list (p0, dir, mask_inlier_dalam_xy)."""
    lines = []
    pool = np.arange(len(xy))
    for _ in range(OBJSEL_WALL_MAX_LINES):
        if len(pool) < OBJSEL_WALL_MIN_INLIERS:
            break
        P = xy[pool]
        best = None; best_cnt = 0
        for _it in range(OBJSEL_WALL_RANSAC_ITERS):
            a, b = rng.choice(len(P), 2, replace=False)
            d = P[b] - P[a]; nd = np.linalg.norm(d)
            if nd < 2.0:
                continue
            d = d / nd
            dist = np.abs((P[:, 0] - P[a, 0]) * d[1] - (P[:, 1] - P[a, 1]) * d[0])
            inl = dist < OBJSEL_WALL_LINE_THR_M
            cnt = int(inl.sum())
            if cnt <= best_cnt or cnt < OBJSEL_WALL_MIN_INLIERS:
                continue
            t = (P[inl] - P[a]) @ d
            ext = float(np.percentile(t, 98) - np.percentile(t, 2))
            if ext < OBJSEL_WALL_MIN_EXTENT_M:
                continue
            cells = np.unique(np.floor((t - t.min()) / 0.5).astype(int))
            cont = len(cells) / max(1.0, (t.max() - t.min()) / 0.5 + 1)
            if cont < OBJSEL_WALL_MIN_CONTINUITY:
                continue
            best, best_cnt = (P[a].copy(), d.copy()), cnt
        if best is None:
            break
        p0, d = best
        dist = np.abs((xy[:, 0] - p0[0]) * d[1] - (xy[:, 1] - p0[1]) * d[0])
        lines.append((p0, d))
        pool = pool[np.abs((xy[pool, 0] - p0[0]) * d[1] - (xy[pool, 1] - p0[1]) * d[0]) >= OBJSEL_WALL_LINE_THR_M]
    return lines


def build_frame_context(points_full7, pc_cam, gf):
    """Sekali per frame: deteksi dinding (garis tegak panjang) di BEV dari semua titik non-tanah. Hasil: _FRAME_CTX['wall'] (bool N)."""
    n = len(points_full7)
    wall = np.zeros(n, bool)
    lines = []
    if OBJSEL_WALL_ENABLED:
        r, f, up, O = gf
        hgt = (pc_cam - O) @ up
        # tinggi relatif thd tanah: median tinggi titik ber-z-kamera sama diaproksimasi lewat persentil rendah per bin jarak
        zc = pc_cam[:, 2]
        base = np.zeros(n)
        for lo in np.arange(0.0, max(zc.max(), 1.0) + 2.0, 2.0):
            s = (zc >= lo) & (zc < lo + 2.0)
            if s.sum() > 20:
                base[s] = np.percentile(hgt[s], 15)
        ng = (hgt - base) > OBJSEL_WALL_MIN_HEIGHT_M
        idx = np.nonzero(ng & (zc > 1.0))[0]
        if len(idx) > 50:
            xy = points_full7[idx, :2].astype(float)
            rng = np.random.default_rng(0)
            lines = _ransac_wall_lines(xy, rng)
            xy_all = points_full7[:, :2].astype(float)
            for p0, d in lines:
                dist = np.abs((xy_all[:, 0] - p0[0]) * d[1] - (xy_all[:, 1] - p0[1]) * d[0])
                t = (xy_all - p0) @ d
                tw = (xy[:, :] - p0) @ d
                wall |= (dist < OBJSEL_WALL_REMOVE_M) & (t > np.percentile(tw, 1) - 1.0) & (t < np.percentile(tw, 99) + 1.0)
    _FRAME_CTX["wall"] = wall; _FRAME_CTX["wall_lines"] = lines
    return wall


def _cluster_shape(xyz_c):
    Q = xyz_c - xyz_c.mean(0)
    if len(Q) < 4:
        return np.array([0.0, 0.0, 0.0])
    return np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(Q.T)), 0.0))


def select_object_points(pc_cam, uv_all, det, gf, H_obj, points_full7, zed_depth, cls):
    """Return dict(idx, uv, z, z_mode, n, objsel) atau None."""
    from sklearn.cluster import DBSCAN
    mask = det.get("mask")
    if mask is None:
        return None
    r, f, up, O = gf
    er = cv2.erode(mask.astype(np.uint8), np.ones((2 * OBJSEL_MASK_ERODE_PX + 1,) * 2, np.uint8)) > 0
    if er.sum() < 30:
        er = mask
    u = np.round(uv_all[:, 0]); v = np.round(uv_all[:, 1])
    ok = np.isfinite(u) & np.isfinite(v) & (pc_cam[:, 2] > 1.0) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    m = np.zeros(len(pc_cam), bool)
    ui = u[ok].astype(int); vi = v[ok].astype(int)
    m[np.nonzero(ok)[0]] = er[vi, ui]
    global _LAST_OBJSEL_INFO
    info = {"n_in_mask": int(m.sum()), "n_after_ground": 0, "n_wall_removed": 0, "n_clusters": 0, "chosen": None, "zed_med": None}
    hgt = (pc_cam - O) @ up
    gl = local_ground_line(pc_cam, uv_all, det["bbox2d"], gf)
    if gl is not None:
        hg = gl[0] + gl[1] * pc_cam[:, 2]
        m &= (hgt > hg + OBJSEL_GROUND_MARGIN_M) & (hgt < hg + H_obj + 0.5)
    else:
        m &= (hgt > GROUND_CLEARANCE_M + 0.35) & (hgt < H_obj + 1.0)
    _LAST_OBJSEL_INFO = info
    # Median depth ZED DI DALAM MASK (bukan bbox yg didilasi: utk objek menempel dinding, dilasi bbox menangkap dinding di belakangnya
    # sehingga jarak & tinggi fisik objek kecil (motor/orang) salah). Dipakai gerbang menapak-tanah & tinggi fisik.
    if zed_depth is not None:
        _dm = zed_depth[er]; _dm = _dm[np.isfinite(_dm) & (_dm > 0.3)]
        if len(_dm) >= 15:
            info["zed_mask_med"] = round(float(np.median(_dm)), 2)
    info["n_after_ground"] = int(m.sum())
    wall = _FRAME_CTX.get("wall")
    if wall is not None:
        info["n_wall_removed"] = int((m & wall).sum()); m &= ~wall
    idx = np.nonzero(m)[0]
    if len(idx) < OBJSEL_MIN_CLUSTER_PTS:
        info["reason"] = "titik_kurang_setelah_buang_tanah_dinding"
        return None
    zm = None
    if zed_depth is not None:
        dm = zed_depth[er]; dm = dm[np.isfinite(dm) & (dm > 0.3)]
        if len(dm) >= 50:
            zm = float(np.median(dm)); info["zed_med"] = round(zm, 2)
    X = points_full7[idx, :3].astype(float)
    eps = OBJSEL_DBSCAN_EPS_M.get(cls, 0.8)
    lab = DBSCAN(eps=eps, min_samples=OBJSEL_DBSCAN_MIN_SAMPLES).fit_predict(X)
    cl = []
    for k in sorted(set(lab) - {-1}):
        s = lab == k
        if s.sum() < OBJSEL_MIN_CLUSTER_PTS:
            continue
        ev = _cluster_shape(X[s])
        flat_line = (ev[0] < OBJSEL_PLANAR_THICK_M) and (ev[1] < OBJSEL_LINE_MAX_WIDTH_M) and s.sum() >= 12
        planar = (ev[0] < OBJSEL_PLANAR_THICK_M) and (ev[1] > OBJSEL_PLANAR_MIN_WIDTH_M) and (ev[2] > 3.0) and s.sum() >= 12
        cl.append(dict(k=k, n=int(s.sum()), z=float(np.median(pc_cam[idx[s], 2])), ev=ev, bad=bool(flat_line or planar), c=X[s].mean(0)))
    info["n_clusters"] = len(cl)
    if not cl:
        info["reason"] = "tidak_ada_klaster"
        return None
    good = [c for c in cl if not c["bad"]] or cl
    if zm is not None:
        # LiDAR pada badan mobil terukur ~1-2 m LEBIH JAUH dari median depth ZED di dalam mask (data nyata 1090-1096: ZED 10.9-12.5 m,
        # LiDAR 13.1-14.5 m) -> bandingkan dgn zm + OBJSEL_ZED_LIDAR_OFFSET_M, bukan zm. Skor = jumlah titik x kecocokan kedalaman.
        lo, hi = zm - OBJSEL_ZED_TOL_NEAR_M, zm + max(OBJSEL_ZED_TOL_FAR_ABS_M, OBJSEL_ZED_TOL_FAR_FRAC * zm)
        near = [c for c in good if lo <= c["z"] <= hi]
        pick = max(near, key=lambda c: c["n"] * np.exp(-((c["z"] - (zm + OBJSEL_ZED_LIDAR_OFFSET_M)) / OBJSEL_ZED_SIGMA_M) ** 2)) if near else None
        if pick is None:
            info["reason"] = f"tak_ada_klaster_dalam_[{lo:.1f},{hi:.1f}]m_dari_depth_ZED"
            return None
    else:
        # tanpa depth ZED (mobil jauh): jarak kontak-tanah dari dasar bbox = prior kasar (jitter pitch membuatnya tidak presisi)
        fx_, fy_, cx_, cy_ = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        bb = det["bbox2d"]
        ray = np.array([((bb[0] + bb[2]) / 2 - cx_) / fx_, (bb[3] - cy_) / fy_, 1.0])
        den = up @ ray
        zc = float((up @ O) / den) if den < -1e-6 else None
        cand_ = [c for c in good if c["n"] >= OBJSEL_NOZED_MIN_PTS]
        if zc is not None and 3 < zc < 80:
            cand_ = [c for c in cand_ if OBJSEL_CONTACT_LO * zc <= c["z"] <= OBJSEL_CONTACT_HI * zc + 2.0]
            info["contact_z"] = round(zc, 1)
        if not cand_:
            info["reason"] = "tanpa_depth_ZED_dan_tak_ada_klaster_cukup_yg_cocok_jarak_kontak"
            return None
        pick = max(cand_, key=lambda c: c["n"])
    keep_k = {pick["k"]}
    for c in good:
        if c["k"] != pick["k"] and np.linalg.norm(c["c"] - pick["c"]) <= OBJSEL_MERGE_DIST_M:
            keep_k.add(c["k"])
    sel = idx[np.isin(lab, list(keep_k))]
    info["chosen"] = dict(n=int(len(sel)), z=round(pick["z"], 2), bad=pick["bad"], clusters_merged=len(keep_k))
    _SEL_LOG.append((det.get("track_id"), sel.copy()))
    return dict(idx=sel, uv=uv_all[sel], z=pc_cam[sel, 2], z_mode=float(np.median(pc_cam[sel, 2])), n=int(len(sel)), objsel=info,
                anchored=bool(len(sel) >= OBJSEL_ANCHOR_MIN_PTS), w_mult=OBJSEL_LIDAR_W_MULT)



def associate_and_classify(points_full7, uv, z_cam, idx_valid, detections, depth, K, T_cam_to_lidar,
                           track_yaw_history, track_motion_history, track_position_history,
                           track_last_valid_info, frame_t):
    objects = []
    REJECTED_LAST_FRAME.clear()
    class_col = points_full7[:, 6].copy()
    img_wh = (IMG_W, IMG_H)
    if depth is not None and DEPTH_SCALE != 1.0:
        depth = depth * DEPTH_SCALE   # FASE 3 BARU: koreksi skala depth (default 1.0 = tidak ada koreksi)
    gp = get_ground_plane(depth, points_full7, T_lidar_to_cam, K)
    if gp is None:
        print("  [PERINGATAN] bidang tanah tidak bisa diestimasi (depth & titik Road kosong, belum ada "
              "riwayat) -- semua deteksi frame ini dilewati.")
        return objects, class_col
    gf = ground_frame(gp["n"], gp["h"])
    pc_cam = (T_lidar_to_cam[:3, :3] @ points_full7[:, :3].T).T + T_lidar_to_cam[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv_all = (K @ pc_cam.T).T; uv_all = uv_all[:, :2] / uv_all[:, 2:3]
    build_frame_context(points_full7, pc_cam, gf)      # Fase 3 baru: dinding per frame (sekali)

    # ---------------- Tahap A: bbox -> box 3D RGB (+ fusi LiDAR) ----------------
    pending = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox2d"]
        cls = det["class"]; track_key = det.get("track_id")
        canon = CLASS_CANONICAL_SIZE_LWH.get(cls, (2.0, 2.0, 1.5))
        ref_depth, ref_dil = get_reference_depth_with_dilation(depth, det["bbox2d"]) if depth is not None else (None, None)
        _sel = None; _objsel_info = None
        if OBJSEL_ENABLED and det.get("mask") is not None:
            # FASE 3 BARU: titik objek dari MASK instance -> buang tanah & dinding -> DBSCAN -> klaster yg cocok dgn depth ZED
            _sel = select_object_points(pc_cam, uv_all, det, gf, canon[2], points_full7, depth, cls)
            _objsel_info = dict(_sel["objsel"]) if _sel is not None else dict(_LAST_OBJSEL_INFO)
            lid_prior, n_prior_pts = ((_sel["z_mode"], _sel["n"]) if _sel is not None else (None, 0))
            lid_set = _sel if cls != "Person" else None
        else:
            _ok = (np.isin(points_full7[:, 6].astype(int), (UNKNOWN_CLASS_ID, 4)) if CANDIDATE_LABEL_FILTER
                   else np.ones(len(points_full7), dtype=bool))
            lid_prior, n_prior_pts = _lidar_depth_prior(pc_cam, uv_all, det["bbox2d"], gf, canon[2], ref_depth, point_ok=_ok)
            lid_set = select_lidar_candidates(pc_cam, uv_all, det["bbox2d"], gf, canon[2], point_ok=_ok) if cls != "Person" else None
        depth_prior = lid_prior if lid_prior is not None else ref_depth
        bbox_area = (x2 - x1) * (y2 - y1)
        if depth_prior is None:
            min_conf = MIN_CONF_FOR_DEPTH_FALLBACK.get(cls, DEFAULT_MIN_CONF_FOR_DEPTH_FALLBACK)
            if det["confidence"] < min_conf or bbox_area < MIN_BBOX_AREA_PX_FOR_DEPTH_FALLBACK:
                if VERBOSE_SKIP:
                    print(f'  [SKIP-geometri-saja] {cls} id={track_key} conf={det["confidence"]:.2f} area={bbox_area:.0f}px '
                          f'-- tanpa dukungan LiDAR/depth & bukti lemah, tidak dipaksakan jadi objek')
                REJECTED_LAST_FRAME.append({"class": cls, "track_id": track_key, "confidence": round(float(det["confidence"]), 3),
                                            "bbox2d": [round(float(v), 1) for v in det["bbox2d"]],
                                            "reason": "bukti_lemah_tanpa_depth_dan_lidar", "range_cam_m": None,
                                            "n_hit": 0, "n_candidates": 0, "fit_iou": None})
                continue
        prev = track_yaw_history.get(track_key) if track_key is not None else None
        s_prev = track_position_history.get(("scale", track_key)) if track_key is not None else None
        _dz_obj = (_objsel_info or {}).get("zed_mask_med") or depth_prior
        _meas_h = ((y2 - y1) * float(_dz_obj) / float(K[1, 1])) if _dz_obj else None   # tinggi fisik dari jarak NYATA (LiDAR/depth), bukan dari lifting
        _prev_backed = bool(track_position_history.get(("yaw_ev", track_key), False)) if track_key is not None else False
        pending.append(dict(det=det, canon=canon, track_key=track_key, ref_depth=ref_depth, ref_dil=ref_dil, lid=lid_set, objsel=_objsel_info,
                            meas_height_m=_meas_h, prev_backed=_prev_backed,
                            depth_prior=depth_prior, n_prior_pts=n_prior_pts, prev=prev,
                            s_prior=(float(s_prev) if s_prev is not None else 1.0), neighbor_used=False))

    # Revisi ke-17: deteksi sisi bbox yg tertutup OBJEK LAIN yg lebih dekat (bukan cuma tepi gambar)
    # -- HARUS dihitung dulu dgn bbox+depth_prior SEMUA deteksi di frame ini, sebelum lifting apapun.
    for rec in pending:
        others = [(r["det"]["bbox2d"], r["depth_prior"]) for r in pending if r is not rec]
        occ_sides, vis_frac = _occlusion_sides(rec["det"]["bbox2d"], rec["depth_prior"], others)
        rec["occ_sides"] = occ_sides; rec["visible_frac"] = vis_frac
        rec["is_partial_view"] = bool(vis_frac < PARTIAL_VIEW_FRAC or any(occ_sides))

    recs = []
    for rec in pending:
        track_key = rec["track_key"]
        streak = track_position_history.get(("yaw_streak", track_key), 0) if track_key is not None else 0
        rec["yaw_streak_in"] = streak
        # Revisi ke-18: bobot prior riwayat NAIK seiring streak (makin lama arah konsisten, makin kuat
        # menahan noise per-frame) -- objek partial_view tetap dapat lantai bobot yg lebih tinggi dari
        # awal (LIFT_YAW_HISTORY_W_PARTIAL), lalu tumbuh dgn streak jg.
        base_w = LIFT_YAW_HISTORY_W_PARTIAL if rec["is_partial_view"] else LIFT_YAW_HISTORY_W
        hist_w = min(YAW_STREAK_HIST_W_MAX, base_w + YAW_STREAK_HIST_W_GROWTH * streak)
        # prior 'sejajar-ego' cuma relevan utk track BENAR2 BARU (belum ada riwayat sama sekali) --
        # kalau sudah ada riwayat, prior ini JUSTRU bisa menarik box menjauh dari arah asli yg sudah
        # diketahui saat bukti edge-fit frame ini lemah/ambigu (ini akar gejala 'jadi lurus' yg dilihat
        # user utk objek yg SAMA di frame lain).
        align_w = LIFT_YAW_ALIGN_W if rec["prev"] is None else 0.0
        yaw_priors = [{"psi": 0.0, "w": align_w}]
        if rec["prev"] is not None:
            yaw_priors.append({"psi": rec["prev"], "w": hist_w})
        rec["yaw_priors"] = yaw_priors
        _run_lift_and_fuse(rec, gf, K, depth, pc_cam, points_full7, img_wh)
        # ---- Revisi ke-18b: uji tanding 'yaw dikunci ke riwayat' vs 'fit bebas yg ambigu' ----
        if (rec["track_key"] is not None and rec["prev"] is not None and not rec["yaw_reliable"] and rec["prev_backed"]
                and rec["det"]["class"] != "Person" and rec["yaw_streak_in"] >= TEMPORAL_LOCK_MIN_STREAK):
            L, W, H = rec["canon"]
            fit_lock = lift_bbox_to_3d(rec["det"]["bbox2d"], depth, K, gf, (L, W, H), img_wh,
                                       depth_prior=rec["depth_prior"], fix_yaw=rec["prev"],
                                       s_prior=rec["s_prior"], occ_sides=rec.get("occ_sides", (False,) * 4),
                                       lid=rec.get("lid"))
            fit_lock.pop("mirror", None)
            _cur_total = min(h["total"] for h in rec["hyps"] if h["name"] == rec["chosen_hyp"])
            if fit_lock["loss"] <= _cur_total + 0.02:     # toleransi kecil: riwayat mapan menang saat nyaris seri
                fit_lock, fus_lock, lab_lock = fuse_lidar_with_box(pc_cam, points_full7, fit_lock, fit_lock["dims"], gf, K,
                                                                   rec["det"]["bbox2d"])
                rec.update(fit=fit_lock, fus=fus_lock, lab_idx=lab_lock, yaw_reliable=True, temporal_locked=True,
                           mirror_ambiguous=False)
        recs.append(rec)

    # ---------------- Gerbang validasi (Revisi ke-21): buang deteksi tanpa bukti yg cukup SEBELUM menyentuh state apapun ----------------
    _kept = []
    for rec in recs:
        reason = _gate_reason(rec, img_wh)
        if reason is None:
            _kept.append(rec)
        else:
            REJECTED_LAST_FRAME.append(_rejected_record(rec, reason))
    recs = _kept

    # ---------------- Tahap B: barisan parkir -- tetangga sejenis yg yaw-nya andal memutus seri hipotesis cermin ----------------
    # Revisi ke-19: (1) jarak tetangga 10 m -> 30 m; (2) kalau objek mirror_ambiguous & tetangga andal, pilih hipotesis
    # yg yaw-nya (kerangka LiDAR, mod pi) PALING DEKAT yaw tetangga -- keputusan tegas antara dua hipotesis yg sama-sama
    # sah secara siluet, bukan prior lembut yg bisa kalah oleh noise.
    for rec in recs:
        if (rec["yaw_reliable"] and not rec.get("mirror_ambiguous")) or rec["prev"] is not None or rec["det"]["class"] == "Person":
            continue
        me = rec["fit"]["corners_cam"][:4].mean(0)
        best_nb, best_d = None, NEIGHBOR_ROW_MAX_DIST_M
        for nb in recs:
            if nb is rec or not nb["yaw_reliable"] or nb.get("mirror_ambiguous") or nb["det"]["class"] != rec["det"]["class"]:
                continue
            d = float(np.linalg.norm(nb["fit"]["corners_cam"][:4].mean(0) - me))
            if d < best_d:
                best_nb, best_d = nb, d
        if best_nb is None:
            continue
        nb_yaw = pose_from_lidar_corners(cam_box_to_lidar(best_nb["fit"]["corners_cam"], T_cam_to_lidar))[1]
        if rec.get("mirror_ambiguous") and rec.get("alt_hyp") is not None:
            cur_name = rec["chosen_hyp"]
            cur = next(h for h in rec["hyps"] if h["name"] == cur_name)
            alt = rec["alt_hyp"]
            d_cur = _ang_diff_mod_pi(cur["lidar_yaw"], nb_yaw); d_alt = _ang_diff_mod_pi(alt["lidar_yaw"], nb_yaw)
            if d_alt + np.radians(15) < d_cur:                 # alternatif JELAS lebih sejajar tetangga -> pindah
                _run_lift_and_fuse(rec, gf, K, depth, pc_cam, points_full7, img_wh, force_hyp=alt["name"])
                rec["mirror_ambiguous"] = False; rec["yaw_reliable"] = True; rec["neighbor_used"] = True
                rec["neighbor_decided"] = True
            elif d_cur <= np.radians(NEIGHBOR_AGREE_DEG):      # hipotesis terpilih sudah searah tetangga -> dikonfirmasi
                rec["mirror_ambiguous"] = False; rec["yaw_reliable"] = True; rec["neighbor_used"] = True
                rec["neighbor_decided"] = True
            else:
                rec["neighbor_used"] = True                    # tidak ada yg searah tetangga -> tetap ambigu (ditandai)
        else:
            w_nb = NEIGHBOR_YAW_W_PARTIAL if rec.get("is_partial_view") else NEIGHBOR_YAW_W
            _run_lift_and_fuse(rec, gf, K, depth, pc_cam, points_full7, img_wh,
                               extra_priors=[{"psi": best_nb["fit"]["psi"], "w": w_nb}])
            rec["neighbor_used"] = True

    # ---------------- Tahap C: frame LiDAR + state + objek ----------------
    for rec in recs:
        det, fit, fus, lab_idx = rec["det"], rec["fit"], rec["fus"], rec["lab_idx"]
        cls, track_key = det["class"], rec["track_key"]
        L, W, H = fit["dims"]
        psi = fit["psi"]
        # ---- Revisi ke-22: bukti yaw dari bentuk titik LiDAR (hanya Car; Person tdk punya yaw) ----
        _corners_l = cam_box_to_lidar(fit["corners_cam"], T_cam_to_lidar)
        _yaw_c = pose_from_lidar_corners(_corners_l)[1]
        _alt0 = rec.get("alt_hyp")
        _yaw_alt0 = (_alt0["lidar_yaw"] if _alt0 is not None else None)
        if cls != "Person" and rec.get("lid") is not None:
            ev = _lidar_yaw_evidence(points_full7[rec["lid"]["idx"], :2], _yaw_c, _yaw_alt0, canon_w=rec["canon"][1])
            # Cincin tunggal = bukan bukti bentuk (garis dari SATU ring bisa permukaan mobil ATAU busur cincin tanah yg lolos filter tanah di jarak jauh)
            if ev["ok"] and _n_rings_ge3(points_full7, rec["lid"]["idx"]) < YAW_EV_MIN_RINGS:
                ev["ok"] = False; ev["reason"] = "cincin_tunggal"
            if ev["ok"]:
                # Ambiguitas 90 deg: bentuk-L cocok dgn yaw DAN yaw+90. Sumbu yg benar: sebaran titik ke arah MELINTANG tidak boleh melebihi lebar mobil.
                _xy = points_full7[rec["lid"]["idx"], :2].astype(float)
                _ux = _xy @ np.array([np.cos(_yaw_c), np.sin(_yaw_c)]); _vx = _xy @ np.array([-np.sin(_yaw_c), np.cos(_yaw_c)])
                _across = float(np.percentile(_vx, 97.5) - np.percentile(_vx, 2.5))
                ev["across_m"] = round(_across, 2)
                if _across > rec["canon"][1] + YAW_EV_ACROSS_TOL_M:
                    ev["ok"] = False; ev["reason"] = "sumbu_salah_90deg"
        else:
            ev = dict(ok=False, n=0, extent=None, axis_deg=None, err_deg=None, alt_err_deg=None)
        rec["yaw_ev"] = ev
        if track_key is not None:
            prev = rec["prev"]
            # Revisi ke-18: JANGAN mencemari riwayat yaw dgn bacaan yg SENDIRI sudah ditandai tidak
            # reliable -- sebelumnya riwayat selalu di-EMA 30% ke arah bacaan frame ini APAPUN
            # reliabilitasnya, jadi satu frame yg goyah (bbox kecil/jauh, bukti lemah) ikut mencemari
            # prior yg dipakai frame SELANJUTNYA juga -- inilah salah satu sumber "yaw lompat2" antar
            # frame utk objek yg sama yg dilaporkan user. Riwayat hanya diperbarui kalau bacaan skrg
            # reliable, atau ini bacaan pertama track ini (tak ada pilihan lain selain pakai apa adanya).
            _backed = bool(track_position_history.get(("yaw_ev", track_key), False))
            if ev["ok"] and not _backed:
                # PERTAMA kali ada bukti LiDAR utk track ini -> TIMPA riwayat lama (yg cuma tebakan siluet) & reset streak
                track_yaw_history[track_key] = float(((psi % np.pi + np.pi / 2) % np.pi) - np.pi / 2)
                track_position_history[("yaw_ev", track_key)] = True
                track_position_history[("yaw_streak", track_key)] = 1
            elif prev is not None and (not rec["yaw_reliable"] or (_backed and not ev["ok"])):
                track_yaw_history[track_key] = prev          # frame tanpa bukti TIDAK mengubah riwayat berbukti
            elif prev is not None and not _backed and not ev["ok"]:
                track_yaw_history[track_key] = prev          # tebakan tdk dibiarkan saling menguatkan
            else:
                new_hist = circular_ema_mod_pi(prev % np.pi, psi % np.pi, YAW_EMA_ALPHA) if prev is not None else psi % np.pi
                track_yaw_history[track_key] = float(((new_hist + np.pi / 2) % np.pi) - np.pi / 2)
            # --- streak konsistensi yaw (Revisi ke-18, dipakai Tahap A frame SELANJUTNYA utk menguatkan
            # bobot prior riwayat secara adaptif -- lihat YAW_STREAK_* di Bagian 4) ---
            if rec["yaw_reliable"] and ev["ok"]:
                if prev is not None:
                    dpsi_deg = float(np.degrees(_ang_diff_mod_pi(psi, prev)))
                    if dpsi_deg <= YAW_STREAK_AGREE_DEG:
                        new_streak = rec["yaw_streak_in"] + 1          # konsisten -> makin dipercaya
                    elif dpsi_deg >= YAW_STREAK_BREAK_DEG:
                        new_streak = 0                                  # menyimpang besar scr reliable -> dianggap belokan asli
                    else:
                        new_streak = max(0, rec["yaw_streak_in"] - 1)   # zona abu-abu -> turunkan pelan
                else:
                    new_streak = 1
                track_position_history[("yaw_streak", track_key)] = new_streak
            # bacaan TIDAK reliable: streak dibiarkan apa adanya (tak disentuh) -- tunggu bacaan bagus lagi
            track_position_history[("scale", track_key)] = fit["size_scale"] if rec["s_prior"] == 1.0 and \
                track_position_history.get(("scale", track_key)) is None else \
                0.5 * fit["size_scale"] + 0.5 * rec["s_prior"]
        corners_lidar = cam_box_to_lidar(fit["corners_cam"], T_cam_to_lidar)
        center, yaw_lidar = pose_from_lidar_corners(corners_lidar)
        rel_speed, move_status = update_track_motion(track_key, center[:2], frame_t, track_motion_history)
        record_valid_track_info(track_key, center, yaw_lidar, L, W, H, track_last_valid_info)

        n_in = int(len(lab_idx))
        if n_in:
            oc, cn = np.unique(points_full7[lab_idx, 6].astype(int), return_counts=True)
            crosscheck = {CLASS_NAMES.get(int(c), str(c)): int(n) for c, n in zip(oc, cn)}
            class_col[lab_idx] = OBJECT_CLASS_ID[cls]
        else:
            crosscheck = {}
        source = "lidar" if n_in >= 3 else ("depth_fallback" if rec["ref_depth"] is not None else "ground_plane_prior")
        n_out = int(max(fus["n_candidates"] - n_in, 0))
        alt_h = rec.get("alt_hyp")
        yaw_src = ("person_upright" if cls == "Person" else
                   "temporal_lock" if rec.get("temporal_locked") else
                   "mirror_resolved_neighbor" if rec.get("neighbor_decided") else
                   "rgb_3d_box+neighbor_prior" if rec["neighbor_used"] else "rgb_3d_box")
        objects.append({
            "class": cls, "track_id": track_key, "confidence": det["confidence"], "bbox2d": det["bbox2d"],
            "center_xyz": center.tolist(), "distance_m": float(np.linalg.norm(center[:2])),
            "length_m": round(L, 3), "width_m": round(W, 3), "height_m": round(H, 3),
            "length_m_raw": None, "width_m_raw": None, "height_m_raw": None,
            "num_lidar_points": n_in,
            "num_rejected_depth_outliers": n_out, "num_rejected_height_outliers": 0, "num_box_outliers_removed": n_out,
            # Revisi ke-15: box_validated = LiDAR punya >=MIN_FUSION_PTS titik yg secara geometris
            # menembus box & konsisten satu sama lain (hit_idx sudah dikosongkan fuse_lidar_with_box
            # kalau MAD-nya kelewat liar) -- TERLEPAS dari apakah posisi box ikut digeser atau tidak.
            "box_validated": bool(n_in >= MIN_FUSION_PTS),
            "source": source,
            "status": ("lidar_ok" if n_in >= MIN_FUSION_PTS else "depth_only"),   # Fase 3 baru
            "objsel": rec.get("objsel"),                                           # diagnosa pemilih titik (mask/tanah/dinding/klaster)
            "yaw_rad": round(float(yaw_lidar), 4),                 # sumbu panjang di bidang x-y LiDAR (mod pi)
            "yaw_from_estimate": True, "yaw_source": yaw_src,
            "fase3_class_crosscheck": crosscheck, "is_split_from_single_detection": False,
            "relative_speed_mps": rel_speed, "movement_status": move_status,
            "is_coasted": False, "coast_count": 0, "depth_dilation_factor": rec["ref_dil"],
            # ---- field BARU Revisi ke-14: geometri RGB = sumber kebenaran ----
            "corners_cam": np.round(fit["corners_cam"], 4).tolist(),
            "corners_lidar": np.round(corners_lidar, 4).tolist(),
            "yaw_ground_rad": round(float(psi), 4),                # yaw di bidang tanah (0 = searah sumbu pandang kamera)
            "size_scale": round(fit["size_scale"], 3),
            "yaw_reliable": bool(rec["yaw_reliable"]), "yaw_loss_gap": round(float(rec["gap"]), 4),
            # ---- field BARU Revisi ke-18: transparansi penguatan temporal yaw ----
            "yaw_streak": int(track_position_history.get(("yaw_streak", track_key), 0)) if track_key is not None else 0,
            # ---- field BARU Revisi ke-19: hipotesis cermin (dipakai pass konsensus Bagian 8a-2) ----
            "yaw_mirror_ambiguous": bool(rec.get("mirror_ambiguous", False)),
            "yaw_mirror_gap": round(float(rec.get("mirror_gap", 1.0)), 4),
            "yaw_chosen_hyp": rec.get("chosen_hyp"),
            "yaw_alt_rad": (round(float(alt_h["lidar_yaw"]), 4) if alt_h is not None else None),
            "alt_corners_lidar": (np.round(cam_box_to_lidar(alt_h["fit"]["corners_cam"], T_cam_to_lidar), 4).tolist()
                                  if alt_h is not None else None),
            "alt_center_xyz": (np.round(alt_h["center"], 4).tolist() if alt_h is not None else None),
            "alt_corners_cam": (np.round(alt_h["fit"]["corners_cam"], 4).tolist() if alt_h is not None else None),
            "alt_yaw_ground_rad": (round(float(alt_h["fit"]["psi"]), 4) if alt_h is not None else None),
            "lidar_hyp_cost": round(float(next((h["lidar_cost"] for h in rec.get("hyps", []) if h["name"] == rec.get("chosen_hyp")), 0.0)), 4),
            # ---- field BARU Revisi ke-17: transparansi occlusion (bbox 2D tertutup objek lain) ----
            "is_partial_view": bool(rec["is_partial_view"]), "visible_frac_2d": round(float(rec["visible_frac"]), 3),
            "occluded_sides": [s for s, o in zip(("left", "top", "right", "bottom"), rec["occ_sides"]) if o],
            "fit_iou": (round(float(rec["fit_iou"]), 3) if rec.get("fit_iou") is not None else None),
            # ---- field BARU Revisi ke-22 ----
            "lfit": ({k: round(float(v), 3) for k, v in rec["lfit"].items()} if rec.get("lfit") else None),
            "yaw_evidence": bool(ev["ok"]), "yaw_ev_n": ev["n"], "yaw_ev_extent_m": ev["extent"],
            "yaw_ev_axis_deg": ev["axis_deg"], "yaw_ev_axis_dev_deg": ev.get("axis_dev_deg"), "yaw_ev_along_m": ev.get("along_m"), "yaw_ev_reason": ev.get("reason"), "yaw_ev_err_deg": ev["err_deg"], "yaw_ev_alt_err_deg": ev["alt_err_deg"],
            "meas_height_m": (round(float(rec["meas_height_m"]), 2) if rec.get("meas_height_m") is not None else None),
            "bbox_wh_px": [round(float(det["bbox2d"][2] - det["bbox2d"][0]), 1), round(float(det["bbox2d"][3] - det["bbox2d"][1]), 1)],
            "track_confirmed": bool(track_key is not None),
            "lift_loss": round(fit["loss"], 4), "range_cam_m": round(float(fit["corners_cam"][:4, 2].mean()), 2),
            "ground_contact_depth_m": (round(fit["ground_contact_depth"], 2) if fit["ground_contact_depth"] else None),
            "depth_map_used": fit["used_depth_probe"], "lidar_depth_prior_pts": rec["n_prior_pts"],
            "fusion": fus, "ground_plane_source": gp["source"],
            "ground_plane_height_m": round(gp["h"], 3), "lidar_z_tilt_deg": round(gp["tilt_lidar_z_deg"], 1),
        })
    return objects, class_col


# ====================================================================================================
# [SEL 8 NOTEBOOK] level_heights
# ====================================================================================================
def level_heights(xyz_lidar):
    '''Tinggi tiap titik LiDAR di atas BIDANG TANAH (median riwayat bidang tanah dari depth kamera), bukan z LiDAR mentah.
    z mentah miring ~17 derajat thd tanah, jadi hanya ukuran ini yang valid utk memfilter "0..2 m di atas jalan" di semua jarak.
    Panggil SETELAH associate_and_classify pada frame yang sama (riwayat bidang tanah sudah ter-update).'''
    hist = _GROUND_STATE.get("hist")
    if not hist:
        return None
    n_c = np.median([x[0] for x in hist], axis=0); n_c /= np.linalg.norm(n_c)
    h_c = float(np.median([x[1] for x in hist]))
    pc = (T_lidar_to_cam[:3, :3] @ xyz_lidar.T).T + T_lidar_to_cam[:3, 3]
    return (pc @ n_c + h_c).astype(np.float16)


# ====================================================================================================
# [SEL 16 NOTEBOOK] geometri kotak 3D (tanpa plot)
# ====================================================================================================
def compute_box_corners_3d(center_xyz, yaw_rad, length_m, width_m, height_m):
    """Revisi ke-9. 8 titik sudut box 3D (frame LiDAR) dari center+yaw+dimensi -- box dianggap
    berpusat di `center_xyz` di SEMUA sumbu (x,y,z), bukan center_xyz sebagai alas bawah.
    Urutan sudut: 0-3 alas bawah (searah jarum jam dari kanan-depan), 4-7 atap (sejajar 0-3)."""
    cx, cy, cz = center_xyz
    l2, w2, h2 = length_m / 2.0, width_m / 2.0, height_m / 2.0
    local = np.array([
        [ l2,  w2, -h2], [ l2, -w2, -h2], [-l2, -w2, -h2], [-l2,  w2, -h2],   # alas bawah
        [ l2,  w2,  h2], [ l2, -w2,  h2], [-l2, -w2,  h2], [-l2,  w2,  h2],   # atap
    ])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local @ R.T + np.array([cx, cy, cz])


def get_obj_corners_lidar(obj):
    """Revisi ke-14: pakai 8 corner LiDAR HASIL TRANSFORMASI dari box 3D RGB (persis geometri yg
    sama utk overlay RGB & BEV). Objek lama/interpolasi/coasting yg belum punya corners_lidar
    jatuh ke rekonstruksi (center,yaw,L,W,H) seperti dulu."""
    if obj.get("corners_lidar") is not None:
        return np.asarray(obj["corners_lidar"], dtype=float)
    return compute_box_corners_3d(obj["center_xyz"], obj["yaw_rad"], obj["length_m"], obj["width_m"], obj["height_m"])


def project_box_corners(corners_xyz, T_lidar_to_cam, K):
    """Proyeksikan 8 sudut box ke piksel gambar. Return (8,2) uv, atau None kalau ADA sudut
    yang jatuh di belakang/terlalu dekat kamera (box gak bisa digambar utuh -- drpd gambar
    wireframe yang terdistorsi/salah, mending dilewati)."""
    pts_h = np.hstack([corners_xyz, np.ones((8, 1))])
    cam_pts = (T_lidar_to_cam @ pts_h.T).T[:, :3]
    z = cam_pts[:, 2]
    if np.any(z <= 0.1):
        return None
    uv = (K @ cam_pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def verify_rgb_lidar_consistency(obj, T_lidar_to_cam, K):
    """Revisi ke-14 (uji konsistensi): (1) proyeksikan corners_lidar kembali ke gambar dan bandingkan
    dgn proyeksi corners_cam (harus ~0 px); (2) sudut sumbu panjang poligon BEV (dari corners_lidar)
    harus sama dgn yaw_rad; (3) selisih center. Return dict angka."""
    cc = np.asarray(obj["corners_cam"]); cl = np.asarray(obj["corners_lidar"])
    uv_cam = (K @ cc.T).T; uv_cam = uv_cam[:, :2] / uv_cam[:, 2:3]
    uv_l = project_box_corners(cl, T_lidar_to_cam, K)
    e = cl[0] - cl[3]; e2 = cl[1] - cl[2]
    yaw_poly = float(np.arctan2((e + e2)[1], (e + e2)[0]) % np.pi)
    dyaw = abs(((yaw_poly - obj["yaw_rad"]) + np.pi / 2) % np.pi - np.pi / 2)
    return {"reproj_err_px_max": float(np.abs(uv_l - uv_cam).max()) if uv_l is not None else None,
            "yaw_poly_deg": float(np.degrees(yaw_poly)), "yaw_rad_deg": float(np.degrees(obj["yaw_rad"])),
            "yaw_diff_deg": float(np.degrees(dyaw)),
            "center_diff_m": float(np.linalg.norm(cl.mean(0) - np.asarray(obj["center_xyz"])))}


# ====================================================================================================
# [SEL 17 NOTEBOOK] state tracker & proses 1 frame
# ====================================================================================================
track_yaw_history = {}      # state arah (yaw) per track_id, persisten sepanjang batch (Revisi ke-7)
track_motion_history = {}   # state posisi/waktu per track_id, persisten sepanjang batch (Revisi ke-8c)
track_position_history = {} # state posisi ter-smoothing per track_id, persisten sepanjang batch (Revisi ke-9)
track_last_valid_info = {}  # state posisi/yaw valid terakhir per track_id, buat coasting (Revisi ke-10)


def process_frame_fase3(frame_idx, rows, base_dir, save_preview=False):
    row = next((r for r in rows if int(r["idx"]) == frame_idx), None)
    if row is None:
        return None
    rgb_path   = resolve(base_dir, row.get("rgb_data"), "rgb")
    depth_path = resolve(base_dir, row.get("depth_data"), "depth")
    points7 = load_lidar_points(row)
    if points7 is None or len(points7) == 0 or not (rgb_path and rgb_path.exists() and depth_path and depth_path.exists()):
        return None
    depth = np.load(depth_path)

    detections = track_objects_2d(rgb_path)
    uv, z_cam, idx_valid = project_lidar_to_image(points7[:, :3], T_lidar_to_cam, K)
    frame_t = get_frame_time(row, frame_idx)
    objects, class_col = associate_and_classify(
        points7, uv, z_cam, idx_valid, detections, depth, K, T_cam_to_lidar,
        track_yaw_history, track_motion_history, track_position_history, track_last_valid_info, frame_t
    )

    out_points = points7.copy()
    out_points[:, 6] = class_col                      # 0 = belum disegmentasi, 4/5 = titik objek
    np.save(POINTS_DIR / f"{frame_idx:06d}.npy", out_points)
    h = level_heights(points7[:, :3])
    if h is not None:
        np.save(HEIGHT_DIR / f"{frame_idx:06d}.npy", h)

    record = {
        "frame_idx":         frame_idx,
        "num_points":        int(len(points7)),
        "num_points_object": int(np.isin(class_col.astype(int), (4, 5)).sum()),
        "depth_scale":       float(DEPTH_SCALE),
        "num_detections_2d": len(detections),
        "num_objects":       len(objects),
        "objects":           objects,
        "rejected":          [dict(r) for r in REJECTED_LAST_FRAME],
    }
    with open(OBJ_DIR / f"{frame_idx:06d}.json", "w") as f:
        json.dump(record, f, indent=2)

    return record


# ====================================================================================================
# [SEL 20 NOTEBOOK] 8a-0 penyambungan track
# ====================================================================================================
import numpy as np
if "load_object_record" not in globals():
    def load_object_record(obj_dir, frame_idx):
        with open(obj_dir / f"{frame_idx:06d}.json") as f:
            return json.load(f)
if "refresh_results_from_disk" not in globals():
    def refresh_results_from_disk(results, obj_dir):
        for i, r in enumerate(results):
            p = obj_dir / f"{r['frame_idx']:06d}.json"
            if p.exists():
                results[i] = load_object_record(obj_dir, r["frame_idx"])

STITCH_MAX_GAP_FRAMES = 45
STITCH_MAX_OVERLAP_FRAMES = 1
STITCH_BASE_DIST_M = 2.0
STITCH_DIST_PER_FRAME_M = 0.25
STITCH_FIT_FRAMES = 12            # jumlah frame ujung/awal track yang dipakai
STITCH_MAX_COST = 1.0
STITCH_CLASSES = ("Car", "Motorcycle", "Bicycle")


def _st_track_table(recs, pos):
    tr = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            tid = o.get("track_id")
            if tid is None or o.get("class") not in STITCH_CLASSES:
                continue
            tr.setdefault(tid, []).append((pos[f], f, oi, o))
    for tid in tr:
        tr[tid].sort(key=lambda t: t[0])
    return tr


def _st_points(items):
    """(t, x, y) posisi tepercaya: sumber lidar bila >= 2 titik, kalau tidak semua (kecuali interpolasi)."""
    allp = [(p, float(o["center_xyz"][0]), float(o["center_xyz"][1]), o.get("source")) for p, _, _, o in items if o.get("source") != "interpolated_temporal"]
    lid = [a for a in allp if a[3] == "lidar"]
    use = lid if len(lid) >= 2 else allp
    return np.array([[a[0], a[1], a[2]] for a in use], float)


def _st_velocity(P):
    """Kecepatan (m per indeks-frame) dari regresi linier posisi; None bila < 4 titik atau rentang waktu < 5 frame."""
    if len(P) < 4 or (P[:, 0].max() - P[:, 0].min()) < 5:
        return None
    return np.array([np.polyfit(P[:, 0], P[:, 1], 1)[0], np.polyfit(P[:, 0], P[:, 2], 1)[0]], float)


def run_track_stitching(results, obj_dir):
    frames = [r["frame_idx"] for r in results]
    pos = {f: i for i, f in enumerate(frames)}
    recs = {f: load_object_record(obj_dir, f) for f in frames if (obj_dir / f"{f:06d}.json").exists()}
    tr = _st_track_table(recs, pos)
    info = {}
    vels = []                                         # (t_tengah, vx, vy) dari track panjang
    for tid, items in tr.items():
        P = _st_points(items); v = _st_velocity(P)
        info[tid] = dict(first=items[0][0], last=items[-1][0], n=len(items), cls=items[0][3].get("class"), v=v,
                         head=_st_points(items[:STITCH_FIT_FRAMES]), tail=_st_points(items[-STITCH_FIT_FRAMES:]))
        if v is not None:
            vels.append((0.5 * (items[0][0] + items[-1][0]), v[0], v[1], items[-1][0] - items[0][0]))

    def prior_velocity(t):
        near = [(abs(tm - t), vx, vy) for tm, vx, vy, span in vels if span >= 8]
        near.sort()
        near = near[:5]
        return np.array([np.median([a[1] for a in near]), np.median([a[2] for a in near])]) if near else np.zeros(2)

    cands = []
    for a, A in info.items():
        for b, B in info.items():
            if a == b or A["cls"] != B["cls"]:
                continue
            gap = B["first"] - A["last"]
            if gap < -STITCH_MAX_OVERLAP_FRAMES or gap > STITCH_MAX_GAP_FRAMES:
                continue
            if len(A["tail"]) == 0 or len(B["head"]) == 0:
                continue
            ta, xa, ya = A["tail"][-1]; tb, xb, yb = B["head"][0]
            v = A["v"] if (A["v"] is not None and A["n"] >= 8) else (B["v"] if (B["v"] is not None and B["n"] >= 8) else prior_velocity(0.5 * (ta + tb)))
            dt = tb - ta
            pred = np.array([xa, ya]) + v * dt
            dist = float(np.hypot(pred[0] - xb, pred[1] - yb))
            lim = STITCH_BASE_DIST_M + STITCH_DIST_PER_FRAME_M * max(dt, 0)
            cost = dist / lim
            if cost <= STITCH_MAX_COST:
                cands.append((cost, a, b, gap, dist))
    cands.sort()
    parent = {t: t for t in info}
    members = {t: {t} for t in info}
    frames_of = {t: {it[1] for it in tr[t]} for t in tr}
    def root(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]; t = parent[t]
        return t
    has_next, has_prev = set(), set()
    merges = []
    for cost, a, b, gap, dist in cands:
        if a in has_next or b in has_prev:
            continue
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        fa = set().union(*[frames_of[m] for m in members[ra]]); fb = set().union(*[frames_of[m] for m in members[rb]])
        if fa & fb:
            continue
        parent[rb] = ra; members[ra] |= members[rb]
        has_next.add(a); has_prev.add(b)
        merges.append((a, b, gap, dist, cost))
    # ID kanonik = ID dgn frame terbanyak dalam gabungan (stabil bila dijalankan ulang)
    rename = {}
    groups = {}
    for t in info:
        groups.setdefault(root(t), []).append(t)
    for r, mem in groups.items():
        if len(mem) < 2:
            continue
        canon = max(mem, key=lambda t: (info[t]["n"], -t))
        for t in mem:
            if t != canon:
                rename[t] = canon
    touched = set()
    for f, rec in recs.items():
        for o in rec["objects"]:
            tid = o.get("track_id")
            if tid in rename:
                o.setdefault("track_id_orig", tid); o["stitched_from"] = tid; o["track_id"] = rename[tid]; touched.add(f)
    for f in touched:
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(recs[f], fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    print(f"[Bagian 8a-0] Penyambungan track: {len(info)} track -> {len(info) - len(rename)} | {len(rename)} ID digabung, {len(touched)} frame ditulis ulang")
    for a, b, gap, dist, cost in merges:
        print(f"  #{a} + #{b} -> track #{rename.get(b, b) if b in rename else rename.get(a, a)}  (celah {gap} frame, selisih prediksi {dist:.2f} m, biaya {cost:.2f})")
    return dict(merges=merges, rename=rename)


# ====================================================================================================
# [SEL 21 NOTEBOOK] 8a-1 konfirmasi track
# ====================================================================================================
# =============================================================================
# REVISI KE-21 -- Bagian 8a-1 (BARU): KONFIRMASI TRACK (buang "hantu") -- jalankan SEKALI setelah Bagian 8,
# SEBELUM 8a-2 (konsensus yaw) dan 8b (interpolasi).
# =============================================================================
# Gerbang per-frame (Bagian 4) hanya bisa menilai SATU frame. Di sini kita menilai riwayat tiap track_id (offline, boleh
# melihat masa depan). Aturan (semua ambang bisa disetel di bawah, alasan penolakan masuk JSON `rejected`):
#   * Car ber-ID   : dibuang kalau muncul < TRACK_MIN_FRAMES_CAR frame DAN tak pernah punya dukungan LiDAR tervalidasi,
#                    atau tak pernah ada LiDAR & rata-rata conf < TRACK_MIN_MEAN_CONF_NO_LIDAR.
#   * Person ber-ID: dibuang kecuali muncul >= PERSON_TRACK_MIN_FRAMES frame dan rata-rata conf >= PERSON_TRACK_MIN_MEAN_CONF.
#   * Objek track_id=None (belum dikonfirmasi tracker) yang lolos gerbang per-frame: dipertahankan HANYA kalau ada objek
#     sekelas di +-UNTRACKED_CORROB_MAX_GAP_FRAMES frame yang posisinya masuk akal sama (ByteTrack baru memberi ID pada
#     sighting KEDUA, jadi mobil asli pasti muncul lagi ber-ID; halusinasi sekali-lewat tidak).
# Titik LiDAR yang sempat di-override jadi Vehicle/Person oleh objek yang dibuang DIKEMBALIKAN ke kelas Fase 3 aslinya.
TRACK_MIN_FRAMES_CAR = 3
TRACK_MIN_MEAN_CONF_NO_LIDAR = 0.55
PERSON_TRACK_MIN_FRAMES = 5
PERSON_TRACK_MIN_MEAN_CONF = 0.70
UNTRACKED_CORROB_MAX_GAP_FRAMES = 3
UNTRACKED_CORROB_BASE_DIST_M = 1.0
UNTRACKED_CORROB_DIST_PER_FRAME_M = 0.8
REVERT_BOX_MARGIN_M = 0.45
PERSON_TRACK_H_RANGE_M = (1.15, 2.30)   # Revisi ke-22: median tinggi fisik terukur sepanjang track (hidran/tiang < 1.15 m ditolak)
PERSON_TRACK_MIN_H_SAMPLES = 3
# v27: ambang veto dinaikkan 0.35 -> 0.60 (lihat konstanta di bawah). Catatan v25 di bawah ini ditulis saat track 34 dianggap rambu; ternyata motor parkir asli.
# v25: objek NYATA yg menapak tanah (orang/motor/sepeda) lolos gerbang fisik di hampir semua frame. Ikon/rambu di tiang lolos hanya saat noise
# depth kebetulan cocok (track 34: 13 frame ditolak 'tidak_menapak_tanah'/'tinggi_fisik_tidak_wajar', 11 lolos). Bila >= fraksi ini dari seluruh
# deteksi track ditolak gerbang fisik, SELURUH track dibuang.
GROUNDED_TRACK_VETO_FRAC = 0.60   # v27: 0.35 membuang motor ASLI (track 34, frame 633: ditolak 13 / lolos 11 = 54%). Objek nyata di jarak ~15 m dgn noise depth bisa gagal ~50% frame; ikon/rambu di tiang gagal jauh lebih sering
GROUNDED_TRACK_VETO_MIN_REJ = 3
_PHYS_REJECT_PREFIXES = ("tidak_menapak_tanah", "kendaraan_kecil_tinggi_fisik_tidak_wajar", "person_tinggi_fisik_tidak_wajar")

from scipy.spatial import cKDTree


def load_object_record(obj_dir, frame_idx):
    with open(obj_dir / f"{frame_idx:06d}.json") as f:
        return json.load(f)


def refresh_results_from_disk(results, obj_dir):
    """`results` (list record dict dari Bagian 8) disegarkan IN-PLACE dari JSON di disk, supaya 8a-2/8b/Bagian 9 membaca
    hasil yang SUDAH dikoreksi (bukan salinan lama di memori)."""
    for i, r in enumerate(results):
        p = obj_dir / f"{r['frame_idx']:06d}.json"
        if p.exists():
            results[i] = load_object_record(obj_dir, r["frame_idx"])


def _points_inside_box(xyz, corners, margin_xy=REVERT_BOX_MARGIN_M, margin_z=0.3):
    c = np.asarray(corners, float)
    ctr = c.mean(0)
    e1, e2, e3 = c[0] - c[3], c[0] - c[1], c[4] - c[0]
    out = np.ones(len(xyz), bool)
    for e, m in ((e1, margin_xy), (e2, margin_xy), (e3, margin_z)):
        n = np.linalg.norm(e)
        u = e / max(n, 1e-9)
        out &= np.abs((xyz - ctr) @ u) <= n / 2.0 + m
    return out


def revert_points_for_dropped(frame_idx, dropped_objs, points_dir, fase3_dir=None):
    """Titik di dalam box objek yg dibuang dikembalikan ke class_id 0 (belum disegmentasi)."""
    p4 = points_dir / f"{frame_idx:06d}.npy"
    if not p4.exists():
        return 0
    P4 = np.load(p4)
    sel = np.isin(P4[:, 6].astype(int), (4, 5))
    if not sel.any():
        return 0
    inside = np.zeros(len(P4), bool)
    for o in dropped_objs:
        if o.get("corners_lidar") is not None:
            inside |= _points_inside_box(P4[:, :3], o["corners_lidar"])
    inside &= sel
    if not inside.any():
        return 0
    P4[inside, 6] = 0
    np.save(p4, P4)
    return int(inside.sum())


def run_track_confirmation(results, obj_dir, points_dir, fase3_dir):
    frame_list = [r["frame_idx"] for r in results]
    pos = {f: i for i, f in enumerate(frame_list)}
    recs = {f: load_object_record(obj_dir, f) for f in frame_list if (obj_dir / f"{f:06d}.json").exists()}
    tracks = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("track_id") is not None:
                tracks.setdefault(o["track_id"], []).append((pos[f], f, oi))

    n_rej_phys = {}
    for rec in recs.values():
        for r in rec.get("rejected", []):
            if r.get("track_id") is not None and str(r.get("reason", "")).startswith(_PHYS_REJECT_PREFIXES):
                n_rej_phys[(r["track_id"], r.get("class"))] = n_rej_phys.get((r["track_id"], r.get("class")), 0) + 1

    drop = {}   # (frame, oi) -> alasan
    for tid, items in tracks.items():
        objs = [recs[f]["objects"][oi] for _, f, oi in items]
        cls = objs[0].get("class")
        n = len(items)
        n_lidar = sum(1 for o in objs if o.get("num_lidar_points", 0) >= MIN_FUSION_PTS and o.get("source") != "interpolated_temporal")
        mean_conf = float(np.mean([o.get("confidence", 0) for o in objs]))
        reason = None
        _nr = n_rej_phys.get((tid, cls), 0)
        if cls in GROUNDED_CLASSES and _nr >= GROUNDED_TRACK_VETO_MIN_REJ and _nr / (_nr + n) >= GROUNDED_TRACK_VETO_FRAC:
            reason = f"track_tidak_menapak_konsisten(ditolak={_nr},lolos={n})"
        elif cls == "Person":
            if n < PERSON_TRACK_MIN_FRAMES or mean_conf < PERSON_TRACK_MIN_MEAN_CONF:
                reason = f"track_person_tidak_stabil(n={n},conf={mean_conf:.2f})"
            else:
                hs = [o["meas_height_m"] for o in objs if o.get("meas_height_m") is not None]
                if len(hs) >= PERSON_TRACK_MIN_H_SAMPLES and not (PERSON_TRACK_H_RANGE_M[0] <= float(np.median(hs)) <= PERSON_TRACK_H_RANGE_M[1]):
                    reason = f"track_person_tinggi_tidak_wajar(median={np.median(hs):.2f}m)"
        else:
            if n < TRACK_MIN_FRAMES_CAR and n_lidar == 0:
                reason = f"track_terlalu_pendek(n={n})"
            elif n_lidar == 0 and mean_conf < TRACK_MIN_MEAN_CONF_NO_LIDAR:
                reason = f"track_tanpa_lidar_conf_rendah(conf={mean_conf:.2f})"
        if reason:
            for _, f, oi in items:
                drop[(f, oi)] = reason

    # objek track_id=None: butuh penguat temporal
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("track_id") is not None or (f, oi) in drop:
                continue
            c0 = np.asarray(o["center_xyz"][:2], float)
            found = False
            for dp in range(-UNTRACKED_CORROB_MAX_GAP_FRAMES, UNTRACKED_CORROB_MAX_GAP_FRAMES + 1):
                if dp == 0:
                    continue
                q = pos[f] + dp
                if q < 0 or q >= len(frame_list) or frame_list[q] not in recs:
                    continue
                lim = UNTRACKED_CORROB_BASE_DIST_M + UNTRACKED_CORROB_DIST_PER_FRAME_M * abs(dp)
                for o2 in recs[frame_list[q]]["objects"]:
                    if o2.get("class") == o.get("class") and np.linalg.norm(np.asarray(o2["center_xyz"][:2], float) - c0) <= lim:
                        found = True
                        break
                if found:
                    break
            if not found:
                drop[(f, oi)] = "tidak_ada_penguat_temporal(track_id=None)"

    by_frame = {}
    for (f, oi), reason in drop.items():
        by_frame.setdefault(f, []).append((oi, reason))
    n_reverted = 0
    summary = {}
    for f, lst in by_frame.items():
        rec = recs[f]
        dropped = []
        for oi, reason in sorted(lst, reverse=True):
            o = rec["objects"].pop(oi)
            dropped.append(o)
            rec.setdefault("rejected", []).append({
                "class": o.get("class"), "track_id": o.get("track_id"), "confidence": o.get("confidence"),
                "bbox2d": o.get("bbox2d"), "reason": reason, "range_cam_m": o.get("range_cam_m"),
                "n_hit": o.get("num_lidar_points"), "n_candidates": None, "fit_iou": o.get("fit_iou")})
            key = (o.get("class"), reason.split("(")[0])
            summary[key] = summary.get(key, 0) + 1
        rec["num_objects"] = len(rec["objects"])
        n_reverted += revert_points_for_dropped(f, [o for o in dropped if o.get("num_lidar_points", 0) > 0], points_dir, fase3_dir)
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    print(f"[Bagian 8a-1] Konfirmasi track: {len(drop)} objek dibuang dari {len(by_frame)} frame; "
          f"{n_reverted} titik LiDAR dikembalikan ke class_id 0.")
    for (cls, why), c in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<7s} {why:<45s} {c}")
    return dict(dropped=sorted(drop.items()), summary=summary)


# ====================================================================================================
# [SEL 22 NOTEBOOK] 8a-2 helper yaw (voting legacy dimatikan)
# ====================================================================================================
# =============================================================================
# FASE 3 FINAL (v24) -- Bagian 8a-2: SMOOTHER YAW PER-ID di KERANGKA DUNIA (ego-compensated) + keselarasan barisan (jalankan SEKALI setelah 8a-1,
# SEBELUM 8b)
# =============================================================================
# KENAPA: mobil PARKIR itu statis, jadi yaw-nya (di kerangka ego) hampir konstan antar frame selama ego tidak berbelok.
# Frame dekat/ber-LiDAR memberi yaw yang andal; frame jauh/tanpa LiDAR memberi tebakan siluet (rawan cermin atau "lurus").
# Offline kita boleh memakai frame-frame KUAT di sekitar (+-YAW_CONS_WINDOW_FRAMES = 45 frame ~ 4.5 s @10 Hz) utk
# mengoreksi frame LEMAH pada track yg sama. Frame kuat TIDAK PERNAH diubah. Tiga aksi (urut dari paling aman):
#   A. TUKAR ke hipotesis cermin yg SUDAH tersimpan di frame itu, kalau yaw-nya searah konsensus (butuh >=1 frame kuat).
#   B. PUTAR box (pada sumbu vertikal lewat pusatnya) ke yaw konsensus, kalau frame lemah menyimpang >= YAW_CONS_ROTATE_BEYOND_DEG
#      (butuh >=2 frame kuat yg saling sepakat -> bukan mobil yg berbelok).
#   C. SNAP: frame lemah yg sudah dekat konsensus dirapatkan persis ke konsensus -> yaw stabil, tidak goyang antar frame.
#   D. Barisan parkir (per frame): objek lemah yg bertetangga (<= ROW_ALIGN_MAX_CENTER_DIST_M) dgn objek berjangkar (kuat /
#      sudah dikonsensus) diputar sejajar tetangga itu (dua mobil parkir berdampingan tak mungkin beda >20 derajat).
# Mobil yang benar2 berbelok / ego yang berbelok: frame kuat di jendela tidak sepakat (spread > batas) -> TIDAK ada koreksi.
# CATATAN JUJUR: jumlah titik LiDAR & label kelas titik (.npy) tidak dihitung ulang utk box yg diputar; field `yaw_source`
# ('consensus_mirror_swap' / 'consensus_rotate' / 'consensus_snap' / 'row_align') + `yaw_before_consensus_rad` mencatat jejaknya.
YAW_CONS_WINDOW_FRAMES = 45
YAW_CONS_MIN_STRONG_SWAP = 1
YAW_CONS_MIN_STRONG_ROTATE = 2
YAW_CONS_SPREAD_DEG = 25.0
YAW_CONS_SWAP_ALT_WITHIN_DEG = 20.0
YAW_CONS_SWAP_CUR_BEYOND_DEG = 30.0
YAW_CONS_ROTATE_BEYOND_DEG = 20.0
YAW_CONS_SNAP_WEAK = True
YAW_CONS_SNAP_MIN_DEG = 2.0
YAW_CONS_DECAY_FRAMES = 20.0
YAW_CONS_STRONG_MIN_PTS = MIN_FUSION_PTS
# ---- Revisi ke-22 ----
# 'Frame kuat' kini = frame dgn `yaw_evidence=True` (titik LiDAR membentuk permukaan searah yaw & asimetris thd cermin),
# BUKAN lagi 'yaw_reliable + >=6 titik di dalam box' (itu yg menyebarkan yaw 'lurus' yg salah ke ~20 frame).
YAW_CONS_NEAR_SINGLE_FRAMES = 10      # 1 frame kuat sudah cukup utk memutar kalau jaraknya <= ini
YAW_CONS_TIER2_WINDOW_FRAMES = 90     # jendela lebar: dipakai HANYA bila >=2 frame kuat saling sepakat ketat
YAW_CONS_TIER2_MIN_STRONG = 2
YAW_CONS_TIER2_SPREAD_DEG = 15.0
SIL_MIN_IOU = 0.35                    # uji siluet: box hasil koreksi harus tetap cocok dgn bbox YOLO
SIL_MAX_DROP = 0.12                   # ...dan tidak boleh turun > ini dari IoU sebelum koreksi
SIL_MAX_DROP_STRONG = 0.25           # v24: konsensus yg ditopang bukti bentuk-L (bobot >= YAW_V24_STRONG_CONS_W) boleh menurunkan IoU siluet lebih banyak
SIL_MIN_IOU_STRONG = 0.50
YAW_V24_STRONG_CONS_W = 0.9
SIL_MAX_SHIFT_M = 1.5                 # geser posisi maksimum saat re-fit setelah memutar
ROW_ALIGN_MAX_LATERAL_DIFF_M = 3.5    # barisan yg sama = offset lateral (y LiDAR) berbeda <= ini
ROW_ALIGN_ENABLED = False   # v25: mobil sebaris BOLEH berbeda sudut (parkir miring: track 44 ~0 deg vs track 60 ~50 deg di sisi yg sama); menyamakan yaw tetangga memutar track 44 ke 56 deg di frame 997/1000
ROW_ALIGN_MAX_CENTER_DIST_M = 15.0
ROW_ALIGN_MIN_DEV_DEG = 20.0
_CONSENSUS_SOURCES = ("consensus_mirror_swap", "consensus_rotate", "consensus_snap", "consensus_confirmed", "row_align")


def _wrap_half_pi(x):
    return float((x + np.pi / 2) % np.pi - np.pi / 2)


def _ang_diff_deg_mod180(a_rad, b_rad):
    return abs((np.degrees(a_rad) - np.degrees(b_rad) + 90.0) % 180.0 - 90.0)


def _cam_pts_to_lidar(P):
    return (T_cam_to_lidar[:3, :3] @ np.asarray(P, float).T).T + T_cam_to_lidar[:3, 3]


def _lidar_yaw_of(cl):
    e_len = ((cl[0] - cl[3]) + (cl[1] - cl[2])) / 2.0
    return float(np.arctan2(e_len[1], e_len[0]) % np.pi)


def _rodrigues(P, center, k, ang):
    d = P - center
    c, s = np.cos(ang), np.sin(ang)
    return center + d * c + np.cross(k, d) * s + np.outer(d @ k, k) * (1.0 - c)


def rotate_obj_to_lidar_yaw(o, target_rad):
    """Putar box objek pada sumbu vertikal-tanah (bukan sumbu z LiDAR yg miring ~17 deg) lewat pusatnya sampai yaw di bidang
    x-y LiDAR = target (mod pi). Posisi pusat & ukuran tidak berubah. Mengembalikan sudut putar (rad) di kerangka kamera."""
    cc = np.asarray(o["corners_cam"], float)
    up = cc[4] - cc[0]
    k = up / np.linalg.norm(up)
    ctr = cc.mean(0)
    a = _wrap_half_pi(target_rad - _lidar_yaw_of(_cam_pts_to_lidar(cc)))
    for _ in range(6):
        y = _lidar_yaw_of(_cam_pts_to_lidar(_rodrigues(cc, ctr, k, a)))
        err = _wrap_half_pi(target_rad - y)
        if abs(err) < 1e-5:
            break
        a += err
    c2 = _rodrigues(cc, ctr, k, a)
    cl = _cam_pts_to_lidar(c2)
    if "yaw_before_consensus_rad" not in o:
        o["yaw_before_consensus_rad"] = o["yaw_rad"]
    o["corners_cam"] = np.round(c2, 4).tolist()
    o["corners_lidar"] = np.round(cl, 4).tolist()
    o["center_xyz"] = cl.mean(0).tolist()
    o["yaw_rad"] = round(_lidar_yaw_of(cl), 4)
    o["yaw_ground_rad"] = round(_wrap_half_pi(o.get("yaw_ground_rad", 0.0) - a), 4)   # psi diukur 'maju'->'kanan' = berlawanan arah putar kanan-tangan thd 'atas'
    o["distance_m"] = float(np.hypot(o["center_xyz"][0], o["center_xyz"][1]))
    o["yaw_alt_rad"] = None; o["alt_corners_lidar"] = None; o["alt_center_xyz"] = None
    o["alt_corners_cam"] = None; o["alt_yaw_ground_rad"] = None
    o["yaw_mirror_ambiguous"] = False; o["yaw_reliable"] = True
    return float(a)


_SNAP_KEYS = ("corners_cam", "corners_lidar", "center_xyz", "yaw_rad", "yaw_ground_rad", "distance_m", "yaw_alt_rad",
              "alt_corners_lidar", "alt_center_xyz", "alt_corners_cam", "alt_yaw_ground_rad", "yaw_mirror_ambiguous",
              "yaw_reliable", "yaw_before_consensus_rad")


def _sil_iou(o, corners_cam):
    """IoU bbox proyeksi box 3D thd bbox YOLO; sisi yg terpotong (tepi gambar / tertutup) tidak dihukum."""
    P = np.asarray(corners_cam, float)
    if (P[:, 2] < 0.2).any():
        return 0.0
    uv = (K @ P.T).T; uv = uv[:, :2] / uv[:, 2:3]
    bb = o["bbox2d"]; occ = o.get("occluded_sides") or []
    trunc = [bb[0] <= 2 or "left" in occ, bb[1] <= 2 or "top" in occ, bb[2] >= IMG_W - 2 or "right" in occ, bb[3] >= IMG_H - 2 or "bottom" in occ]
    pb = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    if trunc[0]: pb[0] = max(pb[0], bb[0])
    if trunc[1]: pb[1] = max(pb[1], bb[1])
    if trunc[2]: pb[2] = min(pb[2], bb[2])
    if trunc[3]: pb[3] = min(pb[3], bb[3])
    iw = max(0.0, min(pb[2], bb[2]) - max(pb[0], bb[0])); ih = max(0.0, min(pb[3], bb[3]) - max(pb[1], bb[1]))
    inter = iw * ih
    union = max(pb[2] - pb[0], 0) * max(pb[3] - pb[1], 0) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _set_cam_corners(o, c2):
    cl = _cam_pts_to_lidar(c2)
    o["corners_cam"] = np.round(c2, 4).tolist(); o["corners_lidar"] = np.round(cl, 4).tolist()
    o["center_xyz"] = cl.mean(0).tolist(); o["distance_m"] = float(np.hypot(o["center_xyz"][0], o["center_xyz"][1]))


def _refit_position_after_rotation(o):
    """Setelah box diputar, geser (sepanjang sinar kamera & lateral di bidang tanah) agar siluet kembali cocok dgn bbox YOLO."""
    from scipy.optimize import minimize
    cc = np.asarray(o["corners_cam"], float)
    up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
    ctr = cc.mean(0); d = ctr - up * (ctr @ up); d = d / max(np.linalg.norm(d), 1e-6)   # arah sinar di bidang tanah
    lat = np.cross(up, d); lat = lat / max(np.linalg.norm(lat), 1e-6)
    def f(p):
        ds, dl = np.clip(p, -SIL_MAX_SHIFT_M, SIL_MAX_SHIFT_M)
        return (1.0 - _sil_iou(o, cc + ds * d + dl * lat)) + 0.03 * (ds ** 2 + dl ** 2)
    best = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=0.02, fatol=1e-4, maxiter=120))
                for x0 in ([0.0, 0.0], [0.7, 0.0], [-0.7, 0.0], [0.0, 0.7], [0.0, -0.7])), key=lambda r: r.fun)
    ds, dl = np.clip(best.x, -SIL_MAX_SHIFT_M, SIL_MAX_SHIFT_M)
    _set_cam_corners(o, cc + ds * d + dl * lat)


LREFIT_MIN_PTS = 6
LREFIT_MAX_SHIFT_M = 3.0
LREFIT_REL_SEEN = 0.6


def _object_points_cam(f, o):
    """Titik LiDAR berlabel Vehicle di sekitar box objek (frame f) dlm kerangka kamera; None bila tak ada/terlalu sedikit."""
    p = POINTS_DIR / f"{f:06d}.npy"
    if not p.exists() or o.get("corners_lidar") is None:
        return None
    P = np.load(p)
    lab = np.isin(P[:, 6].astype(int), (4, 5))
    if lab.sum() < LREFIT_MIN_PTS:
        return None
    inside = _points_inside_box(P[:, :3], o["corners_lidar"], margin_xy=0.8) & lab
    if inside.sum() < LREFIT_MIN_PTS:
        return None
    return (T_lidar_to_cam[:3, :3] @ P[inside, :3].T).T + T_lidar_to_cam[:3, 3]


def _lidar_recenter(o, Pc):
    """Setelah box DIPUTAR ke yaw baru: geser di bidang tanah sehingga titik LiDAR objek berada di sisi yg menghadap sensor
    (sisi nyata terlihat = di tepi box; sisi yg tak terlihat = tengah sebaran). Tanpa ini box ikut yaw konsensus tapi lepas dari titiknya."""
    cc = np.asarray(o["corners_cam"], float)
    up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
    hd = cc[0] - cc[3]; Lc = float(np.linalg.norm(hd)); hd = hd / Lc
    sd = cc[0] - cc[1]; Wc = float(np.linalg.norm(sd)); sd = sd / Wc
    s0 = np.asarray(T_lidar_to_cam)[:3, 3]
    q = Pc - s0; u = q @ hd; v = q @ sd
    G = cc[:4].mean(0) - s0; gu, gv = float(G @ hd), float(G @ sd)
    eu = float(np.percentile(u, 95) - np.percentile(u, 5)); ev = float(np.percentile(v, 95) - np.percentile(v, 5))
    rel_u, rel_v = eu / Lc, ev / Wc
    anch_u = not (rel_u >= LREFIT_REL_SEEN and rel_v < LREFIT_REL_SEEN)
    anch_v = not (rel_v >= LREFIT_REL_SEEN and rel_u < LREFIT_REL_SEEN)
    def center(x, dim, anch):
        if not anch:
            return float((np.percentile(x, 5) + np.percentile(x, 95)) / 2.0)
        pos = x.mean() > 0
        return float((np.percentile(x, 5) if pos else np.percentile(x, 95)) + (dim / 2 if pos else -dim / 2))
    cu = center(u, Lc, anch_u); cv = center(v, Wc, anch_v)
    d = (cu - gu) * hd + (cv - gv) * sd
    n = float(np.linalg.norm(d))
    if n > LREFIT_MAX_SHIFT_M:
        d = d * (LREFIT_MAX_SHIFT_M / n)
    _set_cam_corners(o, cc + d)
    o["lidar_recenter_shift_m"] = round(float(np.linalg.norm(d)), 2)


def apply_target_yaw(o, target_rad, source, max_drop=None, min_iou=None, pts_cam=None):
    """Putar ke yaw target, atur ulang posisi (dari titik LiDAR objek bila ada, kalau tidak dari siluet), LALU uji siluet:
    koreksi hanya diterima bila box masih cocok dgn bbox YOLO. True bila diterima."""
    snap = {k: o.get(k) for k in _SNAP_KEYS}
    snap["yaw_source"] = o.get("yaw_source")
    iou0 = _sil_iou(o, o["corners_cam"])
    rotate_obj_to_lidar_yaw(o, target_rad)
    if pts_cam is not None and len(pts_cam) >= LREFIT_MIN_PTS:
        _lidar_recenter(o, pts_cam)
    else:
        _refit_position_after_rotation(o)
    iou1 = _sil_iou(o, o["corners_cam"])
    if iou1 < max(SIL_MIN_IOU if min_iou is None else min_iou, iou0 - (SIL_MAX_DROP if max_drop is None else max_drop)):
        for k, v in snap.items():
            if v is None and k not in o:
                continue
            o[k] = v
        o["yaw_consensus_rejected"] = f"siluet (IoU {iou0:.2f}->{iou1:.2f})"
        return False
    o["yaw_source"] = source
    o["yaw_consensus_iou"] = [round(iou0, 3), round(iou1, 3)]
    return True



# ---- v26: konstanta voting ----
YAW_V26_SIGMA_DEG = 10.0
YAW_V26_LONG_MARGIN_M = 0.6          # garis (p10-p90, dikoreksi 1.25x) >= lebar + ini = pasti sisi PANJANG
YAW_V26_RGB_W = 0.15
YAW_V26_RGB_ALT_W = 0.05
YAW_V26_MIN_SCORE = 0.8
YAW_V26_MIN_RAW = 1.0
YAW_V26_PEAK_RATIO = 1.8
YAW_V26_PEAK_SEP_DEG = 40.0
YAW_V26_OVERRIDE_OWN_SCORE = 2.5
YAW_V26_MIN_SHAPE_PTS = 8


def _obj_lidar_shape(f, o):
    """Arah & panjang garis titik LiDAR objek (kerangka xy LiDAR). phi = sumbu utama (mod pi); l1/l2 = sebaran sepanjang/melintang
    (p10-p90 x1.25: tahan outlier ujung); n titik; syarat >= 2 ring (masing2 >= 3 titik) agar bukan busur cincin tanah tunggal."""
    p = POINTS_DIR / f"{f:06d}.npy"
    if not p.exists() or o.get("corners_lidar") is None:
        return None
    P = np.load(p)
    lab = np.isin(P[:, 6].astype(int), (4, 5))
    if lab.sum() < YAW_V26_MIN_SHAPE_PTS:
        return None
    inside = _points_inside_box(P[:, :3], o["corners_lidar"], margin_xy=0.8) & lab
    if inside.sum() < YAW_V26_MIN_SHAPE_PTS:
        return None
    if P.shape[1] >= 5:
        _, cn = np.unique(P[inside, 4].astype(int), return_counts=True)
        if int((cn >= 3).sum()) < 2:
            return None
    xy = P[inside, :2].astype(float)
    # arah sisi: yaw yg meminimalkan jarak titik ke 2 sisi terdekat-sensor (sensor di 0,0). Bentuk-L -> arah SISI (bukan sumbu PCA yg
    # jatuh di antara kedua sisi). Orientasi phi vs phi+90 sama baiknya utk residu -> sumbu panjang = arah dgn sebaran lebih besar.
    best = None
    for thd in np.arange(0.0, 90.0, 1.0):
        th_ = np.radians(thd)
        hd = np.array([np.cos(th_), np.sin(th_)]); sd = np.array([-hd[1], hd[0]])
        u = xy @ hd; v = xy @ sd
        un = u.min() if u.mean() > 0 else u.max(); vn = v.min() if v.mean() > 0 else v.max()
        res = float(np.median(np.minimum(np.abs(u - un), np.abs(v - vn))))
        if best is None or res < best[0]:
            best = (res, th_, u, v)
    res, th_, u, v = best
    eu = 1.25 * float(np.percentile(u, 90) - np.percentile(u, 10)); ev = 1.25 * float(np.percentile(v, 90) - np.percentile(v, 10))
    if eu >= ev:
        phi, l1, l2 = th_, eu, ev
    else:
        phi, l1, l2 = th_ + np.pi / 2, ev, eu
    return dict(phi=float(phi % np.pi), l1=l1, l2=l2, n=int(inside.sum()), res=res)

def _is_strong_yaw(o):
    return (o.get("class") == "Car" and bool(o.get("yaw_evidence", False))
            and o.get("source") != "interpolated_temporal"
            and o.get("yaw_source") not in ("interpolated_temporal", "temporal_lock") + _CONSENSUS_SOURCES)


def _is_weak_yaw(o):
    return (o.get("class") == "Car" and not _is_strong_yaw(o) and o.get("source") != "interpolated_temporal"
            and o.get("yaw_source") not in _CONSENSUS_SOURCES)


def _weighted_circ_mean_mod_pi(yaws_rad, weights):
    y = np.asarray(yaws_rad, float); w = np.asarray(weights, float)
    z = (w * np.exp(2j * y)).sum() / max(w.sum(), 1e-9)
    mean = (np.angle(z) / 2.0) % np.pi
    spread_deg = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(abs(z), 1e-9)), 0.0)) / 2.0))
    return float(mean), spread_deg


def _swap_to_mirror(o):
    old = dict(yaw_rad=o["yaw_rad"], center_xyz=o["center_xyz"], corners_lidar=o["corners_lidar"],
               corners_cam=o["corners_cam"], yaw_ground_rad=o["yaw_ground_rad"])
    o["yaw_before_consensus_rad"] = o.get("yaw_before_consensus_rad", old["yaw_rad"])
    o["yaw_rad"] = o["yaw_alt_rad"]
    o["center_xyz"] = o["alt_center_xyz"]; o["corners_lidar"] = o["alt_corners_lidar"]
    o["corners_cam"] = o["alt_corners_cam"]; o["yaw_ground_rad"] = o["alt_yaw_ground_rad"]
    o["distance_m"] = float(np.hypot(o["center_xyz"][0], o["center_xyz"][1]))
    o["yaw_alt_rad"] = old["yaw_rad"]; o["alt_center_xyz"] = old["center_xyz"]
    o["alt_corners_lidar"] = old["corners_lidar"]; o["alt_corners_cam"] = old["corners_cam"]
    o["alt_yaw_ground_rad"] = old["yaw_ground_rad"]
    o["yaw_mirror_ambiguous"] = False; o["yaw_reliable"] = True



# ---- v24: smoother yaw per-ID, UNIVERSAL (tanpa tuning per frame) -------------------------------------------------
# Ide (sesuai arahan): satu ID = satu objek fisik. Yaw yang benar diambil dari frame-frame lain pada ID yang sama (+-45 frame, melebar
# bertahap) lalu dipakai di frame yang yaw-nya salah / tanpa titik LiDAR / tanpa depth. Beda dgn v22:
#  1. KERANGKA DUNIA: yaw tiap frame dikoreksi rotasi ego (yaw ego kumulatif dari ICP 2D antar-scan LiDAR statis) -> tetap benar
#     walau ego berbelok. (Translasi ICP tidak dipakai: koridor lurus membuatnya degenerate; yaw tidak.)
#  2. PEMBOBOTAN BERTINGKAT, bukan biner kuat/lemah: bukti bentuk-L (1.0-2.0) > LiDAR tervalidasi & yaw andal (0.35) > RGB andal tanpa LiDAR (0.1).
#     Frame hasil konsensus/interpolasi TIDAK jadi jangkar (cegah umpan balik).
#  3. ESTIMATOR ROBUST: rata-rata melingkar berbobot (mod pi) + penolakan outlier iteratif; jangkar tidak sepakat (spread besar / banyak outlier)
#     -> objek berbelok / ID tertukar -> TIDAK ada koreksi.
#  4. Leave-one-out: frame kuat yg dikalahkan mayoritas jangkar lain (>= YAW_V24_OUTLIER_DEG, bobot mayoritas >= 2x bobotnya) ikut DIKOREKSI.
#  5. Jendela bertingkat: +-45 -> +-120 -> seluruh track bila jangkar track sangat sepakat (objek parkir: yaw dunia konstan).
YAW_V24_WINDOW_FRAMES = 45
YAW_V24_WIDE_FRAMES = 120
YAW_V24_DECAY_FRAMES = 40.0
YAW_V24_MIN_ANCHOR_FRAMES = 2       # v25: satu frame jangkar saja TIDAK boleh memutar puluhan frame (satu jangkar salah menyebar ke seluruh track)
YAW_V24_MIN_WEIGHT = 1.0          # bobot MENTAH minimum (tanpa peluruhan jarak) = >=1 frame berbukti ~30 titik, atau 2 frame berbukti lebih lemah
YAW_V24_OUTLIER_DEG = 25.0
YAW_V24_MAX_SPREAD_DEG = 18.0
YAW_V24_WIDE_MAX_SPREAD_DEG = 10.0
YAW_V24_MAX_OUTLIER_FRAC = 0.4
YAW_V24_FIX_STRONG_DEG = 30.0
YAW_V24_FIX_STRONG_RATIO = 2.0
EGO_ICP_MAX_RANGE_M = 30.0
EGO_ICP_VOXEL_M = 0.25
EGO_ICP_MIN_PTS = 200
EGO_ICP_MAX_DYAW_DEG = 3.0            # per frame; lebih besar dianggap gagal -> 0


def _ego_icp_points(f):
    P = np.load(POINTS_DIR / f"{f:06d}.npy")
    hp = HEIGHT_DIR / f"{f:06d}.npy"
    if not hp.exists():
        return None
    h = np.load(hp).astype(float)
    m = (h > 0.4) & (h < 3.0) & (~np.isin(P[:, 6].astype(int), (4, 5))) & (np.hypot(P[:, 0], P[:, 1]) < EGO_ICP_MAX_RANGE_M)
    xy = P[m, :2].astype(float)
    if len(xy) < EGO_ICP_MIN_PTS:
        return None
    k = np.floor(xy / EGO_ICP_VOXEL_M).astype(int)
    _, i = np.unique(k, axis=0, return_index=True)
    return xy[i]


def _icp2d_yaw(src, dst, iters=40):
    """ICP 2D (trimmed, point-to-point). Return (dyaw_rad, resid_med). p_dst = R(dyaw) p_src + t."""
    from scipy.spatial import cKDTree
    tree = cKDTree(dst); tx = ty = th = 0.0; resid = 9.9
    for _ in range(iters):
        c, s_ = np.cos(th), np.sin(th); R = np.array([[c, -s_], [s_, c]])
        p = src @ R.T + [tx, ty]; d, j = tree.query(p)
        m = (d < np.percentile(d, 70)) & (d < 1.0)
        if m.sum() < 30:
            return 0.0, 9.9
        a = p[m]; b = dst[j[m]]; ca, cb = a.mean(0), b.mean(0)
        U, S, Vt = np.linalg.svd((a - ca).T @ (b - cb)); Rr = Vt.T @ U.T
        if np.linalg.det(Rr) < 0:
            Vt[1] *= -1; Rr = Vt.T @ U.T
        t = cb - Rr @ ca; R2 = Rr @ R
        th = float(np.arctan2(R2[1, 0], R2[0, 0])); tx, ty = (Rr @ np.array([tx, ty]) + t)
        resid = float(np.median(d[m]))
    return th, resid


def estimate_ego_yaw(frames):
    """Yaw ego kumulatif (rad) per frame, relatif frame pertama. Frame tanpa file/gagal ICP -> delta 0. Hasil di-cache ke OUT_DIR/ego_yaw.json."""
    cache = OUT_DIR / "ego_yaw.json"
    if cache.exists():
        try:
            c = json.load(open(cache))
            if all(str(f) in c for f in frames):
                return {f: float(c[str(f)]) for f in frames}
        except Exception:
            pass
    theta = {}; cum = 0.0; prev = None; n_fail = 0
    for f in frames:
        cur = _ego_icp_points(f)
        if prev is not None and cur is not None:
            d, r = _icp2d_yaw(prev, cur)
            if r < 0.25 and abs(np.degrees(d)) <= EGO_ICP_MAX_DYAW_DEG:
                cum += d
            else:
                n_fail += 1
        theta[f] = cum
        prev = cur if cur is not None else prev
    json.dump({str(k): v for k, v in theta.items()}, open(cache, "w"))
    print(f"[ego yaw] {len(frames)} frame; ICP gagal/dilewati {n_fail}; total rotasi ego kumulatif {np.degrees(cum):.2f} deg")
    return theta


def _anchor_weight(o):
    """Bobot sebuah frame sbg jangkar yaw (0 = bukan jangkar)."""
    if o.get("class") != "Car" or o.get("source") == "interpolated_temporal":
        return 0.0
    if o.get("yaw_source") in ("interpolated_temporal", "temporal_lock") + _CONSENSUS_SOURCES:
        return 0.0
    if o.get("yaw_evidence"):
        return float(np.clip(min(o.get("yaw_ev_n", 10), 60) / 30.0, 1.0, 2.0))
    # v24b: HANYA frame berbukti bentuk (>=2 ring, sumbu benar) yg jadi jangkar. Frame 'reliable' hasil RGB-lift BUKAN jangkar:
    # 30 frame lemah searah (bias RGB yg sama) tadinya mengalahkan bukti LiDAR dan menarik yaw seluruh track (track 70 -> ~172 deg).
    return 0.0


def _robust_world_yaw(obs, p):
    """obs: list (pos, yaw_ref_rad, weight). Return (yaw_ref, total_w, spread_deg, n) atau None bila tidak sepakat / kurang bukti."""
    if not obs:
        return None
    y = np.array([a[1] for a in obs]); w0 = np.array([a[2] / (1.0 + abs(a[0] - p) / YAW_V24_DECAY_FRAMES) for a in obs])
    keep = np.ones(len(y), bool)
    for _ in range(4):
        cons, spread = _weighted_circ_mean_mod_pi(y[keep], w0[keep])
        dev = np.array([_ang_diff_deg_mod180(a, cons) for a in y])
        new = dev <= YAW_V24_OUTLIER_DEG
        if new.sum() == 0 or (new == keep).all():
            break
        keep = new
    cons, spread = _weighted_circ_mean_mod_pi(y[keep], w0[keep])
    frac_out = 1.0 - w0[keep].sum() / max(w0.sum(), 1e-9)
    if frac_out > YAW_V24_MAX_OUTLIER_FRAC:
        return None
    return cons, float(sum(a[2] for a, k in zip(obs, keep) if k)), spread, int(keep.sum()), float(max(a[2] for a, k in zip(obs, keep) if k))



def run_yaw_consensus_pass(results, obj_dir, frame_idx_list=None):
    frame_idx_list = frame_idx_list or [r["frame_idx"] for r in results]
    recs = {f: load_object_record(obj_dir, f) for f in frame_idx_list if (obj_dir / f"{f:06d}.json").exists()}
    pos = {f: i for i, f in enumerate(frame_idx_list)}
    theta = estimate_ego_yaw(frame_idx_list)
    tracks = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("track_id") is not None and o.get("class") == "Car":
                tracks.setdefault(o["track_id"], []).append((pos[f], f, oi))
    stats = dict(swapped=[], rotated=[], snapped=[], row_aligned=[], confirmed=[], still_ambiguous=[], sil_rejected=[], fixed_strong=[], skipped_inconsistent=[])

    # ---- A/B/C (v26): VOTING yaw per-ID di kerangka dunia -- LiDAR memberi ARAH garis (mod 90 ambigu), RGB memilih sumbu, histori track menyatukan ----
    # Pelajaran dari frame 1024 (SUV, sisi BELAKANG menghadap sensor): garis titik ~2.5 m LATERAL = sisi pendek mobil, bukan sisi panjang;
    # menganggapnya sisi panjang memutar yaw 90 deg (salah). Garis titik hanya membuktikan arah MOD 90. Sumbu ditentukan oleh:
    #   (a) panjang garis yg tak mungkin sisi pendek (>= lebar + 0.6 m, diukur ROBUST p10-p90) -> memilih sumbu garis;
    #   (b) selain itu: suara RGB-lift & kandidat cermin pada track yg sama; suara LiDAR dibagi dua sumbu (phi dan phi+90).
    # Skor Psi(dunia) = sum suara * Gauss(sudut). Dipakai bila puncak jelas (rasio thd puncak lain >= YAW_V26_PEAK_RATIO) -> universal, tanpa tuning per frame.
    for tid, items in tracks.items():
        items.sort()
        votes = []                         # (pos, f, angle_ref, weight)
        shape_of = {}
        for p, f, oi in items:
            o = recs[f]["objects"][oi]
            if o.get("source") == "interpolated_temporal" or o.get("yaw_source") in _CONSENSUS_SOURCES:
                continue
            sh = _obj_lidar_shape(f, o); shape_of[(f, oi)] = sh
            th = theta[f]
            if sh is not None:
                nw = min(sh["n"], 60) / 30.0
                if sh["l1"] >= o.get("width_m", 1.9) + YAW_V26_LONG_MARGIN_M:
                    votes.append((p, f, (sh["phi"] - th) % np.pi, 1.0 * nw, False, 'L'))
                else:
                    votes.append((p, f, (sh["phi"] - th) % np.pi, 0.5 * nw, True, 'A'))
                    votes.append((p, f, (sh["phi"] + np.pi / 2 - th) % np.pi, 0.5 * nw, True, 'A'))
            if not o.get("lfit"):         # frame hasil fit-L LiDAR: yaw sudah dari titik, jangan dihitung 2x sbg suara RGB
                votes.append((p, f, (o["yaw_rad"] - th) % np.pi, YAW_V26_RGB_W, False, 'R'))
                if o.get("yaw_alt_rad") is not None:
                    votes.append((p, f, (o["yaw_alt_rad"] - th) % np.pi, YAW_V26_RGB_ALT_W, False, 'R'))
        if not votes:
            continue
        va = np.array([v[2] for v in votes]); vw = np.array([v[3] for v in votes]); vp = np.array([v[0] for v in votes]); vf = np.array([v[1] for v in votes]); vamb = np.array([v[4] for v in votes]); vkind = np.array([v[5] for v in votes])
        grid = np.radians(np.arange(0.0, 180.0, 1.0))
        for p, f, oi in items:
            o = recs[f]["objects"][oi]
            if o.get("source") == "interpolated_temporal" or o.get("yaw_source") in _CONSENSUS_SOURCES:
                continue
            res = None; last = None
            for tier_name, win in (("near", YAW_V24_WINDOW_FRAMES), ("wide", YAW_V24_WIDE_FRAMES), ("track", 10 ** 9)):
                m = (vf != f) & (np.abs(vp - p) <= win)
                if m.sum() == 0:
                    continue
                wd = vw[m] / (1.0 + np.abs(vp[m] - p) / YAW_V24_DECAY_FRAMES)
                am = vamb[m]
                d = np.abs((grid[:, None] - va[m][None, :] + np.pi / 2) % np.pi - np.pi / 2)
                G = np.exp(-0.5 * (d / np.radians(YAW_V26_SIGMA_DEG)) ** 2)
                # Pemilihan SUMBU hanya dari suara INFORMATIF (sisi panjang LiDAR jelas + RGB). Suara LiDAR ambigu (phi & phi+90 sama berat)
                # tak membedakan sumbu -> tidak boleh menambah massa puncak; dipakai hanya utk menghaluskan sudut di sekitar puncak terpilih.
                S = (wd[None, ~am] * G[:, ~am]).sum(1)
                k1 = int(S.argmax()); s1 = float(S[k1])
                far = np.abs((grid - grid[k1] + np.pi / 2) % np.pi - np.pi / 2) > np.radians(YAW_V26_PEAK_SEP_DEG)
                s2 = float(S[far].max()) if far.any() else 0.0
                raw = float(vw[m][~am].sum()); last = dict(tier=tier_name, s1=round(s1, 2), s2=round(s2, 2), raw=round(raw, 2), peak_deg=round(float(np.degrees(grid[k1])), 0))
                if s1 >= YAW_V26_MIN_SCORE and s1 >= YAW_V26_PEAK_RATIO * max(s2, 1e-9) and raw >= YAW_V26_MIN_RAW:
                    near = np.abs((va[m] - grid[k1] + np.pi / 2) % np.pi - np.pi / 2) <= np.radians(20.0)
                    z = (wd[near] * np.exp(2j * va[m][near])).sum()
                    ang = float((np.angle(z) / 2.0) % np.pi) if abs(z) > 1e-9 else float(grid[k1])
                    _hasL = bool(((vkind[m] == 'L') & near).any())
                    res = (ang, s1, s2, tier_name, _hasL); break
            if res is None:
                if last is not None:
                    o["yaw_vote_fail"] = last
                continue
            cons_ref, s1, s2, tier, has_L = res
            cons = float((cons_ref + theta[f]) % np.pi)
            d_cur = _ang_diff_deg_mod180(o["yaw_rad"], cons)
            o["yaw_consensus_rad"] = round(cons, 4); o["yaw_consensus_dev_deg"] = round(d_cur, 1); o["yaw_consensus_tier"] = tier
            o["yaw_vote_score"] = round(s1, 2); o["yaw_vote_second"] = round(s2, 2)
            o["yaw_basis"] = "lidar+rgb" if has_L else "rgb_only"
            sh = shape_of.get((f, oi))
            own_long = bool(sh is not None and sh["l1"] >= o.get("width_m", 1.9) + YAW_V26_LONG_MARGIN_M and _ang_diff_deg_mod180(sh["phi"], o["yaw_rad"]) <= 15.0)
            if own_long and d_cur >= YAW_CONS_ROTATE_BEYOND_DEG and s1 < YAW_V26_OVERRIDE_OWN_SCORE:
                continue                               # bukti sendiri (sisi panjang jelas) kecuali konsensus sangat kuat
            alt_ok = o.get("yaw_alt_rad") is not None and o.get("alt_corners_lidar") is not None
            if alt_ok and d_cur >= YAW_CONS_SWAP_CUR_BEYOND_DEG and _ang_diff_deg_mod180(o["yaw_alt_rad"], cons) <= YAW_CONS_SWAP_ALT_WITHIN_DEG:
                before = o["yaw_rad"]
                _swap_to_mirror(o); o["yaw_source"] = "consensus_mirror_swap"
                stats["swapped"].append((tid, f, round(float(np.degrees(before)), 1), round(float(np.degrees(o["yaw_rad"])), 1)))
                if _ang_diff_deg_mod180(o["yaw_rad"], cons) > YAW_CONS_SNAP_MIN_DEG:
                    apply_target_yaw(o, cons, "consensus_mirror_swap", pts_cam=_object_points_cam(f, o))
                continue
            if d_cur >= YAW_CONS_SNAP_MIN_DEG:
                before = o["yaw_rad"]
                kind = "consensus_rotate" if d_cur >= YAW_CONS_ROTATE_BEYOND_DEG else "consensus_snap"
                if apply_target_yaw(o, cons, kind, max_drop=SIL_MAX_DROP_STRONG, min_iou=SIL_MIN_IOU_STRONG,
                                    pts_cam=(_object_points_cam(f, o) if d_cur >= YAW_CONS_ROTATE_BEYOND_DEG else None)):
                    stats["rotated" if kind == "consensus_rotate" else "snapped"].append(
                        (tid, f, round(float(np.degrees(before)), 1), round(float(np.degrees(o["yaw_rad"])), 1)))
                else:
                    stats["sil_rejected"].append((tid, f, o.get("yaw_consensus_rejected")))
                continue
            if d_cur <= YAW_CONS_SWAP_ALT_WITHIN_DEG:
                o["yaw_mirror_ambiguous"] = False; o["yaw_reliable"] = True; o["yaw_source"] = "consensus_confirmed"
                stats["confirmed"].append((tid, f))

    # ---- D: keselarasan barisan parkir per frame (jangkar: frame berbukti / hasil konsensus; harus SATU barisan) ----
    if ROW_ALIGN_ENABLED:
        for f, rec in recs.items():
            cars = [o for o in rec["objects"] if o.get("class") == "Car" and o.get("source") != "interpolated_temporal"]
            anchors = [o for o in cars if _is_strong_yaw(o) or o.get("yaw_source") in _CONSENSUS_SOURCES]
            for o in cars:
                if o in anchors or not _is_weak_yaw(o) or not anchors:
                    continue
                c0 = np.asarray(o["center_xyz"][:2], float)
                cand = [a for a in anchors
                        if np.linalg.norm(np.asarray(a["center_xyz"][:2], float) - c0) <= ROW_ALIGN_MAX_CENTER_DIST_M
                        and abs(a["center_xyz"][1] - c0[1]) <= ROW_ALIGN_MAX_LATERAL_DIFF_M]     # sisi/baris jalan yg sama
                if not cand:
                    continue
                nb = min(cand, key=lambda a: np.linalg.norm(np.asarray(a["center_xyz"][:2], float) - c0))
                dev = _ang_diff_deg_mod180(o["yaw_rad"], nb["yaw_rad"])
                if dev < ROW_ALIGN_MIN_DEV_DEG:
                    if dev < YAW_CONS_SNAP_MIN_DEG * 5:
                        o["yaw_source"] = o.get("yaw_source") if o.get("yaw_source") in _CONSENSUS_SOURCES else "consensus_confirmed"
                    continue
                before = o["yaw_rad"]
                if apply_target_yaw(o, nb["yaw_rad"], "row_align", pts_cam=_object_points_cam(f, o)):
                    o["row_align_with_track"] = nb.get("track_id")
                    stats["row_aligned"].append((o.get("track_id"), f, round(float(np.degrees(before)), 1), round(float(np.degrees(o["yaw_rad"])), 1)))
                else:
                    stats["sil_rejected"].append((o.get("track_id"), f, o.get("yaw_consensus_rejected")))

    # ---- flag yaw_confident + tulis ----
    for f, rec in recs.items():
        for o in rec["objects"]:
            if o.get("class") == "Person":
                o["yaw_confident"] = True
                continue
            src = o.get("yaw_source")
            # Revisi ke-22: confident HANYA bila ada bukti LiDAR pada frame ini, atau yaw hasil konsensus/baris yg lolos uji siluet.
            # v26: 'confident' = bukti LiDAR frame ini, ATAU konsensus yg ditopang sisi panjang LiDAR (basis lidar+rgb). Konsensus yg hanya
            # ditopang RGB (basis rgb_only) = konsisten antar frame tapi TANPA bukti LiDAR -> yaw_track_consistent=True, yaw_confident=False.
            o["yaw_track_consistent"] = bool(src in _CONSENSUS_SOURCES)
            o["yaw_confident"] = bool(o.get("yaw_evidence", False) or (src in _CONSENSUS_SOURCES and o.get("yaw_basis") == "lidar+rgb"))
            if o.get("yaw_mirror_ambiguous"):
                stats["still_ambiguous"].append((o.get("track_id"), f))
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    n_conf = sum(1 for r in results for o in r["objects"] if o.get("class") == "Car" and o.get("yaw_confident"))
    n_car = sum(1 for r in results for o in r["objects"] if o.get("class") == "Car")
    print(f"[Bagian 8a-2 v26] Voting yaw per-ID, kerangka dunia (jendela +-{YAW_V24_WINDOW_FRAMES}/{YAW_V24_WIDE_FRAMES}/track): "
          f"tukar-cermin {len(stats['swapped'])}, putar-ke-konsensus {len(stats['rotated'])}, snap {len(stats['snapped'])}, "
          f"frame-kuat-dikoreksi {len(stats['fixed_strong'])}, sejajar-barisan {len(stats['row_aligned'])}, dikonfirmasi {len(stats['confirmed'])}; "
          f"masih ambigu {len(stats['still_ambiguous'])}, ditolak uji siluet {len(stats['sil_rejected'])}.")
    print(f"             yaw_confident: {n_conf}/{n_car} Car ({100 * n_conf / max(n_car, 1):.1f}%)")
    n_cons = sum(1 for r in results for o in r["objects"] if o.get("class") == "Car" and o.get("yaw_track_consistent"))
    print(f"             yaw konsisten antar frame (voting track, termasuk basis RGB saja): {n_cons}/{n_car} Car ({100 * n_cons / max(n_car, 1):.1f}%)")
    print(f"             track tanpa konsensus karena jangkar tidak sepakat: {len(stats['skipped_inconsistent'])} frame")
    for key in ("swapped", "rotated", "fixed_strong", "row_aligned"):
        for tid, f, a, b in stats[key][:15]:
            print(f"  [{key}] track_id={tid} frame {f}: yaw {a:.0f} -> {b:.0f} deg")
    return stats


# v27: eksekusi voting v26 DIMATIKAN (penyebab yaw goyang antar frame, lihat sel 8a-2b). Sel ini tetap WAJIB dijalankan karena
# fungsi bantunya (rotate_obj_to_lidar_yaw, apply_target_yaw, estimate_ego_yaw, _obj_lidar_shape, ...) dipakai 8a-2b.
RUN_LEGACY_VOTING = False


# ====================================================================================================
# [SEL 24 NOTEBOOK] 8a-1b yaw dari permukaan mask+depth
# ====================================================================================================
# =============================================================================
# v29 -- Bagian 8a-1b: YAW DARI PERMUKAAN MOBIL (mask segmentasi + depth ZED + LiDAR yang DIVERIFIKASI mask). Jalankan SETELAH 8a-1,
# SEBELUM 8a-2b (kunci yaw). Aman dijalankan ulang (permukaan yang sudah ada di disk dipakai ulang; ukur-ulang selalu dijalankan).
# =============================================================================
# KENAPA (jujur): sebelum v29, yaw per frame hanya berasal dari (a) pencarian pada bbox 2D (bbox hampir tak peka thd yaw) dan (b) garis titik
# LiDAR "berlabel kendaraan" di sekitar kotak. Mask YOLO-seg HANYA dipakai memilih titik, tidak pernah dipakai mengukur yaw. Kalau titik terpilih
# tercampur sesuatu yang lurus & sejajar jalan (dinding/tepi jalan/mobil lain), garisnya dibaca "sisi panjang", dan karena sama di banyak frame,
# kunci per-track menganggapnya konsensus -> yaw salah tampil HIJAU. Kunci hanya menyebarkan nilai; ia tidak bisa memperbaiki ukuran yang salah.
#
# CARA BARU (tiga pembaca terpisah, hijau hanya bila DUA yang independen sepakat):
#   1. Segmentasi ulang tiap frame (YOLO-seg, tanpa tracker) -> mask siluet mobil, dicocokkan ke objek lewat IoU bbox.
#   2. Piksel di DALAM mask (dierosi) + peta depth ZED = titik permukaan 3D yang HANYA milik mobil itu (dinding di belakangnya tidak ikut).
#      Skala depth dikalibrasi per objek dgn titik LiDAR di dalam mask. Titik LiDAR dipakai hanya bila kedalamannya cocok dgn depth ZED di
#      pikselnya (titik tembus dari dinding di belakang mobil ditolak).
#   3. Ambil pita tinggi badan (bawah-pintu s/d pinggang jendela), buang titik bagian dalam (kap, atap) dgn kontur terlihat per azimut, lalu
#      L-SHAPE FITTING (Zhang dkk. 2017): cari sudut yg membuat titik jatuh pada 1-2 sisi tegak lurus. Panjang tiap sisi menentukan mana sisi
#      panjang (~4.2 m) / pendek (~1.8 m). Hasilnya diukur DI KERANGKA DUNIA lintas frame oleh sel kunci (8a-2b), bukan dicopy mentah.
# Field per objek: surf_kind ('L','long','short','amb'), surf_phi_long_rad, surf_phi_line_rad, surf_w, surf_ext (m), surf_res (m), surf_n, surf_meta.
SURF_ENABLED         = True
SURF_MODEL_CONF      = 0.25
SURF_MATCH_IOU       = 0.5
SURF_ERODE_PX        = 3
SURF_MIN_MASK_PX     = 150
SURF_MAX_SAMPLES     = 2500
SURF_MAX_DEPTH_M     = 45.0
SURF_DEPTH_BAND_M    = 4.5          # piksel mask dgn kedalaman > median +- ini dibuang (kebocoran tepi mask ke latar)
SURF_Z_BAND          = (0.30, 1.15) # tinggi di atas dasar box (m): badan tegak (bawah pintu -> pinggang jendela)
SURF_CAL_MIN_PAIRS   = 6
SURF_CAL_RATIO_RANGE = (0.6, 1.6)
SURF_LID_TOL_ABS     = 0.5
SURF_LID_TOL_REL     = 0.06
SURF_LID_W           = 4.0          # bobot titik LiDAR terverifikasi relatif thd satu piksel depth ZED
SURF_MAX_RANGE_M     = 32.0         # lebih jauh dari ini permukaan terlalu noisy -> tidak diukur
SURF_MIN_PTS         = 30
SURF_AZ_BIN_DEG      = 0.45
SURF_CONTOUR_MARGIN_M = 0.35
SURF_FACE_TOL_MIN    = 0.20
SURF_FACE_TOL_PER_M  = 0.015
SURF_FACE_TOL_MAX    = 0.45
SURF_FACE_MIN_PTS    = 4
SURF_FACE_MIN_EXT    = 0.7
SURF_LONG_MIN_M      = 3.0          # satu sisi saja terlihat: panjang >= ini -> sisi panjang
SURF_SHORT_MAX_M     = 2.3          # <= ini -> mungkin sisi pendek (atau sisi panjang yang tertutup) -> hanya suara lemah
SURF_L_RATIO         = 1.3          # dua sisi terlihat: rasio panjang minimal agar sumbu panjang jelas
SURF_L_BIG_MIN       = 2.5
SURF_RES_GOOD        = 0.12
SURF_RES_OK          = 0.20

_SURF_MODEL = None


def _sf_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _sf_segment(rgb_path):
    """Segmentasi kendaraan (car/truck/bus) satu gambar -> list dict(bbox2d, mask). Model terpisah dari tracker batch (tidak mengganggu ID)."""
    global _SURF_MODEL
    if _SURF_MODEL is None:
        _SURF_MODEL = YOLO(YOLO_MODEL_NAME)
    r = _SURF_MODEL.predict(str(rgb_path), verbose=False, conf=SURF_MODEL_CONF, classes=[2, 5, 7], retina_masks=True, agnostic_nms=True)[0]
    out = []
    if r.masks is None:
        return out
    m = r.masks.data.cpu().numpy(); b = r.boxes.xyxy.cpu().numpy()
    for i in range(len(b)):
        mk = m[i] > 0.5
        if mk.shape != (IMG_H, IMG_W):
            mk = cv2.resize(mk.astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 0
        out.append(dict(bbox2d=[float(x) for x in b[i]], mask=mk))
    return out


def _sf_box_ground_z(o):
    cl = o.get("corners_lidar")
    if cl is not None:
        return float(np.mean(np.asarray(cl, float)[:4, 2]))
    return float(o["center_xyz"][2]) - float(o.get("height_m", 1.5)) / 2.0


def _sf_extract(mask, depth, o, P_xyz, depth_scale=1.0):
    """Titik permukaan 3D (kerangka LiDAR) dari mask + depth ZED (+ titik LiDAR terverifikasi). None bila tak cukup."""
    er = cv2.erode(mask.astype(np.uint8), np.ones((2 * SURF_ERODE_PX + 1,) * 2, np.uint8)) > 0
    if int(er.sum()) < SURF_MIN_MASK_PX:
        er = mask.astype(bool)
    n_mask = int(er.sum())
    if n_mask < SURF_MIN_MASK_PX:
        return None
    vs, us = np.nonzero(er)
    if len(vs) > SURF_MAX_SAMPLES:
        sel = np.linspace(0, len(vs) - 1, SURF_MAX_SAMPLES).astype(int); vs, us = vs[sel], us[sel]
    d = depth[vs, us].astype(float) * depth_scale
    ok = np.isfinite(d) & (d > 1.0) & (d < SURF_MAX_DEPTH_M)
    if int(ok.sum()) < 40:
        return None
    vs, us, d = vs[ok], us[ok], d[ok]
    md = float(np.median(d)); g = np.abs(d - md) <= SURF_DEPTH_BAND_M
    vs, us, d = vs[g], us[g], d[g]
    # ---- kalibrasi skala depth thd LiDAR (titik LiDAR di dalam mask; rasio median tahan titik tembus) ----
    scale, n_pairs, cal_ok = 1.0, 0, False
    lid = np.zeros((0, 3))
    if P_xyz is not None and len(P_xyz):
        pc = (T_lidar_to_cam[:3, :3] @ P_xyz.T).T + T_lidar_to_cam[:3, 3]
        zc = pc[:, 2]; inz = np.nonzero(zc > 1.0)[0]
        uvw = (K @ pc[inz].T).T; ui = np.round(uvw[:, 0] / uvw[:, 2]).astype(int); vi = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
        inb = (ui >= 0) & (ui < IMG_W) & (vi >= 0) & (vi < IMG_H)
        inz, ui, vi = inz[inb], ui[inb], vi[inb]
        inm = er[vi, ui]; inz, ui, vi = inz[inm], ui[inm], vi[inm]
        zl = zc[inz]; zd = depth[vi, ui].astype(float) * depth_scale
        good = np.isfinite(zd) & (zd > 1.0)
        if good.any():
            ratio = zl[good] / zd[good]
            rr = (ratio > SURF_CAL_RATIO_RANGE[0]) & (ratio < SURF_CAL_RATIO_RANGE[1])
            # LAPISAN TERDEKAT: permukaan mobil = yang paling dekat kamera. Titik LiDAR 'tembus' dari dinding di belakang mobil selalu LEBIH JAUH
            # dari depth ZED di pikselnya (rasio lebih besar) dan bisa lebih banyak -> median semua rasio akan memilih dinding. Maka dipilih klaster
            # rasio TERKECIL yg cukup padat (>= SURF_CAL_MIN_PAIRS titik dan >= 25% pasangan).
            rs = np.sort(ratio[rr])
            if len(rs) >= SURF_CAL_MIN_PAIRS:
                need = max(SURF_CAL_MIN_PAIRS, int(np.ceil(0.25 * len(rs))))
                for i0 in range(len(rs) - need + 1):
                    seg = rs[i0:i0 + need]
                    if seg[-1] - seg[0] <= 0.10 * seg[0]:
                        sel_ = (ratio > seg[0] - 0.05 * seg[0]) & (ratio < seg[-1] + 0.05 * seg[-1]) & rr
                        scale, n_pairs, cal_ok = float(np.median(ratio[sel_])), int(sel_.sum()), True
                        break
            if cal_ok:
                keep = good & (np.abs(zl - zd * scale) <= np.maximum(SURF_LID_TOL_ABS, SURF_LID_TOL_REL * zl))
                lid = P_xyz[inz[keep]]
    d2 = d * scale
    xc = (us - K[0, 2]) * d2 / K[0, 0]; yc = (vs - K[1, 2]) * d2 / K[1, 1]
    Pl = (T_cam_to_lidar[:3, :3] @ np.stack([xc, yc, d2])).T + T_cam_to_lidar[:3, 3]
    z0 = _sf_box_ground_z(o); lo, hi = z0 + SURF_Z_BAND[0], z0 + SURF_Z_BAND[1]
    Pl = Pl[(Pl[:, 2] >= lo) & (Pl[:, 2] <= hi)]
    if len(lid):
        lid = lid[(lid[:, 2] >= lo) & (lid[:, 2] <= hi)]
    return dict(zed=Pl.astype(np.float32), lid=np.asarray(lid, np.float32), scale=float(scale), n_pairs=n_pairs, cal_ok=cal_ok, n_mask=n_mask)


def build_surface_points(results, obj_dir, frames=None, force=False, verbose=True):
    """Segmentasi ulang + ekstraksi permukaan per frame -> OUT_DIR/surface/<frame>.npz (kunci z<i>=ZED, l<i>=LiDAR terverifikasi)."""
    SURF_DIR.mkdir(parents=True, exist_ok=True)
    rows_by = {int(r["idx"]): r for r in rows}
    frames = frames or [r["frame_idx"] for r in results]
    ds = float(globals().get("DEPTH_SCALE", 1.0))
    n_obj = n_ok = n_nomatch = n_frames = n_cached = 0
    for k, f in enumerate(frames):
        jp = obj_dir / f"{f:06d}.json"
        if not jp.exists():
            continue
        rec = load_object_record(obj_dir, f)
        cars = [(oi, o) for oi, o in enumerate(rec["objects"]) if o.get("class") == "Car" and o.get("source") != "interpolated_temporal"
                and o.get("bbox2d") is not None and float(o.get("distance_m") or 0.0) <= SURF_MAX_RANGE_M + 8.0]
        if not cars:
            continue
        out = SURF_DIR / f"{f:06d}.npz"
        if out.exists() and not force:
            n_cached += 1; continue
        row = rows_by.get(f)
        rgb_p = resolve(BASE_DIR, row.get("rgb_data"), "rgb") if row else None
        dep_p = resolve(BASE_DIR, row.get("depth_data"), "depth") if row else None
        pp = POINTS_DIR / f"{f:06d}.npy"
        if rgb_p is None or dep_p is None or not rgb_p.exists() or not dep_p.exists() or not pp.exists():
            continue
        dets = _sf_segment(rgb_p); depth = np.load(dep_p); P = np.load(pp)[:, :3].astype(float)
        arrays = {}; changed = False
        for oi, o in cars:
            n_obj += 1
            best, bi = None, 0.0
            for dt in dets:
                v = _sf_iou(dt["bbox2d"], o["bbox2d"])
                if v > bi:
                    best, bi = dt, v
            if best is None or bi < SURF_MATCH_IOU:
                n_nomatch += 1; continue
            res = _sf_extract(best["mask"], depth, o, P, ds)
            if res is None:
                continue
            arrays[f"z{oi}"] = res["zed"]; arrays[f"l{oi}"] = res["lid"]
            o["surf_meta"] = dict(scale=round(res["scale"], 3), pairs=res["n_pairs"], cal=res["cal_ok"], mask_px=res["n_mask"],
                                  n_zed=int(len(res["zed"])), n_lid=int(len(res["lid"])), iou=round(bi, 2))
            n_ok += 1; changed = True
        np.savez_compressed(out, **arrays)
        if changed:
            with open(jp, "w") as fh:
                json.dump(rec, fh, indent=2)
        n_frames += 1
        if verbose and n_frames % 100 == 0:
            print(f"  ... {n_frames} frame disegmentasi ulang ({k + 1}/{len(frames)})")
    if verbose:
        print(f"[Bagian 8a-1b] Permukaan: {n_frames} frame baru, {n_cached} frame dari cache | objek Car {n_obj}: permukaan OK {n_ok}, "
              f"tak cocok dgn mask {n_nomatch}")
    return dict(n_frames=n_frames, n_cached=n_cached, n_obj=n_obj, n_ok=n_ok, n_nomatch=n_nomatch)


# --------------------------------------------------------------------------------------------------------------------
# Pengukuran yaw dari titik permukaan (fungsi murni; tidak butuh data lain)
# --------------------------------------------------------------------------------------------------------------------
def _sm_wpercentile(x, w, q):
    o = np.argsort(x); xs = x[o]; cw = np.cumsum(w[o]); cw = cw / cw[-1]
    return float(xs[min(int(np.searchsorted(cw, q / 100.0)), len(xs) - 1)])


def _sm_contour(xy, w):
    """Kontur TERLIHAT dari sensor (0,0): per bin azimut hanya titik terdekat (+margin) -> titik kap/atap/kaca di bagian dalam jejak dibuang;
    tiap bin diwakili SATU titik (median kedalaman titik terpilih) supaya derau depth per-piksel merata-rata keluar.
    Return (xy_wakil, bobot_wakil)."""
    r = np.hypot(xy[:, 0], xy[:, 1]); az = np.arctan2(xy[:, 1], xy[:, 0])
    b = np.floor(np.degrees(az) / SURF_AZ_BIN_DEG).astype(int)
    out_xy, out_w = [], []
    for bb in np.unique(b):
        ii = np.nonzero(b == bb)[0]
        if len(ii) < 2:
            continue
        rr = np.sort(r[ii]); r0 = float(np.median(rr[:max(len(rr) // 2, 1)]))
        k = ii[r[ii] <= r0 + SURF_CONTOUR_MARGIN_M]
        if len(k) == 0:
            continue
        rc = float(np.median(r[k])); ac = float(np.average(az[k], weights=w[k]))
        out_xy.append((rc * np.cos(ac), rc * np.sin(ac))); out_w.append(float(np.sqrt(w[k].sum())))
    if not out_xy:
        return np.zeros((0, 2)), np.zeros(0)
    return np.array(out_xy), np.array(out_w)


def _sm_edges(xy, w, th):
    e1 = np.array([np.cos(th), np.sin(th)]); e2 = np.array([-e1[1], e1[0]])
    u = xy @ e1; v = xy @ e2
    un = _sm_wpercentile(u, w, 4.0) if np.average(u, weights=w) > 0 else _sm_wpercentile(u, w, 96.0)
    vn = _sm_wpercentile(v, w, 4.0) if np.average(v, weights=w) > 0 else _sm_wpercentile(v, w, 96.0)
    return e1, e2, u, v, un, vn


def _sm_resid(xy, w, th):
    _, _, u, v, un, vn = _sm_edges(xy, w, th)
    r = np.minimum(np.abs(u - un), np.abs(v - vn))
    o = np.argsort(r); cw = np.cumsum(w[o]); m = cw <= 0.85 * cw[-1]
    m[0] = True
    return float(np.average(r[o][m], weights=w[o][m]))


def surface_measure(xy, w):
    """xy (n,2) titik permukaan di bidang tanah (kerangka LiDAR), w (n,) bobot. Return dict atau None.
    kind: 'L' = dua sisi terlihat & sumbu panjang jelas | 'long' = satu sisi >= SURF_LONG_MIN_M | 'short' = satu sisi pendek
          (bisa juga sisi panjang yang tertutup) | 'amb' = sumbu belum jelas (hanya arah garis, mod 90)."""
    xy = np.asarray(xy, float); w = np.asarray(w, float)
    if len(xy) < SURF_MIN_PTS:
        return None
    cxy, cw = _sm_contour(xy, w)
    if len(cxy) < 12:
        return None
    r_med = float(np.median(np.hypot(cxy[:, 0], cxy[:, 1])))
    if r_med > SURF_MAX_RANGE_M:
        return None
    grid = np.arange(0.0, 90.0, 1.0)
    res = np.array([_sm_resid(cxy, cw, np.radians(t)) for t in grid])
    k = int(res.argmin())
    fine = np.arange(grid[k] - 1.5, grid[k] + 1.51, 0.25)
    rf = np.array([_sm_resid(cxy, cw, np.radians(t)) for t in fine])
    th = np.radians(float(fine[int(rf.argmin())])); rbest = float(rf.min())
    e1, e2, u, v, un, vn = _sm_edges(cxy, cw, th)
    tol = float(np.clip(SURF_FACE_TOL_PER_M * r_med, SURF_FACE_TOL_MIN, SURF_FACE_TOL_MAX))
    mu = np.abs(u - un) <= tol; mv = np.abs(v - vn) <= tol
    def face(m, along):
        if int(m.sum()) < SURF_FACE_MIN_PTS:
            return 0.0, int(m.sum())
        return _sm_wpercentile(along[m], cw[m], 95.0) - _sm_wpercentile(along[m], cw[m], 5.0), int(m.sum())
    ext_u, n_u = face(mu, v)        # sisi dgn u konstan: membentang sepanjang e2 (arah th+90)
    ext_v, n_v = face(mv, u)        # sisi dgn v konstan: membentang sepanjang e1 (arah th)
    pu = ext_u >= SURF_FACE_MIN_EXT; pv = ext_v >= SURF_FACE_MIN_EXT
    ang_u = (th + np.pi / 2) % np.pi; ang_v = th % np.pi
    kind, phi_long, phi_line = "amb", None, float(th % (np.pi / 2))
    if pu and pv:
        big, small, ang_big = (ext_u, ext_v, ang_u) if ext_u >= ext_v else (ext_v, ext_u, ang_v)
        if big >= SURF_L_BIG_MIN and big / max(small, 1e-6) >= SURF_L_RATIO:
            kind, phi_long = "L", float(ang_big)
    elif pu or pv:
        ell, ang = (ext_u, ang_u) if pu else (ext_v, ang_v)
        if ell >= SURF_LONG_MIN_M:
            kind, phi_long = "long", float(ang)
        elif ell <= SURF_SHORT_MAX_M:
            kind, phi_line = "short", float(ang % (np.pi / 2))
    q = 1.0 if rbest <= SURF_RES_GOOD else (0.5 if rbest <= SURF_RES_OK else 0.25)
    nf = float(np.clip(len(xy) / 250.0, 0.25, 2.0)); rf_ = float(np.clip(15.0 / max(r_med, 1.0), 0.4, 1.5))
    kf = 1.0 if kind in ("L", "long") else (0.6 if kind == "short" else 0.4)
    return dict(kind=kind, phi_long=phi_long, phi_line=phi_line, w=float(nf * rf_ * q * kf), ext=(round(ext_u, 2), round(ext_v, 2)),
                res=round(rbest, 3), n=int(len(cxy)), range=round(r_med, 1), theta=float(th))


def run_surface_measurement(results, obj_dir, verbose=True):
    """Baca permukaan di disk -> ukur yaw tiap objek -> tulis field surf_* ke JSON. Boleh diulang tanpa segmentasi ulang."""
    n = Counter(); frames = [r["frame_idx"] for r in results]
    for f in frames:
        jp = obj_dir / f"{f:06d}.json"; sp = SURF_DIR / f"{f:06d}.npz"
        if not jp.exists():
            continue
        rec = load_object_record(obj_dir, f); z = np.load(sp) if sp.exists() else None; changed = False
        for oi, o in enumerate(rec["objects"]):
            for kk in ("surf_kind", "surf_phi_long_rad", "surf_phi_line_rad", "surf_w", "surf_ext", "surf_res", "surf_n"):
                o.pop(kk, None)
            if o.get("class") != "Car" or z is None or f"z{oi}" not in z.files:
                continue
            zed = z[f"z{oi}"][:, :2].astype(float); lid = z[f"l{oi}"][:, :2].astype(float) if f"l{oi}" in z.files else np.zeros((0, 2))
            xy = np.vstack([zed, lid]); w = np.concatenate([np.ones(len(zed)), np.full(len(lid), SURF_LID_W)])
            m = surface_measure(xy, w)
            if m is None:
                n["tanpa_ukuran"] += 1; continue
            cal = bool((o.get("surf_meta") or {}).get("cal", False))
            o["surf_kind"] = m["kind"]; o["surf_phi_long_rad"] = (round(m["phi_long"], 4) if m["phi_long"] is not None else None)
            o["surf_phi_line_rad"] = round(m["phi_line"], 4); o["surf_w"] = round(m["w"] * (1.0 if cal else 0.5), 3)
            o["surf_ext"] = list(m["ext"]); o["surf_res"] = m["res"]; o["surf_n"] = m["n"]
            n[m["kind"]] += 1; changed = True
        with open(jp, "w") as fh:
            json.dump(rec, fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    if verbose:
        print("[Bagian 8a-1b] Ukuran yaw dari permukaan per objek: " + ", ".join(f"{k}={v}" for k, v in n.items()))
        print("   L=dua sisi & sumbu panjang jelas | long=satu sisi panjang | short=sisi pendek/tertutup | amb=arah garis saja")
    return n


from collections import Counter


# ====================================================================================================
# [SEL 26 NOTEBOOK] 8a-2b kunci yaw per-track
# ====================================================================================================
# =============================================================================
# v27/v28 -- Bagian 8a-2b: KUNCI YAW PER-TRACK (offline, kerangka dunia). Pengganti voting v26 (8a-2).
# Jalankan SEKALI setelah 8a-1, SEBELUM 8a-3 dan 8b.
# =============================================================================
# MASALAH v26 (kenapa yaw satu ID beda-beda tiap ~10 frame):
#   1. Suara dibobot PELURUHAN WAKTU 1/(1+jarak/40). Objek parkir yaw DUNIA-nya konstan -- bukti LiDAR di frame 90 sama kuatnya
#      untuk frame 5 maupun frame 85. Peluruhan membuat frame awal (jauh dari jangkar) gagal lolos ambang -> dibiarkan salah.
#   2. Konsensus dihitung ULANG per frame dengan jendela geser (+-45 / +-120 / track) dan leave-one-out -> tiap frame dapat nilai
#      sedikit beda (goyang), dan jendela beda bisa memilih puncak beda.
#   3. Koreksi yang gagal uji siluet DIBATALKAN -> frame itu tetap salah. Padahal siluet dipengaruhi posisi/ukuran (monokular),
#      bukan cuma yaw; membatalkan yaw karena posisi meleset adalah keputusan yang terbalik.
# PERBAIKAN (pola "offboard / track-level refinement": objek statis = SATU box dunia untuk seluruh track, bukan tebakan per frame):
#   A. Estimasi SATU yaw dunia per track dari SEMUA frame sekaligus, tanpa peluruhan waktu, tanpa leave-one-out.
#      Hierarki bukti (bukan jumlah suara): garis sisi-PANJANG LiDAR (>= 2 frame sepakat) > garis sisi-pendek LiDAR (memperhalus sudut,
#      sumbu dipilih dari bukti panjang) > RGB (hanya bila tak ada bukti LiDAR yang cukup).
#   B. Statis vs dinamis diputuskan dari DATA: bukti sepakat & tanpa tren (drift) sepanjang track -> KUNCI (sama di semua frame,
#      di kerangka dunia dikoreksi rotasi ego -> tetap benar saat ego berbelok). Bukti tidak sepakat / ada tren -> DINAMIS:
#      frame berbukti tak disentuh, frame lain diinterpolasi sirkuler di antara jangkar terdekat (bukan dibiarkan salah).
#   C. Koreksi frame TIDAK dibatalkan karena siluet. Yaw dipaksa, posisi disesuaikan ulang (titik LiDAR atau siluet, dipilih yg IoU-nya
#      lebih baik), IoU dicatat (`yaw_lock_iou`, `yaw_lock_low_iou`) sebagai flag kualitas untuk dicek di viewer.
#   D. `yaw_cont_rad`: yaw yang di-unwrap kontinu per track (yaw_rad ada di [0, pi) sehingga 1 deg dan 179 deg = arah sama tapi
#      beda angka; interpolasi/animasi di Flutter yang memakai yaw_rad mentah akan berputar liar di batas itu). Pakai field ini
#      kalau Fase 5/7 menginterpolasi sudut antar-frame.
# Idempoten: bisa dijalankan ulang di hasil yang sudah pernah dikunci (observasi memakai yaw ASLI: yaw_before_lock_rad).
# Catatan jujur: yaw ini modulo 180 deg (box simetris) -- belum menentukan sisi DEPAN vs BELAKANG mobil.

LOCK_SIGMA_DEG        = 10.0   # lebar Gaussian suara sudut
LOCK_INLIER_DEG       = 20.0   # suara dalam +-ini dari puncak = inlier
LOCK_MIN_L_FRAMES     = 2      # >= frame berbukti sisi-panjang yang sepakat (satu jangkar saja TIDAK boleh mengunci seluruh track)
LOCK_MIN_L_WEIGHT     = 1.5    # total bobot inlier minimum (bobot frame = min(n_titik, 60)/30)
LOCK_L_INLIER_FRAC    = 0.6    # inlier harus >= fraksi ini dari seluruh bobot bukti panjang (sisanya = outlier / ID tertukar)
LOCK_PEAK_RATIO       = 1.8    # puncak utama >= rasio x puncak terkuat yang berjarak >= LOCK_PEAK_SEP_DEG
LOCK_PEAK_SEP_DEG     = 40.0
LOCK_MAX_DRIFT_DEG    = 15.0   # tren yaw dunia sepanjang track > ini -> objek berbelok -> DINAMIS (bukan dikunci)
LOCK_DRIFT_MIN_SPAN   = 20     # tren hanya dinilai bila inlier membentang >= ini frame (urutan)
LOCK_RGB_W            = 0.15
LOCK_RGB_ALT_W        = 0.05
LOCK_RGB_REF_DIST_M   = 15.0   # bobot RGB x clip((ref/jarak)^2, 0.15, 1.5): galat lifting monokular tumbuh ~ kuadrat jarak -> frame DEKAT lebih dipercaya
LOCK_RGB_MIN_SCORE    = 0.8
LOCK_RGB_MIN_RAW      = 1.0
LOCK_AXIS_RATIO       = 1.5    # mode 'static_axis': massa suara sumbu pemenang >= rasio x sumbu lain (hanya 2 hipotesis, mod 90)
LOCK_AXIS_MIN_MASS    = 0.5
LOCK_SNAP_MIN_DEG     = 0.5    # selisih < ini: geometri tidak diputar (cukup dicatat)
LOCK_BIG_CHANGE_DEG   = 25.0   # perubahan >= ini ditandai di viewer
LOCK_LOW_IOU          = 0.35
DYN_MAX_GAP_FRAMES    = 40     # mode dinamis: jarak maks ke jangkar (urutan frame)
DYN_MAX_ANCHOR_DIFF_DEG = 30.0 # dua jangkar pengapit harus sepakat sebelum diinterpolasi
LOCK_ENABLED_CLASSES  = ("Car",)

# ---- v29: dua pembaca independen + aturan HIJAU ----
LOCK_USE_RGB_VOTES      = False   # suara RGB-lift = prior (bbox + prior searah jalan), BUKAN pengukuran -> tidak boleh ikut 'konsensus'
LOCK_AGREE_DEG          = 12.0    # permukaan (mask+depth) & garis LiDAR dianggap sepakat bila selisih sumbu <= ini
LOCK_S_STRONG_FRAMES    = 3       # pembaca permukaan sendirian boleh HIJAU bila >= segini frame sepakat ...
LOCK_S_STRONG_SPREAD_DEG = 12.0   # ... dgn sebaran <= ini
LOCK_DYN_CONF_FRAMES    = 10      # mode dinamis: frame hanya HIJAU bila <= ini frame dari jangkar terdekat (di luar itu = ekstrapolasi)
LOCK_SURF_DEV_FLAG_DEG  = 20.0    # frame dgn ukuran permukaan menyimpang > ini dari yaw akhir ditandai (yaw_lock_surf_dev_deg)

# ---- v28: VALIDASI SUMBU dgn siluet RGB. Konsistensi antar-frame TIDAK membuktikan benar: garis titik sisi-pendek yang terkontaminasi
# (tanah/dinding/semak) berulang kali terbaca 'sisi panjang' di banyak frame -> 11 jangkar sepakat, tapi SUMBUNYA SALAH 90 deg.
# Tiap kandidat yaw dunia diuji di frame ber-LiDAR: kotak berukuran kanonik diputar ke kandidat, posisinya hanya boleh bergeser kecil
# (+-LOCK_VAL_SHIFT_M sepanjang sinar, +-0.4 m lateral; jarak sudah dikunci LiDAR), lalu dibandingkan dgn bbox YOLO (IoU toleran pitch).
LOCK_VAL_ENABLED     = False   # v29: validasi siluet-bbox v28 dimatikan (bbox hampir buta thd yaw -> memberi rasa aman palsu); diganti pembaca permukaan
LOCK_VAL_MIN_FRAMES  = 3
LOCK_VAL_MAX_FRAMES  = 40
LOCK_VAL_SHIFT_M     = 1.5
LOCK_VAL_PITCH_PX    = 12.0
LOCK_VAL_MARGIN      = 0.06     # selisih skor minimum agar kandidat lain menggantikan kandidat garis-LiDAR / agar dianggap tidak ambigu
LOCK_VAL_RGB_SEP_DEG = 25.0

if "track_lock" not in _CONSENSUS_SOURCES:
    _CONSENSUS_SOURCES = tuple(_CONSENSUS_SOURCES) + ("track_lock", "track_lock_dynamic")


def _lk_wrap(a):
    """Bungkus selisih sudut (rad) ke [-pi/2, pi/2) -- sudut modulo pi."""
    return (np.asarray(a, float) + np.pi / 2) % np.pi - np.pi / 2


def _lk_circ_mean(ang, w):
    z = (np.asarray(w, float) * np.exp(2j * np.asarray(ang, float))).sum()
    mean = float((np.angle(z) / 2.0) % np.pi)
    R = abs(z) / max(float(np.sum(w)), 1e-9)
    spread = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-9)), 0.0)) / 2.0))
    return mean, spread


def _lk_peak(ang, w):
    """Puncak Gaussian-mixture pada sudut mod pi. Return (pusat_rad, s1, s2, mask_inlier)."""
    ang = np.asarray(ang, float); w = np.asarray(w, float)
    grid = np.radians(np.arange(0.0, 180.0, 1.0))
    d = np.abs(_lk_wrap(grid[:, None] - ang[None, :]))
    S = (w[None, :] * np.exp(-0.5 * (d / np.radians(LOCK_SIGMA_DEG)) ** 2)).sum(1)
    k = int(S.argmax()); s1 = float(S[k])
    far = np.abs(_lk_wrap(grid - grid[k])) > np.radians(LOCK_PEAK_SEP_DEG)
    s2 = float(S[far].max()) if far.any() else 0.0
    c = float(grid[k])
    for _ in range(3):
        m = np.abs(_lk_wrap(ang - c)) <= np.radians(LOCK_INLIER_DEG)
        if not m.any():
            break
        c, _ = _lk_circ_mean(ang[m], w[m])
    m = np.abs(_lk_wrap(ang - c)) <= np.radians(LOCK_INLIER_DEG)
    return c, s1, s2, m


def _lk_original_yaw(o):
    """Yaw ASLI frame (sebelum konsensus v26 / kunci v27 apa pun) -- supaya observasi tidak bergantung pada koreksi sebelumnya."""
    for k in ("yaw_before_lock_rad", "yaw_before_consensus_rad"):
        if o.get(k) is not None:
            return float(o[k])
    return float(o["yaw_rad"])


def collect_track_observations(items, recs, theta):
    """items: [(pos, frame, obj_index)] satu track. Return dict L (sisi panjang LiDAR), A (garis ambigu mod 90), R (RGB), shape_of.
    Sudut semua di KERANGKA DUNIA (dikurangi rotasi ego kumulatif theta[f]); bobot L/A = min(n_titik,60)/30."""
    L, A, R, shape_of = [], [], [], {}
    for p, f, oi in items:
        o = recs[f]["objects"][oi]
        if o.get("source") == "interpolated_temporal":
            continue
        th = theta[f]
        sh = _obj_lidar_shape(f, o)
        shape_of[(f, oi)] = sh
        if sh is not None:
            nw = min(sh["n"], 60) / 30.0
            if sh["l1"] >= o.get("width_m", 1.9) + YAW_V26_LONG_MARGIN_M:
                L.append((p, f, float((sh["phi"] - th) % np.pi), nw))
            else:
                A.append((p, f, float((sh["phi"] - th) % np.pi), nw))
        if LOCK_USE_RGB_VOTES and not o.get("lfit"):
            y0 = _lk_original_yaw(o)
            dm = max(float(o.get("distance_m") or LOCK_RGB_REF_DIST_M), 3.0)
            mult = float(np.clip((LOCK_RGB_REF_DIST_M / dm) ** 2, 0.15, 1.5))
            R.append((p, f, float((y0 - th) % np.pi), LOCK_RGB_W * mult))
            if o.get("yaw_alt_rad") is not None and o.get("yaw_source") not in _CONSENSUS_SOURCES:
                R.append((p, f, float((float(o["yaw_alt_rad"]) - th) % np.pi), LOCK_RGB_ALT_W * mult))
    return dict(L=L, A=A, R=R, shape_of=shape_of)


def _lk_refine_with_A(c, ang, w, A):
    """Garis sisi-pendek hanya tahu arah mod 90: ambil hipotesis (phi atau phi+90) yang dekat pusat c, tambahkan sebagai suara halus."""
    ang = list(ang); w = list(w)
    used = 0
    for _, _, a, wa in A:
        cands = [a, (a + np.pi / 2) % np.pi]
        d = [abs(float(_lk_wrap(x - c))) for x in cands]
        j = int(np.argmin(d))
        if d[j] <= np.radians(LOCK_INLIER_DEG):
            ang.append(cands[j]); w.append(0.5 * wa); used += 1
    return ang, w, used


def _lk_wrap90(a):
    return (np.asarray(a, float) + np.pi / 4) % (np.pi / 2) - np.pi / 4


def _lk_axis_from_lines(L, A):
    """Garis titik LiDAR (sisi panjang MAUPUN pendek) memberi arah yang tepat tapi hanya MODULO 90 deg. Kumpulkan dari semua frame,
    cari puncak di lingkaran mod-90. Return dict(phi, n, w, spread) atau None bila < LOCK_MIN_L_FRAMES frame sepakat."""
    lines = [(x[1], x[2], x[3]) for x in list(L) + list(A)]
    if len({l[0] for l in lines}) < LOCK_MIN_L_FRAMES:
        return None
    ang = np.array([l[1] for l in lines]); w = np.array([l[2] for l in lines])
    grid = np.radians(np.arange(0.0, 90.0, 1.0))
    d = np.abs(_lk_wrap90(grid[:, None] - ang[None, :]))
    S = (w[None, :] * np.exp(-0.5 * (d / np.radians(LOCK_SIGMA_DEG)) ** 2)).sum(1)
    k = int(S.argmax()); s1 = float(S[k])
    far = np.abs(_lk_wrap90(grid - grid[k])) > np.radians(25.0)
    s2 = float(S[far].max()) if far.any() else 0.0
    c = float(grid[k])
    for _ in range(3):
        m = np.abs(_lk_wrap90(ang - c)) <= np.radians(LOCK_INLIER_DEG)
        if not m.any():
            break
        c = float((np.angle((w[m] * np.exp(4j * ang[m])).sum()) / 4.0) % (np.pi / 2))
    m = np.abs(_lk_wrap90(ang - c)) <= np.radians(LOCK_INLIER_DEG)
    n_fr = len({lines[i][0] for i in np.nonzero(m)[0]}); w_in = float(w[m].sum())
    if n_fr < LOCK_MIN_L_FRAMES or w_in < LOCK_MIN_L_WEIGHT or w_in / max(float(w.sum()), 1e-9) < LOCK_L_INLIER_FRAC \
            or s1 < LOCK_PEAK_RATIO * max(s2, 1e-9):
        return None
    Rr = abs((w[m] * np.exp(4j * (ang[m] - c))).sum()) / max(w_in, 1e-9)
    spread = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(Rr, 1e-9)), 0.0)) / 4.0))
    return dict(phi=c, n=n_fr, w=w_in, spread=spread)


def _lk_choose_axis(phi, L, R):
    """Pilih di antara 2 hipotesis (phi, phi+90) memakai suara sisi-panjang LiDAR (kuat) dan RGB (berbobot jarak). None bila belum jelas."""
    cands = [phi % np.pi, (phi + np.pi / 2) % np.pi]
    mass = [0.0, 0.0]
    for x in list(L) + list(R):
        for j in (0, 1):
            if abs(float(_lk_wrap(x[2] - cands[j]))) <= np.radians(25.0):
                mass[j] += x[3]
    j = int(np.argmax(mass)); hi, lo = mass[j], mass[1 - j]
    if hi >= LOCK_AXIS_MIN_MASS and hi >= LOCK_AXIS_RATIO * max(lo, 1e-9):
        return cands[j], hi, lo
    return None


def estimate_track_world_yaw(obs):
    """Satu yaw dunia per track. Return dict(mode, yaw_ref, ...) dengan mode:
      'static_lidar'  : >=2 frame bukti sisi-panjang sepakat, tanpa tren                    -> KUNCI, confident
      'static_axis'   : garis titik LiDAR (sisi panjang/pendek) sepakat mod 90 di >=2 frame; sumbu dipilih dari L/RGB -> KUNCI, confident
      'static_mixed'  : bukti LiDAR kurang, tapi puncak suara (LiDAR+RGB) jelas               -> KUNCI, confident hanya bila ada L dekat puncak
      'dynamic'       : bukti LiDAR ada tapi tidak sepakat / ada tren (berbelok / ID tertukar) -> interpolasi antar jangkar
      'no_consensus'  : tidak ada bukti cukup                                                   -> tidak disentuh"""
    L, A, R = obs["L"], obs["A"], obs["R"]
    res = dict(mode="no_consensus", yaw_ref=None, n_anchor=0, spread_deg=None, drift_deg=None, anchor_frames=[], reason=None)
    if L:
        ang = np.array([x[2] for x in L]); w = np.array([x[3] for x in L]); ps = np.array([x[0] for x in L], float)
        c, s1, s2, m = _lk_peak(ang, w)
        n_in = int(m.sum()); w_in = float(w[m].sum()); frac = w_in / max(float(w.sum()), 1e-9)
        peak_ok = s1 >= LOCK_PEAK_RATIO * max(s2, 1e-9)
        enough = (n_in >= LOCK_MIN_L_FRAMES and w_in >= LOCK_MIN_L_WEIGHT)
        agree = (frac >= LOCK_L_INLIER_FRAC and peak_ok)
        if enough and agree:
            ang2, w2, n_a = _lk_refine_with_A(c, ang[m], w[m], A)
            yaw, spread = _lk_circ_mean(ang2, w2)
            drift = None
            if n_in >= 3 and (ps[m].max() - ps[m].min()) >= LOCK_DRIFT_MIN_SPAN:
                resid = _lk_wrap(ang[m] - yaw)
                slope = float(np.polyfit(ps[m], resid, 1, w=np.sqrt(w[m]))[0])
                drift = float(np.degrees(slope * (ps[m].max() - ps[m].min())))
            if drift is not None and abs(drift) > LOCK_MAX_DRIFT_DEG:
                res.update(mode="dynamic", reason=f"tren_yaw_dunia={drift:.0f}deg")
            else:
                res.update(mode="static_lidar", yaw_ref=yaw, n_anchor=n_in, spread_deg=spread, drift_deg=drift,
                           anchor_frames=[x[1] for x, k in zip(L, m) if k], n_short_side=n_a,
                           outlier_frames=[x[1] for x, k in zip(L, m) if not k], basis="lidar+rgb")
                return res
        elif len(L) >= 2 and not agree:       # bukti ada & cukup banyak tapi SALING BERTENTANGAN (bukan sekadar kurang kuat)
            res.update(mode="dynamic", reason=f"bukti_panjang_tidak_sepakat(inlier={n_in}/{len(L)},frac={frac:.2f},rasio_puncak={s1 / max(s2, 1e-9):.1f})")
    if res["mode"] == "dynamic":
        return res
    # ---- garis LiDAR (panjang+pendek) tepat tapi mod 90 -> sumbu dipilih dari L/RGB berbobot jarak ----
    ax = _lk_axis_from_lines(L, A)
    if ax is not None:
        pick = _lk_choose_axis(ax["phi"], L, R)
        if pick is not None:
            res.update(mode="static_axis", yaw_ref=float(pick[0]), n_anchor=ax["n"], spread_deg=ax["spread"], anchor_frames=[],
                       basis="lidar+rgb", n_short_side=len(A), axis_mass=(round(pick[1], 2), round(pick[2], 2)))
            return res
        res["reason"] = "garis_LiDAR_sepakat_tapi_sumbu_(panjang/pendek)_belum_jelas"
    # ---- campuran LiDAR(sisi panjang, <2 frame) + RGB: puncak suara informatif (sumbu dipilih RGB bila perlu) ----
    votes = [(x[2], x[3], True) for x in L] + [(x[2], x[3], False) for x in R]
    if votes:
        va = np.array([v[0] for v in votes]); vw = np.array([v[1] for v in votes]); vl = np.array([v[2] for v in votes])
        c, s1, s2, m = _lk_peak(va, vw)
        raw = float(vw[m].sum())
        if s1 >= LOCK_RGB_MIN_SCORE and s1 >= LOCK_PEAK_RATIO * max(s2, 1e-9) and raw >= LOCK_RGB_MIN_RAW:
            ang2, w2, n_a = _lk_refine_with_A(c, va[m], vw[m], A)
            yaw, spread = _lk_circ_mean(ang2, w2)
            has_L = bool((m & vl).any())
            res.update(mode="static_mixed", yaw_ref=yaw, n_anchor=int((m & vl).sum()), spread_deg=spread, anchor_frames=[],
                       basis=("lidar+rgb" if has_L else "rgb_only"), n_short_side=n_a)
            return res
        res["reason"] = f"puncak_tidak_jelas(s1={s1:.2f},s2={s2:.2f},raw={raw:.2f})"
    else:
        res["reason"] = "tanpa_observasi"
    return res


def _lk_dynamic_targets(items, recs, theta, obs):
    """Mode dinamis (mobil berbelok / ID tertukar): TIDAK ada satu yaw untuk seluruh track. Jangkar = frame dgn garis sisi-PANJANG LiDAR
    yang sepakat dgn median-lokal jangkar tetangga (+-15 frame; membuang outlier). Frame jangkar -> sudut jangkar; frame lain -> interpolasi
    sirkuler (mod pi) antar jangkar pengapit di kerangka dunia, atau tahan sudut jangkar terdekat bila <= DYN_MAX_GAP_FRAMES.
    Return (targets {(frame, idx): yaw_di_kerangka_frame}, anchors [(pos, frame, yaw_dunia)])."""
    idx_of = {f: oi for _, f, oi in items}
    pos_of = {f: p for p, f, _ in items}
    Ls = sorted(obs["L"])
    anchors = []
    for p, f, a, w in Ls:
        nb = [(q, aq, wq) for q, _, aq, wq in Ls if abs(q - p) <= 15]
        c, _s1, _s2, _m = _lk_peak([x[1] for x in nb], [x[2] for x in nb])
        if abs(float(_lk_wrap(a - c))) <= np.radians(LOCK_INLIER_DEG):
            anchors.append((p, f, float(c)))
    targets = {}
    if not anchors:
        return targets, anchors
    for p, f, oi in items:
        hit = [a for a in anchors if a[1] == f]
        if hit:
            tgt = hit[0][2]
        else:
            before = [a for a in anchors if a[0] < p]; after = [a for a in anchors if a[0] > p]
            b = before[-1] if before else None; a_ = after[0] if after else None
            tgt = None
            if b is not None and a_ is not None and (a_[0] - b[0]) <= 2 * DYN_MAX_GAP_FRAMES and \
                    np.degrees(abs(float(_lk_wrap(a_[2] - b[2])))) <= DYN_MAX_ANCHOR_DIFF_DEG:
                t = (p - b[0]) / max(a_[0] - b[0], 1e-9)
                tgt = (b[2] + t * float(_lk_wrap(a_[2] - b[2]))) % np.pi
            else:
                cand = [x for x in (b, a_) if x is not None and abs(x[0] - p) <= DYN_MAX_GAP_FRAMES]
                if cand:
                    tgt = min(cand, key=lambda x: abs(x[0] - p))[2]
        if tgt is not None:
            targets[(f, oi)] = float((tgt + theta[f]) % np.pi)       # kembali ke kerangka frame f
    return targets, anchors


# ---------------------------- v28: validasi sumbu dgn siluet ----------------------------
def _lk_sil_iou(o, cc):
    """IoU siluet box (corners kamera cc) vs bbox YOLO; toleran guncangan pitch (geser vertikal +-LOCK_VAL_PITCH_PX); sisi terpotong tak dihukum."""
    P = np.asarray(cc, float)
    if (P[:, 2] < 0.2).any():
        return 0.0
    uv = (K @ P.T).T; uv = uv[:, :2] / uv[:, 2:3]
    bb = o["bbox2d"]; occ = o.get("occluded_sides") or []
    trunc = [bb[0] <= 2 or "left" in occ, bb[1] <= 2 or "top" in occ, bb[2] >= IMG_W - 2 or "right" in occ, bb[3] >= IMG_H - 2 or "bottom" in occ]
    best = 0.0
    for dv in np.arange(-LOCK_VAL_PITCH_PX, LOCK_VAL_PITCH_PX + 0.1, 4.0):
        pb = [float(uv[:, 0].min()), float(uv[:, 1].min() + dv), float(uv[:, 0].max()), float(uv[:, 1].max() + dv)]
        if trunc[0]: pb[0] = max(pb[0], bb[0])
        if trunc[1]: pb[1] = max(pb[1], bb[1])
        if trunc[2]: pb[2] = min(pb[2], bb[2])
        if trunc[3]: pb[3] = min(pb[3], bb[3])
        iw = max(0.0, min(pb[2], bb[2]) - max(pb[0], bb[0])); ih = max(0.0, min(pb[3], bb[3]) - max(pb[1], bb[1]))
        inter = iw * ih
        union = max(pb[2] - pb[0], 0) * max(pb[3] - pb[1], 0) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return float(best)


def _lk_resize_corners(cc, L, W, H):
    cc = np.asarray(cc, float)
    hd = cc[0] - cc[3]; hd = hd / np.linalg.norm(hd)
    sd = cc[0] - cc[1]; sd = sd / np.linalg.norm(sd)
    up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
    G = cc[:4].mean(0)
    sx = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    bot = G + (L / 2) * sx[:, [0]] * hd + (W / 2) * sx[:, [1]] * sd
    return np.vstack([bot, bot + H * up])


def _lk_val_frames(items, recs):
    ok = []
    for p, f, oi in items:
        o = recs[f]["objects"][oi]
        if (o.get("source") == "lidar" and o.get("box_validated") and o.get("num_lidar_points", 0) >= MIN_FUSION_PTS
                and o.get("corners_cam") is not None and o.get("bbox2d") is not None):
            ok.append((p, f, oi))
    if len(ok) > LOCK_VAL_MAX_FRAMES:
        ok = [ok[i] for i in np.linspace(0, len(ok) - 1, LOCK_VAL_MAX_FRAMES).astype(int)]
    return ok


def lk_candidate_score(items, recs, theta, yaw_world):
    """Skor siluet rata-rata (0..1) sebuah kandidat yaw dunia di frame ber-LiDAR track ini. None bila frame valid < LOCK_VAL_MIN_FRAMES."""
    fr = _lk_val_frames(items, recs)
    if len(fr) < LOCK_VAL_MIN_FRAMES:
        return None
    ds_grid = np.arange(-LOCK_VAL_SHIFT_M, LOCK_VAL_SHIFT_M + 0.01, 0.25)
    ious = []
    for p, f, oi in fr:
        o = dict(recs[f]["objects"][oi])
        try:
            rotate_obj_to_lidar_yaw(o, float((yaw_world + theta[f]) % np.pi))
            cc = _lk_resize_corners(o["corners_cam"], o["length_m"], o["width_m"], o["height_m"])
            up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
            ctr = cc.mean(0); d = ctr - up * (ctr @ up); d = d / max(np.linalg.norm(d), 1e-6)
            lat = np.cross(up, d); lat = lat / max(np.linalg.norm(lat), 1e-6)
            ious.append(max(_lk_sil_iou(o, cc + ds * d + dl * lat) for ds in ds_grid for dl in (-0.4, 0.0, 0.4)))
        except Exception:
            continue
    return (float(np.mean(ious)), len(ious)) if len(ious) >= LOCK_VAL_MIN_FRAMES else None


def lk_validate_axis(items, recs, theta, obs, yaw_lines):
    """Uji 'lines' (yaw dari garis LiDAR), 'lines+90' (sumbu tertukar panjang<->pendek) dan 'rgb' (puncak RGB, bila berbeda) dgn siluet.
    Return dict(chosen_yaw, chosen, scores, margin, ambiguous, flipped) atau None bila tak bisa divalidasi."""
    cands = {"lines": float(yaw_lines), "lines+90": float((yaw_lines + np.pi / 2) % np.pi)}
    R = obs["R"]
    if R:
        c, s1, s2, m = _lk_peak([x[2] for x in R], [x[3] for x in R])
        if s1 >= LOCK_RGB_MIN_SCORE and all(abs(float(_lk_wrap(c - v))) > np.radians(LOCK_VAL_RGB_SEP_DEG) for v in cands.values()):
            cands["rgb"] = float(c)
    scores = {}
    for name, y in cands.items():
        sc = lk_candidate_score(items, recs, theta, y)
        if sc is None:
            return None
        scores[name] = sc[0]
    order = sorted(scores, key=lambda k: -scores[k])
    best, second = order[0], order[1]
    margin = scores[best] - scores[second]
    chosen = best if (best == "lines" or margin >= LOCK_VAL_MARGIN) else "lines"
    amb = bool(margin < LOCK_VAL_MARGIN)
    if chosen == "lines" and best != "lines":      # kandidat lain unggul tipis -> tetap garis LiDAR tapi ditandai ambigu
        amb = True
    return dict(chosen_yaw=cands[chosen], chosen=chosen, scores={k: round(v, 3) for k, v in scores.items()},
                margin=round(float(margin), 3), ambiguous=amb, flipped=(chosen != "lines"))



# ---------------------------- v29: pembaca permukaan + rekonsiliasi ----------------------------
def collect_surface_observations(items, recs, theta):
    """Observasi dari pembaca PERMUKAAN (mask+depth, sel 8a-1b). 'L'/'long' -> suara sisi-panjang (arah tepat); 'short'/'amb' -> garis mod 90."""
    L, A = [], []
    for p, f, oi in items:
        o = recs[f]["objects"][oi]
        if o.get("source") == "interpolated_temporal" or o.get("surf_kind") is None:
            continue
        th = theta[f]; w = float(o.get("surf_w") or 0.0)
        if w <= 0:
            continue
        if o["surf_kind"] in ("L", "long") and o.get("surf_phi_long_rad") is not None:
            L.append((p, f, float((float(o["surf_phi_long_rad"]) - th) % np.pi), w))
        elif o.get("surf_phi_line_rad") is not None:
            A.append((p, f, float((float(o["surf_phi_line_rad"]) - th) % np.pi), w))
    return dict(L=L, A=A, R=[], shape_of={})


def _lk_ok(e):
    """Estimasi statis yang boleh dipakai mengunci. 'static_mixed' (1 frame + RGB) TIDAK dihitung: bukan pengukuran independen."""
    return e is not None and e["mode"] in ("static_lidar", "static_axis") and e.get("yaw_ref") is not None


def lk_reconcile(estS, estL):
    """Gabungkan pembaca PERMUKAAN (S) dan garis LiDAR (L). HIJAU (confident) hanya bila S & L independen sepakat, atau S sendirian kuat.
    Return dict(mode, yaw_ref, basis, confident, n_anchor, spread_deg, reason, S_yaw, L_yaw)."""
    sy = float(estS["yaw_ref"]) if _lk_ok(estS) else None
    ly = float(estL["yaw_ref"]) if _lk_ok(estL) else None
    out = dict(mode="no_consensus", yaw_ref=None, basis=None, confident=False, n_anchor=0, spread_deg=None, reason=None, S_yaw=sy, L_yaw=ly,
               anchor_frames=[], drift_deg=None)
    strongS = bool(sy is not None and estS["n_anchor"] >= LOCK_S_STRONG_FRAMES and (estS["spread_deg"] is None or estS["spread_deg"] <= LOCK_S_STRONG_SPREAD_DEG))
    if sy is not None and ly is not None:
        d = float(np.degrees(abs(float(_lk_wrap(sy - ly)))))
        if d <= LOCK_AGREE_DEG:
            wS, wL = float(estS["n_anchor"]), float(estL["n_anchor"])
            yaw, spread = _lk_circ_mean([sy, ly], [max(wS, 1.0), max(1.5 * wL, 1.0)])
            out.update(mode="static_verif", yaw_ref=yaw, basis="permukaan+lidar", confident=True, n_anchor=int(wS + wL), spread_deg=spread,
                       anchor_frames=list(estS.get("anchor_frames", [])), drift_deg=estS.get("drift_deg"))
        else:
            pick = sy if strongS else ly
            out.update(mode="static_konflik", yaw_ref=pick, basis=("permukaan(konflik_dgn_lidar)" if strongS else "lidar(konflik_dgn_permukaan)"),
                       confident=False, n_anchor=(estS if strongS else estL)["n_anchor"], spread_deg=(estS if strongS else estL)["spread_deg"],
                       reason=f"KONFLIK: permukaan={np.degrees(sy):.0f} vs LiDAR={np.degrees(ly):.0f} (dunia, mod 180); dipakai {'permukaan' if strongS else 'LiDAR'}",
                       drift_deg=(estS if strongS else estL).get("drift_deg"))
    elif sy is not None:
        out.update(mode="static_surf", yaw_ref=sy, basis="permukaan", confident=strongS, n_anchor=estS["n_anchor"], spread_deg=estS["spread_deg"],
                   reason=(None if strongS else "permukaan_saja_belum_cukup_kuat"), anchor_frames=list(estS.get("anchor_frames", [])), drift_deg=estS.get("drift_deg"))
    elif ly is not None:
        out.update(mode="static_lidar", yaw_ref=ly, basis="lidar_saja", confident=False, n_anchor=estL["n_anchor"], spread_deg=estL["spread_deg"],
                   reason="hanya_garis_LiDAR(belum_diverifikasi_mask)", anchor_frames=list(estL.get("anchor_frames", [])), drift_deg=estL.get("drift_deg"))
    return out


def _lk_surface_pts_cam(f, oi):
    """Titik permukaan (LiDAR terverifikasi bila >= 12, selain itu + ZED) dlm kerangka KAMERA utk memusatkan ulang box. None bila tak ada."""
    p = SURF_DIR / f"{f:06d}.npz" if "SURF_DIR" in globals() else None
    if p is None or not p.exists():
        return None
    z = np.load(p)
    if f"z{oi}" not in z.files:
        return None
    lid = z[f"l{oi}"] if f"l{oi}" in z.files else np.zeros((0, 3), np.float32)
    P = lid if len(lid) >= 12 else np.vstack([lid, z[f"z{oi}"]])
    if len(P) < LREFIT_MIN_PTS:
        return None
    return (T_lidar_to_cam[:3, :3] @ P.astype(float).T).T + T_lidar_to_cam[:3, 3]


def force_target_yaw(o, target_rad, pts_cam=None):
    """Putar box ke yaw target LALU atur ulang posisi. Beda dgn apply_target_yaw: yaw TIDAK dibatalkan kalau siluet jelek; dicoba dua
    strategi posisi (titik LiDAR objek, siluet YOLO) dan dipakai yang IoU-nya lebih tinggi. Return (iou_sebelum, iou_sesudah)."""
    iou0 = _sil_iou(o, o["corners_cam"])
    if "yaw_before_lock_rad" not in o:
        o["yaw_before_lock_rad"] = _lk_original_yaw(o)
    rotate_obj_to_lidar_yaw(o, target_rad)
    base = np.asarray(o["corners_cam"], float).copy()
    cands = []
    if pts_cam is not None and len(pts_cam) >= LREFIT_MIN_PTS:
        _lidar_recenter(o, pts_cam)
        cands.append((_sil_iou(o, o["corners_cam"]), np.asarray(o["corners_cam"], float).copy()))
        _set_cam_corners(o, base)
    _refit_position_after_rotation(o)
    cands.append((_sil_iou(o, o["corners_cam"]), np.asarray(o["corners_cam"], float).copy()))
    best = max(cands, key=lambda t: t[0])
    _set_cam_corners(o, best[1])
    return float(iou0), float(best[0])


def _lk_mark(o, mode, source, basis, confident, lock_info, before):
    o["yaw_source"] = source
    o["yaw_basis"] = basis
    o["yaw_track_consistent"] = True
    o["yaw_confident"] = bool(confident)
    o["yaw_lock_status"] = mode
    o.update(lock_info)
    o["yaw_lock_changed_deg"] = round(float(np.degrees(abs(float(_lk_wrap(o["yaw_rad"] - before))))), 1)
    o["yaw_mirror_ambiguous"] = False


def run_track_yaw_lock(results, obj_dir, frame_idx_list=None):
    frame_idx_list = frame_idx_list or [r["frame_idx"] for r in results]
    recs = {f: load_object_record(obj_dir, f) for f in frame_idx_list if (obj_dir / f"{f:06d}.json").exists()}
    pos = {f: i for i, f in enumerate(frame_idx_list)}
    theta = estimate_ego_yaw(frame_idx_list)
    tracks = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("track_id") is not None and o.get("class") in LOCK_ENABLED_CLASSES:
                tracks.setdefault(o["track_id"], []).append((pos[f], f, oi))
    table, audit = [], {}
    for tid, items in sorted(tracks.items()):
        items.sort()
        obsL = collect_track_observations(items, recs, theta)           # pembaca 1: garis titik LiDAR (RGB-prior TIDAK ikut)
        obsS = collect_surface_observations(items, recs, theta)         # pembaca 2: permukaan mask+depth (independen dari pembaca 1)
        has_S = bool(obsS["L"] or obsS["A"])
        estL = estimate_track_world_yaw(obsL)
        estS = estimate_track_world_yaw(obsS) if has_S else None
        est = lk_reconcile(estS, estL)
        mode = est["mode"]; basis = est.get("basis"); confident = bool(est["confident"])
        dyn_obs = None
        # pembaca permukaan menunjukkan yaw BERUBAH (mobil berbelok / ID tertukar) dan tidak mendukung yaw statis dari garis LiDAR -> dinamis
        # menurut permukaan (garis LiDAR yg tercemar dinding/curb bisa tampak 'statis' padahal tidak).
        if estS is not None and estS["mode"] == "dynamic" and len(obsS["L"]) >= 6:
            sup = 0.0
            if _lk_ok(estL):
                sup = float(np.mean([abs(float(_lk_wrap(x[2] - estL["yaw_ref"]))) <= np.radians(LOCK_INLIER_DEG) for x in obsS["L"]]))
            if not _lk_ok(estL) or sup < 0.5:
                mode, dyn_obs, confident, basis = "dynamic", obsS, True, "permukaan"
                est = dict(est, reason=estS.get("reason"), yaw_ref=None, S_yaw=None, L_yaw=est.get("L_yaw"))
        if mode == "no_consensus":                                      # tak ada kunci statis -> mobil bergerak/berbelok? coba dinamis
            if estS is not None and estS["mode"] == "dynamic":
                mode, dyn_obs, confident, basis = "dynamic", obsS, True, "permukaan"; est["reason"] = estS.get("reason")
            elif estL["mode"] == "dynamic":
                mode, dyn_obs, confident, basis = "dynamic", obsL, False, "lidar_saja"; est["reason"] = estL.get("reason")
            else:
                est["reason"] = (estS or {}).get("reason") or estL.get("reason")
        n_changed = 0; max_change = 0.0; n_low = 0; n_fail = 0
        targets = {}
        if mode in ("static_verif", "static_surf", "static_konflik", "static_lidar"):
            for p, f, oi in items:
                targets[(f, oi)] = float((est["yaw_ref"] + theta[f]) % np.pi)
            src = "track_lock"
            info_common = dict(yaw_lock_world_rad=round(est["yaw_ref"], 4), yaw_lock_n_anchor=est["n_anchor"],
                               yaw_lock_spread_deg=(round(est["spread_deg"], 1) if est["spread_deg"] is not None else None),
                               yaw_lock_S_world_deg=(round(float(np.degrees(est["S_yaw"])), 1) if est["S_yaw"] is not None else None),
                               yaw_lock_L_world_deg=(round(float(np.degrees(est["L_yaw"])), 1) if est["L_yaw"] is not None else None))
        elif mode == "dynamic":
            targets, _anch = _lk_dynamic_targets(items, recs, theta, dyn_obs)
            anch_pos = [a[0] for a in _anch]
            est["n_anchor"] = len(_anch)
            src = "track_lock_dynamic"
            info_common = dict(yaw_lock_world_rad=None, yaw_lock_n_anchor=len(_anch), yaw_lock_spread_deg=None)
        else:
            src = None; info_common = {}
        for p, f, oi in items:
            o = recs[f]["objects"][oi]
            o.pop("yaw_lock_status", None); o.pop("yaw_lock_surf_dev_deg", None)
            if (f, oi) not in targets:
                o["yaw_lock_status"] = "dynamic_not_changed" if mode == "dynamic" else mode
                o["yaw_confident"] = False
                o["yaw_track_consistent"] = False
                continue
            tgt = targets[(f, oi)]
            before = _lk_original_yaw(o)
            o.setdefault("yaw_before_lock_rad", before)
            d_cur = _ang_diff_deg_mod180(o["yaw_rad"], tgt)
            iou = [None, None]
            if d_cur >= LOCK_SNAP_MIN_DEG and o.get("corners_cam") is not None:
                try:
                    pts_c = _lk_surface_pts_cam(f, oi)
                    if pts_c is None:
                        pts_c = _object_points_cam(f, o)
                    iou = list(force_target_yaw(o, tgt, pts_cam=pts_c))
                except Exception as e:               # satu objek gagal tidak boleh menghentikan seluruh track
                    o["yaw_lock_error"] = repr(e)[:120]; n_fail += 1
                    continue
            elif d_cur >= LOCK_SNAP_MIN_DEG:         # objek tanpa corners (mis. hasil interpolasi): cukup set yaw, box direkonstruksi dari center+yaw
                o["yaw_rad"] = round(float(tgt), 4)
            conf_here = confident
            if mode == "dynamic":
                conf_here = bool(confident and anch_pos and min(abs(p - q) for q in anch_pos) <= LOCK_DYN_CONF_FRAMES)
            _lk_mark(o, mode, src if o.get("source") != "interpolated_temporal" else o.get("yaw_source"),
                     basis, conf_here, dict(info_common, yaw_lock_iou=[None if v is None else round(v, 3) for v in iou],
                     yaw_lock_low_iou=bool(iou[1] is not None and iou[1] < LOCK_LOW_IOU)), before)
            if o.get("surf_phi_long_rad") is not None and o.get("surf_kind") in ("L", "long"):   # seberapa jauh ukuran permukaan frame ini dari yaw akhir
                o["yaw_lock_surf_dev_deg"] = round(float(_ang_diff_deg_mod180(o["yaw_rad"], float(o["surf_phi_long_rad"]))), 1)
            ch = o["yaw_lock_changed_deg"]
            if d_cur >= LOCK_SNAP_MIN_DEG:
                n_changed += 1
            max_change = max(max_change, ch)
            n_low += int(o["yaw_lock_low_iou"])
        # yaw kontinu (unwrap) per track, semua objek ber-track (termasuk hasil interpolasi)
        order = sorted(items)
        ys = np.array([recs[f]["objects"][oi]["yaw_rad"] for _, f, oi in order], float)
        yc = np.unwrap(2.0 * ys) / 2.0
        for (_, f, oi), v in zip(order, yc):
            recs[f]["objects"][oi]["yaw_cont_rad"] = round(float(v), 4)
        devs = [recs[f]["objects"][oi].get("yaw_lock_surf_dev_deg") for _, f, oi in order]; devs = [d for d in devs if d is not None]
        n_dev = int(sum(d > LOCK_SURF_DEV_FLAG_DEG for d in devs))
        audit[str(tid)] = dict(mode=mode, basis=basis, confident=confident, yaw_world_deg=(round(float(np.degrees(est["yaw_ref"])), 1) if est.get("yaw_ref") is not None else None),
                               S_world_deg=(round(float(np.degrees(est["S_yaw"])), 1) if est.get("S_yaw") is not None else None),
                               L_world_deg=(round(float(np.degrees(est["L_yaw"])), 1) if est.get("L_yaw") is not None else None),
                               n_frames=len(items), n_surf_obs=len(obsS["L"]) + len(obsS["A"]), n_lidar_obs=len(obsL["L"]) + len(obsL["A"]),
                               n_anchor=est.get("n_anchor"), spread_deg=est.get("spread_deg"), n_changed=n_changed, max_change_deg=round(max_change, 1),
                               n_low_iou=n_low, n_error=n_fail, n_surf_dev_gt=n_dev, n_surf_dev=len(devs), reason=est.get("reason"))
        a_ = audit[str(tid)]
        table.append((tid, len(items), mode, a_["yaw_world_deg"], a_["S_world_deg"], a_["L_world_deg"], a_["n_surf_obs"], a_["n_lidar_obs"], confident,
                      f"{n_dev}/{len(devs)} frame permukaan menyimpang>{LOCK_SURF_DEV_FLAG_DEG:.0f}" if devs else "-", est.get("reason") or ""))
    # ---- flag yaw_confident utk objek lain (Person/Motor/Sepeda) spt v26 + tulis ----
    for f, rec in recs.items():
        for o in rec["objects"]:
            if o.get("class") == "Person":
                o["yaw_confident"] = True
            elif o.get("class") not in LOCK_ENABLED_CLASSES:
                o["yaw_confident"] = bool(o.get("yaw_evidence", False))
            elif o.get("track_id") is None:
                o["yaw_confident"] = bool(o.get("yaw_evidence", False))
                o.setdefault("yaw_lock_status", "no_track_id")
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
    with open(OUT_DIR / "track_yaw_lock.json", "w") as fh:
        json.dump(audit, fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    modes = Counter(a["mode"] for a in audit.values())
    print(f"[Bagian 8a-2b v29] Kunci yaw per-track: {len(audit)} track Car | " + ", ".join(f"{k}={v}" for k, v in modes.items()))
    print("  HIJAU = static_verif (permukaan mask+depth DAN garis LiDAR sepakat) atau static_surf kuat | KUNING = konflik / hanya satu pembaca | MERAH = no_consensus")
    print(f"{'track':>6s} {'frame':>6s} {'mode':<14s} {'final':>6s} {'perm.':>6s} {'LiDAR':>6s} {'n_p':>4s} {'n_L':>4s} {'hijau':>6s}  catatan")
    for tid, n, mode, yw, sy_, ly_, nS, nL, cf, dv, why in table:
        fm = lambda x: ('%.0f' % x) if x is not None else '-'
        print(f"{tid:>6} {n:>6d} {mode:<14s} {fm(yw):>6s} {fm(sy_):>6s} {fm(ly_):>6s} {nS:>4d} {nL:>4d} {'ya' if cf else 'TIDAK':>6s}  {dv} {why}")
    n_car = sum(1 for r in results for o in r["objects"] if o.get("class") == "Car")
    n_conf = sum(1 for r in results for o in r["objects"] if o.get("class") == "Car" and o.get("yaw_confident"))
    print(f"yaw_confident (HIJAU): {n_conf}/{n_car} Car ({100 * n_conf / max(n_car, 1):.1f}%) | audit tersimpan di {OUT_DIR / 'track_yaw_lock.json'}")
    return audit


from collections import Counter


# ====================================================================================================
# [SEL 28 NOTEBOOK] 8a-3 konsistensi ukuran & jarak
# ====================================================================================================
# =============================================================================
# REVISI KE-23 -- Bagian 8a-3: KONSISTENSI UKURAN & POSISI per track (jalankan SEKALI setelah 8a-2, SEBELUM 8b)
# =============================================================================
# TEMUAN dari frame 1111-1119 (mobil parkir #78, statis): frame ber-LiDAR memberi box 4.70 x 2.02 m di jarak 17-18 m, tapi frame
# TANPA LiDAR (1118/1119) memberi 3.57 x 1.53 m di jarak 12.2 m -- ukuran & jarak saling menukar secara monokular (box kecil+dekat
# memberi siluet yg sama dgn box besar+jauh), dan tanpa LiDAR tak ada yg menahannya. Akibatnya mobil diam "melompat" ~5 m di BEV.
# PERBAIKAN (offline, per track Car):
#   1. Ukuran: semua frame memakai ukuran median frame ber-LiDAR (box_validated) -- mobil fisik yg sama berukuran tetap.
#   2. Jarak: frame TANPA LiDAR dicari ulang sepanjang sinar kamera agar LEBAR siluet cocok dgn bbox YOLO utk ukuran tetap (jarak ~ 1/lebar),
#      TOLERAN guncangan pitch kamera (dasar bbox bisa bergeser belasan piksel tanpa mobil bergerak). Prediksi garis lurus posisi frame
#      ber-LiDAR (+-GEOM_WINDOW frame) hanya penentu seri. Frame ber-LiDAR tidak digeser.
#   3. Setiap perubahan diuji siluet (IoU thd bbox YOLO tidak boleh turun > GEOM_SIL_MAX_DROP & >= GEOM_SIL_MIN_IOU); gagal -> dibatalkan.
# Syarat aman: >= GEOM_MIN_VALID_FRAMES frame ber-LiDAR & sisa garis lurus <= GEOM_MAX_RESID_M (ego berbelok / objek bergerak -> dilewati).
GEOM_MIN_VALID_FRAMES = 3
GEOM_WINDOW_FRAMES = 25
GEOM_MAX_RESID_M = 1.0
GEOM_SIL_MIN_IOU = 0.35
GEOM_SIL_MAX_DROP = 0.20
GEOM_MIN_TRACK_LEN = 4
GEOM_PITCH_TOL_PX = 16.0     # toleransi guncangan pitch utk uji siluet (geser vertikal +-16 px)
GEOM_MAX_SHIFT_M = 8.0       # pencarian jarak sepanjang sinar: +-8 m dari posisi awal


def _geom_valid(o):
    return (o.get("class") == "Car" and o.get("box_validated") and o.get("source") == "lidar"
            and o.get("num_lidar_points", 0) >= MIN_FUSION_PTS)


def _resize_corners_cam(cc, L, W, H):
    cc = np.asarray(cc, float)
    hd = cc[0] - cc[3]; hd = hd / np.linalg.norm(hd)
    sd = cc[0] - cc[1]; sd = sd / np.linalg.norm(sd)
    up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
    G = cc[:4].mean(0)
    sx = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)
    bot = G + (L / 2) * sx[:, [0]] * hd + (W / 2) * sx[:, [1]] * sd
    return np.vstack([bot, bot + H * up])


def _sil_iou_pt(o, cc):
    """IoU siluet TOLERAN GUNCANGAN PITCH kamera: boleh menggeser box proyeksi secara vertikal +-GEOM_PITCH_TOL_PX (1 deg ~ 8 px).
    Guncangan pitch menggeser dasar bbox YOLO belasan piksel tanpa mengubah lebar -> jarak dari dasar bbox jadi salah, tapi LEBAR bbox
    (yg menentukan jarak utk ukuran tetap) tetap benar."""
    P = np.asarray(cc, float)
    if (P[:, 2] < 0.2).any():
        return 0.0
    uv = (K @ P.T).T; uv = uv[:, :2] / uv[:, 2:3]
    bb = o["bbox2d"]; occ = o.get("occluded_sides") or []
    trunc = [bb[0] <= 2 or "left" in occ, bb[1] <= 2 or "top" in occ, bb[2] >= IMG_W - 2 or "right" in occ, bb[3] >= IMG_H - 2 or "bottom" in occ]
    best = 0.0
    for dv in np.arange(-GEOM_PITCH_TOL_PX, GEOM_PITCH_TOL_PX + 0.1, 2.0):
        pb = [float(uv[:, 0].min()), float(uv[:, 1].min() + dv), float(uv[:, 0].max()), float(uv[:, 1].max() + dv)]
        if trunc[0]: pb[0] = max(pb[0], bb[0])
        if trunc[1]: pb[1] = max(pb[1], bb[1])
        if trunc[2]: pb[2] = min(pb[2], bb[2])
        if trunc[3]: pb[3] = min(pb[3], bb[3])
        iw = max(0.0, min(pb[2], bb[2]) - max(pb[0], bb[0])); ih = max(0.0, min(pb[3], bb[3]) - max(pb[1], bb[1]))
        inter = iw * ih
        union = max(pb[2] - pb[0], 0) * max(pb[3] - pb[1], 0) + (bb[2] - bb[0]) * (bb[3] - bb[1]) - inter
        best = max(best, inter / union if union > 0 else 0.0)
    return float(best)


def _fit_range_by_width(o, cc, prior_xy=None):
    """Geser box (ukuran & yaw tetap) sepanjang sinar kamera (+ sedikit lateral) di bidang tanah agar siluet cocok dgn bbox YOLO
    (toleran pitch). Return corners baru."""
    up = cc[4] - cc[0]; up = up / np.linalg.norm(up)
    ctr = cc.mean(0); d = ctr - up * (ctr @ up); d = d / max(np.linalg.norm(d), 1e-6)
    lat = np.cross(up, d); lat = lat / max(np.linalg.norm(lat), 1e-6)
    def cost(ds, dl):
        c2 = cc + ds * d + dl * lat
        v = 1.0 - _sil_iou_pt(o, c2) + 0.01 * dl ** 2
        if prior_xy is not None:
            cur = _cam_pts_to_lidar(c2).mean(0)[:2]
            v += 0.004 * float(np.linalg.norm(cur - prior_xy)) ** 2 / 4.0
        return v
    grid = [(cost(ds, 0.0), ds) for ds in np.arange(-GEOM_MAX_SHIFT_M, GEOM_MAX_SHIFT_M + 0.01, 0.25)]
    ds0 = min(grid)[1]
    best = min(((cost(ds, dl), ds, dl) for ds in np.arange(ds0 - 0.25, ds0 + 0.26, 0.05) for dl in (-0.4, -0.2, 0.0, 0.2, 0.4)))
    return cc + best[1] * d + best[2] * lat


def _geom_apply(o, cc_new, dims=None):
    _set_cam_corners(o, cc_new)
    if dims is not None:
        o["length_m"], o["width_m"], o["height_m"] = [round(float(v), 3) for v in dims]


def run_geometry_consistency_pass(results, obj_dir):
    frame_list = [r["frame_idx"] for r in results]
    pos = {f: i for i, f in enumerate(frame_list)}
    recs = {f: load_object_record(obj_dir, f) for f in frame_list if (obj_dir / f"{f:06d}.json").exists()}
    tracks = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("track_id") is not None and o.get("class") == "Car" and o.get("source") != "interpolated_temporal":
                tracks.setdefault(o["track_id"], []).append((pos[f], f, oi))
    R_l2c = np.asarray(T_cam_to_lidar)[:3, :3].T
    stats = dict(resized=0, moved=0, rejected=0, skipped_tracks=0, shifts=[])
    for tid, items in tracks.items():
        items.sort()
        if len(items) < GEOM_MIN_TRACK_LEN:
            continue
        objs = [(p, f, recs[f]["objects"][oi]) for p, f, oi in items]
        V = [(p, o) for p, f, o in objs if _geom_valid(o)]
        if len(V) < GEOM_MIN_VALID_FRAMES:
            stats["skipped_tracks"] += 1
            continue
        dims = tuple(float(np.median([o[k] for _, o in V])) for k in ("length_m", "width_m", "height_m"))
        for p, f, o in objs:
            iou0 = _sil_iou_pt(o, o["corners_cam"])
            snap = {k: o.get(k) for k in ("corners_cam", "corners_lidar", "center_xyz", "distance_m", "length_m", "width_m", "height_m")}
            cc = _resize_corners_cam(o["corners_cam"], *dims)
            moved = False
            if not _geom_valid(o):
                # prior lemah: prediksi garis lurus posisi frame ber-LiDAR di sekitar (kalau cocok), HANYA sbg penentu seri
                near = [(pp, oo) for pp, oo in V if abs(pp - p) <= GEOM_WINDOW_FRAMES]
                prior = None
                if len(near) >= GEOM_MIN_VALID_FRAMES:
                    ts = np.array([pp for pp, _ in near], float); xy = np.array([oo["center_xyz"][:2] for _, oo in near], float)
                    cx = np.polyfit(ts, xy[:, 0], 1); cy = np.polyfit(ts, xy[:, 1], 1)
                    if float(np.median(np.hypot(np.polyval(cx, ts) - xy[:, 0], np.polyval(cy, ts) - xy[:, 1]))) <= GEOM_MAX_RESID_M:
                        prior = np.array([np.polyval(cx, p), np.polyval(cy, p)])
                cc = _fit_range_by_width(o, cc, prior)
                moved = True
            tmp = dict(o)
            iou1 = _sil_iou_pt(tmp, cc)
            floor = GEOM_SIL_MIN_IOU if not moved else 0.60
            if iou1 >= max(floor, iou0 - GEOM_SIL_MAX_DROP):
                before = np.asarray(o["center_xyz"][:2], float)
                _geom_apply(o, cc, dims)
                o["geom_iou"] = [round(iou0, 3), round(iou1, 3)]
                o["geom_source"] = "track_consistency" + ("+jarak_dari_lebar_bbox" if moved else "")
                stats["resized"] += 1
                if moved:
                    stats["moved"] += 1
                    stats["shifts"].append((tid, f, round(float(np.linalg.norm(np.asarray(o["center_xyz"][:2]) - before)), 2)))
            else:
                for k, v in snap.items():
                    o[k] = v
                o["geom_consistency_rejected"] = f"siluet (IoU {iou0:.2f}->{iou1:.2f})"
                stats["rejected"] += 1
    for f, rec in recs.items():
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(rec, fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    print(f"[Bagian 8a-3] Konsistensi geometri per track: ukuran diseragamkan {stats['resized']} objek, posisi digeser {stats['moved']} "
          f"(frame tanpa LiDAR), dibatalkan uji siluet {stats['rejected']}, track dilewati (frame ber-LiDAR < {GEOM_MIN_VALID_FRAMES}) {stats['skipped_tracks']}.")
    for tid, f, d in stats["shifts"][:15]:
        print(f"  [geser] track_id={tid} frame {f}: {d:.2f} m")
    return stats


# ====================================================================================================
# [SEL 29 NOTEBOOK] 8b isi celah temporal
# ====================================================================================================
# =============================================================================
# REVISI KE-13 -- Bagian 8b (BARU): isi celah temporal antar-frame ("antisipasi
# hilang di satu frame saja")
# =============================================================================
# KENAPA INI HARUS JADI SEL TERPISAH SETELAH BAGIAN 8 (bukan ditaruh di dalam Bagian 4):
# `associate_and_classify` (Bagian 4) memproses SATU frame pd satu waktu, berurutan maju --
# COASTING (Revisi ke-10) di situ cuma bisa "pinjam" posisi dari frame SEBELUMnya, krn saat
# frame ke-N diproses, frame ke-N+1 belum ada. Supaya bisa "ambil posisi sebelum ATAU
# SESUDAH" (persis permintaanmu) -- bahkan idealnya INTERPOLASI dari KEDUANYA sekaligus,
# yang secara geometris lebih akurat drpd cuma menempel ke posisi lama -- proses ini WAJIB
# jalan SETELAH seluruh batch Bagian 8 selesai, krn baru saat itu "masa depan" tiap track
# sudah diketahui semua.
#
# Kasus nyata yang memicu ini (dari uji frame 1090): mobil jauh, LiDAR 0 titik DAN peta
# depth 100% kosong (inf/nan) tepat di bbox-nya -- YOLO tetap yakin (conf tinggi) ada mobil
# di situ. Kalau ini kejadian cuma di 1-2 frame berturut-turut sedang track_id yg sama ADA
# di frame sebelum & sesudahnya, hasil interpolasi jauh lebih masuk akal drpd:
#   (a) di-skip total (objek "hilang berkedip" di visualisasi/BEV), atau
#   (b) di-coast dari frame sebelum saja (padahal kita SUDAH TAHU ke mana ia bergerak,
#       krn frame sesudahnya juga sudah diproses).
#
# CARA KERJA (2-pass, dijalankan SEKALI setelah `results` dari Bagian 8 lengkap):
#   1. Kumpulkan, per track_id, di frame_idx mana saja objeknya ADA (dari `results` yg
#      sudah tersimpan sbg JSON per-frame di OBJ_DIR).
#   2. Utk tiap pasang frame TERDEKAT yg py sama track_id-nya (fa, fb) dengan celah 1..
#      MAX_GAP_FRAMES_TO_INTERPOLATE frame di antaranya yang track ini TIDAK muncul, isi
#      tiap frame kosong di antaranya lewat interpolasi:
#        - posisi (center_xyz): interpolasi linear terhadap WAKTU asli (`lidar_t`, bukan
#          cuma index urutan -- supaya benar walau interval antar-frame gak seragam)
#        - arah (yaw_rad): interpolasi sirkuler (mod pi, sama seperti smoothing EMA yaw
#          yg sudah ada di Bagian 4)
#        - ukuran (L/W/H): rata-rata kedua sisi (ukuran objek harusnya gak berubah drastis)
#        - bbox2d: interpolasi linear juga (buat referensi visual, bukan sumber utama)
#      lalu SISIPKAN/GANTI entri track itu di JSON frame tsb (`source="interpolated_temporal"`)
#      dan tulis ulang filenya.
#   3. Kalau celahnya KEPANJANGAN (> MAX_GAP_FRAMES_TO_INTERPOLATE), SENGAJA TIDAK diisi --
#      itu tandanya objek memang benar2 pergi cukup lama (atau track_id kebetulan dipakai
#      ulang utk objek lain), menebak posisi di sini lebih berisiko drpd membiarkannya
#      kosong. Nilai ini bisa disetel di bawah.
#   4. Entri yg SUDAH ADA di suatu frame tapi berasal dari COASTING (Revisi ke-10, cuma
#      nempel ke posisi lama) DIGANTI oleh hasil interpolasi -- interpolasi dari KEDUA sisi
#      selalu lebih akurat drpd ekstrapolasi dari satu sisi saja.
#   CATATAN JUJUR: titik LiDAR mentah frame yg diisi TIDAK ikut di-override jadi kelas
#   Vehicle/Person (tidak ada titik LiDAR asli utk direlabel di frame itu -- itulah kenapa
#   dia butuh diisi). Jadi hasil interpolasi ini muncul di JSON objek (dipakai box 3D di
#   RGB/BEV Bagian 6 & konsumsi Fase 5/7 dst), TAPI titik point-cloud (.npy) frame tsb TETAP
#   apa adanya. Ini batasan yg jujur: kita tidak "menciptakan" titik LiDAR
#   yg sebenarnya gak pernah tertangkap sensor.

MAX_GAP_FRAMES_TO_INTERPOLATE = 5  # celah maksimal (dlm jumlah frame yg py TERPROSES,
                                     # bukan selisih idx mentah) yg py boleh diisi interpolasi


def _lerp_linear(a, b, t):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return (a + (b - a) * t).tolist()


def _lerp_yaw_mod_pi(yaw_a, yaw_b, t):
    va = np.array([np.cos(2 * yaw_a), np.sin(2 * yaw_a)])
    vb = np.array([np.cos(2 * yaw_b), np.sin(2 * yaw_b)])
    v = va * (1 - t) + vb * t
    return float((np.arctan2(v[1], v[0]) / 2) % np.pi)


def _build_interpolated_object(obj_a, obj_b, t, fa, fb, fm):
    center = _lerp_linear(obj_a["center_xyz"], obj_b["center_xyz"], t)
    bbox2d = _lerp_linear(obj_a["bbox2d"], obj_b["bbox2d"], t)
    yaw = _lerp_yaw_mod_pi(obj_a["yaw_rad"], obj_b["yaw_rad"], t)
    return {
        "class": obj_a["class"],
        "track_id": obj_a["track_id"],
        "confidence": round(min(obj_a["confidence"], obj_b["confidence"]), 3),
        "bbox2d": bbox2d,
        "center_xyz": center,
        "distance_m": float(np.hypot(center[0], center[1])),
        "length_m": (obj_a["length_m"] + obj_b["length_m"]) / 2.0,
        "width_m": (obj_a["width_m"] + obj_b["width_m"]) / 2.0,
        "height_m": (obj_a["height_m"] + obj_b["height_m"]) / 2.0,
        "length_m_raw": None, "width_m_raw": None, "height_m_raw": None,
        "num_lidar_points": 0,
        "num_rejected_depth_outliers": None, "num_rejected_height_outliers": None,
        "num_box_outliers_removed": None,
        "box_validated": False,
        "source": "interpolated_temporal",
        "yaw_rad": round(yaw, 4),
        "yaw_from_estimate": False,
        "yaw_source": "interpolated_temporal",
        "fase3_class_crosscheck": {},
        "is_split_from_single_detection": False,
        "relative_speed_mps": obj_a.get("relative_speed_mps"),
        "movement_status": obj_a.get("movement_status", "belum_diketahui"),
        "is_coasted": False,
        "coast_count": 0,
        "depth_dilation_factor": None,
        "interpolated_from_frames": [fa, fb],   # jejak audit: frame asal interpolasi
        "yaw_confident": bool(obj_a.get("yaw_confident") and obj_b.get("yaw_confident")),   # Revisi ke-21
    }


def fill_temporal_gaps(results, rows, obj_dir, max_gap=MAX_GAP_FRAMES_TO_INTERPOLATE):
    """Jalankan SEKALI setelah Bagian 8 (batch semua frame) selesai.
    results : list of record dict, SAMA seperti yang dikembalikan Bagian 8 (urutan waktu
              harus terjaga -- persis seperti yg sudah dipastikan Bagian 8).
    rows    : sync_table (dipakai ulang get_frame_time buat interpolasi yang sadar waktu
              asli, bukan cuma sadar urutan index).
    obj_dir : Path folder JSON per-frame (OBJ_DIR) -- file akan ditulis ulang kalau ada
              celah yg diisi.
    Return: dict ringkasan {track_id: [frame_idx yg diisi]}.
    """
    frame_idx_list = [r["frame_idx"] for r in results]
    frame_pos = {f: i for i, f in enumerate(frame_idx_list)}

    # 1) kumpulkan per track_id, frame mana objeknya py ADA
    track_frames = {}   # track_id -> {frame_idx: object_dict}
    for rec in results:
        for obj in rec["objects"]:
            tid = obj.get("track_id")
            if tid is None:
                continue   # gak bisa dikorelasikan antar-frame tanpa identitas track
            track_frames.setdefault(tid, {})[rec["frame_idx"]] = obj

    filled_summary = {}
    frames_to_rewrite = {}   # frame_idx -> record (di-mutate in-place, ditulis di akhir)

    for tid, fmap in track_frames.items():
        known_frames = sorted(fmap.keys())
        if len(known_frames) < 2:
            continue
        for fa, fb in zip(known_frames[:-1], known_frames[1:]):
            pa, pb = frame_pos[fa], frame_pos[fb]
            gap_frames = frame_idx_list[pa + 1: pb]   # frame yg SUDAH terproses (ada JSON-nya)
                                                        # tapi track ini gak muncul di situ
            if not gap_frames or len(gap_frames) > max_gap:
                continue   # celah kosong (bersebelahan) ATAU kepanjangan -- lewati

            obj_a, obj_b = fmap[fa], fmap[fb]
            row_a = next((r for r in rows if int(r["idx"]) == fa), None)
            row_b = next((r for r in rows if int(r["idx"]) == fb), None)
            t_a = get_frame_time(row_a, fa) if row_a else float(fa)
            t_b = get_frame_time(row_b, fb) if row_b else float(fb)
            if t_b <= t_a:
                continue   # data waktu gak masuk akal, jangan paksa interpolasi

            for fm in gap_frames:
                row_m = next((r for r in rows if int(r["idx"]) == fm), None)
                t_m = get_frame_time(row_m, fm) if row_m else float(fm)
                t = float(np.clip((t_m - t_a) / (t_b - t_a), 0.0, 1.0))

                interp_obj = _build_interpolated_object(obj_a, obj_b, t, fa, fb, fm)

                rec = frames_to_rewrite.get(fm)
                if rec is None:
                    json_path = obj_dir / f"{fm:06d}.json"
                    if not json_path.exists():
                        continue   # frame ini belum punya file JSON sama sekali -- lewati
                    with open(json_path) as f:
                        rec = json.load(f)
                    frames_to_rewrite[fm] = rec

                # ganti kalau track ini SUDAH ada di frame tsb (mis. hasil COASTING) --
                # interpolasi dari 2 sisi selalu lebih akurat drpd ekstrapolasi 1 sisi.
                existing_idx = next((i for i, o in enumerate(rec["objects"])
                                     if o.get("track_id") == tid), None)
                if existing_idx is not None:
                    rec["objects"][existing_idx] = interp_obj
                else:
                    rec["objects"].append(interp_obj)
                rec["num_objects"] = len(rec["objects"])

                filled_summary.setdefault(tid, []).append(fm)

    for fm, rec in frames_to_rewrite.items():
        with open(obj_dir / f"{fm:06d}.json", "w") as f:
            json.dump(rec, f, indent=2)

    total_filled = sum(len(v) for v in filled_summary.values())
    print(f"[Bagian 8b] Interpolasi temporal selesai: {total_filled} slot objek diisi "
          f"di {len(frames_to_rewrite)} frame, mencakup {len(filled_summary)} track_id.")
    for tid, frames in filled_summary.items():
        print(f"  track_id={tid}: frame {frames}")
    return filled_summary


# ====================================================================================================
# [SEL 31 NOTEBOOK] 8c object_point_idx + cek dinding
# ====================================================================================================
# =============================================================================
# 8c -- (1) object_point_idx/*.npy dari HASIL AKHIR (setelah 8a-1/2/3/8b), (2) checkpoint Fase 3.
# =============================================================================
OBJ_PT_MARGIN_M = 0.35


def write_object_point_idx(results):
    n_obj_pts = 0; n_orphan = 0
    for r in results:
        f = r["frame_idx"]
        p = POINTS_DIR / f"{f:06d}.npy"
        if not p.exists():
            continue
        P = np.load(p)
        rec = load_object_record(OBJ_DIR, f)
        lab = np.isin(P[:, 6].astype(int), (4, 5))
        pairs = []
        claimed = np.zeros(len(P), bool)
        for k, o in enumerate(rec["objects"]):
            if o.get("corners_lidar") is None:
                continue
            inside = _points_inside_box(P[:, :3], o["corners_lidar"], margin_xy=OBJ_PT_MARGIN_M) & lab
            idx = np.nonzero(inside)[0]
            claimed |= inside
            pairs.extend((k, int(i)) for i in idx)
        n_obj_pts += len(pairs); n_orphan += int((lab & ~claimed).sum())
        np.save(PTIDX_DIR / f"{f:06d}.npy", np.asarray(pairs, dtype=np.int32).reshape(-1, 2))
    print(f"object_point_idx: {n_obj_pts} pasangan (objek, titik) ditulis; titik berlabel 4/5 yang tidak lagi masuk kotak mana pun: {n_orphan}")
    return n_obj_pts, n_orphan

# ---- checkpoint: dinding / tanah di dalam kotak, jarak kotak ke titik objek ----
def check_boxes_for_walls(frame_ids):
    from matplotlib.path import Path as _MPath
    n_wall_in_box = 0; n_lab_wall = 0; dist_med = []
    for f in frame_ids:
        row = next((r for r in rows if int(r["idx"]) == f), None)
        pts = np.load(POINTS_DIR / f"{f:06d}.npy"); rec = load_object_record(OBJ_DIR, f)
        depth_f = np.load(resolve(BASE_DIR, row.get("depth_data"), "depth"))
        reset_ground_state()
        gp = get_ground_plane(depth_f, pts, T_lidar_to_cam, K)
        if gp is None:
            continue
        gf = ground_frame(gp["n"], gp["h"])
        pc = (T_lidar_to_cam[:3, :3] @ pts[:, :3].T).T + T_lidar_to_cam[:3, 3]
        wall = build_frame_context(pts, pc, gf)
        lab = np.nonzero(np.isin(pts[:, 6].astype(int), (4, 5)))[0]
        n_lab_wall += int(wall[lab].sum())
        for o in rec["objects"]:
            if o.get("corners_lidar") is None or o.get("source") == "interpolated_temporal":
                continue
            poly = np.asarray(o["corners_lidar"])[:4, :2]
            inside = _MPath(poly).contains_points(pts[:, :2])
            n_wall_in_box += int((inside & wall).sum())
            if len(lab):
                d = np.full(len(lab), np.inf)
                for a, b in zip(poly, np.roll(poly, -1, 0)):
                    ab = b - a; t = np.clip(((pts[lab, :2] - a) @ ab) / (ab @ ab), 0, 1)
                    d = np.minimum(d, np.linalg.norm(pts[lab, :2] - (a + t[:, None] * ab), axis=1))
                d[_MPath(poly).contains_points(pts[lab, :2])] = 0
                near = d[d < 3.0]
                if len(near):
                    dist_med.append(float(np.median(near)))
    reset_ground_state()
    return dict(n_lab_wall=n_lab_wall, n_wall_in_box=n_wall_in_box, dist_med=dist_med)


# ====================================================================================================
# [SEL 34 NOTEBOOK] 8d arah hadap
# ====================================================================================================
# =============================================================================
# v27 -- Bagian 8d: ARAH HADAP (HEADING) mobil: depan vs belakang. Jalankan SETELAH 8c / ringkasan, SEBELUM viewer.
# =============================================================================
# KENAPA: Flutter (Fase 7) membaca `yaw` sebagai sudut -pi..pi terhadap ego (0 = searah ego, +-pi = lawan arah) lalu memutar MODEL 3D
# mobil (bodi + lampu). Yaw hasil Fase 3 hanya SUMBU (modulo 180 deg: kotak simetris), jadi hidung model bisa terbalik 50% waktu.
# Data yang ada TIDAK memuat sisi depan/belakang (LiDAR cuma lihat permukaan, belum ada odometri ego). Maka arah ditentukan dengan
# PRIOR yang dicatat jujur di `heading_source`, dan bisa DIKOREKSI MANUAL per track lewat viewer (tombol r) -- hanya ~10 track Car.
#   Aturan prior (lalu lintas KIRI di Indonesia; ubah HEADING_LEFT_HAND_TRAFFIC bila perlu):
#     - sumbu sejajar arah ego (<= HEADING_PARALLEL_DEG): mobil di sisi KIRI ego searah ego (heading ~0), di sisi KANAN lawan arah (~180).
#     - sumbu miring/tegak lurus (parkir serong/tegak): hidung diarahkan ke SISI objek (menjauhi jalan = parkir "maju"). TIDAK terverifikasi.
#   Keputusan diambil SEKALI per track di kerangka dunia (frame terdekat = geometri terbaik), lalu dipakai di semua frame (dikoreksi rotasi
#   ego). Untuk track tanpa kunci yaw (no_consensus/dinamis) hidung tiap frame = kandidat (sumbu atau sumbu+180) terdekat ke arah itu.
# Field baru: yaw_heading_rad (-pi..pi, kerangka LiDAR = ego: x maju, y kiri), yaw_axis_rad (sumbu mod pi, cadangan), heading_source,
#   heading_verified (True hanya bila dikoreksi manual), yaw_cont_rad (kini unwrap 2*pi). Bila HEADING_WRITE_YAW_RAD=True, `yaw_rad`
#   juga diisi heading (-pi..pi) supaya Fase 5 yang membaca yaw_rad langsung menghasilkan `yaw` berarah. Cuma kelas Car.
HEADING_LEFT_HAND_TRAFFIC = True
HEADING_PARALLEL_DEG = 35.0
HEADING_WRITE_YAW_RAD = True


def _hd_load_override():
    try:
        with open(HEADING_OVERRIDE_FILE) as fh:
            return {str(k): bool(v) for k, v in json.load(fh).items()}
    except Exception:
        return {}


def _hd_pi(a):
    """ke (-pi, pi]"""
    return float(np.pi - ((np.pi - float(a)) % (2.0 * np.pi)))


def _hd_prior(o0):
    """Arah hidung (rad, kerangka frame o0) dari prior. Return (heading, sumber)."""
    a = float(o0["yaw_rad"]) % np.pi
    dev = float((a + np.pi / 2) % np.pi - np.pi / 2)               # sumbu relatif sumbu-x ego, [-90, 90)
    y0 = float(o0["center_xyz"][1])                                # +y = kiri
    if abs(np.degrees(dev)) <= HEADING_PARALLEL_DEG:
        same_dir = (y0 > 0) if HEADING_LEFT_HAND_TRAFFIC else (y0 < 0)
        return (dev if same_dir else dev + np.pi), "prior_sisi_jalan"
    cands = [a, a + np.pi]
    toward_side = [c for c in cands if np.sin(c) * np.sign(y0 if y0 != 0 else 1.0) > 0]
    return (toward_side[0] if toward_side else cands[0]), "prior_parkir_serong_tak_terverifikasi"


def apply_track_headings(results, obj_dir, only=None, verbose=True):
    frames = [r["frame_idx"] for r in results]
    pos = {f: i for i, f in enumerate(frames)}
    recs = {f: load_object_record(obj_dir, f) for f in frames if (obj_dir / f"{f:06d}.json").exists()}
    theta = estimate_ego_yaw(frames)
    override = _hd_load_override()
    tracks = {}
    for f, rec in recs.items():
        for oi, o in enumerate(rec["objects"]):
            if o.get("class") == "Car" and o.get("track_id") is not None and (only is None or o["track_id"] in only):
                tracks.setdefault(o["track_id"], []).append((pos[f], f, oi))
    touched = set(); summary = {}
    for tid, items in sorted(tracks.items()):
        items.sort()
        base = [(p, f, oi) for p, f, oi in items if recs[f]["objects"][oi].get("source") != "interpolated_temporal"] or items
        p0, f0, oi0 = min(base, key=lambda t: float(recs[t[1]]["objects"][t[2]].get("distance_m") or 1e9))
        o0 = recs[f0]["objects"][oi0]
        h0, src = _hd_prior(o0)
        world = (h0 - theta[f0]) % (2 * np.pi)
        flipped = bool(override.get(str(tid), False))
        if flipped:
            world = (world + np.pi) % (2 * np.pi); src = "manual_balik"
        hs = []
        for p, f, oi in items:
            o = recs[f]["objects"][oi]
            ref = (world + theta[f]) % (2 * np.pi)
            a = float(o["yaw_rad"]) % np.pi
            c = min((a, a + np.pi), key=lambda x: abs(float((x - ref + np.pi) % (2 * np.pi) - np.pi)))
            h = _hd_pi(c)
            o["yaw_axis_rad"] = round(a, 4); o["yaw_heading_rad"] = round(h, 4)
            o["heading_source"] = src; o["heading_verified"] = flipped
            if HEADING_WRITE_YAW_RAD:
                o["yaw_rad"] = round(h, 4)
            hs.append(h)
            touched.add(f)
        yc = np.unwrap(np.array(hs))                                  # kontinu 2*pi untuk interpolasi/animasi
        for (p, f, oi), v in zip(items, yc):
            recs[f]["objects"][oi]["yaw_cont_rad"] = round(float(v), 4)
        summary[tid] = (len(items), src, round(float(np.degrees(_hd_pi(world))), 0))
    for f in touched:
        with open(obj_dir / f"{f:06d}.json", "w") as fh:
            json.dump(recs[f], fh, indent=2)
    refresh_results_from_disk(results, obj_dir)
    if verbose:
        print(f"[Bagian 8d] Arah hadap: {len(summary)} track Car | yaw_rad {'DIISI heading (-pi..pi)' if HEADING_WRITE_YAW_RAD else 'tetap sumbu mod pi'}")
        print(f"{'track':>6s} {'frame':>6s} {'heading_dunia(deg)':>19s}  sumber")
        for tid, (n, src, hd) in summary.items():
            print(f"{tid:>6} {n:>6d} {hd:>19.0f}  {src}")
        print("Sumber 'prior_*' = TEBAKAN dari posisi sisi jalan, bukan terukur. Cek di viewer: pilih track (t) lalu tekan r bila hidung terbalik.")
    return summary


# ====================================================================================================================
# ORKESTRASI (BARU, bukan dari notebook)
# ====================================================================================================================
def configure(out):
    """Isi semua global yang di notebook diisi sel config (path, kalibrasi, sync table)."""
    global BASE_DIR, OUT_DIR, POINTS_DIR, HEIGHT_DIR, PTIDX_DIR, PREVIEW_DIR, SURF_DIR, HEADING_OVERRIDE_FILE, OBJ_DIR
    global T_lidar_to_cam, T_cam_to_lidar, K, D, IMG_W, IMG_H, rows, candidates
    BASE_DIR = Path(out)
    OUT_DIR = BASE_DIR / "fase3_deteksi"
    POINTS_DIR, HEIGHT_DIR = OUT_DIR / "points", OUT_DIR / "heights"
    PTIDX_DIR, PREVIEW_DIR, SURF_DIR = OUT_DIR / "object_point_idx", OUT_DIR / "preview", OUT_DIR / "surface"
    HEADING_OVERRIDE_FILE = OUT_DIR / "track_heading_override.json"
    OBJ_DIR = MemDir()
    for d in (OUT_DIR, POINTS_DIR, HEIGHT_DIR, PTIDX_DIR):
        d.mkdir(parents=True, exist_ok=True)
    calib = BASE_DIR / "calib" / "calib_config.json"
    sync = BASE_DIR / "sync_table.csv"
    for p in (calib, sync):
        if not p.exists():
            raise SystemExit(f"Tidak ketemu: {p} (jalankan Fase 1 dan 2 dulu)")
    T_lidar_to_cam, T_cam_to_lidar, K, D, IMG_W, IMG_H = load_calib(calib)
    rows = load_sync_rows(sync)
    candidates = [r for r in rows
                  if r.get("lidar_path") and r.get("rgb_data") and r.get("depth_data")
                  and resolve(BASE_DIR, r.get("lidar_path"), "lidar") is not None
                  and resolve(BASE_DIR, r.get("lidar_path"), "lidar").exists()]
    print(f"[Fase 3] frame lengkap (lidar mentah+rgb+depth): {len(candidates)} dari {len(rows)} | K fx={K[0, 0]:.1f} | {IMG_W}x{IMG_H}")


def _reset_state():
    global track_yaw_history, track_motion_history, track_position_history, track_last_valid_info, _SURF_MODEL
    track_yaw_history, track_motion_history, track_position_history, track_last_valid_info = {}, {}, {}, {}
    _SURF_MODEL = None
    reset_ground_state()


def _clean_previous_outputs():
    """Run baru = hasil baru: buang berkas turunan lama (cache surface/ego_yaw bisa basi bila data berbeda)."""
    for d in (POINTS_DIR, HEIGHT_DIR, PTIDX_DIR):
        for p in d.glob("*.npy"):
            p.unlink()
    if SURF_DIR.exists():
        shutil.rmtree(SURF_DIR)
    for name in ("ego_yaw.json", "track_yaw_lock.json", "fase3_done.json"):
        p = OUT_DIR / name
        if p.exists():
            p.unlink()


def run_batch(frame_idxs):
    """Bagian 8: YOLO+ByteTrack+fusi per frame (urutan kronologis WAJIB). Titik/heights ke disk, objek ke memori."""
    reset_tracker()
    _reset_state()
    results = []
    for idx in tqdm(frame_idxs, desc="Fase 3 batch (YOLO + fusi LiDAR)"):
        rec = process_frame_fase3(idx, rows, BASE_DIR, save_preview=False)
        if rec:
            results.append(rec)
    return results


def run_offline_passes(results):
    """Bagian 8a-0 ... 8d, urutan persis notebook. Semua baca/tulis objects/*.json terjadi di memori."""
    out = {}
    with Timer("8a-0 sambung track"):
        out["stitch"] = run_track_stitching(results, OBJ_DIR)
    with Timer("8a-1 konfirmasi track"):
        out["confirm"] = run_track_confirmation(results, OBJ_DIR, POINTS_DIR, None)
    with Timer("8a-1b yaw dari permukaan (YOLO-seg ulang)"):
        if SURF_ENABLED:
            out["surface"] = build_surface_points(results, OBJ_DIR, force=True)
            out["surface_measure"] = run_surface_measurement(results, OBJ_DIR)
    with Timer("8a-2b kunci yaw per-track"):
        out["yaw_lock"] = run_track_yaw_lock(results, OBJ_DIR)
    with Timer("8a-3 konsistensi ukuran & jarak"):
        out["geometry"] = run_geometry_consistency_pass(results, OBJ_DIR)
    with Timer("8b isi celah temporal"):
        out["gaps"] = fill_temporal_gaps(results, rows, OBJ_DIR)
    with Timer("8c object_point_idx"):
        out["ptidx"] = write_object_point_idx(results)
    with Timer("8d arah hadap"):
        out["heading"] = apply_track_headings(results, OBJ_DIR)
    return out


def _final_records():
    recs = {}
    for name, txt in OBJ_DIR.store.items():
        recs[int(Path(name).stem)] = _json.loads(txt)
    return dict(sorted(recs.items()))


def run_checkpoints(n_candidates, passes, strict=False, wall_sample=60):
    """Checkpoint Fase 3 otomatis (pengganti ringkasan plot & viewer)."""
    ck = Checker("FASE 3", BASE_DIR / "checkpoints" / "fase3_report.json", strict=strict)
    recs = _final_records()
    frames = list(recs)
    npts = sorted(int(p.stem) for p in POINTS_DIR.glob("*.npy"))
    nh = sorted(int(p.stem) for p in HEIGHT_DIR.glob("*.npy"))
    nidx = sorted(int(p.stem) for p in PTIDX_DIR.glob("*.npy"))
    ck.ok("Semua frame lengkap diproses", len(frames) == n_candidates and len(npts) == n_candidates,
          f"objects={len(frames)} points={len(npts)} dari {n_candidates}")
    ck.ok("heights/*.npy ada untuk setiap frame (Fase 4 bergantung padanya)", nh == npts, f"{len(nh)} vs {len(npts)}")
    ck.ok("object_point_idx/*.npy ada untuk setiap frame", nidx == npts, f"{len(nidx)} vs {len(npts)}")

    objs = [o for r in recs.values() for o in r["objects"]]
    cls = Counter(o.get("class") for o in objs)
    ck.info("Objek", f"{len(objs)} total ({dict(cls)}), rata-rata {len(objs) / max(len(frames), 1):.2f}/frame")
    real = [o for o in objs if o.get("source") != "interpolated_temporal"]
    bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 999)]
    parts = []
    for lo, hi in bins:
        s = [o for o in real if lo <= o["distance_m"] < hi]
        if s:
            ok = sum(1 for o in s if o.get("status") == "lidar_ok")
            parts.append(f"{lo}-{hi}m: n={len(s)} lidar_ok={100 * ok / len(s):.0f}% titik~{np.median([o['num_lidar_points'] for o in s]):.0f}")
    ck.info("lidar_ok vs depth_only per jarak", " | ".join(parts))
    ck.info("Titik objek per frame", f"rata-rata {np.mean([r['num_points_object'] for r in recs.values()]):.1f}" if recs else "-")

    cars = [o for o in objs if o.get("class") == "Car"]
    bad_range = sum(1 for o in cars if not (-math.pi - 1e-6 <= o.get("yaw_rad", 0.0) <= math.pi + 1e-6))
    ck.ok("yaw_rad semua Car dalam [-pi, pi]", bad_range == 0, f"{bad_range} di luar rentang")
    n_head = sum(1 for o in cars if o.get("yaw_heading_rad") is not None)
    ck.ok("Car berarah hidung (8d) >= 95%", n_head >= 0.95 * max(len(cars), 1), f"{n_head}/{len(cars)}", fatal=False)
    conf = sum(1 for o in cars if o.get("yaw_confident"))
    ck.info("yaw_confident (HIJAU)", f"{conf}/{len(cars)}")
    by_track = defaultdict(list)
    for f, r in recs.items():
        for o in r["objects"]:
            if o.get("class") == "Car" and o.get("track_id") is not None:
                by_track[o["track_id"]].append((f, o.get("yaw_rad", 0.0)))
    flips = 0
    for tid, seq in by_track.items():
        seq.sort()
        for (f0, y0), (f1, y1) in zip(seq, seq[1:]):
            if f1 - f0 == 1:
                d = abs(((y1 - y0 + math.pi / 2) % math.pi) - math.pi / 2)      # selisih sumbu (mod pi)
                flips += int(math.degrees(d) > 45)
    ck.ok("Lompatan yaw > 45 deg antar frame berurutan (track sama) ~ 0", flips <= max(2, 0.01 * len(cars)), f"{flips}", fatal=False)
    pts_pairs, orphans = passes.get("ptidx") or (None, None)
    if pts_pairs is not None:
        ck.ok("Titik berlabel 4/5 yang tak masuk kotak mana pun (yatim) kecil", orphans <= 0.05 * max(pts_pairs, 1),
              f"{orphans} yatim dari {pts_pairs} titik objek", fatal=False)
    ck.info("Person lolos semua gerbang", sum(1 for o in objs if o.get("class") == "Person"))

    sample = frames[:wall_sample]
    if sample:
        w = check_boxes_for_walls(sample)
        ck.ok("Titik berlabel Vehicle/Person yang ternyata titik DINDING = 0", w["n_lab_wall"] == 0, f"{w['n_lab_wall']} (sampel {len(sample)} frame)", fatal=False)
        if w["dist_med"]:
            med, p90 = float(np.median(w["dist_med"])), float(np.percentile(w["dist_med"], 90))
            ck.ok("Jarak titik objek ke kotak < 0.5 m (median)", med < 0.5, f"median {med:.2f} m, p90 {p90:.2f} m", fatal=False)
    ck.finish()


def run(out, force=False, strict=False, reuse_batch=False, max_frames=None, skip_checks=False):
    t_all = time.time()
    out = Path(out)
    marker = out / "fase3_deteksi" / "fase3_done.json"
    if marker.exists() and not force:
        print(f"[Fase 3] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    configure(out)
    snap = OUT_DIR / "_batch_records.json.gz"
    frame_idxs = [int(r["idx"]) for r in candidates]
    if max_frames:
        frame_idxs = frame_idxs[:max_frames]

    if reuse_batch and snap.exists():
        with gzip.open(snap, "rt", encoding="utf-8") as f:
            OBJ_DIR.store.update(_json.load(f))
        results = [_json.loads(OBJ_DIR.store[k]) for k in sorted(OBJ_DIR.store)]
        print(f"[Fase 3] batch dipakai ulang dari {snap.name}: {len(results)} frame")
        for d in (SURF_DIR,):
            if d.exists():
                shutil.rmtree(d)
        _reset_state()
    else:
        _clean_previous_outputs()
        with Timer("Fase 3 batch"):
            results = run_batch(frame_idxs)
        with gzip.open(snap, "wt", encoding="utf-8") as f:                    # titik henti untuk --reuse-batch
            _json.dump(OBJ_DIR.store, f)
    print(f"[Fase 3] batch selesai: {len(results)} frame")

    passes = run_offline_passes(results)
    with Timer("tulis objects/*.json (sekali)"):
        OBJ_DIR.flush(OUT_DIR / "objects")
    if not skip_checks:
        run_checkpoints(len(frame_idxs), passes, strict=strict)
    with _builtin_open(marker, "w") as f:
        _json.dump({"frames": len(results), "seconds": round(time.time() - t_all, 1)}, f)
    print(f"[Fase 3] SELESAI dalam {time.time() - t_all:.0f} s -> {OUT_DIR}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--reuse-batch", action="store_true", help="lewati YOLO batch, pakai _batch_records.json.gz + points/heights yang ada")
    ap.add_argument("--max-frames", type=int, help="uji cepat: hanya N frame pertama")
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.reuse_batch, a.max_frames)
