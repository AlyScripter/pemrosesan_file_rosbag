#!/usr/bin/env python3
r"""
fase8_road_polygon.py
=====================
Fase 8 - Polygon jalan berbasis bukti (R0 ringan + R1a + R1b + R1c). Sisipan BEV ROAD. Menggantikan `r1_run.py`.
Algoritma = `road_evidence_lib.py` + `road_polygon_lib.py` (rev 3) kode asli Anda, DISALIN TANPA UBAH.

INPUT : fase7_road_mask/masks/<idx>.npz  |  fase3_deteksi/objects/<idx>.json (hanya dibaca: jejak objek = TAK-TAHU)
        meta/odom.csv + lidar/ (pose ego, untuk prior temporal & IoU antar-frame)
OUTPUT: fase8_road_poly/<idx:06d>.json  {idx,t,road{polygons,holes,res_equiv_m,vq,inf,ir},ego_edge_m,road_type}
        fase8_road_poly/report.csv (metrik per frame), hole_causes.csv (R0: penyebab lubang), summary.csv,
        fase8_done.json, checkpoints/fase8_report.json
CHECKPOINT (otomatis): JSON == npz (gagal <= 2%); semua masker punya *_hits; lubang non-pulau turun >= 70% vs pipeline lama (rev 2);
  recall vs bukti frame tetangga tidak turun; konflik rintangan <= 0,05; sel tebakan (inferred) median <= 25%; IoU antar-frame >= 0,9;
  pergeseran tepi median <= 0,1 m / maks <= 0,3 m; IoU R1c vs hasil perbaikan >= 0,97; kekasaran tepi turun >= 40%; JSON <= 6 KB;
  ego di atas jalan >= 95% (di luar pemanasan); >= 80% lubang punya penyebab (R0).
GOAL: polygon jalan yang tidak bolong dan tidak bergerigi, jujur terhadap bukti (lubang hanya diisi bila tak ada bukti BUKAN-JALAN),
  dengan tepi bertipe (berbukti / batas sensor) dan area tebakan (inferred) yang dilaporkan.

Keputusan penyatuan:
  * Perbaikan lubang & penghalusan tepi memakai R1Config bawaan (nilai Anda). Prior temporal kausal (log-odds) AKTIF secara default
    (bagian rencana R1b); matikan dgn --no-temporal untuk ablasi (R4).
  * Pipeline lama (rev 2) tetap dijalankan per frame HANYA untuk membandingkan checkpoint (hasilnya tidak ditulis). --no-baseline = lewati.
  * Paralel: potongan frame berurutan per proses. Prior temporal dipanaskan 15 frame sebelum tiap potongan (peluruhan 0,85^15 x 4 < ambang 0,5
    -> hasil identik dgn serial; diuji).
  * PNG perbandingan hanya dengan --debug-plots (r1_run.plot_compare tidak dibawa; diagnosis lewat report.csv).
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import road_evidence_lib as rel
import road_polygon_lib as rp
from common import Checker, Paths, Timer

try:
    from tqdm import tqdm
except ImportError:                             # pragma: no cover
    def tqdm(x, **kw):
        return x

WARMUP_FRAMES = 15
MIN_CHUNK = 40
MAX_FAIL_FRAC = 0.02
RECALL_K = 10
RECALL_STRIDE = 5


# ---------------------------------------------------------------------------------------------------------------------
# Metrik (dari r1_run.py)
# ---------------------------------------------------------------------------------------------------------------------
def holes_nonisland(raster, notm, res, min_area_m2=0.2):
    """Lubang tertutup pada raster jalan yang TIDAK mengandung bukti BUKAN-JALAN (= bolong yang tampak aneh)."""
    holes = ndi.binary_fill_holes(raster) & ~raster
    if not holes.any():
        return 0
    lbl, n = ndi.label(holes, structure=np.ones((3, 3), bool))
    cnt = 0
    for i in range(1, n + 1):
        c = lbl == i
        if c.sum() * res * res >= min_area_m2 and not (notm & c).any():
            cnt += 1
    return cnt


def all_rings(road):
    return [np.asarray(p, float) for p in road.get("polygons", [])] + [np.asarray(h["ring"], float) for h in road.get("holes", [])]


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def neighbor_recall(cfg, i, j, final, poses, valid, x_range, y_range, res, rng_m=15.0):
    """Persen sel yang JALAN pada frame tetangga j (bukti, di-warp ke frame i; radius <= rng_m) yang tercakup `final` frame i."""
    if not (0 <= i < len(valid) and 0 <= j < len(valid)) or not (valid[i] and valid[j]):
        return None
    p = cfg.masks_dir / f"{j:06d}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        mj = d[cfg.mask_key].astype(np.float32)
    w = rel.warp_grid(mj, np.linalg.inv(poses[j]) @ poses[i], x_range, y_range, res) > 0.5
    w &= rel.range_map(x_range, y_range, res) <= rng_m
    n = int(w.sum())
    return float((w & final).sum() / n) if n >= 50 else None


# ---------------------------------------------------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------------------------------------------------
_W = {}


def make_cfgs(out, temporal=True, refine=True, baseline=True, **r1_over):
    out = Path(out)
    masks = out / "fase7_road_mask" / "masks"
    cfg_r = rp.PolygonConfig(root=out, masks_dir=masks, out_dir=out / "fase8_road_poly")
    cfg_r.r1 = rel.R1Config(temporal=temporal, refine=refine, objects_dir=out / "fase3_deteksi" / "objects", **r1_over)
    cfg_b = rp.PolygonConfig(root=out, masks_dir=masks) if baseline else None
    return cfg_r, cfg_b


def _init_worker(out, opts):
    cfg_r, cfg_b = make_cfgs(out, **opts)
    _, poses, valid = rp.load_ego_poses(out)
    _W.update(cfg_r=cfg_r, cfg_b=cfg_b, poses=poses, valid=valid)


def _process_one(cfg_r, cfg_b, poses, valid, state, i, recall):
    rr, ir = rp.process_frame(cfg_r, i, return_internal=True, state=state)
    xr, yr, res = ir["x_range"], ir["y_range"], ir["res"]
    rr_r = rp.polygons_to_mask(rr["road"], xr, yr, res)
    ev = ir["ev"]
    notm = ev == rel.NOT_ROAD
    obst = ir["aux"]["obstacle"] & ~ir["aux"]["footprint"]          # rintangan di jejak objek = objeknya sendiri, bukan konflik
    c = rel.classify_holes(ev, ir["aux"], xr, yr, res)
    disp = ir["r1"]["disp"]
    row = {"idx": i, "t": rr["t"],
           "holes_raw": c["total"], "holes_r1": holes_nonisland(rr_r, notm, res), "drawn_holes_r1": len(rr["road"]["holes"]),
           "area_r1_m2": float(rr_r.sum() * res * res), "inferred_frac": rr["road"]["inf"],
           "obst_conflict": float((rr_r & obst).sum() / max(rr_r.sum(), 1)),
           "iou_refine_vs_repaired": iou(rr_r, ir["smooth"]),
           "disp_median_m": float(np.median(disp)) if len(disp) else np.nan,
           "disp_max_m": float(disp.max()) if len(disp) else np.nan,
           "rough_raw_outer": rel.roughness(ir["r1"]["dense_raw"]),
           "rough_r1_outer": rel.roughness([np.asarray(p, float) for p in rr["road"]["polygons"]]),
           "edge_observed_frac": sum(v.count("0") for v in rr["road"]["vq"]) / max(sum(len(v) for v in rr["road"]["vq"]), 1),
           "json_bytes_r1": len(json.dumps(rr, separators=(",", ":"))),
           "ego_on_road": bool(rr["ego_edge_m"]["ego_on_road"]), "n_polygons": len(rr["road"]["polygons"]),
           "n_objects": ir["n_boxes"], "has_hits": ir["aux"]["has_hits"]}
    row.update({f"cause_{k}": v for k, v in c.items()})
    if cfg_b is not None:
        rb, ib = rp.process_frame(cfg_b, i, return_internal=True)
        rb_r = rp.polygons_to_mask(rb["road"], xr, yr, res)
        row.update({"holes_base": holes_nonisland(rb_r, notm, res), "drawn_holes_base": len(rb["road"]["holes"]),
                    "area_base_m2": float(rb_r.sum() * res * res), "iou_r1_vs_base": iou(rb_r, rr_r),
                    "rough_base": rel.roughness(all_rings(rb["road"])), "rough_r1": rel.roughness(all_rings(rr["road"])),
                    "json_bytes_base": len(json.dumps(rb, separators=(",", ":")))})
    if recall and poses is not None:
        for tag, j in (("past", i - RECALL_K), ("future", i + RECALL_K)):
            row[f"recall_{tag}_r1"] = neighbor_recall(cfg_r, i, j, rr_r, poses, valid, xr, yr, res)
            if cfg_b is not None:
                row[f"recall_{tag}_base"] = neighbor_recall(cfg_r, i, j, rb_r, poses, valid, xr, yr, res)
    return rr, row


def _run_chunk(job):
    warm, frames, stride_offset = job
    cfg_r, cfg_b, poses, valid = _W["cfg_r"], _W["cfg_b"], _W["poses"], _W["valid"]
    state = rel.TemporalState(cfg_r.r1, poses, valid) if cfg_r.r1.temporal else None
    for i in warm:                                           # pemanasan prior temporal: hasil dibuang
        try:
            rp.process_frame(cfg_r, int(i), state=state)
        except Exception:                                    # noqa: BLE001
            pass
    rows, failed = [], []
    for n, i in enumerate(frames):
        try:
            rr, row = _process_one(cfg_r, cfg_b, poses, valid, state, int(i), recall=((stride_offset + n) % RECALL_STRIDE == 0))
            rp.save_frame_json(cfg_r, rr)
            rows.append(row)
        except Exception as e:                               # noqa: BLE001
            failed.append((int(i), repr(e)))
    return rows, failed


def make_jobs(frames, workers, temporal):
    n = len(frames)
    k = max(1, min(workers * 2, n // MIN_CHUNK))
    b = np.linspace(0, n, k + 1).round().astype(int)
    jobs = []
    for a in range(k):
        lo, hi = b[a], b[a + 1]
        if hi <= lo:
            continue
        warm = frames[max(0, lo - WARMUP_FRAMES):lo] if temporal else []
        jobs.append((warm, frames[lo:hi], int(lo)))
    return jobs


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------------------------------------------------
def _med(s):
    s = pd.Series(s).dropna()
    return float(s.median()) if len(s) else float("nan")


def run_checkpoints(out, S, failed, n_expected, cfg_r, poses, valid, t_start, strict=False):
    d8 = Path(out) / "fase8_road_poly"
    chk = Checker("Fase 8", report_path=Paths(out).checkpoints / "fase8_report.json", strict=strict)
    n_json = len([p for p in d8.glob("??????.json")])
    chk.ok("jumlah JSON == jumlah masker", n_json == n_expected - len(failed), f"{n_json} dari {n_expected} (gagal {len(failed)})")
    chk.ok(f"frame gagal <= {MAX_FAIL_FRAC:.0%}", len(failed) <= MAX_FAIL_FRAC * max(n_expected, 1), f"{len(failed)}; contoh {failed[:2]}")
    if not len(S):
        chk.ok("ada hasil", False, "tidak ada frame berhasil")
        chk.finish()
        return
    chk.ok("semua masker membawa *_hits (BUKAN-JALAN terisi)", bool(S["has_hits"].all()), f"{int((~S['has_hits']).sum())} frame tanpa hits")

    if "holes_base" in S:
        hb, hr = int(S["holes_base"].sum()), int(S["holes_r1"].sum())
        red = 1 - hr / hb if hb else float("nan")
        if hb:
            chk.ok("lubang non-pulau turun >= 70% vs pipeline lama", red >= 0.7, f"{hb} -> {hr} ({red:.0%})", fatal=False)
        else:
            chk.info("lubang non-pulau (lama -> R1)", f"{hb} -> {hr} (tak ada lubang awal: tak bisa dinilai)")
        chk.info("lubang tergambar (lama -> R1)", f"{int(S['drawn_holes_base'].sum())} -> {int(S['drawn_holes_r1'].sum())}")
        for tag in ("past", "future"):
            b, r = _med(S.get(f"recall_{tag}_base", [])), _med(S.get(f"recall_{tag}_r1", []))
            if not np.isnan(b):
                chk.ok(f"recall vs bukti {tag} (1 dtk) tidak turun", r >= b - 0.005, f"lama {b:.3f} -> R1 {r:.3f}", fatal=False)
        chk.info("kekasaran vs pipeline lama (rad/m)", f"{S['rough_base'].mean():.3f} -> {S['rough_r1'].mean():.3f}")
        chk.info("IoU R1 vs pipeline lama (median)", f"{_med(S['iou_r1_vs_base']):.3f}")

    oc = float(S["obst_conflict"].mean())
    chk.ok("konflik rintangan LiDAR <= 0,05", oc <= 0.05, f"{oc:.3f}", fatal=False)
    inf = S["inferred_frac"]
    chk.ok("sel tebakan (inferred) median <= 25%", inf.median() <= 0.25, f"median {inf.median():.1%} | p95 {inf.quantile(.95):.1%}", fatal=False)

    st = rp.temporal_stability(cfg_r, poses, valid, indices=list(S["idx"]), lags=(1,), log=lambda *_: None)
    if len(st):
        m = _med(st["iou"])
        chk.ok(f"IoU antar-frame lag 1 >= {cfg_r.target_iou}", m >= cfg_r.target_iou, f"median {m:.3f} ({len(st)} pasangan)", fatal=False)
    else:
        chk.ok("IoU antar-frame lag 1 terukur", False, "0 pasangan berurutan", fatal=False)

    dm, dx = _med(S["disp_median_m"]), float(S["disp_max_m"].quantile(0.95))
    chk.ok("pergeseran tepi: median <= 0,1 m dan p95(maks) <= 0,3 m", dm <= 0.1 and dx <= 0.3, f"{dm:.3f} m / {dx:.3f} m", fatal=False)
    chk.ok("IoU polygon R1c vs hasil perbaikan >= 0,97", _med(S["iou_refine_vs_repaired"]) >= 0.97, f"median {_med(S['iou_refine_vs_repaired']):.3f}", fatal=False)
    rb, rr = float(S["rough_raw_outer"].mean()), float(S["rough_r1_outer"].mean())
    if rb:
        chk.ok("kekasaran tepi turun >= 40% (sebelum -> sesudah R1c)", 1 - rr / rb >= 0.4, f"{rb:.3f} -> {rr:.3f} ({1 - rr / rb:.0%})", fatal=False)
    kb = S["json_bytes_r1"] / 1024
    chk.ok("ukuran JSON <= 6 KB/frame", kb.max() <= 6.0, f"median {kb.median():.1f} KB | maks {kb.max():.1f} KB", fatal=False)
    late = S[S["t"] >= t_start + cfg_r.warmup_s]
    if len(late):
        chk.ok(f"ego di atas jalan >= {cfg_r.target_ego_on_road:.0%} (di luar pemanasan)", late["ego_on_road"].mean() >= cfg_r.target_ego_on_road,
               f"{late['ego_on_road'].mean():.1%}", fatal=False)
    chk.ok("poligon kosong <= 2% frame", float((S["n_polygons"] == 0).mean()) <= 0.02, f"{int((S['n_polygons'] == 0).sum())} frame")
    chk.info("tepi berbukti (fraksi titik vq='0', median)", f"{S['edge_observed_frac'].median():.0%}")
    tot = S[[c for c in S.columns if c.startswith("cause_")]].sum()
    n_h = int(tot.get("cause_total", 0))
    if n_h:
        explained = 1 - int(tot.get("cause_ambigu", 0)) / n_h
        chk.ok("R0: >= 80% lubang mask mentah punya penyebab", explained >= 0.8,
               f"{explained:.0%} dari {n_h} lubang | " + ", ".join(f"{k[6:]}={int(v)}" for k, v in tot.items() if k != "cause_total"), fatal=False)
    else:
        chk.info("R0: lubang pada mask mentah", "0")
    chk.finish()


# ---------------------------------------------------------------------------------------------------------------------
def run(out, force=False, strict=False, workers=None, max_frames=None, temporal=True, refine=True, baseline=True, debug_plots=False):
    t_all = time.time()
    out = Path(out)
    d8 = out / "fase8_road_poly"
    marker = d8 / "fase8_done.json"
    if marker.exists() and not force:
        print(f"[Fase 8] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    for need in (out / "fase7_road_mask" / "fase7_done.json", out / "meta" / "odom.csv"):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    opts = dict(temporal=temporal, refine=refine, baseline=baseline)
    cfg_r, _ = make_cfgs(out, **opts)
    frames = rp.list_mask_frames(cfg_r)
    if max_frames:
        frames = frames[:max_frames]
    if not frames:
        raise SystemExit(f"Tidak ada masker di {cfg_r.masks_dir}")
    t_l, poses, valid = rp.load_ego_poses(out)
    if d8.exists():
        for p in d8.glob("??????.json"):
            p.unlink()
    d8.mkdir(parents=True, exist_ok=True)
    n_obj = len(list((out / "fase3_deteksi" / "objects").glob("*.json"))) if (out / "fase3_deteksi" / "objects").exists() else 0
    print(f"[Fase 8] {len(frames)} frame | objek Fase 3: {n_obj} berkas | temporal={temporal} refine={refine} baseline={baseline}")

    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    jobs = make_jobs(frames, workers, temporal)
    rows, failed = [], []
    with Timer("Fase 8 polygon"):
        if workers == 1 or len(jobs) == 1:
            _init_worker(out, opts)
            for j in tqdm(jobs, desc="Fase 8"):
                r, f = _run_chunk(j)
                rows += r
                failed += f
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(out), opts)) as ex:
                for r, f in tqdm(ex.map(_run_chunk, jobs), total=len(jobs), desc="Fase 8"):
                    rows += r
                    failed += f
    S = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    S.to_csv(d8 / "report.csv", index=False)
    S[["idx"] + [c for c in S.columns if c.startswith("cause_")]].to_csv(d8 / "hole_causes.csv", index=False)
    t_start = float(t_l[int(np.argmax(valid))]) if valid.any() else 0.0
    run_checkpoints(out, S, sorted(failed), len(frames), cfg_r, poses, valid, t_start, strict=strict)
    if debug_plots:
        print("[Fase 8] --debug-plots: pakai r1_run.py (arsip) bila perlu PNG perbandingan; report.csv memuat semua metrik.")
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"frames": int(len(S)), "failed": len(failed), "temporal": temporal, "seconds": round(time.time() - t_all, 1)}, fh)
    print(f"[Fase 8] SELESAI dalam {time.time() - t_all:.0f} s -> {d8}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-temporal", action="store_true", help="matikan prior temporal (ablasi R4)")
    ap.add_argument("--no-refine", action="store_true", help="matikan R1c (hanya perbaikan lubang)")
    ap.add_argument("--no-baseline", action="store_true", help="lewati pembanding pipeline lama (lebih cepat, checkpoint perbandingan hilang)")
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.workers, a.max_frames, not a.no_temporal, not a.no_refine, not a.no_baseline)
    sys.exit(0)
