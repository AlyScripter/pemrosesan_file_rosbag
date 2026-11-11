#!/usr/bin/env python3
r"""
fase9_road_scene.py
===================
Fase 9 - Export road_scene (kontrak /bev/scene v4, "road-only"). Sisipan BEV ROAD. Menggantikan `export_viewer_data.py` (bagian stream).

INPUT : fase8_road_poly/<idx>.json (polygon, lubang, tipe tepi vq, tebakan ir, jarak tepi ego)
        fase5_payload/<idx>.json (HANYA kunci waktu: "idx" dan "t" -> timeline & sinkronisasi dgn /bev/frame)
        meta/odom.csv + lidar/ (pose ego SE(2))
OUTPUT: fase9_road_scene/<idx:06d>.json  = {"sv":4,"i","t","pose":[x,y,yaw]|null,"road":{p,h,hp,q,ir}|null,"e":[front,back,left,right],"o":[],"wp":[]}
        fase9_road_scene/fase9_done.json, checkpoints/fase9_report.json
CHECKPOINT (otomatis): satu file per idx payload; kunci persis (sv,i,t,pose,road,e,o,wp); sv=4; o==[] dan wp==[] (objek/waypoint MILIK Fase 5);
  t identik dgn payload; road.q sepanjang titik poligon; koordinat dalam grid; pose ada bila odom valid; frame tanpa jalan dilaporkan;
  ukuran rata-rata / maks.
GOAL: record jalan per frame yang tidak bisa menimbulkan objek dobel di Flutter dan terhubung 1:1 ke /bev/frame lewat idx.

Frame tanpa polygon jalan (pose belum valid di awal rekaman / Fase 8 tidak menghasilkan jalan) TETAP dibuat dengan road=null, sehingga
timeline lengkap; klien menahan jalan terakhir ("tak ada jalan baru di frame ini"). Viewer standalone (viewer_data.js) tidak dibawa:
render akhir di Flutter (Fase 11).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from common import Checker, Paths

STREAM_KEYS = ("sv", "i", "t", "pose", "road", "e", "o", "wp")
ROAD_KEYS = {"p", "h", "hp", "q", "ir"}
X_RANGE, Y_RANGE, GRID_TOL = (-15.0, 40.0), (-20.0, 20.0), 0.5
MAX_KB_WARN = 20.0
_T_RE = re.compile(r'"t":\s*(-?[0-9.eE+-]+)')


def r2(v, d=2):
    return round(float(v), d)


def payload_time(path):
    """"t" persis seperti di payload Fase 5 (dari 300 byte pertama; urutan kunci payload: idx, t, ...)."""
    with open(path, "rb") as f:
        m = _T_RE.search(f.read(300).decode("utf-8", "ignore"))
    if not m:
        raise ValueError(f"{path.name}: tak ada kunci 't' di awal berkas")
    return float(m.group(1))


def road_obj(polys, holes, holes_p, vq, ir):
    """road v4. q hanya bila konsisten dgn polygon (panjang string == jumlah titik), ir hanya bila ada."""
    r = {"p": polys, "h": holes, "hp": holes_p}
    if vq and len(vq) == len(polys) and all(len(q) == len(p) for q, p in zip(vq, polys)):
        r["q"] = list(vq)
    if ir:
        r["ir"] = ir
    return r


def build_record(i, t, d, pose):
    road = (d or {}).get("road", {}) or {}
    polys = [[[r2(x), r2(y)] for x, y in ring] for ring in road.get("polygons", [])]
    holes = [[[r2(x), r2(y)] for x, y in h["ring"]] for h in road.get("holes", [])]
    holes_p = [h["parent"] for h in road.get("holes", [])]
    ir = [[[r2(x), r2(y)] for x, y in ring] for ring in road.get("ir", [])]
    e = ((d or {}).get("ego_edge_m")) or {}
    return {"sv": 4, "i": i, "t": round(t, 6), "pose": pose,
            "road": road_obj(polys, holes, holes_p, road.get("vq"), ir) if polys else None,
            "e": [e.get(k) for k in ("front", "back", "left", "right")], "o": [], "wp": []}


def check_record(d, i, name):
    """Daftar pelanggaran kontrak untuk satu record (kosong = OK)."""
    if tuple(d.keys()) != STREAM_KEYS:
        return [f"{name}: kunci {list(d.keys())}"]
    if d["sv"] != 4 or d["o"] != [] or d["wp"] != []:
        return [f"{name}: o/wp tidak kosong atau sv salah"]
    if d["i"] != i or not math.isfinite(d["t"]):
        return [f"{name}: i/t tidak valid"]
    r = d["road"]
    if r:
        if not set(r.keys()) <= ROAD_KEYS:
            return [f"{name}: kunci road tak dikenal {list(r.keys())}"]
        if "q" in r and [len(x) for x in r["q"]] != [len(x) for x in r["p"]]:
            return [f"{name}: panjang road.q != titik polygon"]
        pts = np.array([pt for ring in r["p"] for pt in ring], float)
        if len(pts) and (not np.isfinite(pts).all() or pts[:, 0].min() < X_RANGE[0] - GRID_TOL or pts[:, 0].max() > X_RANGE[1] + GRID_TOL
                         or pts[:, 1].min() < Y_RANGE[0] - GRID_TOL or pts[:, 1].max() > Y_RANGE[1] + GRID_TOL):
            return [f"{name}: koordinat di luar grid"]
    return []


def run(out, force=False, strict=False, max_frames=None):
    import road_polygon_lib as rp
    t_all = time.time()
    out = Path(out)
    d9, d8, pdir = out / "fase9_road_scene", out / "fase8_road_poly", out / "fase5_payload"
    marker = d9 / "fase9_done.json"
    if marker.exists() and not force:
        print(f"[Fase 9] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    for need in (d8 / "fase8_done.json", pdir / "fase5_done.json", out / "meta" / "odom.csv"):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    pay = sorted(int(p.stem) for p in pdir.glob("??????.json") if p.stem.isdigit())
    if max_frames:
        pay = pay[:max_frames]
    if not pay:
        raise SystemExit(f"Tidak ada payload di {pdir}")
    t_l, poses, valid = rp.load_ego_poses(out)
    d9.mkdir(parents=True, exist_ok=True)
    for p in d9.glob("??????.json"):
        p.unlink()
    road_set = {int(p.stem) for p in d8.glob("??????.json")}

    sizes, n_noroad, n_nopose, t_bad = [], 0, 0, []
    for i in pay:
        t = payload_time(pdir / f"{i:06d}.json")
        d = json.loads((d8 / f"{i:06d}.json").read_text(encoding="utf-8")) if i in road_set else None
        ok = i < len(poses) and bool(valid[i])
        pose = [r2(poses[i][0, 2], 3), r2(poses[i][1, 2], 3), r2(math.atan2(poses[i][1, 0], poses[i][0, 0]), 4)] if ok else None
        rec = build_record(i, t, d, pose)
        txt = json.dumps(rec, separators=(",", ":"))
        (d9 / f"{i:06d}.json").write_text(txt, encoding="utf-8")
        sizes.append(len(txt))
        n_noroad += rec["road"] is None
        n_nopose += pose is None
        if i < len(t_l) and abs(t - t_l[i]) > 0.01:
            t_bad.append(i)

    chk = Checker("Fase 9", report_path=Paths(out).checkpoints / "fase9_report.json", strict=strict)
    files = sorted(int(p.stem) for p in d9.glob("??????.json"))
    chk.ok("satu file per idx payload", files == pay, f"{len(files)} file vs {len(pay)} payload")
    bad = []
    for i in pay:
        bad += check_record(json.loads((d9 / f"{i:06d}.json").read_text(encoding="utf-8")), i, f"{i:06d}.json")
    chk.ok("kontrak v4 (kunci, sv=4, o=[] & wp=[], q, koordinat dlm grid)", not bad, f"{len(bad)} pelanggaran; {bad[:3]}")
    chk.ok("t identik dgn payload & dekat timestamp LiDAR (<= 10 ms)", not t_bad, f"{len(t_bad)} idx menyimpang {t_bad[:5]}", fatal=False)
    miss_road = [i for i in pay if i not in road_set]
    chk.ok("frame tanpa JSON jalan <= 10% (sisanya road=null, klien menahan jalan terakhir)", len(miss_road) <= 0.10 * len(pay),
           f"{len(miss_road)} frame (mis. {miss_road[:5]})", fatal=False)
    chk.info("road=null / pose=null", f"{n_noroad} / {n_nopose}")
    kb = np.array(sizes) / 1024
    chk.ok(f"ukuran maks <= {MAX_KB_WARN:.0f} KB", kb.max() <= MAX_KB_WARN, f"rata-rata {kb.mean():.1f} KB | maks {kb.max():.1f} KB", fatal=False)
    chk.finish()
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"frames": len(pay), "road_null": int(n_noroad), "seconds": round(time.time() - t_all, 1)}, fh)
    print(f"[Fase 9] SELESAI dalam {time.time() - t_all:.0f} s -> {d9}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.max_frames)
    sys.exit(0)
