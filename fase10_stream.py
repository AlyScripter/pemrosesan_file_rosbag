#!/usr/bin/env python3
r"""
fase10_stream.py
================
Fase 10 - Streaming playback ke Flutter lewat rosbridge (roslibpy). Menggantikan `fase6_stream_to_flutter.py` (logika SAMA, path & kontrak disatukan).
INI PLAYBACK, BUKAN live-process rosbag. Tidak masuk urutan otomatis `run_pipeline.py` (berjalan real-time, butuh rosbridge): jalankan
`python run_pipeline.py --out . --only 10`  atau  `python fase10_stream.py --out .`

TOPIC -- SATU PEMILIK PER JENIS DATA (mencegah objek/waypoint terender dobel di Flutter):
  /bev/class_color_map  sekali di awal   (dari fase5_payload/meta.json = palet final Fase 4)
  /bev/frame            payload Fase 5 mentah (idx/t/pts/obj/wp)           -- satu-satunya sumber objek, titik, waypoint
  /bev/scene            record Fase 9 road-only v4 (sv,i,t,pose,road,e,o=[],wp=[])   -- hanya geometri jalan
  /vehicle/speed|imu_yaw|distance|nearest|gps   telemetri (dari sync_table_with_waypoints.csv; "nearest" dari wp payload)
Kunci sinkron = idx (BUKAN t). Timeline = fase5_payload. Tiap idx payload punya record scene (road=null bila tak ada jalan baru -> klien menahan jalan terakhir).

PRA-STREAM (checkpoint, tanpa rosbridge; --check-only hanya ini): payload ada; satu scene per idx; kontrak v4 SEMUA scene; kolom telemetri ada;
  t payload naik. PASCA-STREAM: durasi nyata vs asli (<= +5%), drift 1 frame maks, ukuran rata-rata, scene terkirim/hilang.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

# SELF-CONTAINED: hanya library standar + roslibpy. Tidak butuh common.py / fase9_road_scene.py / numpy (cukup 1 file di WSL).

ROSBRIDGE_HOST, ROSBRIDGE_PORT = "localhost", 9090
BEV_TOPIC, SCENE_TOPIC, COLOR_MAP_TOPIC = "/bev/frame", "/bev/scene", "/bev/class_color_map"
MSG_TYPE = "std_msgs/String"          # isi = teks JSON di field "data" -> Flutter: jsonDecode(msg["data"])
TELEMETRY_TOPICS = {"spd": "/vehicle/speed", "yaw": "/vehicle/imu_yaw", "dist": "/vehicle/distance",
                    "near": "/vehicle/nearest", "gps": "/vehicle/gps"}
MAX_FRAME_GAP_S = 2.0
FRAME_RE = re.compile(r"^\d{6}$")
NEED_COLS = ("idx", "global_x", "global_y", "yaw_enu", "velocity_data")
FALLBACK_COLORS = {0: ("Unknown", "#e6007e"), 1: ("Road", "#1f4e9c"), 2: ("Vegetation", "#2e8b2e"),
                   3: ("Building", "#b5651d"), 4: ("Vehicle", "#7b2cbf"), 5: ("Person", "#e00000")}


# ---------------------------------------------------------------------------------------------------------------------
# Utilitas mandiri (salinan ringkas dari common.Checker & fase9_road_scene.check_record agar file ini berdiri sendiri)
# ---------------------------------------------------------------------------------------------------------------------
class CheckpointError(RuntimeError):
    pass


class Paths:
    def __init__(self, out):
        self.out = Path(out)
        self.checkpoints = self.out / "checkpoints"
        self.wp_csv = self.out / "sync_table_with_waypoints.csv"


class Checker:
    def __init__(self, phase, report_path=None, strict=False):
        self.phase, self.report_path, self.strict, self.items = phase, report_path, strict, []

    def ok(self, name, cond, detail="", fatal=True):
        self.items.append({"name": name, "status": "OK" if cond else ("FAIL" if fatal else "WARN"), "detail": str(detail)})
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
            try:
                Path(self.report_path).parent.mkdir(parents=True, exist_ok=True)
                json.dump({"phase": self.phase, "items": self.items, "n_fail": len(fails), "n_warn": len(warns)},
                          open(self.report_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            except OSError:
                pass
        if fails or (self.strict and warns):
            raise CheckpointError(f"{self.phase}: " + "; ".join(i["name"] for i in fails + (warns if self.strict else [])))
        print(f"-> {self.phase}: LULUS ({len(warns)} peringatan)")


STREAM_KEYS = ("sv", "i", "t", "pose", "road", "e", "o", "wp")
ROAD_KEYS = {"p", "h", "hp", "q", "ir"}


def check_record(d, i, name):
    """Pelanggaran kontrak road-only v4 untuk satu scene (kosong = OK)."""
    if tuple(d.keys()) != STREAM_KEYS:
        return [f"{name}: kunci {list(d.keys())}"]
    if d["sv"] != 4 or d["o"] != [] or d["wp"] != []:
        return [f"{name}: o/wp tidak kosong atau sv salah"]
    if d["i"] != i or not isinstance(d["t"], (int, float)) or d["t"] != d["t"] or abs(d["t"]) == float("inf"):
        return [f"{name}: i/t tidak valid"]
    r = d["road"]
    if r:
        if not set(r.keys()) <= ROAD_KEYS:
            return [f"{name}: kunci road tak dikenal {list(r.keys())}"]
        if "q" in r and [len(x) for x in r["q"]] != [len(x) for x in r["p"]]:
            return [f"{name}: panjang road.q != titik polygon"]
    return []


# ---------------------------------------------------------------------------------------------------------------------
def build_index(payload_dir):
    """[(t, idx, path)] terurut waktu dari payload Fase 5."""
    files = [p for p in sorted(Path(payload_dir).glob("*.json")) if FRAME_RE.fullmatch(p.stem)]
    if not files:
        raise RuntimeError(f"Tidak ada payload di {payload_dir} -- jalankan Fase 5 dulu")
    out = []
    for p in files:
        with open(p) as f:
            d = json.load(f)
        out.append((d["t"], d["idx"], p))
    out.sort(key=lambda e: e[0])
    return out


def build_telemetry_lookup(sync_csv):
    """Dari sync_table_with_waypoints.csv: speed, yaw, jarak tempuh kumulatif, GPS per idx (nilai terakhir yang valid bila kosong)."""
    with open(sync_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    spd, yaw, dist, gps = {}, {}, {}, {}
    pgx = pgy = None
    run_d = last_spd = last_lat = last_lon = 0.0
    for r in rows:
        i = int(r["idx"])
        gx, gy = float(r["global_x"]), float(r["global_y"])
        if pgx is not None:
            run_d += ((gx - pgx) ** 2 + (gy - pgy) ** 2) ** 0.5
        pgx, pgy = gx, gy
        dist[i] = run_d
        if r.get("velocity_data"):
            last_spd = float(r["velocity_data"])
        spd[i] = last_spd
        yaw[i] = float(r["yaw_enu"])
        for key, cols in (("lat", ("lat_interp", "lat_data")), ("lon", ("lon_interp", "lon_data"))):
            v = next((r[c] for c in cols if r.get(c)), None)
            if v:
                try:
                    if key == "lat":
                        last_lat = float(v)
                    else:
                        last_lon = float(v)
                except ValueError:
                    pass
        gps[i] = {"lat": last_lat, "lon": last_lon}
    return spd, yaw, dist, gps


def extract_wp_fast(text):
    """"wp" = kunci terakhir payload -> ambil tanpa parse penuh."""
    pos = text.rfind('"wp":')
    if pos == -1:
        return []
    tail = text[pos + 5:].rstrip()
    if not tail.endswith("}"):
        return json.loads(text).get("wp", [])
    try:
        return json.loads(tail[:-1])
    except json.JSONDecodeError:
        return json.loads(text).get("wp", [])


def class_color_map(payload_dir):
    meta = Path(payload_dir) / "meta.json"
    if meta.exists():
        cl = json.load(open(meta, encoding="utf-8")).get("classes")
        if cl:
            return {k: {"name": v["name"], "hex": v["color"]} for k, v in cl.items()}
    return {str(k): {"name": n, "hex": h} for k, (n, h) in FALLBACK_COLORS.items()}


# ---------------------------------------------------------------------------------------------------------------------
def preflight(out, entries, strict=False):
    paths = Paths(out)
    chk = Checker("Fase 10 (pra-stream)", report_path=paths.checkpoints / "fase10_preflight.json", strict=strict)
    scene_dir = Path(out) / "fase9_road_scene"
    idxs = [e[1] for e in entries]
    chk.ok("payload ada & t naik", all(entries[k][0] < entries[k + 1][0] for k in range(len(entries) - 1)), f"{len(entries)} frame")
    have = {int(p.stem) for p in scene_dir.glob("??????.json")} if scene_dir.exists() else set()
    miss = sorted(set(idxs) - have)
    chk.ok("satu scene per idx payload", not miss, f"{len(have)} scene; hilang {len(miss)} {miss[:5]}")
    bad = []
    for i in idxs:
        if i in have:
            bad += check_record(json.loads((scene_dir / f"{i:06d}.json").read_text(encoding="utf-8")), i, f"{i:06d}.json")
    chk.ok("kontrak road-only v4 pada SEMUA scene (tak ada objek/waypoint -> tak ada render dobel)", not bad, f"{len(bad)} pelanggaran {bad[:3]}")
    sync = paths.wp_csv
    cols = next(csv.reader(open(sync, newline=""))) if sync.exists() else []
    chk.ok("kolom telemetri di sync_table_with_waypoints.csv", all(c in cols for c in NEED_COLS), f"kurang: {[c for c in NEED_COLS if c not in cols]}")
    old = Path(out) / "fase5_scene"
    if old.exists():
        chk.ok("tidak ada folder lama fase5_scene (penyebab objek dobel)", False, f"hapus {old}", fatal=False)
    chk.finish()


def stream(entries, scene_dir, topics, telemetry, speed=1.0, loop=False, publish_points=True, make_msg=None, realtime=True, log=print):
    """topics = {"bev","scene","tel":{key:topic}}; make_msg(dict) -> pesan (roslibpy.Message). Return statistik putaran terakhir."""
    spd, yaw, dist, gps = telemetry
    scene_dir = Path(scene_dir)
    n = len(entries)
    log(f"Mulai streaming {n} frame -- {BEV_TOPIC} ({'aktif' if publish_points else 'NONAKTIF'}), {SCENE_TOPIC}, telemetri {len(topics['tel'])} topic "
        f"(speed={speed}x, loop={loop})")
    while True:
        prev_t, w0 = None, time.perf_counter()
        bev_b = scene_b = sent = missing = 0
        drift = 0.0
        for k, (t, idx, path) in enumerate(entries):
            delay = 0.0 if prev_t is None else min(max(0.0, (t - prev_t) / speed), MAX_FRAME_GAP_S)
            prev_t = t
            text = path.read_text()
            p0 = time.perf_counter()
            if publish_points:
                topics["bev"].publish(make_msg({"data": text}))
                bev_b += len(text)
            sp = scene_dir / f"{idx:06d}.json"
            if sp.exists():
                st = sp.read_text(encoding="utf-8")
                topics["scene"].publish(make_msg({"data": st}))
                scene_b += len(st)
                sent += 1
            else:
                missing += 1
            wp = extract_wp_fast(text)
            near = round(min((x ** 2 + y ** 2) ** 0.5 for x, y in wp), 3) if wp else None
            vals = {"spd": spd.get(idx), "yaw": yaw.get(idx), "dist": dist.get(idx), "near": near, "gps": gps.get(idx)}
            for key, tp in topics["tel"].items():
                v = vals.get(key)
                if key == "gps" and isinstance(v, dict):
                    msg = {"idx": idx, "t": t, "lat": v.get("lat"), "lon": v.get("lon"), "value": v}
                else:
                    msg = {"idx": idx, "t": t, "value": v}
                tp.publish(make_msg({"data": json.dumps(msg, separators=(",", ":"))}))
            rem = delay - (time.perf_counter() - p0)
            if rem > 0:
                if realtime:
                    time.sleep(rem)
            else:
                drift = max(drift, -rem)
            if k % 100 == 0 or k == n - 1:
                el = time.perf_counter() - w0
                log(f"  [{k + 1}/{n}] idx={idx} t={t:.3f} bev={len(text) / 1024:.1f}KB spd={vals['spd']} near={near} fps~{(k + 1) / max(el, 1e-9):.1f}")
        wall = time.perf_counter() - w0
        target = (entries[-1][0] - entries[0][0]) / speed
        stats = {"frames": n, "bev_kb": bev_b / n / 1024 if n else 0, "scene_kb": scene_b / max(sent, 1) / 1024, "scene_sent": sent,
                 "scene_missing": missing, "target_s": target, "wall_s": wall, "drift_ms": drift * 1000}
        if not loop:
            return stats


def report(out, st, realtime=True, log=print):
    chk = Checker("Fase 10 (pasca-stream)", report_path=Paths(out).checkpoints / "fase10_report.json")
    chk.ok("semua scene terkirim", st["scene_missing"] == 0, f"{st['scene_sent']} terkirim / {st['scene_missing']} hilang")
    if realtime:
        chk.ok("streaming real-time (selisih <= 5% durasi asli)", st["wall_s"] <= st["target_s"] * 1.05,
               f"asli {st['target_s']:.1f} s | nyata {st['wall_s']:.1f} s | drift maks {st['drift_ms']:.0f} ms", fatal=False)
    chk.info("ukuran rata-rata", f"bev {st['bev_kb']:.1f} KB | scene {st['scene_kb']:.1f} KB")
    chk.finish()


def run(out, force=False, strict=False, host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT, speed=1.0, loop=False, publish_points=True,
        check_only=False, max_frames=None):
    out = Path(out)
    for need in (out / "fase5_payload" / "fase5_done.json", out / "fase9_road_scene" / "fase9_done.json", Paths(out).wp_csv):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    entries = build_index(out / "fase5_payload")
    if max_frames:
        entries = entries[:max_frames]
    dur = entries[-1][0] - entries[0][0]
    print(f"[Fase 10] {len(entries)} frame, durasi asli ~{dur:.1f} s, rata-rata {sum(e[2].stat().st_size for e in entries) / len(entries) / 1024:.1f} KB/frame")
    preflight(out, entries, strict=strict)
    if check_only:
        return
    telemetry = build_telemetry_lookup(Paths(out).wp_csv)
    import roslibpy                                    # import malas: hanya Fase 10 yang butuh
    client = roslibpy.Ros(host=host, port=port)
    client.run()
    print(f"Terhubung ke rosbridge di ws://{host}:{port}")
    mk = roslibpy.Message
    cm_topic = roslibpy.Topic(client, COLOR_MAP_TOPIC, MSG_TYPE)
    cm_topic.advertise()
    cm = json.dumps(class_color_map(out / "fase5_payload"), separators=(",", ":"))
    cm_topic.publish(mk({"data": cm}))
    print(f"class_color_map terkirim ke '{COLOR_MAP_TOPIC}' ({len(cm)} bytes)")
    tops = {"bev": roslibpy.Topic(client, BEV_TOPIC, MSG_TYPE), "scene": roslibpy.Topic(client, SCENE_TOPIC, MSG_TYPE),
            "tel": {k: roslibpy.Topic(client, v, MSG_TYPE) for k, v in TELEMETRY_TOPICS.items()}}
    advertised = [cm_topic, tops["bev"], tops["scene"], *tops["tel"].values()]
    for tp in advertised[1:]:
        tp.advertise()
    try:
        st = stream(entries, out / "fase9_road_scene", tops, telemetry, speed, loop, publish_points, make_msg=mk)
        report(out, st)
    except KeyboardInterrupt:
        print("\nDihentikan manual (Ctrl+C).")
    finally:
        for tp in advertised:
            tp.unadvertise()
        client.terminate()
        print("Koneksi ke rosbridge ditutup.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=ROSBRIDGE_HOST)
    ap.add_argument("--port", type=int, default=ROSBRIDGE_PORT)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-points", action="store_true", help="jangan kirim /bev/frame (hemat bandwidth; wp tetap dibaca utk telemetri)")
    ap.add_argument("--check-only", action="store_true", help="hanya checkpoint pra-stream (tanpa rosbridge)")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args()
    run(a.out, strict=a.strict, host=a.host, port=a.port, speed=a.speed, loop=a.loop, publish_points=not a.no_points,
        check_only=a.check_only, max_frames=a.max_frames)
    sys.exit(0)
