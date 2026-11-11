#!/usr/bin/env python3
r"""
run_pipeline.py
===============
Menjalankan fase pipeline BEV berurutan & otomatis. Tiap fase: cek hasil lama (lewati bila sudah ada,
kecuali --force), jalankan, lalu checkpoint otomatis. Checkpoint FAIL -> pipeline berhenti.

Fase yang sudah ada di paket ini:  1 -> 2 -> 1b -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
(Fase 2 menyusun calib_config.json otomatis dari camera_info.json + rig_extrinsics.json.)
(10 = streaming: jalankan terpisah dgn --only 10, tidak ada di urutan otomatis; 11 = Flutter. Lihat ALUR_PIPELINE_TERPADU.md)

CONTOH:
    python run_pipeline.py --bag /home/kali/rosbag_analysis/Rosbag_Rute_02.bag --out /home/kali/rosbag_analysis/output_rute02
    python run_pipeline.py --out D:\KULIAH SMT 7\MAGANG\output_rute02 --from 1b      # bag tak perlu bila Fase 1 sudah jadi
    python run_pipeline.py --bag ... --out ... --only 1 --force
"""
from __future__ import annotations

import argparse
import sys
import time

import fase1_extract_sync
import fase1b_waypoints
import fase2_calibration

ORDER = ["1", "2", "1b", "3", "4", "5", "6", "7", "8", "9"]   # urutan eksekusi (1b butuh gps.csv dari Fase 1; 3 butuh 1 & 2; 4 butuh 3; 5 butuh 1b, 3, 4; 6 butuh 1 & 4; 7 butuh 1, 2, 4, 6; 8 butuh 3, 7; 9 butuh 5, 8)


def phase_1(a):
    if not a.bag and not _exists(a, "gps_csv"):
        raise SystemExit("Fase 1 butuh --bag (hasil lama tidak ditemukan).")
    if a.bag:
        fase1_extract_sync.run(a.bag, a.out, force=a.force, save_unmatched=a.save_unmatched, strict=a.strict)


def phase_2(a):
    fase2_calibration.run(a.out, rig_path=a.rig_calib, force=a.force, strict=a.strict)


def phase_1b(a):
    fase1b_waypoints.run(a.out, force=a.force, plots=a.debug_plots, strict=a.strict)


def phase_3(a):
    import fase3_detection                      # import malas: torch/ultralytics berat, hanya dimuat bila Fase 3 jalan
    fase3_detection.run(a.out, force=a.force, strict=a.strict, reuse_batch=a.reuse_batch, max_frames=a.max_frames)


def phase_4(a):
    import fase4_segmentation
    fase4_segmentation.run(a.out, force=a.force, strict=a.strict, workers=a.workers, max_frames=a.max_frames)


def phase_5(a):
    import fase5_payload
    fase5_payload.run(a.out, force=a.force, strict=a.strict, workers=a.workers, max_frames=a.max_frames)


def phase_6(a):
    import fase6_road_camera                    # import malas: torch/transformers hanya dimuat bila Fase 6 jalan
    fase6_road_camera.run(a.out, force=a.force, strict=a.strict, workers=a.io_threads, max_frames=a.max_frames, fp32=a.fp32)


def phase_7(a):
    import fase7_road_mask
    fase7_road_mask.run(a.out, force=a.force, strict=a.strict, workers=a.workers, max_frames=a.max_frames)


def phase_8(a):
    import fase8_road_polygon
    fase8_road_polygon.run(a.out, force=a.force, strict=a.strict, workers=a.workers, max_frames=a.max_frames,
                           temporal=not a.no_temporal, refine=not a.no_refine, baseline=not a.no_baseline)


def phase_9(a):
    import fase9_road_scene
    fase9_road_scene.run(a.out, force=a.force, strict=a.strict, max_frames=a.max_frames)


def phase_10(a):
    import fase10_stream                        # streaming real-time ke rosbridge; TIDAK ada di ORDER, jalankan dgn --only 10
    fase10_stream.run(a.out, strict=a.strict, host=a.host, port=a.port, speed=a.speed, loop=a.loop,
                      publish_points=not a.no_points, check_only=a.check_only, max_frames=a.max_frames)


PHASES = {"1": phase_1, "2": phase_2, "1b": phase_1b, "3": phase_3, "4": phase_4, "5": phase_5, "6": phase_6, "7": phase_7, "8": phase_8, "9": phase_9, "10": phase_10}


def _exists(a, attr):
    from common import Paths
    return getattr(Paths(a.out), attr).exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="jalankan satu fase saja, mis. 1b")
    ap.add_argument("--from", dest="from_phase", help="mulai dari fase ini")
    ap.add_argument("--force", action="store_true", help="ulang walau hasil sudah ada")
    ap.add_argument("--strict", action="store_true", help="peringatan checkpoint dianggap gagal")
    ap.add_argument("--rig-calib", help="Fase 2: file ekstrinsik LiDAR->kamera (default rig_extrinsics.json)")
    ap.add_argument("--debug-plots", action="store_true", help="simpan PNG checkpoint (default mati = lebih cepat)")
    ap.add_argument("--save-unmatched", action="store_true", help="Fase 1: simpan semua rgb/depth (perilaku lama)")
    ap.add_argument("--reuse-batch", action="store_true", help="Fase 3: pakai snapshot batch YOLO (lewati YOLO, ulang pass offline saja)")
    ap.add_argument("--max-frames", type=int, default=None, help="Fase 3/4: batasi jumlah frame (uji cepat)")
    ap.add_argument("--workers", type=int, default=None, help="Fase 4/5/7/8: jumlah proses paralel (default min(6, CPU-1))")
    ap.add_argument("--fp32", action="store_true", help="Fase 6: paksa fp32 (jaga-jaga bila fp16 bermasalah)")
    ap.add_argument("--no-temporal", action="store_true", help="Fase 8: matikan prior temporal (ablasi)")
    ap.add_argument("--no-refine", action="store_true", help="Fase 8: matikan penghalusan tepi R1c (ablasi)")
    ap.add_argument("--no-baseline", action="store_true", help="Fase 8: lewati pembanding pipeline lama (lebih cepat)")
    ap.add_argument("--host", default="localhost", help="Fase 10: host rosbridge")
    ap.add_argument("--port", type=int, default=9090, help="Fase 10: port rosbridge")
    ap.add_argument("--speed", type=float, default=1.0, help="Fase 10: kecepatan playback")
    ap.add_argument("--loop", action="store_true", help="Fase 10: ulang terus")
    ap.add_argument("--no-points", action="store_true", help="Fase 10: jangan kirim /bev/frame")
    ap.add_argument("--check-only", action="store_true", help="Fase 10: hanya checkpoint pra-stream")
    ap.add_argument("--io-threads", type=int, default=None, help="Fase 6: thread baca/tulis gambar (default 4)")
    a = ap.parse_args()

    todo = ORDER
    if a.only:
        todo = [a.only]
    elif a.from_phase:
        todo = ORDER[ORDER.index(a.from_phase):]
    bad = [p for p in todo if p not in PHASES]
    if bad:
        raise SystemExit(f"Fase tidak dikenal: {bad}. Tersedia: {ORDER}")

    t_all = time.time()
    for p in todo:
        print(f"\n{'=' * 70}\n FASE {p}\n{'=' * 70}")
        t0 = time.time()
        PHASES[p](a)
        print(f"[Fase {p}] selesai dalam {time.time() - t0:.1f} s")
    print(f"\nSEMUA FASE SELESAI ({time.time() - t_all:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
