#!/usr/bin/env python3
r"""
fase6_road_camera.py
====================
Fase 6 - Segmentasi jalan dari kamera (SETELAH Fase 5). Sisipan BEV ROAD. Menggantikan `02a_segment_road.py`.

INPUT : rgb/ + sync_table.csv (Fase 1), daftar frame = fase4_segmentasi/points (sama dengan Fase 3-5)
OUTPUT: fase6_seg_cam/<idx:06d>.png  uint8: 0 = abaikan, 1 = jalan, 2 = bukan-jalan   (idx = idx frame LiDAR, 1:1 dgn Fase 4)
        fase6_seg_cam/summary_frames.csv, fase6_seg_cam/fase6_done.json, checkpoints/fase6_report.json
CHECKPOINT (otomatis): jumlah PNG == jumlah frame yang punya RGB; tak ada frame yang seluruhnya 0 (>2%);
  fraksi "jalan" di sepertiga bawah gambar (median) >= 30%; frame tanpa RGB dilaporkan (bukan error: Fase 7 menahan bukti kamera).
GOAL: bukti jalan dari kamera sebagai sumber kedua yang independen dari LiDAR (dipakai Fase 7).

Peta kelas -> kode (sama dengan kode asli): road -> 1; trotoar/bangunan/dinding/pagar/tiang/rambu/vegetasi/terrain -> 2;
sisanya (mobil, orang, langit, ...) -> 0 (jalan di bawah mobil parkir tidak dicoret).

Beda dengan skrip asli (hasil identik, hanya lebih cepat):
  * Semua frame otomatis (tanpa --frames), argmax + LUT kode dilakukan di GPU (transfer 1 byte/piksel).
  * PENGATUR OTOMATIS (tune): sebelum memproses, 4 frame contoh dijalankan dalam mode fp32 murni (REFERENSI), amp-fp16 dan
    half-fp16, lalu dipilih yang tercepat DENGAN SYARAT hasilnya sama dengan referensi (kesamaan piksel >= 99,5%, IoU jalan
    >= 0,98, logit finite). Gagal syarat (mis. NaN fp16 di GTX 16xx) -> otomatis tetap fp32 murni. Ukuran batch dipilih dari
    pengukuran (bukan tebakan), sehingga jatuhnya ke RAM sistem karena VRAM habis (4,8 s/frame di GTX 1650) tidak terjadi.
  * Baca + resize gambar di thread terpisah dengan prefetch TERBATAS (RAM tidak membengkak), tumpang-tindih dengan GPU.
  * PNG ditulis atomik (tmp -> rename) -> aman di-resume. Tanpa pratinjau/overlay kecuali --debug-plots.
  * torch/transformers dimuat malas: hanya Fase 6 yang membutuhkannya.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import numpy as np

from common import Checker, Timer, resolve_by_name

try:
    from tqdm import tqdm
except ImportError:                             # pragma: no cover
    def tqdm(x, **kw):
        return x

MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
INPUT_WIDTH = 640             # disesuaikan dengan resolusi native kamera (640x360 -> 640x384, hemat VRAM >60%)
BATCH = 2
SEG_IGNORE, SEG_ROAD, SEG_NONROAD = 0, 1, 2
_ROAD_NAMES = {"road", "service-lane", "parking"}
_NONROAD_NAMES = {"sidewalk", "building", "wall", "fence", "guard rail", "guard-rail", "pole", "polegroup",
                  "traffic light", "traffic sign", "traffic-sign-frame", "vegetation", "terrain", "bridge",
                  "tunnel", "other-barrier", "barrier", "curb", "rail track", "ground"}

MIN_BOTTOM_ROAD_FRAC = 0.30   # median fraksi jalan di sepertiga bawah gambar
MAX_EMPTY_FRAC = 0.02         # maksimum frame yang seluruhnya "abaikan"
MAX_ROAD_MEAN = 0.90          # rata-rata piksel jalan di atas ini = model rusak (mis. NaN fp16 -> semua 'road')
MIN_NONROAD_MEAN = 0.02
TUNE_FRAMES = 4               # jumlah frame contoh untuk pengatur otomatis
MIN_AGREE = 0.995             # kesamaan piksel minimum terhadap referensi fp32 murni
MIN_ROAD_IOU = 0.98           # IoU kelas jalan minimum terhadap referensi
MIN_GAIN = 0.10               # mode non-fp32 dipakai hanya bila >= 10% lebih cepat (selebihnya fp32 murni)


# ---------------------------------------------------------------------------------------------------------------------
# Peta kelas
# ---------------------------------------------------------------------------------------------------------------------
def name_to_code(name):
    last = str(name).lower().strip().split("--")[-1]       # dukung label gaya Mapillary: construction--flat--road
    if last in _ROAD_NAMES:
        return SEG_ROAD
    if last in _NONROAD_NAMES:
        return SEG_NONROAD
    return SEG_IGNORE


def build_lut(id2label):
    """LUT id kelas model -> kode 0/1/2. Gagal keras bila model tak punya kelas 'road'."""
    ids = [int(k) for k in id2label]
    lut = np.zeros(max(ids) + 1, np.uint8)
    for k, v in id2label.items():
        lut[int(k)] = name_to_code(v)
    if not (lut == SEG_ROAD).any():
        raise ValueError(f"Model tidak punya kelas 'road': {sorted(set(map(str, id2label.values())))[:12]} ...")
    return lut


# ---------------------------------------------------------------------------------------------------------------------
# Model (dimuat malas)
# ---------------------------------------------------------------------------------------------------------------------
def make_predictor(model_name=MODEL_NAME, device=None, width=INPUT_WIDTH, fp32=False, model=None):
    """Return predict(list[HxWx3 uint8]) -> list[HxW uint8 kode 0/1/2]. LUT & argmax dijalankan di GPU.

    Mode inferensi (dipilih predict.tune):
      fp32 = bobot & aktivasi fp32 (REFERENSI)
      amp = bobot fp32 + autocast fp16
      half = bobot fp16 murni (sangat hemat memori & cepat di Jetson Tegra GPU)
    Pengaman: logit non-finite -> beralih permanen ke mode cadangan.
    predict.prep(img) meresize di thread pemuat; predict menerima gambar mentah ATAU hasil prep."""
    import copy
    import gc
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if model is None:
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as e:
            raise SystemExit("Fase 6 butuh: pip install torch transformers pillow  (belum terpasang)") from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.startswith("cuda")
    use_half = cuda and not fp32
    if model is None:
        dtype = torch.float16 if use_half else torch.float32
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device).eval()
    if cuda:
        torch.backends.cudnn.benchmark = True            # ukuran input tetap -> autotune sekali
    lut = build_lut({int(k): v for k, v in model.config.id2label.items()})
    lut_t = torch.from_numpy(lut).to(device)
    tensor_dtype = torch.float16 if use_half else torch.float32
    mean32 = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=tensor_dtype).view(1, 3, 1, 1)
    std32 = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=tensor_dtype).view(1, 3, 1, 1)
    w_r = int(round(width / 32)) * 32
    state = {"mode": "half" if use_half else "fp32", "first": True, "m16": None, "t_pred": 0.0, "n_pred": 0}
    amp_dtype = torch.float16 if cuda else torch.bfloat16
    print(f"[Fase 6] model {model_name} di {device} ({'fp16 (hemat VRAM)' if use_half else 'fp32'}; input width={w_r})")

    def prep(im):
        if isinstance(im, tuple):
            return im
        H, W = im.shape[:2]
        h_r = max(int(round(width * H / W / 32)) * 32, 64)
        return np.asarray(Image.fromarray(im).resize((w_r, h_r), Image.BILINEAR)), (H, W)

    def _logits(x, mode):
        xn = (x - mean32) / std32
        if use_half or mode == "half":
            if not use_half:
                if state["m16"] is None:
                    state["m16"] = copy.deepcopy(model).half()
                return state["m16"](pixel_values=xn.half()).logits.float()
            return model(pixel_values=xn).logits.float()
        if mode == "amp":
            with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=amp_dtype):
                return model(pixel_values=xn).logits.float()
        return model(pixel_values=xn).logits

    @torch.inference_mode()
    def _run(images, mode):
        """-> (list kode HxW uint8, logit finite?, kelas unik batch)"""
        items = [prep(im) for im in images]
        H, W = items[0][1]
        x = torch.from_numpy(np.stack([a for a, _ in items])).to(device).permute(0, 3, 1, 2).to(dtype=tensor_dtype).div_(255.0)
        logits = _logits(x, mode)
        finite = bool(torch.isfinite(logits).all())
        full = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False).argmax(1)
        uniq = torch.unique(full).tolist() if state["first"] else None
        codes = list(lut_t[full.clamp_(0, len(lut) - 1)].cpu().numpy())      # .cpu() = sinkronisasi -> waktu terukur akurat
        return codes, finite, uniq

    def predict(images):
        t0 = time.perf_counter()
        codes, finite, uniq = _run(images, state["mode"])
        if not finite and state["mode"] != "fp32" and not use_half:
            print(f"[Fase 6] PERINGATAN: logit {state['mode']} tidak finite (NaN/Inf) -> beralih ke fp32 murni untuk sisa proses")
            state["mode"] = "fp32"
            predict.mode = "fp32"
            codes, finite, uniq = _run(images, "fp32")
        if not finite:
            raise RuntimeError("Fase 6: logit model tidak finite (cek versi torch/transformers & bobot model)")
        if state["first"]:
            state["first"] = False
            print(f"[Fase 6] batch pertama: logit finite, {len(uniq)} kelas Cityscapes terdeteksi {uniq}")
            if len(uniq) < 2:
                raise RuntimeError(f"Fase 6: model hanya memprediksi 1 kelas ({uniq}) -> keluaran tidak valid. "
                                   f"Coba --fp32; bila tetap, periksa torch/transformers.")
        state["t_pred"] += time.perf_counter() - t0
        state["n_pred"] += len(images)
        return codes

    def _agreement(codes, ref):
        same = float(np.mean([(a == b).mean() for a, b in zip(codes, ref)]))
        inter = sum(int(((a == SEG_ROAD) & (b == SEG_ROAD)).sum()) for a, b in zip(codes, ref))
        union = sum(int(((a == SEG_ROAD) | (b == SEG_ROAD)).sum()) for a, b in zip(codes, ref))
        return same, (inter / union if union else 1.0)

    def tune(samples, modes=None, tune_batch=True, min_agree=MIN_AGREE, min_iou=MIN_ROAD_IOU, min_gain=MIN_GAIN):
        """Pilih mode & batch tercepat yang hasilnya SAMA dengan referensi. Mengembalikan laporan (dict)."""
        imgs = [prep(s) for s in samples]
        if use_half:
            modes = ["half"]
        else:
            modes = ["fp32"] + [m for m in (modes or (["fp32"] if (fp32 or not cuda) else ["amp", "half"])) if m != "fp32"]
        cand, ref = [], None
        print(f"[Fase 6] pengatur otomatis: {len(imgs)} frame contoh, mode = {'half (fp16)' if use_half else 'fp32'}")
        for m in modes:
            try:
                _run(imgs[:1], m)                                          # pemanasan (autotune cudnn, alokasi)
                t0 = time.perf_counter()
                res = [_run([im], m) for im in imgs]
                dt = (time.perf_counter() - t0) / len(imgs)
            except RuntimeError as e:                                      # mis. op tak didukung di perangkat ini
                if cuda:
                    torch.cuda.empty_cache()
                cand.append({"mode": m, "ok": False, "why": f"error: {str(e)[:80]}"})
                print(f"   {m:5s} DITOLAK ({str(e)[:80]})")
                continue
            codes, fin = [r[0][0] for r in res], all(r[1] for r in res)
            if ref is None:
                if not fin:
                    raise RuntimeError("Fase 6: logit model tidak finite (cek versi torch/transformers & bobot model)")
                ref, same, iou = codes, 1.0, 1.0
            else:
                same, iou = _agreement(codes, ref) if fin else (0.0, 0.0)
            ok = fin and (len(cand) == 0 or (same >= min_agree and iou >= min_iou))
            why = "" if ok else ("logit NaN/Inf" if not fin else f"hasil beda (sama {same:.2%}, IoU jalan {iou:.3f})")
            cand.append({"mode": m, "sec_per_frame": round(dt, 4), "same_pixels": round(same, 5), "road_iou": round(iou, 4), "ok": ok, "why": why})
            print(f"   {m:5s} {dt:6.3f} s/frame | sama {same:7.2%} | IoU jalan {iou:.3f} | {'LAYAK' if ok else 'DITOLAK: ' + why}")

        valid = [c for c in cand if c.get("ok") and "sec_per_frame" in c]
        if not valid:
            best = {"mode": "half" if use_half else "fp32", "sec_per_frame": 0.25}
        else:
            base = next((c for c in valid if c["mode"] == "fp32"), None)
            if base:
                pool = [c for c in valid if c["mode"] != "fp32" and c["sec_per_frame"] <= base["sec_per_frame"] * (1 - min_gain)]
                best = min(pool, key=lambda c: c["sec_per_frame"]) if pool else base
            else:
                best = min(valid, key=lambda c: c["sec_per_frame"])

        state["mode"] = best["mode"]
        predict.mode = best["mode"]
        spf, batch, peak = best["sec_per_frame"], 1, None
        # ukuran batch: DIUKUR
        if tune_batch and cuda:
            vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            for b in (2,) if vram < 8 else (2, 4):
                try:
                    torch.cuda.reset_peak_memory_stats()
                    group = [imgs[i % len(imgs)] for i in range(b)]
                    _run(group, best["mode"])
                    t0 = time.perf_counter()
                    for _ in range(2):
                        _run(group, best["mode"])
                    per = (time.perf_counter() - t0) / (2 * b)
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    print(f"   batch {b}: kehabisan VRAM -> berhenti")
                    break
                full = peak > 0.8 * vram
                print(f"   batch {b}: {per:.3f} s/frame (puncak VRAM {peak:.2f}/{vram:.1f} GB){' -> terlalu penuh' if full else ''}")
                if full or per > spf * (1 - min_gain):
                    break
                spf, batch = per, b
            torch.cuda.empty_cache()
        predict.suggested_batch = batch
        predict.sec_per_frame = spf
        rep = {"mode": best["mode"], "batch": batch, "sec_per_frame": round(spf, 4), "candidates": cand}
        predict.report = rep
        print(f"[Fase 6] DIPAKAI: mode {best['mode']}, batch {batch} (~{spf:.3f} s/frame)")
        return rep

    predict.suggested_batch = 1
    predict.mode = state["mode"]
    predict.prep = prep
    predict.tune = tune
    predict.stats = lambda: (state["t_pred"], state["n_pred"])
    return predict


# ---------------------------------------------------------------------------------------------------------------------
# Inti
# ---------------------------------------------------------------------------------------------------------------------
def _load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _save_codes(args):
    from PIL import Image
    path, codes = args
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(codes).save(tmp, compress_level=1)
    tmp.replace(path)


def iter_prefetch(pool, chunks, loader, depth=3):
    """Hasilkan (chunk, hasil loader) berurutan dengan prefetch TERBATAS: paling banyak `depth` chunk menunggu di RAM.
    (Sebelumnya pool.map menyerahkan SEMUA chunk sekaligus -> seluruh frame menumpuk di RAM, bisa memicu swap.)"""
    from collections import deque
    it, q = iter(chunks), deque()
    for ch in it:
        q.append((ch, pool.submit(loader, ch)))
        if len(q) >= depth:
            break
    while q:
        ch, fut = q.popleft()
        nxt = next(it, None)
        if nxt is not None:
            q.append((nxt, pool.submit(loader, nxt)))
        yield ch, fut.result()


def frame_stats(codes):
    h = codes.shape[0]
    low = codes[int(h * 2 / 3):]
    n = codes.size
    return {"road_frac": float((codes == SEG_ROAD).sum()) / n,
            "nonroad_frac": float((codes == SEG_NONROAD).sum()) / n,
            "bottom_road_frac": float((low == SEG_ROAD).mean()),
            "empty": bool(not codes.any())}


def run_checkpoints(out, S, n_with_rgb, n_no_rgb, strict=False):
    from common import Paths
    chk = Checker("Fase 6", report_path=Paths(out).checkpoints / "fase6_report.json", strict=strict)
    d = out / "fase6_seg_cam"
    n_png = len([p for p in d.glob("??????.png")])
    chk.ok("jumlah PNG == frame ber-RGB", n_png == n_with_rgb, f"{n_png} vs {n_with_rgb}")
    chk.info("frame tanpa RGB (tanpa bukti kamera)", n_no_rgb)
    if len(S):
        empty = float(S["empty"].mean())
        chk.ok(f"frame kosong total <= {MAX_EMPTY_FRAC:.0%}", empty <= MAX_EMPTY_FRAC, f"{empty:.1%}")
        rf, nf = float(S["road_frac"].mean()), float(S["nonroad_frac"].mean())
        chk.ok("keluaran tidak degenerat (jalan < 90% piksel & bukan-jalan > 2%)", rf < MAX_ROAD_MEAN and nf > MIN_NONROAD_MEAN,
               f"jalan {rf:.2f}, bukan-jalan {nf:.2f}")
        med = float(S["bottom_road_frac"].median())
        chk.ok(f"median fraksi jalan sepertiga bawah >= {MIN_BOTTOM_ROAD_FRAC:.0%}", med >= MIN_BOTTOM_ROAD_FRAC,
               f"{med:.1%}", fatal=False)
        chk.info("rata-rata proporsi piksel", f"jalan {S['road_frac'].mean():.2f} | bukan-jalan {S['nonroad_frac'].mean():.2f}")
    chk.finish()


def run(out, force=False, strict=False, workers=None, max_frames=None, predict=None, batch=None, fp32=False, tune=True):
    import pandas as pd
    t_all = time.time()
    out = Path(out)
    d = out / "fase6_seg_cam"
    marker = d / "fase6_done.json"
    if marker.exists() and not force:
        print(f"[Fase 6] sudah ada: {marker.name} -> dilewati (pakai --force untuk ulang)")
        return
    f4 = out / "fase4_segmentasi"
    for need in (f4 / "fase4_done.json", out / "sync_table.csv"):
        if not need.exists():
            raise SystemExit(f"Tidak ketemu: {need} (jalankan fase sebelumnya dulu)")
    frames = sorted(int(p.stem) for p in (f4 / "points").glob("*.npy"))
    if max_frames:
        frames = frames[:max_frames]
    sync = pd.read_csv(out / "sync_table.csv")
    col = "rgb_data" if "rgb_data" in sync.columns else "rgb_path"
    rgb_of = {}
    for i, p in zip(sync["idx"].astype(int), sync[col]):
        q = resolve_by_name(out, p, "rgb")
        if q is not None and q.exists():
            rgb_of[i] = q
    todo_all = [f for f in frames if f in rgb_of]
    n_no_rgb = len(frames) - len(todo_all)
    if force and d.exists():
        for p in d.glob("*.png"):
            p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    todo = [f for f in todo_all if not (d / f"{f:06d}.png").exists()]
    print(f"[Fase 6] {len(frames)} frame | ber-RGB {len(todo_all)} | tanpa RGB {n_no_rgb} | akan diproses {len(todo)}")

    stats, tuning, gpu_s = {}, None, 0.0
    if todo:
        predict = predict or make_predictor(fp32=fp32)
        if tune and hasattr(predict, "tune"):
            k = min(TUNE_FRAMES, len(todo))
            pick = [todo[i * (len(todo) - 1) // max(k - 1, 1)] for i in range(k)]      # contoh tersebar merata
            tuning = predict.tune([_load_rgb(rgb_of[f]) for f in pick], tune_batch=batch is None)
            print(f"[Fase 6] perkiraan: {len(todo) * tuning['sec_per_frame'] / 60:.0f} menit untuk {len(todo)} frame")
        batch = batch or getattr(predict, "suggested_batch", BATCH)
        prep = getattr(predict, "prep", None) or (lambda im: im)
        n_io = workers or 4
        with ThreadPoolExecutor(max_workers=n_io) as io, Timer("Fase 6 segmentasi kamera"):
            chunks = [todo[s:s + batch] for s in range(0, len(todo), batch)]
            loader = lambda ch: [prep(_load_rgb(rgb_of[f])) for f in ch]               # baca + resize di thread (lepas GIL)
            writes = []
            t_loop = time.time()
            for ch, imgs in tqdm(iter_prefetch(io, chunks, loader, depth=max(3, n_io)), total=len(chunks), desc="Fase 6"):
                for f, codes in zip(ch, predict(imgs)):
                    stats[f] = frame_stats(codes)
                    writes.append(io.submit(_save_codes, (d / f"{f:06d}.png", codes)))
            for w in writes:
                w.result()
            t_loop = time.time() - t_loop
        if hasattr(predict, "stats"):
            gpu_s, n_pred = predict.stats()
            print(f"[Fase 6] waktu prediksi {gpu_s:.0f} s dari {t_loop:.0f} s ({gpu_s / max(t_loop, 1e-9):.0%}); "
                  f"{n_pred and gpu_s / n_pred:.3f} s/frame. Bila persentase rendah, bottleneck = baca/tulis gambar (naikkan --workers).")
    # statistik untuk frame hasil resume (sudah ada di disk) dihitung ulang murah (PNG kecil)
    from PIL import Image
    for f in todo_all:
        if f not in stats:
            stats[f] = frame_stats(np.asarray(Image.open(d / f"{f:06d}.png")))
    S = pd.DataFrame([{"idx": f, **stats[f]} for f in sorted(stats)])
    S.to_csv(d / "summary_frames.csv", index=False)
    run_checkpoints(out, S, len(todo_all), n_no_rgb, strict=strict)
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump({"frames": len(todo_all), "no_rgb": n_no_rgb, "seconds": round(time.time() - t_all, 1),
                   "mode": (tuning or {}).get("mode"), "batch": batch if todo else None, "tuning": tuning}, fh, indent=1)
    print(f"[Fase 6] SELESAI dalam {time.time() - t_all:.0f} s -> {d}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--workers", type=int, help="thread I/O (default 4)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--batch", type=int, default=None, help="default otomatis menurut VRAM")
    ap.add_argument("--fp32", action="store_true", help="paksa fp32 murni (pengatur hanya memilih batch)")
    ap.add_argument("--no-tune", action="store_true", help="lewati pengatur otomatis (fp32 murni, batch menurut VRAM)")
    a = ap.parse_args()
    run(a.out, a.force, a.strict, a.workers, a.max_frames, batch=a.batch, fp32=a.fp32, tune=not a.no_tune)
    sys.exit(0)
