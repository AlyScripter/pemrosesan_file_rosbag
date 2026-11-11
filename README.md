# Modular BEV Perception & Road Reconstruction Pipeline

## Deskripsi Singkat: Kode Ini Untuk Apa?
Proyek ini adalah **Pipeline Persepsi dan Rekonstruksi Jalan *Bird's Eye View* (BEV) secara modular** yang dirancang untuk memproses data rekaman sensor kendaraan otonom dari file **ROSbag** (`.bag`). 

Secara garis besar, sistem ini melakukan:
1. Ekstraksi data mentah sinkron (LiDAR, Kamera RGB, Depth, IMU, GPS/Odom).
2. Kalibrasi ekstrinsik sensor LiDAR dan Kamera.
3. Deteksi objek 3D (Kendaraan, Pejalan Kaki, dll.) menggunakan YOLO dan fusi titik LiDAR.
4. Segmentasi semantik jalan dan pemrosesan mask kamera (SegFormer).
5. Fusi data mask jalan LiDAR dan kamera secara geometris dan spasial.
6. Pembuatan poligon jalan berbasis bukti (*Evidence-based Road Polygon*) dengan perbaikan tepi dan lubang (*hole filling*).
7. Ekspor data pemandangan akhir ke format standar (*contract v4*) untuk divisualisasikan atau di-*streaming* ke *dashboard* (seperti aplikasi Flutter via Rosbridge).

---

## Prasyarat & Lingkungan
* **Python:** 3.8+ (Direkomendasikan menggunakan virtual environment)
* **Pustaka Utama:** `torch`, `transformers`, `opencv-python`, `scipy`, `pandas`, `numpy`, `roslibpy` (opsional untuk streaming).
* **Perangkat Keras:** Diuji dan dioptimalkan untuk GPU kelas konsumer (seperti NVIDIA GTX 1650 4GB dengan fitur *auto-tuner* memori dinamis) hingga perangkat komputasi tepi (*edge computing*) seperti NVIDIA Jetson.

---

## Struktur Alur Pipeline (Fase 1 s.d. 10)
Pipeline ini dirancang secara sekuensial dari Fase 1 hingga Fase 10:
* **Fase 1 & 1b:** Ekstraksi ROSbag, sinkronisasi waktu (*sync_table*), dan penghitungan *waypoints*.
* **Fase 2:** Validasi dan kalibrasi ekstrinsik LiDAR terhadap Kamera/Depth ZED.
* **Fase 3:** Deteksi objek 3D & fusi LiDAR-kamera.
* **Fase 4 & 5:** Segmentasi semantik titik LiDAR dan pembangunan payload BEV.
* **Fase 6:** Segmentasi jalan berbasis kamera (menggunakan SegFormer dengan *auto-tuner* memori VRAM).
* **Fase 7:** Fusi mask jalan antara LiDAR dan Kamera.
* **Fase 8:** Konstruksi poligon jalan berbasis bukti (penanganan lubang & penghalusan tepi).
* **Fase 9:** Ekspor data pemandangan akhir (*road_scene v4*).
* **Fase 10:** *Streaming* data real-time ke *dashboard* eksternal via Rosbridge.

---

## Cara Menjalankan Pipeline

### 1. Menjalankan Seluruh Pipeline dari Awal (Full Run)
Jika Anda ingin memproses file ROSbag dari awal secara berurutan hingga selesai:

```powershell
python run_pipeline.py --bag "PATH/KE/FILE/Rosbag_Rute_03.bag" --out "." --force
```
* Penjelasan Flag:
  * `--bag`: Menentukan lokasi absolut atau relatif file ROSbag sumber.
  * `--out`: Direktori tempat penyimpanan seluruh hasil keluaran (*output cache*).
  * `--force`: Memaksa eksekusi ulang dari Fase 1 (mengabaikan cache lama).

### 2. Menjalankan Satu Fase Tertentu Saja (--only)
Jika Anda sudah memiliki hasil dari fase sebelumnya dan hanya ingin mengeksekusi fase tertentu (misalnya **Fase 6** saja):

```powershell
python run_pipeline.py --out "." --only 6
```
*(Ganti angka `6` dengan nomor fase lain yang ingin Anda uji, misal `3`, `7`, `8`, dll).*

### 3. Melanjutkan Pipeline dari Fase Tertentu (--from)
Jika Anda ingin menjalankan proses mulai dari fase tertentu hingga selesai:

```powershell
python run_pipeline.py --out "." --from 7
```

---

## Penanganan File ROSbag di Folder Berbeda
Anda tidak harus meletakkan file ROSbag di dalam folder proyek. Anda dapat mereferensikannya langsung dari direktori eksternal manapun dengan menuliskan jalur lengkapnya pada argumen `--bag`:

```powershell
python run_pipeline.py --bag "D:\Data_Autonom\Dataset_Baru\Rosbag_Rute_01.bag" --out "./output_rute01" --force
```
* Program akan otomatis membaca ROSbag dari folder luar tersebut dan menyimpan seluruh hasil *output* ke dalam folder tujuan yang ditentukan pada `--out`.

---

## Pembersihan Cache (*Cleanup*)
Jika Anda ingin mereset hasil pemrosesan suatu fase atau memulai dari bersih tanpa argumen `--force` total, Anda dapat menghapus folder *cache* terkait di direktori kerja, seperti:
* `fase6_seg_cam/` (untuk segmentasi kamera)
* `fase7_road_mask/` (untuk fusi mask jalan)
* `fase8_road_poly/` (untuk poligon jalan)
* `fase9_road_scene/` (untuk ekspor pemandangan akhir)