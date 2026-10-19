# Threck — Thread Check (Inspeksi Ulir Lubang)

Aplikasi Python untuk mendeteksi apakah sebuah lubang **berulir (OK)** atau **tanpa ulir (NG)**
menggunakan kamera dan model deep learning **YOLO26n-cls**. Seluruh pemrosesan berjalan lokal (PC
atau Raspberry Pi 5), dan kamera hanya berfungsi sebagai "mata".

## Fitur

- **Live display** dengan kotak ROI berwarna sesuai hasil (hijau OK, merah NG, kuning N/A).
- **Judgement OK / NG** besar, keyakinan model, dan ambang OK yang bisa diatur.
- **Dua mode inspeksi**:
  - *Trigger*: tombol INSPEKSI / tombol **Spasi**. Hasil dirata-rata dari beberapa frame, dihitung
    ke statistik, dicatat ke log CSV, dan gambar NG disimpan.
  - *Kontinu*: judgement live terus-menerus (untuk setting dan monitoring).
- **Training langsung di aplikasi**: ambil sampel ULIR / TANPA ULIR (tombol **1** / **2**), kelola
  dataset (galeri, hapus, pindah kelas, impor gambar), lalu latih YOLO26n-cls.
- **Penolak objek tidak dikenal**: gambar yang jauh berbeda dari sampel training (wajah, tangan,
  tidak ada part, part lain) dinyatakan **NG: Unidentified Object**, bukan OK.
- **Profil kamera**: setiap kamera punya kamera, ROI, pengaturan, dataset, dan model sendiri.
- **Pilih kamera**: webcam/USB (OpenCV), Raspberry Pi Camera (Picamera2), atau file video/gambar.
- Statistik (Total / OK / NG / Yield), riwayat inspeksi, sidebar yang bisa diciutkan (Ctrl+B).
- **Mode tampilan**: normal, layar penuh, atau ringkas untuk layar Raspberry Pi 800×480.

## Cara kerja

1. Gambar di dalam **ROI** dipotong dari frame kamera.
2. **YOLO26n-cls** (model klasifikasi ±2,8 juta parameter, format ONNX) menilai potongan itu dan
   memberi **keyakinan berulir** 0–100%. Keyakinan ≥ ambang OK → **OK**.
3. **Penolak objek tidak dikenal** membandingkan tampilan gambar (tata letak, tekstur, arah tepi,
   warna) dengan sampel training. Bila terlalu berbeda → **NG: Unidentified Object**.
4. Belum ada model → **N/A** (aplikasi tidak pernah menyatakan OK tanpa model).

Model hanya akurat untuk kamera dan setup tempat ia dilatih. Bila kamera, lensa, jarak, atau lampu
berubah, buat profil baru atau latih ulang.

## Instalasi

### Windows / PC
```bash
pip install -r requirements.txt
pip install ultralytics
python main.py
```
`ultralytics` hanya dibutuhkan untuk **melatih** model. Atau klik dua kali `jalankan_windows.bat`.

### Raspberry Pi (Raspberry Pi OS Bookworm 64-bit, disarankan Pi 5)
```bash
sudo apt update
sudo apt install -y python3-pyqt5 python3-opencv python3-sklearn python3-joblib python3-picamera2
python3 -m venv --system-site-packages ~/threck-venv
~/threck-venv/bin/pip install onnxruntime
~/threck-venv/bin/python main.py
```
Aplikasi otomatis memakai PyQt5 bila PySide6 tidak ada.

**Layar Raspberry Pi 800×480**: pilih *Pengaturan → Tampilan → Raspberry Pi 800×480*, atau jalankan
`./jalankan_pi.sh --800x480`. Tata letaknya diringkas (header tipis, sidebar ikon, panel lebih
sempit, Pengaturan satu kolom) dan otomatis layar penuh. Untuk layar besar pakai `--fullscreen`.
F11 = keluar/masuk layar penuh.

**Training** disarankan di PC/laptop (±3–5 menit), lalu salin folder `profiles/` ke Raspberry Pi.
Inferensi di Pi memakai file ONNX lewat `onnxruntime`, tanpa PyTorch.

## Cara pakai

1. **Profil**: pilih atau buat profil kamera di kotak **PROFIL** pada header.
2. **Pengaturan → Kamera**: pilih sumber dan perangkat (*Cari kamera*), resolusi, lalu
   *Terapkan & Hubungkan*. Di Windows, *Pengaturan driver…* membuka dialog exposure/fokus.
3. **Atur ROI**: seret kotak di video sehingga ulir terlihat jelas di dalamnya. Usahakan posisi
   part selalu sama (pakai jig) agar ROI tidak perlu diubah-ubah.
4. **Training**:
   - Ambil minimal 5 (disarankan **≥ 50**) sampel per kelas dari banyak part berbeda.
   - *(Opsional)* centang **Early stopping**: training berhenti otomatis bila akurasi validasi tidak
     membaik selama *patience* epoch ("Lama training" menjadi batas maksimum). Model akhir lalu
     dilatih sebanyak epoch yang dijalankan tahap validasi, sehingga training lebih cepat.
   - Klik **Latih Model** dan baca laporannya. Baris *"Tanpa ulir lolos sebagai OK"* harus 0.
5. **Inspeksi**: mode *Trigger*, tekan Spasi/INSPEKSI untuk setiap part.
6. Bila part tanpa ulir masih ada yang lolos, naikkan **Ambang OK** di Pengaturan (mis. 0,7),
   tambah sampel, lalu latih ulang. Part yang salah dinilai sebaiknya difoto dan disimpan ke
   kelas yang benar.

## Profil kamera

- **Ganti profil**: kotak **PROFIL** di header. Kamera, ROI, dan model ikut berganti.
- **Profil baru**: header → *＋ Profil baru…* atau *Pengaturan → Profil kamera*. Pengaturan profil
  aktif boleh disalin, tetapi dataset & model tidak (wajib training sendiri).
- **Pindah sampel** yang salah tempat: *Training → Pindah ke profil kamera lain…*.
- Ganti nama / hapus profil di *Pengaturan → Profil kamera*. Menghapus profil menghapus dataset &
  modelnya, sedangkan gambar hasil inspeksi dan log tetap disimpan.

## Struktur folder

```
CekUlir/
├── main.py                       # titik masuk aplikasi
├── app/
│   ├── vision.py                 # potong ROI, keputusan OK/NG, fitur penolak objek asing, overlay
│   ├── model.py                  # model aktif, penolak objek asing, laporan training
│   ├── yolo_backend.py           # training YOLO26n-cls (Ultralytics) & inferensi ONNX
│   ├── camera.py                 # sumber kamera: OpenCV, Picamera2, file
│   ├── workers.py                # thread kamera & thread training
│   ├── storage.py                # dataset, simpan hasil, log CSV
│   ├── config.py                 # pengaturan & profil kamera
│   ├── context.py                # state aplikasi bersama
│   ├── qt.py                     # kompatibilitas PySide6 / PyQt5
│   └── ui/                       # halaman Inspeksi, Training, Pengaturan, tema, logo
├── config.json                   # profil aktif & preferensi tampilan
├── profiles/<nama profil>/
│   ├── config.json               # kamera, ROI, ambang profil ini
│   ├── dataset/ulir, dataset/tanpa_ulir
│   └── models/                   # model_ulir.joblib, yolo26n_cls_ulir.onnx (+ .pt)
├── weights/yolo26n-cls.pt        # bobot dasar YOLO (dipakai semua profil)
├── captures/<profil>/OK|NG/<tanggal>/
├── logs/<profil>/inspeksi_YYYYMMDD.csv
├── logo/                         # logo Threck (PNG, ICO)
└── tools/export_logo.py
```

## Tips agar akurat

- **Ketajaman & cahaya lebih penting daripada resolusi.** Pastikan garis ulir tajam (fokus tepat)
  dan cahaya konsisten (kunci exposure, hindari cahaya ruangan/matahari).
- Sudut kamera dan jarak harus **selalu sama** (pakai bracket tetap).
- Latar di balik lubang tembus sebaiknya seragam (mis. pelat gelap di bawah part).

## Lisensi YOLO

Ultralytics YOLO berlisensi **AGPL-3.0**. Untuk pemakaian internal pabrik umumnya tidak masalah.
Bila aplikasi didistribusikan atau dijual ke pihak lain, cek kebutuhan lisensi Enterprise dari
Ultralytics.

## Pengembangan berikutnya (opsional)

- Integrasi I/O: input trigger dari sensor dan output OK/NG ke PLC atau reject via GPIO Raspberry Pi
  (wajib memakai optocoupler/relay untuk level 24 V).
- *Teach & locate*: ROI otomatis mengikuti posisi lubang bila part sedikit bergeser.
- Multi-ROI untuk part dengan beberapa lubang.
