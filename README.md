# Threck — Thread Check (Inspeksi Ulir Lubang Ø3,5 mm)

Aplikasi Python untuk mendeteksi apakah sebuah lubang **berulir (OK)** atau **tanpa ulir (NG)**
menggunakan kamera. Seluruh pemrosesan berjalan lokal (PC atau Raspberry Pi), dan kamera hanya
berfungsi sebagai "mata".

## Fitur

- **Dua sudut pandang kamera**:
  - **Sudut miring** (default): kamera melihat ke dalam lubang dari samping, ulir tampak sebagai
    garis-garis lengkung sejajar di dinding lubang.
  - **Tegak lurus (top view)**: kamera tepat di atas lubang, lubang tampak sebagai lingkaran.
- **Live display** dengan overlay: area inspeksi (ROI) berwarna sesuai hasil, arah garis ulir
  (mode miring), atau lingkaran lubang (mode tegak lurus).
- **Judgement OK / NG** berukuran besar, skor keyakinan, dan ambang batas yang bisa diatur.
- **Dua mode inspeksi**:
  - *Trigger*: tombol INSPEKSI / tombol **Spasi**. Hasil dirata-rata dari beberapa frame, dihitung ke statistik, dicatat ke log CSV, dan gambar NG disimpan.
  - *Kontinu*: judgement live terus-menerus (untuk setting dan monitoring).
- **Training langsung di aplikasi**:
  - Ambil sampel ULIR / TANPA ULIR dari kamera (tombol **1** / **2**).
  - Kelola dataset: galeri, hapus, pindah kelas (salah label), impor gambar.
  - Augmentasi otomatis (rotasi kecil/bebas sesuai mode, cahaya, blur, noise).
  - Mode *Otomatis* membandingkan 4 algoritma (SVM, Random Forest, Gradient Boosting, Logistic
    Regression) dengan validasi silang, lalu memilih yang terbaik berdasarkan **akurasi dan
    keyakinan** (persentase sampel yang diprediksi benar dengan keyakinan ≥ 80%).
  - Laporan akurasi dan confusion matrix, termasuk jumlah part *tanpa ulir yang lolos sebagai OK*.
- **Metode Rule-based** sebagai cadangan, sehingga aplikasi sudah bisa dipakai sebelum ada model.
- **Pilih kamera**: webcam/USB (OpenCV, dengan pemilihan perangkat, backend, dan resolusi),
  Raspberry Pi Camera (Picamera2), atau file video/gambar untuk simulasi.
- Statistik (Total / OK / NG / Yield), riwayat inspeksi, dan pengaturan yang tersimpan otomatis.

## Cara kerja deteksi

### Mode sudut miring (default)
1. ROI (area dinding lubang) dinormalisasi ukurannya, lalu **diluruskan**: sudut garis dicari
   otomatis (±20° di sekitar arah yang diatur) agar garis ulir menjadi mendatar.
2. ROI dibagi menjadi **12 strip vertikal**. Profil terang-gelap tiap strip dianalisis dengan
   FFT dan autokorelasi untuk mengukur **seberapa teratur garis berulang**.
3. Nilai **median antar strip** dipakai, sehingga pantulan cahaya (garis terang vertikal) tidak
   merusak hasil.
4. **Deteksi garis (untuk ulir yang hanya terlihat 2–3 garis, mis. pelat tipis/burring)**: filter
   *top-hat* per strip hanya meloloskan garis terang sempit (puncak ulir), sehingga gradasi cahaya
   dan batas gelap keluar lubang tidak terhitung. Ulir = **≥ 2 garis** yang cukup kontras dan
   **menerus melintasi strip**. Bibir lubang hanya 1 garis, jadi tidak terhitung sebagai ulir.
5. **Kontras** diukur hanya pada frekuensi garis yang terdeteksi. Ulir menghasilkan garis
   terang-gelap kuat, sedangkan bekas bor pada lubang polos jauh lebih rapat dan samar.
6. Total 180 fitur (spektrum, periodisitas, jumlah/tinggi/kontinuitas garis, bentuk profil,
   gradien, tekstur LBP) masuk ke model ML. Metode Rule = skor terbaik dari (pola berulang ×
   gerbang kontras) dan (deteksi ≥ 2 garis).

### Mode tegak lurus
1. **Cari lubang** di dalam ROI (Hough Circle, dengan fallback kontur), lalu radiusnya
   disempurnakan dari profil tepi radial.
2. **Polar unwrap**: area cincin di sekitar lubang "dibuka" ke koordinat polar, sehingga
   cincin/spiral ulir menjadi garis-garis lurus berulang.
3. 189 fitur periodisitas dan tekstur, dihitung untuk seluruh band dan khusus dinding dalam lubang.

### Keputusan
- *ML* (disarankan untuk produksi): probabilitas "berulir" dari model hasil training ≥ ambang → OK.
  - **Penolak objek tidak dikenal**: model juga mengingat seperti apa sampel training-nya.
    Gambar yang jauh berbeda (wajah, tangan, tidak ada part, part lain) → **NG: objek tidak
    dikenal**. Kelonggarannya diatur di *Pengaturan → Toleransi objek asing* (0 = nonaktif).
- *Rule* (tanpa training): skor pola ulir ≥ ambang → OK.
  - Garis sejajar bisa berasal dari benda apa saja (rak, lampu, wajah). Karena itu, di mode
    sudut miring Rule **wajib dikalibrasi** dulu: letakkan part berulir lalu klik
    *Pengaturan → Kalibrasi dari part OK*. Jarak garis ulirnya disimpan sebagai acuan, dan garis
    dengan jarak lain ditolak. **Tanpa kalibrasi, Rule selalu menampilkan N/A, tidak pernah OK.**
  - Rule hanya untuk memulai dengan cepat. Untuk produksi tetap gunakan ML dengan sampel asli.

Model hanya berlaku untuk sudut pandang saat ia dilatih. Bila sudut pandang diganti, latih ulang.

## Instalasi

### Windows / PC
```bash
pip install -r requirements.txt
python main.py
```
Atau klik dua kali `jalankan_windows.bat`.

### Raspberry Pi (Raspberry Pi OS Bookworm 64-bit, disarankan Pi 5)
```bash
sudo apt update
sudo apt install -y python3-pyqt5 python3-opencv python3-sklearn python3-joblib python3-picamera2
python3 main.py
```
Untuk menjalankan model **YOLO26n-cls** di Pi, tambahkan runtime ONNX:
```bash
python3 -m venv --system-site-packages ~/threck-venv
~/threck-venv/bin/pip install onnxruntime
~/threck-venv/bin/python main.py
```

Aplikasi otomatis memakai PyQt5 bila PySide6 tidak ada. Mode layar penuh untuk HMI:
`./jalankan_pi.sh --fullscreen` (F11 untuk keluar/masuk layar penuh).

## Model YOLO26n-cls (deep learning)

Pilih **Algoritma → YOLO26n-cls** di halaman Training. Pada data foto part asli (32 OK / 21 NG),
YOLO26n-cls mencapai akurasi validasi silang **100%** dengan keyakinan rata-rata **99,9%**,
sedangkan model klasik terbaik (SVM) mencapai 96,1%. YOLO lebih tahan terhadap variasi arah garis
ulir, warna, dan pencahayaan.

- **Training** butuh `pip install ultralytics` (PyTorch), ±5 menit di laptop. Training di Raspberry
  Pi bisa, tapi jauh lebih lama, jadi disarankan **latih di PC lalu salin folder `models/`** ke Pi.
- **Inferensi** memakai file ONNX (`models/yolo26n_cls_ulir.onnx`) lewat `onnxruntime`, tanpa
  PyTorch, sehingga ringan untuk Raspberry Pi 5.
- Penolak **objek tidak dikenal** tetap aktif.
- **Lisensi**: Ultralytics YOLO berlisensi **AGPL-3.0**. Untuk pemakaian internal pabrik umumnya
  tidak masalah. Bila aplikasi didistribusikan atau dijual ke pihak lain, cek kebutuhan lisensi
  Enterprise dari Ultralytics.

## Profil kamera

Model hanya akurat untuk kamera tempat ia dilatih (ketajaman, warna, cahaya, dan sudut pandang
tiap kamera berbeda). Karena itu, setiap **profil kamera** punya kamera, ROI, pengaturan
analisis, dataset, dan model sendiri.

- **Ganti profil**: pilih di kotak **PROFIL** pada header. Kamera, ROI, dan model ikut berganti.
- **Profil baru**: header → *＋ Profil baru…* atau *Pengaturan → Profil kamera → Profil baru…*.
  Pengaturan profil aktif boleh disalin, tetapi dataset & model tidak (wajib training sendiri).
- **Pindah sampel** yang salah tempat: *Training → Pindah ke profil kamera lain…*.
- Ganti nama / hapus profil di *Pengaturan → Profil kamera*. Menghapus profil menghapus dataset &
  modelnya, sedangkan gambar hasil inspeksi dan log tetap disimpan.

## Cara pakai (alur yang disarankan)

1. **Pengaturan → Kamera**: pilih sumber dan perangkat (klik *Cari kamera*), pilih resolusi, lalu
   *Terapkan & Hubungkan*. Untuk webcam di Windows, *Pengaturan driver…* membuka dialog
   exposure/fokus.
2. **Pengaturan → Analisis ulir**: pilih **Sudut pandang** sesuai posisi kamera.
3. **Atur ROI**: seret kotak di video.
   - *Sudut miring*: tepat di dinding lubang tempat garis ulir terlihat. Hindari bibir lubang,
     permukaan luar, dan dasar lubang yang gelap. Usahakan terlihat minimal ±5 garis ulir.
   - *Tegak lurus*: mengelilingi lubang (lubang mengisi ±30–90% ROI).
4. *(Opsional, bila belum ada model)* **Kalibrasi Rule**: letakkan part berulir, klik
   *Pengaturan → Kalibrasi dari part OK*.
5. **Training**:
   - Ambil minimal 5 (disarankan **≥ 30**) sampel per kelas, dengan sedikit variasi posisi dan cahaya.
   - Klik **Latih Model** dan baca laporannya. Perhatikan baris *"Tanpa ulir lolos sebagai OK"*: idealnya 0.
6. **Inspeksi**: mode *Trigger*, tekan Spasi/INSPEKSI untuk setiap part.
7. Bila part tanpa ulir masih ada yang lolos, naikkan **Ambang OK (ML)** di Pengaturan
   (misalnya 0,7), tambah sampel, lalu latih ulang.

## Mencoba tanpa kamera (data demo)

```bash
python tools/buat_data_demo.py
```
Perintah ini membuat data sintetis untuk kedua sudut pandang:
- `demo_data/miring/` (sudut miring, hanya 2–3 garis ulir seperti part asli),
  `demo_data/miring_banyak/` (sudut miring, banyak garis), dan `demo_data/tegak/` (top view).
  Masing-masing berisi `ulir/`, `tanpa_ulir/`, dan `demo_video.mp4`.
- `demo_data/contoh_miring_ulir.png` adalah contoh foto ulir dari sudut miring.

Cara mencoba:
- Pengaturan → Sumber: *File video / gambar*, pilih `demo_data/miring/demo_video.mp4`.
- Training → *Impor gambar…* dari `demo_data/miring/ulir` dan `demo_data/miring/tanpa_ulir`.

> Data demo hanya untuk mencoba alur aplikasi. Model untuk produksi **harus** dilatih dari
> gambar part asli dengan kamera, lensa, dan pencahayaan yang sebenarnya.

## Struktur folder

```
CekUlir/
├── main.py                  # titik masuk aplikasi
├── app/
│   ├── vision.py            # analisis sudut miring & tegak lurus, fitur, keputusan
│   ├── model.py             # training, augmentasi, validasi silang, simpan/muat model
│   ├── camera.py            # sumber kamera: OpenCV, Picamera2, file
│   ├── workers.py           # thread kamera & thread training
│   ├── storage.py           # dataset, simpan hasil, log CSV
│   ├── config.py            # konfigurasi (disimpan ke config.json)
│   ├── qt.py                # kompatibilitas PySide6 / PyQt5
│   └── ui/                  # halaman Inspeksi, Training, Pengaturan + tema
├── config.json                        # profil aktif & preferensi tampilan
├── profiles/<nama profil>/
│   ├── config.json                    # kamera, ROI, analisis, ambang profil ini
│   ├── dataset/ulir, dataset/tanpa_ulir
│   └── models/                        # model_ulir.joblib (+ yolo26n_cls_ulir.onnx)
├── weights/yolo26n-cls.pt             # bobot dasar YOLO (bersama)
├── captures/<profil>/OK|NG/<tanggal>/ # gambar hasil inspeksi
├── logs/<profil>/inspeksi_YYYYMMDD.csv
└── tools/buat_data_demo.py
```

## Tips agar akurat

- **Pencahayaan paling menentukan.** Untuk sudut miring, lampu dari arah kamera (atau sedikit
  di atasnya) membuat puncak ulir tampak sebagai garis terang. Jaga agar cahaya konsisten
  (hindari cahaya ruangan/matahari).
- Sudut kamera dan jarak harus **selalu sama** (pakai bracket tetap), fokus tajam pada dinding
  lubang. Untuk lubang Ø3,5 mm perlu lensa macro dan kemiringan yang cukup agar dinding terlihat.
- Posisi part dibuat konsisten dengan jig, agar ROI tidak perlu diubah-ubah.
- Bila kamera, lensa, jarak, atau lampu diganti, **latih ulang** model.

## Pengembangan berikutnya (opsional)

- Integrasi I/O: input trigger dari sensor dan output OK/NG ke PLC atau reject via GPIO Raspberry Pi
  (wajib memakai optocoupler/relay untuk level 24 V).
- Kontrol strobe lampu yang disinkronkan dengan pengambilan gambar.
- Multi-ROI untuk part dengan beberapa lubang.
