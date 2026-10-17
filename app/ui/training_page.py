"""Halaman Training: ambil sampel, kelola dataset, latih model."""
from pathlib import Path

from .. import storage
from .. import config
from ..config import CLASS_DISPLAY, CLASS_NG, CLASS_OK, AppConfig
from ..model import MIN_PER_CLASS, MODEL_KINDS, format_report
from ..qt import QtCore, QtGui, QtWidgets, Qt
from ..vision import VIEW_MODES, DetectParams
from ..workers import TrainWorker
from .widgets import (Card, StatTile, VideoView, button, label, scroll_panel, set_prop,
                      video_frame)

GALLERY_MAX = 400
THUMB = 84


def open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


class TrainingPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._last = None
        self._train = None

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # ---- kiri: video ----
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        self.video = VideoView()
        self.video.roi_selected.connect(lambda r: self.ctx.update(roi=r))
        left.addWidget(video_frame(self.video), 1)
        bar = QtWidgets.QHBoxLayout()
        b = button("Reset ROI")
        b.clicked.connect(lambda: self.ctx.update(roi=AppConfig().roi))
        bar.addWidget(b)
        bar.addStretch(1)
        self.lbl_detect = label("", "status")
        bar.addWidget(self.lbl_detect)
        left.addLayout(bar)
        root.addLayout(left, 1)

        # ---- kanan ----
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 10, 0)
        pl.setSpacing(12)

        # Ambil sampel
        cap = Card("1 · Ambil sampel")
        cap.body.addWidget(label("Letakkan part, pastikan lingkaran lubang terdeteksi di video, "
                                 "lalu simpan sesuai kondisi sebenarnya. Variasikan posisi & cahaya "
                                 "sedikit agar model lebih tangguh.", "hint", wrap=True))
        row = QtWidgets.QHBoxLayout()
        self.btn_ok = button("＋ ULIR\nOK  ·  tombol 1", "ok", "lg")
        self.btn_ng = button("＋ TANPA ULIR\nNG  ·  tombol 2", "ng", "lg")
        self.btn_ok.clicked.connect(lambda: self.capture(CLASS_OK))
        self.btn_ng.clicked.connect(lambda: self.capture(CLASS_NG))
        row.addWidget(self.btn_ok)
        row.addWidget(self.btn_ng)
        cap.body.addLayout(row)
        cnt = QtWidgets.QHBoxLayout()
        self.t_ok = StatTile("Sampel ulir", "ok")
        self.t_ng = StatTile("Sampel tanpa ulir", "ng")
        cnt.addWidget(self.t_ok)
        cnt.addWidget(self.t_ng)
        cap.body.addLayout(cnt)
        self.lbl_capture = label("Disarankan ≥ 30 sampel per kelas.", "status", wrap=True)
        cap.body.addWidget(self.lbl_capture)
        pl.addWidget(cap)

        # Dataset
        ds = Card("2 · Dataset")
        top = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QComboBox()
        self.filter.addItem("Semua kelas", None)
        self.filter.addItem(CLASS_DISPLAY[CLASS_OK], CLASS_OK)
        self.filter.addItem(CLASS_DISPLAY[CLASS_NG], CLASS_NG)
        self.filter.currentIndexChanged.connect(self.refresh_gallery)
        top.addWidget(self.filter, 1)
        ds.body.addLayout(top)
        self.gallery = QtWidgets.QListWidget()
        self.gallery.setViewMode(QtWidgets.QListView.IconMode)
        self.gallery.setIconSize(QtCore.QSize(THUMB, THUMB))
        self.gallery.setGridSize(QtCore.QSize(THUMB + 14, THUMB + 30))
        self.gallery.setResizeMode(QtWidgets.QListView.Adjust)
        self.gallery.setMovement(QtWidgets.QListView.Static)
        self.gallery.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.gallery.setMinimumHeight(230)
        self.gallery.setWordWrap(True)
        ds.body.addWidget(self.gallery)
        btns = QtWidgets.QGridLayout()
        btns.setSpacing(8)
        b_del = button("Hapus", tooltip="Hapus sampel terpilih (Delete)")
        b_move = button("Pindah kelas", tooltip="Pindahkan sampel terpilih ke kelas lain (salah label)")
        b_imp = button("Impor gambar…", tooltip="Tambahkan gambar dari folder lain ke dataset")
        b_open = button("Buka folder")
        b_del.clicked.connect(self.delete_selected)
        b_move.clicked.connect(self.move_selected)
        b_imp.clicked.connect(self.import_images)
        b_open.clicked.connect(lambda: open_folder(config.dataset_dir()))
        b_prof = button("Pindah ke profil kamera lain…",
                        tooltip="Pindahkan sampel terpilih ke dataset profil kamera lain (kelas tetap sama)")
        b_prof.clicked.connect(self.move_to_profile)
        for i, w in enumerate((b_del, b_move, b_imp, b_open)):
            btns.addWidget(w, i // 2, i % 2)
        btns.addWidget(b_prof, 2, 0, 1, 2)
        ds.body.addLayout(btns)
        pl.addWidget(ds)

        # Training
        tr = Card("3 · Latih model")
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.cb_model = QtWidgets.QComboBox()
        for k, v in MODEL_KINDS.items():
            self.cb_model.addItem(v, k)
        self.cb_model.setCurrentIndex(max(0, self.cb_model.findData(ctx.cfg.model_kind)))
        self.cb_model.currentIndexChanged.connect(self._model_kind_changed)
        form.addRow(label("Algoritma", "muted"), self.cb_model)
        self.chk_aug = QtWidgets.QCheckBox("Augmentasi data")
        self.chk_aug.setChecked(ctx.cfg.augment)
        self.chk_aug.setToolTip("Membuat variasi gambar (rotasi, cahaya, blur, noise) agar model lebih tangguh")
        self.sp_aug = QtWidgets.QSpinBox()
        self.sp_aug.setRange(1, 30)
        self.sp_aug.setValue(ctx.cfg.aug_per_image)
        self.sp_aug.setSuffix("  variasi/gambar")
        self.chk_aug.toggled.connect(lambda v: (self.ctx.update(augment=v), self.sp_aug.setEnabled(v)))
        self.sp_aug.valueChanged.connect(lambda v: self.ctx.update(aug_per_image=v))
        self.sp_aug.setEnabled(ctx.cfg.augment)
        form.addRow(label("Data", "muted"), self.chk_aug)
        form.addRow(label("Variasi", "muted"), self.sp_aug)
        self.lbl_algo = label("", "hint", wrap=True)
        form.addRow(self.lbl_algo)
        tr.body.addLayout(form)
        self.btn_train = button("Latih Model", "primary", "lg")
        self.btn_train.clicked.connect(self.start_training)
        tr.body.addWidget(self.btn_train)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        tr.body.addWidget(self.progress)
        self.lbl_train = label("", "hint", wrap=True)
        tr.body.addWidget(self.lbl_train)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(280)
        self.report.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        tr.body.addWidget(self.report)
        b_delm = button("Hapus model aktif")
        b_delm.clicked.connect(self.delete_model)
        tr.body.addWidget(b_delm)
        pl.addWidget(tr)
        pl.addStretch(1)

        root.addWidget(scroll_panel(panel, 440))

        self.gallery.installEventFilter(self)

        self.refresh_counts()
        self.refresh_gallery()
        self.refresh_model_info()
        self._update_algo_hint()

    def _model_kind_changed(self, *_):
        kind = self.cb_model.currentData()
        self.ctx.update(model_kind=kind)
        self._update_algo_hint()

    def _update_algo_hint(self):
        kind = self.cb_model.currentData()
        is_yolo = kind == "yolo"
        self.chk_aug.setEnabled(not is_yolo)
        self.sp_aug.setEnabled(not is_yolo and self.chk_aug.isChecked())
        if is_yolo:
            from ..yolo_backend import training_available
            if training_available():
                self.lbl_algo.setText("YOLO26n-cls: paling akurat untuk foto part asli. Training ±5 menit di "
                                      "laptop (jauh lebih lama di Raspberry Pi, disarankan latih di PC lalu "
                                      "salin folder models/). Augmentasi diatur otomatis oleh YOLO.")
            else:
                self.lbl_algo.setText("⚠ Ultralytics belum terpasang (pip install ultralytics). Model YOLO "
                                      "hasil training di PC tetap bisa dipakai di sini lewat ONNX.")
        else:
            self.lbl_algo.setText("Model klasik: training cepat (±30 detik), ringan untuk Raspberry Pi.")

    # ------------------------------------------------------------------
    def eventFilter(self, obj, ev):
        if obj is self.gallery and ev.type() == QtCore.QEvent.KeyPress and ev.key() == Qt.Key_Delete:
            self.delete_selected()
            return True
        return super().eventFilter(obj, ev)

    def on_result(self, qimg, res):
        self.video.set_image(qimg)
        self._last = res
        if not self.isVisible():
            return
        if res.circle is not None:
            self.lbl_detect.setText(f"✓ Lubang terdeteksi · r = {res.circle[2]:.1f} px")
            set_prop(self.lbl_detect, "tone", "ok")
        elif "period_px" in res.info:
            strong = (res.rule_score or 0) >= self.ctx.cfg.rule_threshold
            self.lbl_detect.setText(
                f"{'Pola garis kuat' if strong else 'Pola garis lemah'} · jarak "
                f"{res.info['period_px']:.1f} px · kontras {res.info.get('contrast', 0):.1f}")
            set_prop(self.lbl_detect, "tone", "ok" if strong else "warn")
        else:
            self.lbl_detect.setText("✕ Lubang tidak terdeteksi" if self.ctx.cfg.view_mode == "top"
                                    else "✕ ROI tidak bisa dianalisis")
            set_prop(self.lbl_detect, "tone", "ng")

    def on_camera_status(self, connected, msg):
        if not connected:
            self.video.set_placeholder(msg)
            self._last = None

    def _status(self, text, tone=""):
        self.lbl_capture.setText(text)
        set_prop(self.lbl_capture, "tone", tone)

    def capture(self, cls):
        res = self._last
        if res is None or res.frame is None:
            self._status("Belum ada gambar dari kamera.", "ng")
            return
        if res.band is None:
            self._status("ROI tidak bisa dianalisis (lubang tidak terdeteksi), sampel TIDAK disimpan. "
                         "Atur ROI, fokus, atau pencahayaan dulu.", "ng")
            return
        x, y, w, h = res.roi_px
        crop = res.frame[y:y + h, x:x + w].copy()
        path = storage.save_sample(cls, crop)
        self._status(f"Tersimpan sebagai {CLASS_DISPLAY[cls]}: {path.name}", "ok")
        self.refresh_counts()
        if self.filter.currentData() in (None, cls):
            self.gallery.insertItem(0, self._make_item(path, cls))

    # ------------------------------------------------------------------
    def refresh_counts(self):
        c = storage.count_samples()
        self.t_ok.set(c[CLASS_OK])
        self.t_ng.set(c[CLASS_NG])
        ready = min(c.values()) >= MIN_PER_CLASS
        self.btn_train.setEnabled(ready and self._train is None)
        if not ready:
            self.lbl_train.setText(f"Butuh minimal {MIN_PER_CLASS} sampel per kelas untuk training "
                                   "(disarankan ≥ 30).")
        return c

    def _make_item(self, path: Path, cls):
        pm = QtGui.QPixmap(str(path))
        if not pm.isNull():
            pm = pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        it = QtWidgets.QListWidgetItem(QtGui.QIcon(pm), "ULIR" if cls == CLASS_OK else "TANPA")
        it.setData(Qt.UserRole, (str(path), cls))
        it.setToolTip(f"{path.name}\n{CLASS_DISPLAY[cls]}")
        it.setForeground(QtGui.QColor("#22c55e" if cls == CLASS_OK else "#ef4444"))
        return it

    def refresh_gallery(self):
        self.gallery.clear()
        want = self.filter.currentData()
        entries = []
        for cls in (CLASS_OK, CLASS_NG):
            if want in (None, cls):
                entries += [(p, cls) for p in storage.list_samples(cls)]
        entries.sort(key=lambda e: e[0].stat().st_mtime, reverse=True)
        for p, cls in entries[:GALLERY_MAX]:
            self.gallery.addItem(self._make_item(p, cls))

    def _selected(self):
        return [it.data(Qt.UserRole) for it in self.gallery.selectedItems()]

    def delete_selected(self):
        sel = self._selected()
        if not sel:
            return
        r = QtWidgets.QMessageBox.question(self, "Hapus sampel", f"Hapus {len(sel)} sampel terpilih?")
        if r != QtWidgets.QMessageBox.Yes:
            return
        for path, _ in sel:
            storage.delete_sample(Path(path))
        self.refresh_counts()
        self.refresh_gallery()

    def move_selected(self):
        sel = self._selected()
        if not sel:
            return
        for path, cls in sel:
            storage.move_sample(Path(path), CLASS_NG if cls == CLASS_OK else CLASS_OK)
        self._status(f"{len(sel)} sampel dipindahkan ke kelas lawannya.", "ok")
        self.refresh_counts()
        self.refresh_gallery()

    def move_to_profile(self):
        sel = self._selected()
        if not sel:
            self._status("Pilih sampel di galeri terlebih dahulu.", "warn")
            return
        others = [p for p in config.list_profiles() if p != self.ctx.profile]
        if not others:
            QtWidgets.QMessageBox.information(self, "Pindah ke profil lain",
                                              "Belum ada profil lain. Buat profil baru di menu Pengaturan.")
            return
        target, ok = QtWidgets.QInputDialog.getItem(
            self, "Pindah ke profil lain", f"Pindahkan {len(sel)} sampel ke profil:", others, 0, False)
        if not ok:
            return
        for path, cls in sel:
            storage.move_to_profile(Path(path), cls, target)
        self._status(f"{len(sel)} sampel dipindahkan ke profil '{target}'.", "ok")
        self.refresh_counts()
        self.refresh_gallery()

    def reload(self, cfg):
        """Muat ulang kontrol, dataset, dan info model (dipakai saat ganti profil)."""
        for w in (self.cb_model, self.chk_aug, self.sp_aug):
            w.blockSignals(True)
        try:
            self.cb_model.setCurrentIndex(max(0, self.cb_model.findData(cfg.model_kind)))
            self.chk_aug.setChecked(cfg.augment)
            self.sp_aug.setValue(cfg.aug_per_image)
        finally:
            for w in (self.cb_model, self.chk_aug, self.sp_aug):
                w.blockSignals(False)
        self._update_algo_hint()
        self._last = None
        self.progress.setValue(0)
        self.lbl_train.setText("")
        self._status(f"Profil aktif: {self.ctx.profile}", "")
        self.refresh_counts()
        self.refresh_gallery()
        self.report.clear()
        self.refresh_model_info(force=True)

    def import_images(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pilih gambar sampel", "", "Gambar (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not files:
            return
        items = [CLASS_DISPLAY[CLASS_OK], CLASS_DISPLAY[CLASS_NG]]
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Kelas sampel",
                                                    f"{len(files)} gambar ini termasuk kelas:", items, 0, False)
        if not ok:
            return
        cls = CLASS_OK if choice == items[0] else CLASS_NG
        n = storage.import_files(files, cls)
        self._status(f"{n} gambar diimpor ke {CLASS_DISPLAY[cls]}. Pastikan ukuran lubang di gambar "
                     "mirip dengan tampilan kamera.", "ok")
        self.refresh_counts()
        self.refresh_gallery()

    # ------------------------------------------------------------------
    def start_training(self):
        if self._train is not None:
            self._train.cancel()
            self.btn_train.setEnabled(False)
            self.btn_train.setText("Membatalkan…")
            return
        self.report.clear()
        self.progress.setValue(0)
        set_prop(self.progress, "tone", "")
        self._train = TrainWorker(self.ctx.cfg, self)
        self.ctx.training_active = True
        self._train.progress.connect(self._on_progress)
        self._train.finished_ok.connect(self._on_trained)
        self._train.failed.connect(self._on_failed)
        self._train.finished.connect(self._on_thread_done)
        self.btn_train.setText("Batalkan training")
        self.btn_train.setProperty("variant", "ng")
        set_prop(self.btn_train, "variant", "ng")
        self._train.start()

    def _on_progress(self, pct, msg):
        self.progress.setValue(pct)
        self.lbl_train.setText(msg)

    def _on_trained(self, bundle, report):
        self.ctx.models.save(bundle)
        self.ctx.model_changed.emit()
        self.report.setPlainText(format_report(report))
        set_prop(self.progress, "tone", "ok")
        self.lbl_train.setText("✓ Model baru aktif dan langsung dipakai di halaman Inspeksi.")

    def _on_failed(self, msg):
        set_prop(self.progress, "tone", "ng")
        self.lbl_train.setText(msg)
        self.report.setPlainText("Training gagal:\n" + msg)

    def _on_thread_done(self):
        self._train = None
        self.ctx.training_active = False
        self.btn_train.setText("Latih Model")
        set_prop(self.btn_train, "variant", "primary")
        self.refresh_counts()

    def delete_model(self):
        if not self.ctx.models.is_ready():
            return
        r = QtWidgets.QMessageBox.question(self, "Hapus model",
                                           "Hapus model aktif? Inspeksi akan memakai metode Rule.")
        if r == QtWidgets.QMessageBox.Yes:
            self.ctx.models.delete()
            self.ctx.model_changed.emit()

    def refresh_model_info(self, force=False):
        """Tampilkan info model di kotak laporan (tidak menimpa laporan training terbaru)."""
        m = self.ctx.models
        current = self.report.toPlainText()
        replaceable = force or not current or current.startswith(("Model aktif", "Belum ada"))
        if not m.is_ready():
            if replaceable or current.startswith("Model terpilih"):
                self.report.setPlainText("Belum ada model terlatih.\n\nLangkah:\n"
                                         "  1. Ambil sampel ULIR dan TANPA ULIR\n"
                                         "  2. Klik 'Latih Model'\n"
                                         "  3. Model otomatis dipakai di halaman Inspeksi\n\n"
                                         "Selama belum ada model, inspeksi memakai metode Rule.")
            return
        meta = m.meta
        txt = (f"Model aktif : {meta.get('model_name')}\n"
               f"Dilatih     : {meta.get('trained_at')}\n"
               f"Sampel      : ulir={meta.get('n_ok')}, tanpa ulir={meta.get('n_ng')}\n"
               f"Akurasi CV  : {meta.get('cv_balanced_accuracy', 0) * 100:.1f}% (balanced)\n")
        trained_p = meta.get("detect_params")
        cur = DetectParams.from_config(self.ctx.cfg)
        model_mode = meta.get("view_mode", "top")
        txt += f"Mode        : {VIEW_MODES.get(model_mode, model_mode)}\n"
        if model_mode != cur.view_mode:
            txt += ("\n⚠ Model dilatih untuk sudut pandang lain dari pengaturan sekarang.\n"
                    "  Latih ulang dengan sampel dari sudut pandang ini.\n")
        elif model_mode == "top" and trained_p and (abs(trained_p["band_inner"] - cur.band_inner) > 1e-6
                                                    or abs(trained_p["band_outer"] - cur.band_outer) > 1e-6):
            txt += ("\n⚠ Pengaturan area analisis (band) berbeda dari saat training.\n"
                    "  Latih ulang model agar hasil konsisten.\n")
        if replaceable:
            self.report.setPlainText(txt)
