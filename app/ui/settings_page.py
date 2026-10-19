"""Halaman Pengaturan: profil kamera, kamera, keputusan, penyimpanan."""
import platform

import cv2

from .. import __version__
from ..camera import (BACKENDS, IS_WINDOWS, RESOLUTIONS, SOURCE_TYPES, list_opencv_cameras,
                      list_picamera2_cameras)
from .. import config
from ..config import APP_NAME, BASE_DIR, DISPLAY_MODES
from ..qt import QT_API, QtWidgets, Qt
from . import profiles as profile_actions
from .training_page import open_folder
from .widgets import Card, button, label, scroll_panel, set_compact_cards, set_prop

def _form():
    f = QtWidgets.QFormLayout()
    f.setHorizontalSpacing(14)
    f.setVerticalSpacing(10)
    f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return f


def _gpio_column(result, enabled, pin, active_high, pulse_ms):
    """Satu kolom pengaturan pin untuk hasil OK atau NG."""
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    lay.addWidget(label(f"HASIL {result}", "cardTitle"))
    chk = QtWidgets.QCheckBox(f"Nyalakan GPIO ketika hasil {result}")
    chk.setChecked(enabled)
    lay.addWidget(chk)
    form = _form()
    sp_pin = _spin(2, 27, pin, 1)
    sp_pin.setToolTip(f"Nomor pin BCM untuk hasil {result}.")
    cb = QtWidgets.QComboBox()
    cb.addItem(f"Aktif HIGH (3,3 V saat {result})", True)
    cb.addItem(f"Aktif LOW (0 V saat {result})", False)
    cb.setCurrentIndex(0 if active_high else 1)
    sp_ms = _spin(0, 5000, pulse_ms, 50, " ms")
    sp_ms.setSpecialValueText(f"tahan selama {result}")
    sp_ms.setToolTip(f"Lama pin menyala setiap hasil {result}. 0 = tetap menyala selama hasil {result}.")
    form.addRow(label("Pin BCM", "muted"), sp_pin)
    form.addRow(label("Level aktif", "muted"), cb)
    form.addRow(label("Durasi pulsa", "muted"), sp_ms)
    lay.addLayout(form)
    btn = button(f"Uji nyala {result}", tooltip=f"Nyalakan pin {result} sebentar untuk mencoba kabel")
    row = QtWidgets.QHBoxLayout()
    row.addWidget(btn)
    row.addStretch(1)
    lay.addLayout(row)
    lbl = label("", "status", wrap=True)
    lay.addWidget(lbl)
    lay.addStretch(1)
    return box, chk, sp_pin, cb, sp_ms, btn, lbl


def _spin(lo, hi, val, step=1, suffix="", decimals=None):
    if decimals is None:
        s = QtWidgets.QSpinBox()
    else:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class SettingsPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        cfg = ctx.cfg

        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setSpacing(16)
        self._grid = grid

        # ================= Profil kamera =================
        prof = Card("Profil kamera")
        prof.body.addWidget(label(
            "Setiap profil punya kamera, ROI, pengaturan analisis, dataset, dan model sendiri. "
            "Ganti profil cepat dari header di atas. Dobel-klik profil di daftar untuk mengaktifkannya.",
            "hint", wrap=True))
        prow = QtWidgets.QHBoxLayout()
        self.lst_profiles = QtWidgets.QListWidget()
        self.lst_profiles.setFixedHeight(110)
        self.lst_profiles.itemDoubleClicked.connect(
            lambda it: profile_actions.switch(self, self.ctx, it.data(Qt.UserRole)))
        prow.addWidget(self.lst_profiles, 1)
        pbtn = QtWidgets.QVBoxLayout()
        b_new = button("Profil baru…", "primary")
        b_new.clicked.connect(lambda: profile_actions.create(self, self.ctx))
        b_ren = button("Ganti nama…")
        b_ren.clicked.connect(lambda: profile_actions.rename(self, self.ctx))
        b_del = button("Hapus…")
        b_del.setToolTip("Hapus profil yang dipilih di daftar")
        b_del.clicked.connect(self._delete_selected_profile)
        for b in (b_new, b_ren, b_del):
            pbtn.addWidget(b)
        pbtn.addStretch(1)
        prow.addLayout(pbtn)
        prof.body.addLayout(prow)


        # ================= Kamera =================
        cam = Card("Kamera")
        f = _form()
        self.cb_source = QtWidgets.QComboBox()
        for k, v in SOURCE_TYPES.items():
            self.cb_source.addItem(v, k)
        self.cb_source.setCurrentIndex(max(0, self.cb_source.findData(cfg.camera_source)))
        self.cb_source.currentIndexChanged.connect(self._source_changed)
        f.addRow(label("Sumber", "muted"), self.cb_source)

        dev = QtWidgets.QHBoxLayout()
        self.cb_device = QtWidgets.QComboBox()
        self.cb_device.addItem(f"#{cfg.camera_index} · Kamera", cfg.camera_index)
        self.btn_scan = button("Cari kamera", tooltip="Deteksi kamera yang terpasang")
        self.btn_scan.clicked.connect(self.scan_cameras)
        dev.addWidget(self.cb_device, 1)
        dev.addWidget(self.btn_scan)
        f.addRow(label("Perangkat", "muted"), dev)

        self.cb_backend = QtWidgets.QComboBox()
        for k, (name, _) in BACKENDS.items():
            self.cb_backend.addItem(name, k)
        self.cb_backend.setCurrentIndex(max(0, self.cb_backend.findData(cfg.camera_backend)))
        f.addRow(label("Backend", "muted"), self.cb_backend)

        self.cb_res = QtWidgets.QComboBox()
        for w, h in RESOLUTIONS:
            self.cb_res.addItem(f"{w} × {h}", (w, h))
        idx = self.cb_res.findData((cfg.camera_width, cfg.camera_height))
        if idx < 0:
            self.cb_res.addItem(f"{cfg.camera_width} × {cfg.camera_height}", (cfg.camera_width, cfg.camera_height))
            idx = self.cb_res.count() - 1
        self.cb_res.setCurrentIndex(idx)
        f.addRow(label("Resolusi", "muted"), self.cb_res)

        fl = QtWidgets.QHBoxLayout()
        self.ed_file = QtWidgets.QLineEdit(cfg.camera_file)
        self.ed_file.setPlaceholderText("Pilih video / gambar untuk simulasi")
        self.btn_file = button("Pilih…")
        self.btn_file.clicked.connect(self.pick_file)
        fl.addWidget(self.ed_file, 1)
        fl.addWidget(self.btn_file)
        f.addRow(label("File", "muted"), fl)
        cam.body.addLayout(f)

        act = QtWidgets.QHBoxLayout()
        self.btn_apply_cam = button("Terapkan && Hubungkan", "primary")
        self.btn_apply_cam.clicked.connect(self.apply_camera)
        self.btn_driver = button("Pengaturan driver…",
                                 tooltip="Buka dialog exposure/fokus/gain dari driver kamera "
                                         "(DirectShow di Windows)")
        self.btn_driver.clicked.connect(lambda: self.ctx.worker and self.ctx.worker.open_driver_settings())
        act.addWidget(self.btn_apply_cam, 1)
        act.addWidget(self.btn_driver)
        cam.body.addLayout(act)
        self.lbl_cam = label("—", "status", wrap=True)
        cam.body.addWidget(self.lbl_cam)
        cam.body.addStretch(1)


        # ================= Keputusan =================
        dec = Card("Keputusan OK / NG")
        f = _form()
        self.sp_thr_ml = _spin(0.05, 0.99, cfg.ok_threshold, 0.05, "", 2)
        self.sp_thr_ml.setToolTip("Probabilitas minimal 'berulir' agar dinyatakan OK. "
                                  "Naikkan untuk lebih ketat (mengurangi part tanpa ulir yang lolos).")
        self.sp_trig = _spin(1, 30, cfg.trigger_frames, 1, "  frame")
        self.sp_novelty = _spin(0.0, 5.0, cfg.novelty_tolerance, 0.1, " ×", 1)
        self.sp_novelty.setToolTip("Gambar yang terlalu berbeda dari sampel training dinyatakan NG "
                                   "'Unidentified Object'. Lebih besar = lebih longgar, 0 = nonaktif.")
        self.sp_smooth = _spin(1, 30, cfg.continuous_smoothing, 1, "  frame")
        f.addRow(label("Ambang OK", "muted"), self.sp_thr_ml)
        f.addRow(label("Toleransi objek asing", "muted"), self.sp_novelty)
        f.addRow(label("Frame per trigger", "muted"), self.sp_trig)
        f.addRow(label("Penghalusan kontinu", "muted"), self.sp_smooth)
        dec.body.addLayout(f)
        self.sp_thr_ml.valueChanged.connect(lambda v: self.ctx.update(ok_threshold=v))
        self.sp_trig.valueChanged.connect(lambda v: self.ctx.update(trigger_frames=v))
        self.sp_novelty.valueChanged.connect(lambda v: self.ctx.update(novelty_tolerance=v))
        self.sp_smooth.valueChanged.connect(lambda v: self.ctx.update(continuous_smoothing=v))
        dec.body.addStretch(1)


        # ================= Output GPIO =================
        gpio = Card("Output GPIO")
        gpio.body.addWidget(label(
            "Hasil OK dan NG masing-masing bisa menyalakan pin Raspberry Pi (nomor BCM) yang berbeda. "
            "Pulsa cocok untuk lampu atau PLC. Isi 0 agar pin tetap menyala selama hasil itu masih aktif. "
            "Lubang tidak terdeteksi ikut menyalakan pin NG. "
            "Di PC Windows aturan hanya disimpan dan baru menggerakkan pin di Raspberry Pi. "
            "Untuk tegangan 24 V gunakan optocoupler atau relay.",
            "hint", wrap=True))
        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(24)
        (ok_box, self.chk_gpio, self.sp_gpio_pin, self.cb_gpio_level, self.sp_gpio_ms,
         self.btn_gpio_test, self.lbl_gpio) = _gpio_column(
            "OK", cfg.gpio_ok_enabled, cfg.gpio_ok_pin, cfg.gpio_ok_active_high, cfg.gpio_ok_pulse_ms)
        (ng_box, self.chk_gpio_ng, self.sp_gpio_ng_pin, self.cb_gpio_ng_level, self.sp_gpio_ng_ms,
         self.btn_gpio_ng_test, self.lbl_gpio_ng) = _gpio_column(
            "NG", cfg.gpio_ng_enabled, cfg.gpio_ng_pin, cfg.gpio_ng_active_high, cfg.gpio_ng_pulse_ms)
        cols.addWidget(ok_box, 1)
        cols.addWidget(ng_box, 1)
        gpio.body.addLayout(cols)
        for w in (self.chk_gpio, self.sp_gpio_pin, self.cb_gpio_level, self.sp_gpio_ms,
                  self.chk_gpio_ng, self.sp_gpio_ng_pin, self.cb_gpio_ng_level, self.sp_gpio_ng_ms):
            signal = w.toggled if isinstance(w, QtWidgets.QCheckBox) else (
                w.currentIndexChanged if isinstance(w, QtWidgets.QComboBox) else w.valueChanged)
            signal.connect(self._sync_gpio)
        self.btn_gpio_test.clicked.connect(lambda: self._test_gpio("ok"))
        self.btn_gpio_ng_test.clicked.connect(lambda: self._test_gpio("ng"))


        # ================= Penyimpanan =================
        sto = Card("Penyimpanan & log")
        self.chk_ng = QtWidgets.QCheckBox("Simpan gambar hasil NG")
        self.chk_ok = QtWidgets.QCheckBox("Simpan gambar hasil OK")
        self.chk_log = QtWidgets.QCheckBox("Catat setiap inspeksi ke log CSV harian")
        self.chk_ng.setChecked(cfg.save_ng_images)
        self.chk_ok.setChecked(cfg.save_ok_images)
        self.chk_log.setChecked(cfg.log_csv)
        self.chk_ng.toggled.connect(lambda v: self.ctx.update(save_ng_images=v))
        self.chk_ok.toggled.connect(lambda v: self.ctx.update(save_ok_images=v))
        self.chk_log.toggled.connect(lambda v: self.ctx.update(log_csv=v))
        for w in (self.chk_ng, self.chk_ok, self.chk_log):
            sto.body.addWidget(w)
        folders = QtWidgets.QHBoxLayout()
        for text, path_fn in (("Folder gambar", config.captures_dir), ("Folder log", config.logs_dir),
                              ("Folder dataset", config.dataset_dir)):
            b = button(text)
            b.setToolTip("Folder milik profil yang sedang aktif")
            b.clicked.connect(lambda _=False, fn=path_fn: open_folder(fn()))
            folders.addWidget(b)
        sto.body.addLayout(folders)
        sto.body.addStretch(1)


        # ================= Info =================
        info = Card("Informasi sistem")
        info.body.addWidget(label(
            f"{APP_NAME} v{__version__}  ·  Python {platform.python_version()}  ·  "
            f"OpenCV {cv2.__version__}  ·  {QT_API}\n"
            f"Sistem: {platform.system()} {platform.machine()}\n"
            f"Folder aplikasi: {BASE_DIR}", "hint", wrap=True))
        info.body.addStretch(1)

        # ================= Tampilan =================
        disp = Card("Tampilan")
        disp.body.addWidget(label(
            "Raspberry Pi 800×480: tata letak ringkas dan layar penuh untuk layar sentuh kecil. "
            "Di PC/laptop mode ini tampil sebagai jendela 800×480 untuk pratinjau. "
            "F11 = keluar/masuk layar penuh.", "hint", wrap=True))
        f = _form()
        self.cb_display = QtWidgets.QComboBox()
        for k, v in DISPLAY_MODES.items():
            self.cb_display.addItem(v, k)
        self.cb_display.setCurrentIndex(max(0, self.cb_display.findData(cfg.display_mode)))
        self.cb_display.currentIndexChanged.connect(
            lambda: self.ctx.update(display_mode=self.cb_display.currentData()))
        f.addRow(label("Mode tampilan", "muted"), self.cb_display)
        disp.body.addLayout(f)
        disp.body.addStretch(1)

        # susunan kartu: (kartu, baris, kolom, rowspan, colspan) untuk layar normal;
        # layar kecil memakai satu kolom dengan urutan yang sama
        self._cards = [(prof, 0, 0, 1, 2), (cam, 1, 0, 1, 1), (dec, 1, 1, 1, 1),
                       (disp, 2, 0, 1, 1), (sto, 2, 1, 1, 1), (gpio, 3, 0, 1, 2), (info, 4, 0, 1, 2)]
        self._arrange(compact=False)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll_panel(content))

        self._source_changed()
        self.refresh_profiles()
        self._show_gpio()

    # ------------------------------------------------------------------
    def _arrange(self, compact: bool):
        grid = self._grid
        for card, *_ in self._cards:
            grid.removeWidget(card)
        for i, (card, r, c, rs, cs) in enumerate(self._cards):
            if compact:
                grid.addWidget(card, i, 0)
            else:
                grid.addWidget(card, r, c, rs, cs)
        for r in range(len(self._cards) + 1):
            grid.setRowStretch(r, 0)
        grid.setRowStretch(len(self._cards) + 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0 if compact else 1)

    def set_compact(self, compact: bool):
        """Satu kolom & jarak rapat untuk layar kecil (Raspberry Pi 800×480)."""
        m = 6 if compact else 16
        self._grid.setContentsMargins(m, m, m, m)
        self._grid.setSpacing(m)
        self.lst_profiles.setFixedHeight(80 if compact else 110)
        self._arrange(compact)
        set_compact_cards(self, compact)
        self.cb_display.blockSignals(True)
        self.cb_display.setCurrentIndex(max(0, self.cb_display.findData(self.ctx.cfg.display_mode)))
        self.cb_display.blockSignals(False)

    def refresh_profiles(self):
        self.lst_profiles.clear()
        for name in config.list_profiles():
            active = name == self.ctx.profile
            it = QtWidgets.QListWidgetItem(
                f"{'●' if active else '○'}  {name}{'  (aktif)' if active else ''}   —   "
                f"{profile_actions.profile_summary(name)}")
            it.setData(Qt.UserRole, name)
            if active:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.lst_profiles.addItem(it)

    def _delete_selected_profile(self):
        it = self.lst_profiles.currentItem()
        if it is None:
            QtWidgets.QMessageBox.information(self, "Hapus profil", "Pilih profil di daftar terlebih dahulu.")
            return
        profile_actions.delete(self, self.ctx, it.data(Qt.UserRole))

    def reload(self, cfg):
        """Isi ulang semua kontrol dari konfigurasi (dipakai saat ganti profil)."""
        widgets = [self.cb_source, self.cb_device, self.cb_backend, self.cb_res, self.ed_file,
                   self.sp_thr_ml, self.sp_novelty, self.sp_trig, self.sp_smooth,
                   self.chk_gpio, self.sp_gpio_pin, self.cb_gpio_level, self.sp_gpio_ms,
                   self.chk_gpio_ng, self.sp_gpio_ng_pin, self.cb_gpio_ng_level, self.sp_gpio_ng_ms,
                   self.chk_ng, self.chk_ok, self.chk_log]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.cb_source.setCurrentIndex(max(0, self.cb_source.findData(cfg.camera_source)))
            if self.cb_device.findData(cfg.camera_index) < 0:
                self.cb_device.addItem(f"#{cfg.camera_index} · Kamera", cfg.camera_index)
            self.cb_device.setCurrentIndex(self.cb_device.findData(cfg.camera_index))
            self.cb_backend.setCurrentIndex(max(0, self.cb_backend.findData(cfg.camera_backend)))
            res = (cfg.camera_width, cfg.camera_height)
            if self.cb_res.findData(res) < 0:
                self.cb_res.addItem(f"{res[0]} × {res[1]}", res)
            self.cb_res.setCurrentIndex(self.cb_res.findData(res))
            self.ed_file.setText(cfg.camera_file)
            self.sp_thr_ml.setValue(cfg.ok_threshold)
            self.sp_novelty.setValue(cfg.novelty_tolerance)
            self.sp_trig.setValue(cfg.trigger_frames)
            self.sp_smooth.setValue(cfg.continuous_smoothing)
            self.chk_gpio.setChecked(cfg.gpio_ok_enabled)
            self.sp_gpio_pin.setValue(cfg.gpio_ok_pin)
            self.cb_gpio_level.setCurrentIndex(0 if cfg.gpio_ok_active_high else 1)
            self.sp_gpio_ms.setValue(cfg.gpio_ok_pulse_ms)
            self.chk_gpio_ng.setChecked(cfg.gpio_ng_enabled)
            self.sp_gpio_ng_pin.setValue(cfg.gpio_ng_pin)
            self.cb_gpio_ng_level.setCurrentIndex(0 if cfg.gpio_ng_active_high else 1)
            self.sp_gpio_ng_ms.setValue(cfg.gpio_ng_pulse_ms)
            self.chk_ng.setChecked(cfg.save_ng_images)
            self.chk_ok.setChecked(cfg.save_ok_images)
            self.chk_log.setChecked(cfg.log_csv)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._source_changed()
        self.ctx.gpio.apply_config(cfg)
        self._show_gpio()
        self.refresh_profiles()
        self._cam_status("Menghubungkan kamera profil…", "warn")

    def _sync_gpio(self, *_):
        self.ctx.update(
            gpio_ok_enabled=self.chk_gpio.isChecked(),
            gpio_ok_pin=int(self.sp_gpio_pin.value()),
            gpio_ok_active_high=bool(self.cb_gpio_level.currentData()),
            gpio_ok_pulse_ms=int(self.sp_gpio_ms.value()),
            gpio_ng_enabled=self.chk_gpio_ng.isChecked(),
            gpio_ng_pin=int(self.sp_gpio_ng_pin.value()),
            gpio_ng_active_high=bool(self.cb_gpio_ng_level.currentData()),
            gpio_ng_pulse_ms=int(self.sp_gpio_ng_ms.value()),
        )
        self._show_gpio()

    def _test_gpio(self, kind):
        self.ctx.gpio.test(kind)
        self._show_gpio()

    def _show_gpio(self):
        for lbl, ch in ((self.lbl_gpio, self.ctx.gpio.ok), (self.lbl_gpio_ng, self.ctx.gpio.ng)):
            tone, text = ch.status()
            lbl.setText(text)
            set_prop(lbl, "tone", tone)

    def _source_changed(self):
        src = self.cb_source.currentData()
        is_cv, is_pi, is_file = src == "opencv", src == "picamera2", src == "file"
        self.cb_device.setEnabled(is_cv or is_pi)
        self.btn_scan.setEnabled(is_cv or is_pi)
        self.cb_backend.setEnabled(is_cv)
        self.cb_res.setEnabled(is_cv or is_pi)
        self.ed_file.setEnabled(is_file)
        self.btn_file.setEnabled(is_file)
        self.btn_driver.setEnabled(is_cv and IS_WINDOWS)

    def scan_cameras(self):
        src = self.cb_source.currentData()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if src == "picamera2":
                cams = list_picamera2_cameras()
            else:
                active = ()
                if self.ctx.cfg.camera_source == "opencv":
                    active = (self.ctx.cfg.camera_index,)
                cams = list_opencv_cameras(self.cb_backend.currentData(), skip_indices=active)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        current = self.cb_device.currentData()
        self.cb_device.clear()
        for c in cams:
            self.cb_device.addItem(c.label, c.index)
        if not cams:
            self.cb_device.addItem("Tidak ada kamera ditemukan", 0)
        i = self.cb_device.findData(current)
        self.cb_device.setCurrentIndex(max(0, i))
        self._cam_status(f"Ditemukan {len(cams)} kamera.", "ok" if cams else "ng")

    def pick_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pilih file simulasi", str(BASE_DIR),
            "Video / gambar (*.mp4 *.avi *.mov *.mkv *.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ed_file.setText(path)

    def apply_camera(self):
        w, h = self.cb_res.currentData()
        self.ctx.update(
            reopen_camera=True,
            camera_source=self.cb_source.currentData(),
            camera_index=int(self.cb_device.currentData() or 0),
            camera_backend=self.cb_backend.currentData(),
            camera_width=int(w), camera_height=int(h),
            camera_file=self.ed_file.text().strip(),
        )
        self._cam_status("Menghubungkan…", "warn")

    def _cam_status(self, text, tone=""):
        self.lbl_cam.setText(text)
        set_prop(self.lbl_cam, "tone", tone)

    def on_camera_status(self, connected, msg):
        self._cam_status(("Aktif: " if connected else "") + msg, "ok" if connected else "ng")
