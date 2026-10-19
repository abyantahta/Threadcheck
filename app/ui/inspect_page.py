"""Halaman Inspeksi: live view, judgement OK/NG, statistik, riwayat."""
import time
from collections import deque

import numpy as np

from .. import storage
from ..config import AppConfig
from ..qt import QtCore, QtGui, QtWidgets, Qt
from ..vision import draw_overlay, square_preview
from .widgets import (Card, JudgementCard, ScoreBar, Segmented, StatTile, ValueLabel, VideoView,
                      button, gray_to_pixmap, label, scroll_panel, video_frame)

HISTORY_MAX = 300
TEXT_REFRESH_S = 0.25      # teks nilai live diperbarui ±4x per detik agar mudah dibaca
SCORE_SMOOTHING = 0.35     # bobot nilai baru pada bar skor (EMA), makin kecil makin halus
PREVIEW_SIZE = 150


class InspectPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._buffer = deque(maxlen=30)
        self._pending = 0
        self._collected = []
        self._t_text = 0.0
        self._score_ema = None
        self.n_total = self.n_ok = self.n_ng = 0

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
        self.btn_reset_roi = button("Reset ROI", tooltip="Kembalikan area inspeksi ke posisi default")
        self.btn_reset_roi.clicked.connect(lambda: self.ctx.update(roi=AppConfig().roi))
        bar.addWidget(self.btn_reset_roi)
        bar.addStretch(1)
        left.addLayout(bar)
        root.addLayout(left, 1)

        # ---- kanan: panel ----
        panel = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 10, 0)
        pl.setSpacing(12)

        self.judge = JudgementCard()
        self.judge.set_state("", "—", "Tekan INSPEKSI atau Spasi")
        pl.addWidget(self.judge)

        # mode
        mode_card = Card("Mode inspeksi")
        self.mode = Segmented([("trigger", "Trigger"), ("continuous", "Kontinu (live)")])
        self.mode.changed.connect(self._on_mode)
        mode_card.body.addWidget(self.mode)
        self.btn_trigger = button("INSPEKSI   ⏎ Spasi", "primary", "xl")
        self.btn_trigger.clicked.connect(self.trigger)
        mode_card.body.addWidget(self.btn_trigger)
        self.lbl_mode_hint = label("", "hint", wrap=True)
        mode_card.body.addWidget(self.lbl_mode_hint)
        pl.addWidget(mode_card)

        # analisis live
        live = Card("Analisis live")
        self.score_bar = ScoreBar()
        live.body.addWidget(label("Keyakinan berulir (garis putih = ambang OK)", "hint"))
        live.body.addWidget(self.score_bar)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.v_reason = ValueLabel()
        self.v_novelty = ValueLabel()
        self.v_time = ValueLabel()
        rows = ((label("Status", "muted"), self.v_reason),
                (label("Kemiripan objek", "muted"), self.v_novelty),
                (label("Waktu proses", "muted"), self.v_time))
        for i, (name, w) in enumerate(rows):
            name.setFixedWidth(118)
            grid.addWidget(name, i, 0, Qt.AlignVCenter)
            grid.addWidget(w, i, 1, Qt.AlignVCenter)
        grid.setColumnStretch(1, 1)
        self.v_novelty.setToolTip("Seberapa mirip gambar dengan sampel training. Di atas batas toleransi, "
                                  "hasil dinyatakan NG 'objek tidak dikenal'.")
        live.body.addLayout(grid)
        live.body.addWidget(label("Gambar yang dinilai model (potongan ROI)", "hint"))
        self.preview = QtWidgets.QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(PREVIEW_SIZE + 8)
        self.preview.setStyleSheet("background:#05080b; border-radius:6px;")
        live.body.addWidget(self.preview)
        pl.addWidget(live)

        # statistik
        stats = Card("Statistik")
        btn_reset = button("Reset")
        btn_reset.setFixedHeight(28)
        btn_reset.clicked.connect(self.reset_stats)
        stats.header.addWidget(btn_reset)
        row = QtWidgets.QGridLayout()
        row.setSpacing(8)
        self.t_total = StatTile("Total", "accent")
        self.t_ok = StatTile("OK", "ok")
        self.t_ng = StatTile("NG", "ng")
        self.t_yield = StatTile("Yield")
        for i, t in enumerate((self.t_total, self.t_ok, self.t_ng, self.t_yield)):
            row.addWidget(t, i // 2, i % 2)
        stats.body.addLayout(row)
        pl.addWidget(stats)

        # riwayat
        hist = Card("Riwayat inspeksi")
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["WAKTU", "HASIL", "KEYAKINAN"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(200)
        self.table.setShowGrid(False)
        hist.body.addWidget(self.table)
        pl.addWidget(hist)
        pl.addStretch(1)

        root.addWidget(scroll_panel(panel, 410))

        self._timeout = QtCore.QTimer(self)
        self._timeout.setSingleShot(True)
        self._timeout.timeout.connect(self._trigger_timeout)

        self.mode.set_value(ctx.cfg.inspect_mode)
        self._apply_mode(ctx.cfg.inspect_mode)

    # ------------------------------------------------------------------
    def _on_mode(self, mode):
        self.ctx.update(inspect_mode=mode)
        self._apply_mode(mode)

    def _apply_mode(self, mode):
        trig = mode == "trigger"
        self.btn_trigger.setEnabled(trig)
        self._buffer.clear()
        self._apply_gpio(None)
        if trig:
            self.judge.set_state("", "—", "Tekan INSPEKSI atau Spasi")
        self._mode_hint(self.ctx.cfg)

    def _mode_hint(self, cfg):
        if cfg.inspect_mode == "trigger":
            self.lbl_mode_hint.setText(f"Setiap trigger merata-rata {cfg.trigger_frames} frame, "
                                       "lalu hasil dihitung, dicatat ke log, dan disimpan.")
        else:
            self.lbl_mode_hint.setText("Judgement diperbarui terus-menerus dari video. "
                                       "Mode ini tidak menghitung statistik/log.")

    def on_camera_status(self, connected, msg):
        if not connected:
            self.video.set_placeholder(msg)

    def on_result(self, qimg, res):
        self.video.set_image(qimg)
        self._update_live(res)
        cfg = self.ctx.cfg
        if cfg.inspect_mode == "continuous":
            self._buffer.append(res)
            recent = list(self._buffer)[-max(1, cfg.continuous_smoothing):]
            j, score, reason, _ = self._combine(recent)
            self._show_judgement(j, score, reason)
            self._apply_gpio(j)
        elif self._pending > 0:
            self._collected.append(res)
            self._pending -= 1
            if self._pending == 0:
                self._finalize()

    def _update_live(self, res):
        if res.score is None:
            self._score_ema = None
        elif self._score_ema is None:
            self._score_ema = res.score
        else:
            self._score_ema += SCORE_SMOOTHING * (res.score - self._score_ema)
        self.score_bar.set(self._score_ema, self.ctx.cfg.ok_threshold)

        now = time.monotonic()
        if now - self._t_text < TEXT_REFRESH_S or not self.isVisible():
            return
        self._t_text = now
        self.v_reason.setText(res.reason)
        tol = self.ctx.cfg.novelty_tolerance
        if res.novelty is None:
            self.v_novelty.setText("—")
        else:
            state = "asing" if (tol > 0 and res.novelty > tol) else "dikenal"
            self.v_novelty.setText(f"{state}  ·  {res.novelty:.2f} (batas {tol:.1f})")
        self.v_time.setText(f"{res.proc_ms:.1f} ms")
        if res.frame is not None:
            x, y, w, h = res.roi_px
            prev = square_preview(res.frame[y:y + h, x:x + w], PREVIEW_SIZE)
            self.preview.setPixmap(gray_to_pixmap(prev))

    # ------------------------------------------------------------------
    def trigger(self):
        if self.ctx.cfg.inspect_mode != "trigger" or self._pending > 0:
            return
        self._collected = []
        self._pending = max(1, self.ctx.cfg.trigger_frames)
        self._apply_gpio(None)
        self.btn_trigger.setEnabled(False)
        self.judge.set_state("WAIT", "…", "Menganalisis")
        self._timeout.start(4000)

    def _trigger_timeout(self):
        if self._pending > 0:
            self._pending = 0
            if self._collected:
                self._finalize()
            else:
                self.btn_trigger.setEnabled(True)
                self._apply_gpio(None)
                self.judge.set_state("WAIT", "!", "Tidak ada gambar dari kamera")

    def _combine(self, results):
        """Gabungkan beberapa frame menjadi satu keputusan: (judgement, skor, alasan, contoh)."""
        n = len(results)
        na = [r for r in results if r.judgement == "N/A"]
        if len(na) * 2 >= n:
            return "N/A", None, na[-1].reason, na[-1]
        unknown = [r for r in results if r.unknown]
        if len(unknown) * 2 >= n:
            return "NG", None, unknown[-1].reason, unknown[-1]
        valid = [r for r in results if r.judgement in ("OK", "NG") and not r.unknown]
        score = float(np.mean([r.score for r in valid]))
        ok = score >= self.ctx.cfg.ok_threshold
        rep = min(valid, key=lambda r: abs(r.score - score))
        return ("OK" if ok else "NG"), score, ("Ulir terdeteksi" if ok else "Ulir tidak terdeteksi"), rep

    def _show_judgement(self, j, score, reason):
        pct = f" · {score * 100:.1f}%" if score is not None else ""
        if j in ("OK", "NG"):
            self.judge.set_state(j, j, reason + pct)
        else:
            self.judge.set_state("WAIT", "N/A", reason)

    def _finalize(self):
        self._timeout.stop()
        self.btn_trigger.setEnabled(True)
        j, score, reason, rep = self._combine(self._collected)
        self._collected = []
        self._show_judgement(j, score, reason)
        self._apply_gpio(j)
        if j == "N/A":
            return
        self.n_total += 1
        if j == "OK":
            self.n_ok += 1
        else:
            self.n_ng += 1
        self._refresh_stats()
        self._add_history(j, score, reason)
        self._persist(j, score, reason, rep)

    def _persist(self, final, score, reason, rep):
        cfg = self.ctx.cfg
        img_path = ""
        if rep is not None and rep.frame is not None and (
                (final == "NG" and cfg.save_ng_images) or (final == "OK" and cfg.save_ok_images)):
            rep.judgement = final
            try:
                img_path = str(storage.save_inspection(rep, draw_overlay(rep.frame.copy(), rep)))
            except OSError:
                img_path = "(gagal disimpan)"
        if cfg.log_csv:
            try:
                storage.append_log({
                    "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hasil": final,
                    "keyakinan_ulir": f"{score:.4f}" if score is not None else "",
                    "keterangan": reason,
                    "gambar": img_path,
                })
            except OSError:
                pass

    def _add_history(self, final, score, reason):
        self.table.insertRow(0)
        items = [time.strftime("%H:%M:%S"), final, f"{score * 100:.1f}%" if score is not None else "asing"]
        for c, text in enumerate(items):
            it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(Qt.AlignCenter)
            it.setToolTip(reason)
            if c == 1:
                it.setForeground(QtGui.QColor("#22c55e" if final == "OK" else "#ef4444"))
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.table.setItem(0, c, it)
        while self.table.rowCount() > HISTORY_MAX:
            self.table.removeRow(self.table.rowCount() - 1)

    def _refresh_stats(self):
        self.t_total.set(self.n_total)
        self.t_ok.set(self.n_ok)
        self.t_ng.set(self.n_ng)
        self.t_yield.set(f"{self.n_ok / self.n_total * 100:.1f}%" if self.n_total else "—")

    def reset_for_profile(self):
        """Profil kamera berganti: kosongkan statistik, riwayat, dan hasil terakhir."""
        self._pending = 0
        self._collected = []
        self._buffer.clear()
        self._apply_gpio(None)
        self._timeout.stop()
        self.btn_trigger.setEnabled(self.ctx.cfg.inspect_mode == "trigger")
        self.judge.set_state("", "—", "Profil diganti · tekan INSPEKSI atau Spasi")
        self._score_ema = None
        self.score_bar.set(None, self.ctx.cfg.ok_threshold)
        self.preview.clear()
        self.reset_stats()
        self.mode.set_value(self.ctx.cfg.inspect_mode)
        self.on_config(self.ctx.cfg)

    def reset_stats(self):
        self.n_total = self.n_ok = self.n_ng = 0
        self._refresh_stats()
        self.table.setRowCount(0)

    def on_config(self, cfg):
        if self.mode.value() != cfg.inspect_mode:
            self.mode.set_value(cfg.inspect_mode)
        self._mode_hint(cfg)

    def _apply_gpio(self, judgement):
        self.ctx.gpio.set_result(judgement)

    def on_fps(self, fps):
        self.video.badge = f"●  LIVE   {fps:.1f} FPS"
