"""Halaman Inspeksi: live view, judgement OK/NG, statistik, riwayat."""
import time
from collections import deque

import cv2
import numpy as np

from .. import storage
from ..config import AppConfig
from ..qt import QtCore, QtGui, QtWidgets, Qt
from ..vision import draw_overlay
from .widgets import (Card, JudgementCard, ScoreBar, Segmented, StatTile, ValueLabel, VideoView,
                      button, gray_to_pixmap, label, scroll_panel, video_frame)

HISTORY_MAX = 300
TEXT_REFRESH_S = 0.25      # teks nilai live diperbarui ±4x per detik agar mudah dibaca
SCORE_SMOOTHING = 0.35     # bobot nilai baru pada bar skor (EMA), makin kecil makin halus


class InspectPage(QtWidgets.QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._buffer = deque(maxlen=30)
        self._pending = 0
        self._collected = []
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
        self.lbl_proc = label("", "muted")
        bar.addWidget(self.lbl_proc)
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
        live.body.addWidget(label("Skor ulir (garis putih = ambang OK)", "hint"))
        live.body.addWidget(self.score_bar)
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self.v_method = ValueLabel()
        self.v_reason = ValueLabel()
        self.v_geo1 = ValueLabel()
        self.v_geo2 = ValueLabel()
        self.v_time = ValueLabel()
        self.n_geo1 = label("Jarak garis", "muted")
        self.n_geo2 = label("Kontras · sudut", "muted")
        rows = ((label("Metode", "muted"), self.v_method), (label("Status", "muted"), self.v_reason),
                (self.n_geo1, self.v_geo1), (self.n_geo2, self.v_geo2),
                (label("Waktu proses", "muted"), self.v_time))
        for i, (name, w) in enumerate(rows):
            name.setFixedWidth(118)
            grid.addWidget(name, i, 0, Qt.AlignVCenter)
            grid.addWidget(w, i, 1, Qt.AlignVCenter)
        grid.setColumnStretch(1, 1)
        self._t_text = 0.0
        self._score_ema = None
        live.body.addLayout(grid)
        self.polar = QtWidgets.QLabel()
        self.polar.setAlignment(Qt.AlignCenter)
        self.polar.setFixedHeight(110)
        self.polar.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.polar.setStyleSheet("background:#05080b; border-radius:6px;")
        self.lbl_view = label("", "hint", wrap=True)
        live.body.addWidget(self.lbl_view)
        live.body.addWidget(self.polar)
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
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["WAKTU", "HASIL", "SKOR", "METODE"])
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
        self.on_config(ctx.cfg)

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
            self.lbl_mode_hint.setText(f"Setiap trigger merata-rata {self.ctx.cfg.trigger_frames} frame, "
                                       "lalu hasil dihitung, dicatat ke log, dan disimpan.")
            self.judge.set_state("", "—", "Tekan INSPEKSI atau Spasi")
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
            n = max(1, cfg.continuous_smoothing)
            recent = list(self._buffer)[-n:]
            j, score, method, reason, _ = self._combine(recent)
            self._show_judgement(j, score, method, reason)
            self._apply_gpio(j)
        elif self._pending > 0:
            self._collected.append(res)
            self._pending -= 1
            if self._pending == 0:
                self._finalize()

    def _update_live(self, res):
        cfg = self.ctx.cfg
        thr = cfg.ok_threshold if res.method == "ML" else cfg.rule_threshold
        if res.score is None:
            self._score_ema = None
        elif self._score_ema is None:
            self._score_ema = res.score
        else:
            self._score_ema += SCORE_SMOOTHING * (res.score - self._score_ema)
        self.score_bar.set(self._score_ema, thr)

        now = time.monotonic()
        if now - self._t_text >= TEXT_REFRESH_S:
            self._t_text = now
            self.v_method.setText(res.method)
            self.v_reason.setText(res.reason)
            if res.circle is not None:
                self.v_geo1.setText(f"{res.circle[2]:.1f} px")
                self.v_geo2.setText("—")
            elif "period_px" in res.info:
                self.v_geo1.setText(f"{res.info['period_px']:.1f} px")
                self.v_geo2.setText(f"{res.info.get('contrast', 0):.1f}  ·  {res.info.get('angle', 0):+.0f}°")
            else:
                self.v_geo1.setText("—")
                self.v_geo2.setText("—")
            self.v_time.setText(f"{res.proc_ms:.1f} ms")
        if res.band is not None:
            w = max(60, self.polar.width())
            img = res.band.T if cfg.view_mode == "top" else res.band
            strip = cv2.resize(img, (w, self.polar.height()), interpolation=cv2.INTER_LINEAR)
            self.polar.setPixmap(gray_to_pixmap(strip))
        else:
            self.polar.clear()

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
        cfg = self.ctx.cfg
        n = len(results)
        na = [r for r in results if r.judgement == "N/A"]
        if len(na) * 2 >= n:
            return "N/A", None, na[-1].method, na[-1].reason, na[-1]
        unknown = [r for r in results if r.info.get("unknown")]
        if len(unknown) * 2 >= n:
            return "NG", 0.0, unknown[-1].method, unknown[-1].reason, unknown[-1]
        valid = [r for r in results if r.judgement in ("OK", "NG") and r.score is not None
                 and not r.info.get("unknown")]
        if len(valid) * 2 < n or not valid:
            nh = [r for r in results if r.judgement == "NO_HOLE"]
            return "NO_HOLE", None, "-", (nh[-1].reason if nh else "Hasil tidak stabil"), results[-1]
        score = float(np.mean([r.score for r in valid]))
        method = valid[-1].method
        thr = cfg.ok_threshold if method == "ML" else cfg.rule_threshold
        ok = score >= thr
        rep = min(valid, key=lambda r: abs(r.score - score))
        return ("OK" if ok else "NG"), score, method, ("Ulir terdeteksi" if ok else "Ulir tidak terdeteksi"), rep

    def _show_judgement(self, j, score, method, reason):
        pct = f"{score * 100:.1f}%" if score is not None else ""
        if j == "OK":
            self.judge.set_state("OK", "OK", f"{reason} · {pct} · {method}")
        elif j == "NG" and pct and "tidak dikenal" not in reason:
            self.judge.set_state("NG", "NG", f"{reason} · {pct} · {method}")
        elif j == "NG":
            self.judge.set_state("NG", "NG", reason)
        elif j == "NO_HOLE":
            self.judge.set_state("NG", "NG", "Lubang tidak terdeteksi")
        else:
            self.judge.set_state("WAIT", "N/A", reason)

    def _finalize(self):
        self._timeout.stop()
        self.btn_trigger.setEnabled(True)
        j, score, method, reason, rep = self._combine(self._collected)
        self._collected = []
        self._show_judgement(j, score, method, reason)
        self._apply_gpio(j)
        if j == "N/A":
            return
        final = "OK" if j == "OK" else "NG"
        self.n_total += 1
        if final == "OK":
            self.n_ok += 1
        else:
            self.n_ng += 1
        self._refresh_stats()
        self._add_history(final, score, method, reason)
        self._persist(final, score, method, reason, rep)

    def _persist(self, final, score, method, reason, rep):
        cfg = self.ctx.cfg
        img_path = ""
        if rep is not None and rep.frame is not None and (
                (final == "NG" and cfg.save_ng_images) or (final == "OK" and cfg.save_ok_images)):
            rep.judgement = final
            annotated = draw_overlay(rep.frame.copy(), rep, cfg)
            try:
                img_path = str(storage.save_inspection(rep, annotated))
            except OSError:
                img_path = "(gagal disimpan)"
        if cfg.log_csv:
            try:
                storage.append_log({
                    "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "hasil": final,
                    "skor": f"{score:.4f}" if score is not None else "",
                    "metode": method,
                    "keterangan": reason,
                    "radius_px": f"{rep.circle[2]:.1f}" if rep is not None and rep.circle else "",
                    "gambar": img_path,
                })
            except OSError:
                pass

    def _add_history(self, final, score, method, reason):
        self.table.insertRow(0)
        items = [time.strftime("%H:%M:%S"), final,
                 f"{score * 100:.1f}%" if score is not None else "—", method]
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
        self.score_bar.set(None, 0.5)
        self.reset_stats()
        self.mode.set_value(self.ctx.cfg.inspect_mode)
        self.on_config(self.ctx.cfg)

    def reset_stats(self):
        self.n_total = self.n_ok = self.n_ng = 0
        self._refresh_stats()
        self.table.setRowCount(0)

    def on_config(self, cfg):
        self.n_geo1.setText("Radius lubang" if cfg.view_mode == "top" else "Jarak garis")
        self.n_geo2.setVisible(cfg.view_mode != "top")
        self.v_geo2.setVisible(cfg.view_mode != "top")
        self.lbl_view.setText(
            "Tampilan polar dinding lubang (pola garis berulang = ulir)" if cfg.view_mode == "top"
            else "Dinding lubang yang diluruskan (garis mendatar berulang = ulir)")
        if self.mode.value() != cfg.inspect_mode:
            self.mode.set_value(cfg.inspect_mode)
        if cfg.inspect_mode == "trigger":
            self.lbl_mode_hint.setText(f"Setiap trigger merata-rata {cfg.trigger_frames} frame, "
                                       "lalu hasil dihitung, dicatat ke log, dan disimpan.")

    def _apply_gpio(self, judgement):
        self.ctx.gpio.set_result(judgement)

    def on_fps(self, fps):
        self.video.badge = f"●  LIVE   {fps:.1f} FPS"
