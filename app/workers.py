"""Thread latar belakang: akuisisi + inspeksi kamera, dan training model."""
import threading
import time
from dataclasses import replace as dc_replace

import cv2

from .camera import CameraError, create_source
from .model import TrainingError, train_model
from .qt import QtCore, QtGui, Signal
from .vision import DetectParams, Inspector, draw_overlay


def bgr_to_qimage(img):
    if img.ndim == 2:
        h, w = img.shape
        return QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


class FrameGrabber:
    """Membaca kamera terus-menerus di thread sendiri. Decoding gambar kamera (MJPG 1080p
    bisa ±15-30 ms) berjalan paralel dengan analisis, dan analisis selalu memakai frame
    terbaru (frame lama dibuang, tidak menumpuk)."""

    def __init__(self, source):
        self.source = source
        self.io_lock = threading.Lock()
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self.fails = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            with self.io_lock:
                ok, frame = self.source.read()
            with self._cond:
                if ok and frame is not None:
                    self._frame, self._seq, self.fails = frame, self._seq + 1, 0
                else:
                    self.fails += 1
                self._cond.notify_all()
            if not ok:
                time.sleep(0.02)

    def latest(self, last_seq, timeout=0.5):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or not self._running, timeout)
            return self._seq, self._frame

    def stop(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(2.0)


def _scaled(res, s):
    """Salinan hasil inspeksi dengan koordinat diskalakan (untuk menggambar di citra kecil)."""
    roi = tuple(int(round(v * s)) for v in res.roi_px)
    circle = None if res.circle is None else (res.circle[0] * s, res.circle[1] * s, res.circle[2] * s)
    return dc_replace(res, roi_px=roi, circle=circle, frame=None)


class CameraWorker(QtCore.QThread):
    frame_ready = Signal(object, object)       # QImage beranotasi, InspectionResult
    status_changed = Signal(bool, str)         # terhubung?, pesan
    fps_changed = Signal(float)

    DISPLAY_MAX_W = 1280

    def __init__(self, cfg, model_manager, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._lock = threading.Lock()
        self._running = True
        self._reopen = True
        self._driver_settings = False
        self._busy_gui = False
        self.inspector = Inspector(model_manager)
        self.source = None
        self.grabber = None

    # --- dipanggil dari thread GUI ---
    def update_config(self, cfg, reopen=False):
        with self._lock:
            self._cfg = cfg
            if reopen:
                self._reopen = True

    def reconnect(self):
        with self._lock:
            self._reopen = True

    def open_driver_settings(self):
        self._driver_settings = True

    def frame_consumed(self):
        self._busy_gui = False

    def stop(self):
        self._running = False
        self.wait(3000)

    # --- thread kamera ---
    def _close(self):
        if self.grabber is not None:
            self.grabber.stop()
            self.grabber = None
        if self.source is not None:
            self.source.release()
            self.source = None

    def _open(self, cfg):
        self._close()
        src = create_source(cfg)
        try:
            src.open()
        except CameraError as e:
            self.status_changed.emit(False, str(e))
            return False
        except Exception as e:  # driver kadang melempar error lain
            self.status_changed.emit(False, f"Gagal membuka kamera: {e}")
            return False
        self.source = src
        self.grabber = FrameGrabber(src)
        self.status_changed.emit(True, src.describe())
        return True

    def run(self):
        seq = 0
        t_fps, n_fps = time.perf_counter(), 0
        while self._running:
            with self._lock:
                cfg = self._cfg
                reopen = self._reopen
                self._reopen = False
            if reopen:
                seq = 0
                if not self._open(cfg):
                    self._sleep_until_reopen()
                    continue
            if self.source is None or self.grabber is None:
                self._sleep_until_reopen()
                continue
            if self._driver_settings:
                self._driver_settings = False
                with self.grabber.io_lock:
                    self.source.open_driver_settings()

            new_seq, frame = self.grabber.latest(seq)
            if new_seq == seq or frame is None:
                if self.grabber.fails > 30:
                    self.status_changed.emit(False, "Koneksi kamera terputus, mencoba ulang...")
                    self.msleep(1000)
                    with self._lock:
                        self._reopen = True
                continue
            seq = new_seq
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            result = self.inspector.inspect(frame, cfg)

            n_fps += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                self.fps_changed.emit(n_fps / (now - t_fps))
                t_fps, n_fps = now, 0

            if self._busy_gui:     # GUI belum selesai menggambar frame sebelumnya -> lewati
                continue
            # perkecil dulu, baru gambar overlay (jauh lebih ringan daripada menggambar di 1080p)
            if frame.shape[1] > self.DISPLAY_MAX_W:
                s = self.DISPLAY_MAX_W / frame.shape[1]
                disp = cv2.resize(frame, (self.DISPLAY_MAX_W, int(frame.shape[0] * s)),
                                  interpolation=cv2.INTER_AREA)
                vis = draw_overlay(disp, _scaled(result, s), cfg)
            else:
                vis = draw_overlay(frame.copy(), result, cfg)
            self._busy_gui = True
            self.frame_ready.emit(bgr_to_qimage(vis), result)

        self._close()

    def _sleep_until_reopen(self):
        for _ in range(10):
            if not self._running or self._reopen:
                return
            self.msleep(50)


class TrainWorker(QtCore.QThread):
    progress = Signal(int, str)
    finished_ok = Signal(object, object)    # bundle, report
    failed = Signal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = False

    def cancel(self):
        self._stop = True

    def run(self):
        try:
            bundle, report = train_model(
                DetectParams.from_config(self.cfg), self.cfg.model_kind, self.cfg.augment,
                self.cfg.aug_per_image, progress=lambda p, m: self.progress.emit(p, m),
                should_stop=lambda: self._stop)
        except TrainingError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:
            self.failed.emit(f"Error saat training: {e}")
            return
        self.finished_ok.emit(bundle, report)
