"""Inspeksi per frame: potong ROI, klasifikasi YOLO26n-cls, penolak objek asing, overlay."""
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

@dataclass
class InspectionResult:
    judgement: str                       # OK | NG | N/A
    score: Optional[float]               # probabilitas "berulir" 0..1 dari YOLO
    reason: str
    roi_px: tuple                        # (x, y, w, h) dalam koordinat frame
    proc_ms: float
    frame: Optional[np.ndarray] = None   # frame mentah (BGR) untuk disimpan
    timestamp: float = 0.0
    novelty: Optional[float] = None      # > 1 = jauh berbeda dari sampel training
    unknown: bool = False                # ditolak sebagai objek tidak dikenal


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------
def roi_to_pixels(roi_norm, frame_shape) -> tuple:
    h, w = frame_shape[:2]
    x, y, rw, rh = roi_norm
    x0 = int(round(np.clip(x, 0, 1) * w))
    y0 = int(round(np.clip(y, 0, 1) * h))
    x1 = int(round(np.clip(x + rw, 0, 1) * w))
    y1 = int(round(np.clip(y + rh, 0, 1) * h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0, 0, w, h
    return x0, y0, x1 - x0, y1 - y0


def square_preview(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Potongan tengah persegi, sama seperti yang dilihat model YOLO."""
    h, w = img.shape[:2]
    s = size / min(h, w)
    r = cv2.resize(img, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (r.shape[0] - size) // 2, (r.shape[1] - size) // 2
    return r[y0:y0 + size, x0:x0 + size]


# ---------------------------------------------------------------------------
# Inspeksi
# ---------------------------------------------------------------------------
class Inspector:
    def __init__(self, model_manager):
        self.models = model_manager

    def inspect(self, frame: np.ndarray, cfg) -> InspectionResult:
        t0 = time.perf_counter()
        x, y, w, h = roi_to_pixels(cfg.roi, frame.shape)
        roi = frame[y:y + h, x:x + w]

        def result(judgement, score, reason, novelty=None, unknown=False):
            return InspectionResult(judgement, score, reason, (x, y, w, h),
                                    (time.perf_counter() - t0) * 1000, frame, time.time(), novelty, unknown)

        if not self.models.is_ready():
            return result("N/A", None, "Model belum dilatih: ambil sampel & latih di menu Training")
        prob, nov = self.models.predict(roi)
        if prob is None:
            return result("N/A", None, "Model tidak bisa dijalankan, latih ulang")
        if cfg.novelty_tolerance > 0 and nov is not None and nov > cfg.novelty_tolerance:
            return result("NG", prob, "Objek tidak dikenal (beda dari sampel training)", nov, True)
        ok = prob >= cfg.ok_threshold
        return result("OK" if ok else "NG", prob, "Ulir terdeteksi" if ok else "Ulir tidak terdeteksi", nov)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
COLOR_OK = (94, 197, 34)      # BGR
COLOR_NG = (68, 68, 239)
COLOR_NA = (11, 158, 245)


def draw_overlay(frame: np.ndarray, res: InspectionResult) -> np.ndarray:
    """Bingkai ROI berwarna sesuai hasil (hijau OK, merah NG, kuning N/A)."""
    vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    t = max(1, int(round(min(h, w) / 360)))
    x, y, rw, rh = res.roi_px
    color = {"OK": COLOR_OK, "NG": COLOR_NG}.get(res.judgement, COLOR_NA)
    cv2.rectangle(vis, (x, y), (x + rw, y + rh), color, t + 1, cv2.LINE_AA)
    L = max(10, int(min(rw, rh) * 0.08))
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + rw, y, -1, 1), (x, y + rh, 1, -1), (x + rw, y + rh, -1, -1)):
        cv2.line(vis, (px, py), (px + dx * L, py), color, t + 2, cv2.LINE_AA)
        cv2.line(vis, (px, py), (px, py + dy * L), color, t + 2, cv2.LINE_AA)
    return vis
