"""Model aktif profil (YOLO26n-cls + penolak objek asing): muat, simpan, prediksi, training."""
import threading

import cv2
import joblib
import numpy as np

from . import config, storage
from .config import CLASS_NG, CLASS_OK

MIN_PER_CLASS = 5


class TrainingError(Exception):
    pass


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Variasi ringan (posisi, skala, cahaya, blur, noise), dipakai saat menyiapkan penolak
    objek asing agar sedikit perbedaan posisi/cahaya tidak dianggap asing."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.92, 1.08))
    M[0, 2] += rng.uniform(-0.04, 0.04) * w
    M[1, 2] += rng.uniform(-0.04, 0.04) * h
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(2, 6), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


class ModelManager:
    """Model aktif profil ini, aman dipakai dari thread kamera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._yolo = None
        self._novelty = None
        self.meta = {}
        self.load_error = ""

    def is_ready(self) -> bool:
        return self._yolo is not None

    def load(self) -> bool:
        self.load_error = ""
        path = config.model_path()
        if not path.exists():
            return False
        try:
            bundle = joblib.load(path)
            if bundle.get("kind") != "yolo":
                self.load_error = "Model lama (bukan YOLO), latih ulang"
                return False
            self.set_bundle(bundle)
        except Exception as e:
            self.load_error = f"Model gagal dimuat: {e}"
            return False
        return True

    def set_bundle(self, bundle):
        from .yolo_backend import NOVELTY_VERSION, YoloRunner
        yolo = YoloRunner(config.models_dir() / bundle["onnx"], bundle["ok_index"], bundle.get("imgsz", 224))
        meta = bundle.get("meta", {})
        novelty = bundle.get("novelty") if meta.get("novelty_version") == NOVELTY_VERSION else None
        with self._lock:
            self._yolo, self._novelty, self.meta = yolo, novelty, meta

    @property
    def novelty_active(self) -> bool:
        return self._novelty is not None

    def save(self, bundle):
        path = config.model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        self.set_bundle(bundle)

    def clear(self):
        """Lepas model dari memori (file tidak dihapus), mis. saat ganti profil."""
        with self._lock:
            self._yolo, self._novelty, self.meta = None, None, {}

    def delete(self):
        self.clear()
        models = config.models_dir()
        for p in [models / config.MODEL_FILE, *models.glob("yolo26n_cls_ulir.*")]:
            p.unlink(missing_ok=True)

    def predict(self, roi):
        """Mengembalikan (P(ulir), rasio objek asing). Rasio > 1 = lebih asing dari sampel
        training mana pun; None bila penolak objek asing tidak aktif."""
        with self._lock:
            yolo, nov = self._yolo, self._novelty
        if yolo is None or roi is None or roi.size == 0:
            return None, None
        try:
            prob, embed = yolo.run(roi)
        except Exception:
            return None, None
        ratio = nov.ratio(embed) if (nov is not None and embed is not None) else None
        return prob, ratio

    def summary(self) -> str:
        if not self.is_ready():
            return self.load_error or "Belum ada model"
        return f"{self.meta.get('model_name', 'YOLO26n-cls')} · akurasi {self.meta.get('cv_balanced_accuracy', 0) * 100:.1f}%"


def train_model(cfg, progress=lambda pct, msg: None, should_stop=lambda: False):
    """Latih YOLO26n-cls dari dataset profil aktif. Mengembalikan (bundle, report)."""
    from .yolo_backend import train_yolo, training_available
    if not training_available():
        raise TrainingError("Library Ultralytics belum terpasang. Jalankan: pip install ultralytics "
                            "(disarankan training di PC, lalu salin folder profiles/ ke Raspberry Pi).")
    for cls in (CLASS_OK, CLASS_NG):
        n = len(storage.list_samples(cls))
        if n < MIN_PER_CLASS:
            raise TrainingError(f"Sampel kelas '{cls}' baru {n}. Minimal {MIN_PER_CLASS} per kelas "
                                "(disarankan ≥ 50).")
    return train_yolo(epochs=cfg.yolo_epochs, progress=progress, should_stop=should_stop)


def format_report(report) -> str:
    m = report["meta"]
    (tp, fn), (fp, tn) = report["confusion"]
    lines = [
        f"Model             : {m['model_name']} ({m.get('epochs', '?')} epoch)",
        f"Waktu training    : {report['duration_s'] / 60:.1f} menit",
        f"Sampel            : ulir={m['n_ok']}, tanpa ulir={m['n_ng']}",
        f"Validasi          : {report['cv_desc']}",
        "",
        f"Akurasi            : {m['cv_accuracy'] * 100:.1f}%",
        f"Balanced accuracy  : {m['cv_balanced_accuracy'] * 100:.1f}%",
        f"Benar & yakin ≥80% : {m.get('cv_confident', 0) * 100:.1f}% dari sampel",
        "",
        "Confusion matrix (gambar validasi):",
        "                     Prediksi ULIR  Prediksi TANPA",
        f"  Aktual ULIR        {tp:>13}  {fn:>14}",
        f"  Aktual TANPA ULIR  {fp:>13}  {tn:>14}",
        "",
        f"  ⚠ Tanpa ulir lolos sebagai OK (paling berbahaya): {fp}",
        f"  Ulir ditolak sebagai NG (false reject)          : {fn}",
        "",
        "Penolak objek tidak dikenal: aktif",
    ]
    if report["misclassified"]:
        lines += ["", "Salah klasifikasi saat validasi:"]
        lines += [f"  - {s}" for s in report["misclassified"][:15]]
    return "\n".join(lines)
