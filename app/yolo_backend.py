"""Backend YOLO26n-cls (deep learning) untuk klasifikasi ulir / tanpa ulir.

* Training memakai Ultralytics (butuh `pip install ultralytics`, paling cepat di PC).
* Inferensi memakai model ONNX lewat onnxruntime (ringan, jalan di Raspberry Pi 5 tanpa
  PyTorch). Bila onnxruntime tidak ada, dipakai OpenCV DNN.

Catatan lisensi: Ultralytics YOLO berlisensi AGPL-3.0. Untuk dipakai di produk yang
didistribusikan ke pihak lain, cek kebutuhan lisensi Enterprise Ultralytics.
"""
import shutil
import time

import cv2
import numpy as np

from . import storage
from . import config
from .config import CLASS_NG, CLASS_OK, WEIGHTS_DIR

YOLO_BASE = "yolo26n-cls.pt"
YOLO_ONNX = "yolo26n_cls_ulir.onnx"
YOLO_IMGSZ = 224


def training_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None


def preprocess(bgr: np.ndarray, size: int = YOLO_IMGSZ) -> np.ndarray:
    """Sama seperti Ultralytics: sisi terpendek -> size, crop tengah, RGB, 0..1, NCHW."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    r = cv2.resize(rgb, (nw, nh), interpolation=interp)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    crop = r[y0:y0 + size, x0:x0 + size]
    return np.ascontiguousarray((crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])


class YoloRunner:
    """Menjalankan model YOLO26n-cls hasil export ONNX."""

    def __init__(self, onnx_path, ok_index: int, imgsz: int = YOLO_IMGSZ):
        self.ok_index = ok_index
        self.imgsz = imgsz
        self.sess = None
        self.net = None
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2      # sisakan CPU untuk kamera & antarmuka
            self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                             providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name
        except ImportError:
            self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def prob_ok(self, roi_bgr: np.ndarray) -> float:
        x = preprocess(roi_bgr, self.imgsz)
        if self.sess is not None:
            out = self.sess.run(None, {self.input_name: x})[0][0]
        else:
            self.net.setInput(x)
            out = self.net.forward()[0]
        out = np.asarray(out, dtype=np.float64)
        if out.min() < 0 or abs(out.sum() - 1.0) > 1e-3:      # jaga-jaga bila keluaran belum softmax
            e = np.exp(out - out.max())
            out = e / e.sum()
        return float(out[self.ok_index])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _base_weights():
    """Bobot awal YOLO26n-cls (diunduh sekali ke folder models/)."""
    from ultralytics import YOLO
    path = WEIGHTS_DIR / YOLO_BASE
    if not path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        YOLO(YOLO_BASE)                         # unduh ke folder kerja
        for cand in (config.BASE_DIR / YOLO_BASE, config.models_dir() / YOLO_BASE):
            if cand.exists():
                shutil.move(str(cand), str(path))
                break
    return str(path) if path.exists() else YOLO_BASE


def _make_split_dir(root, items):
    shutil.rmtree(root, ignore_errors=True)
    for split, entries in items.items():
        for path, cls in entries:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, d / path.name)


def _train_once(data_dir, epochs, name, progress, should_stop, p_from, p_to, label):
    from ultralytics import YOLO

    from .model import TrainingError
    model = YOLO(_base_weights())

    def on_epoch_end(trainer):
        if should_stop():
            raise TrainingError("Training dibatalkan")
        e = trainer.epoch + 1
        progress(p_from + int((p_to - p_from) * e / epochs), f"{label}: epoch {e}/{epochs}")

    model.add_callback("on_train_epoch_end", on_epoch_end)
    model.train(data=str(data_dir), epochs=epochs, imgsz=YOLO_IMGSZ, batch=16, workers=0,
                project=str(config.models_dir() / "_yolo_work" / "runs"), name=name, exist_ok=True,
                plots=False, verbose=False, seed=0, patience=epochs, val=True)
    best = config.models_dir() / "_yolo_work" / "runs" / name / "weights" / "best.pt"
    return YOLO(str(best)) if best.exists() else model


def train_yolo(params, epochs=40, progress=lambda pct, msg: None, should_stop=lambda: False):
    """Latih YOLO26n-cls: (1) validasi 80/20 untuk mengukur akurasi & keyakinan,
    (2) latih model akhir dengan semua data, (3) export ONNX, (4) penolak objek asing."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    from .model import MIN_PER_CLASS, NoveltyDetector, TrainingError, augment
    from .vision import FEATURE_VERSION, analyze_roi

    t_start = time.time()
    progress(0, "Menyiapkan data YOLO…")
    entries = [(p, c) for c in (CLASS_OK, CLASS_NG) for p in storage.list_samples(c)]
    counts = {c: sum(1 for _, cc in entries if cc == c) for c in (CLASS_OK, CLASS_NG)}
    for c, n in counts.items():
        if n < MIN_PER_CLASS:
            raise TrainingError(f"Sampel kelas '{c}' baru {n}. Minimal {MIN_PER_CLASS} per kelas.")

    labels = [1 if c == CLASS_OK else 0 for _, c in entries]
    tr, va = train_test_split(entries, test_size=0.2, stratify=labels, random_state=0)
    work = config.models_dir() / "_yolo_work"

    # (1) validasi
    _make_split_dir(work / "data_val", {"train": tr, "val": va})
    m_val = _train_once(work / "data_val", epochs, "val", progress, should_stop, 2, 45, "Validasi")
    ok_idx = [k for k, v in m_val.names.items() if v == CLASS_OK][0]
    probs, ys, names = [], [], []
    for path, cls in va:
        img = storage.read_image(path)
        r = m_val.predict(img, imgsz=YOLO_IMGSZ, verbose=False)[0]
        probs.append(float(r.probs.data[ok_idx]))
        ys.append(1 if cls == CLASS_OK else 0)
        names.append(path.name)
    probs, ys = np.array(probs), np.array(ys)
    pred = (probs >= 0.5).astype(int)
    conf_true = np.where(ys == 1, probs, 1 - probs)
    bal = float(balanced_accuracy_score(ys, pred))
    confident = float((conf_true >= 0.8).mean())

    # (2) model akhir dengan semua data
    if should_stop():
        raise TrainingError("Training dibatalkan")
    _make_split_dir(work / "data_all", {"train": entries, "val": va})
    m_final = _train_once(work / "data_all", epochs, "final", progress, should_stop, 45, 88, "Model akhir")

    # (3) export ONNX untuk inferensi ringan
    progress(90, "Export ONNX…")
    onnx_src = m_final.export(format="onnx", imgsz=YOLO_IMGSZ, dynamic=False, simplify=True, verbose=False)
    shutil.copy2(onnx_src, config.models_dir() / YOLO_ONNX)
    final_ok_idx = [k for k, v in m_final.names.items() if v == CLASS_OK][0]
    # simpan bobot PyTorch final (untuk export ulang, mis. ke NCNN di Raspberry Pi), lalu
    # hapus folder kerja (salinan dataset & log training)
    best_pt = work / "runs" / "final" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, config.models_dir() / YOLO_ONNX.replace(".onnx", ".pt"))
    shutil.rmtree(work, ignore_errors=True)

    # (4) penolak objek tidak dikenal dari fitur citra (independen dari YOLO)
    progress(94, "Menyiapkan penolak objek tidak dikenal")
    X, groups = [], []
    rng = np.random.default_rng(0)
    for gi, (path, _) in enumerate(entries):
        img = storage.read_image(path)
        for k in range(5):
            a = analyze_roi(img if k == 0 else augment(img, rng, params.view_mode), params)
            if a.features is not None:
                X.append(a.features)
                groups.append(gi)
    novelty = NoveltyDetector().fit(np.array(X, dtype=np.float32), np.array(groups))

    cm = confusion_matrix(ys, pred, labels=[1, 0])
    meta = {
        "model_name": "YOLO26n-cls",
        "model_kind": "yolo",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": FEATURE_VERSION,
        "n_ok": counts[CLASS_OK],
        "n_ng": counts[CLASS_NG],
        "n_train_vectors": len(entries),
        "cv_accuracy": float((pred == ys).mean()),
        "cv_balanced_accuracy": bal,
        "cv_confident": confident,
        "view_mode": params.view_mode,
        "epochs": epochs,
    }
    report = {
        "meta": meta,
        "candidates": {"YOLO26n-cls": (bal, confident)},
        "confusion": cm.tolist(),
        "skipped": [],
        "misclassified": [f"{n} (aktual: {'ulir' if t == 1 else 'tanpa ulir'})"
                          for n, t, pr in zip(names, ys, pred) if t != pr],
        "n_splits": 0,
        "cv_desc": f"hold-out {len(va)} gambar (20%), lalu model akhir dilatih dengan semua data",
        "duration_s": time.time() - t_start,
    }
    progress(100, "Selesai")
    bundle = {"kind": "yolo", "pipeline": None, "onnx": YOLO_ONNX, "ok_index": final_ok_idx,
              "imgsz": YOLO_IMGSZ, "novelty": novelty, "meta": meta}
    return bundle, report
