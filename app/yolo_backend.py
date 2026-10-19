"""Backend YOLO26n-cls (deep learning) untuk klasifikasi ulir / tanpa ulir.

* Training memakai Ultralytics (butuh `pip install ultralytics`, paling cepat di PC).
* Inferensi memakai model ONNX lewat onnxruntime (ringan, jalan di Raspberry Pi 5 tanpa
  PyTorch). Bila onnxruntime tidak ada, dipakai OpenCV DNN.
* Model ONNX mengeluarkan dua hal sekaligus: probabilitas kelas dan *embedding* (vektor fitur
  internal YOLO). Embedding dipakai penolak objek asing: gambar yang embedding-nya jauh dari
  semua sampel training dinyatakan "objek tidak dikenal".

Catatan lisensi: Ultralytics YOLO berlisensi AGPL-3.0. Untuk dipakai di produk yang
didistribusikan ke pihak lain, cek kebutuhan lisensi Enterprise Ultralytics.
"""
import shutil
import time

import cv2
import numpy as np

from . import config, storage
from .config import CLASS_NG, CLASS_OK, WEIGHTS_DIR

YOLO_BASE = "yolo26n-cls.pt"
YOLO_ONNX = "yolo26n_cls_ulir.onnx"
YOLO_PT = "yolo26n_cls_ulir.pt"
YOLO_IMGSZ = 224
NOVELTY_VERSION = 3          # 3 = berbasis embedding YOLO


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
    """Menjalankan model YOLO26n-cls (ONNX): mengembalikan (P(ulir), embedding)."""

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

    def run(self, roi_bgr: np.ndarray):
        x = preprocess(roi_bgr, self.imgsz)
        if self.sess is not None:
            outs = self.sess.run(None, {self.input_name: x})
        else:
            self.net.setInput(x)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        probs = np.asarray(outs[0][0], dtype=np.float64)
        embed = np.asarray(outs[1][0], dtype=np.float64) if len(outs) > 1 else None
        return float(probs[self.ok_index]), embed


class EmbeddingNovelty:
    """Penolak objek asing: jarak kosinus embedding ke sampel training terdekat.

    Ambang = jarak terjauh sampel training saat diuji *leave-one-image-out* (seperti part baru
    yang belum pernah dilihat) dikali margin. Rasio > 1 berarti gambar lebih asing dari
    sampel asli mana pun."""

    K = 3
    MARGIN = 1.25

    def fit(self, embeds: np.ndarray, groups: np.ndarray):
        self.E = embeds.astype(np.float32)
        loo = []
        for g in np.unique(groups):
            others = self.E[groups != g]
            for e in self.E[(groups == g)][:1]:          # gambar asli (bukan variasi)
                d = np.sort(1.0 - others @ e)
                loo.append(d[min(self.K, len(d)) - 1])
        self.threshold = float(max(loo) * self.MARGIN) if loo else 1.0
        return self

    def ratio(self, embed: np.ndarray) -> float:
        d = np.sort(1.0 - self.E @ embed.astype(np.float32))
        return float(d[min(self.K, len(d)) - 1] / (self.threshold + 1e-9))


# ---------------------------------------------------------------------------
# Export ONNX (probabilitas + embedding)
# ---------------------------------------------------------------------------
def export_onnx(yolo, onnx_path):
    """Export model Ultralytics ke ONNX dengan dua keluaran: probs dan embedding."""
    import torch

    cm = yolo.model.float().eval()

    class ProbsAndEmbed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(*list(cm.model)[:-1])
            self.head = cm.model[-1]

        def forward(self, x):
            h = self.head
            e = h.pool(h.conv(self.backbone(x))).flatten(1)
            return h.linear(h.drop(e)).softmax(1), torch.nn.functional.normalize(e, dim=1)

    torch.onnx.export(ProbsAndEmbed().eval(), torch.zeros(1, 3, YOLO_IMGSZ, YOLO_IMGSZ), str(onnx_path),
                      input_names=["images"], output_names=["probs", "embed"], opset_version=17,
                      dynamo=False)


def fit_novelty(runner: YoloRunner, paths, variations: int = 4) -> EmbeddingNovelty:
    from .model import augment
    E, G = [], []
    rng = np.random.default_rng(0)
    for gi, path in enumerate(paths):
        img = storage.read_image(path)
        if img is None:
            continue
        for k in range(variations + 1):
            E.append(runner.run(img if k == 0 else augment(img, rng))[1])
            G.append(gi)
    return EmbeddingNovelty().fit(np.array(E), np.array(G))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _base_weights():
    """Bobot awal YOLO26n-cls (diunduh sekali ke folder weights/, dipakai semua profil)."""
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


def finalize_model(yolo, paths):
    """Export ONNX (probs + embedding) ke folder model profil aktif dan siapkan penolak
    objek asing. Mengembalikan isi bundle tanpa meta."""
    onnx_path = config.models_dir() / YOLO_ONNX
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(yolo, onnx_path)
    ok_index = [k for k, v in yolo.names.items() if v == CLASS_OK][0]
    runner = YoloRunner(onnx_path, ok_index)
    return {"kind": "yolo", "onnx": YOLO_ONNX, "ok_index": ok_index, "imgsz": YOLO_IMGSZ,
            "novelty": fit_novelty(runner, paths)}


def train_yolo(epochs=40, progress=lambda pct, msg: None, should_stop=lambda: False):
    """Latih YOLO26n-cls: (1) validasi 80/20 untuk mengukur akurasi & keyakinan,
    (2) latih model akhir dengan semua data, (3) export ONNX + penolak objek asing."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    from .model import TrainingError

    t_start = time.time()
    progress(0, "Menyiapkan data YOLO…")
    entries = [(p, c) for c in (CLASS_OK, CLASS_NG) for p in storage.list_samples(c)]
    counts = {c: sum(1 for _, cc in entries if cc == c) for c in (CLASS_OK, CLASS_NG)}
    labels = [1 if c == CLASS_OK else 0 for _, c in entries]
    tr, va = train_test_split(entries, test_size=0.2, stratify=labels, random_state=0)
    work = config.models_dir() / "_yolo_work"

    # (1) validasi
    _make_split_dir(work / "data_val", {"train": tr, "val": va})
    m_val = _train_once(work / "data_val", epochs, "val", progress, should_stop, 2, 45, "Validasi")
    ok_idx = [k for k, v in m_val.names.items() if v == CLASS_OK][0]
    probs, ys, names = [], [], []
    for path, cls in va:
        r = m_val.predict(storage.read_image(path), imgsz=YOLO_IMGSZ, verbose=False)[0]
        probs.append(float(r.probs.data[ok_idx]))
        ys.append(1 if cls == CLASS_OK else 0)
        names.append(path.name)
    probs, ys = np.array(probs), np.array(ys)
    pred = (probs >= 0.5).astype(int)
    conf_true = np.where(ys == 1, probs, 1 - probs)

    # (2) model akhir dengan semua data
    if should_stop():
        raise TrainingError("Training dibatalkan")
    _make_split_dir(work / "data_all", {"train": entries, "val": va})
    m_final = _train_once(work / "data_all", epochs, "final", progress, should_stop, 45, 88, "Model akhir")

    # (3) export ONNX + penolak objek asing; simpan bobot PyTorch (untuk export ulang, mis. NCNN)
    progress(90, "Export ONNX & menyiapkan penolak objek tidak dikenal…")
    bundle = finalize_model(m_final, [p for p, _ in entries])
    best_pt = work / "runs" / "final" / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, config.models_dir() / YOLO_PT)
    shutil.rmtree(work, ignore_errors=True)

    meta = {
        "model_name": "YOLO26n-cls",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "novelty_version": NOVELTY_VERSION,
        "n_ok": counts[CLASS_OK],
        "n_ng": counts[CLASS_NG],
        "cv_accuracy": float((pred == ys).mean()),
        "cv_balanced_accuracy": float(balanced_accuracy_score(ys, pred)),
        "cv_confident": float((conf_true >= 0.8).mean()),
        "epochs": epochs,
    }
    report = {
        "meta": meta,
        "confusion": confusion_matrix(ys, pred, labels=[1, 0]).tolist(),
        "misclassified": [f"{n} (aktual: {'ulir' if t == 1 else 'tanpa ulir'})"
                          for n, t, pr in zip(names, ys, pred) if t != pr],
        "cv_desc": f"hold-out {len(va)} gambar (20%), lalu model akhir dilatih dengan semua data",
        "duration_s": time.time() - t_start,
    }
    progress(100, "Selesai")
    bundle["meta"] = meta
    return bundle, report
