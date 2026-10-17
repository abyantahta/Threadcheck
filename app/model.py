"""Manajemen model machine learning: training, evaluasi, simpan/muat, prediksi."""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import cv2
import joblib
import numpy as np

from . import storage
from . import config
from .config import CLASS_LABEL, CLASS_NG, CLASS_OK
from .vision import FEATURE_VERSION, DetectParams, analyze_roi

MIN_PER_CLASS = 5

MODEL_KINDS = {
    "auto": "Otomatis (pilih terbaik)",
    "svm": "SVM (RBF)",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting",
    "logreg": "Logistic Regression",
    "yolo": "YOLO26n-cls (deep learning)",
}


class NoveltyDetector:
    """Penolak "objek tidak dikenal": mengukur seberapa jauh gambar baru dari data training.

    Fitur di-standarisasi dan diringkas dengan PCA, lalu dihitung jarak ke tetangga
    terdekat di data training. Ambang diambil dari jarak antar gambar training yang
    berbeda (bukan hasil augmentasi dari gambar yang sama), sehingga gambar yang mirip
    sampel training tetap diterima, sedangkan wajah, tangan, latar kosong, dll. ditolak.
    """

    K = 3

    def fit(self, X, groups):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=int(min(20, Xs.shape[0] - 1, Xs.shape[1])), random_state=0).fit(Xs)
        Z = self.pca.transform(Xs)
        self.nn = NearestNeighbors().fit(Z)
        n_query = min(len(Z), 64)
        dist, idx = self.nn.kneighbors(Z, n_neighbors=n_query)
        ref = []
        for i in range(len(Z)):
            other = dist[i][groups[idx[i]] != groups[i]]     # tetangga dari gambar lain
            if len(other) >= self.K:
                ref.append(other[self.K - 1])
        ref = np.array(ref) if ref else dist[:, -1]
        self.threshold = float(np.quantile(ref, 0.99) * 1.5)
        self._prepare_fast()
        return self

    def _prepare_fast(self):
        scale = np.where(self.scaler.scale_ == 0, 1.0, self.scaler.scale_)
        self._mu = (self.scaler.mean_).astype(np.float64)
        self._inv = (1.0 / scale).astype(np.float64)
        self._pmean = self.pca.mean_.astype(np.float64)
        self._comp = self.pca.components_.astype(np.float64)
        self._Z = self.nn._fit_X.astype(np.float64)

    def ratio(self, features) -> float:
        if not hasattr(self, "_Z"):
            self._prepare_fast()
        x = (features.astype(np.float64) - self._mu) * self._inv
        z = (x - self._pmean) @ self._comp.T
        d = np.sqrt(((self._Z - z) ** 2).sum(axis=1))
        k = min(self.K, len(d)) - 1
        return float(np.partition(d, k)[k] / (self.threshold + 1e-9))


class ModelManager:
    """Menyimpan model aktif secara thread-safe (dipakai oleh thread kamera)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = None
        self._yolo = None
        self._novelty = None
        self.meta = {}

    def is_ready(self) -> bool:
        return self._pipeline is not None or self._yolo is not None

    def load(self) -> bool:
        path = config.model_path()
        if not path.exists():
            return False
        try:
            bundle = joblib.load(path)
        except Exception:
            return False
        if bundle.get("meta", {}).get("feature_version") != FEATURE_VERSION:
            return False
        self.set_bundle(bundle)
        return True

    def set_bundle(self, bundle):
        yolo = None
        if bundle.get("kind") == "yolo":
            from .yolo_backend import YoloRunner
            yolo = YoloRunner(config.models_dir() / bundle["onnx"], bundle["ok_index"], bundle.get("imgsz", 224))
        else:
            _single_thread(bundle["pipeline"])
        with self._lock:
            self._yolo = yolo
            self._pipeline = bundle["pipeline"]
            self._novelty = bundle.get("novelty")
            self.meta = bundle.get("meta", {})

    def save(self, bundle):
        path = config.model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)
        self.set_bundle(bundle)

    def clear(self):
        """Lepas model dari memori (file tidak dihapus), mis. saat ganti profil."""
        with self._lock:
            self._pipeline = None
            self._yolo = None
            self._novelty = None
            self.meta = {}

    def delete(self):
        self.clear()
        path = config.model_path()
        if path.exists():
            path.unlink()

    def predict_proba(self, features, roi=None):
        with self._lock:
            pipe, yolo = self._pipeline, self._yolo
        if yolo is not None:
            if roi is None:
                return None
            try:
                return yolo.prob_ok(roi)
            except Exception:
                return None
        if pipe is None or features is None:
            return None
        try:
            proba = pipe.predict_proba(features.reshape(1, -1))[0]
            classes = list(pipe.classes_)
            return float(proba[classes.index(1)])
        except Exception:
            return None

    def novelty_ratio(self, features):
        """> 1 berarti gambar jauh berbeda dari semua data training (objek tidak dikenal)."""
        with self._lock:
            nov = self._novelty
        if nov is None or features is None:
            return None
        try:
            return nov.ratio(features)
        except Exception:
            return None

    def summary(self) -> str:
        if not self.is_ready():
            return "Belum ada model"
        m = self.meta
        return f"{m.get('model_name', '?')} · akurasi CV {m.get('cv_balanced_accuracy', 0) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Augmentasi
# ---------------------------------------------------------------------------
def augment(img: np.ndarray, rng: np.random.Generator, view_mode: str = "oblique") -> np.ndarray:
    h, w = img.shape[:2]
    if view_mode == "top":
        angle = rng.uniform(0, 360)        # lubang dilihat dari atas: rotasi bebas
    else:
        angle = rng.uniform(-6, 6)         # sudut miring: garis ulir tetap kira-kira mendatar
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)
    scale = rng.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += rng.uniform(-0.04, 0.04) * w
    M[1, 2] += rng.uniform(-0.04, 0.04) * h
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.75, 1.25), beta=rng.uniform(-25, 25))
    gamma = rng.uniform(0.8, 1.25)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        noise = rng.normal(0, rng.uniform(2, 7), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Model kandidat
# ---------------------------------------------------------------------------
def _make_model(kind: str, for_cv: bool = False):
    """for_cv=True: versi ringan untuk validasi silang (tanpa kalibrasi probabilitas)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale", probability=not for_cv,
                                                   class_weight="balanced", random_state=0))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=120 if for_cv else 250, class_weight="balanced_subsample",
                                      min_samples_leaf=1, n_jobs=-1, random_state=0)
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=40 if for_cv else 150, learning_rate=0.1, min_samples_leaf=5,
                                              max_leaf_nodes=15,
                                              class_weight="balanced", random_state=0)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000,
                                                                  class_weight="balanced"))
    raise ValueError(kind)


def _single_thread(model):
    """Prediksi satu gambar per frame: paralelisme justru menambah overhead besar."""
    steps = model.steps if hasattr(model, "steps") else [(None, model)]
    for _, est in steps:
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1
    return model


class TrainingError(Exception):
    pass


def train_model(params: DetectParams, model_kind="auto", use_augment=True, aug_per_image=8,
                progress=lambda pct, msg: None, should_stop=lambda: False):
    """Latih model dari folder dataset. Mengembalikan (bundle, report)."""
    if model_kind == "yolo":
        from .yolo_backend import train_yolo, training_available
        if not training_available():
            raise TrainingError("Library Ultralytics belum terpasang. Jalankan: pip install ultralytics "
                                "(disarankan training di PC, lalu salin folder models/ ke Raspberry Pi).")
        return train_yolo(params, epochs=40, progress=progress, should_stop=should_stop)
    progress(0, "Menyiapkan library & dataset…")
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    t_start = time.time()
    files = {c: storage.list_samples(c) for c in (CLASS_OK, CLASS_NG)}
    for c, lst in files.items():
        if len(lst) < MIN_PER_CLASS:
            raise TrainingError(f"Sampel kelas '{c}' baru {len(lst)}. Minimal {MIN_PER_CLASS} per kelas "
                                f"(disarankan ≥ 30).")

    n_aug = aug_per_image if use_augment else 0
    pairs = [(path, CLASS_LABEL[cls]) for cls, lst in files.items() for path in lst]
    jobs = [(i, path, label) for i, (path, label) in enumerate(pairs)]
    total = len(jobs)

    def extract(job):
        i, path, _ = job
        if should_stop():
            return None
        img = storage.read_image(path)
        a = analyze_roi(img, params) if img is not None else None
        if a is None or a.features is None:
            return None
        feats = [a.features]
        rng = np.random.default_rng(1000 + i)
        for _ in range(n_aug):
            aa = analyze_roi(augment(img, rng, params.view_mode), params)
            if aa.features is not None:
                feats.append(aa.features)
        return feats

    # Paralel per gambar: OpenCV & NumPy melepas GIL, jadi thread memberi percepatan nyata.
    # Thread internal OpenCV dimatikan sementara agar tidak terjadi oversubscription CPU.
    outputs = [None] * total
    progress(1, f"Ekstraksi fitur 0/{total}")
    cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
            futures = {pool.submit(extract, job): job[0] for job in jobs}
            for done, fut in enumerate(as_completed(futures), 1):
                outputs[futures[fut]] = fut.result()
                progress(1 + int(done / total * 59), f"Ekstraksi fitur {done}/{total}")
    finally:
        cv2.setNumThreads(cv_threads)
    if should_stop():
        raise TrainingError("Training dibatalkan")

    X, y, groups, is_orig, src = [], [], [], [], []
    skipped = []
    for (i, path, label), feats in zip(jobs, outputs):
        if feats is None:
            skipped.append(path.name)
            continue
        for k, f in enumerate(feats):
            X.append(f); y.append(label); groups.append(i); is_orig.append(k == 0); src.append(path.name)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    is_orig = np.asarray(is_orig)
    src = np.asarray(src)
    counts = {lbl: len(set(groups[(y == lbl) & is_orig])) for lbl in (0, 1)}
    if min(counts.values()) < MIN_PER_CLASS:
        raise TrainingError(f"Setelah membuang sampel tanpa lubang terdeteksi, sampel valid: "
                            f"ulir={counts[1]}, tanpa ulir={counts[0]}. Minimal {MIN_PER_CLASS} per kelas.")

    n_splits = int(min(5, min(counts.values())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    kinds = ["svm", "rf", "hgb", "logreg"] if model_kind == "auto" else [model_kind]

    results = {}
    for i, kind in enumerate(kinds):
        if should_stop():
            raise TrainingError("Training dibatalkan")
        progress(60 + int(i / len(kinds) * 30), f"Validasi silang: {MODEL_KINDS[kind]}")
        # SVM perlu kalibrasi probabilitas agar keyakinannya bisa dinilai
        est = _make_model(kind, for_cv=(kind != "svm"))
        proba = cross_val_predict(est, X, y, groups=groups, cv=cv, n_jobs=1, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        # evaluasi hanya pada gambar asli (bukan hasil augmentasi)
        yo_, po_ = y[is_orig], proba[is_orig]
        bal = balanced_accuracy_score(yo_, pred[is_orig])
        conf_true = np.where(yo_ == 1, po_, 1 - po_)      # keyakinan pada jawaban yang benar
        confident = float((conf_true >= 0.8).mean())
        # skor pemilihan: akurasi dulu, keyakinan sebagai penentu bila akurasi mirip
        results[kind] = {"bal": bal, "conf": float(conf_true.mean()), "confident": confident,
                         "pred": pred, "score": bal + 0.5 * confident}

    best_kind = max(results, key=lambda k: results[k]["score"])
    best_bal, best_pred = results[best_kind]["bal"], results[best_kind]["pred"]
    progress(92, f"Melatih model akhir: {MODEL_KINDS[best_kind]}")
    final = _make_model(best_kind)
    final.fit(X, y)
    _single_thread(final)
    progress(97, "Menyiapkan penolak objek tidak dikenal")
    novelty = NoveltyDetector().fit(X, groups)
    orig_ratio = np.array([novelty.ratio(x) for x in X[is_orig]])

    yo, po = y[is_orig], best_pred[is_orig]
    cm = confusion_matrix(yo, po, labels=[1, 0])   # baris: aktual ulir, aktual tanpa ulir
    acc = float((yo == po).mean())
    wrong = [f"{s} (aktual: {'ulir' if t == 1 else 'tanpa ulir'})"
             for s, t, pr in zip(src[is_orig], yo, po) if t != pr]

    meta = {
        "model_name": MODEL_KINDS[best_kind],
        "model_kind": best_kind,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": FEATURE_VERSION,
        "n_ok": counts[1],
        "n_ng": counts[0],
        "n_train_vectors": int(len(y)),
        "cv_accuracy": acc,
        "cv_balanced_accuracy": float(best_bal),
        "cv_confident": results[best_kind]["confident"],
        "detect_params": asdict(params),
        "view_mode": params.view_mode,
    }
    report = {
        "meta": meta,
        "candidates": {MODEL_KINDS[k]: (float(v["bal"]), v["confident"]) for k, v in results.items()},
        "confusion": cm.tolist(),
        "skipped": skipped,
        "misclassified": wrong,
        "n_splits": n_splits,
        "duration_s": time.time() - t_start,
    }
    progress(100, "Selesai")
    report["novelty_flagged"] = int((orig_ratio > 1.0).sum())
    return {"pipeline": final, "meta": meta, "novelty": novelty}, report


def format_report(report) -> str:
    m = report["meta"]
    cm = report["confusion"]
    tp, fn = cm[0]
    fp, tn = cm[1]
    lines = [
        f"Model terpilih    : {m['model_name']}",
        f"Waktu training    : {report['duration_s']:.1f} detik",
        f"Sampel valid      : ulir={m['n_ok']}, tanpa ulir={m['n_ng']}"
        + ("" if m.get("model_kind") == "yolo" else f" (total vektor + augmentasi: {m['n_train_vectors']})"),
        f"Validasi          : {report.get('cv_desc') or str(report['n_splits']) + '-fold (dikelompokkan per gambar)'}",
        "",
        "Perbandingan model    akurasi   benar & yakin >= 80%",
    ]
    for name, (bal, conf) in sorted(report["candidates"].items(), key=lambda kv: -(kv[1][0] + 0.5 * kv[1][1])):
        mark = "  ◀ dipilih" if name == m["model_name"] else ""
        lines.append(f"  {name:<20} {bal * 100:6.1f}%   {conf * 100:6.1f}%{mark}")
    lines += [
        "",
        f"Akurasi            : {m['cv_accuracy'] * 100:.1f}%",
        f"Balanced accuracy  : {m['cv_balanced_accuracy'] * 100:.1f}%",
        f"Benar & yakin ≥80% : {m.get('cv_confident', 0) * 100:.1f}% dari sampel",
        "",
        "Confusion matrix (gambar asli):",
        "                     Prediksi ULIR  Prediksi TANPA",
        f"  Aktual ULIR        {tp:>13}  {fn:>14}",
        f"  Aktual TANPA ULIR  {fp:>13}  {tn:>14}",
        "",
        f"  ⚠ Tanpa ulir lolos sebagai OK (paling berbahaya): {fp}",
        f"  Ulir ditolak sebagai NG (false reject)          : {fn}",
    ]
    if "novelty_flagged" in report:
        lines += ["", f"Penolak objek tidak dikenal aktif (sampel training yang akan ditolak: "
                      f"{report['novelty_flagged']})"]
    if report["skipped"]:
        lines += ["", f"Dilewati (lubang tidak terdeteksi): {len(report['skipped'])}"]
        lines += [f"  - {s}" for s in report["skipped"][:15]]
    if report["misclassified"]:
        lines += ["", "Salah klasifikasi saat validasi:"]
        lines += [f"  - {s}" for s in report["misclassified"][:15]]
    return "\n".join(lines)
