"""Penyimpanan dataset training, hasil inspeksi, dan log CSV."""
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .config import CLASS_NG, CLASS_OK

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def class_dir(cls: str, profile: str = None) -> Path:
    d = config.dataset_dir(profile) / cls
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_samples(cls: str, profile: str = None):
    d = class_dir(cls, profile)
    return sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT), key=lambda p: p.name)


def count_samples():
    return {c: len(list_samples(c)) for c in (CLASS_OK, CLASS_NG)}


def read_image(path) -> np.ndarray:
    # imdecode agar path dengan karakter non-ASCII tetap terbaca di Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))
    return ok


def _stamp():
    t = time.time()
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"_{int((t % 1) * 1000):03d}"


def save_sample(cls: str, img: np.ndarray) -> Path:
    path = class_dir(cls) / f"{cls}_{_stamp()}.png"
    write_image(path, img)
    return path


def delete_sample(path: Path):
    Path(path).unlink(missing_ok=True)


def move_sample(path: Path, new_cls: str) -> Path:
    path = Path(path)
    old_prefix = (CLASS_OK if new_cls == CLASS_NG else CLASS_NG) + "_"
    name = path.name
    if name.startswith(old_prefix):
        name = new_cls + "_" + name[len(old_prefix):]
    dest = class_dir(new_cls) / name
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}_{_stamp()}{dest.suffix}")
    shutil.move(str(path), str(dest))
    return dest


def move_to_profile(path: Path, cls: str, profile: str) -> Path:
    """Pindahkan sampel ke dataset profil lain (kelas sama)."""
    path = Path(path)
    dest = class_dir(cls, profile) / path.name
    if dest.exists():
        dest = dest.with_name(f"{dest.stem}_{_stamp()}{dest.suffix}")
    shutil.move(str(path), str(dest))
    return dest


def import_files(files, cls: str) -> int:
    n = 0
    for f in files:
        f = Path(f)
        if f.suffix.lower() not in IMAGE_EXT:
            continue
        dest = class_dir(cls) / f"{cls}_import_{f.stem}{f.suffix.lower()}"
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}_{_stamp()}{dest.suffix}")
        shutil.copy2(f, dest)
        n += 1
    return n


def save_inspection(res, annotated: np.ndarray = None) -> Path:
    """Simpan frame mentah + crop ROI (+ versi beranotasi) hasil inspeksi."""
    folder = config.captures_dir() / ("OK" if res.judgement == "OK" else "NG") / time.strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{res.judgement}_{_stamp()}"
    frame_path = folder / f"{stem}_frame.jpg"
    write_image(frame_path, res.frame)
    x, y, w, h = res.roi_px
    write_image(folder / f"{stem}_roi.png", res.frame[y:y + h, x:x + w])
    if annotated is not None:
        write_image(folder / f"{stem}_hasil.jpg", annotated)
    return frame_path


def append_log(row: dict):
    logs = config.logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"inspeksi_{time.strftime('%Y%m%d')}.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
