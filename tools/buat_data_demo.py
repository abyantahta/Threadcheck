"""Membuat data demo sintetis (lubang berulir vs tanpa ulir) untuk mencoba aplikasi
tanpa kamera.

Hasil (per mode):
  demo_data/miring/ulir/*.png        - ROI dinding lubang berulir, sudut miring, hanya 2-3 garis
                                       ulir (pelat tipis, seperti part asli)
  demo_data/miring/tanpa_ulir/*.png  - ROI dinding lubang polos, sudut miring
  demo_data/miring/demo_video.mp4    - video simulasi lini produksi (bergantian OK/NG)
  demo_data/miring_banyak/...        - sudut miring dengan banyak garis ulir (lubang dalam)
  demo_data/tegak/...                - kamera tegak lurus (top view)

Jalankan:  python tools/buat_data_demo.py [miring|miring_banyak|tegak|semua] [jumlah_per_kelas]
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo_data"
DEFAULT_ROI = (0.30, 0.20, 0.40, 0.60)   # sama dengan ROI default aplikasi
ROI_FEW_LINES = (0.30, 0.12, 0.40, 0.38)  # ROI dinding pelat tipis (2-3 garis ulir)


def _surface(h, w, rng, base=None):
    base = rng.uniform(120, 165) if base is None else base
    tex = rng.normal(0, 1, (h, w)).astype(np.float32)
    tex = cv2.GaussianBlur(tex, (0, 0), sigmaX=6, sigmaY=0.6)   # tekstur bubut/brushed
    tex = tex / (tex.std() + 1e-6) * rng.uniform(4, 9)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / w - 0.5) * rng.uniform(-30, 30) + (yy / h - 0.5) * rng.uniform(-30, 30)
    return base + tex + grad


def _finish(img, rng):
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.4))
    img = img + rng.normal(0, rng.uniform(2, 5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Kamera tegak lurus (top view)
# ---------------------------------------------------------------------------
def render_top(h, w, cx, cy, r, threaded, rng):
    img = _surface(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    rho = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    dark = rng.uniform(15, 40)
    inside = rho < r
    wall = dark + (rng.uniform(25, 60)) * np.clip((rho / r - 0.45) / 0.55, 0, 1) ** 2
    if threaded:
        pitch = r * rng.uniform(0.10, 0.17)
        spiral = rho + pitch * (theta + rng.uniform(0, 2 * np.pi)) / (2 * np.pi)
        crest = np.clip(np.cos(2 * np.pi * spiral / pitch), 0, 1) ** rng.uniform(1.5, 3.0)
        amp = rng.uniform(55, 110) * np.clip((rho / r - 0.40) / 0.6, 0, 1)
        light = 0.65 + 0.35 * np.cos(theta - rng.uniform(0, 2 * np.pi))
        wall = wall + amp * crest * light
    else:
        marks = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5)
        wall = wall + marks * rng.uniform(3, 8)
    img[inside] = wall[inside]
    if rng.random() < 0.7:
        ring = (rho >= r) & (rho < r + r * rng.uniform(0.04, 0.10))
        img[ring] = np.clip(img[ring] + rng.uniform(30, 70), 0, 255)
    return _finish(img, rng)


def frame_top(h, w, threaded, rng):
    cx, cy = w / 2 + rng.uniform(-0.012, 0.012) * w, h / 2 + rng.uniform(-0.02, 0.02) * h
    r = h * rng.uniform(0.15, 0.17)
    return render_top(h, w, cx, cy, r, threaded, rng)


# ---------------------------------------------------------------------------
# Kamera sudut miring: melihat dinding seberang di dalam lubang
# ---------------------------------------------------------------------------
def frame_oblique(h, w, threaded, rng, few_lines=False, pitch_frac=None):
    """Mulut lubang tampak sebagai elips; dinding seberang terlihat di dalamnya.

    Titik dinding pada kedalaman z tampak pada busur elips yang sama dengan bibir
    lubang tetapi tergeser ke bawah, sehingga garis ulir menjadi busur-busur sejajar.
    """
    img = _surface(h, w, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2 + rng.uniform(-0.015, 0.015) * w
    a = w * rng.uniform(0.40, 0.44)                  # setengah sumbu horizontal bibir lubang
    b = h * rng.uniform(0.40, 0.46)                  # setengah sumbu vertikal (elips karena miring)
    y_top = h * rng.uniform(0.13, 0.17)              # bibir lubang sisi jauh
    cy = y_top + b
    u = (xx - cx) / a
    inside = (u ** 2 + ((yy - cy) / b) ** 2) < 1.0
    arc = cy - b * np.sqrt(np.clip(1 - u ** 2, 0, 1))      # busur bibir pada kolom x
    z = yy - arc                                            # "kedalaman" di dinding (px)

    depth_fade = np.exp(-np.clip(z, 0, None) / (h * rng.uniform(0.35, 0.5)))
    side = np.sqrt(np.clip(1 - u ** 2, 0, 1))               # dinding samping lebih gelap
    wall = rng.uniform(25, 45) + rng.uniform(50, 80) * depth_fade * (0.5 + 0.5 * side)

    if pitch_frac is not None:
        pitch = h * pitch_frac * rng.uniform(0.97, 1.03)     # setup produksi: pitch nyaris tetap
    elif few_lines:
        pitch = h * rng.uniform(0.08, 0.11)
    else:
        pitch = h * rng.uniform(0.035, 0.055)
    if threaded:
        phase = rng.uniform(0, 2 * np.pi)
        tilt = rng.uniform(-0.03, 0.03) * (xx - cx)          # heliks: garis sedikit miring
        t = 2 * np.pi * (z + tilt) / pitch + phase
        crest = np.clip(np.cos(t), 0, 1) ** rng.uniform(2.0, 4.0)
        flank = np.clip(np.cos(t + rng.uniform(1.8, 2.6)), 0, 1) ** 6   # pantulan sisi ulir
        wall = wall + depth_fade * (0.4 + 0.6 * side) * (rng.uniform(60, 110) * crest
                                                         + rng.uniform(15, 40) * flank)
    else:
        # dinding bor/reamer: bekas pahat jauh lebih rapat dari pitch ulir (feed bor
        # ±0,05-0,1 mm vs pitch M4 0,7 mm), amplitudo kecil, ditambah gelombang tidak teratur
        for _ in range(3):
            period = pitch * rng.uniform(0.07, 0.18)
            wall = wall + rng.uniform(1.5, 4) * np.cos(2 * np.pi * z / period + rng.uniform(0, 6.28))
        wall = wall + rng.uniform(2, 6) * np.cos(2 * np.pi * z / (pitch * rng.uniform(2.5, 5)))
        wall = wall + cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.5) * 4

    # pantulan cahaya vertikal (seperti pada foto asli)
    sx = cx + rng.uniform(-0.25, 0.25) * a
    streak = np.exp(-((xx - sx) / (w * rng.uniform(0.006, 0.012))) ** 2)
    if threaded:
        streak = streak * (0.3 + 0.7 * crest)               # terpecah oleh puncak ulir
    wall = wall + rng.uniform(60, 130) * streak * depth_fade

    if few_lines:
        # pelat tipis: setelah ±2-3 garis ulir, pandangan keluar dari bawah lubang (gelap)
        thickness = pitch * rng.uniform(2.2, 3.4)
        exit_mask = 1.0 / (1.0 + np.exp(-np.clip((z - thickness) / 2.0, -50, 50)))
        wall = wall * (1 - exit_mask) + rng.uniform(8, 25) * exit_mask

    img[inside] = wall[inside]
    # chamfer terang di bibir lubang
    rim = np.abs(np.sqrt(u ** 2 + ((yy - cy) / b) ** 2) - 1.0) < rng.uniform(0.012, 0.025)
    img[rim] = np.clip(img[rim] + rng.uniform(40, 80), 0, 255)
    return _finish(img, rng)


# ---------------------------------------------------------------------------
def _crop_roi(img, roi=DEFAULT_ROI):
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    return img[int(y * h):int((y + rh) * h), int(x * w):int((x + rw) * w)]


def _renderer(mode):
    if mode == "miring":
        return lambda h, w, t, rng: frame_oblique(h, w, t, rng, few_lines=True), ROI_FEW_LINES
    if mode == "miring_banyak":
        return frame_oblique, DEFAULT_ROI
    return frame_top, DEFAULT_ROI


def make_roi_samples(mode, n_per_class=40, seed=1):
    rng = np.random.default_rng(seed)
    render, roi = _renderer(mode)
    for cls, threaded in (("ulir", True), ("tanpa_ulir", False)):
        d = OUT / mode / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            frame = render(360, 640, threaded, rng)
            cv2.imwrite(str(d / f"demo_{cls}_{i:03d}.png"), _crop_roi(frame, roi))


def make_video(mode, seconds_per_part=2.5, parts=8, fps=15, seed=7):
    rng = np.random.default_rng(seed)
    w, h = 1280, 720
    render, _ = _renderer(mode)
    path = OUT / mode / "demo_video.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k in range(parts):
        threaded = (k % 2 == 0) if k < 4 else bool(rng.random() < 0.6)
        base = render(h, w, threaded, rng).astype(np.float32)
        for _ in range(int(seconds_per_part * fps)):
            vw.write(np.clip(base + rng.normal(0, 2.5, base.shape), 0, 255).astype(np.uint8))
    vw.release()
    return path


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "semua"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    modes = ["miring", "miring_banyak", "tegak"] if which == "semua" else [which]
    for m in modes:
        make_roi_samples(m, n)
        p = make_video(m)
        print(f"[{m}] {n} gambar ulir + {n} tanpa ulir, video: {p.relative_to(ROOT)}")
