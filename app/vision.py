"""Inti computer vision: ekstraksi fitur ulir dan keputusan OK/NG.

Dua mode sudut pandang kamera:

* **Sudut miring (oblique)**: kamera melihat ke dalam lubang dari samping. Ulir di
  dinding seberang tampak sebagai tumpukan garis lengkung sejajar. ROI diluruskan
  (sudut garis dicari otomatis), lalu dibagi menjadi strip vertikal; profil intensitas
  tiap strip diukur periodisitasnya (FFT, autokorelasi, kesepakatan antar strip).
  Median antar strip membuatnya tahan terhadap pantulan cahaya (highlight).

* **Tegak lurus (top view)**: kamera tegak lurus lubang. Lubang dideteksi sebagai
  lingkaran, area cincin di sekitarnya di-"buka" ke koordinat polar sehingga cincin
  ulir menjadi garis lurus berulang sepanjang sumbu radius.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

FEATURE_VERSION = 1
POLAR_ANG = 256   # jumlah sampel sudut
POLAR_RAD = 64    # jumlah sampel radius
N_SECTORS = 8

VIEW_MODES = {
    "oblique": "Sudut miring (melihat dinding lubang)",
    "top": "Tegak lurus / top view (lubang tampak lingkaran)",
}
OBL_N = 192        # ROI miring dinormalisasi ke OBL_N x OBL_N
OBL_STRIPS = 12    # jumlah strip vertikal
OBL_FFT = 256
OBL_KMIN, OBL_KMAX = 8, 100   # rentang frekuensi garis ulir yang dicari (≥ ±5 garis terlihat)
OBL_FLOOR = 0.004             # batas bawah derau spektrum (citra terlalu halus tidak memicu)
OBL_TOPHAT = 0.14             # lebar top-hat relatif tinggi ROI: hanya garis terang sempit yang lolos
OBL_PROFILE_BINS = 24         # bentuk profil garis (fitur ML)


@dataclass(frozen=True)
class DetectParams:
    r_min_pct: float = 12.0
    r_max_pct: float = 48.0
    hough_sensitivity: int = 30
    band_inner: float = 0.55
    band_outer: float = 1.25
    proc_max_side: int = 640
    view_mode: str = "oblique"
    line_angle: float = 0.0        # perkiraan arah garis ulir (derajat, 0 = mendatar)
    angle_search: float = 20.0     # rentang pencarian sudut otomatis (±derajat, 0 = mati)
    line_contrast: float = 5.0     # kontras minimum garis ulir (level abu-abu) untuk metode Rule
    expected_pitch: float = 0.0    # jarak garis hasil kalibrasi (px ROI asli), 0 = tanpa kalibrasi
    pitch_tol: float = 0.25        # toleransi jarak garis (±fraksi)

    @classmethod
    def from_config(cls, cfg) -> "DetectParams":
        return cls(cfg.r_min_pct, cfg.r_max_pct, cfg.hough_sensitivity,
                   cfg.band_inner, cfg.band_outer, cfg.proc_max_side,
                   cfg.view_mode, cfg.line_angle, cfg.angle_search, cfg.line_contrast,
                   cfg.rule_pitch_px, cfg.rule_pitch_tol)


@dataclass
class Analysis:
    features: Optional[np.ndarray]
    rule_score: Optional[float]
    circle: Optional[tuple]          # (cx, cy, r) dalam koordinat ROI asli (mode top)
    band: Optional[np.ndarray]       # citra analisis (polar / ROI yang diluruskan), uint8
    info: dict = field(default_factory=dict)


@dataclass
class InspectionResult:
    judgement: str                   # OK | NG | NO_HOLE | N/A
    score: Optional[float]           # probabilitas/skor "berulir" 0..1
    method: str                      # ML | Rule | -
    reason: str
    circle: Optional[tuple]          # (cx, cy, r) dalam koordinat frame
    roi_px: tuple                    # (x, y, w, h) dalam koordinat frame
    band: Optional[np.ndarray]
    rule_score: Optional[float]
    proc_ms: float
    frame: Optional[np.ndarray] = None   # frame mentah (BGR) untuk disimpan
    timestamp: float = 0.0
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


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


def _prepare(gray: np.ndarray, max_side: int):
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray, scale


# ---------------------------------------------------------------------------
# Deteksi lubang
# ---------------------------------------------------------------------------
def _radial_edge_profile(gray, cx, cy, r_lo, r_hi):
    """Rata-rata |turunan radial| intensitas untuk tiap radius (r_lo..r_hi)."""
    r_hi = max(r_hi, r_lo + 4)
    n_rad = int(r_hi) + 1
    pad = int(r_hi) + 2
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    polar = cv2.warpPolar(g.astype(np.float32), (n_rad, 180), (cx + pad, cy + pad), r_hi,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
    prof = polar.mean(axis=0)
    prof = cv2.GaussianBlur(prof.reshape(1, -1), (0, 0), 1.0).ravel()
    deriv = np.abs(np.gradient(prof))
    radii = np.arange(n_rad, dtype=np.float32)
    mask = (radii >= r_lo) & (radii <= r_hi)
    return radii[mask], deriv[mask]


def _refine_radius(gray, cx, cy, r, r_min, r_max):
    """Pilih radius dengan transisi terang-gelap terkuat di sekitar estimasi awal.

    Hough bisa saja mengunci salah satu cincin ulir; tepi lubang (lubang ke
    permukaan benda) umumnya kontras paling kuat.
    """
    lo = max(r_min, 0.7 * r)
    hi = min(r_max, 1.3 * r)
    if hi - lo < 3:
        return r
    radii, deriv = _radial_edge_profile(gray, cx, cy, lo, hi)
    if len(deriv) == 0:
        return r
    return float(radii[int(np.argmax(deriv))])


def _detect_by_contour(gray, r_min, r_max):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None
    for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):   # lubang gelap, lalu lubang terang
        _, bw = cv2.threshold(blur, 0, 255, flag | cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < np.pi * r_min ** 2 * 0.6:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            if not (r_min <= rad <= r_max):
                continue
            circularity = area / (np.pi * rad ** 2 + 1e-6)
            if circularity < 0.6:
                continue
            dist = np.hypot(cx - w / 2, cy - h / 2) / max(w, h)
            score = circularity - dist
            if best is None or score > best[0]:
                best = (score, (cx, cy, rad))
        if best is not None:
            return best[1]
    return None


def detect_hole(gray: np.ndarray, p: DetectParams):
    """Kembalikan (cx, cy, r) lubang pada citra gray, atau None."""
    h, w = gray.shape
    m = min(h, w)
    r_min = max(4, int(m * p.r_min_pct / 100.0))
    r_max = max(r_min + 2, int(m * p.r_max_pct / 100.0))

    g = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=m / 3.0,
                               param1=100, param2=max(5, int(p.hough_sensitivity)),
                               minRadius=r_min, maxRadius=r_max)
    found = None
    if circles is not None and len(circles[0]):
        cands = circles[0][:5]
        d = np.hypot(cands[:, 0] - w / 2, cands[:, 1] - h / 2)
        cx, cy, r = cands[int(np.argmin(d))]
        found = (float(cx), float(cy), float(r))
    else:
        found = _detect_by_contour(gray, r_min, r_max)
    if found is None:
        return None
    cx, cy, r = found
    r = _refine_radius(gray, cx, cy, r, r_min, r_max)
    return cx, cy, r


# ---------------------------------------------------------------------------
# Polar unwrap & fitur
# ---------------------------------------------------------------------------
def polar_band(gray: np.ndarray, circle, p: DetectParams) -> np.ndarray:
    cx, cy, r = circle
    r_out = r * p.band_outer
    frac_in = p.band_inner / p.band_outer
    width = int(np.ceil(POLAR_RAD / max(1e-3, 1.0 - frac_in))) + 1
    pad = int(np.ceil(r_out)) + 2
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    full = cv2.warpPolar(g, (width, POLAR_ANG), (cx + pad, cy + pad), r_out,
                         cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
    start = int(round(width * frac_in))
    band = full[:, start:]
    return cv2.resize(band, (POLAR_RAD, POLAR_ANG), interpolation=cv2.INTER_AREA)


def _build_lbp_lut():
    lut = np.zeros(256, dtype=np.int32)
    idx = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            lut[code] = idx
            idx += 1
        else:
            lut[code] = 58
    return lut


_LBP_LUT = _build_lbp_lut()
_LBP_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _lbp_hist(img: np.ndarray) -> np.ndarray:
    c = img[1:-1, 1:-1]
    code = np.zeros(c.shape, dtype=np.int32)
    h, w = img.shape
    for i, (dy, dx) in enumerate(_LBP_OFFSETS):
        nb = img[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= (nb >= c).astype(np.int32) << i
    hist = np.bincount(_LBP_LUT[code].ravel(), minlength=59).astype(np.float32)
    return hist / (hist.sum() + 1e-6)


def _periodicity(z: np.ndarray):
    """Ukuran periodisitas sepanjang radius pada segmen polar z (sudut x radius).

    Mengembalikan (spektrum ternormalisasi 24 bin, peak_frac, ac_peak, zero_cross,
    konsistensi antar sektor).
    """
    n = z.shape[1]
    prof = z.reshape(N_SECTORS, z.shape[0] // N_SECTORS, n).mean(axis=1)
    # buang trend (tepi lubang, gradasi cahaya) agar tersisa pola berulang
    smooth = cv2.blur(prof, (9, 1), borderType=cv2.BORDER_REFLECT)
    d = prof - smooth
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(d * win, n=POLAR_RAD, axis=1)).mean(axis=0)
    spec_n = spec / (spec[1:].sum() + 1e-6)
    peak_frac = float(spec_n[3:24].max())

    f = np.fft.rfft(d, n=2 * n, axis=1)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=1)[:, :n]
    ac = ac / (ac[:, :1] + 1e-6)
    hi = max(4, n // 2)
    ac_peak = float(np.clip(ac[:, 3:hi].max(axis=1), 0, 1).mean())

    zc = float((np.diff(np.sign(d), axis=1) != 0).mean())

    # cincin ulir muncul di semua sudut -> profil antar sektor mirip
    dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-6)
    corr = dn @ dn.T
    consistency = float(np.abs(corr[np.triu_indices(N_SECTORS, 1)]).mean())
    return spec_n[1:25].astype(np.float32), peak_frac, ac_peak, zc, consistency


def band_features(band: np.ndarray, p: DetectParams):
    """Hitung vektor fitur (float32) dan skor rule-based (0..1) dari citra polar."""
    b = band.astype(np.float32)
    raw_mean, raw_std = float(b.mean()), float(b.std())
    z = (b - raw_mean) / (raw_std + 1e-6)

    # Kolom polar yang bersesuaian dengan tepi lubang (radius = 1.0 x r)
    edge_col = int(round((1.0 - p.band_inner) / (p.band_outer - p.band_inner) * POLAR_RAD))
    edge_col = int(np.clip(edge_col, 1, POLAR_RAD - 1))
    margin = max(2, POLAR_RAD // 16)
    in_end = max(12, edge_col - margin)     # area dinding lubang, tepi dikecualikan

    # Periodisitas seluruh band dan khusus dinding dalam lubang
    fft_all, pk_all, ac_all, zc_all, cons_all = _periodicity(z)
    zin = b[:, :in_end]
    zin = (zin - zin.mean()) / (zin.std() + 1e-6)
    fft_in, pk_in, ac_in, zc_in, cons_in = _periodicity(zin)

    # Anisotropi gradien: cincin -> gradien dominan searah radius
    def aniso_of(zz):
        gx = cv2.Sobel(zz, cv2.CV_32F, 1, 0, ksize=3)   # sepanjang radius
        gy = cv2.Sobel(zz, cv2.CV_32F, 0, 1, ksize=3)   # sepanjang sudut
        ex, ey = float((gx ** 2).mean()), float((gy ** 2).mean())
        return ex / (ex + ey + 1e-6), np.sqrt(gx ** 2 + gy ** 2)

    aniso_all, gm = aniso_of(z)
    aniso_in, gm_in = aniso_of(zin)

    u8 = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(u8, 50, 150) > 0
    edge_density = float(edges.mean())
    edge_density_in = float(edges[:, :in_end].mean())
    lap_var = float(cv2.Laplacian(z, cv2.CV_32F).var())

    inside = float(b[:, :edge_col].mean()) / 255.0
    outside = float(b[:, edge_col:].mean()) / 255.0
    inside_rel_std = float(b[:, :in_end].std() / (raw_std + 1e-6))

    lbp_all = _lbp_hist(u8)
    lbp_in = _lbp_hist(u8[:, :in_end])

    scalars = np.array([
        raw_mean / 255.0, raw_std / 255.0, inside, outside, inside_rel_std,
        pk_all, ac_all, zc_all, cons_all, aniso_all,
        pk_in, ac_in, zc_in, cons_in, aniso_in,
        float(gm.mean()), float(gm.std()), float(np.percentile(gm, 90)),
        float(gm_in.mean()), float(np.percentile(gm_in, 90)),
        edge_density, edge_density_in, float(np.log1p(lap_var)),
    ], dtype=np.float32)
    feats = np.concatenate([scalars, fft_all, fft_in, lbp_all, lbp_in]).astype(np.float32)

    def clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    # Skor rule-based: periodisitas di dinding dalam lubang
    rule = (0.40 * clip01((ac_in - 0.10) / 0.45)
            + 0.25 * clip01((pk_in - 0.10) / 0.15)
            + 0.20 * clip01((aniso_in - 0.55) / 0.35)
            + 0.15 * clip01((0.32 - zc_in) / 0.14))
    return feats, rule


def analyze_top(roi_img: np.ndarray, p: DetectParams) -> Analysis:
    """Mode tegak lurus: deteksi lubang + polar unwrap."""
    gray = to_gray(roi_img)
    small, scale = _prepare(gray, p.proc_max_side)
    circle = detect_hole(small, p)
    if circle is None:
        return Analysis(None, None, None, None)
    band = polar_band(small, circle, p)
    feats, rule = band_features(band, p)
    cx, cy, r = circle
    return Analysis(feats, rule, (cx / scale, cy / scale, r / scale), band,
                    {"radius_px": r / scale})


# ---------------------------------------------------------------------------
# Mode sudut miring
# ---------------------------------------------------------------------------
def _obl_rotate(g: np.ndarray, angle: float) -> np.ndarray:
    n = g.shape[0]
    M = cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1.0)
    r = cv2.warpAffine(g, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    c = int(n * 0.12)      # buang tepi yang terkena efek rotasi
    return r[c:n - c, c:n - c]


def _obl_detrended(patch: np.ndarray) -> np.ndarray:
    """Profil intensitas vertikal per strip dengan trend (gradasi cahaya) dibuang."""
    h = patch.shape[0]
    prof = np.stack([s.mean(axis=1) for s in np.array_split(patch, OBL_STRIPS, axis=1)])
    win = max(5, h // 10) | 1
    smooth = cv2.blur(prof, (win, 1), borderType=cv2.BORDER_REFLECT)
    return prof - smooth


def _obl_profiles(patch: np.ndarray) -> np.ndarray:
    d = _obl_detrended(patch)
    return d / (d.std(axis=1, keepdims=True) + 1e-6)


def _obl_spectrum(d: np.ndarray) -> np.ndarray:
    h = d.shape[1]
    spec = np.abs(np.fft.rfft(d * np.hanning(h), n=OBL_FFT, axis=1))
    return spec / (spec[:, 1:].sum(axis=1, keepdims=True) + 1e-6)


def _obl_ridges(patch: np.ndarray) -> np.ndarray:
    """Profil garis terang sempit per strip (white top-hat), dalam level abu-abu.

    Top-hat hanya meloloskan puncak terang yang lebih sempit dari jendela, sehingga
    gradasi cahaya dan batas gelap keluar lubang (pelat tipis) tidak dihitung sebagai garis.
    """
    h = patch.shape[0]
    prof = np.stack([s.mean(axis=1) for s in np.array_split(patch, OBL_STRIPS, axis=1)])
    prof = cv2.GaussianBlur(prof.astype(np.float32), (5, 1), 1.0)
    w = max(5, int(h * OBL_TOPHAT)) | 1
    return cv2.morphologyEx(prof, cv2.MORPH_TOPHAT, np.ones((1, w), np.uint8),
                            borderType=cv2.BORDER_REPLICATE)


def _strip_coherence(r: np.ndarray, max_lag: int = 0) -> float:
    """Median korelasi profil strip bersebelahan: tinggi bila garis menerus melintasi ROI."""
    rn = r - r.mean(axis=1, keepdims=True)
    rn = rn / (np.linalg.norm(rn, axis=1, keepdims=True) + 1e-6)
    vals = []
    for i in range(len(rn) - 1):
        a, b = rn[i], rn[i + 1]
        best = -1.0
        for lag in range(-max_lag, max_lag + 1):
            c = float(np.dot(a[max(0, lag):len(a) + min(0, lag)], b[max(0, -lag):len(b) - max(0, lag)]))
            best = max(best, c)
        vals.append(best)
    return float(np.median(vals))


def _angle_score(patch: np.ndarray) -> float:
    return _strip_coherence(_obl_ridges(patch))


def _search_angle(g: np.ndarray, center: float, search: float) -> float:
    """Cari sudut pelurusan garis. Rentang lebar (mis. ±90° = semua arah) dicari kasar dulu
    pada citra kecil, lalu diperhalus di sekitar kandidat terbaik."""
    small = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)

    def best_of(angles, img):
        scores = np.array([_angle_score(_obl_rotate(img, a)) for a in angles])
        # beberapa sudut sering hampir sama baiknya (garis melengkung): pilih yang paling
        # dekat dengan arah yang diharapkan di antara kandidat terbaik
        good = angles[scores >= scores.max() - 0.1 * abs(scores.max())]
        return float(good[np.argmin(np.abs(good - center))])

    if search <= 30:
        return best_of(np.arange(-search, search + 0.1, 5.0) + center, g)
    coarse = best_of(np.arange(-search, search + 0.1, 15.0) + center, small)
    return best_of(np.arange(coarse - 10, coarse + 10.1, 5.0), g)


def _track_ridges(r: np.ndarray, min_height: float, min_dist: int, tol: float):
    """Lacak garis terang strip demi strip (mengikuti lengkungan).

    Mengembalikan daftar garis: dict(ys={strip: y}, coverage, height).
    """
    S = r.shape[0]
    peaks = []
    for row in r:
        pk, props = find_peaks(row, height=0.5 * min_height, prominence=0.3 * min_height, distance=min_dist)
        peaks.append((pk.astype(float), props["peak_heights"]))

    tracks = []
    for s0 in sorted({S // 2 - 2, S // 2, S // 2 + 2}):
        for y0, h0 in zip(*peaks[s0]):
            ys, hs = {s0: y0}, [h0]
            for step in (-1, 1):
                y, s, miss = y0, s0, 0
                while 0 <= s + step < S:
                    s += step
                    pk, ph = peaks[s]
                    if len(pk):
                        j = int(np.argmin(np.abs(pk - y)))
                        if abs(pk[j] - y) <= tol:
                            y = pk[j]
                            ys[s] = y
                            hs.append(ph[j])
                            miss = 0
                            continue
                    miss += 1
                    if miss > 1:       # izinkan 1 strip hilang (mis. tertutup pantulan cahaya)
                        break
            tracks.append({"ys": ys, "coverage": len(ys) / S, "height": float(np.median(hs))})

    # buang duplikat (garis sama dari benih berbeda)
    tracks.sort(key=lambda t: -t["coverage"])
    unique = []
    for t in tracks:
        dup = False
        for u in unique:
            common = set(t["ys"]) & set(u["ys"])
            if common and np.median([abs(t["ys"][k] - u["ys"][k]) for k in common]) <= tol:
                dup = True
                break
        if not dup:
            unique.append(t)
    return unique


def _thread_lines(tracks, min_height: float, tol: float, period: float = 0.0):
    """Jumlah garis ulir valid: garis utama (puncak) yang menerus (>= 75% lebar), sejajar,
    tinggi mirip, dan berjarak ±1 pitch (bila pitch diketahui dari autokorelasi). Garis
    pantulan sisi ulir yang lebih redup diabaikan.
    Mengembalikan (jumlah garis valid, tinggi rata-rata, jarak rata-rata, jumlah garis kuat)."""
    full = [t for t in tracks if t["coverage"] >= 0.75 and t["height"] >= min_height]
    if not full:
        return 0, 0.0, 0.0, 0
    hmax = max(t["height"] for t in full)
    strong = [t for t in full if t["height"] >= 0.45 * hmax]
    strong.sort(key=lambda t: np.mean(list(t["ys"].values())))

    def pair_gap(a, b):
        common = sorted(set(a["ys"]) & set(b["ys"]))
        if len(common) < 4:
            return None
        gaps = np.array([b["ys"][k] - a["ys"][k] for k in common])
        g = float(gaps.mean())
        if g <= 0 or gaps.std() > max(0.2 * g, tol * 0.6):     # tidak sejajar
            return None
        return g

    best = (1, strong[0]["height"], 0.0)
    for i in range(len(strong)):
        chain, gaps = [strong[i]], []
        for nxt in strong[i + 1:]:
            g = pair_gap(chain[-1], nxt)
            if g is None:
                continue
            if period > 0:
                ok = 0.7 <= g / period <= 1.3
            else:
                ok = not gaps or 0.6 <= g / np.mean(gaps) <= 1.67
            if ok:
                chain.append(nxt)
                gaps.append(g)
            elif period > 0 and g > 1.3 * period:
                break
        if len(chain) > best[0]:
            best = (len(chain), float(np.mean([c["height"] for c in chain])), float(np.mean(gaps)))
    return best[0], best[1], best[2], len(strong)


def _ridge_period(r: np.ndarray, lo: int, hi: int):
    """Pitch garis dari autokorelasi profil top-hat. Pola puncak + pantulan sisi ulir berulang
    tepat satu kali per pitch, sehingga puncak autokorelasi tertinggi = pitch sebenarnya.
    Mengembalikan (lag, nilai korelasi) atau (0, 0) bila tidak ada pola berulang."""
    h = r.shape[1]
    rn = r - r.mean(axis=1, keepdims=True)
    f = np.fft.rfft(rn, n=2 * h, axis=1)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=1)[:, :h]
    ac = ac / (ac[:, :1] + 1e-6)
    m = np.median(ac, axis=0)
    hi = min(hi, h - 2)
    if hi <= lo + 2:
        return 0.0, 0.0
    pk, props = find_peaks(m[lo:hi], height=0.05)
    if not len(pk):
        return 0.0, 0.0
    j = int(np.argmax(props["peak_heights"]))
    return float(pk[j] + lo), float(props["peak_heights"][j])


def _count_ridges(r: np.ndarray, min_height: float, min_dist: int):
    """Hitung garis terang per strip; kembalikan (median jumlah, median tinggi 2 puncak teratas)."""
    counts, heights = [], []
    for row in r:
        pk, props = find_peaks(row, height=min_height, prominence=0.6 * min_height, distance=min_dist)
        counts.append(len(pk))
        hs = np.sort(props["peak_heights"])[::-1][:2] if len(pk) else np.array([0.0])
        heights.append(float(hs.mean()))
    return float(np.median(counts)), float(np.median(heights))


def analyze_oblique(roi_img: np.ndarray, p: DetectParams) -> Analysis:
    """Mode sudut miring: ukur garis-garis ulir yang berulang di dinding lubang."""
    gray = to_gray(roi_img)
    if min(gray.shape[:2]) < 16:
        return Analysis(None, None, None, None)
    g = cv2.resize(gray, (OBL_N, OBL_N), interpolation=cv2.INTER_AREA).astype(np.float32)

    # cari sudut yang membuat garis paling mendatar (garis menerus antar strip)
    if p.angle_search > 0:
        angle = _search_angle(g, p.line_angle, p.angle_search)
    else:
        angle = float(p.line_angle)
    patch = _obl_rotate(g, angle)
    h = patch.shape[0]

    lo, hi = np.percentile(patch, [2, 98])
    z = np.clip((patch - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    saturated = float((patch >= 250).mean())
    # amplitudi garis berulang dalam level abu-abu asli: ulir kontras tinggi,
    # bekas pahat pada lubang polos kontras rendah
    line_amp = float(np.median(_obl_detrended(patch).std(axis=1)))

    # --- deteksi garis (untuk ulir yang hanya terlihat 2-3 garis) ---
    minc = max(0.5, p.line_contrast)
    ridges = _obl_ridges(patch)
    n_ridges, ridge_h = _count_ridges(ridges, minc, max(3, int(h * 0.05)))
    ridge_coh = _strip_coherence(ridges, max_lag=max(2, h // 30))
    tol = max(3.0, h * 0.04)
    tracks = _track_ridges(ridges, minc, max(3, int(h * 0.05)), tol)
    ac_lag, ac_val = _ridge_period(ridges, max(4, int(h * 0.08)), int(h * 0.7))
    n_lines, lines_h, lines_gap, n_full = _thread_lines(tracks, minc, tol,
                                                        ac_lag if ac_val >= 0.2 else 0.0)
    ridge_shape = np.median(ridges, axis=0)
    ridge_shape = cv2.resize(ridge_shape.reshape(1, -1), (OBL_PROFILE_BINS, 1),
                             interpolation=cv2.INTER_AREA).ravel()
    ridge_shape = ridge_shape / (ridge_shape.max() + 1e-6)

    d = _obl_profiles(z)
    spec = _obl_spectrum(d)                                   # strip x frekuensi
    med = np.median(spec, axis=0)                             # median: tahan highlight
    band = med[OBL_KMIN:OBL_KMAX]
    k = int(np.argmax(band)) + OBL_KMIN
    peak = float(med[k])
    prominence = float(np.log(max(peak, 1e-9) / max(float(np.median(band)), OBL_FLOOR)))
    band_energy = float(med[max(OBL_KMIN, k - 2):k + 3].sum())
    period_px = OBL_FFT / k * (gray.shape[0] / OBL_N)        # dalam piksel ROI asli

    # kontras pola berulang diukur HANYA di sekitar frekuensi garis (band-pass), agar tepi
    # terang bibir lubang tidak ikut terhitung dan bekas bor yang samar tidak lolos
    d_raw = _obl_detrended(patch)
    win_h = np.hanning(h)
    F = np.fft.rfft(d_raw * win_h, n=OBL_FFT, axis=1)
    mask = np.zeros(F.shape[1])
    mask[max(1, k - 2):k + 3] = 1.0
    recon = np.fft.irfft(F * mask, n=OBL_FFT, axis=1)[:, :h] / (win_h.mean() + 1e-6)
    periodic_amp = float(np.median(recon.std(axis=1)))

    # konsistensi antar strip: garis ulir menerus melintasi seluruh dinding
    sn = spec[:, 1:] / (np.linalg.norm(spec[:, 1:], axis=1, keepdims=True) + 1e-6)
    consistency = float((sn @ sn.T)[np.triu_indices(OBL_STRIPS, 1)].mean())
    k_strip = np.argmax(spec[:, OBL_KMIN:OBL_KMAX], axis=1) + OBL_KMIN
    agree = float((np.abs(k_strip - k) <= max(1, k // 8)).mean())

    f = np.fft.rfft(d, n=2 * h, axis=1)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=1)[:, :h]
    ac = ac / (ac[:, :1] + 1e-6)
    lag = int(round(OBL_FFT / k))
    ac_at_period = float(np.median(ac[:, min(lag, h - 1)]))
    zc = float((np.diff(np.sign(d), axis=1) != 0).mean())

    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
    ex, ey = float((gx ** 2).mean()), float((gy ** 2).mean())
    aniso = ey / (ex + ey + 1e-6)          # garis mendatar -> gradien vertikal dominan
    gm = np.sqrt(gx ** 2 + gy ** 2)
    u8 = (z * 255).astype(np.uint8)
    edge_density = float((cv2.Canny(u8, 50, 150) > 0).mean())
    lap_var = float(cv2.Laplacian(z, cv2.CV_32F).var())

    scalars = np.array([
        float(patch.mean()) / 255.0, float(patch.std()) / 255.0, saturated,
        peak, prominence, band_energy, k / OBL_FFT, consistency, agree,
        ac_at_period, zc, aniso, float(gm.mean()), float(gm.std()),
        float(np.percentile(gm, 90)), edge_density, float(np.log1p(lap_var)),
        abs(angle - p.line_angle) / 45.0, line_amp / 32.0,
        line_amp / (float(patch.std()) + 1e-6),
        n_ridges / 5.0, ridge_h / 64.0, ridge_coh, float(ridges.max()) / 64.0, periodic_amp / 32.0,
        n_lines / 5.0, n_full / 5.0, lines_h / 64.0, lines_gap / h, ac_lag / h, ac_val,
        (n_lines / n_full) if n_full else 0.0,
        float(ridges.mean()) / 16.0,
    ], dtype=np.float32)
    feats = np.concatenate([scalars, med[1:65].astype(np.float32), ridge_shape.astype(np.float32),
                            _lbp_hist(u8)]).astype(np.float32)

    def clip01(v):
        return float(np.clip(v, 0.0, 1.0))

    periodic = (0.40 * clip01((prominence - 2.0) / 1.0)
                + 0.20 * clip01((agree - 0.25) / 0.5)
                + 0.20 * clip01((ac_at_period + 0.1) / 0.6)
                + 0.20 * clip01((peak - 0.03) / 0.05))
    contrast_gate = clip01((periodic_amp - 0.5 * minc) / minc)
    periodic_score = periodic * contrast_gate

    # skor garis: >= 2 garis ulir valid (menerus, sejajar, tinggi mirip, jarak teratur).
    # Garis lain yang tidak masuk pola (mis. fitur wajah, lampu) menurunkan skor.
    count_term = 0.0 if n_lines < 2 else (0.85 if n_lines < 3 else 1.0)
    purity = n_lines / n_full if n_full else 0.0
    ridge_score = (count_term * clip01((lines_h - minc) / (1.5 * minc))
                   * clip01((purity - 0.3) / 0.4))
    # jarak garis tiap jalur (px ROI asli)
    scale = gray.shape[0] / OBL_N
    ridge_period = ac_lag * scale if ac_lag > 0 else (lines_gap * scale if lines_gap > 0 else 0.0)
    periodic_period = period_px

    # kalibrasi: pada setup kamera tetap, jarak garis ulir selalu sama. Garis lain (rak,
    # lampu, fitur wajah) jaraknya acak sehingga ditolak.
    if p.expected_pitch > 0:
        def pitch_match(period):
            if period <= 0:
                return 0.0
            dev = abs(period / p.expected_pitch - 1.0)
            return clip01(1.0 - (dev - p.pitch_tol) / max(p.pitch_tol, 1e-3))
        periodic_score *= pitch_match(periodic_period)
        ridge_score *= pitch_match(ridge_period)
    rule = max(periodic_score, ridge_score)

    # jarak garis: dari autokorelasi garis bila pola garis jelas, selain itu dari spektrum
    if ridge_period > 0 and (n_lines >= 2 or ac_val >= 0.25):
        period_px = ridge_period
    info = {"period_px": period_px, "angle": angle, "prominence": prominence, "agree": agree,
            "ac": ac_at_period, "peak": peak,
            "contrast": lines_h if ridge_score >= periodic_score else periodic_amp,
            "lines": n_lines, "lines_full": n_full, "ridge_h": lines_h, "ridge_coh": ridge_coh,
            "ridge_ac": ac_val,
            "periodic": periodic_score, "ridge": ridge_score}
    return Analysis(feats, rule, None, u8, info)


def analyze_roi(roi_img: np.ndarray, p: DetectParams) -> Analysis:
    """Analisis satu citra ROI sesuai mode sudut pandang kamera."""
    if p.view_mode == "top":
        return analyze_top(roi_img, p)
    return analyze_oblique(roi_img, p)


# ---------------------------------------------------------------------------
# Inspeksi lengkap (ROI + keputusan)
# ---------------------------------------------------------------------------
class Inspector:
    def __init__(self, model_manager):
        self.models = model_manager

    def inspect(self, frame: np.ndarray, cfg) -> InspectionResult:
        t0 = time.perf_counter()
        p = DetectParams.from_config(cfg)
        x, y, w, h = roi_to_pixels(cfg.roi, frame.shape)
        roi = frame[y:y + h, x:x + w]
        a = analyze_roi(roi, p)

        if a.features is None:
            reason = ("Lubang tidak terdeteksi di ROI" if p.view_mode == "top"
                      else "ROI terlalu kecil untuk dianalisis")
            return InspectionResult("NO_HOLE", None, "-", reason, None, (x, y, w, h), None, None,
                                    (time.perf_counter() - t0) * 1000, frame, time.time())

        circle = None
        if a.circle is not None:
            cx, cy, r = a.circle
            circle = (cx + x, cy + y, r)
        judgement, score, method, reason, extra = self.decide(a, cfg, roi)
        info = dict(a.info)
        info.update(extra)
        return InspectionResult(judgement, score, method, reason, circle, (x, y, w, h), a.band,
                                a.rule_score, (time.perf_counter() - t0) * 1000, frame, time.time(),
                                info)

    def decide(self, a: Analysis, cfg, roi=None):
        has_model = self.models.is_ready()
        use_ml = cfg.decision_method == "ml" or (cfg.decision_method == "auto" and has_model)
        if use_ml:
            if not has_model:
                return "N/A", None, "ML", "Model belum dilatih", {}
            model_mode = self.models.meta.get("view_mode", "top")
            if model_mode != cfg.view_mode:
                return "N/A", None, "ML", (f"Model dilatih untuk mode '{VIEW_MODES.get(model_mode, model_mode)}'"
                                           ", latih ulang untuk mode ini"), {}
            prob = self.models.predict_proba(a.features, roi)
            if prob is None:
                return "N/A", None, "ML", "Model tidak cocok, latih ulang", {}
            nov = self.models.novelty_ratio(a.features)
            extra = {"novelty": nov}
            if cfg.novelty_tolerance > 0 and nov is not None and nov > cfg.novelty_tolerance:
                extra["unknown"] = True
                return "NG", 0.0, "ML", "Objek tidak dikenal (beda dari sampel training)", extra
            ok = prob >= cfg.ok_threshold
            return ("OK" if ok else "NG"), prob, "ML", ("Ulir terdeteksi" if ok else "Ulir tidak terdeteksi"), extra
        if cfg.view_mode != "top" and cfg.rule_pitch_px <= 0:
            # tanpa referensi, "garis sejajar" bisa berasal dari apa saja: jangan pernah OK
            return "N/A", a.rule_score, "Rule", "Belum ada referensi: kalibrasi Rule atau latih model", {}
        ok = a.rule_score >= cfg.rule_threshold
        return ("OK" if ok else "NG"), a.rule_score, "Rule", ("Pola ulir terdeteksi" if ok else "Pola ulir lemah"), {}


# ---------------------------------------------------------------------------
# Overlay visual
# ---------------------------------------------------------------------------
COLOR_OK = (94, 197, 34)      # BGR
COLOR_NG = (68, 68, 239)
COLOR_NA = (11, 158, 245)
COLOR_ROI = (246, 130, 59)


def draw_overlay(frame: np.ndarray, res: InspectionResult, cfg) -> np.ndarray:
    vis = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = vis.shape[:2]
    t = max(1, int(round(min(h, w) / 360)))
    x, y, rw, rh = res.roi_px
    judge_color = {"OK": COLOR_OK, "NG": COLOR_NG, "NO_HOLE": COLOR_NG}.get(res.judgement, COLOR_NA)

    if cfg.view_mode != "top":
        # mode miring: bingkai ROI berwarna sesuai hasil + garis arah ulir yang terdeteksi
        cv2.rectangle(vis, (x, y), (x + rw, y + rh), judge_color, t + 1, cv2.LINE_AA)
        _corner_marks(vis, x, y, rw, rh, judge_color, t + 2)
        ang = res.info.get("angle")
        if ang is not None:
            # sudut rotasi pelurusan = kemiringan garis ulir (searah jarum jam) di gambar
            rad = np.radians(ang)
            dx, dy = np.cos(rad), np.sin(rad)
            cx, cy = x + rw / 2, y + rh / 2
            L = rw * 0.35
            for off in (-0.25, 0.0, 0.25):
                oy = off * rh
                p0 = (int(cx - dx * L), int(cy + oy - dy * L))
                p1 = (int(cx + dx * L), int(cy + oy + dy * L))
                _dashed_line(vis, p0, p1, (230, 230, 230), t)
        return vis

    cv2.rectangle(vis, (x, y), (x + rw, y + rh), COLOR_ROI, t, cv2.LINE_AA)
    _corner_marks(vis, x, y, rw, rh, COLOR_ROI, t + 1)

    if res.circle is not None:
        cx, cy, r = res.circle
        color = {"OK": COLOR_OK, "NG": COLOR_NG}.get(res.judgement, COLOR_NA)
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(vis, c, int(round(r)), color, t + 1, cv2.LINE_AA)
        for k in (cfg.band_inner, cfg.band_outer):
            _dashed_circle(vis, c, r * k, (200, 200, 200), t)
        cv2.drawMarker(vis, c, color, cv2.MARKER_CROSS, int(r * 0.25) + 6, t, cv2.LINE_AA)
    return vis


def _corner_marks(img, x, y, w, h, color, t):
    L = max(10, int(min(w, h) * 0.08))
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * L, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * L), color, t, cv2.LINE_AA)


def _dashed_line(img, p0, p1, color, t, n=14):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    for i in range(0, n, 2):
        a = p0 + (p1 - p0) * i / n
        b = p0 + (p1 - p0) * (i + 1) / n
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, t, cv2.LINE_AA)


def _dashed_circle(img, center, radius, color, t, segments=48):
    for i in range(0, segments, 2):
        a0 = 360.0 * i / segments
        a1 = 360.0 * (i + 1) / segments
        cv2.ellipse(img, center, (int(radius), int(radius)), 0, a0, a1, color, t, cv2.LINE_AA)
