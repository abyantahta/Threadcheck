"""Konfigurasi aplikasi, profil kamera, path folder, dan konstanta kelas.

Setiap **profil kamera** punya dataset, model, dan pengaturannya sendiri:

    profiles/<nama profil>/config.json      pengaturan (kamera, ROI, analisis, keputusan)
    profiles/<nama profil>/dataset/...      sampel training ulir / tanpa_ulir
    profiles/<nama profil>/models/...       model hasil training
    captures/<nama profil>/OK|NG/<tanggal>  gambar hasil inspeksi
    logs/<nama profil>/inspeksi_*.csv       log inspeksi

`config.json` di folder utama hanya menyimpan profil aktif dan preferensi tampilan.
"""
import json
import re
import shutil
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = BASE_DIR / "profiles"
CAPTURES_ROOT = BASE_DIR / "captures"
LOGS_ROOT = BASE_DIR / "logs"
WEIGHTS_DIR = BASE_DIR / "weights"           # bobot dasar YOLO, dipakai bersama semua profil
GLOBAL_CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_PROFILE = "Kamera Jernih"
MODEL_FILE = "model_ulir.joblib"

# Label kelas: 1 = berulir (OK), 0 = tanpa ulir (NG)
CLASS_OK = "ulir"
CLASS_NG = "tanpa_ulir"
CLASS_LABEL = {CLASS_OK: 1, CLASS_NG: 0}
CLASS_DISPLAY = {CLASS_OK: "ULIR (OK)", CLASS_NG: "TANPA ULIR (NG)"}

APP_NAME = "Threck"
APP_SUBTITLE = "Inspeksi Ulir Lubang"


@dataclass(frozen=True)
class AppConfig:
    # --- Kamera ---
    camera_source: str = "opencv"          # opencv | picamera2 | file
    camera_index: int = 0
    camera_backend: str = "auto"           # auto | dshow | msmf | v4l2 | any
    camera_width: int = 1280
    camera_height: int = 720
    camera_file: str = ""

    # --- ROI (dinormalisasi 0..1: x, y, w, h) ---
    roi: tuple = (0.30, 0.20, 0.40, 0.60)

    # --- Keputusan ---
    ok_threshold: float = 0.50             # ambang probabilitas "berulir" untuk OK
    trigger_frames: int = 5                # jumlah frame dirata-rata per trigger
    novelty_tolerance: float = 1.0         # toleransi "objek tidak dikenal" (0 = nonaktif)
    continuous_smoothing: int = 5          # jumlah frame untuk menghaluskan tampilan kontinu

    # --- Output GPIO (Raspberry Pi, nomor BCM) ---
    gpio_ok_enabled: bool = False          # nyalakan pin saat hasil inspeksi OK
    gpio_ok_pin: int = 17
    gpio_ok_active_high: bool = True       # True = 3,3 V saat OK; False = 0 V saat OK
    gpio_ok_pulse_ms: int = 500            # 0 = pin tetap menyala selama hasil masih OK
    gpio_ng_enabled: bool = False          # nyalakan pin saat hasil inspeksi NG
    gpio_ng_pin: int = 27
    gpio_ng_active_high: bool = True
    gpio_ng_pulse_ms: int = 500            # 0 = pin tetap menyala selama hasil masih NG

    # --- Penyimpanan ---
    save_ng_images: bool = True
    save_ok_images: bool = False
    log_csv: bool = True

    # --- Training ---
    yolo_epochs: int = 40                  # jumlah epoch training YOLO26n-cls (maksimum bila early stopping)
    yolo_early_stop: bool = False          # hentikan training bila akurasi validasi tidak membaik
    yolo_patience: int = 10                # jumlah epoch tanpa perbaikan sebelum berhenti

    # --- UI ---
    inspect_mode: str = "trigger"          # trigger | continuous
    sidebar_collapsed: bool = False        # sidebar navigasi diciutkan (global, bukan per profil)
    display_mode: str = "normal"           # normal | fullscreen | pi800 (global, bukan per profil)

    def with_(self, **kw) -> "AppConfig":
        return replace(self, **kw)


GLOBAL_KEYS = ("sidebar_collapsed", "display_mode")

DISPLAY_MODES = {
    "normal": "Normal (jendela)",
    "fullscreen": "Layar penuh",
    "pi800": "Raspberry Pi 800×480 (layar penuh, ringkas)",
}

# ---------------------------------------------------------------------------
# Profil aktif & path
# ---------------------------------------------------------------------------
_state = {"profile": DEFAULT_PROFILE}


def active_profile() -> str:
    return _state["profile"]


def profile_dir(name=None) -> Path:
    return PROFILES_DIR / (name or active_profile())


def dataset_dir(name=None) -> Path:
    return profile_dir(name) / "dataset"


def models_dir(name=None) -> Path:
    return profile_dir(name) / "models"


def model_path(name=None) -> Path:
    return models_dir(name) / MODEL_FILE


def captures_dir(name=None) -> Path:
    return CAPTURES_ROOT / (name or active_profile())


def logs_dir(name=None) -> Path:
    return LOGS_ROOT / (name or active_profile())


def list_profiles():
    if not PROFILES_DIR.exists():
        return []
    return sorted((d.name for d in PROFILES_DIR.iterdir()
                   if d.is_dir() and (d / "config.json").exists()), key=str.lower)


def profile_name_error(name: str, exclude: str = None):
    """Pesan kesalahan bila nama profil tidak valid, atau None bila valid."""
    name = (name or "").strip()
    if not name:
        return "Nama profil tidak boleh kosong."
    if len(name) > 40:
        return "Nama profil maksimal 40 karakter."
    if not re.fullmatch(r"[A-Za-z0-9 _.\-()]+", name) or name.startswith("."):
        return "Gunakan huruf, angka, spasi, titik, tanda hubung, garis bawah, atau kurung saja."
    if any(p.lower() == name.lower() for p in list_profiles() if p != exclude):
        return f"Profil '{name}' sudah ada."
    return None


# ---------------------------------------------------------------------------
# Baca/tulis
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _cfg_from_dict(data: dict) -> AppConfig:
    valid = {f.name for f in fields(AppConfig)}
    data = {k: v for k, v in data.items() if k in valid}
    if "roi" in data:
        data["roi"] = tuple(float(v) for v in data["roi"])
    try:
        return AppConfig(**data)
    except TypeError:
        return AppConfig()


def _cfg_to_dict(cfg: AppConfig) -> dict:
    data = asdict(cfg)
    data["roi"] = list(cfg.roi)
    for k in GLOBAL_KEYS:
        data.pop(k, None)
    return data


def _move_contents(src: Path, dst: Path):
    """Pindahkan isi folder src ke dst (tanpa menimpa file yang sudah ada)."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            if item.is_dir():
                _move_contents(item, target)
                try:
                    item.rmdir()
                except OSError:
                    pass
            continue
        shutil.move(str(item), str(target))
    try:
        src.rmdir()
    except OSError:
        pass


def _migrate_legacy():
    """Versi lama menyimpan dataset/model/config langsung di folder utama.
    Pindahkan semuanya ke profil pertama agar tidak ada data yang hilang."""
    if list_profiles():
        return
    old = _read_json(GLOBAL_CONFIG_PATH)
    legacy_cfg = {k: v for k, v in old.items() if k not in ("active_profile",)}
    prof = profile_dir(DEFAULT_PROFILE)
    _move_contents(BASE_DIR / "dataset", prof / "dataset")
    legacy_models = BASE_DIR / "models"
    base_w = legacy_models / "yolo26n-cls.pt"
    if base_w.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        if not (WEIGHTS_DIR / base_w.name).exists():
            shutil.move(str(base_w), str(WEIGHTS_DIR / base_w.name))
    _move_contents(legacy_models, prof / "models")
    for sub in ("OK", "NG"):
        _move_contents(CAPTURES_ROOT / sub, CAPTURES_ROOT / DEFAULT_PROFILE / sub)
    if LOGS_ROOT.exists():
        for f in list(LOGS_ROOT.glob("*.csv")):
            (LOGS_ROOT / DEFAULT_PROFILE).mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(LOGS_ROOT / DEFAULT_PROFILE / f.name))
    _write_json(prof / "config.json", _cfg_to_dict(_cfg_from_dict(legacy_cfg)))
    _write_json(GLOBAL_CONFIG_PATH, {"active_profile": DEFAULT_PROFILE,
                                     "sidebar_collapsed": bool(old.get("sidebar_collapsed", False))})


def _read_profile_cfg(name: str) -> AppConfig:
    cfg = _cfg_from_dict(_read_json(profile_dir(name) / "config.json"))
    g = _read_json(GLOBAL_CONFIG_PATH)
    return cfg.with_(**{k: g[k] for k in GLOBAL_KEYS if k in g})


def load_config() -> AppConfig:
    _migrate_legacy()
    profiles = list_profiles()
    if not profiles:
        create_profile(DEFAULT_PROFILE, AppConfig())
        profiles = [DEFAULT_PROFILE]
    name = _read_json(GLOBAL_CONFIG_PATH).get("active_profile")
    if name not in profiles:
        name = profiles[0]
    _state["profile"] = name
    return _read_profile_cfg(name)


def save_config(cfg: AppConfig) -> None:
    _write_json(profile_dir() / "config.json", _cfg_to_dict(cfg))
    g = _read_json(GLOBAL_CONFIG_PATH)
    g["active_profile"] = active_profile()
    for k in GLOBAL_KEYS:
        g[k] = getattr(cfg, k)
    _write_json(GLOBAL_CONFIG_PATH, g)


def ensure_dirs() -> None:
    for d in (dataset_dir() / CLASS_OK, dataset_dir() / CLASS_NG, models_dir(), logs_dir(),
              captures_dir() / "OK", captures_dir() / "NG"):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Kelola profil
# ---------------------------------------------------------------------------
def create_profile(name: str, cfg: AppConfig) -> None:
    name = name.strip()
    for d in (dataset_dir(name) / CLASS_OK, dataset_dir(name) / CLASS_NG, models_dir(name)):
        d.mkdir(parents=True, exist_ok=True)
    _write_json(profile_dir(name) / "config.json", _cfg_to_dict(cfg))


def switch_profile(name: str) -> AppConfig:
    _state["profile"] = name
    g = _read_json(GLOBAL_CONFIG_PATH)
    g["active_profile"] = name
    _write_json(GLOBAL_CONFIG_PATH, g)
    ensure_dirs()
    return _read_profile_cfg(name)


def rename_profile(old: str, new: str) -> None:
    new = new.strip()
    for root in (PROFILES_DIR, CAPTURES_ROOT, LOGS_ROOT):
        if (root / old).exists():
            shutil.move(str(root / old), str(root / new))
    if active_profile() == old:
        _state["profile"] = new
        g = _read_json(GLOBAL_CONFIG_PATH)
        g["active_profile"] = new
        _write_json(GLOBAL_CONFIG_PATH, g)


def delete_profile(name: str) -> None:
    """Hapus dataset, model, dan pengaturan profil. Gambar hasil inspeksi & log disimpan."""
    shutil.rmtree(profile_dir(name), ignore_errors=True)
