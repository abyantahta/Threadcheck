"""Sumber gambar: webcam/kamera USB (OpenCV), Raspberry Pi Camera (Picamera2), atau file."""
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

SOURCE_TYPES = {
    "opencv": "Kamera USB / Webcam (OpenCV)",
    "picamera2": "Raspberry Pi Camera (Picamera2)",
    "file": "File video / gambar (simulasi)",
}

BACKENDS = {
    "auto": ("Otomatis", None),
    "dshow": ("DirectShow (Windows)", cv2.CAP_DSHOW),
    "msmf": ("Media Foundation (Windows)", cv2.CAP_MSMF),
    "v4l2": ("V4L2 (Linux / Raspberry Pi)", cv2.CAP_V4L2),
    "any": ("OpenCV default", cv2.CAP_ANY),
}

RESOLUTIONS = [(640, 480), (800, 600), (1280, 720), (1280, 960), (1600, 1200),
               (1920, 1080), (2592, 1944), (3840, 2160)]

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"


@dataclass
class CameraInfo:
    index: int
    label: str


class CameraError(Exception):
    pass


def _resolve_backend(name: str) -> int:
    if name == "auto":
        if IS_WINDOWS:
            return cv2.CAP_DSHOW
        if IS_LINUX:
            return cv2.CAP_V4L2
        return cv2.CAP_ANY
    return BACKENDS.get(name, ("", cv2.CAP_ANY))[1]


class OpenCVSource:
    def __init__(self, index=0, backend="auto", width=1280, height=720):
        self.index, self.backend, self.width, self.height = index, backend, width, height
        self.cap = None

    def open(self):
        api = _resolve_backend(self.backend)
        cap = cv2.VideoCapture(self.index, api)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Kamera #{self.index} tidak bisa dibuka")
        if self.width >= 1280:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise CameraError(f"Kamera #{self.index} terbuka, tetapi tidak mengirim gambar")
        self.cap = cap

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def resolution(self):
        if self.cap is None:
            return 0, 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def describe(self):
        w, h = self.resolution()
        return f"Kamera #{self.index} · {w}×{h}"

    def open_driver_settings(self):
        """Buka dialog pengaturan driver (exposure, fokus, dll). Hanya DirectShow."""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class Picamera2Source:
    def __init__(self, index=0, width=1280, height=720):
        self.index, self.width, self.height = index, width, height
        self.cam = None

    def open(self):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError("Library picamera2 tidak tersedia (hanya untuk Raspberry Pi)") from e
        try:
            cam = Picamera2(self.index)
            conf = cam.create_video_configuration(main={"size": (self.width, self.height),
                                                        "format": "RGB888"})
            cam.configure(conf)
            cam.start()
        except Exception as e:
            raise CameraError(f"Raspberry Pi Camera #{self.index} gagal dibuka: {e}") from e
        self.cam = cam

    def read(self):
        if self.cam is None:
            return False, None
        try:
            # format "RGB888" picamera2 tersusun B,G,R di memori -> langsung cocok untuk OpenCV
            return True, self.cam.capture_array("main")
        except Exception:
            return False, None

    def resolution(self):
        return self.width, self.height

    def describe(self):
        return f"Pi Camera #{self.index} · {self.width}×{self.height}"

    def open_driver_settings(self):
        pass

    def release(self):
        if self.cam is not None:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception:
                pass
            self.cam = None


class FileSource:
    IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, path: str):
        self.path = Path(path) if path else None
        self.cap = None
        self.image = None
        self.fps = 15.0
        self._t_last = 0.0

    def open(self):
        if self.path is None or not self.path.exists():
            raise CameraError("File simulasi belum dipilih atau tidak ditemukan")
        if self.path.suffix.lower() in self.IMAGE_EXT:
            from .storage import read_image
            self.image = read_image(self.path)
            if self.image is None:
                raise CameraError("Gambar tidak bisa dibaca")
        else:
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise CameraError("Video tidak bisa dibuka")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            self.cap = cap

    def _throttle(self):
        dt = 1.0 / max(1.0, self.fps) - (time.perf_counter() - self._t_last)
        if dt > 0:
            time.sleep(dt)
        self._t_last = time.perf_counter()

    def read(self):
        self._throttle()
        if self.image is not None:
            return True, self.image.copy()
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        if not ok:   # ulang dari awal
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def resolution(self):
        if self.image is not None:
            return self.image.shape[1], self.image.shape[0]
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0, 0

    def describe(self):
        w, h = self.resolution()
        return f"File: {self.path.name} · {w}×{h}"

    def open_driver_settings(self):
        pass

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.image = None


def create_source(cfg):
    if cfg.camera_source == "picamera2":
        return Picamera2Source(cfg.camera_index, cfg.camera_width, cfg.camera_height)
    if cfg.camera_source == "file":
        return FileSource(cfg.camera_file)
    return OpenCVSource(cfg.camera_index, cfg.camera_backend, cfg.camera_width, cfg.camera_height)


# ---------------------------------------------------------------------------
# Enumerasi perangkat
# ---------------------------------------------------------------------------
def _device_names():
    """Nama kamera bila bisa didapat dari OS (tanpa membuka kamera)."""
    names = {}
    if IS_LINUX:
        for p in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                idx = int(p.name.replace("video", ""))
                names[idx] = (p / "name").read_text().strip()
            except (OSError, ValueError):
                pass
    elif IS_WINDOWS:
        try:
            from pygrabber.dshow_graph import FilterGraph   # opsional
            for i, n in enumerate(FilterGraph().get_input_devices()):
                names[i] = n
        except Exception:
            pass
    return names


def list_opencv_cameras(backend="auto", max_index=8, skip_indices=()):
    """Cari kamera yang bisa dibuka. Index pada skip_indices dianggap tersedia
    (mis. kamera yang sedang aktif dipakai, karena tidak bisa dibuka dua kali)."""
    names = _device_names()
    api = _resolve_backend(backend)
    found = []
    misses = 0
    for i in range(max_index):
        if i in skip_indices:
            found.append(CameraInfo(i, f"#{i} · {names.get(i, 'Kamera')} (aktif)"))
            continue
        cap = cv2.VideoCapture(i, api)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            found.append(CameraInfo(i, f"#{i} · {names.get(i, 'Kamera')}"))
            misses = 0
        else:
            misses += 1
            if misses >= 3 and not IS_LINUX:   # di Linux index bisa loncat (video0, video2, ...)
                break
    return found


def list_picamera2_cameras():
    try:
        from picamera2 import Picamera2
        info = Picamera2.global_camera_info()
    except Exception:
        return []
    return [CameraInfo(i, f"#{i} · {c.get('Model', 'Pi Camera')}") for i, c in enumerate(info)]
