"""State aplikasi yang dibagi antar halaman."""
from . import config
from .config import load_config, save_config
from .gpio_out import GpioOutputs
from .model import ModelManager
from .qt import QtCore, Signal


class AppContext(QtCore.QObject):
    config_changed = Signal(object)
    model_changed = Signal()
    profile_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.models = ModelManager()
        self.models.load()
        self.gpio = GpioOutputs(self)
        self.gpio.apply_config(self.cfg)
        self.worker = None
        self.training_active = False
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(lambda: save_config(self.cfg))

    def update(self, reopen_camera=False, **changes):
        self.cfg = self.cfg.with_(**changes)
        self._save_timer.start()
        if self.worker is not None:
            self.worker.update_config(self.cfg, reopen=reopen_camera)
        self.config_changed.emit(self.cfg)

    def flush(self):
        self._save_timer.stop()
        save_config(self.cfg)

    # --- profil kamera ---
    @property
    def profile(self) -> str:
        return config.active_profile()

    def switch_profile(self, name: str):
        """Aktifkan profil lain: pengaturan, kamera, ROI, dataset, dan model ikut berganti."""
        if name == self.profile:
            return
        self.flush()
        sidebar = self.cfg.sidebar_collapsed
        self.cfg = config.switch_profile(name).with_(sidebar_collapsed=sidebar)
        self.models.clear()
        self.models.load()
        if self.worker is not None:
            self.worker.update_config(self.cfg, reopen=True)
        self.profile_changed.emit(name)
        self.config_changed.emit(self.cfg)
        self.model_changed.emit()

    def create_profile(self, name: str, copy_current: bool):
        base = self.cfg if copy_current else config.AppConfig()
        if not copy_current:
            base = base.with_(sidebar_collapsed=self.cfg.sidebar_collapsed)
        self.flush()
        config.create_profile(name, base)
        self.switch_profile(name)

    def rename_profile(self, old: str, new: str):
        self.flush()
        config.rename_profile(old, new)
        self.profile_changed.emit(self.profile)

    def delete_profile(self, name: str):
        config.delete_profile(name)
        self.profile_changed.emit(self.profile)
