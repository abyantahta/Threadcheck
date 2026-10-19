"""Jendela utama: header, sidebar navigasi, dan halaman-halaman."""
import time

from .. import __version__
from .. import config
from ..config import APP_NAME, APP_SUBTITLE
from ..qt import QShortcut, QtCore, QtGui, QtWidgets, Qt
from ..workers import CameraWorker
from .inspect_page import InspectPage
from . import profiles as profile_actions
from .logo import logo_icon, logo_pixmap
from .settings_page import SettingsPage
from .training_page import TrainingPage
from .widgets import Chip, button, label

NAV = [("◉", "Inspeksi", 0), ("▣", "Training", 1), ("⚙", "Pengaturan", 2)]
SIDEBAR_W, SIDEBAR_W_COLLAPSED = 200, 64


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"{APP_NAME} — {APP_SUBTITLE} · {ctx.profile}")
        self.setWindowIcon(logo_icon())
        self.resize(1480, 920)
        self.setMinimumSize(1100, 700)

        root = QtWidgets.QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._build_header())

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        self.stack = QtWidgets.QStackedWidget()
        self.inspect = InspectPage(ctx)
        self.training = TrainingPage(ctx)
        self.settings = SettingsPage(ctx)
        for p in (self.inspect, self.training, self.settings):
            self.stack.addWidget(p)
        body.addWidget(self.stack, 1)
        v.addLayout(body, 1)

        # Shortcut keyboard (aktif sesuai halaman)
        self.sc_trigger = QShortcut(QtGui.QKeySequence(Qt.Key_Space), self)
        self.sc_trigger.activated.connect(self.inspect.trigger)
        self.sc_ok = QShortcut(QtGui.QKeySequence(Qt.Key_1), self)
        self.sc_ok.activated.connect(self.training.btn_ok.click)
        self.sc_ng = QShortcut(QtGui.QKeySequence(Qt.Key_2), self)
        self.sc_ng.activated.connect(self.training.btn_ng.click)
        self.sc_sidebar = QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        self.sc_sidebar.activated.connect(self.toggle_sidebar)
        self._apply_sidebar(ctx.cfg.sidebar_collapsed, animate=False)

        # Worker kamera
        self.worker = CameraWorker(ctx.cfg, ctx.models)
        ctx.worker = self.worker
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_changed.connect(self._on_camera_status)
        self.worker.fps_changed.connect(self._on_fps)

        ctx.profile_changed.connect(self._on_profile_changed)
        self._refresh_profile_combo()
        ctx.config_changed.connect(self._on_config)
        ctx.model_changed.connect(self._on_model_changed)

        self._clock = QtCore.QTimer(self)
        self._clock.timeout.connect(self._tick)
        self._clock.start(1000)
        self._tick()

        self._on_model_changed()
        self.go(0)
        self.worker.start()

    # ------------------------------------------------------------------
    def _build_header(self):
        h = QtWidgets.QFrame()
        h.setObjectName("Header")
        h.setFixedHeight(64)
        lay = QtWidgets.QHBoxLayout(h)
        lay.setContentsMargins(12, 8, 20, 8)
        lay.setSpacing(10)
        self.btn_side = QtWidgets.QPushButton("☰")
        self.btn_side.setObjectName("SideToggle")
        self.btn_side.setFixedSize(40, 40)
        self.btn_side.setCursor(Qt.PointingHandCursor)
        self.btn_side.setToolTip("Sembunyikan / tampilkan menu (Ctrl+B)")
        self.btn_side.clicked.connect(self.toggle_sidebar)
        lay.addWidget(self.btn_side)
        logo = QtWidgets.QLabel()
        logo.setFixedSize(40, 40)
        logo.setPixmap(logo_pixmap(40, self.devicePixelRatioF()))
        lay.addWidget(logo)
        tbox = QtWidgets.QVBoxLayout()
        tbox.setSpacing(0)
        t = QtWidgets.QLabel(APP_NAME)
        t.setObjectName("AppTitle")
        s = QtWidgets.QLabel(APP_SUBTITLE)
        s.setObjectName("AppSubtitle")
        tbox.addWidget(t)
        tbox.addWidget(s)
        lay.addLayout(tbox)
        lay.addSpacing(18)
        plabel = QtWidgets.QLabel("PROFIL")
        plabel.setObjectName("ProfileLabel")
        lay.addWidget(plabel)
        self.cb_profile = QtWidgets.QComboBox()
        self.cb_profile.setObjectName("ProfileCombo")
        self.cb_profile.setMinimumWidth(190)
        self.cb_profile.setToolTip("Profil kamera aktif: kamera, ROI, dataset & model ikut berganti")
        self.cb_profile.activated.connect(self._on_profile_selected)
        lay.addWidget(self.cb_profile)
        lay.addStretch(1)
        self.chip_cam = Chip("Kamera: menghubungkan…", "warn")
        self.chip_model = Chip("Model: —")
        self.chip_fps = Chip("0.0 FPS")
        for c in (self.chip_cam, self.chip_model, self.chip_fps):
            lay.addWidget(c)
        lay.addSpacing(12)
        self.lbl_clock = QtWidgets.QLabel()
        self.lbl_clock.setObjectName("Clock")
        lay.addWidget(self.lbl_clock)
        return h

    def _build_sidebar(self):
        sb = QtWidgets.QFrame()
        sb.setObjectName("Sidebar")
        sb.setFixedWidth(SIDEBAR_W)
        self.sidebar = sb
        lay = QtWidgets.QVBoxLayout(sb)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(6)
        self.nav_group = QtWidgets.QButtonGroup(self)
        self.nav_buttons = []
        for icon, text, idx in NAV:
            b = button(f"{icon}   {text}", checkable=True)
            b.setProperty("nav", True)
            b.setProperty("navIcon", icon)
            b.setProperty("navText", text)
            b.clicked.connect(lambda _=False, i=idx: self.go(i))
            self.nav_group.addButton(b)
            self.nav_buttons.append(b)
            lay.addWidget(b)
        lay.addStretch(1)
        self.side_hint = label("Pintasan keyboard:\nSpasi  ·  inspeksi\n1 / 2  ·  simpan sampel\n"
                               "Ctrl+B  ·  menu\nF11   ·  layar penuh", "hint")
        lay.addWidget(self.side_hint)
        self.side_foot = QtWidgets.QLabel(f"v{__version__}")
        self.side_foot.setObjectName("SidebarFoot")
        lay.addWidget(self.side_foot)
        self._side_layout = lay
        return sb

    # ------------------------------------------------------------------
    NEW_PROFILE = "__new__"

    def _refresh_profile_combo(self):
        self.cb_profile.blockSignals(True)
        self.cb_profile.clear()
        for name in config.list_profiles():
            self.cb_profile.addItem(name, name)
        self.cb_profile.insertSeparator(self.cb_profile.count())
        self.cb_profile.addItem("＋  Profil baru…", self.NEW_PROFILE)
        self.cb_profile.setCurrentIndex(max(0, self.cb_profile.findData(self.ctx.profile)))
        self.cb_profile.blockSignals(False)

    def _on_profile_selected(self, idx):
        name = self.cb_profile.itemData(idx)
        if name == self.NEW_PROFILE:
            profile_actions.create(self, self.ctx)
        elif name:
            profile_actions.switch(self, self.ctx, name)
        self._refresh_profile_combo()

    def _on_profile_changed(self, name):
        self._refresh_profile_combo()
        cfg = self.ctx.cfg
        self.settings.reload(cfg)
        self.training.reload(cfg)
        self.inspect.reset_for_profile()
        self.setWindowTitle(f"{APP_NAME} — {APP_SUBTITLE} · {name}")

    def toggle_sidebar(self):
        collapsed = not self.ctx.cfg.sidebar_collapsed
        self.ctx.update(sidebar_collapsed=collapsed)
        self._apply_sidebar(collapsed, animate=True)

    def _apply_sidebar(self, collapsed, animate=True):
        """Ciutkan sidebar menjadi strip ikon (atau buka kembali), dengan animasi lebar."""
        target = SIDEBAR_W_COLLAPSED if collapsed else SIDEBAR_W
        for b in self.nav_buttons:
            icon, text = b.property("navIcon"), b.property("navText")
            b.setText(icon if collapsed else f"{icon}   {text}")
            b.setToolTip(text if collapsed else "")
            b.setProperty("collapsed", bool(collapsed))
            b.style().unpolish(b)
            b.style().polish(b)
        self.side_hint.setVisible(not collapsed)
        self.side_foot.setVisible(not collapsed)
        m = 8 if collapsed else 12
        self._side_layout.setContentsMargins(m, 16, m, 16)
        self.btn_side.setText("☰")
        if not animate:
            self.sidebar.setFixedWidth(target)
            return
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(self.sidebar.width())
        anim.setEndValue(target)
        anim.setDuration(180)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(lambda v: self.sidebar.setFixedWidth(int(v)))
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        self._side_anim = anim

    def go(self, idx):
        self.stack.setCurrentIndex(idx)
        self.nav_buttons[idx].setChecked(True)
        self.sc_trigger.setEnabled(idx == 0)
        self.sc_ok.setEnabled(idx == 1)
        self.sc_ng.setEnabled(idx == 1)
        if idx == 1:
            self.training.refresh_counts()

    # ------------------------------------------------------------------
    def _on_frame(self, qimg, res):
        try:
            self.inspect.on_result(qimg, res)
            self.training.on_result(qimg, res)
        finally:
            self.worker.frame_consumed()

    def _on_camera_status(self, connected, msg):
        if connected:
            self.chip_cam.set("Kamera: aktif", "ok")
        else:
            self.chip_cam.set("Kamera: tidak terhubung", "ng")
            self.chip_fps.set("0.0 FPS")
        self.chip_cam.setToolTip(msg)
        self.inspect.on_camera_status(connected, msg)
        self.training.on_camera_status(connected, msg)
        self.settings.on_camera_status(connected, msg)

    def _on_fps(self, fps):
        self.chip_fps.set(f"{fps:.1f} FPS")
        self.inspect.on_fps(fps)

    def _on_config(self, cfg):
        self.ctx.gpio.apply_config(cfg)
        self.inspect.on_config(cfg)

    def _on_model_changed(self):
        m = self.ctx.models
        if m.is_ready():
            self.chip_model.set(f"Model: {m.summary()}", "ok")
        else:
            self.chip_model.set(f"Model: {m.summary().lower()}", "warn")
        self.training.refresh_model_info()

    def _tick(self):
        self.lbl_clock.setText(time.strftime("%d %b %Y   %H:%M:%S"))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.worker.stop()
        if self.training._train is not None:
            self.training._train.cancel()
            self.training._train.wait(5000)
        self.ctx.gpio.close()
        self.ctx.flush()
        super().closeEvent(e)
