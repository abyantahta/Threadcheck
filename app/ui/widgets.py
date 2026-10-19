"""Komponen UI yang dipakai ulang."""
import cv2
import numpy as np

from ..qt import QtCore, QtGui, QtWidgets, Qt, Signal


def repolish(w):
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


def set_prop(w, name, value):
    if w.property(name) != value:
        w.setProperty(name, value)
        repolish(w)


def button(text, variant=None, size=None, tooltip=None, checkable=False):
    b = QtWidgets.QPushButton(text)
    b.setCursor(Qt.PointingHandCursor)
    if variant:
        b.setProperty("variant", variant)
    if size:
        b.setProperty("size", size)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCheckable(checkable)
    return b


def label(text="", role=None, wrap=False):
    lb = QtWidgets.QLabel(text)
    if role:
        lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def gray_to_pixmap(img: np.ndarray) -> QtGui.QPixmap:
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        q = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        q = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q.copy())


class ValueLabel(QtWidgets.QLabel):
    """Label nilai satu baris dengan tinggi & lebar stabil.

    Teks yang terlalu panjang dipotong dengan "…" (teks lengkap di tooltip), sehingga
    perubahan nilai live tidak menggeser tata letak panel.
    """

    def __init__(self, text="—", role="value"):
        super().__init__()
        self._full = ""
        self.setProperty("role", role)
        self.setWordWrap(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(20)
        self.setText(text)

    def setText(self, text):
        if text == self._full:
            return
        self._full = text
        self.setToolTip(text)
        self._apply()

    def _apply(self):
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, Qt.ElideRight, max(20, self.width())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self._apply()

    def sizeHint(self):
        return QtCore.QSize(60, self.fontMetrics().height() + 4)

    def minimumSizeHint(self):
        return QtCore.QSize(20, self.fontMetrics().height() + 4)


class Card(QtWidgets.QFrame):
    """Panel dengan judul kecil di atas."""

    def __init__(self, title=None, parent=None, spacing=10):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(spacing)
        self.header = QtWidgets.QHBoxLayout()
        self.header.setSpacing(8)
        if title:
            self.title = label(title.upper(), "cardTitle")
            self.header.addWidget(self.title)
        self.header.addStretch(1)
        lay.addLayout(self.header)
        self.body = lay


class Chip(QtWidgets.QLabel):
    def __init__(self, text="", tone=None):
        super().__init__(text)
        self.setProperty("chip", True)
        if tone:
            self.setProperty("tone", tone)

    def set(self, text, tone=None):
        self.setText(text)
        set_prop(self, "tone", tone or "")


class StatTile(QtWidgets.QFrame):
    def __init__(self, title, tone=None):
        super().__init__()
        self.setObjectName("Stat")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(0)
        self.value = QtWidgets.QLabel("0")
        self.value.setObjectName("StatValue")
        if tone:
            self.value.setProperty("tone", tone)
        t = QtWidgets.QLabel(title.upper())
        t.setObjectName("StatTitle")
        lay.addWidget(self.value)
        lay.addWidget(t)

    def set(self, text):
        self.value.setText(str(text))


class JudgementCard(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Judgement")
        self.setMinimumHeight(190)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 14)
        lay.setSpacing(2)
        self.text = QtWidgets.QLabel("—")
        self.text.setObjectName("JudgeText")
        self.text.setAlignment(Qt.AlignCenter)
        self.sub = QtWidgets.QLabel("Menunggu")
        self.sub.setObjectName("JudgeSub")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setWordWrap(True)
        lay.addStretch(1)
        lay.addWidget(self.text)
        lay.addWidget(self.sub)
        lay.addStretch(1)

    def set_state(self, state, text, sub):
        set_prop(self, "state", state)
        self.text.setText(text)
        self.sub.setText(sub)


class ScoreBar(QtWidgets.QWidget):
    """Bar skor 0..1 dengan penanda ambang batas."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(26)
        self.score = None
        self.threshold = 0.5

    def set(self, score, threshold):
        self.score, self.threshold = score, threshold
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(1, 4, -1, -4)
        p.setPen(QtGui.QPen(QtGui.QColor("#2a3644"), 1))
        p.setBrush(QtGui.QColor("#0d1117"))
        p.drawRoundedRect(r, 6, 6)
        if self.score is not None:
            fill = QtCore.QRectF(r)
            fill.setWidth(max(0.0, r.width() * float(np.clip(self.score, 0, 1))))
            color = "#22c55e" if self.score >= self.threshold else "#ef4444"
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(color))
            p.drawRoundedRect(fill, 6, 6)
        x = r.left() + r.width() * self.threshold
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        p.drawLine(QtCore.QPointF(x, r.top() - 3), QtCore.QPointF(x, r.bottom() + 3))
        p.setPen(QtGui.QColor("#ffffff"))
        f = p.font()
        f.setBold(True)
        f.setPointSize(9)
        p.setFont(f)
        txt = "—" if self.score is None else f"{self.score * 100:.1f}%"
        p.drawText(r, Qt.AlignCenter, txt)
        p.end()


class Segmented(QtWidgets.QWidget):
    changed = Signal(str)

    def __init__(self, options):
        super().__init__()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for i, (key, text) in enumerate(options):
            b = button(text, checkable=True)
            b.setProperty("segment", True)
            b.setProperty("pos", "left" if i == 0 else ("right" if i == len(options) - 1 else "mid"))
            self.group.addButton(b)
            lay.addWidget(b, 1)
            self.buttons[key] = b
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))

    def set_value(self, key):
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def value(self):
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return None


class VideoView(QtWidgets.QWidget):
    """Menampilkan live video, dan pengguna dapat menyeret kotak untuk mengatur ROI."""

    roi_selected = Signal(tuple)

    def __init__(self, placeholder="Kamera belum terhubung"):
        super().__init__()
        self.setMinimumSize(420, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)
        self._img = None
        self._placeholder = placeholder
        self._drag_start = None
        self._drag_end = None
        self.badge = ""

    def set_image(self, img):
        self._img = img
        if self.isVisible():
            self.update()

    def set_placeholder(self, text):
        self._placeholder = text
        self._img = None
        self.update()

    def _target(self):
        if self._img is None or self._img.isNull():
            return None
        iw, ih = self._img.width(), self._img.height()
        W, H = self.width(), self.height()
        s = min(W / iw, H / ih)
        w, h = iw * s, ih * s
        return QtCore.QRectF((W - w) / 2, (H - h) / 2, w, h)

    @staticmethod
    def _pos(e):
        return e.position() if hasattr(e, "position") else QtCore.QPointF(e.pos())

    def mousePressEvent(self, e):
        t = self._target()
        if e.button() == Qt.LeftButton and t is not None and t.contains(self._pos(e)):
            self._drag_start = self._drag_end = self._pos(e)
            self.update()

    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            self._drag_end = self._pos(e)
            self.update()

    def mouseReleaseEvent(self, e):
        if self._drag_start is None:
            return
        t = self._target()
        a, b = self._drag_start, self._pos(e)
        self._drag_start = self._drag_end = None
        self.update()
        if t is None:
            return
        x0 = (min(a.x(), b.x()) - t.left()) / t.width()
        y0 = (min(a.y(), b.y()) - t.top()) / t.height()
        x1 = (max(a.x(), b.x()) - t.left()) / t.width()
        y1 = (max(a.y(), b.y()) - t.top()) / t.height()
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(1.0, x1), min(1.0, y1)
        if x1 - x0 > 0.03 and y1 - y0 > 0.03:
            self.roi_selected.emit((x0, y0, x1 - x0, y1 - y0))

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor("#05080b"))
        t = self._target()
        if t is None:
            p.setPen(QtGui.QColor("#5b6672"))
            f = p.font()
            f.setPointSize(13)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self._placeholder)
            p.end()
            return
        p.drawImage(t, self._img)

        if self._drag_start is not None and self._drag_end is not None:
            r = QtCore.QRectF(self._drag_start, self._drag_end).normalized()
            p.setBrush(QtGui.QColor(59, 130, 246, 50))
            pen = QtGui.QPen(QtGui.QColor("#5294ff"), 2, Qt.DashLine)
            p.setPen(pen)
            p.drawRect(r)

        # badge kiri atas
        if self.badge:
            f = p.font()
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            fm = QtGui.QFontMetrics(f)
            w = fm.horizontalAdvance(self.badge) + 20
            box = QtCore.QRectF(t.left() + 10, t.top() + 10, w, 24)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 160))
            p.drawRoundedRect(box, 12, 12)
            p.setPen(QtGui.QColor("#e6edf3"))
            p.drawText(box, Qt.AlignCenter, self.badge)

        hint = "Seret pada gambar untuk mengatur area inspeksi (ROI)"
        f = p.font()
        f.setPointSize(8)
        f.setBold(False)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        w = fm.horizontalAdvance(hint) + 18
        box = QtCore.QRectF(t.left() + 10, t.bottom() - 32, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor(0, 0, 0, 140))
        p.drawRoundedRect(box, 11, 11)
        p.setPen(QtGui.QColor("#aab6c3"))
        p.drawText(box, Qt.AlignCenter, hint)
        p.end()


def video_frame(view: VideoView) -> QtWidgets.QFrame:
    """Bungkus VideoView dengan bingkai membulat."""
    fr = QtWidgets.QFrame()
    fr.setObjectName("VideoFrame")
    lay = QtWidgets.QVBoxLayout(fr)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.addWidget(view)
    return fr


def scroll_panel(widget, width=None):
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sa.setWidget(widget)
    if width:
        sa.setFixedWidth(width)
    return sa
