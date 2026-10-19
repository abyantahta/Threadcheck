"""Tema gelap ala HMI industri."""
from pathlib import Path

C = {
    "bg": "#0d1117",
    "sidebar": "#10161d",
    "surface": "#161d26",
    "surface2": "#1e2732",
    "surface3": "#263241",
    "border": "#2a3644",
    "text": "#e6edf3",
    "muted": "#8b98a6",
    "accent": "#3b82f6",
    "accent_hover": "#5294ff",
    "ok": "#22c55e",
    "ok_dark": "#0f3d24",
    "ng": "#ef4444",
    "ng_dark": "#4a1717",
    "warn": "#f59e0b",
    "warn_dark": "#45300a",
}

QSS = f"""
* {{
    color: {C['text']};
    outline: none;
}}
QMainWindow, QWidget#Root, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{
    background: {C['bg']};
}}
QToolTip {{
    background: {C['surface3']}; color: {C['text']};
    border: 1px solid {C['border']}; padding: 6px; border-radius: 6px;
}}

/* ---------- Header ---------- */
QFrame#Header {{
    background: {C['sidebar']};
    border-bottom: 1px solid {C['border']};
}}
QLabel#AppTitle {{ font-size: 19px; font-weight: 700; }}
QLabel#AppSubtitle {{ color: {C['muted']}; font-size: 12px; }}
QLabel#Clock {{ color: {C['muted']}; font-size: 13px; font-weight: 600; }}
QLabel[chip="true"] {{
    background: {C['surface2']}; border: 1px solid {C['border']};
    border-radius: 13px; padding: 4px 12px; font-size: 12px; font-weight: 600;
}}
QLabel[chip="true"][tone="ok"] {{ background: {C['ok_dark']}; border-color: {C['ok']}; color: #bbf7d0; }}
QLabel[chip="true"][tone="ng"] {{ background: {C['ng_dark']}; border-color: {C['ng']}; color: #fecaca; }}
QLabel[chip="true"][tone="warn"] {{ background: {C['warn_dark']}; border-color: {C['warn']}; color: #fde68a; }}

/* ---------- Sidebar ---------- */
QFrame#Sidebar {{
    background: {C['sidebar']};
    border-right: 1px solid {C['border']};
}}
QPushButton[nav="true"] {{
    text-align: left; padding: 12px 16px; border: none; border-radius: 8px;
    background: transparent; color: {C['muted']}; font-size: 14px; font-weight: 600;
}}
QPushButton[nav="true"]:hover {{ background: {C['surface']}; color: {C['text']}; }}
QPushButton[nav="true"]:checked {{
    background: {C['surface2']}; color: {C['text']};
    border-left: 3px solid {C['accent']};
}}
QPushButton[nav="true"][collapsed="true"] {{
    text-align: center; padding: 12px 0px; font-size: 17px;
}}
QLabel#ProfileLabel {{ color: {C['muted']}; font-size: 10px; font-weight: 700; letter-spacing: 1px; }}
QComboBox#ProfileCombo {{
    background: {C['surface2']}; border: 1px solid {C['accent']}; border-radius: 13px;
    padding: 4px 28px 4px 12px; font-size: 13px; font-weight: 700; min-height: 18px;
}}
QComboBox#ProfileCombo:hover {{ background: {C['surface3']}; }}
QPushButton#SideToggle {{
    background: transparent; border: none; border-radius: 8px;
    font-size: 20px; color: {C['muted']}; padding: 0px;
}}
QPushButton#SideToggle:hover {{ background: {C['surface2']}; color: {C['text']}; }}
QLabel#SidebarFoot {{ color: {C['muted']}; font-size: 11px; }}

/* ---------- Card ---------- */
QFrame#Card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
}}
QLabel[role="cardTitle"] {{
    color: {C['muted']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;
}}
QLabel[role="muted"] {{ color: {C['muted']}; }}
QLabel[role="value"] {{ font-size: 14px; font-weight: 600; }}
QLabel[role="hint"] {{ color: {C['muted']}; font-size: 12px; }}
QLabel[role="status"] {{ font-size: 12px; padding: 6px 10px; border-radius: 6px;
    background: {C['surface2']}; }}
QLabel[role="status"][tone="ok"] {{ background: {C['ok_dark']}; color: #bbf7d0; }}
QLabel[role="status"][tone="ng"] {{ background: {C['ng_dark']}; color: #fecaca; }}
QLabel[role="status"][tone="warn"] {{ background: {C['warn_dark']}; color: #fde68a; }}

/* ---------- Video ---------- */
QFrame#VideoFrame {{
    background: #05080b; border: 1px solid {C['border']}; border-radius: 12px;
}}

/* ---------- Judgement ---------- */
QFrame#Judgement {{
    border-radius: 14px; border: 2px solid {C['border']}; background: {C['surface2']};
}}
QFrame#Judgement[state="OK"] {{
    border-color: {C['ok']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #14532d, stop:1 #0b2e1a);
}}
QFrame#Judgement[state="NG"] {{
    border-color: {C['ng']};
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7f1d1d, stop:1 #3b0d0d);
}}
QFrame#Judgement[state="WAIT"] {{
    border-color: {C['warn']}; background: {C['warn_dark']};
}}
QLabel#JudgeText {{ font-size: 76px; font-weight: 800; letter-spacing: 4px; }}
QLabel#JudgeSub {{ font-size: 14px; font-weight: 600; color: #d6dde5; }}

/* ---------- Stat tile ---------- */
QFrame#Stat {{
    background: {C['surface2']}; border-radius: 10px; border: 1px solid {C['border']};
}}
QLabel#StatValue {{ font-size: 22px; font-weight: 700; }}
QLabel#StatValue[tone="ok"] {{ color: {C['ok']}; }}
QLabel#StatValue[tone="ng"] {{ color: {C['ng']}; }}
QLabel#StatValue[tone="accent"] {{ color: {C['accent_hover']}; }}
QLabel#StatTitle {{ color: {C['muted']}; font-size: 11px; font-weight: 700; }}

/* ---------- Tombol ---------- */
QPushButton {{
    background: {C['surface2']}; border: 1px solid {C['border']}; border-radius: 8px;
    padding: 8px 14px; font-size: 13px; font-weight: 600;
}}
QPushButton:hover {{ background: {C['surface3']}; }}
QPushButton:pressed {{ background: {C['border']}; }}
QPushButton:disabled {{ color: #5b6672; background: {C['surface']}; }}
QPushButton[variant="primary"] {{ background: {C['accent']}; border-color: {C['accent']}; color: white; }}
QPushButton[variant="primary"]:hover {{ background: {C['accent_hover']}; }}
QPushButton[variant="ok"] {{ background: #15803d; border-color: {C['ok']}; color: white; }}
QPushButton[variant="ok"]:hover {{ background: #16a34a; }}
QPushButton[variant="ng"] {{ background: #b91c1c; border-color: {C['ng']}; color: white; }}
QPushButton[variant="ng"]:hover {{ background: #dc2626; }}
QPushButton[variant]:disabled {{ background: {C['surface']}; border-color: {C['border']}; color: #5b6672; }}
QPushButton[size="xl"] {{ font-size: 18px; font-weight: 800; padding: 16px; border-radius: 10px; }}
QPushButton[size="lg"] {{ font-size: 15px; font-weight: 700; padding: 12px; }}
QPushButton[segment="true"] {{
    background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 0px; padding: 9px 10px;
    color: {C['muted']};
}}
QPushButton[segment="true"][pos="left"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton[segment="true"][pos="right"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QPushButton[segment="true"]:checked {{ background: {C['accent']}; border-color: {C['accent']}; color: white; }}

/* ---------- Input ---------- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;
    padding: 6px 8px; min-height: 18px; selection-background-color: {C['accent']};
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: #3a4a5c; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {C['accent']}; }}
QComboBox QAbstractItemView {{
    background: {C['surface2']}; border: 1px solid {C['border']};
    selection-background-color: {C['accent']}; padding: 4px;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 4px; border: 1px solid #3a4a5c; background: {C['bg']};
}}
QCheckBox::indicator:checked {{ background: {C['accent']}; border-color: {C['accent']}; }}
QSlider::groove:horizontal {{ height: 6px; background: {C['bg']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {C['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: white; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
}}

/* ---------- Progress ---------- */
QProgressBar {{
    background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 6px;
    text-align: center; height: 18px; font-size: 11px; font-weight: 600;
}}
QProgressBar::chunk {{ background: {C['accent']}; border-radius: 5px; }}
QProgressBar[tone="ok"]::chunk {{ background: {C['ok']}; }}
QProgressBar[tone="ng"]::chunk {{ background: {C['ng']}; }}

/* ---------- Tabel / list ---------- */
QTableWidget, QListWidget, QPlainTextEdit {{
    background: {C['bg']}; border: 1px solid {C['border']}; border-radius: 8px;
    gridline-color: transparent; alternate-background-color: #121821;
}}
QTableWidget::item, QListWidget::item {{ padding: 4px; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: #1e3a5f; }}
QHeaderView::section {{
    background: {C['surface2']}; color: {C['muted']}; border: none;
    padding: 6px; font-size: 11px; font-weight: 700;
}}
QPlainTextEdit {{ font-family: "Consolas", "DejaVu Sans Mono", monospace; font-size: 12px; padding: 6px; }}

/* ---------- Scrollbar ---------- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {C['surface3']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3a4a5c; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {C['surface3']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMessageBox, QInputDialog, QFileDialog {{ background: {C['surface']}; }}

/* ---------- Mode ringkas (layar 800x480) ---------- */
QWidget#Root[compact="true"] QLabel#AppTitle {{ font-size: 15px; }}
QWidget#Root[compact="true"] QLabel[chip="true"] {{ padding: 2px 8px; font-size: 11px; border-radius: 10px; }}
QWidget#Root[compact="true"] QComboBox#ProfileCombo {{ padding: 2px 24px 2px 8px; font-size: 12px; }}
QWidget#Root[compact="true"] QLabel#JudgeText {{ font-size: 40px; letter-spacing: 2px; }}
QWidget#Root[compact="true"] QLabel#JudgeSub {{ font-size: 11px; }}
QWidget#Root[compact="true"] QLabel#StatValue {{ font-size: 16px; }}
QWidget#Root[compact="true"] QLabel#StatTitle {{ font-size: 9px; }}
QWidget#Root[compact="true"] QLabel[role="cardTitle"] {{ font-size: 10px; }}
QWidget#Root[compact="true"] QLabel[role="hint"] {{ font-size: 10px; }}
QWidget#Root[compact="true"] QLabel[role="value"] {{ font-size: 12px; }}
QWidget#Root[compact="true"] QPushButton {{ padding: 5px 8px; font-size: 12px; }}
QWidget#Root[compact="true"] QPushButton[size="xl"] {{ font-size: 14px; padding: 8px; }}
QWidget#Root[compact="true"] QPushButton[size="lg"] {{ font-size: 12px; padding: 6px; }}
QWidget#Root[compact="true"] QPushButton[segment="true"] {{ padding: 5px 6px; }}
QWidget#Root[compact="true"] QPushButton[nav="true"][collapsed="true"] {{ padding: 8px 0px; font-size: 15px; }}
QWidget#Root[compact="true"] QComboBox, QWidget#Root[compact="true"] QSpinBox,
QWidget#Root[compact="true"] QDoubleSpinBox, QWidget#Root[compact="true"] QLineEdit {{ padding: 3px 6px; }}
QWidget#Root[compact="true"] QFrame#Card {{ border-radius: 8px; }}
QWidget#Root[compact="true"] QFrame#Judgement {{ border-radius: 10px; }}
"""

ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _make_assets():
    """Gambar ikon kecil (panah, centang) sekali saja, dipakai oleh stylesheet."""
    from ..qt import QtCore, QtGui, Qt
    ASSET_DIR.mkdir(exist_ok=True)

    def draw(name, points, width=2.0, closed=True):
        path = ASSET_DIR / name
        if path.exists():
            return
        pm = QtGui.QPixmap(24, 24)
        pm.fill(Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        poly = QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in points])
        if closed:
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor("#aab6c3"))
            p.drawPolygon(poly)
        else:
            pen = QtGui.QPen(QtGui.QColor("white"), width * 1.6)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(poly)
        p.end()
        pm.save(str(path))

    draw("arrow_up.png", [(6, 15), (18, 15), (12, 8)])
    draw("arrow_down.png", [(6, 9), (18, 9), (12, 16)])
    draw("check.png", [(5, 12.5), (10, 17.5), (19, 7)], closed=False)


def build_qss():
    try:
        _make_assets()
    except Exception:
        return QSS
    a = ASSET_DIR.as_posix()
    return QSS + f"""
QSpinBox, QDoubleSpinBox {{ padding-right: 26px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 24px;
    border-left: 1px solid {C['border']}; border-top-right-radius: 6px; background: {C['surface2']};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 24px;
    border-left: 1px solid {C['border']}; border-bottom-right-radius: 6px; background: {C['surface2']};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {C['surface3']}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url("{a}/arrow_up.png"); width: 12px; height: 12px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url("{a}/arrow_down.png"); width: 12px; height: 12px; }}
QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: none; }}
QComboBox {{ padding-right: 26px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: top right; width: 24px; border: none;
}}
QComboBox::down-arrow {{ image: url("{a}/arrow_down.png"); width: 12px; height: 12px; }}
QCheckBox::indicator:checked {{ image: url("{a}/check.png"); }}
"""

