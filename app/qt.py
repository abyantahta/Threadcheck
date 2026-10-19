"""Lapisan kompatibilitas Qt.

Utama memakai PySide6. Jika tidak tersedia (mis. Raspberry Pi yang memasang
Qt lewat `sudo apt install python3-pyqt5`), otomatis memakai PyQt5.
Paksa pilihan dengan environment variable ULIR_QT_API=pyside6 / pyqt5.
"""
import importlib
import os
import sys

# Muat library berat SEBELUM Qt. PySide6 memasang hook impor yang membaca kode sumber
# setiap modul yang dimuat setelahnya; untuk library besar hal ini bisa membuat aplikasi
# lambat terbuka (bahkan puluhan detik).
for _mod in ("numpy", "cv2", "joblib", "onnxruntime"):
    try:
        importlib.import_module(_mod)
    except Exception:
        pass


def _disable_pyside_import_hook():
    """Matikan pemeriksaan `__feature__` PySide6 pada modul yang dimuat belakangan
    (mis. PyTorch/Ultralytics saat training YOLO). Aplikasi ini tidak memakai fitur itu."""
    feature = sys.modules.get("shibokensupport.feature")
    if feature is not None and hasattr(feature, "_mod_uses_pyside"):
        feature._mod_uses_pyside = lambda module: False


def _load_pyside6():
    from PySide6 import QtCore, QtGui, QtWidgets
    _disable_pyside_import_hook()
    return QtCore, QtGui, QtWidgets, QtCore.Signal, "PySide6"


def _load_pyqt5():
    from PyQt5 import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets, QtCore.pyqtSignal, "PyQt5"


_order = [_load_pyside6, _load_pyqt5]
if os.environ.get("ULIR_QT_API", "").lower() == "pyqt5":
    _order.reverse()

for _loader in _order:
    try:
        QtCore, QtGui, QtWidgets, Signal, QT_API = _loader()
        break
    except ImportError:
        continue
else:
    raise ImportError(
        "PySide6 atau PyQt5 tidak ditemukan. Install dengan: pip install PySide6 "
        "(atau di Raspberry Pi: sudo apt install python3-pyqt5)"
    )

Qt = QtCore.Qt
QShortcut = getattr(QtGui, "QShortcut", None) or QtWidgets.QShortcut


def exec_app(app):
    return app.exec() if hasattr(app, "exec") else app.exec_()
