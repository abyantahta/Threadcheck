"""Threck — aplikasi inspeksi ulir lubang berbasis kamera.

Jalankan:  python main.py
"""
import os
import sys

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # sembunyikan warning saat scan kamera

from app.qt import QT_API, QtCore, QtWidgets, exec_app


def main():
    if QT_API == "PyQt5":
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    from app.config import APP_NAME, ensure_dirs
    from app.context import AppContext
    from app.ui.main_window import MainWindow
    from app.ui.style import build_qss

    if sys.platform.startswith("win"):
        # agar taskbar Windows memakai ikon Threck, bukan ikon Python
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Threck.ThreadCheck")
        except Exception:
            pass
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    font = app.font()
    if sys.platform.startswith("win"):
        font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(build_qss())
    from app.ui.logo import logo_icon
    app.setWindowIcon(logo_icon())

    ctx = AppContext()
    ensure_dirs()
    win = MainWindow(ctx)
    if "--fullscreen" in sys.argv:
        win.showFullScreen()
    else:
        win.show()
    return exec_app(app)


if __name__ == "__main__":
    sys.exit(main())
