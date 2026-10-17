"""Simpan logo Threck sebagai file gambar (PNG 512 px dan ICO) untuk shortcut/dokumen.

Jalankan:  python tools/export_logo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.qt import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from app.ui.logo import logo_pixmap  # noqa: E402

out = ROOT / "logo"
out.mkdir(exist_ok=True)
logo_pixmap(512).save(str(out / "threck_logo.png"))
logo_pixmap(256).save(str(out / "threck.ico"))
print(f"Logo disimpan di: {out}")
