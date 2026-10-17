"""Logo Threck — Thread Check.

Mur heksagonal (komponen berulir), lubang dengan garis ulir miring (seperti yang dilihat
kamera dari sudut miring), dan lencana centang hijau (lolos inspeksi). Digambar sebagai
vektor dengan QPainter agar tajam di semua ukuran tanpa dependensi tambahan.
"""
import math

from ..qt import QtCore, QtGui, Qt

BLUE_TOP = "#3b82f6"
BLUE_BOTTOM = "#1e3a8a"
GREEN = "#22c55e"


def paint_logo(p: QtGui.QPainter, size: float, x: float = 0.0, y: float = 0.0):
    s = float(size)
    p.save()
    p.translate(x, y)
    p.setRenderHint(QtGui.QPainter.Antialiasing)

    # latar: kotak membulat dengan gradasi biru
    grad = QtGui.QLinearGradient(0, 0, s, s)
    grad.setColorAt(0.0, QtGui.QColor(BLUE_TOP))
    grad.setColorAt(1.0, QtGui.QColor(BLUE_BOTTOM))
    p.setPen(Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(QtCore.QRectF(0, 0, s, s), s * 0.23, s * 0.23)

    cx, cy = s * 0.47, s * 0.47
    white = QtGui.QColor("white")

    # mur heksagonal
    R = s * 0.31
    hexagon = QtGui.QPolygonF([
        QtCore.QPointF(cx + R * math.cos(math.radians(60 * k)), cy + R * math.sin(math.radians(60 * k)))
        for k in range(6)
    ])
    pen = QtGui.QPen(white, s * 0.06)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPolygon(hexagon)

    # lubang + garis ulir miring (dipotong di dalam lubang)
    r = s * 0.165
    hole = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
    p.setPen(Qt.NoPen)
    p.setBrush(QtGui.QColor("#0b1e4f"))
    p.drawEllipse(hole)
    if s >= 24:
        clip = QtGui.QPainterPath()
        clip.addEllipse(hole)
        p.save()
        p.setClipPath(clip)
        tpen = QtGui.QPen(QtGui.QColor("#93c5fd"), s * 0.026)
        tpen.setCapStyle(Qt.FlatCap)
        p.setPen(tpen)
        slope = 0.30 * r
        gap = r * 0.44
        for k in (-2, -1, 0, 1, 2):
            yy = cy + k * gap
            p.drawLine(QtCore.QPointF(cx - r * 1.2, yy + slope), QtCore.QPointF(cx + r * 1.2, yy - slope))
        p.restore()
    ring = QtGui.QPen(white, s * 0.04)
    p.setPen(ring)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(hole)

    # lencana centang
    br = s * 0.17
    bx, by = s * 0.775, s * 0.775
    p.setPen(QtGui.QPen(QtGui.QColor(BLUE_BOTTOM), s * 0.035))
    p.setBrush(QtGui.QColor(GREEN))
    p.drawEllipse(QtCore.QPointF(bx, by), br, br)
    cpen = QtGui.QPen(white, s * 0.045)
    cpen.setCapStyle(Qt.RoundCap)
    cpen.setJoinStyle(Qt.RoundJoin)
    p.setPen(cpen)
    p.setBrush(Qt.NoBrush)
    check = QtGui.QPolygonF([
        QtCore.QPointF(bx - br * 0.48, by + br * 0.02),
        QtCore.QPointF(bx - br * 0.12, by + br * 0.38),
        QtCore.QPointF(bx + br * 0.50, by - br * 0.34),
    ])
    p.drawPolyline(check)
    p.restore()


def logo_pixmap(size: int, dpr: float = 1.0) -> QtGui.QPixmap:
    px = int(round(size * dpr))
    pm = QtGui.QPixmap(px, px)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    paint_logo(p, px)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def logo_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(logo_pixmap(s))
    return icon
