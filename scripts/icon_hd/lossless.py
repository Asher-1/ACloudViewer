"""v8 lossless pixel-exact SVG transcription (ScalablePixels paradigm).

Pure functions: no paths, no Qt application object, no side effects.
The caller (icon_hd.py) owns the QGuiApplication and every file write.

Gate contract: the transcribed SVG must raster within <=2/255 of the original
PNG through the SAME Qt paint pipeline (1x on 4 backgrounds + 2/3/4x HiDPI).
The residual +-1-2 comes only from Qt's 8-bit premultiplied round-trip, which
affects ANY raster including the original PNG itself.
"""
import numpy as np
from PySide6 import QtCore, QtGui, QtSvg

# per-channel tolerance of the Qt-pipeline SVG-vs-original gate (see verify)
VERIFY_TOL = 2


def qimage_rgba(img):
    """QImage -> (h, w, 4) int64 RGBA (ARGB32 memory order is BGRA)."""
    img = img.convertToFormat(QtGui.QImage.Format_ARGB32)
    w, h = img.width(), img.height()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4]
    return arr.reshape(h, w, 4)[:, :, [2, 1, 0, 3]].astype(np.int64)


def transcribe_lossless(rgba):
    """Every source pixel with alpha>=1 becomes an exact rect; horizontal runs
    of identical RGBA merge, then identical stacked runs merge vertically.
    No smoothing, no alpha skipping, no edge extension: the SVG composite is
    information-identical to the source at 1x and crisp at every integer scale.
    Returns (svg_text, rect_count)."""
    h, w = rgba.shape[:2]
    # horizontal runs of exact RGBA (alpha==0 carries no visible information)
    rows = []
    for y in range(h):
        runs = []
        x = 0
        while x < w:
            c = tuple(rgba[y, x])
            if c[3] == 0:
                x += 1
                continue
            run = 1
            while x + run < w and tuple(rgba[y, x + run]) == c:
                run += 1
            runs.append((x, run, c))
            x += run
        rows.append(runs)
    # vertical merge: identical (x, run, rgba) runs on consecutive rows
    rects = []
    active = {}
    for y, runs in enumerate(rows):
        seen = set()
        for x, run, c in runs:
            key = (x, run, c)
            seen.add(key)
            if key in active:
                active[key][1] += 1
            else:
                active[key] = [y, 1]
        for key in list(active):
            if key not in seen:
                y0, hh = active.pop(key)
                rects.append((key[0], y0, key[1], hh, key[2]))
    for key, (y0, hh) in active.items():
        rects.append((key[0], y0, key[1], hh, key[2]))
    rects.sort(key=lambda r: (r[1], r[0]))
    out = []
    for x, y, rw, rh, c in rects:
        fill = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
        if c[3] == 255:
            out.append(f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" fill="{fill}"/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{rw}" height="{rh}" '
                       f'fill="{fill}" fill-opacity="{c[3] / 255.0:.6f}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">\n'
           f'<!-- Lossless pixel-exact transcription v8 ({w}x{h}, {len(out)} rects): '
           f'every source pixel preserved, crisp at any scale -->\n'
           + '\n'.join(out) + '\n</svg>\n')
    return svg, len(out)


def render_rgba(svg_path, w, h, bg, scale=1):
    """Rasterize an SVG file through Qt; bg=None renders on transparency."""
    r = QtSvg.QSvgRenderer(str(svg_path))
    if not r.isValid():
        return None
    img = QtGui.QImage(w * scale, h * scale, QtGui.QImage.Format_ARGB32)
    if bg is None:
        img.fill(QtGui.QColor(0, 0, 0, 0))
    else:
        img.fill(QtGui.QColor(*bg, 255))
    p = QtGui.QPainter(img)
    r.render(p)
    p.end()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
        h * scale, img.bytesPerLine())[:, :w * scale * 4]
    return arr.reshape(h * scale, w * scale, 4)[:, :, [2, 1, 0, 3]].astype(np.int64)


def composite(src, bg):
    """numpy source-over of straight-alpha RGBA onto an opaque background."""
    a = src[:, :, 3:4] / 255.0
    return np.round(src[:, :, :3] * a + np.array(bg) * (1 - a)).astype(np.int64)


def render_orig(src_path, w, h, bg, scale=1):
    """The original PNG through the SAME Qt paint pipeline the app uses."""
    img = QtGui.QImage(w * scale, h * scale, QtGui.QImage.Format_ARGB32)
    img.fill(QtGui.QColor(*bg, 255))
    p = QtGui.QPainter(img)
    src_img = QtGui.QImage(str(src_path))
    if scale > 1:
        src_img = src_img.scaled(w * scale, h * scale,
                                 QtCore.Qt.AspectRatioMode.IgnoreAspectRatio,
                                 QtCore.Qt.TransformationMode.FastTransformation)
    p.drawImage(0, 0, src_img)
    p.end()
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
        h * scale, img.bytesPerLine())[:, :w * scale * 4]
    return arr.reshape(h * scale, w * scale, 4)[:, :, [2, 1, 0, 3]].astype(np.int64)


def verify(svg_path, src, src_path):
    """Hard gate: SVG raster vs original-PNG raster in the SAME Qt pipeline,
    per pixel, on 4 backgrounds at 1x and 2/3/4x (nearest).
    Returns (worst1, worst_hi, worst_vs, worst_hi_vs):
      worst*_vs = vs the original PNG (the publish gate, <=VERIFY_TOL),
      worst*    = vs the numpy composite (info only: Qt premul round-trip).
    Returns None if the SVG fails to load."""
    h, w = src.shape[:2]
    worst1 = worst_vs = 0
    for bg in ((255, 255, 255), (128, 128, 128), (0, 0, 0), (30, 60, 90)):
        ren = render_rgba(svg_path, w, h, bg)
        org = render_orig(src_path, w, h, bg)
        if ren is None:
            return None
        worst1 = max(worst1, int(np.abs(ren[:, :, :3] - composite(src, bg)).max()))
        worst_vs = max(worst_vs, int(np.abs(ren[:, :, :3] - org[:, :, :3]).max()))
    worst_hi = worst_hi_vs = 0
    for k in (2, 3, 4):
        for bg in ((255, 255, 255), (40, 40, 40)):
            ren = render_rgba(svg_path, w, h, bg, scale=k)
            org = render_orig(src_path, w, h, bg, scale=k)
            if ren is None:
                return None
            truth = np.kron(composite(src, bg), np.ones((k, k, 1), dtype=np.int64))
            worst_hi = max(worst_hi, int(np.abs(ren[:, :, :3] - truth).max()))
            worst_hi_vs = max(worst_hi_vs, int(np.abs(ren[:, :, :3] - org[:, :, :3]).max()))
    return worst1, worst_hi, worst_vs, worst_hi_vs