"""Gallery builder: orig / v8 lossless / v10 commercial tri-view + drawer.

Single self-contained HTML (all payloads base64-inlined) rendered from
gallery_template.html. v8 metrics come from the live lossless transcription
report (v8 is a pure function of the orig fixture, so no backup dir is
needed); v10 payloads are the published repo SVGs plus data/report.json.
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui

import lossless
import redesign
import fit


def rast(img):
    arr = np.frombuffer(img.constBits(), dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine())[:, :img.width() * 4]
    return arr.reshape(img.height(), img.width(), 4)[:, :, [2, 1, 0, 3]].astype(int)


def heat_b64(svg_path, src_img, w, h, win=None):
    """Per-pixel |SVG - orig| on white, as a grayscale PNG data payload.
    The published render is inverse-mapped through the declared fit window
    so the heat map proves fidelity modulo the canvas-fit pass."""
    o = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
    o.fill(QtGui.QColor(255, 255, 255, 255))
    pt = QtGui.QPainter(o)
    pt.drawImage(0, 0, src_img)
    pt.end()
    s = fit.align_render(svg_path, w, h, (255, 255, 255), win=win)
    d = np.abs(s[:, :, :3] - rast(o)[:, :, :3]).max(axis=2).astype(np.uint8)
    hi = QtGui.QImage(d.tobytes(), w, h, w, QtGui.QImage.Format_Grayscale8)
    bb = QtCore.QByteArray()
    bf = QtCore.QBuffer(bb)
    bf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    hi.save(bf, 'PNG')
    return base64.b64encode(bytes(bb)).decode()


def design_b64(design_path):
    """96px preview of the raw 1024px AI design."""
    img = cv2.imread(str(design_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return base64.b64encode(buf.tobytes()).decode()


def build_gallery(here, root, rep8, rep10, manifest, designs_dir, native_svg):
    """Assemble the DATA payload and write data/gallery.html. Returns the
    output path. `here` is scripts/icon_hd, `root` the repo root."""
    here, root = Path(here), Path(root)
    relocate = json.loads((here / 'qa.json').read_text()).get('relocate', {})
    fitmap = json.loads((here / 'qa.json').read_text()).get('fit', {})
    scratch = here / 'data' / 'gallery_v8_scratch.svg'
    scratch.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for p, m in sorted(rep8.items()):
        sp = here / 'orig' / p.replace('/', '__')
        si = QtGui.QImage(str(sp))
        w, h = si.width(), si.height()
        repo_svg = root / relocate.get(p, p[:-4] + '.svg')
        win = fitmap.get(relocate.get(p, p[:-4] + '.svg'))
        src = lossless.qimage_rgba(si)
        svg8 = lossless.transcribe_lossless(src)[0]
        m10 = rep10.get(p, {})
        svg10 = None
        hd = None
        if m10.get('method') == 'v10-embed' and p in manifest:
            # HD preview so large/4K zoom downsamples the 1024px design instead
            # of upscaling the <=256px embedded raster; same declared window
            k = 512 / max(w, h)
            rgba = fit.hd_align(redesign.hd_raster(src, designs_dir / manifest[p], 512),
                                w, h, k, win)
            okc, buf = cv2.imencode('.png', cv2.cvtColor(rgba.astype(np.uint8),
                                                       cv2.COLOR_RGBA2BGRA),
                                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
            hd = base64.b64encode(buf.tobytes()).decode()
        if p in native_svg:
            svg8 = repo_svg.read_text()
            svg10 = svg8
            heat = heat_b64(repo_svg, si, w, h, win)
        else:
            # v8 is a pure function of the fixture: transcribe live, no backup;
            # show it in the published (canvas-fitted) state
            if win:
                svg8 = fit.reframe_text(svg8, win)
            scratch.write_text(svg8)
            heat = heat_b64(scratch, si, w, h, win)
            if m10.get('method') in ('v10-vector', 'v10-embed'):
                svg10 = repo_svg.read_text()
        data.append(dict(p=p, w=w, h=h,
                         png=base64.b64encode(sp.read_bytes()).decode(),
                         svg=svg8, svg10=svg10,
                         heat=heat,
                         heat10=heat_b64(repo_svg, si, w, h, win) if svg10 else None,
                         design=design_b64(designs_dir / manifest[p]) if p in manifest else None,
                         hd=hd,
                         m=m, m10=m10))
    tpl = (here / 'gallery_template.html').read_text()
    html = tpl.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    out = here / 'data' / 'gallery.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    scratch.unlink(missing_ok=True)
    return out, data