"""Static 4K (3840x2160) orig / v8-lossless / v10-final comparison sheets.

Crispness contract (never upscale a raster):
  * orig  -> nearest-neighbor at an INTEGER scale (every source pixel visible,
             i.e. the detail truth), via the same Qt paint pipeline as the app;
  * v8    -> the lossless rect transcription, crispEdges at the same integer scale;
  * v10   -> published vector SVG rendered at that scale (infinite crisp), or for
             HD-embed icons the 1024px design downsampled to the tile (the embed
             repo cap is <=256px, meant for in-app <=128px@2x, not for 4K walls).

Pure rendering: the Qt application and all path ownership live in icon_hd.py.
"""
from pathlib import Path

import json
import cv2
import numpy as np
from PySide6 import QtCore, QtGui

import lossless
import redesign
import fit

W, H = 3840, 2160
CELL, GAP_IN, GAP_OUT = 224, 16, 32
COLS, ROWS = 5, 6
HEADER, LABEL, ROW_GAP = 120, 40, 28

BGS = {
    'light': dict(canvas='#eef0f3', tile='#ffffff', ink='#20242a', sub='#5a6270'),
    'dark': dict(canvas='#14181e', tile='#22262c', ink='#e8ecf2', sub='#9aa4b2'),
}


def _rgb_img(arr):
    a = np.ascontiguousarray(arr.astype(np.uint8))
    h, w = a.shape[:2]
    return QtGui.QImage(a.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)


def _tiles(sp, repo_svg, svg8_scratch, src, m10, design_path, bg, k, win=None):
    """(orig, v8, v10) RGB rasters at w*k x h*k on `bg`, all crisp."""
    h, w = src.shape[:2]
    orig = lossless.render_orig(str(sp), w, h, bg, scale=k)[:, :, :3]
    v8 = lossless.render_rgba(str(svg8_scratch), w, h, bg, scale=k)[:, :, :3]
    if m10.get('method') == 'v10-embed' and design_path is not None:
        hd = redesign.hd_raster(src, design_path, ed=512)
        # hd_raster is already in the ORIG frame (crop_fit places the content
        # at the orig footprint), so the fidelity column is a pure downsample;
        # the fit window only matters on the display path (gallery hd pane)
        hd = cv2.resize(hd, (w * k, h * k), interpolation=cv2.INTER_AREA).astype(np.int64)
        v10 = lossless.composite(hd, bg)
    else:
        v10 = fit.align_render(str(repo_svg), w, h, bg, scale=k, win=win)[:, :, :3]
    return orig, v8, v10


def build_sheets(here, root, rep8, rep10, manifest, designs_dir, native_svg, outdir):
    here, root, outdir = Path(here), Path(root), Path(outdir)
    relocate = json.loads((here / 'qa.json').read_text()).get('relocate', {})
    fitmap = json.loads((here / 'qa.json').read_text()).get('fit', {})
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = outdir / '_v8_scratch.svg'
    icons = sorted(rep8)
    per = COLS * ROWS
    triplet_w = 3 * CELL + 2 * GAP_IN
    grid_w = COLS * triplet_w + (COLS - 1) * GAP_OUT
    mx = (W - grid_w) // 2
    paths = []
    for bgname, pal in BGS.items():
        bg = tuple(int(pal['tile'][i:i + 2], 16) for i in (1, 3, 5))
        for s in range((len(icons) + per - 1) // per):
            chunk = icons[s * per:(s + 1) * per]
            cv = QtGui.QImage(W, H, QtGui.QImage.Format_ARGB32)
            cv.fill(QtGui.QColor(pal['canvas']))
            pt = QtGui.QPainter(cv)
            pt.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            f = pt.font(); f.setPixelSize(40); f.setBold(True); pt.setFont(f)
            pt.setPen(QtGui.QColor(pal['ink']))
            pt.drawText(mx, 66, f'ICON HD · 4K comparison · {bgname} · sheet {s + 1}')
            f.setPixelSize(24); f.setBold(False); pt.setFont(f)
            pt.setPen(QtGui.QColor(pal['sub']))
            pt.drawText(mx, 102, 'orig (nearest, detail truth)  |  v8 lossless  |  v10 final '
                                 '· 3840×2160 · vector crisp / embed downsampled from 1024px')
            for i, p in enumerate(chunk):
                col, row = i % COLS, i // COLS
                x0 = mx + col * (triplet_w + GAP_OUT)
                y0 = HEADER + row * (CELL + LABEL + ROW_GAP)
                sp = here / 'orig' / p.replace('/', '__')
                src = lossless.qimage_rgba(QtGui.QImage(str(sp)))
                h, w = src.shape[:2]
                k = max(1, CELL // max(w, h))
                m10 = rep10.get(p, {})
                svg8, _ = lossless.transcribe_lossless(src)
                svg_rel = relocate.get(p, p[:-4] + '.svg')
                win = fitmap.get(svg_rel)
                if win:
                    svg8 = fit.reframe_text(svg8, win)
                scratch.write_text(svg8)
                dp = designs_dir / manifest[p] if p in manifest else None
                tiles = _tiles(sp, root / svg_rel,
                               scratch, src, m10, dp, bg, k, win)
                for j, arr in enumerate(tiles):
                    tx = x0 + j * (CELL + GAP_IN)
                    pt.fillRect(tx, y0, CELL, CELL, QtGui.QColor(pal['tile']))
                    im = _rgb_img(arr)
                    pt.drawImage(tx + (CELL - im.width()) // 2,
                                 y0 + (CELL - im.height()) // 2, im)
                f.setPixelSize(22); pt.setFont(f)
                pt.setPen(QtGui.QColor(pal['sub']))
                tag = m10.get('method', 'native')
                pt.drawText(QtCore.QRect(x0, y0 + CELL + 4, triplet_w, LABEL),
                            QtCore.Qt.AlignmentFlag.AlignCenter,
                            f"{p.split('/')[-1]}  ·  {w}×{h}  ·  {tag}")
            pt.end()
            out = outdir / f'compare4k_{bgname}_{s + 1:02d}.png'
            cv.save(str(out))
            paths.append(out)
    scratch.unlink(missing_ok=True)
    return paths
