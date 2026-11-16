"""Canvas-fit pass: square ink-tight viewBox reframe at 96% occupancy.

The toolbar complaint ("content too small") has exactly two causes, both
geometric: padding inside a square canvas, and letterboxing of a non-square
canvas in a square toolbar cell. One operation removes both: move the viewBox
window to a square centred on the ink bbox whose side is max(ink)/0.96.
It is a pure window change in unchanged user space -- vector paths, gradients
(userSpaceOnUse) and embedded rasters are untouched, no resampling happens.

Icons already square with occupancy >= 0.90 are identity (cube family, full
bleed art). The declared window is stored in qa.json['fit'] keyed by repo SVG
path so fidelity gates / heat maps / 4K sheets can inverse-map the published
render back into the original canvas frame (a single affine warp).
"""
import re

import cv2
import numpy as np

import lossless

TARGET = 0.96   # ink longest side / canvas side after the pass
THRESH = 0.90   # occupancy at or above which a square canvas is left alone
ALPHA = 128     # ink threshold on the 0..255 alpha channel


def size_of(text):
    m = re.search(r'viewBox="([0-9.eE\-+ ]+)"', text)
    if m:
        x, y, w, h = map(float, m.group(1).split())
        return max(1.0, w), max(1.0, h)
    w = re.search(r'width="([0-9.]+)', text)
    h = re.search(r'height="([0-9.]+)', text)
    return float(w.group(1)), float(h.group(1))


def ink_window(rgba, scale=1):
    """(X, Y, S) square window in canvas units, or None for identity.
    rgba is the transparent render of the icon at `scale` px per unit."""
    h, w = rgba.shape[:2]
    ys, xs = np.where(rgba[:, :, 3] >= ALPHA)
    if len(xs) == 0:
        return None
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if max(bw / w, bh / h) >= THRESH and w == h:
        return None
    s = max(bw, bh) / TARGET / scale
    x = (xs.min() + xs.max() + 1) / 2 / scale - s / 2
    y = (ys.min() + ys.max() + 1) / 2 / scale - s / 2
    return (round(x, 3), round(y, 3), round(s, 3))


def svg_window(path, scale=4):
    """Window of a repo SVG, measured at 4x for sub-pixel bbox precision."""
    text = path.read_text(errors='ignore')
    w, h = size_of(text)
    ren = lossless.render_rgba(str(path), int(w), int(h), None, scale=scale)
    if ren is None:
        return None
    return ink_window(ren, scale)


def reframe_text(text, win):
    """Rewrite only the root <svg> tag: viewBox + width/height to the square
    window. Child width/height attributes (embedded images) must stay
    untouched, and an XML prolog / leading comment must survive intact."""
    x, y, s = win
    m = re.search(r'<svg\b[^>]*>', text, re.S)
    if m is None:
        raise ValueError('no root <svg> tag')
    head = m.group(0)
    if re.search(r'viewBox\s*=', head):
        head = re.sub(r'viewBox\s*=\s*"[^"]*"', f'viewBox="{x} {y} {s} {s}"', head, count=1)
    else:
        head = head[:-1] + f' viewBox="{x} {y} {s} {s}">'
    if 'width="' in head:
        head = re.sub(r'width="[^"]*"', f'width="{s}"', head, count=1)
    else:
        head = head[:-1] + f' width="{s}">'
    if 'height="' in head:
        head = re.sub(r'height="[^"]*"', f'height="{s}"', head, count=1)
    else:
        head = head[:-1] + f' height="{s}">'
    return text[:m.start()] + head + text[m.end():]


def align_render(path, w, h, bg, scale=1, win=None):
    """Render a published (possibly reframed) SVG back into the ORIGINAL
    w x h canvas frame at `scale`, so |render - orig| stays a fidelity proof.
    win=None renders plainly. One bilinear warp, sub-pixel exact."""
    if win is None:
        return lossless.render_rgba(str(path), w, h, bg, scale=scale)
    x, y, s = win
    n = max(1, int(round(s * scale)))
    ren = lossless.render_rgba(str(path), n, n, bg)
    if ren is None:
        return None
    f = s * scale / n
    # warpAffine's default M is FORWARD (src->dst, inverted internally): the
    # reframed render fills [0,n] in window coords and lands at (x,y)*scale
    m = np.float32([[f, 0, x * scale], [0, f, y * scale]])
    border = (0, 0, 0, 0) if bg is None else (bg[0], bg[1], bg[2], 255)
    out = cv2.warpAffine(ren.astype(np.uint8), m,
                         (int(round(w * scale)), int(round(h * scale))),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=border)
    return out.astype(np.int64)


def hd_align(hd, w, h, k, win, bg=None, k_hd=None):
    """Inverse-map an HD embed preview (orig-frame RGBA at k_hd px/unit) through
    the declared window, then meet-place it in the w*k x h*k tile frame.
    k_hd defaults to k; callers whose HD raster scale differs from the tile
    scale (4K sheets: ed=512 vs CELL-derived k) must pass it explicitly."""
    if win is None:
        return hd
    if k_hd is None:
        k_hd = k
    x, y, s = win
    side = max(1, int(round(s * k_hd)))
    # forward M: shifting src by -(x,y)*k_hd crops the window at (x,y)
    m = np.float32([[1, 0, -x * k_hd], [0, 1, -y * k_hd]])
    crop = cv2.warpAffine(hd.astype(np.uint8), m, (side, side),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))
    ow, oh = int(round(w * k)), int(round(h * k))
    f = min(ow, oh) / side
    nw, nh = max(1, int(round(side * f))), max(1, int(round(side * f)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((oh, ow, 4), np.uint8)
    x0, y0 = (ow - nw) // 2, (oh - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out.astype(np.int64)