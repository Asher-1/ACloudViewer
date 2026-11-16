"""v10 commercial AI redesign: ImageGen 1024px art -> keyed RGBA -> smooth
bezier SVG, with a dual gate (vector fidelity + semantic drift safety-net).

Pure functions: paths are arguments, the Qt application lives in icon_hd.py.
The only writes are scratch SVG files inside a caller-provided workdir, needed
because QSvgRenderer rasterizes from a file path.

Decision data (which gate-fails were visually approved, which icons are solid
fixtures or line art) lives in qa.json, never in this module.
"""
import base64
import re
import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import potrace
from scipy import ndimage
from skimage import measure as skm

import lossless

# vector-vs-design fidelity gate at 4x (pure vector must keep the design)
VECTOR_MEAN_MAX = 10.0
VECTOR_FRAC_MAX = 0.10
# semantic drift gate for a DESIGN REPLACEMENT (not pixel fidelity): catch
# wrong-object / wrong-topology generations; palette & footprint may
# legitimately differ from the 16px original (opaque panels -> transparency)
SEM_FRAC_MAX = 0.70
SEM_IOU_MIN = 0.45
SEM_TOPO_RANGE = (0.5, 2.0)
# logical vectorization resolution multiplier
TARGET = 16


def key_white(img_bgr):
    """Remove border-connected white background; matte the AA band against
    white so no fringe remains; interior whites stay opaque."""
    rgb = img_bgr[:, :, ::-1].astype(np.float64)
    mn = rgb.min(axis=2)
    near_white = mn >= 236
    lab, n = ndimage.label(near_white)
    border_labels = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border_labels.discard(0)
    bg = np.isin(lab, list(border_labels))
    band = ndimage.binary_dilation(bg, iterations=4) & ~bg
    alpha = np.where(bg, 0.0, 1.0)
    # white matting inside the band: a = 1 - min/255 clamped, then un-premultiply
    a_band = np.clip((255.0 - mn) / (255.0 - 200.0), 0, 1)
    alpha = np.where(band, np.maximum(a_band, 0.0), alpha)
    alpha = np.where(bg, 0.0, alpha)
    a3 = alpha[..., None]
    safe = np.maximum(a3, 1e-3)
    rgb_true = np.clip((rgb - (1 - a3) * 255.0) / safe, 0, 255)
    rgb_true = np.where(a3 > 0.02, rgb_true, 0)
    return np.dstack([rgb_true, alpha * 255.0])


def trace_bezier(mask, scale):
    bm = potrace.Bitmap(~np.asarray(mask, dtype=bool))
    path = bm.trace(turdsize=2, alphamax=1.0, opticurve=1, opttolerance=0.2)
    parts = []
    for curve in path:
        sp = curve.start_point
        d = [f'M{sp.x / scale:.3f} {sp.y / scale:.3f}']
        for seg in curve.segments:
            if seg.is_corner:
                d.append(f'L{seg.c.x / scale:.3f} {seg.c.y / scale:.3f}'
                         f'L{seg.end_point.x / scale:.3f} {seg.end_point.y / scale:.3f}')
            else:
                d.append(f'C{seg.c1.x / scale:.3f} {seg.c1.y / scale:.3f} '
                         f'{seg.c2.x / scale:.3f} {seg.c2.y / scale:.3f} '
                         f'{seg.end_point.x / scale:.3f} {seg.end_point.y / scale:.3f}')
        d.append('Z')
        parts.append(''.join(d))
    return ' '.join(parts)


def gradient_domains(rgb8, core, tol=24):
    """4-connected flood-fill regions of near-constant colour inside the core
    mask. O(H*W) amortized: every pixel is visited once."""
    h, w = core.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = []
    for y0 in range(h):
        for x0 in range(w):
            if visited[y0, x0] or not core[y0, x0]:
                continue
            stack = [(x0, y0)]
            visited[y0, x0] = True
            members = []
            while stack:
                x, y = stack.pop()
                members.append((x, y))
                cur = rgb8[y, x]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx] \
                            and core[ny, nx] and np.abs(rgb8[ny, nx] - cur).max() <= tol:
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            m = np.zeros((h, w), dtype=bool)
            for x, y in members:
                m[y, x] = True
            regions.append(m)
    return regions


def fit_linear_gradient(rgb01, mask):
    ys, xs = np.nonzero(mask)
    colors = rgb01[ys, xs]
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    A = np.stack([np.ones_like(x), x, y], axis=1)
    coef, *_ = np.linalg.lstsq(A, colors, rcond=None)
    pred = A @ coef
    resid = float(np.sqrt(((colors - pred) ** 2).mean(axis=0)).max() * 255)
    gx, gy = coef[1].mean(), coef[2].mean()
    if abs(gx) + abs(gy) < 1e-6:
        return resid, None
    dirv = np.array([gx, gy])
    dirv /= np.linalg.norm(dirv) + 1e-9
    proj = (x - x.mean()) * dirv[0] + (y - y.mean()) * dirv[1]
    p0, p1 = proj.min(), proj.max()
    cx, cy = x.mean(), y.mean()
    c0 = np.clip(np.array([1, cx + dirv[0] * p0, cy + dirv[1] * p0]) @ coef, 0, 1)
    c1 = np.clip(np.array([1, cx + dirv[0] * p1, cy + dirv[1] * p1]) @ coef, 0, 1)
    return resid, dict(x1=cx + dirv[0] * p0, y1=cy + dirv[1] * p0,
                       x2=cx + dirv[0] * p1, y2=cy + dirv[1] * p1, c0=c0, c1=c1)


def hexc(c):
    r, g, b = (np.clip(c, 0, 1) * 255).round().astype(int)
    return f'#{r:02x}{g:02x}{b:02x}'


def vectorize(rgba, scale, w, h, tol=12):
    """Gradient domains + dominant-colour bins -> bezier paths at `scale`x
    logical resolution. Returns (svg_text, info) or (None, reason)."""
    H, W = rgba.shape[:2]
    rgb01 = rgba[:, :, :3] / 255.0
    rgb8 = rgba[:, :, :3].astype(int)
    alpha = rgba[:, :, 3] / 255.0
    solid = alpha >= 0.5
    core = alpha >= 0.9
    if solid.sum() < 16:
        return None, 'empty'
    thinline = core.sum() < 0.25 * solid.sum()
    defs, paths = [], []
    claimed = np.zeros((H, W), dtype=bool)
    gi = 0
    if not thinline:
        regions = [m for m in gradient_domains(rgb8, core, tol=tol) if m.sum() >= 12]
        regions.sort(key=lambda m: -m.sum())
        regions = regions[:48]
        near = np.full((H, W), -1, dtype=int)
        dq = deque()
        for k, m in enumerate(regions):
            for y, x in zip(*np.nonzero(m)):
                near[y, x] = k
                dq.append((y, x, k))
        while dq:
            y, x, k = dq.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and near[ny, nx] == -1 and alpha[ny, nx] > 0.16:
                    near[ny, nx] = k
                    dq.append((ny, nx, k))
        for k, m in enumerate(regions):
            cols = rgb01[m]
            cspan = float((cols.max(axis=0) - cols.min(axis=0)).max())
            resid, grad = fit_linear_gradient(rgb01, m)
            mask = np.where(near == k, np.minimum(alpha / 0.9, 1.0), 0.0)
            d = trace_bezier(mask >= 0.5, scale)
            if not d:
                continue
            claimed |= m
            if grad is not None and resid <= 22 and cspan > 0.08:
                defs.append(
                    f'<linearGradient id="g{gi}" gradientUnits="userSpaceOnUse" '
                    f'x1="{grad["x1"] / scale:.3f}" y1="{grad["y1"] / scale:.3f}" '
                    f'x2="{grad["x2"] / scale:.3f}" y2="{grad["y2"] / scale:.3f}">'
                    f'<stop offset="0" stop-color="{hexc(grad["c0"])}"/>'
                    f'<stop offset="1" stop-color="{hexc(grad["c1"])}"/></linearGradient>')
                fill = f'url(#g{gi})'
                gi += 1
            else:
                fill = hexc(cols.mean(axis=0))
            paths.append((float(m.sum()), f'<path d="{d}" fill="{fill}"/>'))
    src_core = (core & ~claimed) if not thinline else solid
    if src_core.sum() >= 4:
        q = (rgb01[src_core] * 15).round().astype(int)
        keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
        uniq, cnt = np.unique(keys, return_counts=True)
        dom = uniq[np.argsort(-cnt)[:20]]
        dom_rgb = np.stack([dom // 256, dom // 16 % 16, dom % 16], axis=1) / 15.0
        base = (solid & ~claimed) if not thinline else solid
        flat = rgb01[base]
        lab = ((flat[:, None, :] - dom_rgb[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        labels = np.full((H, W), -1, dtype=int)
        labels[base] = lab
        for k in range(len(dom)):
            if thinline:
                mask = np.where(labels == k, alpha, 0.0)
                if mask.max() < 0.3:
                    continue
                d = trace_bezier(mask >= 0.25, scale)
                mean_a = float(alpha[(labels == k) & (alpha >= 0.25)].mean())
                op = '' if mean_a >= 0.95 else f' fill-opacity="{mean_a:.2f}"'
            else:
                mask = np.where(labels == k, np.minimum(alpha / 0.5, 1.0), 0.0)
                if mask.max() < 0.5:
                    continue
                d = trace_bezier(mask >= 0.5, scale)
                op = ''
            if not d:
                continue
            paths.append((float((labels == k).sum()),
                          f'<path d="{d}" fill="{hexc(dom_rgb[k])}"{op}/>'))
    if not paths:
        return None, 'no-paths'
    paths.sort(key=lambda t: -t[0])
    body = '\n'.join(t[1] for t in paths)
    defs_xml = ('<defs>' + ''.join(defs) + '</defs>') if defs else ''
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">\n'
           f'<!-- Commercial HD v10: AI-designed 1024px art, keyed + bezier '
           f'vectorized ({len(paths)} paths, {gi} gradients) -->\n{defs_xml}\n{body}\n</svg>\n')
    return svg, f'{len(paths)} paths/{gi} grad'


def crop_fit(keyed, w, h, target, src):
    """Crop to content bbox and fit into the logical canvas matching the
    ORIGINAL icon's footprint (bbox fill ratio + centre) so the redesign keeps
    the same optical size and position in the UI."""
    a = keyed[:, :, 3]
    ys, xs = np.nonzero(a > 8)
    if len(xs) == 0:
        return keyed
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = keyed[y0:y1 + 1, x0:x1 + 1]
    ch, cw = crop.shape[:2]
    sa = src[:, :, 3]
    sy, sx = np.nonzero(sa > 8)
    if len(sx) == 0:
        fx0, fx1, fy0, fy1 = 0, w - 1, 0, h - 1
    else:
        fx0, fx1, fy0, fy1 = sx.min(), sx.max(), sy.min(), sy.max()
    tw, th = w * target, h * target
    fw, fh = (fx1 - fx0 + 1) / w, (fy1 - fy0 + 1) / h
    s = min(tw * fw / cw, th * fh / ch)
    nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
    rz = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float64)
    canvas = np.zeros((th, tw, 4), dtype=np.float64)
    cx, cy = (fx0 + fx1 + 1) / 2 * target, (fy0 + fy1 + 1) / 2 * target
    ox, oy = int(cx - nw / 2), int(cy - nh / 2)
    ox, oy = max(0, min(tw - nw, ox)), max(0, min(th - nh, oy))
    canvas[oy:oy + nh, ox:ox + nw] = rz
    return canvas


def ink_mask(rgba):
    """Content silhouette: alpha for transparent art, colour distance from the
    opaque corner background for panel-style art (symmetric for src & render)."""
    if float((rgba[:, :, 3] >= 250).mean()) >= 0.98:
        corners = np.stack([rgba[0, 0, :3], rgba[0, -1, :3], rgba[-1, 0, :3], rgba[-1, -1, :3]])
        bgc = np.median(corners, axis=0)
        return np.abs(rgba[:, :, :3] - bgc).max(axis=2) > 48
    return rgba[:, :, 3] / 255.0 > 0.5


def gates(svg_path, src):
    """Semantic drift safety-net: white-bg composite frac, ink-silhouette IoU,
    topology ratio. Transparency replacing an opaque panel is intended, so no
    grey-bg term. Returns (frac, iou, topo) or None."""
    h, w = src.shape[:2]
    a = src[:, :, 3:4] / 255.0
    ren = lossless.render_rgba(svg_path, w, h, (255, 255, 255))
    if ren is None:
        return None
    truth = src[:, :, :3] * a + 255 * (1 - a)
    d = np.abs(ren[:, :, :3] - truth).max(axis=2)
    worst_frac = float((d > 64).mean())
    ra = lossless.render_rgba(svg_path, w, h, None)
    if ra is None:
        return None
    rb = ink_mask(ra)
    sb = ink_mask(src)
    union = (rb | sb).sum()
    iou = float((rb & sb).sum() / union) if union else 1.0
    n_src = int(skm.label(sb, connectivity=2).max())
    n_ren = int(skm.label(rb, connectivity=2).max())
    topo = n_ren / n_src if n_src else 1.0
    return worst_frac, iou, topo


def gate_ok(g):
    return g is not None and g[0] <= SEM_FRAC_MAX and g[1] >= SEM_IOU_MIN \
        and SEM_TOPO_RANGE[0] <= g[2] <= SEM_TOPO_RANGE[1]


def embed_hd(small, w, h):
    """Embed the keyed HD raster inside the SVG container (lossless design);
    cap embed resolution: >=128px for HiDPI UI, <=256px to bound repo size."""
    ed = int(min(256, max(128, max(w, h) * 8)))
    es = ed / max(w * TARGET, h * TARGET)
    emb = cv2.resize(small.astype(np.uint8),
                     (max(1, int(w * TARGET * es)), max(1, int(h * TARGET * es))),
                     interpolation=cv2.INTER_AREA)
    okc, buf = cv2.imencode('.png', cv2.cvtColor(emb, cv2.COLOR_RGBA2BGRA),
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
            f'<!-- Commercial HD v10: AI-designed art, keyed HD raster embedded -->\n'
            f'<image x="0" y="0" width="{w}" height="{h}" '
            f'xlink:href="data:image/png;base64,{b64}"/>\n</svg>\n')


def hd_raster(src, gen_path, ed=512):
    """High-res transparent RGBA of the redesign, footprint-fitted exactly like
    the published embed (same key_white + crop_fit) but at `ed` px instead of
    the <=256px repo cap. Used only for HD previews / 4K comparison sheets so
    large displays downsample instead of upscaling the embedded raster."""
    h, w = src.shape[:2]
    keyed = key_white(cv2.imread(str(gen_path), cv2.IMREAD_COLOR))
    small = crop_fit(keyed, w, h, TARGET, src)
    es = ed / max(w * TARGET, h * TARGET)
    return cv2.resize(small.astype(np.uint8),
                      (max(1, int(w * TARGET * es)), max(1, int(h * TARGET * es))),
                      interpolation=cv2.INTER_AREA)


def hd_preview_b64(src, gen_path, ed=512):
    """base64 PNG of hd_raster (RGBA) for inlining into the gallery."""
    rgba = hd_raster(src, gen_path, ed)
    okc, buf = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return base64.b64encode(buf.tobytes()).decode()


def build_candidate(src_path, gen_path, workdir):
    """Full v10 pipeline for one icon: key -> footprint fit -> vectorize ->
    4x vector-fidelity gate (embed fallback) -> semantic gate.
    Returns (svg_text, metrics); metrics['gate_ok'] drives auto-publish,
    everything else is report data. Scratch renders live in workdir."""
    from PySide6 import QtGui
    src = lossless.qimage_rgba(QtGui.QImage(str(src_path)))
    h, w = src.shape[:2]
    keyed = key_white(cv2.imread(str(gen_path), cv2.IMREAD_COLOR))
    small = crop_fit(keyed, w, h, TARGET, src)
    svg, info = vectorize(small, TARGET, w, h, tol=12)
    if svg is None:
        return None, dict(method='v10-fail', reason=info, gate_ok=False)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / 'candidate.svg'
    tmp.write_text(svg)
    # vector-vs-design quality gate at 4x: does the pure vector keep the design?
    k = 4
    design = cv2.resize(small.astype(np.uint8), (w * k, h * k),
                        interpolation=cv2.INTER_AREA).astype(np.float64)
    da = design[:, :, 3:4] / 255.0
    dtruth = design[:, :, :3] * da + 255 * (1 - da)
    ren = lossless.render_rgba(tmp, w, h, (255, 255, 255), scale=k)
    vmean = float(np.abs(ren[:, :, :3] - dtruth).mean()) if ren is not None else 999
    vfrac = float((np.abs(ren[:, :, :3] - dtruth).max(axis=2) > 32).mean()) if ren is not None else 1
    pure = vmean <= VECTOR_MEAN_MAX and vfrac <= VECTOR_FRAC_MAX
    if not pure:
        svg = embed_hd(small, w, h)
        tmp.write_text(svg)
        info = 'embedded-hd'
    g = gates(tmp, src)
    ok = gate_ok(g)
    m = dict(method=('v10-vector' if pure else 'v10-embed') if ok else 'v10-gate-fail',
             info=info, vmean=round(vmean, 1), vfrac=round(vfrac, 3),
             frac=round(g[0], 3) if g else None,
             iou=round(g[1], 3) if g else None, topo=round(g[2], 2) if g else None,
             kb=round(len(svg.encode()) / 1024, 1), gate_ok=ok)
    return svg, m


def solid_svg(src):
    """Solid-color test fixture: exact median source colour, crisp full-canvas
    rect. Returns (svg_text, hex_colour)."""
    h, w = src.shape[:2]
    op = src[:, :, 3] >= 250
    col = np.median(src[:, :, :3][op], axis=0).round().astype(int)
    hexcol = '#%02x%02x%02x' % tuple(col)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">\n<!-- Commercial HD v10: solid-color fixture, '
           f'exact source colour -->\n<rect width="{w}" height="{h}" fill="{hexcol}" '
           f'shape-rendering="crispEdges"/>\n</svg>\n')
    return svg, hexcol


def line_trace_svg(keyed, src, workdir):
    """Monochrome line art: per-colour binning fragments the AA line into dots
    that potrace drops; trace the whole ink mask as one solid path instead.
    Stretch to the ORIGINAL footprint (non-uniform) and dilate the ink so the
    stroke stays >=1px at 1x (uniform fit left it sub-pixel/faint); remap path
    coords so the traced stroke lands exactly on the original bbox.
    Returns (svg_text, gate_tuple)."""
    h, w = src.shape[:2]
    ka = keyed[:, :, 3]
    ys, xs = np.nonzero(ka > 8)
    crop = keyed[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sa = src[:, :, 3]
    sy, sx = np.nonzero(sa > 8)
    fx0, fx1, fy0, fy1 = sx.min(), sx.max(), sy.min(), sy.max()
    nw, nh = (fx1 - fx0 + 1) * TARGET, (fy1 - fy0 + 1) * TARGET
    rz = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float64)
    small = np.zeros((h * TARGET, w * TARGET, 4))
    small[fy0 * TARGET:fy0 * TARGET + nh, fx0 * TARGET:fx0 * TARGET + nw] = rz
    a = small[:, :, 3] / 255.0
    mask = ndimage.binary_dilation(a >= 0.2, iterations=6)
    dom = np.median(small[:, :, :3][a > 0.5], axis=0)
    d = trace_bezier(mask, TARGET)
    # affine remap of every coordinate pair (token walk over the path data)
    nums = [float(t) for t in re.findall(r'-?\d+\.\d+', d)]
    xs_, ys_ = nums[0::2], nums[1::2]
    x0_, x1_, y0_, y1_ = min(xs_), max(xs_), min(ys_), max(ys_)

    def _mx(x):
        return fx0 + (x - x0_) * (fx1 - fx0) / max(1e-6, x1_ - x0_)

    def _my(y):
        return fy0 + (y - y0_) * (fy1 - fy0) / max(1e-6, y1_ - y0_)

    toks = re.findall(r'[MLCZ]|-?\d+\.\d+', d)
    res, k = [], 0
    while k < len(toks):
        t = toks[k]
        if t in ('M', 'L', 'C', 'Z'):
            res.append(t)
            k += 1
        else:
            x, y = float(toks[k]), float(toks[k + 1])
            res.append(f'{_mx(x):.3f} {_my(y):.3f}')
            k += 2
    d = ''.join(res)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
           f'<!-- Commercial HD v10: AI-designed line art, single solid trace -->\n'
           f'<path d="{d}" fill="{hexc(dom / 255.0)}"/>\n</svg>\n')
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = workdir / 'line_trace.svg'
    tmp.write_text(svg)
    return svg, gates(tmp, src)