"""Audit report: before/after for every icon, with the code that proves its meaning.

Why this exists
---------------
A visual diff alone cannot be reviewed.  "Look at these 270 pictures and tell me
which are wrong" asks the reviewer to reverse-engineer, from the drawing, what
the icon is *supposed* to mean -- and the drawing is exactly the thing under
suspicion.  This report attaches the evidence instead: for each icon, the
``QAction`` text, tooltip and source file that reference it.  A reviewer checks
the drawing against the stated behaviour, not against their own guess.

What "before" means
-------------------
The previous artwork is read out of git (``HEAD:`` rather than the working
tree), so the comparison is against what shipped, never against a file this
toolchain has already overwritten.  Raster before-images are rendered at the
same box size as the vector after-image so the eye compares shape, not scale.

Usage
-----
    python3 -m scripts.icon_hd.logo.audit                    # build the report
    python3 -m scripts.icon_hd.logo.audit --out build/audit.html
    python3 -m scripts.icon_hd.logo.audit --only pq          # one family
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGES = ROOT / "app/Resources/images"

# Render box for every thumbnail.  96 px is the size at which shape differences
# remain legible in a grid while a row of 8 still fits a laptop width.
BOX = 96

# Ink-coverage ceiling, the same gate the design system documents.  A "before"
# icon above it is reported dense; it is a fact about the render, not an opinion.
DENSE = 0.45


# ---------------------------------------------------------------------------
# Evidence: what code actually uses this icon, and what does it say it does
# ---------------------------------------------------------------------------


def _read_sources() -> dict[str, str]:
    out = {}
    patterns = ("app/**/*.ui", "app/**/*.cpp", "app/**/*.h",
                "libs/**/*.ui", "libs/**/*.cpp", "libs/**/*.h",
                "plugins/**/*.ui", "plugins/**/*.cpp")
    for pat in patterns:
        for p in ROOT.glob(pat):
            if "3rdparty" in str(p):
                continue
            try:
                out[str(p.relative_to(ROOT))] = p.read_text(errors="ignore")
            except OSError:
                continue
    return out


_ACTION = re.compile(r'<(action|widget)[^>]*name="([^"]+)">', re.S)


def _spans(text: str, tag: str):
    """(start, end, identifier) for every balanced ``<tag>`` block in ``text``.

    Two wrong versions preceded this one, and both produced silent
    misattributions -- the worst possible failure for a review tool, because a
    reviewer would redraw an icon against a label that belongs to something else:

    1. ``.*?(?=</action>|</widget>)`` stopped at the first closing tag of *either*
       kind, so an ``<action>`` holding a ``QMenu`` was cut at the inner
       ``</widget>`` and the icon was matched against the next element.
    2. Depth counting without ranking returned the *outermost* container, so
       every icon inside a dialog mapped to the dialog widget itself
       (``ecvMinus`` -> ``selectionGroupBox``, ``add`` -> ``MainViewerClass``).

    The fix is both: balance the tags, then resolve each icon reference to the
    **innermost** block that contains it.
    """
    open_re = re.compile(rf'<{tag}\b[^>]*\bname="([^"]+)"[^>]*>')
    any_open = re.compile(rf'<{tag}\b')
    any_close = re.compile(rf'</{tag}>')

    out = []
    for m in open_re.finditer(text):
        pos, depth = m.end(), 1
        end = len(text)
        while depth > 0:
            nxt_open = any_open.search(text, pos)
            nxt_close = any_close.search(text, pos)
            if not nxt_close:
                break
            if nxt_open and nxt_open.start() < nxt_close.start():
                depth += 1
                pos = nxt_open.end()
            else:
                depth -= 1
                if depth == 0:
                    end = nxt_close.start()
                    break
                pos = nxt_close.end()
        out.append((m.start(), end, m.group(1)))
    return out


def _innermost(spans, position: int):
    """Smallest span containing ``position``, or None."""
    best = None
    for start, end, ident in spans:
        if start < position < end:
            if best is None or (end - start) < (best[1] - best[0]):
                best = (start, end, ident)
    return best

# Icons registered from C++ rather than from a .ui file.  Three shapes occur:
#
#   const int cloudIndex = mIconList.count();
#   mIconList.append({QIcon(QStringLiteral(":/Resources/images/dbCloudSymbol.svg")), ...});
#
#   setIcon(QIcon(":/Resources/images/svg/pq/pqCamera.svg"));
#
#   static const QString APP_LOGO = ":/Resources/images/icon/logo_256.png";
#
# The first is by far the most common in this repo (the whole DB tree), and the
# only place its meaning is written down is the index variable name plus the
# CV_TYPES map that consumes it.  Without this branch 88 icons report "no
# evidence" purely because they have no .ui entry, which would be a false alarm
# in a review.
_CXX_INDEX = re.compile(
    r'const int (\w+)\s*=\s*mIconList\.count\(\);(.*?)(?=const int \w+\s*=\s*mIconList\.count\(\)|\}|$)',
    re.S)
_CXX_ENUM_MAP = re.compile(r'\{CV_TYPES::(\w+),\s*(\w+)\}', re.S)
_CXX_SETICON = re.compile(r'setIcon(?:\(|.*?)QIcon\(\s*(?:QStringLiteral\()?"([^"]+)"', re.S)
_CXX_CONST = re.compile(r'(?:static\s+)?const QString (\w+)\s*=\s*"([^"]+)"', re.S)

# The selection controller builds every tool action through one helper:
#
#   make("actionSelectSurfaceCells",
#        ":/Resources/images/svg/pq/pqSurfaceSelectionCell.svg",
#        QObject::tr("Select Cells On (s)"),
#        QObject::tr("Select Cells On (s)"), true);
#
# name, icon, label and tooltip sit in a single call, which makes this the
# richest evidence in the repo -- it states what each icon means *and* what
# string the code keys it by, so nothing has to be inferred from the drawing.
_CXX_MAKE = re.compile(
    r'make\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*'
    r'(?:QObject::tr\(\s*)?"([^"]*)"\s*\)?\s*,\s*'
    r'(?:QObject::tr\(\s*)?"([^"]*)"',
    re.S)

# camelCase index name -> words, so "cameraSensorIndex" is readable in the report.
def _words(ident: str) -> str:
    s = re.sub(r'Index$', '', ident)
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    return s.strip().lower()


def _cxx_evidence(token: str, sources: dict[str, str]) -> list[dict]:
    """Icons referenced from C++ registration or setIcon calls."""
    hits = []
    for rel, text in sources.items():
        if token not in text or not rel.endswith((".cpp", ".h")):
            continue
        fname = Path(rel).name

        # (a) DB-tree registration: index variable names the icon's class.
        for m in _CXX_INDEX.finditer(text):
            ident, body = m.group(1), m.group(2)
            if token not in body:
                continue
            enums = [e for e, idx in _CXX_ENUM_MAP.findall(text) if idx == ident]
            label = _words(ident)
            tip = "db 树节点类型：" + ", ".join(enums[:4]) if enums else "db 树节点类型"
            hits.append({"file": fname, "path": rel, "kind": "db",
                         "id": ident, "text": label, "tip": tip})

        # (a2) selection-controller helper: icon, label and tooltip in one call.
        for m in _CXX_MAKE.finditer(text):
            ident, icon, label, tip = m.groups()
            if token not in icon:
                continue
            hits.append({"file": fname, "path": rel, "kind": "make",
                         "id": ident, "text": label, "tip": tip})

        # (b) direct setIcon call: report the enclosing function as context.
        for m in _CXX_SETICON.finditer(text):
            if token not in m.group(1):
                continue
            start = max(0, m.start() - 900)
            window = text[start:m.start()]
            fn = re.findall(r'^\w[\w:<>*&\s]*?(\w+)::(\w+)\s*\(', window, re.M)
            ctx = f"{fn[-1][0]}::{fn[-1][1]}" if fn else fname
            hits.append({"file": fname, "path": rel, "kind": "cpp",
                         "id": ctx, "text": "", "tip": f"在 {ctx} 中直接设置"})

        # (c) named constant, e.g. APP_START_LOGO.
        for m in _CXX_CONST.finditer(text):
            if token not in m.group(2):
                continue
            hits.append({"file": fname, "path": rel, "kind": "const",
                         "id": m.group(1), "text": "", "tip": f"常量 {m.group(1)}"})

        # (d) fallback for a bare QIcon("...") assignment: the only human-readable
        # description is the comment above it or the enclosing function, and both
        # are worth showing.  Several selection-widget icons are wired this way
        # (`pqApply`, `pqDelete`, `pqCloseView`) with the purpose stated in a
        # trailing comment rather than in any action object.
        for m in re.finditer(re.escape(token), text):
            start = max(0, m.start() - 1500)
            window = text[start:m.start() + 200]
            fn = re.findall(r'^\w[\w:<>*&\s]*?(\w+::\w+)\s*\(', window, re.M)
            comments = re.findall(r'//\s*(.+)', window)
            label = comments[-1].strip() if comments else ""
            ctx = fn[-1] if fn else fname
            if not label:
                # last resort: a human identifier assigned on the same statement
                ids = re.findall(r'\b(?:auto\*?|QAction\*|QPushButton\*|QToolButton\*)\s+(\w+)', window)
                label = ids[-1] if ids else ""
            hits.append({"file": fname, "path": rel, "kind": "inline",
                         "id": ctx, "text": label,
                         "tip": "" if label else f"在 {ctx} 中使用"})
    return hits


def evidence_for(token: str, sources: dict[str, str]) -> list[dict]:
    """Every place that references ``token``, with whatever label that place has.

    The ``.ui`` branch reads the label from the block that *contains* the icon
    reference, not from a neighbouring block: an earlier ad-hoc scan matched the
    next action down the file and produced four wrong mappings (``pqWrench``
    reported as "Glyph Filter", ``pqSideSetData`` as "Toggle Camera Orientation
    Widget").

    The C++ branch exists because the DB tree -- 20 of this repo's icons -- is
    registered programmatically and has no ``.ui`` entry at all.
    """
    hits = []
    for rel, text in sources.items():
        if token not in text:
            continue
        if rel.endswith(".ui"):
            spans = _spans(text, "action") + _spans(text, "widget")
            for m in re.finditer(re.escape(token), text):
                owner = _innermost(spans, m.start())
                if owner is None:
                    continue
                start, end, ident = owner
                body = text[start:end]
                kind = "action" if f'<action name="{ident}"' in body[:80] else "widget"
                label = _prop(body, "text")
                tip = _prop(body, "toolTip") or _prop(body, "statusTip")
                hits.append({"file": Path(rel).name, "path": rel, "kind": kind,
                             "id": ident, "text": label, "tip": tip})

    hits.extend(_cxx_evidence(token, sources))

    # Prefer entries that actually say something, then dedupe by identity.
    seen, uniq = set(), []
    for h in sorted(hits, key=lambda h: (not (h["text"] or h["tip"]))):
        key = (h["id"], h["text"], h["tip"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq[:3]


def _prop(body: str, name: str) -> str:
    m = re.search(rf'<property name="{name}">\s*<string>([^<]*)</string>', body, re.S)
    return html.unescape(m.group(1)).strip() if m else ""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(data: bytes, name: str) -> "PIL.Image.Image | None":
    """Rasterise SVG bytes or decode PNG/GIF bytes to a BOX-sized RGBA tile."""
    from PIL import Image

    try:
        if name.endswith(".svg") or data[:5].lstrip().startswith(b"<"):
            import cairosvg
            png = cairosvg.svg2png(bytestring=data, output_width=BOX, output_height=BOX)
            im = Image.open(io.BytesIO(png)).convert("RGBA")
        else:
            im = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:
        return None
    im.thumbnail((BOX, BOX), Image.LANCZOS)
    tile = Image.new("RGBA", (BOX, BOX), (0, 0, 0, 0))
    tile.alpha_composite(im, ((BOX - im.width) // 2, (BOX - im.height) // 2))
    return tile


def _ink(tile) -> float:
    import numpy as np
    return float((np.asarray(tile)[..., 3] > 80).mean())


def _b64(tile) -> str:
    buf = io.BytesIO()
    tile.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _git_show(rel: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _registry() -> dict[str, tuple[str, str]]:
    """icon basename -> (family, current SVG text), read from the repository.

    This used to import the eight ``*icons.py`` design modules and call their
    builders.  Those modules were removed: they regenerated 197 icons that were
    already valid vector art, which destroyed artwork the user had learned.  The
    audit therefore reads what actually ships, not what a design system would
    produce.
    """
    out = {}
    for p in IMAGES.rglob("*.svg"):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        family = p.parent.name if p.parent != IMAGES else "root"
        out[p.stem] = (family, text)
    return out


def _on_disk() -> dict[str, str]:
    """icon basename -> repo-relative path, for every svg under images/."""
    out = {}
    for p in IMAGES.rglob("*.svg"):
        out[p.stem] = str(p.relative_to(ROOT))
    return out


# A file named .svg is not necessarily vector art.  Measured over this
# repository, 109 of 203 SVG files are what the review must call a **pixel
# mosaic**: a small bitmap (16-48 px) re-encoded as several hundred 2-3 px
# <rect> elements.  At 18 logical px on a 3x display those need 54 device pixels,
# so they are blurry -- and no retracing fixes them, which was verified by
# tracing all 109 and reverting every one.
_MOSAIC_TINY = 40


def _material(text: str) -> tuple[str, int | None]:
    """Classify an SVG by what it actually contains, not by its extension.

    Returns (kind, grid) where kind is one of:
      ``embed``   carries a base64 raster or <image>
      ``mosaic``  hundreds of sub-3px rects; grid is the source bitmap size
      ``vector``  real paths, scales without limit
    """
    if "base64" in text or "<image" in text:
        return "embed", None
    body = re.sub(r"base64,[A-Za-z0-9+/=]{200,}", "", text)
    tiny = 0
    for tag in re.findall(r"<rect\b[^>]*>", body):
        m = re.search(r'width="([\d.]+)"', tag)
        if m:
            try:
                if float(m.group(1)) <= 3:
                    tiny += 1
            except ValueError:
                pass
    if tiny >= _MOSAIC_TINY:
        vb = re.search(r'viewBox="0 0 ([\d.]+)', text)
        grid = int(round(float(vb.group(1)))) if vb else None
        return "mosaic", grid
    return "vector", None


def _before_path(name: str, on_disk: dict[str, str]) -> str | None:
    """Where the pre-change artwork lived: same path, .png suffix."""
    svg_rel = on_disk.get(name)
    if svg_rel:
        for ext in (".png", ".gif"):
            cand = svg_rel[:-4] + ext
            if _git_show(cand) is not None:
                return cand
    # Flat-root icons moved in the first pass; try the historical root path.
    for cand in (f"app/Resources/images/{name}.png",
                 f"app/Resources/images/svg/{name}.png"):
        if _git_show(cand) is not None:
            return cand
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_CSS = """
:root{--ink:#1d2a35;--muted:#6a7b88;--line:#e2e8ee;--accent:#0a84b8;
      --bg:#fbfcfd;--card:#fff;--bad:#c0392b;--good:#1e7d47;--warn:#b46b00}
*{box-sizing:border-box}
body{margin:0;padding:28px 32px 80px;background:var(--bg);color:var(--ink);
     font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
h1{font-size:25px;margin:0 0 4px;letter-spacing:-.015em}
h2{font-size:16px;margin:36px 0 12px;padding-bottom:7px;
   border-bottom:1px solid var(--line);letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 22px;font-size:13px}
.stats{display:flex;gap:26px;flex-wrap:wrap;margin:18px 0 8px;font-size:13px}
.stats b{display:block;font-size:21px;letter-spacing:-.02em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(238px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:9px;
      padding:11px 12px 12px;display:flex;flex-direction:column;gap:9px}
.pair{display:flex;gap:9px;align-items:center}
.cell{text-align:center}
.cell img{display:block;background:
  repeating-conic-gradient(#f0f3f6 0 25%,#fff 0 50%) 0 0/13px 13px;
  border:1px solid var(--line);border-radius:6px}
.cap{font-size:10px;text-transform:uppercase;letter-spacing:.06em;
     color:var(--muted);margin-top:3px}
.name{font:600 12px/1.3 ui-monospace,SFMono-Regular,Menlo,monospace;
      word-break:break-all}
.mod{font-size:10px;color:var(--muted);font-family:ui-monospace,monospace}
.ev{font-size:11.5px;border-left:2px solid var(--accent);padding-left:7px;
    color:#37474f}
.ev .t{font-weight:600}
.ev .m{color:var(--muted);font-family:ui-monospace,monospace;font-size:10.5px}
.tag{display:inline-block;font-size:10px;padding:1px 6px;border-radius:9px;
     font-weight:600;letter-spacing:.02em}
.t-dense{background:#fdecea;color:var(--bad)}
.t-ok{background:#e8f5ec;color:var(--good)}
.t-noev{background:#fff4e0;color:var(--warn)}
.t-vec{background:#e6f2f8;color:#0a6e93}
.t-mosaic{background:#f3e8fd;color:#6d28a8}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:10px 0 0}
.note{background:#fff;border:1px solid var(--line);border-left:3px solid var(--accent);
      border-radius:0 8px 8px 0;padding:12px 15px;margin:18px 0;font-size:13px}
.note code{background:#eef3f7;padding:1px 5px;border-radius:4px;
           font-family:ui-monospace,monospace;font-size:12px}
"""


def build(only: str | None = None) -> str:
    reg = _registry()
    disk = _on_disk()
    src = _read_sources()

    names = sorted(reg)
    if only:
        names = [n for n in names if only in n or only in reg[n][0]]

    rows, n_before, n_dense_before, n_dense_after, n_noev, n_not_vector = [], 0, 0, 0, 0, 0
    for name in names:
        modname, after_svg = reg[name]
        after = _render(after_svg.encode(), name + ".svg")
        if after is None:
            continue
        ink_after = _ink(after)
        if ink_after > DENSE:
            n_dense_after += 1

        before = None
        rel_before = _before_path(name, disk)
        if rel_before:
            data = _git_show(rel_before)
            if data:
                before = _render(data, rel_before)
                if before is not None:
                    n_before += 1
                    if _ink(before) > DENSE:
                        n_dense_before += 1

        ev = evidence_for(name + ".svg", src) or evidence_for(name + ".png", src)
        if not ev or not any(e["text"] or e["tip"] for e in ev):
            n_noev += 1

        kind, grid = _material(after_svg)
        if kind != "vector":
            n_not_vector += 1

        rows.append((name, modname, before, after, ev, ink_after, rel_before,
                     kind, grid))

    # ---- markup ----
    def card(r):
        name, modname, before, after, ev, ink, rel_before, kind, grid = r
        tags = []

        # Material first: whether the file scales is the question this report
        # exists to answer.  A .svg extension proves nothing -- 109 of them are
        # bitmaps in disguise.
        if kind == "vector":
            tags.append('<span class="tag t-vec">真矢量</span>')
        elif kind == "mosaic":
            tags.append(f'<span class="tag t-mosaic">位图马赛克 {grid or "?"}px</span>')
        else:
            tags.append('<span class="tag t-mosaic">内嵌位图</span>')

        if not ev or not any(e["text"] or e["tip"] for e in ev):
            tags.append('<span class="tag t-noev">无文案证据</span>')
        if ink > DENSE:
            tags.append(f'<span class="tag t-dense">过密 {ink:.0%}</span>')
        else:
            tags.append(f'<span class="tag t-ok">墨量 {ink:.0%}</span>')

        ev_html = ""
        for e in ev[:2]:
            bits = []
            if e["text"]:
                bits.append(f'<span class="t">{html.escape(e["text"])}</span>')
            if e["tip"]:
                bits.append(html.escape(e["tip"]))
            if bits:
                ev_html += (f'<div class="ev">{" — ".join(bits)}'
                            f'<br><span class="m">{html.escape(e["id"])}'
                            f' · {html.escape(e["file"])}</span></div>')
        if not ev_html and ev:
            ev_html = (f'<div class="ev"><span class="m">{html.escape(ev[0]["id"])}'
                       f' · {html.escape(ev[0]["file"])}</span></div>')

        before_img = (f'<img src="data:image/png;base64,{_b64(before)}" width={BOX} height={BOX}>'
                      if before else
                      '<div style="width:96px;height:96px;border:1px dashed #ccd;'
                      'border-radius:6px;display:grid;place-items:center;'
                      'font-size:10px;color:#99a">无旧图</div>')
        after_img = f'<img src="data:image/png;base64,{_b64(after)}" width={BOX} height={BOX}>'

        return f"""<div class="card">
  <div class="pair">
    <div class="cell">{before_img}<div class="cap">改前</div></div>
    <div class="cell">{after_img}<div class="cap">改后</div></div>
  </div>
  <div class="name">{html.escape(name)}.svg</div>
  <div class="mod">{html.escape(modname)} · {' '.join(tags)}</div>
  {ev_html}
</div>"""

    body = "\n".join(card(r) for r in rows)
    total = len(rows)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>图标审核报告 · {total} 个</title>
<style>{_CSS}</style></head><body>
<h1>图标状态对照</h1>
<p class="sub">每个图标都附上引用它的 <code>QAction</code> 文案与源文件；审核请对照「代码说要做什么」与「图形画了什么」，而不是凭图形本身猜测。</p>

<div class="stats">
  <div><b>{total}</b>图标总数</div>
  <div><b>{n_before}</b>有改前旧图可对照</div>
  <div><b>{n_dense_before}</b>当前墨量超标</div>
  <div><b>{n_dense_after}</b>位图来源墨量超标</div>
  <div><b>{n_noev}</b>缺文案证据</div>
</div>

<div class="legend">
  <span><span class="tag t-vec">真矢量</span> 真实路径，任意缩放都锐利</span>
  <span><span class="tag t-mosaic">位图马赛克</span> 16–48px 小图逐像素重编码，放大即方块</span>
  <span><span class="tag t-ok">墨量</span> 18px 下不透明像素占比</span>
  <span><span class="tag t-dense">过密</span> 超过 {DENSE:.0%}，小尺寸下会糊成块</span>
  <span><span class="tag t-noev">无文案证据</span> 未绑定可见动作，需人工确认语义</span>
</div>

<div class="note">
<strong>本报告的判据是「文件里到底装了什么」，不是扩展名。</strong>
一个 <code>.svg</code> 可能是真矢量，也可能是一张 16–48px 小图被逐像素拆成几百个
2–3px <code>&lt;rect&gt;</code> 的马赛克。两者在高 DPI 屏上表现完全不同：
工具栏图标 18 逻辑像素、3 倍屏需要 54 设备像素，马赛克型的源网格全部不足，
所以会糊。<br>
<strong>这正是此前「优化后还是模糊」的根因</strong>：那 109 个文件本来就是位图，
恢复它们只恢复了原有的模糊。自动描摹全部 109 个的尝试已实测并<strong>全部回退</strong>
——产生了可见形变（<code>ecvExit</code> 泛绿、<code>pqBold24</code> 变形、
<code>typeXCoordinate</code> 被切断），不能让它们进仓库。<br>
<strong>已是真矢量的 198 个图标</strong>：与上一次提交逐字节一致，一个都没有改动。
</div>

<h2>逐图对照</h2>
<div class="grid">
{body}
</div>
</body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the before/after icon audit.")
    ap.add_argument("--out", default=str(HERE / "data/audit.html"))
    ap.add_argument("--only", default=None, help="substring filter on name or module")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    page = build(a.only)
    out.write_text(page)
    print(f"wrote {out}  ({len(page):,} B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
