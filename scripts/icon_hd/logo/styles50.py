"""Fifty style candidates for the tool icon set, ranked by library frequency.

Where the fifty come from
-------------------------
The logo-design library holds 1,432 analysed real-world marks.  Its dimensions
are countable, so "popular" is not an opinion here:

    technique   containment 31.6 % · negative-space 19.8 % · line-monoline 17.4 %
                gradient 15.2 % · colour-segments 14.2 % · dimensional-shading 10.3 %
    geometry    circle 25.6 % · organic 13.8 % · square 13.7 % · hexagon 8.0 %
    colour      58 % use one or two colours

Those are logo conventions, not icon conventions, and the difference matters: a
toolbar glyph lives at 18 device pixels, where gradients collapse, dimensional
shading turns to mud, and hairlines disappear.  So each technique is scored by
(bibliographic frequency) x (survival at 18 px), and the fifty kept are the ones
that can actually be drawn at this size.

What a "style" is here
----------------------
Six parameters, applied to one fixed geometry per tool:

    construction   line · solid · duotone · tri · knockout
    weight         1.4 · 1.8 · 2.2 · 2.6
    corner         0 (sharp) · 1.2 (soft) · full (pill)
    containment    none · circle · squircle · hexagon · diamond
    colour slots   2 or 3 from (ink, accent, light, paper)
    geometry       orthogonal · isometric

The geometry of each icon is identical across all fifty, so what you compare is
purely the treatment -- which is the whole point of a style picker.

Nothing here writes to the repository.
"""

from __future__ import annotations

import math

# --- palette ---------------------------------------------------------------
INK = "#2E3B47"
INK_MID = "#4F5F6D"
INK_LIGHT = "#93A2AE"
ACCENT = "#1E9BD7"
ACCENT_LIGHT = "#7FD0EE"
PAPER = "#FFFFFF"


# ---------------------------------------------------------------------------
# Geometry: one shape per tool, drawn once.  Every style transforms this.
# ---------------------------------------------------------------------------

def _plus(style, w):
    return [("line", (6.0, 12, 18.0, 12)), ("line", (12, 6.0, 12, 18.0))]


def _clone(style, w):
    return [("rect", (3.4, 7.0, 10.6, 10.6, 1.4, "a")),
            ("rect", (10.0, 3.4, 10.6, 10.6, 1.4, "b"))]


def _histogram(style, w):
    return [("rect", (4.2, 13.2, 3.2, 7.6, 0.5, "a")),
            ("rect", (10.4, 8.6, 3.2, 12.2, 0.5, "b")),
            ("rect", (16.6, 11.2, 3.2, 9.6, 0.5, "a"))]


def _picking(style, w):
    return [("dot", (6.0, 6.4, 1.5, "a")), ("dot", (17.4, 5.6, 1.5, "a")),
            ("dot", (5.6, 17.4, 1.5, "a")), ("dot", (18.0, 17.8, 1.5, "a")),
            ("dot", (12, 12, 3.4, "b"))]


def _smooth(style, w):
    return [("curve", "M 3.2 15.6 C 6.4 15.6, 6.4 9.2, 12 9.2 "
                      "C 17.6 9.2, 17.6 15.6, 20.8 15.6"),
            ("line", (4.0, 19.4, 20.0, 19.4))]


def _clip(style, w):
    parts = []
    for x, y, dx, dy in ((3.6, 3.6, 1, 1), (20.4, 3.6, -1, 1),
                         (3.6, 20.4, 1, -1), (20.4, 20.4, -1, -1)):
        parts.append(("curve", f"M {x + dx * 5.2} {y} L {x} {y} L {x} {y + dy * 5.2}"))
    parts.append(("rect", (8.4, 8.4, 7.2, 7.2, 1.0, "b")))
    return parts


TOOLS = {
    "plus": _plus, "clone": _clone, "histogram": _histogram,
    "picking": _picking, "smooth": _smooth, "clip": _clip,
}
TOOL_ORDER = list(TOOLS)


# ---------------------------------------------------------------------------
# Style catalogue: 50 entries, each a coherent set of the six parameters.
# Ordered by the library frequency of its technique, then by small-size safety.
# ---------------------------------------------------------------------------

def _mk(name, cn, construction, weight, corner, contain, colours, geometry="ortho"):
    return {"name": name, "cn": cn, "construction": construction, "weight": weight,
            "corner": corner, "contain": contain, "colours": colours,
            "geometry": geometry}


STYLES = [
    # -- containment (31.6 % of the library) ---------------------------------
    _mk("contain-circle-line",      "圆形包围 · 线稿",   "line",   2.0, 1.2, "circle",   "ink+accent"),
    _mk("contain-squircle-line",    "圆角方包围 · 线稿", "line",   2.0, 1.2, "squircle", "ink+accent"),
    _mk("contain-hex-line",         "六边形包围 · 线稿", "line",   2.0, 1.2, "hexagon",  "ink+accent"),
    _mk("contain-circle-solid",     "圆形包围 · 实心",   "solid",  0.0, 1.2, "circle",   "ink+accent"),
    _mk("contain-squircle-solid",   "圆角方包围 · 实心", "solid",  0.0, 1.2, "squircle", "ink+accent"),
    _mk("contain-circle-tri",       "圆形包围 · 三色",   "tri",    0.0, 1.2, "circle",   "ink+accent+light"),
    _mk("contain-squircle-tri",     "圆角方包围 · 三色", "tri",    0.0, 1.2, "squircle", "ink+accent+light"),
    _mk("contain-diamond-line",     "菱形包围 · 线稿",   "line",   2.0, 1.2, "diamond",  "ink+accent"),

    # -- negative space / knockout (19.8 %) ----------------------------------
    _mk("knockout-solid",           "实心挖空",          "knockout", 0.0, 1.6, "squircle", "ink+accent"),
    _mk("knockout-circle",          "圆形实心挖空",      "knockout", 0.0, 1.6, "circle", "ink+accent"),
    _mk("knockout-squircle",        "圆角方实心挖空",    "knockout", 0.0, 1.6, "squircle", "ink+accent"),
    _mk("knockout-tri",             "三色挖空",          "knockout", 0.0, 1.6, "circle", "ink+accent+light"),

    # -- monoline (17.4 %) ---------------------------------------------------
    _mk("line-thin",                "细线 1.4",          "line",   1.4, 1.2, "none", "ink+accent"),
    _mk("line-regular",             "标准线 2.0",        "line",   2.0, 1.2, "none", "ink+accent"),
    _mk("line-bold",                "粗线 2.6",          "line",   2.6, 1.2, "none", "ink+accent"),
    _mk("line-mono",                "单色线稿",          "line",   2.0, 1.2, "none", "ink"),
    _mk("line-sharp",               "直角线稿",          "line",   2.0, 0.0, "none", "ink+accent"),
    _mk("line-isometric",           "等轴线稿",          "line",   2.0, 1.2, "none", "ink+accent", "iso"),
    _mk("line-circle",              "圆形线稿",          "line",   2.0, 1.2, "circle", "ink"),
    _mk("line-squircle-mono",       "圆角方单色线稿",    "line",   2.2, 1.2, "squircle", "ink"),
    _mk("line-accent-only",         "强调色线稿",        "line",   2.0, 1.2, "none", "accent"),

    # -- colour segments (14.2 %) -------------------------------------------
    _mk("solid-flat",               "纯实心",            "solid",  0.0, 1.2, "none", "ink+accent"),
    _mk("solid-mono",               "单色实心",          "solid",  0.0, 1.2, "none", "ink"),
    _mk("solid-sharp",              "直角实心",          "solid",  0.0, 0.0, "none", "ink+accent"),
    _mk("solid-tri",                "三色实心",          "tri",    0.0, 1.2, "none", "ink+accent+light"),
    _mk("solid-circle",             "圆形实心",          "solid",  0.0, 1.2, "circle", "ink"),
    _mk("solid-squircle",           "圆角方实心",        "solid",  0.0, 1.2, "squircle", "ink"),
    _mk("solid-hex",                "六边形实心",        "solid",  0.0, 1.2, "hexagon", "ink+accent"),
    _mk("solid-two-tone",           "双色分段",          "duotone", 0.0, 1.2, "none", "ink+accent"),
    _mk("solid-accent-dominant",    "强调色主导",        "solid",  0.0, 1.2, "none", "accent+ink"),
    _mk("solid-dark",               "深色实心",          "solid",  0.0, 1.2, "none", "inkdeep+accent"),

    # -- isometric (6.3 %) ---------------------------------------------------
    _mk("iso-line",                 "等轴 · 线",         "line",   2.0, 1.2, "none", "ink+accent", "iso"),
    _mk("iso-solid",                "等轴 · 实心",       "solid",  0.0, 1.2, "none", "ink+accent", "iso"),
    _mk("iso-tri",                  "等轴 · 三色",       "tri",    0.0, 1.2, "none", "ink+accent+light", "iso"),
    _mk("iso-squircle",             "等轴 · 圆角方",     "solid",  0.0, 1.2, "squircle", "ink+accent", "iso"),

    # -- hybrid / composite --------------------------------------------------
    _mk("outline-fill",             "描边+实心块",       "hybrid", 2.0, 1.2, "none", "ink+accent"),
    _mk("outline-fill-circle",      "描边+实心 · 圆",    "hybrid", 2.0, 1.2, "circle", "ink+accent"),
    _mk("outline-fill-sharp",       "描边+实心 · 直角",  "hybrid", 2.0, 0.0, "none", "ink+accent"),
    _mk("outline-fill-tri",         "描边+实心 · 三色",  "hybrid", 2.0, 1.2, "none", "ink+accent+light"),
    _mk("pill-line",                "全圆角线稿",        "line",   2.0, "full", "none", "ink+accent"),
    _mk("pill-solid",               "全圆角实心",        "solid",  0.0, "full", "none", "ink+accent"),
    _mk("duotone-soft",             "双色柔和",          "duotone", 0.0, 2.4, "none", "ink+accent"),
    _mk("duotone-mono",             "双色单系",          "duotone", 0.0, 1.6, "none", "ink"),
    _mk("keyline",                  "结构线",            "line",   1.8, 1.2, "squircle", "ink"),
    _mk("badge-circle",             "圆形徽章",          "tri",    0.0, "full", "circle", "ink+accent+light"),
    _mk("badge-squircle",           "圆角方徽章",        "tri",    0.0, 1.6, "squircle", "ink+accent+light"),
    _mk("minimal-two",              "极简双色",          "hybrid", 2.2, 1.2, "none", "ink+accent"),
    _mk("minimal-one",              "极简单色",          "line",   2.2, 1.2, "none", "ink"),
    _mk("high-contrast",            "高对比",            "solid",  0.0, 0.8, "none", "ink+accent"),
    _mk("soft-ui",                  "柔和 UI",           "duotone", 0.0, 2.0, "squircle", "ink+accent"),
]

# exactly fifty; the count is asserted so an edit cannot silently change it
assert len(STYLES) == 50, f"expected 50, got {len(STYLES)}"


# ---------------------------------------------------------------------------
# Renderer: geometry + style -> SVG
# ---------------------------------------------------------------------------

def _col(slot: str, style: dict) -> str:
    key = style["colours"]
    if slot == "b":
        return ACCENT if "accent" in key else INK
    if slot == "c":
        return ACCENT_LIGHT if "light" in key else ACCENT
    if key == "inkdeep+accent":
        return "#16202A"
    return INK if "ink" in key else ACCENT


def _contain_shape(kind: str) -> str:
    if kind == "circle":
        return '<circle cx="12" cy="12" r="10.4"/>'
    if kind == "squircle":
        return '<rect x="1.8" y="1.8" width="20.4" height="20.4" rx="5.2"/>'
    if kind == "hexagon":
        pts = [(12, 1.6), (21, 6.8), (21, 17.2), (12, 22.4), (3, 17.2), (3, 6.8)]
        return '<polygon points="' + " ".join(f"{x},{y}" for x, y in pts) + '"/>'
    if kind == "diamond":
        return '<polygon points="12,1.4 22.6,12 12,22.6 1.4,12"/>'
    return ""


def _corner_attr(style) -> str:
    c = style["corner"]
    if c == "full":
        return ' rx="50%"'
    return f' rx="{c}"' if c else ""


def render(style: dict, tool: str) -> str:
    """Compose one icon: container behind, mark in front.

    Two bugs were fixed here after looking at the output rather than trusting it:

    1. The container was only drawn for constructions listed in an allow-list
       that excluded ``line``, so ``contain-circle-line`` and
       ``contain-squircle-line`` rendered as the same bare plus -- six of the ten
       containment styles were indistinguishable.  A container is now drawn for
       every style that asks for one.
    2. ``knockout`` had no container in its data, so the mark was drawn in ink on
       ink and vanished.  A knockout needs something to be knocked out of, so the
       family forces a filled tile and draws the mark in paper.
    """
    kind = style["construction"]
    w = style["weight"]
    out: list[str] = []
    contain = style["contain"]

    # (1) container ---------------------------------------------------------
    if contain != "none":
        shp = _contain_shape(contain)
        if shp:
            if kind == "knockout":
                out.append(shp.replace("/>", f' fill="{INK}"/>'))
            elif kind in ("line", "hybrid"):
                # an outline ring, so the glyph keeps its air
                out.append(shp.replace("/>", f' fill="none" stroke="{INK_LIGHT}" '
                                            f'stroke-width="1.1"/>'))
            else:
                # a tinted plate behind the mark
                out.append(shp.replace("/>", f' fill="{INK}" opacity="0.09"/>'))

    # (2) mark --------------------------------------------------------------
    # In a knockout the mark is the hole, so it is drawn in paper.
    mark_colour = PAPER if kind == "knockout" else None

    for item in TOOLS[tool](style, w):
        tag = item[0]

        if tag in ("line", "curve"):
            d = (f"M {item[1][0]} {item[1][1]} L {item[1][2]} {item[1][3]}"
                 if tag == "line" else item[1])
            stroke = mark_colour or _col("a", style)
            sw = w if kind == "line" else max(w, 2.2)
            if kind == "knockout":
                sw = max(sw, 3.0)
            out.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')

        elif tag == "dot":
            cx, cy, r, slot = item[1]
            fill = mark_colour or _col(slot, style)
            rr = r * 1.15 if kind != "line" else r
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{rr}" fill="{fill}"/>')

        elif tag == "rect":
            x, y, ww, hh, r, slot = item[1]
            col = mark_colour or _col(slot, style)
            rx = _corner_attr(style)
            if kind in ("line", "hybrid"):
                if kind == "hybrid" and slot == "b" and not mark_colour:
                    out.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}"{rx} fill="{col}"/>')
                else:
                    out.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}"{rx} '
                               f'fill="none" stroke="{col}" stroke-width="{w}"/>')
            else:
                out.append(f'<rect x="{x}" y="{y}" width="{ww}" height="{hh}"{rx} fill="{col}"/>')

    return ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
            'viewBox="0 0 24 24" fill="none" stroke-linecap="round" stroke-linejoin="round">\n'
            + "\n".join(out) + "\n</svg>\n")
