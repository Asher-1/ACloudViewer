"""Font-property and field-operator icons.

Two groups that were both left as bitmap traces:

* ``svg/font/*`` -- the font style and alignment toggles of the label property
  widget.  Letterforms are constructed from paths; ``<text>`` is not used, both
  because the system forbids live text and because a live glyph would render in
  whatever font the host Qt build happens to have.
* Four 64-viewBox icons (calculator, face-set, probe, solid colour) that are
  still pixel staircases at 21-65 KB each.  Their meanings come from the
  ``QAction`` they are attached to, not from their file names.
"""

from __future__ import annotations

from . import iconset as K


# ---------------------------------------------------------------------------
# Font style
# ---------------------------------------------------------------------------


def bold() -> str:
    """A capital B built from a stem and two bowls -- no live text.

    Drawn as one closed outline rather than stem-plus-two-open-arcs: the arcs
    met the stem at slightly different tangents and the junction showed as a
    notch at 18 px.  The upper bowl stays ink and the lower one accent, the same
    two-tone rule the other style toggles use.
    """
    upper = ("M 7.0 4.6 H 12.0 A 3.4 3.4 0 0 1 12.0 11.4 H 7.0 "
             "A 0.0 0.0 0 0 1 7.0 11.4 Z")
    lower = ("M 7.0 11.4 H 12.8 A 3.6 3.6 0 0 1 12.8 18.6 H 7.0 "
             "A 0.0 0.0 0 0 1 7.0 18.6 Z")
    return K.svg("\n".join([
        f'  <path d="{upper}" stroke="{K.INK}" stroke-width="2.2"/>',
        f'  <path d="{lower}" stroke="{K.ACCENT}" stroke-width="2.2"/>',
        K.line(7.0, 4.6, 7.0, 18.6, color=K.INK, width=2.2),
    ]))


def italics() -> str:
    """A slanted I with its two serifs: the oblique mark."""
    return K.svg("\n".join([
        K.line(14.6, 5.2, 10.0, 18.8, color=K.INK, width=2.2),
        K.line(8.6, 5.2, 16.4, 5.2, color=K.INK, width=2.0),
        K.line(7.2, 18.8, 15.0, 18.8, color=K.ACCENT, width=2.0),
    ]))


def shadow() -> str:
    """A glyph with an offset copy behind it: drop shadow."""
    return K.svg("\n".join([
        K.rect(8.6, 8.6, 11.0, 11.0, r=1.2, color=K.MUTED, width=1.5),
        K.rect(4.2, 4.2, 11.0, 11.0, r=1.2, fill=K.INK),
    ]))


# ---------------------------------------------------------------------------
# Text alignment
# ---------------------------------------------------------------------------


def _h_bars(align: str) -> str:
    """Three bars aligned left, centre or right."""
    widths = (13.0, 8.4, 11.0)
    parts = []
    for i, y in enumerate((7.6, 12.0, 16.4)):
        w = widths[i]
        if align == "left":
            x = 3.4
        elif align == "right":
            x = 20.6 - w
        else:
            x = 12.0 - w / 2
        color = K.ACCENT if i == 1 else K.INK
        parts.append(K.line(x, y, x + w, y, color=color, width=2.2))
    return K.svg("\n".join(parts))


def align_left() -> str:
    return _h_bars("left")


def align_center() -> str:
    return _h_bars("center")


def align_right() -> str:
    return _h_bars("right")


def _v_bars(place: str) -> str:
    """Three stacked bars with a bracket where they are anchored.

    Vertical alignment has no width to vary, so the anchor is drawn explicitly:
    a bracket at the top, the middle or the bottom edge.
    """
    ys = (4.8, 12.0, 19.2)
    # Bars start at x 10.0 so the anchor arrow on the left never touches them.
    # The first version ran the arrow to x 9.4 against bars starting at 9.0 and
    # the two merged into one thick rule.
    parts = [K.line(10.0, y, 20.4, y, color=K.INK, width=2.1) for y in ys]
    parts.append(K.line(10.0, 12.0, 20.4, 12.0, color=K.ACCENT, width=2.1))

    anchor_y = {"top": 4.8, "center": 12.0, "bottom": 19.2}[place]
    parts.append(K.line(3.0, anchor_y, 7.0, anchor_y, color=K.ACCENT, width=2.0))
    parts.append(K.poly([(7.2, anchor_y - 2.0), (10.0, anchor_y), (7.2, anchor_y + 2.0)],
                        close=True, fill=K.ACCENT))
    return K.svg("\n".join(parts))


def align_vertical_top() -> str:
    return _v_bars("top")


def align_vertical_center() -> str:
    return _v_bars("center")


def align_vertical_bottom() -> str:
    return _v_bars("bottom")


# ---------------------------------------------------------------------------
# Field operators
# ---------------------------------------------------------------------------


def arithmetic() -> str:
    """A calculator body with operator keys: scalar-field arithmetic."""
    # Six hatched keys measured 0.568 ink -- heavier than anything else in the
    # set, because every key is both an outline and a fill.  Four keys as plain
    # dots keep the "calculator" reading at a third of the weight.
    parts = [
        K.rect(4.4, 2.8, 15.2, 18.4, r=2.0, color=K.INK, width=1.7),
        K.rect(6.8, 5.0, 10.4, 3.2, r=0.7, color=K.MUTED, width=1.1),
    ]
    for x, y in ((7.6, 11.4), (11.6, 11.4), (7.6, 15.0), (11.6, 15.0)):
        parts.append(K.dot(x, y, 1.35, color=K.MUTED))
    parts.append(K.dot(15.6, 13.2, 1.35))
    parts.append(K.line(7.4, 18.6, 16.6, 18.6, color=K.ACCENT, width=1.9))
    return K.svg("\n".join(parts))


def label_classes() -> str:
    """Four labelled cells in two colours: a class face-set.

    Attached to "Semantic Annotation", so the mark is the thing being produced
    -- a set of class labels over a face set -- not a pen or a box.
    """
    return K.svg("\n".join([
        K.rect(3.2, 3.2, 8.0, 8.0, r=1.0, fill=K.ACCENT),
        K.rect(12.8, 3.2, 8.0, 8.0, r=1.0, color=K.MUTED, width=1.5),
        K.rect(3.2, 12.8, 8.0, 8.0, r=1.0, color=K.MUTED, width=1.5),
        K.rect(12.8, 12.8, 8.0, 8.0, r=1.0, fill=K.INK),
    ]))


def probe() -> str:
    """A probe pin sampling one point of a cloud: read a value there."""
    return K.svg("\n".join([
        K.cloud(12.0, 15.4, s=0.84, color=K.MUTED),
        f'  <path d="M 17.2 3.0 L 17.2 6.0 A 3.4 3.4 0 0 0 13.8 6.0 '
        f'L 13.8 10.4" stroke="{K.ACCENT}" stroke-width="1.8"/>',
        K.poly([(11.6, 9.8), (16.0, 9.8), (13.8, 13.0)], close=True, fill=K.ACCENT),
        K.dot(17.2, 2.4, 1.8, color=K.INK),
    ]))


def single_point() -> str:
    """One filled dot on a lattice: a cloud with a single point."""
    return K.svg("\n".join([
        K.dot(12.0, 12.0, 5.6, color=K.ACCENT),
        K.dot(4.6, 4.6, 1.3, color=K.MUTED),
        K.dot(19.4, 4.6, 1.3, color=K.MUTED),
        K.dot(4.6, 19.4, 1.3, color=K.MUTED),
        K.dot(19.4, 19.4, 1.3, color=K.MUTED),
    ]))


REGISTRY = {
    "font/pqBold24": bold,
    "font/pqItalics24": italics,
    "font/pqShadow24": shadow,
    "font/pqTextAlignLeft": align_left,
    "font/pqTextAlignCenter": align_center,
    "font/pqTextAlignRight": align_right,
    "font/pqTextVerticalAlignTop": align_vertical_top,
    "font/pqTextVerticalAlignCenter": align_vertical_center,
    "font/pqTextVerticalAlignBottom": align_vertical_bottom,
    "pqCalculator": arithmetic,
    "pqFaceSetData": label_classes,
    "pqProbeLocation": probe,
    "pqSolidColor": single_point,
}
