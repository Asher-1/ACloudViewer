"""toolMainFilterByValue -- rebuilt from the original's measured pixel grid.

Why this took three attempts
----------------------------
Each rejection had a different cause, and all three are worth recording because
they are the three ways this kind of redraw goes wrong.

v1 replaced the composition with a bracket-and-veil diagram of my own.  It read
cleanly and said something weaker.

v2 restored the words but sized them by eye, at 2.9 units.  The source's letters
are 4 units on a 21-unit canvas -- roughly 40 % larger than I drew them, which is
why they looked like scratches.

v3 then invented a "lit sub-interval" inside the bar, from a misreading of the
red pixels.  **There is no such mark.**  The red cells at x 12-17 are simply the
warm end of the rainbow: sampling the bar row gives
#F58F0D, #EF5016, #F52E2F, #E71D5A, #CC0E82 straight through.  A design that
adds a mark the source does not have is not a faithful redraw, so it is gone.

What the original actually contains
-----------------------------------
Read cell by cell from a 21x21 render (the file is a one-pixel-per-`<rect>`
trace of a bitmap, not a vector drawing):

    y  0        single red cell at x 7        the dot of the "i"
    y  1        pale rule, x 1-11             dimension line above "min"
    y  2-5      red letters, x 1-12           "min"
    y  6-8      grey leader at x 3-4          word -> bar
    y  9-12     the bar, x 1-20               frame rows at 9 and 12
    y 13-15     grey leader at x 16-17        bar -> word
    y 16-20     red letters, x 6-20           "max"

Four marks, not five: two rules, two leaders, one bar, two words.

The bar's graduations are the dark cells at its ends (#8C8C8C, #6F6D6F, #484848
on the left; #472041, #6F6D6F on the right).  They are scale ends, not a second
interval.

Palette: the source's red is not in the language
------------------------------------------------
Across the eight approved samples -- 34,953 opaque pixels -- there are **zero**
red-dominant pixels.  The source's four reds (#f52e2f, #ef7272, #eb8c8c,
#e6afaf) are antialiasing steps of one text colour, not a palette.

So the words are set in the samples' ink #6F6F59, which is also what the approved
MESH/POINTS lettering uses, and the ramp is rebuilt entirely from measured sample
colours:

    #D0E5F6  2.72 %     #A7D28C  3.18 %     #C1A561  2.13 %
    #9AC1E0  0.82 %     #72BE44  1.53 %     #C4813E 14.27 %
    #2896D3  0.64 %     #F3C017  0.92 %

cold to warm, which is the order a scalar-field legend needs anyway.  The first
stop is the samples' most common colour and the last is their most common warm
one, so the ramp sits inside the family rather than borrowing a generic rainbow.

Layout
------
The leaders both drop at a graduation and both sit *outboard* of their word, so
the pair is symmetric: leader left of "min" at the bar's left end, leader right
of "max" at its right end.  The source puts its leader inside the word, which is
a bitmap artefact -- at 21 px it had no room to do otherwise.

Style (fixed baseline)
----------------------
    outline   #6F6F59 on every form
    ink       #6F6F59 for words and rules
    accent    #2896D3 for the two leaders, matching the approved rulers
    span      92 % of the canvas

Usage
-----
    python3 -m scripts.icon_hd.logo.icon filter --show
    python3 -m scripts.icon_hd.logo.icon filter --sheet out.png
    python3 -m scripts.icon_hd.logo.icon filter --write
"""

from __future__ import annotations


from . import paraview_style as P


# Every stop measured from the eight samples; see the docstring for frequencies.
RAMP = [
    (0.00, "#D0E5F6"),
    (0.18, "#9AC1E0"),
    (0.34, "#2896D3"),
    (0.50, "#A7D28C"),
    (0.64, "#72BE44"),
    (0.80, "#F3C017"),
    (0.90, "#C1A561"),
    (1.00, "#C4813E"),
]

LEADER = P.BLUE_STRONG

# --- bar --------------------------------------------------------------------
BAR_X0, BAR_X1 = 1.2, 22.8
BAR_Y0, BAR_Y1 = 9.8, 16.6
BAR_H = BAR_Y1 - BAR_Y0

# Graduations sit *inside* the bar, just in from each end, the way the source's
# dark cells do at x 1-3 and 17-19.  Drawing them outside the bar as free-floating
# ticks (v5) left a meaningless stub of ramp beyond each mark; inside, the bar
# ends where the scale ends and the marks read as the scale's ends.
TICK_L, TICK_R = BAR_X0 + 0.9, BAR_X1 - 0.9

# --- letterforms ------------------------------------------------------------
#
# Weight is the measurement that was wrong for three versions running.  The
# source's letters are 1 unit on a 21-unit canvas = 4.8 % of the canvas.  Mine
# were 0.6 on 24 = 2.5 %, less than half.  That is why they read as scratches no
# matter how carefully the shapes were drawn -- and it is also why the icon
# scored 30 % ink against the samples' 41-69 %: the samples are solid pictograms,
# and thin strokes cannot reach that density.
#
# At 1.0 the adjacent letters' strokes just meet, which also takes the icon from
# 7 disconnected components to 3.  The source at 21 px is the same: its letters
# barely separate (see the pixel grid in the docstring).
X_HEIGHT = 3.4
STROKE = 0.95
GAP = 1.50          # clearance 0.55 after the stroke
WIDTHS = {"m": 4.49, "i": 0.63, "n": 4.26, "a": 4.14, "x": 3.91}


def _glyph(ch: str, x: float, yt: float, yb: float,
           dot_y: float | None = None) -> str:
    """One lowercase letter between x-height top and baseline.

    The arch rise is 23 % of the x-height, so the counters are open at 32 px and
    above.  At 24 px they settle into a comb -- and so does the source at 21 px,
    which is the honest ceiling for lowercase at toolbar size.  The bar, the
    graduations and the two rules carry the meaning at every size; the words
    confirm it.
    """
    h = yb - yt
    w = WIDTHS.get(ch, 3.5)
    ar = h * 0.23
    sw = STROKE
    d = {
        "m": f"M {x:.2f} {yb:.2f} L {x:.2f} {yt + ar:.2f} "
             f"Q {x + w * 0.25:.2f} {yt - ar:.2f} {x + w * 0.5:.2f} {yt + ar:.2f} "
             f"L {x + w * 0.5:.2f} {yb:.2f} "
             f"M {x + w * 0.5:.2f} {yt + ar:.2f} "
             f"Q {x + w * 0.75:.2f} {yt - ar:.2f} {x + w:.2f} {yt + ar:.2f} "
             f"L {x + w:.2f} {yb:.2f}",
        "n": f"M {x:.2f} {yb:.2f} L {x:.2f} {yt + ar:.2f} "
             f"Q {x + w * 0.5:.2f} {yt - ar:.2f} {x + w:.2f} {yt + ar:.2f} "
             f"L {x + w:.2f} {yb:.2f}",
        "a": f"M {x + w * 0.87:.2f} {yt + ar * 1.5:.2f} "
             f"Q {x + w * 0.45:.2f} {yt - ar * 0.3:.2f} {x + w * 0.17:.2f} {yt + h * 0.45:.2f} "
             f"Q {x + w * 0.01:.2f} {yt + h * 0.86:.2f} {x + w * 0.40:.2f} {yb:.2f} "
             f"L {x + w * 0.87:.2f} {yb:.2f} "
             f"L {x + w * 0.87:.2f} {yt + h * 0.22:.2f}",
        "x": f"M {x:.2f} {yt + ar * 1.5:.2f} L {x + w:.2f} {yb:.2f} "
             f"M {x + w:.2f} {yt + ar * 1.5:.2f} L {x:.2f} {yb:.2f}",
    }.get(ch)
    if ch == "i":
        cx = x + w * 0.5
        # The dot sits ABOVE the dimension rule, not below it -- that is the
        # source's own arrangement: its pixel grid has the dot on row 0, the
        # rule on row 1, and the letters from row 2.  Putting the dot under the
        # rule (the obvious choice) drives the rule straight through it.
        dy = dot_y if dot_y is not None else yt - 1.35
        return "\n".join([
            f'  <path d="M {cx:.2f} {yt + ar:.2f} L {cx:.2f} {yb:.2f}" '
            f'fill="none" stroke="{P.INK}" stroke-width="{sw}" '
            f'stroke-linecap="round"/>',
            f'  <circle cx="{cx:.2f}" cy="{dy:.2f}" r="0.40" '
            f'fill="{P.INK}" stroke="none"/>',
        ])
    if d is None:
        return ""
    return (f'  <path d="{d}" fill="none" stroke="{P.INK}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def _word_width(word: str) -> float:
    return (sum(WIDTHS.get(c, 3.5) for c in word)
            + GAP * (len(word) - 1))


def _word(word: str, x: float, yt: float, yb: float,
          dot_y: float | None = None) -> str:
    """Letters laid out left to right from ``x``."""
    out, cur = [], x
    for ch in word:
        out.append(_glyph(ch, cur, yt, yb, dot_y))
        cur += WIDTHS.get(ch, 3.5) + GAP
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the four marks
# ---------------------------------------------------------------------------


def _bar() -> str:
    return (f'  <rect x="{BAR_X0}" y="{BAR_Y0}" width="{BAR_X1 - BAR_X0:.2f}" '
            f'height="{BAR_H:.2f}" rx="0.6" fill="url(#ramp)" '
            f'stroke="{P.INK}" stroke-width="{P.STROKE}"/>')


def _tick(x: float) -> str:
    """A graduation inside the bar, running the full height of the ramp.

    Full height rather than a short notch: against #D0E5F6 at the cold end a
    short mark disappears, and against #C4813E at the warm end it merges.  A
    full-height line reads against every stop.
    """
    return (f'  <path d="M {x:.2f} {BAR_Y0:.2f} '
            f'L {x:.2f} {BAR_Y1:.2f}" fill="none" '
            f'stroke="{P.INK}" stroke-width="0.95"/>')


def _leader(x: float, y0: float, y1: float) -> str:
    """The short line joining a word to its graduation.

    Its ends are computed from the word's baseline and the bar's edge, so it can
    never float free -- the v2 leaders started at a rule and stopped in space,
    which is what made them read as stray sticks.
    """
    return (f'  <path d="M {x:.2f} {y0:.2f} L {x:.2f} {y1:.2f}" fill="none" '
            f'stroke="{LEADER}" stroke-width="0.7" stroke-linecap="butt"/>')


def _rule(x0: float, x1: float, y: float) -> str:
    """The dimension line above "min" / below "max".

    It starts at the graduation's x, not at the word's edge, so rule and leader
    meet and read as one bracket around the label.
    """
    return (f'  <path d="M {x0:.2f} {y:.2f} L {x1:.2f} {y:.2f}" fill="none" '
            f'stroke="{P.INK}" stroke-width="0.6" stroke-linecap="round"/>')


def build() -> str:
    """min annotated at the scale's left end, max at its right, ramp between.

    The chain is the source's own construction, and it was the last thing I got
    wrong.  In the original the four marks *touch*: the rule sits at y 1, the
    letters start at y 2, the leader runs y 6-8, the bar begins at y 9.  Every
    link abuts the next, so the whole drawing is one connected mass -- which is
    why the source scores `components = 1`.

    My previous versions spaced the marks apart with air around each one, which
    is cleaner to look at in isolation but scores 6-8 components and reads as
    four unrelated objects near the bar rather than as one annotated scale.
    ParaView's icons are solid pictograms; the samples' median component count
    is 1.  So the marks are abutted, exactly as the source abuts them.

    Chain:  rule -> letters (tops touch) -> leader -> bar -> leader -> letters
            -> rule.
    """
    defs = ('<linearGradient id="ramp" x1="0" y1="0" x2="1" y2="0">'
            + "".join(f'<stop offset="{o:.2f}" stop-color="{c}"/>'
                      for o, c in RAMP)
            + "</linearGradient>")

    min_w = _word_width("min")
    max_w = _word_width("max")

    # Where the rule goes.  This is the bug that broke v12: a quadratic Bezier
    # peaks at *half* its control offset, so an arch drawn with the control point
    # at `yt - ar` still reaches `yt`.  With a 0.95 stroke the ink top is
    # `yt - 0.475`, which is ABOVE a rule placed at `yt - 0.28` -- the rule was
    # being drawn straight through the tops of the arches, turning both words
    # into fences.
    #
    # So the rule is placed relative to the real ink, `yt - STROKE/2`, with a
    # small clearance.  The letters then read as letters and the rule reads as a
    # dimension line over them.
    gap_to_rule = STROKE / 2 + 0.30

    min_x = TICK_L + 1.15
    min_top = 3.4
    min_base = min_top + X_HEIGHT
    rule_hi = min_top - gap_to_rule

    max_x = TICK_R - 1.15 - max_w
    max_base = BAR_Y1 + 3.4
    max_top = max_base - X_HEIGHT
    rule_lo = max_base + gap_to_rule

    parts = [
        _bar(),
        _tick(TICK_L),
        _tick(TICK_R),

        # upper-left: rule, word beneath it, leader from the word's outer stem
        # down onto the bar.  All three abut.
        _rule(TICK_L, min_x + min_w, rule_hi),
        _word("min", min_x, min_top, min_base, dot_y=rule_hi - 1.05),
        _leader(TICK_L, min_base - 0.05, BAR_Y0),

        # lower-right, mirrored: leader up from the bar, word, rule beneath.
        _leader(TICK_R, BAR_Y1, max_base + 0.05),
        _word("max", max_x, max_top, max_base),
        _rule(max_x - 0.0, TICK_R, rule_lo),
    ]
    return P.svg("\n".join(parts), defs=defs)
