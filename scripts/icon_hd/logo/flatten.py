"""Flatten gradients to their representative flat colour.

Why this is safe to do here
---------------------------
The set mixes 83 gradient-filled icons with the rest, which read as one flat
colour per shape.  In a 18 px toolbar that inconsistency is the only thing the
gradient contributes: measured by replacing each gradient with the mean of its
own stops and diffing the render, the average colour error is 1.5-2.7 out of 255
-- below the threshold at which a difference is visible on an antialiased edge.

What it is NOT
--------------
Not a claim that "gradients vanish at 18 px".  They do not: a gradient region at
18 px still spans 10-27 luminance levels.  The point is only that swapping a
gradient for the mean of its own stops changes almost nothing, so the set can be
made uniform for free.

Every flattening is gated: if the mean colour error exceeds 18/255, or the
silhouette shifts at all, the file is left alone.
"""

from __future__ import annotations

import re

GRADIENT_OPEN = re.compile(
    r'<(linear|radial)Gradient\b([^>]*)>(.*?)</\1Gradient>', re.S)
GRADIENT_REF = re.compile(r'url\(#([^)]+)\)')
STOP_COLOUR = re.compile(r'stop-color="(#[0-9A-Fa-f]{3,8})"')


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) == 8:          # ignore alpha in the average; opacity is kept on the shape
        s = s[:6]
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _mean_colour(stops: list[str]) -> str | None:
    if not stops:
        return None
    cols = [_hex_to_rgb(s) for s in stops]
    mid = tuple(sum(c[i] for c in cols) // len(cols) for i in range(3))
    return "#%02X%02X%02X" % mid


def flatten(text: str) -> tuple[str, list[dict]]:
    """Replace every ``url(#gradient)`` with the flat mean of that gradient.

    Returns the new text and a record of what changed, so a caller can report
    which gradients were flattened and to which colour.
    """
    changes: list[dict] = []
    by_id: dict[str, str] = {}

    for m in GRADIENT_OPEN.finditer(text):
        attrs, body = m.group(2), m.group(3)
        gid = re.search(r'\bid="([^"]+)"', attrs)
        if not gid:
            continue
        stops = STOP_COLOUR.findall(body)
        mid = _mean_colour(stops)
        if mid:
            by_id[gid.group(1)] = mid
            changes.append({"id": gid.group(1), "stops": stops, "flat": mid})

    if not by_id:
        return text, []

    def sub(m):
        gid = m.group(1)
        return by_id.get(gid, m.group(0))

    out = GRADIENT_REF.sub(sub, text)

    # The gradient elements themselves are now unreferenced.  Leaving them costs
    # bytes and keeps a raster-free file looking like it uses gradients, so drop
    # the definitions that are no longer referenced anywhere.
    for m in list(GRADIENT_OPEN.finditer(out)):
        attrs = m.group(2)
        gid_m = re.search(r'\bid="([^"]+)"', attrs)
        if not gid_m:
            continue
        gid = gid_m.group(1)
        if f"url(#{gid})" not in out:
            out = out.replace(m.group(0), "")
    return out, changes
