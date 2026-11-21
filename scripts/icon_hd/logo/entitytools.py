"""Entity creation, clipping and cloning in the approved ParaView palette.

The source AddConstSF is a plain plus, also reused to add a polyline vertex.
Keep that shared meaning.  The clipping family shares the same volume and
projection: one horizontal slice, or three repeated slices, with a green plus
only on the two actions that create output entities.  Export here creates a
cloud in the database, so a disk or file-download arrow would be misleading.

Clone keeps the original pair of sheep: the repeated subject is the visual
identifier of this action. Both sheep use exactly the same geometry and finish.
Only SVG geometry is authored here; preview and installation use icon.py.
"""

from __future__ import annotations

from . import paraview_style as P


def _plus(cx: float, cy: float, radius: float, half_stem: float,
          fill: str, sw: float = P.STROKE) -> str:
    a, b, r = cx, cy, radius
    s = half_stem
    return P.fill(
        f"M {a-s:g} {b-r:g} H {a+s:g} V {b-s:g} H {a+r:g} "
        f"V {b+s:g} H {a+s:g} V {b+r:g} H {a-s:g} V {b+s:g} "
        f"H {a-r:g} V {b-s:g} H {a-s:g} Z", fill, sw=sw)


def add_constant_sf() -> str:
    """The original plus with a clean outline and gently shaded green body."""
    defs = P.linear("add", [(0, P.GREEN_LT), (0.48, P.GREEN),
                            (1, P.GREEN_DK)])
    body = '<title>Add constant scalar field</title>\n'
    body += _plus(12, 12, 10.5, 2.9, "url(#add)")
    return P.svg(body, defs=defs)


def _clipping(*, multiple: bool = False, export: bool = False) -> str:
    # An oblique box. Every corresponding back vertex is offset by (4.8, -6).
    # Its edges and slice corners therefore stay in one coherent projection.
    left, right, top, bottom = 2.4, 16.8, 8.6, 21.5
    dx, dy = 4.8, -6.0
    gradients = [
        P.linear("glass", [(0, P.GREY_LT), (1, P.GREY)]),
        P.linear("slice", [(0, P.BLUE_PALE), (0.4, P.BLUE_MID),
                           (1, P.BLUE_STRONG)]),
    ]
    if export:
        gradients.append(P.linear("add", [(0, P.GREEN_LT), (0.5, P.GREEN),
                                           (1, P.GREEN_DK)]))
    defs = "\n".join(gradients)
    title = ("Export repeated clipping slices" if multiple else
             "Export clipping selection" if export else "Clipping box")
    body = [
        f"<title>{title}</title>",
        P.poly([(left, top), (left+dx, top+dy), (right+dx, top+dy),
                (right, top)], P.GREY_LT),
        P.poly([(right, top), (right+dx, top+dy),
                (right+dx, bottom+dy), (right, bottom)], P.GREY),
        P.rect(left, top, right-left, bottom-top, "url(#glass)", r=0),
    ]
    levels = (12.8, 16.1, 19.4) if multiple else (16.1,)
    for y in levels:
        # A filled horizontal section, including a narrow front edge for depth.
        body.extend([
            P.fill(f"M {left} {y} L {left+dx} {y+dy} "
                   f"H {right+dx} L {right} {y} Z", "url(#slice)", sw=0.55),
            P.fill(f"M {left} {y} H {right} V {y+0.65} H {left} Z",
                   P.BLUE_STRONG, outline=None),
        ])
    # The translucent right wall sits in front of the section's far corner.
    body.append(
        f'<path d="M {right} {top} L {right+dx} {top+dy} '
        f'V {bottom+dy} L {right} {bottom} Z" '
        f'fill="{P.BLUE_PALE}" fill-opacity="0.52" stroke="none"/>')
    body.extend([
        P.stroke(f"M {left} {top} L {left+dx} {top+dy} H {right+dx} "
                 f"V {bottom+dy} L {right} {bottom} H {left} Z"),
        P.stroke(f"M {left} {top} H {right} L {right+dx} {top+dy} "
                 f"M {right} {top} V {bottom}"),
    ])
    if export:
        # Stay inside the volume's ink bounds so all three family members
        # retain exactly the same scale and position after canvas fitting.
        body.append(_plus(5.9, 6.0, 3.4, 1.1, "url(#add)", sw=0.75))
    return P.svg("\n".join(body), defs=defs)


def clipping_box() -> str:
    return _clipping()


def clipping_multi_export() -> str:
    return _clipping(multiple=True, export=True)


def clipping_single_export() -> str:
    return _clipping(export=True)


def clone_entity() -> str:
    """Two identical left-facing sheep, preserving the source icon's identity."""
    defs = "\n".join([
        P.linear("wool", [(0, "#FAFAF6"), (0.5, P.GREY_LT),
                          (1, P.GREY)]),
        P.linear("face", [(0, P.GREY_MID), (1, P.GREY_DK)]),
    ])
    # One continuous fleece silhouette avoids seams between the wool lobes.
    # Feet remain broad enough to distinguish the subject at toolbar sizes.
    sheep = "\n".join([
        P.fill("M 8.1 8 H 10 V 12 H 8.1 Z", P.GREY_DK, sw=0.65),
        P.fill("M 17.1 8 H 19 V 12 H 17.1 Z", P.GREY_DK, sw=0.65),
        P.fill("M 6.5 4.5 C 6.1 3.1 7.5 2.2 8.7 2.6 "
               "C 9.2 1.1 11.1 1.0 12 2.2 "
               "C 13.2 1.1 15 1.5 15.5 2.7 "
               "C 17 1.9 18.7 2.6 18.9 3.9 "
               "C 20.8 3.8 21.6 5.1 20.8 6.5 "
               "C 21.5 7.9 20.6 9.3 19.1 9.3 "
               "C 18.2 10.5 16.8 10.6 15.7 9.7 "
               "C 14.7 10.8 13 10.8 12 9.8 "
               "C 10.7 10.8 9 10.3 8.7 9.2 "
               "C 6.9 9.4 5.8 8.2 6.3 6.7 "
               "C 5.6 6.1 5.7 5 6.5 4.5 Z", "url(#wool)", sw=0.75),
        # A drooping ear and elongated dark muzzle carry the sheep identity.
        P.fill("M 6.5 3.6 L 3.8 2.9 L 4.3 5.2 L 6.4 5.7 Z",
               "url(#face)", sw=0.65),
        P.fill("M 5.9 3.8 C 7.4 3.5 8.4 4.5 7.9 5.9 "
               "L 6.8 8 C 6.3 9 4.5 8.6 4.3 7.6 "
               "L 4.6 5 C 4.7 4.4 5.2 4 5.9 3.8 Z",
               "url(#face)", sw=0.75),
        P.circle(6.3, 5.1, 0.34, "#FAFAF6", outline=None),
    ])
    body = [
        "<title>Clone selected entities — two identical sheep</title>",
        f'<g transform="translate(1,0)">{sheep}</g>',
        f'<g transform="translate(-2,9.5)">{sheep}</g>',
    ]
    return P.svg("\n".join(body), defs=defs)
