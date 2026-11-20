"""Five navigation icons, preserving the original action silhouettes.

Projection icons retain the source's offset rectangular ends and translucent
connecting faces. Perspective uses a small lower-left end and a large upper-right
end; orthographic uses equal ends. Do not substitute a generic solid cube.
"""
from __future__ import annotations

from . import paraview_style as P
from .icon import fit


def global_zoom() -> str:
    """Fit the scene: four bounds and two outward diagonal arrows."""
    body = ['<title>Fit the entire scene in view</title>']
    for x, y, sx, sy in [(2.4, 2.4, 1, 1), (21.6, 2.4, -1, 1),
                          (2.4, 21.6, 1, -1), (21.6, 21.6, -1, -1)]:
        body.append(P.stroke(f'M {x+sx*5} {y} H {x} V {y+sy*5}', sw=1.6))
    body.extend([
        P.fill('M 4.5 4.5 L 11 5.2 L 9.1 7.1 L 11.2 9.2 '
               'L 9.2 11.2 L 7.1 9.1 L 5.2 11 Z', P.BLUE_MID, sw=0.7),
        P.fill('M 19.5 19.5 L 13 18.8 L 14.9 16.9 L 12.8 14.8 '
               'L 14.8 12.8 L 16.9 14.9 L 18.8 13 Z', P.BLUE_MID, sw=0.7),
    ])
    return P.svg('\n'.join(body))


def _projection(perspective: bool) -> str:
    # Coordinates come from the original 32px source, scaled by 0.75.
    # Far end: (8,2)..(30,25). Perspective near end: (2,17)..(14,30).
    # Equal end sizes and a constant (6,-5) offset express orthographic mode.
    if perspective:
        back = [(6, 1.5), (22.5, 3.75), (22.5, 18.75), (6, 13)]
        front = [(1.5, 12.75), (10.5, 14), (10.5, 22.5), (1.5, 21.75)]
    else:
        back = [(7.5, 1.5), (22.5, 1.5), (22.5, 18), (7.5, 18)]
        front = [(1.5, 5.25), (16.5, 5.25), (16.5, 21.75), (1.5, 21.75)]
    defs = '\n'.join([
        P.linear('back', [(0, P.BLUE_PALE), (1, '#EAF2F8')]),
        P.linear('side', [(0, P.BLUE_MID), (1, P.BLUE_STRONG)]),
        P.linear('front', [(0, P.BLUE_MID), (1, '#789CB7')]),
    ])
    title = 'Perspective projection' if perspective else 'Orthographic projection'
    body = [f'<title>{title}</title>', P.poly(back, 'url(#back)'),
            P.poly([front[0], back[0], back[3], front[3]], 'url(#side)'),
            P.poly([front[3], back[3], back[2], front[2]], P.BLUE_MID),
            P.poly([front[1], back[1], back[2], front[2]], P.BLUE_PALE),
            P.poly([front[0], back[0], back[1], front[1]], 'url(#back)')]
    # The transparent front preserves the original glass-volume reading.
    body.append(P.poly(front, 'url(#front)').replace(' stroke=', ' fill-opacity="0.78" stroke=', 1))
    body.append(P.stroke(f'M {back[0][0]} {back[0][1]} L {back[3][0]} {back[3][1]} '
                         f'L {back[2][0]} {back[2][1]}', sw=0.65))
    return P.svg('\n'.join(body), defs=defs)


def ortho_projection() -> str:
    return _projection(False)


def perspective_projection() -> str:
    return _projection(True)


def point_picking() -> str:
    """Keep the original target ring and overlapping arrow cursor."""
    defs = P.linear('metal', [(0, '#FAFAF6'), (1, P.GREY)])
    body = [
        '<title>Pick a point</title>',
        P.stroke('M 11.5 1.5 V 6 M 11.5 16.5 V 21 '
                 'M 1.5 11.5 H 6 M 17 11.5 H 21', sw=1.7),
        '<circle cx="11.5" cy="11.5" r="4.1" fill="url(#metal)" '
        f'stroke="{P.INK}" stroke-width="1.35"/>',
        '<circle cx="11.5" cy="11.5" r="2.3" fill="none" '
        f'stroke="{P.INK}" stroke-width="0.7"/>',
        P.fill('M 13.6 13.4 L 21.8 19.4 L 18.5 19.7 L 20.1 22.1 '
               'L 17.9 23 L 16.6 20.2 L 14.4 22.1 Z', '#FAFAF6', sw=1.0),
    ]
    return P.svg('\n'.join(body), defs=defs)


def point_list_picking() -> str:
    """Retain the source's tall sheet and four marker/line rows."""
    defs = P.linear('paper', [(0, '#FAFAF6'), (1, P.BLUE_PALE)])
    body = ['<title>Pick a list of points</title>',
            P.rect(3.6, 1.5, 16.8, 21, 'url(#paper)', r=0.6)]
    for y in (5.4, 9.8, 14.2, 18.6):
        body.extend([P.circle(7.4, y, 1.25, P.GREEN, outline=P.INK),
                     P.stroke(f'M 11 {y} H 17.6', colour=P.BLUE_STRONG, sw=1.5)])
    return P.svg('\n'.join(body), defs=defs)


ICONS = {
    'zoom': ('toolMainGlobalZoom', global_zoom),
    'ortho': ('toolMainOrthoProjection', ortho_projection),
    'perspective': ('toolMainPerspectiveProjection', perspective_projection),
    'pointpicking': ('toolMainPointPicking', point_picking),
    'pointlist': ('toolMainPointListPicking', point_list_picking),
}


def built(key: str) -> str:
    return fit(ICONS[key][1]())
