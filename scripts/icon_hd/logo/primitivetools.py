"""Primitive-factory icons: retain box, sphere, cylinder, cone and dish.

Only geometry is authored here. The batch migration/preview tool owns paths and
rendering. All silhouettes fit the 24-unit source canvas before normalisation.
"""
from . import paraview_style as P


def box():
    defs = '\n'.join([
        P.linear('left', [(0, P.BLUE_PALE), (1, P.BLUE_MID)]),
        P.linear('right', [(0, P.BLUE_MID), (1, P.BLUE_STRONG)]),
    ])
    return P.svg('<title>Box primitive</title>\n' +
                 P.cube(12, 12, 10, top=P.BLUE_PALE,
                        left='url(#left)', right='url(#right)'), defs=defs)


def sphere():
    defs = P.radial('surface', [(0, '#EDF4E7'), (0.3, P.GREEN_LT),
                                (0.72, P.GREEN), (1, P.GREEN_DK)],
                    cx=0.32, cy=0.27, r=0.85)
    return P.svg('<title>Sphere primitive or sphere fitting</title>\n' +
                 P.circle(12, 12, 10, 'url(#surface)'), defs=defs)


def cylinder():
    defs = P.linear('side', [(0, P.GREY_MID), (0.32, P.GREY_LT),
                            (0.6, P.GREY), (1, P.GREY_DK)], x2=1, y2=0)
    body = [
        '<title>Cylinder primitive</title>',
        P.fill('M 3.5 5.5 C 3.5 10.2 20.5 10.2 20.5 5.5 V 18.5 '
               'C 20.5 23.2 3.5 23.2 3.5 18.5 Z', 'url(#side)'),
        f'<ellipse cx="12" cy="5.5" rx="8.5" ry="3.5" '
        f'fill="{P.BLUE_PALE}" stroke="{P.INK}"/>',
    ]
    return P.svg('\n'.join(body), defs=defs)


def cone():
    defs = P.linear('surface', [(0, P.GREEN_DK), (0.4, P.GREEN_LT),
                               (0.65, P.GREEN), (1, P.GREEN_DK)], x2=1, y2=0)
    body = [
        '<title>Cone or truncated-cone primitive</title>',
        P.fill('M 12 2 L 22 18.5 C 22 23.2 2 23.2 2 18.5 Z',
               'url(#surface)'),
        P.stroke('M 2 18.5 C 5 21.8 19 21.8 22 18.5', sw=0.65),
    ]
    return P.svg('\n'.join(body), defs=defs)


def dish():
    # ccDish is a spherical/ellipsoidal cap, not an antenna: no stand or feed.
    defs = '\n'.join([
        P.linear('outer', [(0, P.GREY_LT), (1, P.GREY_DK)]),
        P.radial('capFace', [(0, P.BLUE_PALE), (0.7, P.GREY_LT),
                            (1, P.BLUE_MID)], cx=0.5, cy=0.2, r=0.9),
    ])
    body = [
        '<title>Dish primitive — spherical or ellipsoidal cap</title>',
        P.fill('M 2 7 C 3 13.5 6.5 19.2 12 20 '
               'C 17.5 19.2 21 13.5 22 7 Z', 'url(#outer)'),
        f'<ellipse cx="12" cy="7" rx="10" ry="3.4" '
        f'fill="url(#capFace)" stroke="{P.INK}"/>',
        P.stroke('M 3.4 7.8 C 6.8 10.4 17.2 10.4 20.6 7.8',
                 colour=P.GREY_MID, sw=0.6),
    ]
    return P.svg('\n'.join(body), defs=defs)


ICONS = {
    'cube.png': ('primitive/primitiveBox.svg', box, '长方体'),
    'sphere.png': ('primitive/primitiveSphere.svg', sphere, '球体 / 球拟合'),
    'cylinder.png': ('primitive/primitiveCylinder.svg', cylinder, '圆柱体'),
    'cone.svg': ('primitive/primitiveCone.svg', cone, '圆锥体'),
    'dish.png': ('primitive/primitiveDish.svg', dish, '碟形曲面'),
}
