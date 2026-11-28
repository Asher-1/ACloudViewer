"""View, color and settings symbols; preserve the source-specific marks."""
import math

from . import paraview_style as P


def zoom_to_box():
    # The source combines a dashed rubber-band box with a magnifying glass.
    defs = P.radial('lens', [(0, P.BLUE_PALE), (1, P.BLUE_MID)],
                    cx=.35, cy=.3, r=.9)
    box = P.stroke('M 2 2 H 17.8 V 17.8 H 2 Z', sw=1.0)
    box = box.replace('/>', ' stroke-dasharray="2.7 1.8"/>')
    body = [box,
            P.fill('M 18.3 16.7 L 23 21.4 L 21.4 23 L 16.7 18.3 Z', P.GREY_LT),
            P.circle(13, 13, 6.7, 'url(#lens)'),
            P.stroke('M 8.5 11.8 C 9 9.5 10.5 8.3 12.9 8.2',
                     colour=P.BLUE_PALE, sw=1.0)]
    return P.svg('<title>Zoom to box — rubber band and magnifier</title>\n' +
                 '\n'.join(body), defs=defs)


def property_editor():
    points = []
    for tooth in range(8):
        for angle, radius in [(0, 7.5), (7, 10), (27, 10), (34, 7.5)]:
            a = math.radians(tooth * 45 + angle - 90)
            points.append((12 + radius * math.cos(a), 12 + radius * math.sin(a)))
    d = 'M ' + ' L '.join(f'{x:.3f} {y:.3f}' for x, y in points) + ' Z '
    d += 'M 15.4 12 A 3.4 3.4 0 1 0 8.6 12 A 3.4 3.4 0 1 0 15.4 12 Z'
    defs = P.linear('gear', [(0, P.GREY_LT), (1, P.BLUE_MID)])
    path = P.fill(d, 'url(#gear)').replace('/>', ' fill-rule="evenodd"/>')
    return P.svg('<title>Property editor — gear</title>\n' + path, defs=defs)


def color_select():
    defs = '\n'.join([
        P.radial('warm', [(0, P.YELLOW), (1, P.BROWN)], cx=.35, cy=.25, r=.9),
        P.radial('green', [(0, P.GREEN_LT), (1, P.GREEN)], cx=.35, cy=.25, r=.9),
        P.radial('blue', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)], cx=.35, cy=.25, r=.9),
    ])
    body = [P.circle(15.6, 7.2, 6, 'url(#green)'),
            P.circle(7.5, 12, 6, 'url(#warm)'),
            P.circle(15.6, 17, 6, 'url(#blue)')]
    return P.svg('<title>Colors — three overlapping color circles</title>\n' +
                 '\n'.join(body), defs=defs)


def settings():
    # The original is crossed tools, not another gear. Keep both silhouettes.
    defs = P.linear('handle', [(0, P.YELLOW), (1, P.BROWN)])
    body = [
        P.fill('M 19.5 1.5 C 15 0 12 4 13.8 7.4 L 2.1 19.1 '
               'C .7 20.5 3.5 23.3 4.9 21.9 L 16.6 10.2 '
               'C 20 12 24 9 22.5 4.5 L 19.7 7.3 L 17 4.6 Z', P.BLUE_MID),
        P.fill('M 2 1.5 L 5 3.1 L 5.7 5.1 L 14.8 14.2 '
               'L 13.2 15.8 L 4.1 6.7 L 2.1 6 Z', P.GREY_LT),
        P.fill('M 12 14.5 L 14.5 12 L 22.1 19.6 '
               'Q 23.2 20.7 21.8 22.1 Q 20.5 23.4 19.4 22.2 Z',
               'url(#handle)'),
        P.stroke('M 15 16 L 20.2 21.2', colour=P.TAN_DK, sw=.65),
    ]
    return P.svg('<title>Settings — wrench and screwdriver</title>\n' +
                 '\n'.join(body), defs=defs)


def plugins():
    # Four puzzle pieces are the original identifier of the plugin manager.
    piece = ('M 0 0 H 2.6 C 1.8 -2.5 6.2 -2.5 5.4 0 H 8 V 2.6 '
             'C 10.5 1.8 10.5 6.2 8 5.4 V 8 H 5.4 '
             'C 6.2 5.5 1.8 5.5 2.6 8 H 0 V 5.4 '
             'C 2.5 6.2 2.5 1.8 0 2.6 Z')
    body = ['<title>About plugins — interlocking puzzle pieces</title>']
    for x, y, color in [(3, 3.5, P.BLUE_MID), (12, 3.5, P.BROWN),
                        (3, 13, P.YELLOW), (12, 13, P.GREEN)]:
        body.append(f'<g transform="translate({x} {y})">' +
                    P.fill(piece, color, sw=.7) + '</g>')
    return P.svg('\n'.join(body))


ICONS = {
    'pqZoomToBox@2x.png': ('view/viewZoomToBox.svg', zoom_to_box, '框选缩放'),
    'ecvGear.png': ('misc/propertyEditor.svg', property_editor, '属性编辑 · 齿轮'),
    'color.png': ('misc/colorSelect.svg', color_select, '颜色 · 三色圆'),
    'settings.png': ('misc/settingsOptions.svg', settings, '设置 · 交叉工具'),
    'Components_72px.png': ('main/toolMainAboutPlugins.svg', plugins, '插件管理 · 拼图'),
}
