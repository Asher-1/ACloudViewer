"""Processing icons: communicate the action, not incidental source-model detail.

ICP is correspondence-driven registration; cloud/primitive distance measures an
analytic surface; Clean groups denoising and resampling, not scene deletion.
"""
from . import paraview_style as P


def segment():
    defs = P.linear('blade', [(0, '#FAFAF6'), (1, P.GREY_MID)])
    ring1 = ('M 8 17.5 A 3.2 3.2 0 1 0 1.6 17.5 A 3.2 3.2 0 1 0 8 17.5 Z '
             'M 6.3 17.5 A 1.5 1.5 0 1 0 3.3 17.5 A 1.5 1.5 0 1 0 6.3 17.5 Z')
    ring2 = ('M 13 20 A 2.8 2.8 0 1 0 7.4 20 A 2.8 2.8 0 1 0 13 20 Z '
             'M 11.5 20 A 1.3 1.3 0 1 0 8.9 20 A 1.3 1.3 0 1 0 11.5 20 Z')
    body = [P.fill('M 9 13 L 16.5 1.5 L 14.7 11.6 L 11.5 15 Z', 'url(#blade)'),
            P.fill('M 9 13 L 22.5 7.7 L 15.3 14.4 L 11.4 15.3 Z', 'url(#blade)'),
            P.stroke('M 7 15.1 L 11.4 12.6 L 11 17.5', colour=P.BLUE_MID, sw=2),
            P.fill(ring1+' '+ring2, P.BLUE_MID).replace('/>', ' fill-rule="evenodd"/>'),
            P.circle(11.4, 13.3, .8, P.GREY_DK)]
    return P.svg('<title>Interactive segmentation — scissors</title>\n' +
                 '\n'.join(body), defs=defs)


def trace_polyline():
    line = P.stroke('M 2 21.8 L 15 19.5 L 13.7 6.8 L 22 2',
                    colour=P.GREY_DK, sw=1.2)
    line = line.replace('/>', ' stroke-dasharray="2 1.3"/>')
    body = [line, P.rect(13.5, 18.1, 2.7, 2.7, P.BLUE_MID, r=.2),
            P.rect(12.4, 5.5, 2.7, 2.7, P.BLUE_MID, r=.2),
            P.fill('M 7 1.8 H 9.5 V 7 H 15 V 9.5 H 9.5 V 15 '
                   'H 7 V 9.5 H 1.5 V 7 H 7 Z', P.BLUE_STRONG, sw=.7)]
    return P.svg('<title>Trace a polyline — picking cross and vertex chain</title>\n' +
                 '\n'.join(body))


def apply_scale():
    defs = P.linear('size', [(0, P.BLUE_PALE), (1, P.BLUE_MID)])
    body = [P.rect(2, 2, 20, 20, 'url(#size)', r=.4),
            P.rect(2, 2, 6.5, 6.5, P.GREEN_LT, r=.2),
            P.fill('M 16.5 8.5 L 18.5 10.5 L 12.5 16.5 L 15 19 '
                   'L 6 20 L 7 11 L 9.5 13.5 Z', P.YELLOW, sw=.7)]
    return P.svg('<title>Multiply or scale coordinates — size change</title>\n' +
                 '\n'.join(body), defs=defs)


def icp():
    # Two congruent sample sets and explicit correspondences: the operation is
    # alignment, not merging, point-pair picking or a particular scanned model.
    reference = [(3.2, 8.2), (8, 5), (13, 6.5), (14.2, 11.8), (8.5, 15), (3, 13)]
    moving = [(x+3, y-2.6) for x, y in reference]
    defs = P.linear('reference', [(0, P.BLUE_PALE), (1, P.BLUE_MID)])
    body = [P.poly(reference, 'url(#reference)'),
            P.stroke('M 6.2 5.6 L 11 2.4 L 16 3.9 L 17.2 9.2 '
                     'L 11.5 12.4 L 6 10.4 Z', colour=P.GREEN_DK, sw=.8)]
    correspondence = ' '.join(f'M {moving[i][0]:g} {moving[i][1]:g} '
                              f'L {reference[i][0]:g} {reference[i][1]:g}'
                              for i in (0, 2, 4))
    body.append(P.stroke(correspondence, colour=P.GREY_DK, sw=.65).replace(
        '/>', ' stroke-dasharray="1 0.8"/>'))
    for x, y in reference:
        body.append(P.circle(x, y, .85, P.BLUE_STRONG, outline=P.INK))
    for x, y in moving:
        body.append(P.circle(x, y, .85, P.GREEN, outline=P.INK))
    body.append(P.fill('M 21 7.5 L 22.3 8.6 L 18.4 12.6 L 20 14 '
                       'L 14.6 15 L 15.7 9.7 L 17.1 11 Z', P.YELLOW, sw=.65))
    # Outlined glyphs; no host-font dependency. ICP distinguishes automatic fine
    # registration from the existing manually picked point-pair alignment tool.
    body.append(P.stroke('M 5 18 H 8 M 6.5 18 V 22 M 5 22 H 8 '
                         'M 13 18 H 10 V 22 H 13 '
                         'M 15.4 22 V 18 H 18.2 V 20 H 15.4', sw=1.05))
    return P.svg('<title>ICP — align corresponding blue and green sample sets</title>\n' +
                 '\n'.join(body), defs=defs)


def cloud_primitive_distance():
    # An analytic sphere stands for the supported equation-based primitives.
    # The measured point projects radially onto its surface, not its centre.
    defs = P.radial('surface', [(0, P.GREEN_LT), (.7, P.GREEN), (1, P.GREEN_DK)],
                    cx=.3, cy=.25, r=.9)
    body = [P.circle(16, 8, 5.6, 'url(#surface)'),
            P.stroke('M 16 2.4 C 12.5 4.1 12.5 11.9 16 13.6 '
                     'M 16 2.4 C 19.5 4.1 19.5 11.9 16 13.6 '
                     'M 10.7 6.1 Q 16 9.2 21.3 6.1 '
                     'M 10.7 9.9 Q 16 6.8 21.3 9.9',
                     colour=P.INK, sw=.55),
            # Surface endpoint Q lies on r=5.6 along the radial P->centre line.
            P.stroke('M 6.5 17.2 L 12.04 11.96', colour=P.INK, sw=1.45),
            P.arrow(6.5, 17.2, 12.04, 11.96, P.YELLOW, head=1.7, sw=.9, both=True),
            P.stroke('M 5.7 16.5 L 7.3 18 M 11.3 11.3 L 12.8 12.8',
                     colour=P.GREY_DK, sw=.75),
            P.circle(12.04, 11.96, .6, P.GREEN_LT, outline=P.INK)]
    for x, y, r in [(2.3, 12.3, .85), (5, 8.8, .8), (6.5, 17.2, 1.05),
                     (3.2, 20.8, .85), (9, 21, .85), (12.3, 18.5, .8)]:
        body.append(P.circle(x, y, r, P.BLUE_STRONG, outline=P.INK))
    return P.svg('<title>Point-cloud to analytic primitive distance — point to surface</title>\n' +
                 '\n'.join(body), defs=defs)


def clean_menu():
    # Menu containing SOR, noise filtering and voxel sampling: clean the data,
    # do not delete the scene. Retained point samples stay visible beside broom.
    defs = '\n'.join([
        P.linear('handle', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)]),
        P.linear('bristles', [(0, P.TAN), (.5, P.YELLOW), (1, P.TAN)]),
    ])
    body = []
    for x, y, color in [(3, 9.2, P.BLUE_MID), (5.8, 7.2, P.BLUE_STRONG),
                         (8.6, 9.5, P.BLUE_MID), (3.1, 13, P.BLUE_STRONG),
                         (6, 12, P.BLUE_STRONG), (8.5, 13.7, P.BLUE_MID),
                         (3.8, 16.7, P.BLUE_STRONG)]:
        body.append(P.circle(x, y, .85, color, outline=P.INK))
    for x, y in [(2, 4.7), (8.2, 2.5)]:
        body.append(P.circle(x, y, .55, P.GREY_MID, outline=None))
    body += [P.fill('M 20.3 1.7 Q 21 1 21.8 1.8 L 22.4 2.4 '
                    'Q 23 3.1 22.2 3.8 L 14.4 13.1 L 12.4 11.3 Z',
                    'url(#handle)', sw=.65),
             P.fill('M 12.3 11 L 15.4 13.8 L 13.6 16 L 10.5 13.2 Z',
                    P.GREY_LT, sw=.65),
             P.fill('M 10.5 13.1 L 13.9 15.9 L 13.1 22 '
                    'Q 8.1 21.8 5.2 18.3 L 8.6 13.5 Z', 'url(#bristles)'),
             P.stroke('M 9 15.5 L 6.8 19.6 M 10.6 16.3 L 9 20.8 '
                      'M 12.1 17 L 11.4 21.5', colour=P.TAN_DK, sw=.6)]
    return P.svg('<title>Clean point-cloud data — samples and sweeping broom</title>\n' +
                 '\n'.join(body), defs=defs)


def filter_tools():
    defs = P.linear('funnel', [(0, P.BLUE_MID), (1, P.BLUE_STRONG)])
    body = [P.fill('M 2 8 H 22 L 14.7 16 V 21 L 10.2 22.5 V 16 Z', 'url(#funnel)'),
            P.rect(2, 8, 20, 2, P.BLUE_PALE, r=.3),
            P.poly([(4,3),(6.3,1.7),(8,4.7),(5.7,6)], P.YELLOW),
            P.rect(10.4, 4.1, 2.8, 2.8, P.BROWN, r=.3),
            P.rect(16.5, 1.4, 3, 5, P.GREEN_LT, r=.4)]
    return P.svg('<title>Filter tools and noise filter — funnel with input samples</title>\n' +
                 '\n'.join(body), defs=defs)


def connected_components():
    # The original has two broad counterposed arcs and two distinct centres.
    top = ('M 18 2.8 A 10.8 10.8 0 0 0 1.2 11.2 '
           'C 4.2 7 9.8 8.7 11.5 11.2 C 12 6.8 14.8 5 18 5.2 Z')
    body = [P.fill(top, P.BLUE_MID),
            '<g transform="rotate(180 12 12)">'+P.fill(top, P.GREEN_LT)+'</g>',
            P.circle(6.5, 12, 2.4, P.BLUE_STRONG),
            P.circle(17.5, 12, 2.4, P.GREEN)]
    return P.svg('<title>Label connected components — two separated centres</title>\n' +
                 '\n'.join(body))


def annotations():
    defs = P.linear('tag', [(0, P.YELLOW), (1, P.TAN)])
    contour = ('M 3 2 H 9.2 L 22.5 15.3 L 14.5 23 L 1.5 10 '
               'V 3.5 Q 1.5 2 3 2 Z '
               'M 6.8 6 A 1.5 1.5 0 1 0 3.8 6 A 1.5 1.5 0 1 0 6.8 6 Z')
    body = [P.stroke('M 4.5 5.2 C 1.2 4.2 1.2 15 4.8 15.4', sw=.7),
            P.fill(contour, 'url(#tag)').replace('/>', ' fill-rule="evenodd"/>')]
    return P.svg('<title>Annotation tools — label tag</title>\n' +
                 '\n'.join(body), defs=defs)


def intersect_selections():
    body = [P.fill('M 2 2 H 15.5 V 15.5 H 2 Z', 'none', sw=1.1),
            P.fill('M 8.5 8.5 H 22 V 22 H 8.5 Z', 'none', sw=1.1),
            P.rect(8.5, 8.5, 7, 7, P.GREEN, r=0)]
    return P.svg('<title>Intersect selections — overlap only</title>\n' +
                 '\n'.join(body))


ICONS = {
    'ecvSegment.png': ('main/toolMainSegment.svg', segment, '交互分割 · 剪刀'),
    'ecvTracePolyline.png': ('main/toolMainTracePolyline.svg', trace_polyline, '描绘折线 · 拾取十字'),
    'transformScale.png': ('main/toolMainApplyScale.svg', apply_scale, '坐标缩放'),
    'ecvicp.png': ('main/toolMainICP.svg', icp, '精配准 · ICP'),
    'ecvCloudPrimitiveDistance.png': ('main/toolMainCloudPrimitiveDistance.svg', cloud_primitive_distance, '点云 / 几何体距离'),
    'clean_72px_27328.png': ('main/toolMainClean.svg', clean_menu, '清理工具'),
    'filter.png': ('filter/filterTools.svg', filter_tools, '滤波 · 滤斗'),
    'ecvCCExtract.svg': ('main/toolMainConnectedComponents.svg', connected_components, '连通分量提取'),
    'annotations.png': ('annotation/annotationTools.svg', annotations, '标注工具 · 标签'),
    'ecvIntersection.svg': ('annotation/annotationIntersectSelections.svg', intersect_selections, '选择交集'),
}
