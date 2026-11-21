"""View pivots, sensor menus and point-pair registration actions."""
from . import paraview_style as P


def new_view():
    body=[P.rect(2,3,19,17,P.BLUE_PALE,r=.6),
          P.rect(2,3,19,2.5,P.BLUE_MID,r=0),
          # The source 3D label stays content, authored as outlines.
          P.stroke('M 5 8 H 8.3 L 6.5 11 H 8.3 V 15 H 5 '
                   'M 11 8 V 15 H 13 Q 16 15 16 11.5 Q 16 8 13 8 Z',sw=1.05),
          P.stroke('M 5 17.5 H 18',colour=P.BLUE_STRONG,sw=.7)]
    # Also used by the existing 3D/2D view toggle: no exclusive add-window badge.
    return P.svg('<title>3D view — new view and 3D-mode controls</title>\n'+''.join(body))


def _rotation_ring():
    return P.fill('M 20 8 A 8.7 8.7 0 1 0 21 14 '
                  'L 18.2 13.5 A 5.9 5.9 0 1 1 17.4 8.8 '
                  'L 15.2 9.5 L 21.5 11.8 L 22 5.1 Z',P.GREEN_LT,sw=.7)


def pick_center():
    body=[_rotation_ring(),P.circle(11.4,11.7,1.65,P.YELLOW),
          P.fill('M 13 13 L 21.2 18.8 L 18.1 19.1 L 19.9 22 '
                 'L 17.8 23 L 16.2 19.7 L 14.3 21.3 Z',P.GREY_LT,sw=.75)]
    return P.svg('<title>Auto-pick rotation center at clicked 3D point</title>\n'+''.join(body))


def reset_center():
    body=[_rotation_ring(),P.circle(11.5,12,2.1,P.YELLOW),
          P.stroke('M 11.5 7 V 9.9 M 11.5 14.1 V 17 '
                   'M 6.5 12 H 9.4 M 13.6 12 H 16.5',sw=1.1)]
    return P.svg('<title>Reset rotation center — recenter target</title>\n'+''.join(body))


def show_center():
    body=[_rotation_ring(),
          P.stroke('M 11.5 12 V 5.8',colour=P.BLUE_STRONG,sw=1.35),
          P.stroke('M 11.5 12 L 5.5 16.5',colour=P.GREEN_DK,sw=1.35),
          P.stroke('M 11.5 12 L 17 16.5',colour=P.TAN_DK,sw=1.35),
          P.circle(11.5,12,2,P.YELLOW)]
    return P.svg('<title>Show rotation-center axes</title>\n'+''.join(body))


def unroll():
    defs=P.linear('sheet',[(0,P.BLUE_PALE),(.5,P.BLUE_MID),(1,P.BLUE_STRONG)],x2=1,y2=0)
    body=[P.fill('M 3 6 C 3 1 13 1 13 6 V 19 '
                 'C 13 22 3 22 3 19 Z','url(#sheet)'),
          f'<ellipse cx="8" cy="6" rx="5" ry="2.5" fill="{P.BLUE_PALE}" stroke="{P.INK}"/>',
          P.fill('M 10 6 C 15 8 19 6 22 4 V 18 C 17 21 14 20 10 19 Z',P.GREEN_LT),
          P.stroke('M 16 7 V 20 M 10 12.5 Q 17 15 22 10.5',sw=.6),
          P.fill('M 12 2 H 18 V 0.8 L 22 3.5 L 18 6 V 4.8 H 12 Z',P.YELLOW,sw=.6)]
    return P.svg('<title>Unroll cylindrical or conical surface to a sheet</title>\n'+''.join(body),defs=defs)


def sensors():
    defs=P.linear('sensor',[(0,P.GREY_LT),(1,P.BLUE_MID)])
    body=[P.fill('M 3 6 H 10 L 12 8 H 17 V 18 H 2 V 8 Z','url(#sensor)'),
          P.circle(9.5,12.5,3.6,P.BLUE_STRONG),P.circle(9.5,12.5,1.9,P.BLUE_PALE),
          P.fill('M 16 7 L 22 3 V 21 L 16 17 Z',P.GREEN_LT),
          P.stroke('M 17.8 8.8 V 16.6 M 17 10.5 L 21 6 M 17 14 L 21 18',sw=.55)]
    return P.svg('<title>Sensor tools — camera and scanning field</title>\n'+''.join(body),defs=defs)


def waveform():
    body=[P.rect(2,3,20,18,P.BLUE_PALE,r=.5),
          P.stroke('M 3.5 16 H 7 L 8.4 6 L 10.5 19 L 12.3 9 '
                   'L 14 15 H 16',colour=P.BLUE_STRONG,sw=1.2),
          P.stroke('M 17.2 7.2 Q 21.4 12 17.2 16.8 '
                   'M 15.3 9 Q 18 12 15.3 15',colour=P.GREEN_DK,sw=1)]
    return P.svg('<title>Full waveform tools — waveform and echoes</title>\n'+''.join(body))


def promote_cylinder():
    defs=P.linear('wall',[(0,P.BLUE_PALE),(.5,P.BLUE_MID),(1,P.BLUE_STRONG)],x2=1,y2=0)
    body=[P.fill('M 3 7 C 3 12 17 12 17 7 V 19 C 17 24 3 24 3 19 Z','url(#wall)'),
          f'<ellipse cx="10" cy="7" rx="7" ry="3.3" fill="{P.GREEN_LT}" stroke="{P.INK}"/>',
          P.fill('M 19 15 V 7 H 17 L 20.3 2 L 23 7 H 21 V 15 Z',P.YELLOW,sw=.6)]
    return P.svg('<title>Promote a selected circle to a cylinder — extrusion</title>\n'+''.join(body),defs=defs)


def manual_point():
    defs=P.linear('pencil',[(0,P.YELLOW),(1,P.TAN)])
    body=[P.fill('M 3 17 L 15 3 L 20 7 L 8 21 L 2 23 Z','url(#pencil)'),
          P.fill('M 15 3 L 17 1 Q 18 0.5 19 1.5 L 21.5 3.5 '
                 'Q 22 4.5 21 5.5 L 20 7 Z',P.GREY_LT),
          P.fill('M 3 17 L 8 21 L 2 23 Z',P.GREY_LT),
          P.fill('M 2.4 20.7 L 4.5 22.1 L 2 23 Z',P.INK,outline=None),
          P.stroke('M 6 17.7 L 17.2 5',colour=P.TAN_DK,sw=.65),
          P.stroke('M 15 19 H 22 M 18.5 15.5 V 22.5',colour=P.BLUE_STRONG,sw=1.5)]
    return P.svg('<title>Enter a point manually for point-pair registration</title>\n'+''.join(body),defs=defs)


def remove_last_point():
    body=[P.rect(2,3,18,4,P.GREEN_LT,r=.4),
          P.rect(2,9,18,4,P.GREEN_LT,r=.4),
          P.rect(2,15,18,4,P.BLUE_PALE,r=.4),
          P.stroke('M 5 5 H 15 M 5 11 H 15',sw=.6),
          P.circle(18,18.5,4.2,P.YELLOW),
          P.stroke('M 15.8 18.5 H 20.2',sw=1.4)]
    return P.svg('<title>Remove the last registration point, not the entire list</title>\n'+''.join(body))


ICONS={
 '3D3.png':('view/view3D.svg',new_view,'3D 视图 / 模式'),
 'pqPickCenter@2x.png':('view/viewPickRotationCenter.svg',pick_center,'拾取旋转中心'),
 'pqResetCenter@2x.png':('view/viewResetRotationCenter.svg',reset_center,'重置旋转中心'),
 'pqShowCenterAxes@2x.png':('view/viewShowRotationCenter.svg',show_center,'显示旋转中心坐标轴'),
 'unroll.png':('main/toolMainUnroll.svg',unroll,'圆柱 / 圆锥展开'),
 'integrated_sensor.png':('view/viewSensorTools.svg',sensors,'传感器工具'),
 'Wavefrom.png':('view/viewWaveformTools.svg',waveform,'全波形工具'),
 'primCylinder.gif':('primitive/primitivePromoteCylinder.svg',promote_cylinder,'圆提升为圆柱'),
 'svg/main/toolMainPencil.svg':('registration/registrationAddManualPoint.svg',manual_point,'手动输入配准点'),
 'svg/main/toolMainUnstack.svg':('registration/registrationRemoveLastPoint.svg',remove_last_point,'移除最后配准点'),
}
