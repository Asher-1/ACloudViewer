"""Smooth, resolution-independent redraws for raster-mosaicked ParaView SVGs."""
from . import paraview_style as P
from .iconlabels import label_path


def _preview_svg(body, *, defs=''):
    """128px intrinsic preview; keep the original 24-unit drawing coordinates."""
    return P.svg(body, defs=defs).replace('width="24" height="24"',
                                          'width="128" height="128"', 1)


def _tag(text,x=4.1,y=21.7,w=15.8,h=3.2):
    return P.rect(x-0.5,y-h-0.15,w+1,h+0.8,P.BLUE_PALE,r=0.45)+label_path(text,x=x,y=y,width=w,height=h)

def axes():
    # Source 32px triad: yellow up, tan right, green down-left; no invented labels.
    defs=P.linear('ax-y',[(0,'#FFE38A'),(1,P.YELLOW)])+P.linear('ax-x',[(0,'#DDC69A'),(1,P.TAN)])+P.linear('ax-z',[(0,'#BDDEA4'),(1,P.GREEN)])
    b=[P.fill('M 7.1 16.1 V 6.2 H 4.7 L 8.3 1.3 L 12.1 6.2 H 9.7 V 16.1 Z','url(#ax-y)',sw=.75),
       P.fill('M 8.2 15.2 H 18 V 12.9 L 22.6 17.3 L 18 21 V 18.8 H 8.2 Z','url(#ax-x)',sw=.75),
       P.fill('M 8.3 16.4 L 3.5 20.7 L 3.2 18.8 L 1.3 22.5 L 5.6 22.3 L 4.5 21.3 L 10.1 17.9 Z','url(#ax-z)',sw=.75)]
    return _preview_svg('<title>ParaView coordinate triad — original yellow/tan/green arrows</title>'+''.join(b),defs=defs)

def contours():
    # Original is an open S-shaped contour with two yellow control handles.
    b=[P.stroke('M 4.4 11.5 C 4.1 7.2 4.1 1.9 7.5 1.9 C 11.9 1.9 10.6 9.4 11.6 15.4 C 12.2 18.7 11.9 21.7 15.1 21.7 C 18.3 21.7 18.6 16.4 18.6 11.8',sw=1.65),
       P.circle(4.4,11.8,2.45,P.YELLOW),P.circle(18.7,11.8,2.45,P.YELLOW)]
    return _preview_svg('<title>ParaView contour widget — open spline and yellow endpoint handles</title>'+''.join(b))

def fixed_repr():
    b=[P.rect(2.4,4.1,19.2,6.8,P.GREY_LT,r=1.1),label_path('10',x=4,y=9.5,width=6.8,height=3.4),
       P.arrow(11,7.5,14.2,7.5,P.GREEN,head=1.15,sw=.8),P.rect(2.4,13.1,19.2,7.1,P.BLUE_PALE,r=1.1),
       label_path('10.00',x=4,y=18.8,width=12.8,height=3.5)]
    return P.svg('<title>Toggle fixed-point numeric representation and precision</title>'+''.join(b))

def invert():
    b=[P.poly([(3.2,17.8),(11.8,4.3),(20.7,17.8)],P.BLUE_PALE),
       P.arrow(11.8,12.9,11.8,3.1,P.GREEN,head=1.65,sw=1.05),
       P.arrow(11.8,14,11.8,22,P.BLUE_STRONG,head=1.65,sw=1.05),
       P.stroke('M 5 12 C 2.8 7 5.4 4.2 8.7 4.6 M 18.9 11 C 22 15 20.6 20 16.7 20',colour=P.INK,sw=.85),
       P.poly([(8.7,4.6),(6.5,3.4),(6.8,6.1)],P.YELLOW),
       P.poly([(16.7,20),(19.2,20.7),(18.7,18.3)],P.YELLOW)]
    return P.svg('<title>Flip triangle winding and reverse the surface normal</title>'+''.join(b))

def ramp():
    # Original pqRamp24 is two yellow endpoint handles joined diagonally.
    defs=P.radial('handle',[(0,'#FFE79A'),(.65,P.YELLOW),(1,'#DDB019')],cx=.3,cy=.25,r=.8)
    b=[P.stroke('M 5.6 18.1 L 18.3 5.3',sw=1.65),
       P.circle(5.1,18.9,2.8,'url(#handle)'),P.circle(18.4,5.1,2.8,'url(#handle)')]
    return _preview_svg('<title>ParaView ramp — two yellow handles and diagonal connector</title>'+''.join(b),defs=defs)

def rect_grid():
    b=[P.rect(3,3,18,18,P.BLUE_PALE,r=.35),P.stroke('M 9 3 V 21 M 15 3 V 21 M 3 9 H 21 M 3 15 H 21',colour=P.INK,sw=.65),
       P.rect(9.2,3.2,5.6,5.6,P.GREEN_LT,r=.2),P.stroke('M 17.8 5.2 H 21 M 17.8 7 H 21',colour=P.BLUE_STRONG,sw=.8)]
    return P.svg('<title>Toggle spreadsheet column visibility</title>'+''.join(b))

def save_table():
    # Original composition: tan folder behind ruled table, green down-arrow above.
    defs=P.linear('folder',[(0,'#D7C18B'),(1,P.TAN_DK)])+P.linear('download',[(0,P.GREEN_LT),(1,P.GREEN)])
    b=[P.fill('M 2 8.2 H 8.3 L 10.2 10.1 H 19 V 20.6 L 3.8 22.4 Z','url(#folder)',sw=.7),
       P.fill('M 4.4 13 H 14.8 L 13.2 21.5 L 4 23 Z','url(#folder)',sw=.7),
       P.rect(11.6,11.3,11.2,11.2,'#FAFAF6',r=.2),
       P.stroke('M 11.6 14.1 H 22.8 M 11.6 16.9 H 22.8 M 11.6 19.7 H 22.8 M 15.3 11.3 V 22.5 M 19 11.3 V 22.5',sw=.65),
       P.fill('M 12.5 1.2 H 16.8 V 4.3 H 20 L 14.7 10.3 L 9.5 4.3 H 12.5 Z','url(#download)',sw=.75)]
    return _preview_svg('<title>ParaView save table — folder, spreadsheet and green download arrow</title>'+''.join(b),defs=defs)

def select_only():
    # Original dotted green selection boundary and dark pointer overlap at lower right.
    b=[P.stroke('M 3 18.2 V 3 H 18.3 V 18.2 Z',colour=P.GREEN,sw=1.65).replace('/>',' stroke-dasharray="2.1 1.4" stroke-linecap="butt"/>'),
       P.fill('M 12.4 10.6 L 22 16.6 L 18.3 17.5 L 21.4 22.5 L 19.5 23 L 16.5 18 L 14 21 Z','#42443B',outline='#42443B',sw=.45)]
    return _preview_svg('<title>ParaView selection — green dashed rectangle and dark mouse pointer</title>'+''.join(b))

def _shift(axis,sign,color):
    # These are axis-aligned camera-view presets, not crop-box translations.
    label=axis+sign
    directions={
      ('X','+'):(20.5,11.2,13.3,12),('X','-'):(3.5,12.8,10.7,12),
      ('Y','+'):(12.7,3.5,12,10.5),('Y','-'):(11.3,20.5,12,13.5),
      ('Z','+'):(20,4.2,13.3,10.7),('Z','-'):(4,19.8,10.7,13.3),
    }
    x1,y1,x2,y2=directions[(axis,sign)]
    b=[P.cube(12,12,6.3,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREY_LT),
       P.circle(12,12,1.25,P.YELLOW),P.arrow(x1,y1,x2,y2,color,head=1.5,sw=1.0),
       P.circle(x1,y1,1.15,P.GREY_LT)]
    b.append(P.rect(7.7,19.1,8.9,4.4,P.BLUE_PALE,r=.55))
    b.append(label_path(label,x=8.2,y=22.3,width=7.9,height=3.0))
    return P.svg(f'<title>Camera view from the {"positive" if sign=="+" else "negative"} {axis} axis ({"1" if axis=="X" and sign=="+" else "-1" if axis=="X" else "0"}, {"1" if axis=="Y" and sign=="+" else "-1" if axis=="Y" else "0"}, {"1" if axis=="Z" and sign=="+" else "-1" if axis=="Z" else "0"})</title>'+''.join(b))

def calculator():
    b=[P.rect(4,1.6,16,20.8,P.GREY,r=1.8),P.rect(6,3.5,12,5,P.BLUE_PALE,r=.7),label_path('1+2',x=7,y=7.1,width=10,height=2.6)]
    for y in [11,15,19]:
        for x in [7.1,12,16.9]:b.append(P.rect(x-1.35,y-1.2,2.7,2.4,P.GREEN_LT if x==16.9 else P.GREY_LT,r=.4))
    return P.svg('<title>Scalar field arithmetic calculator</title>'+''.join(b))

def annotation_overlay():
    """Exact annotationTools.svg paths/transform, composed as a scaled overlay.

    User selected the existing label-tag shape explicitly; never replace it with
    an invented pencil. Outer placement changes, source geometry stays intact.
    """
    return '''<g data-source="annotation/annotationTools.svg" transform="translate(12.0 11.7) scale(.49)">
<g transform="translate(-0.0482,-0.5523) scale(1.00822)">
<defs><linearGradient id="tag" x1="0.0" y1="0.0" x2="0.0" y2="1.0"><stop offset="0.000" stop-color="#F3C017"/><stop offset="1.000" stop-color="#C1A561"/></linearGradient></defs>
<path d="M 4.5 5.2 C 1.2 4.2 1.2 15 4.8 15.4" fill="none" stroke-width="0.7"/>
<path d="M 3 2 H 9.2 L 22.5 15.3 L 14.5 23 L 1.5 10 V 3.5 Q 1.5 2 3 2 Z M 6.8 6 A 1.5 1.5 0 1 0 3.8 6 A 1.5 1.5 0 1 0 6.8 6 Z" fill="url(#tag)" stroke="#6F6F59" fill-rule="evenodd"/>
</g></g>'''


def faceset():
    # ANN-01: semantic annotation supports both point clouds and meshes.
    # FACE lettering is intentionally removed by the user's explicit correction.
    b = [P.fill('M 2.2 3.4 L 9 1.8 L 14 5 L 10.8 11 L 3.2 10.2 Z', P.BLUE_PALE, sw=.85),
         P.stroke('M 2.2 3.4 L 10.8 11 M 9 1.8 L 7.4 6.8 L 14 5 M 3.2 10.2 L 7.4 6.8', colour=P.BLUE_STRONG, sw=.7)]
    for x, y, color in [(3.4,14.2,P.GREEN),(7.3,13.4,P.BLUE_STRONG),
                         (10.6,15.7,P.GREEN),(3.4,18.7,P.BLUE_STRONG),
                         (7.2,20.3,P.GREEN),(10.8,19.7,P.BLUE_STRONG)]:
        b.append(P.circle(x,y,1.35,color,outline=P.INK))
    b.append(annotation_overlay())
    return _preview_svg('<title>ANN-01 — Point-cloud and mesh semantic annotation</title>'+''.join(b))


def probe():
    b=[P.cube(10.2,12,8.8,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREY_LT),P.circle(12.2,11.6,2.2,P.GREEN_LT),P.stroke('M 12.2 7.5 V 15.7 M 8 11.6 H 16.4',colour=P.BLUE_STRONG,sw=.9),P.fill('M 18 2 L 20.5 4.5 L 14.6 10.4 L 12.7 11.2 L 13.5 9.3 Z',P.YELLOW,sw=.65)]
    return P.svg('<title>Probe samples a field at a chosen location</title>'+''.join(b))

def recent():
    b=[P.fill('M 2 5 H 9 L 11 7 H 21 V 20 H 2 Z',P.TAN),P.fill('M 3 9 H 22 L 19 21 H 1.5 Z',P.YELLOW),P.circle(16.5,8,5.1,P.BLUE_PALE),P.stroke('M 16.5 4.6 V 8 L 19.1 9.4',colour=P.INK,sw=.95)]
    return P.svg('<title>Recent files — folder with history clock</title>'+''.join(b))

def solid_color():
    defs=P.radial('solid',[(0,'#E4C093'),(.55,P.BROWN),(1,'#9A602E')],cx=.32,cy=.28,r=.9)
    return P.svg('<title>Solid color swatch</title>'+P.circle(12,12,9.6,'url(#solid)'),defs=defs)

def orientation():
    # Interactive camera orientation cue differs from the plain axes triad.
    return P.svg('<title>Interactive camera orientation axes</title>'+P.circle(12,12,10,P.BLUE_PALE)+P.arrow(10,14,19.5,14,P.BLUE_STRONG,head=1.6,sw=.9)+P.arrow(10,14,10,4,P.GREEN,head=1.6,sw=.9)+P.arrow(10,14,4.4,19.6,P.YELLOW,head=1.6,sw=.9)+label_path('X',x=17,y=12,width=3,height=3)+label_path('Y',x=8.5,y=6.8,width=3,height=3)+label_path('Z',x=3,y=21,width=3,height=3))

ICONS={
'pqAxes.svg':axes,'pqContours.svg':contours,'pqFixedRepr32.svg':fixed_repr,'pqInvert.svg':invert,
'pqRamp24.svg':ramp,'pqRectilinearGrid16.svg':rect_grid,'pqSaveTable32.svg':save_table,'pqSelect16.svg':select_only,
'pqXMinus.svg':lambda:_shift('X','-',P.BLUE_STRONG),'pqXPlus.svg':lambda:_shift('X','+',P.BLUE_STRONG),
'pqYMinus.svg':lambda:_shift('Y','-',P.GREEN),'pqYPlus.svg':lambda:_shift('Y','+',P.GREEN),
'pqZMinus.svg':lambda:_shift('Z','-',P.YELLOW),'pqZPlus.svg':lambda:_shift('Z','+',P.YELLOW),
'pqCalculator.svg':calculator,'pqFaceSetData.svg':faceset,'pqProbeLocation.svg':probe,
'pqRecentFile.svg':recent,'pqSolidColor.svg':solid_color,'pqShowOrientationAxes.svg':orientation,
}
