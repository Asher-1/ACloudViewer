"""True vector redraws for ParaView-origin rasters formerly loose in svg/."""
from . import paraview_style as P
from .icon import fit

def camera():
 b=[P.fill('M 2 6 H 7 L 9 3 H 15 L 17 6 H 22 V 20 H 2 Z',P.GREY),P.circle(12,12.5,6,P.BLUE_MID),P.circle(12,12.5,3.5,P.BLUE_STRONG),P.circle(10.8,11.2,1.3,P.BLUE_PALE,outline=None),P.rect(18,7.5,2,1.4,P.YELLOW,r=.2)]
 return P.svg('<title>Camera sensor</title>'+''.join(b))
def exit_app():
 return P.svg('<title>Exit application</title>'+P.fill('M 3 2 H 15 V 22 H 3 Z',P.TAN)+P.fill('M 5 4 L 13 2 V 22 L 5 20 Z',P.GREY_LT)+P.fill('M 10 10 H 17 V 6 L 23 12 L 17 18 V 14 H 10 Z',P.GREEN))
def eye():
 return P.svg('<title>Viewport eye</title>'+P.fill('M 1.5 12 Q 12 0 22.5 12 Q 12 24 1.5 12 Z',P.BLUE_PALE)+P.circle(12,12,5,P.BLUE_MID)+P.circle(12,12,2.5,P.INK)+P.circle(10.8,10.8,.8,'#FAFAF6',outline=None))
def full():
 b=[P.rect(3,3,18,18,P.BLUE_PALE,r=.4)]
 for d in ['M 2 8 V 2 H 8 L 6.1 4 L 9 6.9 L 6.9 9 L 4 6.1 Z','M 16 2 H 22 V 8 L 20 6.1 L 17.1 9 L 15 6.9 L 17.9 4 Z','M 2 16 V 22 H 8 L 6.1 20 L 9 17.1 L 6.9 15 L 4 17.9 Z','M 16 22 H 22 V 16 L 20 17.9 L 17.1 15 L 15 17.1 L 17.9 20 Z']:b.append(P.fill(d,P.YELLOW,sw=.55))
 return P.svg('<title>Fullscreen view</title>'+''.join(b))
def glyph():
 b=[P.poly([(3,18),(11,13),(21,16),(13,22)],P.BLUE_PALE)]
 for x,y in [(5,14),(11,11),(17,14)]:
  b+=[P.fill(f'M {x-1} {y+3} H {x+1} V {y-3} H {x+2.4} L {x} {y-6} L {x-2.4} {y-3} H {x-1} Z',P.GREEN,sw=.55)]
 return P.svg('<title>Glyph vectors on sampled locations</title>'+''.join(b))
def paste():
 return P.svg('<title>Paste cloud data from clipboard</title>'+P.rect(3,3,15,19,P.TAN,r=.8)+P.rect(6,5,13,16,P.GREY_LT,r=.3)+P.rect(7,1.5,7,4,P.GREY,r=.5)+P.rect(10,9,11,12,P.BLUE_PALE,r=.3)+P.stroke('M 12 12 H 19 M 12 15 H 19 M 12 18 H 17',sw=.7))
def preview():
 return P.svg('<title>Preview selected result</title>'+P.rect(2,3,20,17,P.GREY,r=.6)+P.rect(4,5,16,12,P.BLUE_PALE,r=.2)+P.fill('M 9 7 L 16 11 L 9 15 Z',P.GREEN)+P.stroke('M 8 22 H 16 M 12 20 V 22',sw=1))
def side_set():
 return P.svg('<title>Representation side-set data</title>'+P.cube(12,12,10,top=P.GREY_LT,left=P.BLUE_PALE,right=P.GREEN)+P.stroke('M 12 12 L 20.66 7 M 12 17 L 20.66 12 M 16.33 9.5 V 19.5',sw=.65))
def stream():
 b=[P.poly([(2,18),(9,14),(22,17),(15,22)],P.BLUE_PALE)]
 for x in [4,10,16]:
  b+=[P.stroke(f'M {x} 18 C {x+9} 13 {x-3} 9 {x+4} 3',colour=P.BLUE_STRONG,sw=1.05),P.poly([(x+4,3),(x+1.8,4),(x+4.5,5.5)],P.YELLOW,outline=P.INK)]
 return P.svg('<title>Stream tracer follows the vector field</title>'+''.join(b))
def wrench():
 return P.svg('<title>General filter settings — wrench</title>'+P.fill('M 15 2 C 8-1 7 7 10 9 L 2.5 17.5 C .5 19.5 4.5 23.5 6.5 21.5 L 14 14 C 20 17 25 10 21 6 L 17 10 L 14 7 Z',P.GREY_LT)+P.circle(5,19,1,P.BLUE_MID))
# Clip, Slice, Threshold and IsoSurface are restored byte-exact from the upstream
# ParaView originals into svg/filter/ (see pq_original_sources.json, dir=filter);
# the former custom redraws here were rejected for not matching the operations.
# Glyph and StreamTracer keep their approved redraws but live in svg/filter/ too,
# so their targets carry the filter/ subpath (resolved relative to the svg root).
ICONS={
 'pqCamera.png':('pqCamera.svg',camera,'Camera'),
 'pqExitApp.png':('pqExitApp.svg',exit_app,'Exit'), 'pqEyeball.png':('pqEyeball.svg',eye,'Viewport'),
 'pqFullscreen.png':('pqFullscreen.svg',full,'Fullscreen'),
 'pqGlyph.png':('filter/filterGlyph.svg',glyph,'Glyph'),
 'pqPaste.png':('pqPaste.svg',paste,'Paste'),
 'pqPreview.png':('pqPreview.svg',preview,'Preview'), 'pqSideSetData.png':('pqSideSetData.svg',side_set,'Side-set data'),
 'pqStreamTracer.png':('filter/filterStreamTracer.svg',stream,'Stream tracer'),
 'pqWrench.png':('pqWrench.svg',wrench,'Filter settings'),
}
