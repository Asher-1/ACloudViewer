"""File and scene actions migrated from loose raster assets; pure SVG geometry."""
from . import paraview_style as P
from .icon import fit


def disk(binary=False):
    defs = P.linear('case', [(0, P.BLUE_PALE), (1, P.BLUE_STRONG)])
    body = [P.fill('M 2 2 H 20 L 22 4 V 22 H 2 Z', 'url(#case)'),
            P.rect(5, 2, 13, 10, P.GREY_LT, r=0.3),
            P.rect(6, 16, 12, 6, P.GREY_LT, r=0.3),
            P.rect(8, 17, 2.2, 4, P.GREY_DK, r=0.1),
            P.rect(5, 2, 13, 2, P.TAN, r=0)]
    if binary:
        # Original 1/0 binary label, authored as paths rather than live text.
        body += [P.rect(9, 5, 13.5, 8, P.GREY_LT, r=0.5),
                 P.stroke('M 11 7 L 12 6.5 V 11 M 10.8 11 H 13.2', sw=1),
                 P.fill('M 15 6.5 H 19 V 11 H 15 Z M 16.2 7.6 V 9.9 H 17.8 V 7.6 Z', P.INK, outline=None).replace('/>', ' fill-rule="evenodd"/>')]
    return P.svg('<title>Save project as BIN</title>' if binary else '<title>Save</title>'+'\n', defs=defs).replace('</svg>', '\n'.join(body)+'\n</svg>')


def open_file():
    defs=P.linear('folder',[(0,P.BLUE_PALE),(1,P.BLUE_STRONG)])
    return P.svg('<title>Open file</title>\n'+'\n'.join([
        P.fill('M 2 4 H 10 L 12 6 H 22 V 21 H 2 Z','url(#folder)'),
        P.fill('M 5 9 H 22 L 19 21 H 2 Z',P.BLUE_MID),
        P.fill('M 12 9 H 18 V 13 H 20 L 15 18 L 10 13 H 12 Z',P.YELLOW,sw=0.7)]),defs=defs)


def colorize():
    defs=P.linear('metal',[(0,P.GREY_LT),(1,P.GREY_MID)])
    body=[]
    for x,y,c in [(13,12,P.GREEN),(18,12,P.BLUE_STRONG),(13,17,P.YELLOW),(18,17,P.TAN)]:
        body.append(P.rect(x,y,5,5,c,r=0))
    body += [P.stroke('M 4 7 V 3 C 4 -0.5 10 -0.5 10 4',sw=0.85),
             P.fill('M 5 3 L 15 11 L 8 18 L 1 11 Z','url(#metal)'),
             P.stroke('M 5 3 L 15 11',colour=P.GREY_DK,sw=1.1),
             P.fill('M 15.5 10 C 14 12 13.6 13.5 15.5 13.5 C 17.4 13.5 17 12 15.5 10 Z',P.BLUE_STRONG,sw=0.6)]
    return P.svg('<title>Colorize entities</title>\n'+'\n'.join(body),defs=defs)


def clear_all():
    defs=P.linear('bristle',[(0,P.TAN),(1,P.YELLOW)])
    body=[P.fill('M 3 1.5 L 5 1 L 13 13 L 11 15 Z',P.BROWN),
          P.fill('M 9 11 L 13 8 L 16 12 L 12 15 Z',P.GREY_LT),
          P.fill('M 12 14 L 16 11 L 23 19 L 15 23 L 8 19 Z','url(#bristle)'),
          P.stroke('M 13 16 L 17 21 M 15.5 15 L 19 20 M 11 18 L 14 21',colour=P.TAN_DK,sw=0.65)]
    return P.svg('<title>Clear all — broom</title>\n'+'\n'.join(body),defs=defs)


ICONS={
 'open.png':('toolMainOpen',open_file),
 'save.png':('toolMainSave',disk),
 'saveBinary.png':('toolMainSaveProject',lambda:disk(True)),
 'bgcolor.png':('toolMainColorize',colorize),
 'clear.png':('toolMainClearAll',clear_all),
}

def built(old):
    return fit(ICONS[old][1]())
