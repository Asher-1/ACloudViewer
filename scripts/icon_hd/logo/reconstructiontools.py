"""Reconstruction module toolbar artwork in the measured ParaView language.

All 24-unit authoring coordinates remain independent of the 128px native preview.
No raster tracing, live font, SVG filter, or external image dependency is used.
"""
import math
from . import paraview_style as P
from .iconlabels import label_path


def svg(title,body,defs=''):
    return P.svg('<title>'+title+'</title>'+body,defs=defs).replace('width="24" height="24"','width="128" height="128"',1)


def cloud(dense=False):
    points=[(7,9),(11,6),(15,8),(6,13),(11,11),(16,12),(8,17),(13,17),(18,16)]
    if dense:points += [(9,8),(13,8),(8,11),(14,11),(9,14),(12,14),(15,15),(11,19),(16,19)]
    return ''.join(P.circle(x,y,.65 if dense else .85,P.GREEN if i%3==0 else P.BLUE_STRONG,outline=None) for i,(x,y) in enumerate(points))


def camera(x=2,y=3,scale=1):
    return f'<g transform="translate({x} {y}) scale({scale})">'+P.fill('M 0 2 H 2 L 3 0 H 6 L 7 2 H 10 V 8 H 0 Z',P.GREY_LT,sw=.7)+P.circle(5,5,2.2,P.BLUE_MID)+P.circle(5,5,.9,P.BLUE_STRONG,outline=None)+'</g>'


def picture(x=2,y=3,w=19,h=15):
    return P.rect(x,y,w,h,P.BLUE_PALE,r=.5)+P.poly([(x+1,y+h-1),(x+w*.34,y+h*.42),(x+w*.62,y+h*.72),(x+w*.79,y+h*.38),(x+w-1,y+h-1)],P.GREEN_LT)+P.circle(x+w*.75,y+h*.23,1.2,P.YELLOW)


def badge(text,x=4,y=22,w=16,h=3.6):
    return P.rect(x-.6,y-h-.4,w+1.2,h+1.1,'#FAFAF6',r=.5)+label_path(text,x=x,y=y,width=w,height=h)


def gear(cx=17,cy=16,r=4.4):
    pts=[]
    for i in range(32):
        a=math.pi*i/16;rr=r if i%4 in (0,1) else r*.76
        pts.append((round(cx+math.cos(a)*rr,3),round(cy+math.sin(a)*rr,3)))
    return P.poly(pts,P.GREY)+P.circle(cx,cy,r*.34,'#FAFAF6')


def folder():
    return P.fill('M 2 5 H 8.5 L 10.5 7 H 21 V 20 H 2 Z',P.TAN)+P.fill('M 3 10 H 22 L 19.5 21 H 1.8 Z',P.YELLOW)


def file_sheet():
    return P.fill('M 4 2 H 14 L 20 8 V 22 H 4 Z','#FAFAF6')+P.fill('M 14 2 V 8 H 20',P.BLUE_MID)


def transfer(direction,kind):
    # Both import and export preserve the original model-file/arrow association.
    b=file_sheet()+P.cube(11,12.1,4,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREEN_LT)
    if direction=='in':b+=P.fill('M 2 13 H 7 V 10 L 12 15 L 7 20 V 17 H 2 Z',P.GREEN,sw=.65)
    else:b+=P.fill('M 12 14 H 17 V 11 L 22.5 16 L 17 21 V 18 H 12 Z',P.GREEN,sw=.65)
    if kind=='all':b= '<g transform="translate(1 .4) scale(.88)">'+file_sheet()+b+'</g>'+badge('ALL',x=1.5,y=22.3,w=8,h=3)
    elif kind=='as':b+=P.circle(18.9,5.2,3.2,P.YELLOW)+P.stroke('M 17 5.2 H 20.8 M 18.9 3.3 V 7.1',sw=.7)
    elif kind=='text':b+=badge('TXT',x=4,y=21.7,w=10,h=3.4)
    elif kind=='from':b+=P.stroke('M 15 4 H 18 M 15 6 H 18',sw=.65)
    return svg(('Import' if direction=='in' else 'Export')+' reconstruction model '+kind,b)


def project_new():
    return svg('Create new reconstruction project',folder()+P.circle(17.9,17.7,4.5,P.GREEN_LT)+P.stroke('M 15.3 17.7 H 20.5 M 17.9 15.1 V 20.3',sw=1.15))


def project_open():
    # Reuse Reconstruction's camera cue to distinguish an image project from
    # the global folder-plus action.
    return svg('Open photogrammetry project (.ini)', folder()+camera(10,2,1.15))


def project_edit():
    pencil=P.fill('M 7 19 L 8 15 L 17 6 L 20 9 L 11 18 Z',P.YELLOW,sw=.7)+P.fill('M 7 19 L 8 15 L 11 18 Z',P.GREY_DK,sw=.6)
    return svg('Edit reconstruction project',folder()+pencil)


def db():
    b=P.fill('M 3 5 H 17 V 18 C 17 23 3 23 3 18 Z',P.BLUE_MID)+ '<ellipse cx="10" cy="5" rx="7" ry="2.8" fill="'+P.BLUE_PALE+'"/>'+P.stroke('M 3 11 C 3 15 17 15 17 11 M 3 16 C 3 20 17 20 17 16',sw=.65)+gear(18,17,4.2)
    return svg('Manage reconstruction images, cameras and matches database',b)


def automatic():
    b=camera(1.5,2,.75)+camera(15,2,.75)+cloud()+P.stroke('M 6 7 L 10 11 M 18 7 L 13 11',sw=.5)+gear(18.3,17,4.1)+badge('AUTO',x=2,y=22,w=11,h=3)
    return svg('Automatic reconstruction pipeline — cameras and reconstructed points',b)


def features():
    b=picture(2,2.5,20,15)
    for x,y in [(6,11),(10,8),(16,10),(18,6)]:
        b+=P.circle(x,y,1.15,'none',outline=P.BLUE_STRONG)+P.stroke(f'M {x-1.8} {y} H {x+1.8} M {x} {y-1.8} V {y+1.8}',colour=P.BLUE_STRONG,sw=.6)
    return svg('Extract image feature keypoints',b+badge('FEAT',x=5,y=22,w=14,h=3.1))


def matching():
    b=picture(1.8,3,8.1,13)+picture(14.1,3,8.1,13)
    for y1,y2,c in [(6,8,P.GREEN_DK),(10,6,P.BLUE_STRONG),(14,13,P.TAN_DK)]:
        b+=P.line(7.8,y1,16.3,y2,c,.65)+P.circle(7.8,y1,.8,P.YELLOW)+P.circle(16.3,y2,.8,P.YELLOW)
    return svg('Match image feature correspondences',b+badge('MATCH',x=3,y=22,w=18,h=3.1))


def bundle():
    b=camera(1,2,.7)+camera(16,2,.7)+P.stroke('M 4.5 7 L 8 14 L 19.5 7 M 4.5 7 L 15 16 L 19.5 7',colour=P.GREY_MID,sw=.6)+P.circle(8,14,1,P.GREEN)+P.circle(15,16,1,P.BLUE_STRONG)+P.stroke('M 7 11 Q 11 8 16 10',colour=P.BLUE_STRONG,sw=.7)+P.poly([(16,10),(13.8,8.6),(14.3,11)],P.BLUE_STRONG,outline=None)
    return svg('Bundle adjustment jointly optimizes camera poses and 3D points',b+badge('BA',x=8,y=22,w=8,h=3.4))


def dense():
    return svg('Dense multi-view reconstruction',P.poly([(3,8),(11,3),(21,8),(19,19),(9,22),(2,16)],P.BLUE_PALE)+P.stroke('M 3 8 L 12 12 L 11 3 M 12 12 L 21 8 M 12 12 L 19 19 M 12 12 L 9 22 M 12 12 L 2 16',sw=.55)+cloud(True))


def play():
    return svg('Start incremental reconstruction',P.fill('M 6 3 L 21 12 L 6 21 Z',P.GREEN))


def pause():
    return svg('Pause reconstruction',P.rect(5,3,5.5,18,P.YELLOW,r=.6)+P.rect(14,3,5.5,18,P.YELLOW,r=.6))


def step():
    return svg('Reconstruct next image — one incremental step',P.fill('M 3 4 L 16 12 L 3 20 Z',P.GREEN)+P.rect(17.5,4,3.5,16,P.BLUE_MID,r=.4)+P.circle(6.5,4,2.8,P.YELLOW)+P.stroke('M 5 4 H 8 M 6.5 2.5 V 5.5',sw=.7))


def reset():
    return svg('Reset reconstruction',P.stroke('M 20 16 A 8.5 8.5 0 1 1 18.6 6.2',colour=P.GREEN,sw=3)+P.fill('M 14.4 6.8 L 21.5 2.5 V 11 Z',P.GREEN,sw=.6)+P.circle(12,12,2,P.BLUE_PALE))


def normalize():
    b=P.cube(12,11,5.7,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREEN_LT)+P.stroke('M 2 5 V 2 H 5 M 19 2 H 22 V 5 M 2 17 V 20 H 5 M 19 20 H 22 V 17',sw=.9)+P.arrow(2.5,11,6,11,P.YELLOW,head=1.3,sw=.8)+P.arrow(21.5,11,18,11,P.YELLOW,head=1.3,sw=.8)
    return svg('Normalize reconstruction scale and center',b)


def options(render=False):
    b=(P.rect(2,2,15,12,P.BLUE_PALE,r=.6)+P.stroke('M 6 17 H 14 M 10 14 V 17',sw=.8)) if render else cloud()
    return svg('Rendering options' if render else 'Reconstruction options',b+gear(17.5,16.5,4.7))


def stats():
    b=P.line(3,3,3,21)+P.line(3,21,22,21)+P.rect(6,13,3,8,P.BLUE_MID,r=.2)+P.rect(11,8,3,13,P.GREEN_LT,r=.2)+P.rect(16,3,3,18,P.YELLOW,r=.2)
    return svg('Reconstruction model statistics',b)


def matrix():
    b=P.rect(3,3,18,18,'#FAFAF6',r=.4)
    colors=[[P.BLUE_STRONG,P.BLUE_MID,P.GREY_LT,P.GREEN_LT],[P.BLUE_MID,P.BLUE_STRONG,P.GREEN,P.GREY_LT],[P.GREY_LT,P.GREEN,P.BLUE_STRONG,P.TAN],[P.GREEN_LT,P.GREY_LT,P.TAN,P.BLUE_STRONG]]
    for y in range(4):
        for x in range(4):b+=P.rect(3.5+x*4.25,3.5+y*4.25,4,4,colors[y][x],r=0,outline=None)
    return svg('Pairwise image match matrix',b)


def log():
    return svg('Show reconstruction log',P.rect(2,3,20,18,P.GREY_LT,r=.7)+P.rect(2,3,20,4,P.BLUE_MID,r=.4)+P.stroke('M 5 10 L 7 12 L 5 14 M 10 15 H 16 M 5 18 H 19',sw=1))


def render_state(enabled):
    b=P.fill('M 1.8 11.8 Q 12 1 22.2 11.8 Q 12 22.6 1.8 11.8 Z',P.BLUE_PALE if enabled else P.GREY_LT)+P.circle(12,11.8,4,P.BLUE_MID if enabled else P.GREY)+P.circle(12,11.8,1.7,P.INK)
    if not enabled:b+=P.stroke('M 3 21 L 21 3',colour=P.GREY_DK,sw=1.8)
    return svg('Rendering enabled' if enabled else 'Rendering disabled',b)


def reset_view():
    b=P.cube(12,12,5.5,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREEN_LT)+P.stroke('M 21 15 A 9 9 0 1 1 19 5.8',colour=P.GREEN_DK,sw=1.4)+P.fill('M 15 6 L 22 2 V 10 Z',P.YELLOW,sw=.6)
    return svg('Reset reconstruction viewport camera',b)


def movie():
    b=P.rect(2,4,16,16,P.GREY,r=.7)+P.rect(5,6,10,12,P.BLUE_PALE,r=.4)+P.fill('M 8 8 L 14 12 L 8 16 Z',P.GREEN)+P.fill('M 18 8 L 23 5 V 19 L 18 16 Z',P.TAN)
    for y in [6,10,14,18]:b+=P.rect(2.8,y,1.5,1.4,'#FAFAF6',r=0,outline=None)
    return svg('Record reconstruction camera movie',b)


def undistort():
    b=P.rect(2,3,9,14,P.BLUE_PALE,r=.5)+P.stroke('M 4 4 Q 7 10 4 16 M 9 4 Q 6 10 9 16 M 3 7 Q 6.5 9 10 7 M 3 13 Q 6.5 11 10 13',sw=.55)+P.rect(15,3,7,14,P.GREEN_LT,r=.5)+P.stroke('M 17.3 3.5 V 16.5 M 19.7 3.5 V 16.5 M 15.5 7.5 H 21.5 M 15.5 12.5 H 21.5',sw=.55)+P.arrow(9.5,20,20,20,P.YELLOW,head=1.6,sw=1)
    return svg('Undistort images — curved lens grid becomes rectilinear',b)


def emblem():
    return svg('Reconstruction application mark — camera network and 3D model',P.circle(12,12,10.8,P.BLUE_PALE)+P.cube(12,13,6.2,top=P.GREEN_LT,left=P.BLUE_MID,right=P.BLUE_STRONG)+camera(1.8,3,.55)+camera(16.7,3,.55)+P.stroke('M 4.5 7.5 L 12 13 L 19.5 7.5',colour=P.INK,sw=.6))


BUILDERS={
'automatic-reconstruction.svg':automatic,'bundle-adjustment.svg':bundle,'database-management.svg':db,
'dense-reconstruction.svg':dense,'export-all.svg':lambda:transfer('out','all'),
'export-as-text.svg':lambda:transfer('out','text'),'export-as.svg':lambda:transfer('out','as'),
'export.svg':lambda:transfer('out','model'),'feature-extraction.svg':features,'feature-matching.svg':matching,
'grab-movie.svg':movie,'import-from.svg':lambda:transfer('in','from'),'import.svg':lambda:transfer('in','model'),
'log.svg':log,'match-matrix.svg':matrix,'project-edit.svg':project_edit,'project-new.svg':project_new,'project-open.svg':project_open,
'reconstruction-normalize.svg':normalize,'reconstruction-options.svg':lambda:options(False),
'reconstruction-pause.svg':pause,'reconstruction-reset.svg':reset,'reconstruction-start.svg':play,
'reconstruction-stats.svg':stats,'reconstruction-step.svg':step,'render-disabled.svg':lambda:render_state(False),
'render-enabled.svg':lambda:render_state(True),'render-options.svg':lambda:options(True),
'render-reset-view.svg':reset_view,'undistort.svg':undistort,
}
# Standard file operations keep ParaView's original authored paths, with only
# intrinsic dimensions adjusted to the explicit preview requirement.
UPSTREAM={'project-save.svg':'pqSave.svg',
          'project-save-as.svg':'pqSaveAs.svg','grab-image.svg':'pqCaptureScreenshot.svg'}
