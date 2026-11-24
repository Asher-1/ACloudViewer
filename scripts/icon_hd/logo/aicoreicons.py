"""ParaView-palette marks for AICore plugin launch icons, with prominent names."""
from . import paraview_style as P
from .iconlabels import label_path

# Full visible plugin names are deliberately split, never silently abbreviated.
NAME_LINES={
 'DA3':('DA3',),'DeepLSD':('DeepLSD',),'FaceDetect':('Face','Detect'),
 'FreeSplatter':('Free','Splatter'),'GKD':('qGKD',),'InstantMesh':('Instant','Mesh'),
 'LightGlue':('Light','Glue'),'LingBot-Map':('LingBot-','Map'),
 'MapAnything':('MAP','ANYTHING'),'OCR':('OCR',),'OpenPCDet':('Open','PCDet'),
 'QuerySplat':('QUERY','SPLAT'),'RAM++':('RAM++',),'RF-DETR':('RF-DETR',),
 'RMBG':('RMBG',),'RelateAnything':('Relate','Anything'),'SAM3':('SAM3',),
 'SAM3D':('Sam3D',),'TRELLIS.2':('TRELLIS.2',),'VLX-Seek':('VLX-Seek',),'YOLO':('YOLO',),
}
SOURCE_CAPTIONS={
 'DA3':('Depth',),'DeepLSD':('Deep LSD',),'FreeSplatter':('FS',),
 'LightGlue':('LG',),'LingBot-Map':('qLingbotMap',),
 'MapAnything':('VGGT','MAP·3D'),'QuerySplat':('VGGT','3DGS'),
 'RelateAnything':('SCENE RELATIONS',),'TRELLIS.2':('qTrellis',),'YOLO':('DEP',),
}

# Low-chroma identity surfaces, not independent saturated product logos.
# The family label describes the verified operation; original captions and full
# plugin names remain font-independent paths. Native toolbar art is unchanged.
FAMILY={
'DA3':('depth','#E8EFF4','#668699'),'DeepLSD':('lines','#EDF0E7','#7C8A61'),
'FaceDetect':('identity','#F2ECE4','#9A846A'),'FreeSplatter':('splats','#E9EFE8','#73896C'),
'GKD':('keypoints','#F1EEE3','#968D68'),'InstantMesh':('mesh','#E7EDF1','#758C9A'),
'LightGlue':('matching','#EBEEF5','#7987A5'),'LingBot-Map':('trajectory','#E8F0EE','#6B8D83'),
'MapAnything':('mapping','#E8EDF1','#6D899C'),'OCR':('text','#F1EEE5','#96896E'),
'OpenPCDet':('boxes','#EDF0E5','#81916A'),'QuerySplat':('query','#EDEAF2','#8E809B'),
'RAM++':('tags','#ECEEE5','#8B916B'),'RF-DETR':('detect','#E7EEEF','#6A8B91'),
'RMBG':('matte','#EEEEEC','#888B85'),'RelateAnything':('relations','#F1EDE6','#998B73'),
'SAM3':('mask','#E8F0E9','#718E77'),'SAM3D':('shape','#E7EEF0','#6D8A94'),
'TRELLIS.2':('generation','#EDEAF0','#8A7E96'),'VLX-Seek':('search','#EBEDF4','#7A84A1'),
'YOLO':('detect-depth','#EEEFE3','#8B9060')}

def _tile(mark,name):
 family,base,accent=FAMILY[name]
 b=f'<title>{name} — {family} plugin</title>'
 # A low-chroma plate gives disabled/light surfaces a visual anchor without
 # turning twenty-one commands into saturated app-store badges.
 b+=f'<rect x="2" y="1" width="60" height="48" rx="5" fill="{base}" stroke="#ABB2B3" stroke-width=".65"/>'
 b+=f'<path d="M 6 10 V 38" stroke="{accent}" stroke-width="2.3" stroke-linecap="round"/>'
 b+=f'<g data-feature="{family}" transform="translate(8 0) scale(2)" stroke-width=".85">{mark}</g>'
 b+='<rect x="2" y="48" width="60" height="15.5" rx="2" fill="#F2F4F2"/>'
 captions=SOURCE_CAPTIONS.get(name,())
 if captions:
  span=60/len(captions)
  b+='<g data-source-captions="true">'
  for i,text in enumerate(captions):b+=label_path(text,x=2+i*span,y=49.7,width=span-1,height=2.5)
  b+='</g>'
 lines=NAME_LINES[name]
 b+=f'<g data-plugin-name="{name}" data-name-lines="{len(lines)}">'
 for i,text in enumerate(lines):
  y=56.4+i*7.1 if len(lines)==2 else 62.2
  b+=label_path(text,x=2,y=y,width=60,height=6.3 if len(lines)==2 else 9.0)
 b+='</g>'
 return P.svg(b,w=64,h=64,sw=1.35).replace('width="64" height="64"','width="128" height="128"',1)

def _ptcloud(pts=None,color=None,r=.58):
    pts=pts or [(4,6),(8,4),(12,7),(17,4.5),(6,11),(11,10),(16,11),(4.5,16),(9,15),(14,17),(19,15)]
    return ''.join(P.circle(x,y,r,color or P.BLUE_STRONG,outline=None) for x,y in pts)
def _camera(x=3,y=3,s=1):
    return f'<g transform="translate({x} {y}) scale({s})">'+P.fill('M 0 2 H 2.5 L 3.6 0 H 7 L 8.2 2 H 11 V 9 H 0 Z',P.GREY_LT)+P.circle(5.5,5.5,2.5,P.BLUE_MID)+P.circle(5.5,5.5,1,P.BLUE_STRONG,outline=None)+'</g>'
def _frame(x,y,w,h,c=P.GREEN):return P.stroke(f'M {x} {y+3} V {y} H {x+4} M {x+w-4} {y} H {x+w} V {y+3} M {x+w} {y+h-3} V {y+h} H {x+w-4} M {x+4} {y+h} H {x} V {y+h-3}',colour=c,sw=.9)
def _mesh():
 pts=[(3,7),(10,3),(19,6),(4,15),(12,12),(21,15),(8,21),(16,21)]
 tris=[(0,1,3),(1,2,4),(2,5,4),(3,4,6),(4,5,7),(6,4,7)]
 b=[P.poly([pts[i] for i in t],P.BLUE_PALE) for t in tris]
 b += [P.stroke('M 3 7 L 10 3 L 19 6 L 21 15 L 16 21 L 8 21 L 4 15 Z M 3 7 L 12 12 L 19 6 M 12 12 L 21 15 M 12 12 L 8 21 M 4 15 L 16 21',sw=.55)]
 return ''.join(b)
def _label_doc():
 return P.fill('M 5 2 H 16 L 20 6 V 21 H 5 Z',P.BLUE_PALE)+P.fill('M 16 2 V 6 H 20',P.BLUE_MID)+P.stroke('M 8 10 H 16 M 8 13 H 16 M 8 16 H 13',sw=.7)
def _relation():
    b=P.rect(1,2,7,6,P.BLUE_PALE,r=1)+P.rect(16,2,7,6,P.GREEN_LT,r=1)+P.rect(8.5,14,7,6,P.YELLOW,r=1)
    # Subject, object and video-frame nodes retain the original relationship story.
    b+=P.circle(4.5,3.7,.65,P.INK,outline=None)+P.stroke('M 4.5 4.7 V 6.1 M 2.9 5.6 L 4.5 4.7 L 6.1 5.6 M 4.5 6.1 L 3.1 7.2 M 4.5 6.1 L 5.9 7.2',sw=.38)
    b+=P.stroke('M 17.1 5.3 H 20.1 L 20.8 3.6 L 22.1 4.1 L 21.7 5.7 H 20.6 M 17.7 5.3 L 17 4.2 M 17.6 5.7 V 7.1 M 18.8 5.7 L 19 7.1 M 20.4 5.7 V 7.1 M 21.4 5.8 L 22 7.1',sw=.4)
    b+=P.rect(9.7,15.3,4.6,3.4,P.BLUE_PALE,r=.2)+P.poly([(11.3,15.9),(13.2,17),(11.3,18.1)],P.INK,outline=None)
    b+=P.arrow(8.2,5,15.6,5,P.GREEN_DK,head=1.1,sw=.6)+P.arrow(5,8.4,9.6,13.5,P.GREY_DK,head=1.1,sw=.65)+P.arrow(19,8.4,14.3,13.5,P.GREY_DK,head=1.1,sw=.65)
    b+=P.rect(7.7,8.6,8.6,3.8,'#FAFAF6',r=.5)+label_path('RIDES',x=8.1,y=11.7,width=7.8,height=2.5)
    return b

def _splats():
    defs=''
    shapes=[]
    for i,(x,y,rx,ry,color,light,angle) in enumerate([
        (6,7,4,2.5,P.BLUE_STRONG,P.BLUE_PALE,-22),
        (16,6,4.2,2.7,P.GREEN,P.GREEN_LT,18),
        (11,12,5.2,3.1,P.TAN_DK,P.YELLOW,-12),
        (5,16,3,1.9,P.GREEN_DK,P.GREEN_LT,30),
        (18,15,3.2,2,P.BLUE_STRONG,P.BLUE_MID,-25)]):
        gid=f'splat{i}'
        defs+=P.radial(gid,[(0,light),(1,color)],cx=.35,cy=.3,r=.8)
        shapes.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" transform="rotate({angle} {x} {y})" fill="url(#{gid})" stroke="{P.INK}" stroke-width=".5"/>')
    return '<defs>'+defs+'</defs>'+''.join(shapes)


def build(name):
 if name=='DA3':
  marks=P.cube(12,11,7,top=P.BLUE_PALE,left=P.BLUE_MID,right=P.GREY_LT)+P.stroke('M 5 8 L 10 11 M 19 8 L 14 11',sw=.65)+P.poly([(16,13),(18,14),(17,17),(14,18)],P.GREEN_LT)
 elif name=='DeepLSD':
  marks=P.poly([(3,4),(18,4),(21,18),(5,20)],P.BLUE_PALE)+P.stroke('M 4 7 L 19 7 M 5 12 L 20 11 M 5 17 L 20 16',colour=P.BLUE_STRONG,sw=.95)+_ptcloud([(4,7),(10,7),(19,7),(5,12),(13,12),(20,11),(5,17),(11,17),(20,16)],P.YELLOW,.5)
 elif name=='FaceDetect':
  marks=_frame(3,3,18,16)+P.fill('M 8 6 Q 12 3 16 6 L 17 11 Q 16 16 12 17 Q 8 16 7 11 Z',P.TAN)+P.circle(10,10,0.55,P.INK,outline=None)+P.circle(14,10,0.55,P.INK,outline=None)+P.stroke('M 10 13 Q 12 14 14 13',sw=.55)
 elif name=='FreeSplatter':
  marks=_splats()+_ptcloud([(4,18),(8,17),(12,19),(17,17),(20,19)],P.INK,.45)
 elif name=='GKD':
  marks=P.rect(2,1.5,20,20,P.BLUE_PALE,r=.8)+P.stroke('M 12 5 L 7 7 L 5 11 M 12 5 L 17 7 L 19 11 M 12 5 V 12 M 12 12 L 7 19 M 12 12 L 17 19',colour=P.BLUE_STRONG,sw=.85)+_ptcloud([(12,5),(7,7),(5,11),(17,7),(19,11),(12,12),(7,19),(17,19)],P.YELLOW,1)
 elif name=='InstantMesh':
  marks=_ptcloud([(2,6),(5,4),(2,12),(6,10),(4,17)],P.BLUE_STRONG,.7)+P.arrow(6.7,11,11,11,P.GREEN,head=1.3,sw=.9)+'<g transform="translate(9 1) scale(.6 .85)">'+_mesh()+'</g>'
 elif name=='LightGlue':
  left=[(4,5),(5,11),(4,18),(8,8),(8,15)];right=[(20,5),(19,11),(21,18),(16,8),(16,15)]
  links=[(0,0),(1,2),(2,4),(3,3),(4,1)]
  marks=''.join(P.line(*left[a],*right[b],P.GREY_MID,.55) for a,b in links)+_ptcloud(left,P.BLUE_STRONG,.78)+_ptcloud(right,P.GREEN,.78)
 elif name=='LingBot-Map':
  marks=P.stroke('M 4 7 C 7 3 17 3 20 8 C 22 12 18 18 12 18 C 7 18 4 14 6 10 C 8 7 13 7 15 10',colour=P.BLUE_STRONG,sw=1.1)+_camera(1,2,.48)+_camera(17,14,.48)+P.circle(12,10,1.5,P.YELLOW)
 elif name=='MapAnything':
  marks=_camera(1,6,.58)+_camera(17,6,.58)+P.arrow(7,11,11,11,P.GREEN,head=1.2,sw=.8)+P.poly([(4,18),(12,14),(21,17),(13,21)],P.BLUE_PALE)+P.stroke('M 4 18 L 13 21 L 21 17 M 13 21 V 17',sw=.55)
 elif name=='OCR':
  marks=_label_doc()+P.circle(17,16,3.1,'none',outline=P.GREEN)+P.line(19.3,18.3,22,21,P.BLUE_STRONG,1.1)
 elif name=='OpenPCDet':
  marks=_ptcloud()+P.stroke('M 4 9 H 14 V 19 H 4 Z M 9 4 H 19 V 14 H 9 Z M 4 9 L 9 4 M 14 9 L 19 4 M 14 19 L 19 14 M 4 19 L 9 14',colour=P.TAN_DK,sw=.85)
 elif name=='QuerySplat':
  marks=_splats()+P.circle(15.5,12,3.8,'none',outline=P.INK)+P.line(18.2,14.7,22,18.5,P.INK,1.4)+P.stroke('M 11.5 12 H 19.5 M 15.5 8 V 16',colour=P.BLUE_STRONG,sw=.65)
 elif name=='RAM++':
  marks=_label_doc()+P.poly([(8,11),(10.5,13.5),(15.5,8.5)],P.GREEN,outline=P.GREEN,close=False)+badge('TAG')
 elif name=='RF-DETR':
  marks=P.rect(3,5,12,13,P.BLUE_PALE,r=.5)+P.rect(11,12,9,9,'none',r=.5,outline=P.GREEN)+_frame(2,4,14,15,P.GREEN)+P.circle(19,5,2.2,'none',outline=P.BLUE_STRONG)+P.stroke('M 19 1 V 9 M 15 5 H 23',colour=P.BLUE_STRONG,sw=.65)
 elif name=='RMBG':
  marks=P.rect(2,2,20,20,P.GREY_LT,r=.6)+''.join(P.rect(x,y,5,5,P.BLUE_PALE,r=0,outline=None) for x,y in [(2,2),(12,2),(7,7),(17,7),(2,12),(12,12),(7,17),(17,17)])+P.circle(12,8,3.7,'#FAFAF6')+P.fill('M 5.5 21 C 5.5 13 18.5 13 18.5 21 Z','#FAFAF6')
 elif name=='RelateAnything':
  marks=_relation()
 elif name=='SAM3':
  marks=P.rect(2,2,20,19,P.BLUE_PALE,r=.8)+P.poly([(3,19),(9,8),(14,16),(18,10),(21,19)],P.GREY_LT)+P.fill('M 6 11 L 12 7 L 18 10 L 20 16 L 14 20 L 7 18 Z',P.GREEN_LT,sw=.7)+P.circle(8,12,1.2,P.GREEN_DK)+P.circle(18,16,1.2,P.ROSE)+_frame(11,5,8,6,P.BLUE_STRONG)
 elif name=='SAM3D':
  marks=_mesh()+P.circle(7,8,1.2,P.YELLOW)+P.circle(14,12,1.2,P.GREEN)+P.circle(10,18,1.2,P.ROSE)
 elif name=='TRELLIS.2':
  marks=P.rect(1,4,7,7,P.BLUE_PALE,r=.5)+P.poly([(1.5,10),(4,6),(7.5,10)],P.GREEN_LT)+P.arrow(9,8,13,8,P.GREEN,head=1.1,sw=.8)+P.circle(17.5,11,5.3,P.BLUE_PALE)+P.stroke('M 12.5 9 Q 17.5 5 22.5 9 M 12.5 13 Q 17.5 17 22.5 13 M 17.5 5.7 C 13 10 13 12 17.5 16.3 C 22 12 22 10 17.5 5.7',colour=P.INK,sw=.5)
 elif name=='VLX-Seek':
  marks=_frame(3,3,18,14,P.BLUE_STRONG)+P.rect(8,6,13,10,P.GREEN_LT,r=.7)+P.circle(9.5,8.5,5,'none',outline=P.GREY_DK)+P.line(13,12,19,18,P.GREY_DK,1.8)+P.rect(6,14,16,5,'#FAFAF6',r=.3)+label_path('<objN>',x=6.4,y=18,width=15.2,height=3.2)
 elif name=='YOLO':
  marks='<defs>'+P.linear('depth',[(i/5,c) for i,c in enumerate(P.FIELD)])+'</defs>'+_ptcloud()+_frame(3,3,9,8,P.GREEN)+_frame(10,11,8,8,P.YELLOW)+P.rect(20.3,2.5,2,18,'url(#depth)',r=.35)
 else:raise KeyError(name)
 return _tile(marks,name)

def badge(text):
 return P.rect(12.6,13.2,8.8,6.5,P.BLUE_PALE,r=.45)+label_path(text,x=13.1,y=18.2,width=7.8,height=3.2)

NAMES={
'qDA3':'DA3','qDeepLSD':'DeepLSD','qFaceDetect':'FaceDetect','qFreeSplatter':'FreeSplatter',
'qGKD':'GKD','qInstantMesh':'InstantMesh','qLightGlue':'LightGlue','qLingbotMap':'LingBot-Map',
'qMapAnything':'MapAnything','qOCR':'OCR','qOpenPCDet':'OpenPCDet','qQuerySplat':'QuerySplat',
'qRAM':'RAM++','qRFDetr':'RF-DETR','qRMBG':'RMBG','qRelateAnything':'RelateAnything',
'qSAM3':'SAM3','qSAM3D':'SAM3D','qTrellis':'TRELLIS.2','qVLXSeek':'VLX-Seek','qYOLO':'YOLO'}
