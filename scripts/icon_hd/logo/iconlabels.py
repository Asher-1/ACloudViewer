"""Font-independent algorithm labels: font used only at authoring time, never SVG runtime."""
from functools import lru_cache
from pathlib import Path
from html import escape
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.boundsPen import BoundsPen
from . import paraview_style as P

FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')

@lru_cache(maxsize=64)
def label_path(text,x=2,y=22,width=20,height=4):
    font=TTFont(FONT);glyphs=font.getGlyphSet();cmap=font.getBestCmap();advance=0;paths=[]
    minx=miny=float('inf');maxx=maxy=float('-inf')
    for ch in text:
        name=cmap[ord(ch)];bounds=BoundsPen(glyphs);glyphs[name].draw(bounds)
        if bounds.bounds:
            a,b,c,d=bounds.bounds;minx=min(minx,advance+a);maxx=max(maxx,advance+c);miny=min(miny,b);maxy=max(maxy,d)
        pen=SVGPathPen(glyphs);glyphs[name].draw(TransformPen(pen,(1,0,0,-1,advance,0)))
        paths.append(pen.getCommands());advance+=font['hmtx'][name][0]
    font.close();scale=min(width/(maxx-minx),height/(maxy-miny))
    tx=x+(width-(maxx-minx)*scale)/2-minx*scale;ty=y+miny*scale
    return f'<path data-label="{escape(text, quote=True)}" d="'+''.join(paths)+f'" transform="translate({tx:.5f} {ty:.5f}) scale({scale:.6f})" fill="{P.INK}" stroke="none"/>'

def with_label(raw,text):
    import xml.etree.ElementTree as E
    node=E.fromstring(raw);defs=[];art=[]
    for child in node:
        name=child.tag.rsplit('}',1)[-1]
        if name=='defs':defs.extend(E.tostring(c,encoding='unicode') for c in child)
        elif name!='title':art.append(E.tostring(child,encoding='unicode'))
    # Uniform scale, rather than anisotropic squeezing, retains source geometry.
    body=f'<title>{text} — labeled functional PCL icon</title>'
    body+='<g transform="translate(3 .2) scale(.75)">'+''.join(art)+'</g>'
    body+=P.rect(1.3,18.1,21.4,5.5,P.BLUE_PALE,r=.6)
    body+=label_path(text,x=2,y=22.6,width=20,height=3.9)
    return P.svg(body,defs=''.join(defs))
