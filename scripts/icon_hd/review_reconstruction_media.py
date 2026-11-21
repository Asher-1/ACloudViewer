"""Numbered before/after review for all reconstruction-module toolbar icons."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageOps
from .logo.icon import _png,_font
ROOT=Path(__file__).resolve().parents[2]
REVIEW=Path.home()/'ACloudViewer-icon-reviews/reconstruction-media'

def render(path,size):
 data=path.read_bytes()
 if data.startswith(b'\x89PNG'):
  with Image.open(path) as im:
   im=ImageOps.contain(im.convert('RGBA'),(size,size),Image.Resampling.LANCZOS)
   c=Image.new('RGBA',(size,size));c.alpha_composite(im,((size-im.width)//2,(size-im.height)//2));return c
 return _png(data.decode(),size)

def main():
 ledger=json.loads((REVIEW/'migration-ledger.json').read_text());rows=ledger['icons'];out=REVIEW/'review';out.mkdir(exist_ok=True);sizes=out/'sizes';sizes.mkdir(exist_ok=True)
 for row in rows:
  target=Path(row['after']);assert hashlib.sha256(target.read_bytes()).hexdigest()==row['after_sha256']
  for s in [16,24,32,48,64,128,384,1024]:render(target,s).save(sizes/f"{row['id']:02}-{target.stem}-{s}.png")
 for g in range((len(rows)+5)//6):
  group=rows[g*6:g*6+6];board=Image.new('RGB',(1550,130+len(group)*230),'#f5f6f8');d=ImageDraw.Draw(board)
  d.text((22,15),f'Reconstruction ParaView 图标 · 第{g+1}/6组 · 固定编号01–33',font=_font(24),fill='#253342')
  d.text((22,53),'原稿 / 新SVG；支持实际16/24/32/48/64/128及384/1024高倍预览。',font=_font(14),fill='#637182')
  for j,row in enumerate(group):
   y=91+j*230;d.rounded_rectangle((13,y,1537,y+218),radius=8,fill='white')
   d.text((27,y+10),f"{row['id']:02} · {row['name']}",font=_font(16),fill='#253342')
   d.text((27,y+36),row['source_kind'],font=_font(12),fill='#637182')
   for x,caption,key in [(286,'原图',row['before']),(455,'ParaView原件' if row['source_kind']=='paraView-svg-root-size-only' else '新SVG','after')]:
    d.text((x,y+11),caption,font=_font(12),fill='#637182');p=Path(key) if key!='after' else Path(row['after']);im=render(p,100);board.paste(im,(x,y+48),im)
   for k,s in enumerate([16,24,32,48,128]):
    x=650+k*93;d.text((x,y+15),f'{s}px',font=_font(11),fill='#637182');im=render(Path(row['after']),s);board.paste(im,(x+(48-s)//2,y+77),im)
   d.rounded_rectangle((1160,y+42,1518,y+207),radius=6,fill='#303640');im=render(Path(row['after']),96);board.paste(im,(1291,y+64),im)
  path=out/f'comparison-{g+1:02}.png';board.save(path)
  with Image.open(path) as im:im.verify()
 for g in range((len(rows)+4)//5):
  group=rows[g*5:g*5+5];board=Image.new('RGB',(1400,105+len(group)*540),'#f5f6f8');d=ImageDraw.Draw(board)
  d.text((22,15),f'Reconstruction图标原设计 / 512px直接渲染 · {g+1}/7',font=_font(23),fill='#253342')
  for j,row in enumerate(group):
   y=75+j*540;d.text((25,y+5),f"{row['id']:02} · {row['name']}\n{row['action']}",font=_font(15),fill='#253342')
   for x,title,path in [(370,'原图',Path(row['before'])),(850,'最终SVG',Path(row['after']))]:
    d.text((x,y),title,font=_font(14),fill='#637182');im=render(path,480);board.paste(im,(x,y+42),im)
  board.save(out/f'zoom-{g+1:02}.png')
 overview=Image.new('RGB',(1420,1840),'#f5f6f8');d=ImageDraw.Draw(overview)
 d.text((24,16),'Reconstruction · 33项旧稿 / ParaView风格128px SVG总览',font=_font(24),fill='#253342')
 for i,row in enumerate(rows):
  x=12+i%5*282;y=62+i//5*254;d.rounded_rectangle((x,y,x+270,y+238),radius=7,fill='white')
  d.text((x+9,y+8),f"{row['id']:02} · {row['name']}",font=_font(12),fill='#253342')
  old=render(Path(row['before']),78);new=render(Path(row['after']),78);overview.paste(old,(x+30,y+51),old);overview.paste(new,(x+156,y+51),new)
  d.text((x+48,y+142),'原',font=_font(12),fill='#637182');d.text((x+177,y+142),'新',font=_font(12),fill='#637182')
 overview.save(out/'overview.png')
 result=[]
 for row in rows:
  p=Path(row['after']);entry={'id':row['id'],'name':row['name'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sizes':{}}
  for s in [16,24,32,48,64,128,384,1024]:entry['sizes'][str(s)]=list(render(p,s).size)
  result.append(entry)
 (out/'render-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Reconstruction media icons · ParaView风格 / 128px预览','',
 '33项功能SVG按RECON-01至RECON-33编号。图标文件名和 `:/media/<filename>.svg`运行时URL保持不变。App与libs/Reconstruction的两份QRC都指向同一app/reconstruction/media文件。','','## 总览','', '[33项总览](<review/overview.png>)','','## 编号分组','','| 固定编号 | 原图 / 最终SVG / 实际尺寸预览 | 高倍直接渲染 |','|---|---|---|']
 groups=(len(rows)+5)//6
 for g in range(groups):
  lo=g*6+1;hi=min((g+1)*6,len(rows));lines.append(f'| RECON-{lo:02}–{hi:02} | [第{g+1}组](<review/comparison-{g+1:02}.png>) | [第{g+1}组512px](<review/zoom-{g+1:02}.png>) |')
 lines+=['','## 逐项来源、动作和代码证据','','| ID | 文件 | 功能 | 源码证据 | 图形处理 |','|---|---|---|---|---|']
 for row in rows:
  src_kind='ParaView原始SVG，仅改root默认尺寸' if row['source_kind']=='paraView-svg-root-size-only' else '按功能重绘纯向量'
  lines.append(f"| RECON-{row['id']:02} | `{row['name']}` | {row['action']} | `{row['code_ui_evidence'][0]}` | {src_kind} |")
 lines+=['','## 源图文字与格式事实','','- 源文件33项：21项以SVG路径为主，12项的SVG内嵌PNG（automatic-reconstruction为1024×1024，其余11项为128×128嵌图）。全部无live `<text>`节点，但路径或嵌图可能含文字，不能由XML推断没有文字。','- 系统Tesseract辅助OCR没有可靠识别到英文词；`reconstruction-start`的“>”置信85与播放三角吻合，`reconstruction-step`“=”置信72仅视为图形线索，reset的“A”置信93视为形状误识别。TXT/LOG等原文字均未确认，台账不伪称逐字保留。','- 四个操作图标直接复用ParaView `pqOpen/pqSave/pqSaveAs/pqCaptureScreenshot`原SVG。其他29项按各自动作重绘；Windows三个ICO另作应用身份资源，保持原字节，不能将它们等同于工具栏品牌或随意替换。','- media/LICENSE中的Fugue署名保留；ParaView版权以PARAVIEW_NOTICE.txt注明。独立程序ICO的COLMAP品牌等同性未证实。','','## 验证与边界','','- 34项重建/图标回归通过。','- App QRC、library UI QRC分别独立及组合Qt加载33个资源：每包默认`QSvgRenderer`/`QImage`尺寸128×128，8档（16/24/32/48/64/128/384/1024）普通/禁用QIcon均加载并绘制。','- 两套QRC均编译；33原始URL闭环。`reconstruction_icon.rc`继续引用原reconstruction_128.ico。','- 所有对比图、PNG和多尺寸渲染均已解码验证。当前模型无法直接查看图像，因此没有声称人工视觉验收通过；源图文字细节仍有未识别项。','','## 机器记录','','- [原图/新图SHA256、来源和代码证据](<migration-ledger.json>)','- [逐项多尺寸渲染清单](<review/render-manifest.json>)','- [全体应用图像保护哈希](<protected-other-images.json>)','- [Fugue许可证与3个ICO保护哈希](<protected-media-sha256.json>)','- [ParaView原件快照](<paraview-originals>) · [旧媒体原稿](<originals>)','','## 重建','','```bash','python3 -m scripts.icon_hd.install_reconstruction_icons','python3 -m scripts.icon_hd.review_reconstruction_media','python3 -m scripts.icon_hd.verify_reconstruction_media','python3 -m unittest scripts.icon_hd.tests.test_reconstruction_media -q','```','','请用RECON-01至RECON-33编号反馈；ICO身份图标不属于本次工具栏SVG审阅范围。']
 (REVIEW/'README.md').write_text('\n'.join(lines)+'\n')
 print(f'PASS: {len(rows)} IDs, {groups} groups, 7 zoom boards, 264 multi-size renders; source/action report written')
if __name__=='__main__':main()
