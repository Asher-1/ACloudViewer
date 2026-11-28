"""Install, review and verify AICore plugin-owned named ParaView SVGs.

AnyLabeling is excluded before discovery, file edits and resource registration.
The existing filenames, metadata URLs and QRC namespaces remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

if __package__:
    from .logo.aicoreicons import NAMES, NAME_LINES, SOURCE_CAPTIONS, build
    from .logo.icon import _font, _png
else:
    from logo.aicoreicons import NAMES, NAME_LINES, SOURCE_CAPTIONS, build
    from logo.icon import _font, _png

ROOT = Path(__file__).resolve().parents[2]
STANDARD = ROOT/'plugins/core/Standard'
EXCLUDED = STANDARD/'qAnyLabeling'
DEFAULT_OUT = Path.home()/'ACloudViewer-icon-reviews/aicore-plugin-icons'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def guard_anylabeling(out):
    current = {str(p.relative_to(ROOT)): digest(p) for p in sorted(EXCLUDED.rglob('*')) if p.is_file()}
    path = out/'anylabeling-protected.json'
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise RuntimeError('AnyLabeling tree differs from its pre-task snapshot; no overwrite is allowed')
    else:
        dump(path, current)
    return len(current)


def catalog():
    discovered = set()
    for folder in STANDARD.iterdir():
        if not folder.is_dir() or folder == EXCLUDED:
            continue
        metadata = folder/'info.json'
        if not metadata.is_file():
            continue
        info = json.loads(metadata.read_text(encoding='utf-8-sig'))
        cmake = (folder/'CMakeLists.txt').read_text(errors='replace')
        if info.get('group') == 'aicore' or 'AICore' in cmake:
            discovered.add(folder.name)
    if discovered != set(NAMES):
        raise RuntimeError(f'AICore plugin scope changed; refresh mapping: {discovered ^ set(NAMES)}')
    rows = []
    for index, (plugin, wordmark) in enumerate(NAMES.items(), 1):
        folder = STANDARD/plugin
        metadata = folder/'info.json'
        info = json.loads(metadata.read_text(encoding='utf-8-sig'))
        icon = folder/'images'/f'{plugin}.svg'
        url = info['icon']
        qrc = folder/f'{plugin}.qrc'
        found = []
        for group in ET.parse(qrc).getroot().findall('qresource'):
            prefix = group.attrib.get('prefix', '/').strip('/')
            for node in group.findall('file'):
                relative = (node.text or '').strip()
                source = (qrc.parent/relative).resolve()
                effective = ':/'+'/'.join(x for x in (prefix, node.attrib.get('alias', relative)) if x)
                if source == icon.resolve():
                    found.append(effective)
        if found != [url]:
            raise RuntimeError(f'{plugin}: icon URL not uniquely owned by its QRC: {found} != {url}')
        consumers = []
        for path in folder.rglob('*'):
            if not path.is_file() or path.suffix not in {'.cpp', '.h', '.hpp', '.ui'}:
                continue
            for line, text in enumerate(path.read_text(errors='replace').splitlines(), 1):
                if url in text or ('getIcon()' in text and path.suffix == '.cpp'):
                    consumers.append({'file': str(path.relative_to(ROOT)), 'line': line, 'text': text.strip()})
        rows.append({'id': index, 'plugin': plugin, 'wordmark': wordmark, 'display_name': info['name'],
                     'icon': str(icon.relative_to(ROOT)), 'url': url, 'qrc': str(qrc.relative_to(ROOT)),
                     'metadata': str(metadata.relative_to(ROOT)), 'consumers': consumers,
                     'name_lines': list(NAME_LINES[wordmark]),
                     'source_captions': list(SOURCE_CAPTIONS.get(wordmark, ()))})
    return rows


def protect_other_art(out):
    targets = {(ROOT/row['icon']).resolve() for row in catalog()}
    paths = [p for p in (ROOT/'app/Resources/images').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'app/reconstruction/media').glob('*') if p.is_file()]
    paths += [p for p in STANDARD.rglob('*') if p.is_file() and p.suffix.lower() in {'.svg', '.png', '.ico'}]
    current = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.resolve() not in targets}
    snapshot = out/'out-of-scope-art.json'
    if snapshot.exists():
        original = json.loads(snapshot.read_text())
        changed = [n for n, value in original.items() if not (ROOT/n).is_file() or digest(ROOT/n) != value]
        if changed:
            raise RuntimeError(f'Out-of-scope artwork changed: {changed}')
    else:
        dump(snapshot, current)
    return len(current)


def install(out):
    out.mkdir(parents=True, exist_ok=True)
    excluded_count = guard_anylabeling(out)
    protected_count = protect_other_art(out)
    rows = catalog()
    receipt = out/'icons-ledger.json'
    previous = {r['plugin']: r for r in json.loads(receipt.read_text())['icons']} if receipt.exists() else {}
    plans = []
    for row in rows:
        target = ROOT/row['icon']
        original = out/'originals'/row['plugin']/target.name
        original.parent.mkdir(parents=True, exist_ok=True)
        data = build(row['wordmark']).encode()
        node = ET.fromstring(data)
        if node.attrib.get('width') != '128' or node.attrib.get('height') != '128':
            raise RuntimeError(f'{target}: wrong preview size')
        if {'image', 'text', 'script', 'filter'} & {n.tag.rsplit('}', 1)[-1] for n in node.iter()}:
            raise RuntimeError(f'{target}: output must be fontless, filter-free vector artwork')
        if original.exists():
            known = {digest(original), hashlib.sha256(data).hexdigest(), previous.get(row['plugin'], {}).get('after_sha256')}
            if digest(target) not in known:
                raise RuntimeError(f'Concurrent unreviewed plugin artwork: {target}')
        else:
            original.write_bytes(target.read_bytes())
        original_root = ET.parse(original).getroot()
        original_text = [''.join(n.itertext()).strip() for n in original_root.iter() if n.tag.rsplit('}', 1)[-1] == 'text']
        references = {row['metadata']: digest(ROOT/row['metadata']), row['qrc']: digest(ROOT/row['qrc'])}
        row.update(original=str(original), original_text=original_text,
                   original_outlined_text=['InstantMesh'] if row['plugin']=='qInstantMesh' else [],
                   before_sha256=digest(original), after_sha256=hashlib.sha256(data).hexdigest(),
                   reference_sha256=references, runtime_url_unchanged=True,
                   primary_name_policy='Exact plugin brand without directory prefix; deliberate line wrapping, original captions retained as paths',
                   acceptance='pending')
        plans.append((row, target, data))
    for _, target, data in plans:
        if target.read_bytes() != data:
            target.write_bytes(data)
    guard_anylabeling(out)
    dump(receipt, {'icons': [row for row, _, _ in plans], 'count': len(plans),
                   'AnyLabeling_untouched_files': excluded_count, 'other_art_protected_count': protected_count,
                   'scope': 'Primary plugin action SVGs only; unregistered PNG/video art and demo results are untouched',
                   'intrinsic_size': [128, 128], 'font_runtime_dependency': False})
    print(f'PASS: {len(plans)} plugin-local named SVGs; {excluded_count} AnyLabeling files unchanged; {protected_count} other artwork hashes preserved')


PROBE = r'''#include <QApplication>
#include <QResource>
#include <QFile>
#include <QImage>
#include <QIcon>
#include <QSvgRenderer>
#include <QPainter>
#include <QTextStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <iostream>
int main(int argc,char**argv){QApplication app(argc,argv);if(argc<3)return 1;
for(int i=1;i<argc-1;i++)if(!QResource::registerResource(argv[i]))return 2;
QFile f(argv[argc-1]);if(!f.open(QIODevice::ReadOnly))return 3;QTextStream in(&f);int n=0;
while(!in.atEnd()){auto a=in.readLine().split('\t');if(a.size()!=4)return 4;
QFile r(a[0]),s(a[1]),m(a[2]);if(!r.open(QIODevice::ReadOnly)||!s.open(QIODevice::ReadOnly)||r.readAll()!=s.readAll()||!m.open(QIODevice::ReadOnly))return 5;
auto doc=QJsonDocument::fromJson(m.readAll());if(doc.object().value("icon").toString()!=a[0])return 6;
QSvgRenderer svg(a[0]);QImage image(a[0]);QIcon icon(a[0]);if(!svg.isValid()||svg.defaultSize()!=QSize(128,128)||image.size()!=QSize(128,128))return 7;
for(int size:{16,24,32,48,64,128,384,1024}){if(icon.pixmap(size,size).isNull()||icon.pixmap(size,size,QIcon::Disabled).isNull())return 8;
QImage x(size,size,QImage::Format_ARGB32_Premultiplied);x.fill(Qt::transparent);QPainter p(&x);svg.render(&p);p.end();bool ink=false;for(int y=0;y<size&&!ink;y++)for(int z=0;z<size;z++)if(qAlpha(x.pixel(z,y))>0){ink=true;break;}if(!ink)return 9;}
++n;}std::cout<<"PASS "<<n<<" plugin icon URLs: metadata, canonical bytes, default128, 8 sizes, normal/disabled\n";return 0;}
'''


def verify(out):
    ledger = json.loads((out/'icons-ledger.json').read_text())
    rows = ledger['icons']
    guard_anylabeling(out)
    protect_other_art(out)
    for row in rows:
        path = ROOT/row['icon']
        if digest(path) != row['after_sha256'] or path.read_text() != build(row['wordmark']):
            raise RuntimeError(f'Installed icon and generator diverged: {path}')
        for name, expected in row['reference_sha256'].items():
            if digest(ROOT/name) != expected:
                raise RuntimeError(f'Metadata/QRC changed unexpectedly: {name}')
    flags = subprocess.check_output(['pkg-config', '--cflags', '--libs', 'Qt5Widgets', 'Qt5Svg'], text=True).split()
    results = []
    with tempfile.TemporaryDirectory(prefix='aicore-icon-check-') as temp:
        work = Path(temp)
        source = work/'probe.cpp'
        source.write_text(PROBE)
        probe = work/'probe'
        subprocess.run(['g++', '-std=c++17', '-fPIC', str(source), '-o', str(probe), *flags], check=True)
        bundles, all_rows = [], []
        env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
        for row in rows:
            qrc = ROOT/row['qrc']
            root = ET.parse(qrc).getroot()
            metadata_url = None
            for group in root.findall('qresource'):
                for item in group.findall('file'):
                    if (qrc.parent/(item.text or '')).resolve() == (ROOT/row['metadata']).resolve():
                        metadata_url = ':/'+'/'.join(x for x in (group.attrib.get('prefix', '/').strip('/'), item.attrib.get('alias', item.text)) if x)
            if not metadata_url:
                raise RuntimeError(f'Metadata not in plugin QRC: {qrc}')
            line = '\t'.join([row['url'], str(ROOT/row['icon']), metadata_url, row['plugin']])
            bundle = work/(row['plugin']+'.rcc')
            subprocess.run(['rcc', '--binary', str(qrc), '-o', str(bundle)], check=True)
            listing = work/(row['plugin']+'.tsv')
            listing.write_text(line+'\n')
            run = subprocess.run([str(probe), str(bundle), str(listing)], env=env, capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(f'{row["plugin"]} Qt failure {run.returncode}: {run.stdout} {run.stderr}')
            results.append(row['plugin']+': '+run.stdout.strip())
            print(results[-1])
            bundles.append(bundle)
            all_rows.append(line)
        listing = work/'combined.tsv'
        listing.write_text('\n'.join(all_rows)+'\n')
        run = subprocess.run([str(probe), *map(str, bundles), str(listing)], env=env, capture_output=True, text=True, check=True)
        print('combined: '+run.stdout.strip())
        results.append('combined: '+run.stdout.strip())
    (out/'qt-verification.txt').write_text('\n'.join(results)+'\n')
    dump(out/'verification.json', {'isolated_plugin_bundles': len(rows), 'combined_plugin_icons': len(rows),
         'requested_sizes': [16, 24, 32, 48, 64, 128, 384, 1024],
         'isolated_size_checks': len(rows)*8, 'combined_size_checks': len(rows)*8,
         'checks_include': ['normal/disabled QIcon', 'QSvgRenderer', 'QImage default128', 'canonical bytes', 'metadata URL'],
         'AnyLabeling_tree_unchanged': True, 'visual_acceptance': 'pending user review'})


def review(out):
    rows = json.loads((out/'icons-ledger.json').read_text())['icons']
    render_dir = out/'renders'
    render_dir.mkdir(exist_ok=True)
    label_checks = []
    for row in rows:
        path = ROOT/row['icon']
        if digest(path) != row['after_sha256']:
            raise RuntimeError(f'Comparison source differs from installed file: {path}')
        root = ET.parse(path).getroot()
        name_group = next(n for n in root.iter() if n.attrib.get('data-plugin-name'))
        # Measure actual name-glyph pixels separately from other diagram detail.
        name_svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'+ET.tostring(name_group, encoding='unicode')+'</svg>'
        im = _png(name_svg, 128)
        bounds = im.getchannel('A').getbbox()
        if not bounds:
            raise RuntimeError(f'Empty name lettering: {row["plugin"]}')
        name_paths = [n for n in name_group.iter() if n.tag.rsplit('}', 1)[-1] == 'path']
        expected = list(NAME_LINES[row['wordmark']])
        if [n.attrib.get('data-label') for n in name_paths] != expected:
            raise RuntimeError(f'Name lettering mismatch: {row["plugin"]}')
        line_bounds = []
        for glyph in name_paths:
            fragment = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'+ET.tostring(glyph, encoding='unicode')+'</svg>'
            box = _png(fragment,128).getchannel('A').getbbox()
            if not box or box[3]-box[1] < 13:
                raise RuntimeError(f'Plugin name is too small at 128px: {row["plugin"]}: {box}')
            line_bounds.append(list(box))
        labels=[n.attrib['data-label'] for n in root.iter() if 'data-label' in n.attrib]
        normalize=lambda text: ''.join(text.split()).casefold()
        shown=normalize(''.join(labels))
        original_words=row['original_text']+row.get('original_outlined_text',[])
        missing=[text for text in original_words if normalize(text) not in shown]
        if missing:
            raise RuntimeError(f'Original plugin lettering omitted: {row["plugin"]}: {missing}')
        label_checks.append({'id': row['id'], 'plugin': row['plugin'], 'wordmark': row['wordmark'],
                             'lines': expected, 'glyph_bbox_at128': list(bounds), 'name_line_boxes_at128':line_bounds,
                             'original_lettering_retained': original_words, 'all_visible_path_labels':labels})
        for size in [16, 24, 32, 48, 64, 128, 384, 1024]:
            image = _png(path.read_text(), size)
            if not image.getbbox():
                raise RuntimeError(f'Empty SVG render: {path}')
            image.save(render_dir/f'{row["id"]:02}-{row["plugin"]}-{size}.png')
    dump(out/'label-checks.json', label_checks)
    groups = (len(rows)+4)//5
    for group_index in range(groups):
        chunk = rows[group_index*5:group_index*5+5]
        canvas = Image.new('RGB', (1600, 100+len(chunk)*235), '#F5F6F8')
        draw = ImageDraw.Draw(canvas)
        draw.text((24, 15), f'AICore plugin icons {group_index+1}/{groups} — stable AI-01..{len(rows):02d}', font=_font(24), fill='#253342')
        for j, row in enumerate(chunk):
            y = 75+j*235
            draw.rounded_rectangle((14,y,1586,y+220),radius=8,fill='white')
            draw.text((28,y+12),f'AI-{row["id"]:02}  {row["plugin"]}',font=_font(21),fill='#253342')
            draw.text((28,y+47),row['wordmark'],font=_font(16),fill='#637182')
            for x, title, path in [(305,'Original',Path(row['original'])),(500,'Named ParaView SVG',ROOT/row['icon'])]:
                draw.text((x,y+12),title,font=_font(13),fill='#637182')
                image = _png(path.read_text(),128)
                canvas.paste(image,(x,y+65),image)
            for k, size in enumerate([16,24,32,48,64]):
                x = 735+k*90
                draw.text((x,y+15),f'{size}px',font=_font(12),fill='#637182')
                image = _png((ROOT/row['icon']).read_text(),size)
                canvas.paste(image,(x+(64-size)//2,y+95),image)
            draw.rounded_rectangle((1230,y+43,1565,y+205),radius=5,fill='#303640')
            image = _png((ROOT/row['icon']).read_text(),128)
            canvas.paste(image,(1333,y+61),image)
        canvas.save(out/f'comparison-{group_index+1:02}.png')
    overview = Image.new('RGB',(1450,70+((len(rows)+3)//4)*255),'#F5F6F8')
    draw = ImageDraw.Draw(overview)
    draw.text((22,14),f'AICore — {len(rows)} named ParaView plugin icons; AnyLabeling excluded',font=_font(23),fill='#253342')
    for i,row in enumerate(rows):
        x=12+i%4*360;y=65+i//4*255
        draw.rounded_rectangle((x,y,x+347,y+241),radius=7,fill='white')
        draw.text((x+9,y+10),f'AI-{row["id"]:02}  {row["plugin"]}',font=_font(17),fill='#253342')
        for xx,path in [(x+22,Path(row['original'])),(x+191,ROOT/row['icon'])]:
            image=_png(path.read_text(),128);overview.paste(image,(xx,y+65),image)
        draw.text((x+60,y+209),'Original',font=_font(12),fill='#637182')
        draw.text((x+235,y+209),'New',font=_font(12),fill='#637182')
    overview.save(out/'overview.png')
    for group_index in range((len(rows)+2)//3):
        chunk=rows[group_index*3:group_index*3+3]
        canvas=Image.new('RGB',(1380,90+len(chunk)*440),'#F5F6F8');draw=ImageDraw.Draw(canvas)
        draw.text((22,15),f'AICore original / final — direct 384px SVG render {group_index+1}/{(len(rows)+2)//3}',font=_font(23),fill='#253342')
        for j,row in enumerate(chunk):
            y=70+j*440;draw.text((25,y+10),f'AI-{row["id"]:02}\n{row["plugin"]}',font=_font(18),fill='#253342')
            for x,path in [(350,Path(row['original'])),(915,ROOT/row['icon'])]:
                image=_png(path.read_text(),384);canvas.paste(image,(x,y+25),image)
        canvas.save(out/f'zoom-{group_index+1:02}.png')
    for path in [out/'overview.png',*out.glob('comparison-*.png'),*out.glob('zoom-*.png')]:
        with Image.open(path) as image:
            image.verify()
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>AICore named SVGs</title>',
          '<style>body{font:16px sans-serif;background:#f5f6f8;color:#253342;margin:24px}main{display:flex;flex-wrap:wrap;gap:20px}section{background:white;padding:16px;border:1px solid #d6dde3;border-radius:8px}svg{width:128px;height:128px}.dark{background:#303640}button{padding:8px;margin:8px}</style>',
          f'<h1>AICore icons AI-01..{len(rows):02d} — live SVG preview</h1><p>AnyLabeling is not part of this set. Native 128px preview; use browser zoom for vector detail.</p><main>']
    for row in rows:
        page.append(f'<section><h2>AI-{row["id"]:02} {row["plugin"]}</h2>'+ (ROOT/row['icon']).read_text()+'<div class="dark">'+(ROOT/row['icon']).read_text()+'</div></section>')
    page.append('</main></html>')
    (out/'live-svg-preview.html').write_text('\n'.join(page)+'\n')
    print(f'PASS: {len(rows)} stable IDs; {(len(rows)+4)//5} detailed groups, {(len(rows)+2)//3} zoom boards, {len(rows)*8} direct vector renders; original/name lettering and 128px glyph height verified')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['install','review','verify'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(argv)
    {'install':install,'review':review,'verify':verify}[args.action](args.out)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
