"""Read-only icon inventory: style evidence, QRC ownership and runtime references.

QRC registration alone is not a consumer. Dynamic resource expressions are listed
separately and prevent a claim that static inspection is a full runtime trace.
"""
from __future__ import annotations
import hashlib
import html
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path.home() / 'ACloudViewer-icon-reviews/final-audit'
SOURCE_EXTS = {'.cpp', '.h', '.hpp', '.ui', '.qss', '.css', '.qrc', '.cmake',
               '.txt', '.json', '.xml', '.py', '.rc', '.desktop'}
SKIP = {'.git', '3rdparty', 'node_modules', 'build', 'build_app', 'build_headless_docs', '_build'}
RESOURCE = re.compile(r':/(?:Resources/images|qPCL/icons)/[^\s\"\'<>;)]+')


def source_files():
    data = subprocess.check_output(['git', 'ls-files', '--cached', '--others',
                                    '--exclude-standard', '-z'], cwd=ROOT)
    result = []
    for name in sorted(set(data.decode().split('\0'))):
        path = ROOT / name
        if not name or not path.is_file() or path.suffix not in SOURCE_EXTS:
            continue
        if set(Path(name).parts) & SKIP:
            continue
        result.append((name, path, path.read_text(errors='replace')))
    return result


def runtime_text(name, path, text):
    if path.suffix == '.qrc' or name.startswith(('scripts/', 'docs/', '.agents/')):
        return None
    if path.suffix in {'.cpp', '.h', '.hpp'}:
        # Preserve string literals (including http://) while removing comments.
        pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/'
        text = re.sub(pattern, lambda m: '' if m.group().startswith(('//', '/*')) else m.group(), text)
        text = re.sub(r'"\s*"', '', text)
    elif path.suffix in {'.ui', '.xml'}:
        text = html.unescape(re.sub(r'<!--[\s\S]*?-->', '', text))
    return text


def registration_map(sources):
    by_path, by_url, missing, malformed = {}, {}, [], []
    for name, path, _ in sources:
        if path.suffix != '.qrc':
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            malformed.append({'file': name, 'error': str(error)})
            continue
        for group in root.findall('qresource'):
            prefix = group.attrib.get('prefix', '/').strip('/')
            for item in group.findall('file'):
                relative = (item.text or '').strip()
                if not relative:
                    continue
                url = ':/' + '/'.join(x for x in (prefix, item.attrib.get('alias', relative).lstrip('/')) if x)
                disk = (path.parent / relative).resolve()
                row = {'qrc': name, 'url': url, 'file': str(disk.relative_to(ROOT)) if disk.is_relative_to(ROOT) else str(disk)}
                by_path.setdefault(disk, []).append(row)
                by_url.setdefault(url, []).append(row)
                if not disk.is_file():
                    missing.append(row)
    return by_path, by_url, missing, malformed


def audit(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    by_path, by_url, missing, malformed = registration_map(sources)
    runtime = [(name, runtime_text(name, path, text)) for name, path, text in sources]
    runtime = [(name, text) for name, text in runtime if text is not None]
    consumers, dynamic, unresolved = {}, [], []
    for name, text in runtime:
        for url in set(RESOURCE.findall(text)):
            if any(mark in url for mark in ('%1', '%2', '%s', '%d', '*', '{')) or url.endswith('/'):
                dynamic.append({'source': name, 'expression': url})
                continue
            consumers.setdefault(url, []).append(name)
            if url not in by_url:
                unresolved.append({'source': name, 'url': url})
    roots = [ROOT / 'app/Resources/images', ROOT / 'plugins/core/Standard/qPCL/resources/icons']
    assets = []
    for base in roots:
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.svg', '.png', '.jpg', '.jpeg', '.gif', '.ico'}:
                continue
            relative = str(path.relative_to(ROOT))
            registered = by_path.get(path.resolve(), [])
            urls = sorted({r['url'] for r in registered})
            uses = sorted({name for url in urls for name in consumers.get(url, [])})
            flags = []
            text = path.read_text(errors='replace') if path.suffix.lower() == '.svg' else ''
            if not text:
                flags.append('raster')
            if '<image' in text or 'base64' in text:
                flags.append('embedded_raster')
            if re.search(r'<(?:\w+:)?text\b', text):
                flags.append('live_text')
            if 'Commercial HD v10' in text or 'Lossless pixel' in text:
                flags.append('legacy_pixel_transcription')
            if '/icon/' in relative or path.name in {'qt.png', 'donate.png'}:
                style = 'brand_or_platform_artwork_not_restyled'
            elif relative.startswith('plugins/core/Standard/qPCL/resources/icons/'):
                style = 'qPCL_functional_vector' if not flags else 'qPCL_needs_review'
            elif 'legacy_pixel_transcription' in flags:
                style = 'legacy_needs_functional_redraw'
            elif path.name.startswith('pq'):
                style = 'ParaView_reference_or_authored_glyph'
            elif '/tree/' in relative or '/QTreeView/' in relative:
                style = 'pixel_exact_Qt_tree_decoration'
            elif '#6f6f59' in text.lower():
                style = 'ParaView_palette_authored_vector'
            else:
                style = 'needs_visual_style_review'
            assets.append({'file': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'urls': urls, 'qrc': registered, 'consumers': uses,
                           'style': style, 'flags': flags,
                           'status': 'used' if uses else 'registered_only' if registered else 'source_only'})
    directories = {str(d.relative_to(ROOT)): sum(p.is_file() for p in d.iterdir())
                   for d in (ROOT/'app/Resources/images/svg').iterdir() if d.is_dir()}
    counts = {}
    for row in assets:
        counts[row['style']] = counts.get(row['style'], 0) + 1
    result = {'asset_count': len(assets), 'assets': assets, 'style_counts': counts,
              'status_counts': {s: sum(a['status'] == s for a in assets) for s in ('used', 'registered_only', 'source_only')},
              'svg_directory_file_counts': directories,
              'single_icon_directories': [n for n, count in directories.items() if count == 1],
              'missing_qrc_files': missing, 'malformed_qrc': malformed,
              'unresolved_runtime_urls': sorted(unresolved, key=lambda x: (x['url'], x['source'])),
              'dynamic_resource_expressions': dynamic,
              'limitations': 'Static source/QRC evidence is not visual approval or complete dynamic execution coverage.'}
    (out/'complete-asset-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('Assets:', len(assets), 'Style:', counts, 'Use:', result['status_counts'])
    print('Missing QRC files:', len(missing), 'Unresolved runtime URLs:', len(unresolved), 'Dynamic:', len(dynamic))
    print('QRC-only SVG candidates:', [a['file'] for a in assets if a['status']=='registered_only' and a['file'].endswith('.svg')])
    for row in unresolved:
        print('UNRESOLVED:', row)
    return result


if __name__ == '__main__':
    audit()
