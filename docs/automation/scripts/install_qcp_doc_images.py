#!/usr/bin/env python3
"""Install referenced, newly authored QCustomPlot examples without changing HTML."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

ASSETS = Path(__file__).resolve().parents[1] / 'assets/qcustomplot-original'


class Images(HTMLParser):

    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.sources.extend(
                value for key, value in attrs if key == 'src' and value)


def install(api_root, assets=ASSETS):
    api_root = Path(api_root).resolve()
    assets = Path(assets)
    if not api_root.is_dir():
        raise FileNotFoundError(f'C++ API directory is missing: {api_root}')
    references = []
    for page in api_root.rglob('*.html'):
        parser = Images()
        parser.feed(page.read_text(encoding='utf-8'))
        for raw in parser.sources:
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue
            path = Path(unquote(url.path))
            if re.fullmatch(
                    r'(QCP[^/]*|AxisNamesOverview|AxisRectSpacingOverview|adaptive-sampling-(line|scatter))\.png',
                    path.name):
                references.append((page, raw, path))
    if not references:
        return 0
    manifest = json.loads(
        (assets / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['assets']
    names = {row['name']: row for row in rows}
    if len(names) != len(rows) or any(
            Path(name).name != name or not name.endswith('.png')
            for name in names):
        raise ValueError('Invalid original QCustomPlot illustration manifest')
    pending = {}
    for page, raw, path in references:
        candidate = page.parent / path
        target = candidate.resolve()
        if candidate.is_symlink() or path.is_absolute(
        ) or not target.is_relative_to(api_root):
            raise ValueError(
                f'Illustration URL escapes API directory: {page}: {raw}')
        if target.exists():
            if not target.is_file():
                raise ValueError(f'Invalid existing illustration: {target}')
            continue
        if path.name not in names:
            raise ValueError(
                f'No reviewed original illustration for {path.name}')
        pending[target] = path.name
    # Validate all inputs before writing any output; no partial repair on a
    # corrupt source asset. The build's image validator also decodes formulas.
    for name in set(pending.values()):
        source = assets / name
        data = source.read_bytes()
        if (hashlib.sha256(data).hexdigest() != names[name]['sha256'] or
                not data.startswith(b'\x89PNG\r\n\x1a\n')):
            raise ValueError(
                f'Original illustration checksum/format mismatch: {name}')
    if pending:
        notices = {}
        for notice in ['README.md', 'LICENSE-NOTICE.md', 'LICENSE']:
            if not (assets / notice).is_file():
                raise FileNotFoundError(
                    f'Missing illustration attribution: {notice}')
            target = api_root / ('QCustomPlot-original-' + notice)
            if (target.is_symlink() or
                    not target.resolve().is_relative_to(api_root) or
                (target.exists() and not target.is_file())):
                raise ValueError(
                    f'Invalid illustration attribution target: {target}')
            if not target.exists():
                notices[target] = assets / notice
        for target, name in pending.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets / name, target)
        for target, source in notices.items():
            shutil.copyfile(source, target)
    print(
        f'Installed {len(pending)} referenced original QCustomPlot example images'
    )
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-root', type=Path, required=True)
    args = parser.parse_args()
    install(args.api_root)


if __name__ == '__main__':
    main()
