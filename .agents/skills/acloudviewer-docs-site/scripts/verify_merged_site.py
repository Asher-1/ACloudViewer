#!/usr/bin/env python3
"""Read-only ACloudViewer merged-tree probe. See references/real-artifact-acceptance.md."""
import argparse
import json
import mimetypes
import re
import shutil
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote, unquote, urljoin, urlsplit

PREFIX = '/ACloudViewer/'
CSS = re.compile(r'''url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*))\s*\)|@import\s+(?:"([^"]*)"|'([^']*)')''', re.I)
CSS_COMMENTS = re.compile(r'/\*.*?\*/', re.S)


def css_urls(text):
    for match in CSS.finditer(CSS_COMMENTS.sub('', text)):
        yield next(v.strip() for v in match.groups() if v is not None)


def srcset_urls(text):
    # URL tokens end at whitespace; commas within data URLs are not delimiters.
    while text:
        text = text.lstrip(' ,\n\t\r\f')
        if not text:
            break
        url, _, rest = text.partition(' ')
        parts = re.split(r'\s+', text, maxsplit=1)
        url, rest = parts[0], parts[1] if len(parts) > 1 else ''
        if url.lower().startswith('data:'):
            if not url.endswith(','):
                rest = rest.partition(',')[2]
        elif ',' in url:
            yield from (v for v in url.split(',') if v)
            if not url.endswith(','):
                rest = rest.partition(',')[2]
        else:
            yield url
            rest = rest.partition(',')[2]
        text = rest


class Assets(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.base = None
        self.refs = []
        self.style = False

    def handle_starttag(self, tag, attributes):
        a = dict(attributes)
        if tag == 'base' and self.base is None and a.get('href') is not None:
            self.base = a['href']
        if tag == 'style':
            self.style = True
        if tag in {'img', 'script', 'source', 'video', 'audio', 'track', 'iframe', 'embed', 'input'}:
            self.refs.append(a.get('src'))
        if tag in {'img', 'source'}:
            self.refs.extend(srcset_urls(a.get('srcset') or ''))
        if tag == 'video':
            self.refs.append(a.get('poster'))
        if tag == 'object':
            self.refs.append(a.get('data'))
        if tag == 'link' and set((a.get('rel') or '').lower().split()) & {
            'stylesheet', 'icon', 'apple-touch-icon', 'preload', 'modulepreload', 'manifest', 'prefetch'
        }:
            self.refs.append(a.get('href'))
        if tag.split(':')[-1] in {'image', 'use', 'feimage', 'a'} and (
            tag.split(':')[-1] != 'a' or 'xlink:href' in a
        ):
            self.refs.extend(a.get(k) for k in ('href', 'xlink:href'))
        for key, value in attributes:
            if value and (key == 'style' or 'url(' in value.lower()):
                self.refs.extend(css_urls(value))

    def handle_endtag(self, tag):
        if tag == 'style':
            self.style = False

    def handle_data(self, data):
        if self.style:
            self.refs.extend(css_urls(data))


def find_path(site, path):
    path = unquote(path)
    if not path.startswith(PREFIX):
        return None
    candidate = (site / path[len(PREFIX):].lstrip('/')).resolve()
    return candidate if candidate.is_relative_to(site) else None


def audit(site, templates, origin):
    rows, counts = [], {'files': 0, 'content_bytes': 0, 'scanned_documents': 0, 'local_references': 0}
    for entry in sorted(site.rglob('*')):
        rel = entry.relative_to(site).as_posix()
        page = PREFIX + quote(rel)
        if entry.is_symlink():
            rows.append(dict(page=rel, page_url=page, reference=rel, resolved_path=str(entry.resolve()), category='symlink_requires_upload_contract_review'))
            continue
        if not entry.is_file():
            continue
        counts['files'] += 1
        counts['content_bytes'] += entry.stat().st_size
        ext = entry.suffix.lower()
        if ext not in {'.html', '.htm', '.css', '.svg'}:
            continue
        counts['scanned_documents'] += 1
        try:
            text = entry.read_text(encoding='utf-8-sig')
        except UnicodeError as exc:
            rows.append(dict(page=rel, page_url=page, reference='', resolved_path=rel, category='unreadable_document', detail=str(exc)))
            continue
        base = origin + page
        if ext == '.css':
            refs = css_urls(text)
        else:
            parser = Assets()
            parser.feed(text)
            parser.close()
            refs = parser.refs
            if ext == '.svg':
                # Include href-only links in standalone SVG; HTML <a> navigation is sampled in browser.
                refs += re.findall(r'''<a\b[^>]*?\bhref\s*=\s*["']([^"']+)''', text, re.I)
            if parser.base is not None:
                base = urljoin(base, parser.base)
        for raw in dict.fromkeys(refs):
            if not raw or not raw.strip() or raw.strip().startswith('#'):
                continue
            resolved = urlsplit(urljoin(base, raw.strip()))
            if resolved.scheme not in {'http', 'https'} or resolved.netloc != urlsplit(origin).netloc:
                continue
            counts['local_references'] += 1
            target = find_path(site, resolved.path)
            if target is not None and (target.is_file() or (target.is_dir() and (target / 'index.html').is_file())):
                continue
            category = 'outside_mount_or_tree' if target is None else 'missing_local_asset'
            if rel in templates:
                category = 'unrendered_template'
            rows.append(dict(page=rel, page_url=page, reference=raw, resolved_url=resolved.geturl(),
                             resolved_path=target.relative_to(site).as_posix() if target is not None else unquote(resolved.path),
                             category=category, detail=templates.get(rel, '')))
    return dict(counts=counts, findings=rows,
                passed=all(row['category'] == 'unrendered_template' for row in rows))


def handler(site):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            target = find_path(site, path)
            status = 200
            if target is not None and target.is_dir() and (target / 'index.html').is_file():
                if not path.endswith('/'):
                    parts = urlsplit(self.path)
                    self.send_response(301)
                    self.send_header('Location', path + '/' + ('?' + parts.query if parts.query else ''))
                    self.end_headers()
                    return
                target = target / 'index.html'
            if target is None or not target.is_file():
                status, target = 404, (site / '404.html').resolve()
            if not target.is_relative_to(site) or not target.is_file():
                self.send_error(404)
                return
            self.send_response(status)
            self.send_header('Content-Type', mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(target.stat().st_size))
            self.end_headers()
            with target.open('rb') as stream:
                try:
                    shutil.copyfileobj(stream, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *args):
            pass
    return Handler


WIDTH = '''() => ({viewport: innerWidth, document: document.documentElement.scrollWidth,
 body: document.body?.scrollWidth, elements: Array.from(document.querySelectorAll('body *')).flatMap(e => {
  const r = e.getBoundingClientRect(), s = getComputedStyle(e);
  if (!r.width || (r.left >= -1 && r.right <= innerWidth + 1)) return [];
  return [{tag:e.tagName,id:e.id,cls:e.getAttribute('class'),left:r.left,right:r.right,
    width:r.width,overflowX:s.overflowX,whiteSpace:s.whiteSpace,minWidth:s.minWidth}];
 }).slice(0, 35)})'''
IMAGE = '''async (img) => {
  img.loading = 'eager'; await img.decode();
  return {url: img.currentSrc || img.src, width: img.naturalWidth, height: img.naturalHeight};
}'''


def browser_checks(args, plan, origin, report):
    from playwright.sync_api import sync_playwright
    rows = report['checks']
    mount = origin + PREFIX
    expected = args.versions
    stable = next((v for v in expected if v != 'dev'), None)
    manifest = json.loads((args.site / 'documentation/versions.json').read_text())
    ids = [v['id'] for v in manifest['versions']]
    kinds = [v['kind'] for v in manifest['versions']]
    correct_kinds = ['dev' if v == 'dev' else 'latest' if v == stable else 'archive' for v in expected]
    rows.append(dict(name='manifest', passed=ids == expected and kinds == correct_kinds and manifest['latest'] == args.latest,
                     observed=manifest, expected=expected, expected_kinds=correct_kinds, expected_latest=args.latest))
    cases = [dict(path='', widths=[1280, 1024, 768, 390, 320]),
             dict(path='documentation/', final=f'documentation/{args.latest}/', directory=True)]
    cases += [dict(path=f'documentation/{v}/', final=f'documentation/{v}/', directory=True) for v in expected]
    cases += plan['pages']
    with sync_playwright() as p:
        with p.chromium.launch() as browser:
            report['browser'] = dict(version=browser.version, executable=p.chromium.executable_path)
            for case in cases:
                for width in case.get('widths', [1280]):
                    run_case(browser, rows, mount, origin, case, width)


def run_case(browser, rows, mount, origin, case, width):
    url = urljoin(mount, case['path'])
    row = dict(name='navigation', url=url, width=width, passed=False, local_errors=[], external_blocked=[],
               page_errors=[], images=[], svg=[], navigation=[])
    rows.append(row)
    context = browser.new_context(viewport={'width': width, 'height': 900}, service_workers='block')

    def route_request(route):
        request = route.request
        if urlsplit(request.url).netloc == urlsplit(origin).netloc:
            route.continue_()
        else:
            row['external_blocked'].append(request.url)
            route.abort()

    def response_event(r):
        if urlsplit(r.url).netloc != urlsplit(origin).netloc:
            return
        if r.request.is_navigation_request() and r.request.frame == page.main_frame:
            row['navigation'].append(dict(url=r.url, status=r.status))
        elif r.status >= 400:
            row['local_errors'].append(dict(url=r.url, status=r.status, type=r.request.resource_type))

    def request_failed(r):
        if urlsplit(r.url).netloc == urlsplit(origin).netloc:
            row['local_errors'].append(dict(url=r.url, error=r.failure, type=r.resource_type))

    try:
        context.route('**/*', route_request)
        page = context.new_page()
        page.on('response', response_event)
        page.on('requestfailed', request_failed)
        page.on('pageerror', lambda error: row['page_errors'].append(str(error)))
        if not url.startswith(mount):
            raise ValueError('plan page must be inside the mount')
        response = page.goto(url, wait_until='load', timeout=15000)
        if 'final' in case:
            final = urljoin(mount, case['final'])

            def matches(u):
                u = str(u)
                return u == final or (case.get('directory') and final.endswith('/') and u == final + 'index.html')

            page.wait_for_url(matches, wait_until='load', timeout=15000)
        if 'selector' in case:
            page.locator(case['selector']).first.wait_for(state='visible', timeout=10000)
        row['final_url'] = page.url
        row['initial_status'] = row['navigation'][0]['status'] if row['navigation'] else (response.status if response else None)
        row['metrics'] = page.evaluate(WIDTH)
        for image in case.get('images', []):
            target = urljoin(page.url, image)
            item = dict(page_url=page.url, url=target, passed=False)
            row['images'].append(item)
            try:
                imgs = page.locator('img')
                idx = imgs.evaluate_all('(es, url) => es.findIndex(e => e.src === url || e.currentSrc === url)', target)
                if idx < 0:
                    raise ValueError('expected img not present')
                img = imgs.nth(idx)
                img.scroll_into_view_if_needed(timeout=5000)
                data = img.evaluate(IMAGE, timeout=5000)
                loaded = next((r for r in reversed(row.get('image_http', [])) if r['url'] == target), None)
                # Load/decode above is browser evidence; obtain a literal status for this URL too.
                status = loaded['status'] if loaded else context.request.get(target, timeout=5000, max_redirects=0).status
                item.update(data, status=status, passed=data['url'] == target and min(data['width'], data['height']) > 0 and status == 200)
            except Exception as exc:
                item['error'] = str(exc)
        if 'svg' in case:
            refs = page.locator(case['svg']).evaluate_all('''es => es.map(e => {
                const ref = e.getAttribute('href') || e.getAttributeNS('http://www.w3.org/1999/xlink', 'href');
                return ref && new URL(ref, e.baseURI).href;
            }).filter(Boolean)''')
            for link in dict.fromkeys(refs):
                status = context.request.get(link, timeout=5000, max_redirects=0).status if link.startswith(mount) else None
                row['svg'].append(dict(url=link, status=status, passed=status == 200))
            if not row['svg']:
                row['svg'].append(dict(passed=False, error='no SVG links matched'))
        row['passed'] = (row['initial_status'] == case.get('status', 200)
                         and all(x['status'] < 400 for x in row['navigation'][1:])
                         and max(row['metrics']['document'], row['metrics']['body'] or 0) <= width + 1
                         and not row['local_errors'] and not row['page_errors']
                         and all(x['passed'] for x in row['images'] + row['svg']))
    except Exception as exc:
        row['error'] = str(exc)
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--versions', nargs='+', required=True)
    parser.add_argument('--latest', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.sha) or not re.fullmatch(r'[1-9][0-9]*', args.run_id):
        parser.error('give full source SHA and positive exact Actions run ID')
    if (any(not re.fullmatch(r'dev|v[0-9]+\.[0-9]+\.[0-9]+', v) for v in args.versions)
            or len(set(args.versions)) != len(args.versions) or args.latest not in args.versions):
        parser.error('invalid independent version inventory')
    args.site, args.out = args.site.resolve(), args.out.resolve()
    if not args.site.is_dir() or args.out.is_relative_to(args.site) or args.out.exists():
        parser.error('site must exist; use a new output directory outside the site')
    plan = json.loads(args.plan.read_text())
    templates = plan.get('templates', {})
    if not isinstance(plan.get('pages'), list) or not isinstance(templates, dict) or not all(
        isinstance(why, str) and why.strip() and key.endswith(('.html', '.htm'))
        and (args.site / key).resolve().is_relative_to(args.site) and (args.site / key).is_file()
        for key, why in templates.items()
    ):
        parser.error('provide pages and evidence for any exact template exemptions')
    args.out.mkdir(parents=True)
    report = dict(scope='local exact-artifact replay; not production', sha=args.sha, run_id=args.run_id,
                  site=str(args.site), plan=plan, checks=[])
    with ThreadingHTTPServer(('127.0.0.1', 0), handler(args.site)) as server:
        origin = f'http://127.0.0.1:{server.server_port}'
        report['mount'] = origin + PREFIX
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            try:
                browser_checks(args, plan, origin, report)
            except Exception as exc:
                report['checks'].append(dict(name='browser/manifest probe error', passed=False, error=str(exc)))
            try:
                report['audit'] = audit(args.site, templates, origin)
            except Exception as exc:
                report['audit'] = dict(passed=False, findings=[], error=str(exc))
        finally:
            server.shutdown()
            worker.join()
    report['passed'] = report['audit']['passed'] and all(row['passed'] for row in report['checks'])
    (args.out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (args.out / 'report.md').open('w', encoding='utf-8') as md:
        print(f'# Local exact-artifact acceptance\n\nSHA `{args.sha}`, run `{args.run_id}`, tree `{args.site}`. Not production.\n', file=md)
        print('Probe:', 'PASS' if report['passed'] else 'FAIL', '(see JSON for raw evidence).\n', file=md)
        print('## Browser\n', file=md)
        for row in report['checks']:
            print('-', 'PASS' if row['passed'] else 'FAIL', row['name'], row.get('url', ''), row.get('width', ''), row.get('metrics', {}).get('document', ''), row.get('error', ''), file=md)
        print('\n## Static audit\n', file=md)
        print(report['audit'].get('counts', report['audit'].get('error', '')), file=md)
        for row in report['audit']['findings']:
            print(f"- {row['category']}: `{row['page']}` → `{row['resolved_path']}` (reference `{row['reference']}`) {row.get('detail', '')}", file=md)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
