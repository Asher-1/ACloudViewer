#!/usr/bin/env python3
"""Icon HD toolchain: lossless v8 transcription, commercial v10 AI redesign,
and the comparison gallery — one self-contained folder.

Usage (from anywhere; paths resolve relative to this file):
    python scripts/icon_hd/icon_hd.py lossless            # dry run + verify gate
    python scripts/icon_hd/icon_hd.py lossless --write    # publish v8 to repo
    python scripts/icon_hd/icon_hd.py redesign            # candidates + report only
    python scripts/icon_hd/icon_hd.py redesign --publish  # publish v10 to repo
    python scripts/icon_hd/icon_hd.py gallery             # data/gallery.html
    python scripts/icon_hd/icon_hd.py compare4k           # data/compare4k/*.png (3840x2160)

Repo writes happen ONLY with --write / --publish. Candidates, reports and the
gallery live in data/ (gitignored). The orig/ PNG fixtures are the single
source of truth (the repo PNGs were replaced by SVGs). Dependencies:
PySide6, numpy, opencv-python, potrace, scikit-image, scipy.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'

# qMeshIO icon.svg is native Inkscape vector - never touch
NATIVE_SVG = {'plugins/core/IO/qMeshIO/images/icon.svg'}


def load_icons():
    return sorted({l.strip() for l in (HERE / 'icons.txt').read_text().splitlines()
                   if l.strip().endswith('.png')})


def orig_path(p):
    # fixture name = repo path with '/' -> '__' (keeps its own .png suffix)
    return HERE / 'orig' / p.replace('/', '__')


_RELOCATE = None


def repo_svg(p):
    # qa['relocate'] maps png repo paths whose published SVG was categorized
    # into images/svg/<class>/ (see the svg-relocate pass)
    global _RELOCATE
    if _RELOCATE is None:
        _RELOCATE = json.loads((HERE / 'qa.json').read_text()).get('relocate', {})
    return ROOT / (_RELOCATE.get(p) or p[:-4] + '.svg')


def cand_svg(p):
    return DATA / 'candidates' / (p.replace('/', '__') + '.svg')


def cmd_lossless(args):
    import lossless
    from PySide6 import QtGui
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = DATA / 'lossless_candidate.svg'
    report, bad = {}, []
    for p in load_icons():
        if p[:-4] + '.svg' in NATIVE_SVG:
            report[p] = dict(method='native-svg')
            continue
        sp = orig_path(p)
        src = lossless.qimage_rgba(QtGui.QImage(str(sp)))
        svg, nrect = lossless.transcribe_lossless(src)
        tmp.write_text(svg)
        v = lossless.verify(tmp, src, sp)
        if v is None or v[2] > lossless.VERIFY_TOL or v[3] > lossless.VERIFY_TOL:
            bad.append((p, v))
            report[p] = dict(method='LOSSLESS-FAIL', verify=v)
            continue
        report[p] = dict(method='v8-lossless', rects=nrect, max_diff_1x=v[0],
                         max_diff_hidpi=v[1], vs_orig_1x=v[2], vs_orig_hidpi=v[3],
                         kb=round(len(svg.encode()) / 1024, 1),
                         size=[int(src.shape[1]), int(src.shape[0])])
        if args.write:
            repo_svg(p).write_text(svg)
    tmp.unlink(missing_ok=True)
    (DATA / 'lossless_report.json').write_text(json.dumps(report, indent=1))
    ok = [r for r in report.values() if r['method'] == 'v8-lossless']
    print(f"== lossless: {len(ok)} verified, "
          f"{sum(1 for r in report.values() if r['method'] == 'native-svg')} native, "
          f"{len(bad)} FAILED ==")
    for p, v in bad:
        print('  FAIL', p, v)
    if ok:
        print('vs ORIGINAL raster (same Qt pipeline): '
              f"1x worst={max(r['vs_orig_1x'] for r in ok)}/255, "
              f"HiDPI worst={max(r['vs_orig_hidpi'] for r in ok)}/255")
    print('repo write:', 'YES (--write)' if args.write else 'no (dry run)')
    return 1 if bad else 0


def cmd_redesign(args):
    import cv2
    import redesign
    from PySide6 import QtGui
    manifest = json.loads((HERE / 'manifest.json').read_text())
    qa = json.loads((HERE / 'qa.json').read_text())
    overrides = set(qa['overrides'])
    designs = Path(args.designs) if args.designs else DATA / 'designs'
    DATA.mkdir(parents=True, exist_ok=True)
    report, publish = {}, []
    hd_embed = qa.get('hd_embed', {})
    for p, base in sorted(manifest.items()):
        if p in hd_embed:
            continue  # user-selected HD art publishes as-is below
        svg, m = redesign.build_candidate(orig_path(p), designs / base, DATA / 'scratch')
        if svg is None:
            report[p] = m
            print('FAIL', p, m, flush=True)
            continue
        cand_svg(p).parent.mkdir(parents=True, exist_ok=True)
        cand_svg(p).write_text(svg)
        if m.pop('gate_ok'):
            m['qa'] = 'auto-gate'
            publish.append(p)
        elif p in overrides:
            # visually approved in the review sheets despite the numeric gate
            m['method'] = 'v10-embed' if m.get('info') == 'embedded-hd' else 'v10-vector'
            m['qa'] = 'manual-qa-override'
            publish.append(p)
        report[p] = m
        print(('OK  ' if p in publish else 'GATE'), p, m, flush=True)
    # solid-color test fixtures: exact original colour, crisp full-canvas rect
    for p in qa['solids']:
        src = redesign.lossless.qimage_rgba(QtGui.QImage(str(orig_path(p))))
        svg, hexcol = redesign.solid_svg(src)
        cand_svg(p).write_text(svg)
        report[p] = dict(method='v10-vector', info='solid-color', vmean=0.0, vfrac=0.0,
                         frac=0.0, iou=1.0, topo=1.0,
                         kb=round(len(svg.encode()) / 1024, 1), qa='solid-fixture')
        publish.append(p)
        print('SOLID', p, hexcol, flush=True)
    # monochrome line art: single dilated ink trace remapped to the orig bbox
    for p in qa['line_trace']:
        src = redesign.lossless.qimage_rgba(QtGui.QImage(str(orig_path(p))))
        keyed = redesign.key_white(cv2.imread(str(designs / manifest[p]), cv2.IMREAD_COLOR))
        svg, g = redesign.line_trace_svg(keyed, src, DATA / 'scratch')
        cand_svg(p).write_text(svg)
        report[p] = dict(method='v10-vector', info='line-trace', vmean=0.0, vfrac=0.0,
                         frac=round(g[0], 3), iou=round(g[1], 3), topo=round(g[2], 2),
                         kb=round(len(svg.encode()) / 1024, 1), qa='manual-qa-override')
        publish.append(p)
        print('LINE', p, report[p], flush=True)
    # hand-authored exact vector reproductions (simple geometry / line art where
    # stochastic generation drifts): published as-is, gates recorded for report
    for p, name in qa.get('author', {}).items():
        svg = (HERE / 'authored' / name).read_text()
        cand_svg(p).parent.mkdir(parents=True, exist_ok=True)
        cand_svg(p).write_text(svg)
        src = redesign.lossless.qimage_rgba(QtGui.QImage(str(orig_path(p))))
        scratch = DATA / 'scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        tmp = scratch / 'author.svg'
        tmp.write_text(svg)
        g = redesign.gates(tmp, src)
        report[p] = dict(method='v10-vector', info='author-exact', vmean=0.0, vfrac=0.0,
                         frac=round(g[0], 3) if g else None,
                         iou=round(g[1], 3) if g else None,
                         topo=round(g[2], 2) if g else None,
                         kb=round(len(svg.encode()) / 1024, 1), qa='author-exact')
        publish.append(p)
        print('AUTHOR', p, report[p], flush=True)
    # user-selected HD embeds (particle/gradient art approved in review sheets):
    # published as-is at 1024px; fidelity gates do not apply to full redesigns
    for p, name in hd_embed.items():
        svg = (HERE / 'authored' / name).read_text()
        cand_svg(p).parent.mkdir(parents=True, exist_ok=True)
        cand_svg(p).write_text(svg)
        report[p] = dict(method='v10-embed', info='user-selected-hd',
                         kb=round(len(svg.encode()) / 1024, 1), qa='user-selected')
        publish.append(p)
        print('HD-SEL', p, report[p], flush=True)
    shutil.rmtree(DATA / 'scratch', ignore_errors=True)
    (DATA / 'report.json').write_text(json.dumps(report, indent=1))
    print(Counter(m['method'] for m in report.values()), flush=True)
    if args.publish:
        for p in publish:
            shutil.copyfile(cand_svg(p), repo_svg(p))
    print(f'publish list: {len(publish)}; repo write:',
          'YES (--publish)' if args.publish else 'no (candidates only)')
    return 0


def cmd_gallery(args):
    import gallery
    rep8_path, rep10_path = DATA / 'lossless_report.json', DATA / 'report.json'
    if not rep8_path.exists() or not rep10_path.exists():
        print('run `lossless` and `redesign` first (reports missing)', file=sys.stderr)
        return 1
    rep8 = json.loads(rep8_path.read_text())
    rep10 = json.loads(rep10_path.read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    out, data = gallery.build_gallery(HERE, ROOT, rep8, rep10, manifest,
                                      DATA / 'designs', NATIVE_SVG)
    print('gallery:', round(out.stat().st_size / 1e6, 2), 'MB,', len(data), 'icons,',
          sum(1 for d in data if d['svg10']), 'v10 published ->', out)
    return 0


def cmd_compare4k(args):
    import compare4k
    rep8_path, rep10_path = DATA / 'lossless_report.json', DATA / 'report.json'
    if not rep8_path.exists() or not rep10_path.exists():
        print('run `lossless` and `redesign` first (reports missing)', file=sys.stderr)
        return 1
    rep8 = json.loads(rep8_path.read_text())
    rep10 = json.loads(rep10_path.read_text())
    manifest = json.loads((HERE / 'manifest.json').read_text())
    paths = compare4k.build_sheets(HERE, ROOT, rep8, rep10, manifest,
                                   DATA / 'designs', NATIVE_SVG, DATA / 'compare4k')
    for out in paths:
        print('sheet:', out.name, round(out.stat().st_size / 1e6, 2), 'MB')
    print(f'compare4k: {len(paths)} sheets @ {compare4k.W}x{compare4k.H}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('lossless', help='v8 pixel-exact transcription + verify gate')
    s.add_argument('--write', action='store_true', help='publish verified SVGs to the repo')
    s.set_defaults(fn=cmd_lossless)
    s = sub.add_parser('redesign', help='v10 commercial AI redesign pipeline')
    s.add_argument('--designs', help='design PNG dir (default: data/designs)')
    s.add_argument('--publish', action='store_true', help='publish gated/QA-approved SVGs to the repo')
    s.set_defaults(fn=cmd_redesign)
    s = sub.add_parser('gallery', help='build the tri-version comparison gallery')
    s.set_defaults(fn=cmd_gallery)
    s = sub.add_parser('compare4k', help='render 4K orig/v8/v10 comparison sheets')
    s.set_defaults(fn=cmd_compare4k)
    args = ap.parse_args()
    # the only place a Qt application is created; modules stay pure
    from PySide6 import QtGui
    app = QtGui.QGuiApplication(sys.argv)
    sys.exit(args.fn(args))


if __name__ == '__main__':
    main()