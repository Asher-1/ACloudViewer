"""Original-source integrity and approved-art protection without host dependencies."""
from pathlib import Path
import hashlib
import json
import re
import unittest
import xml.etree.ElementTree as ET
from scripts.icon_hd.logo.icon import _png

ROOT = Path(__file__).resolve().parents[3]
PQ = ROOT/'app/Resources/images/svg/pq'
MANIFEST = ROOT/'scripts/icon_hd/pq_original_sources.json'


def _loc(row):
    # Most restored originals live in pq/; the filter-toolbar restorations
    # (dir=filter) live in svg/filter/. The manifest row carries the directory.
    return ROOT/'app/Resources/images/svg'/row.get('dir', 'pq')/row['name']


def _strip_root_size(data: bytes) -> bytes:
    # Remove only the root <svg> width/height (the approved intrinsic preview
    # size, recorded as default_size). Geometry, viewBox and encoding stay
    # byte-exact, so the upstream pin still protects the drawing itself.
    i = data.find(b'<svg')
    if i < 0:
        return data
    j = data.find(b'>', i)
    tag = re.sub(rb'\s(?:width|height)="[^"]*"', b'', data[i:j])
    return data[:i] + tag + data[j:]


class PQOriginalsTests(unittest.TestCase):
    def test_pq_tree_has_no_pixel_mosaic_payloads(self):
        files = list(PQ.glob('*.svg'))
        self.assertEqual(len(files), 72)
        for path in files:
            with self.subTest(file=path.name):
                data = path.read_text()
                root = ET.fromstring(data)
                self.assertNotIn('Commercial HD', data)
                self.assertNotIn('Lossless pixel', data)
                self.assertLess(sum(n.tag.rsplit('}', 1)[-1] == 'rect' for n in root.iter()), 20)

    def test_fixed_ids_and_approved_assets_are_unchanged(self):
        rows = json.loads(MANIFEST.read_text())['items']
        self.assertEqual([r['id'] for r in rows], list(range(1, 25)))
        self.assertEqual([r['id'] for r in rows if r['status'] == 'approved-keep'], [15, 16, 19])
        for row in rows:
            self.assertEqual(hashlib.sha256(_loc(row).read_bytes()).hexdigest(), row['sha256'])

    def test_original_svg_replacements_are_byte_exact(self):
        rows = json.loads(MANIFEST.read_text())['items']
        direct = [r for r in rows if r['status'] == 'direct-svg']
        self.assertEqual(len(direct), 14)
        for row in direct:
            with self.subTest(file=row['name']):
                installed = _loc(row).read_bytes()
                # Installed pin covers the approved artwork including any
                # approved intrinsic-size attributes (default_size).
                self.assertEqual(hashlib.sha256(installed).hexdigest(), row['sha256'])
                if 'default_size' in row:
                    # We added root width/height to a size-less upstream: the
                    # geometry pin is the upstream raw bytes with our added
                    # attributes normalised away. Upstream files that already
                    # carry their own width/height keep the raw pin below.
                    w, h = row['default_size']
                    root = installed[:installed.find(b'>')]
                    self.assertIn(f'width="{w}"'.encode(), root)
                    self.assertIn(f'height="{h}"'.encode(), root)
                    self.assertEqual(hashlib.sha256(_strip_root_size(installed)).hexdigest(),
                                     row['source_sha256'])
                else:
                    self.assertEqual(row['sha256'], row['source_sha256'])
                source = Path(row['source'])
                # CI need not have a second ParaView checkout; pinned hashes are
                # the portable contract. Locally additionally compare raw bytes.
                if source.is_file():
                    self.assertEqual(_strip_root_size(installed),
                                     _strip_root_size(source.read_bytes()))

    def test_png_trace_and_non_upstream_source_are_explicit(self):
        rows = json.loads(MANIFEST.read_text())['items']
        traces = [r for r in rows if r['status'] in {'vtracer-png', 'pixel-svg-to-vector', 'source-measured-geometry'}]
        self.assertEqual(len(traces), 7)
        for row in traces:
            with self.subTest(file=row['name']):
                self.assertIsNotNone(_png(_loc(row).read_text(), 384).getbbox())
                if row['name'] == 'pqContours.svg':
                    self.assertEqual(row['status'], 'source-measured-geometry')
                else:
                    self.assertTrue(row['source'].endswith('.png'))

    def test_qrc_urls_unchanged_and_registered(self):
        tree = ET.parse(ROOT/'app/MainWindow.qrc').getroot()
        files = {f.text for g in tree for f in g.findall('file')}
        for row in json.loads(MANIFEST.read_text())['items']:
            self.assertIn('Resources/images/svg/'+row.get('dir', 'pq')+'/'+row['name'], files)


if __name__ == '__main__':
    unittest.main()
