"""Regression checks for the reconstruction-media ParaView conversion."""
from pathlib import Path
import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
MEDIA=ROOT/'app/reconstruction/media'
LEDGER=Path.home()/'ACloudViewer-icon-reviews/reconstruction-media/migration-ledger.json'

class ReconstructionMediaTests(unittest.TestCase):
 def test_33_icons_have_stable_128_intrinsic_views_and_no_rasters(self):
  ledger=json.loads(LEDGER.read_text());self.assertEqual(len(ledger['icons']),33)
  self.assertEqual([r['id'] for r in ledger['icons']],list(range(1,34)))
  self.assertEqual(sum(r['source_kind']=='paraView-svg-root-size-only' for r in ledger['icons']),3)
  overlays=[r for r in ledger['icons'] if r.get('source_kind')=='user-requested-camera-workflow-cue']
  self.assertEqual([r['id'] for r in overlays],[19])
  self.assertIn('User explicitly requested',overlays[0]['authorization'])
  for row in ledger['icons']:
   with self.subTest(icon=row['name']):
    path=Path(row['after']);self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['after_sha256'])
    root=ET.parse(path).getroot();self.assertEqual((root.attrib['width'],root.attrib['height']),('128','128'))
    self.assertFalse(any(n.tag.rsplit('}',1)[-1]=='image' for n in root.iter()))
    self.assertFalse({'script','filter'} & {n.tag.rsplit('}',1)[-1] for n in root.iter()})
 def test_both_qrc_bundles_register_same_physical_media(self):
  for qrc in [ROOT/'app/reconstruction/resources.qrc',ROOT/'libs/Reconstruction/src/ui/resources.qrc']:
   tree=ET.parse(qrc).getroot();media=[(qrc.parent/(f.text or '').strip()).resolve() for g in tree for f in g.findall('file') if Path(f.text or '').name.endswith('.svg')]
   actual={p.name for p in media if p.parent==MEDIA.resolve()};self.assertEqual(len(actual),33);self.assertEqual(len(set(media)),len(media))
 def test_windows_brand_icons_licensed_and_byte_preserved(self):
  protected=json.loads((Path.home()/'ACloudViewer-icon-reviews/reconstruction-media/protected-media-sha256.json').read_text())
  for name in ['reconstruction_32.ico','reconstruction_64.ico','reconstruction_128.ico','LICENSE','PARAVIEW_NOTICE.txt']:
   self.assertEqual(hashlib.sha256((MEDIA/name).read_bytes()).hexdigest(),protected[name])
  self.assertIn('COLMAP_ADD_EXECUTABLE',(ROOT/'libs/Reconstruction/cmake/CMakeHelper.cmake').read_text())
if __name__=='__main__':unittest.main()
