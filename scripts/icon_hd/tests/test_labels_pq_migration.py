"""PCL label paths, fixed numbering, PQ-root categorisation and QRC closure."""
import unittest
import xml.etree.ElementTree as E
from scripts.icon_hd.logo import pcltools as P
from scripts.icon_hd.logo.icon import fit
from scripts.icon_hd.logo.paraviewextras import ICONS as PQ
ROOT=__import__('pathlib').Path(__file__).resolve().parents[3]
VECTOR_NAMES = ['pqCalculator.svg', 'pqFaceSetData.svg', 'pqRecentFile.svg', 'pqSolidColor.svg']


class LabelPQTests(unittest.TestCase):
 def test_all_pcl_candidates_have_outlined_semantic_labels(self):
  self.assertEqual(len(P.PLUGIN_ICONS),22);self.assertEqual(len(P.LABELS),22)
  for name,build,title in P.PLUGIN_ICONS.values():
   root=E.fromstring(fit(build()));labels=[n for n in root.iter() if 'data-label' in n.attrib]
   self.assertEqual([n.attrib['data-label'] for n in labels],[P.LABELS[name]])
   self.assertTrue(all(n.tag.rsplit('}',1)[-1]=='path' for n in labels))
   self.assertFalse({'text','image','script','filter'} & {n.tag.rsplit('}',1)[-1] for n in root.iter()})
 def test_original_algorithm_identifiers_retained(self):
  required={'pclMLSSmoothing.svg':'MLS','pclSIFT.svg':'SIFT','pclStatisticalOutliers.svg':'SOR','pclDifferenceOfNormals.svg':'DoN','pclSACSegmentation.svg':'SAC','pclFastGlobalRegistration.svg':'FGR','pclTemplateAlignment.svg':'(R,t)'}
  for name,label in required.items():self.assertEqual(P.LABELS[name],label)
 def test_root_svg_directory_has_no_loose_icons(self):
  root=ROOT/'app/Resources/images/svg'
  self.assertFalse([p for p in root.iterdir() if p.is_file()])
  for name in VECTOR_NAMES:self.assertTrue((root/'pq'/name).is_file())
  # The former raster generators are historical design references; the current
  # ParaView-source manifest, not those rejected sketches, is the runtime authority.
  for old,(name,build,label) in PQ.items():
   self.assertTrue((root/(name if '/' in name else 'pq/'+name)).is_file(),name)
   self.assertFalse((root/old).exists())
 def test_pq_qrc_urls_are_the_canonical_location(self):
  tree=E.parse(ROOT/'app/MainWindow.qrc').getroot();files={n.text for group in tree for n in group.findall('file')}
  for name in VECTOR_NAMES+[row[0] for row in PQ.values()]:self.assertIn('Resources/images/svg/'+(name if '/' in name else 'pq/'+name),files)
if __name__=='__main__':unittest.main()
