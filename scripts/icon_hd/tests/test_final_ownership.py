"""Plugin ownership, no dependency inversion, and complete resource migration."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as E
from scripts.icon_hd.logo import pcltools
from scripts.icon_hd.logo.icon import fit
ROOT=Path(__file__).resolve().parents[3]
class OwnershipTests(unittest.TestCase):
 def test_pcl_owns_20_consumer_backed_vectors(self):
  folder=ROOT/'plugins/core/Standard/qPCL/resources/icons'
  self.assertFalse((ROOT/'app/Resources/images/svg/pcl').exists())
  self.assertEqual(len(list(folder.glob('*.svg'))),20)
  for name,fn,_ in pcltools.PLUGIN_ICONS.values():
   if name in pcltools.SOURCE_ONLY:continue
   data=(folder/name).read_text();self.assertEqual(data,fit(fn()))
   self.assertFalse({'text','image','filter','script'} & {x.tag.rsplit('}',1)[-1] for x in E.fromstring(data).iter()})
 def test_plugin_qrc_resolves_locally_without_app_paths(self):
  qrc=ROOT/'plugins/core/Standard/qPCL/qPCL.qrc';tree=E.parse(qrc)
  group=tree.find("qresource[@prefix='/qPCL/icons']")
  self.assertEqual(len(group.findall('file')),20)
  for node in group:
   self.assertEqual(node.attrib.get('alias'),Path(node.text).name)
   self.assertTrue(node.text.startswith('resources/icons/'))
   self.assertTrue((qrc.parent/node.text).is_file())
  self.assertNotIn('app/Resources',qrc.read_text())
 def test_main_application_has_no_optional_pcl_icon_dependency(self):
  for s in ['app/MainWindow.qrc','app/ui_templates/MainWindow.ui']:
   text=(ROOT/s).read_text();self.assertNotIn('/svg/pcl/',text);self.assertNotIn(':/qPCL/',text)
 def test_no_single_file_subdirectory_remains(self):
  root=ROOT/'app/Resources/images/svg'
  for d in root.iterdir():
   if d.is_dir():self.assertNotEqual(len([p for p in d.iterdir() if p.is_file()]),1,str(d))
 def test_root_resources_migrated_and_container_is_live(self):
  self.assertFalse((ROOT/'app/Resources/images/frame.png').exists())
  self.assertFalse((ROOT/'app/Resources/images/dbContainerSymbol.svg').exists())
  self.assertTrue((ROOT/'app/Resources/images/svg/misc/frame.svg').is_file())
  self.assertIn(':/Resources/images/svg/db/dbContainerSymbol.svg',(ROOT/'app/db_tree/ecvDBRoot.cpp').read_text())
if __name__=='__main__':unittest.main()
