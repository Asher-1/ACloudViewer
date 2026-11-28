"""Regression tests for exact resource ownership and XML/C++ URL scanning."""
import tempfile,unittest
from pathlib import Path
from scripts.icon_hd.audit_svg_inventory import runtime_text,registration_map
class ResourceAuditTests(unittest.TestCase):
 def test_cpp_split_literals_and_comment_removal(self):
  t='QIcon(":/Resources/images/svg/db/"\n"dbCloudSymbol.svg"); // :/bad.png\nconst char* url="https://a/b";'
  result=runtime_text('app/x.cpp',Path('app/x.cpp'),t)
  self.assertIn(':/Resources/images/svg/db/dbCloudSymbol.svg',result)
  self.assertNotIn(':/bad.png',result)
  self.assertIn('https://a/b',result)
 def test_ui_html_entities_are_decoded(self):
  result=runtime_text('app/a.ui',Path('a.ui'),'&lt;img src=&quot;:/Resources/images/donate.png&quot;&gt;')
  self.assertIn('src=":/Resources/images/donate.png"',result)
 def test_qrc_alias_not_disk_path_defines_runtime_url(self):
  with tempfile.TemporaryDirectory(prefix='acloudviewer-qrc-audit-') as tmp:
   p=Path(tmp);(p/'icons').mkdir();(p/'icons/a.svg').write_text('<svg/>')
   q=p/'test.qrc';q.write_text('<RCC><qresource prefix="/plugin"><file alias="a.svg">icons/a.svg</file></qresource></RCC>')
   bypath,byurl,missing,bad=registration_map([('test.qrc',q,q.read_text())])
   self.assertIn(':/plugin/a.svg',byurl);self.assertNotIn(':/plugin/icons/a.svg',byurl)
   self.assertFalse(missing);self.assertFalse(bad)
if __name__=='__main__':unittest.main()
