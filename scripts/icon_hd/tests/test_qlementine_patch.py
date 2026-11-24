"""Regression gates for the CMake-owned third-party Git patch boundary."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / '3rdparty/qlementine/patches/apply_qlementine_patch.cmake'


@unittest.skipUnless(
    shutil.which('cmake') and shutil.which('git'),
    'CMake and Git are required for dependency patch gates')
class QlementinePatchTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='acv-patch-gate-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # A nested checkout reproduced a historical git-apply silent no-op.
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.source = self.root / 'nested/source'
        self.source.mkdir(parents=True)
        (self.source / 'existing.txt').write_text('before\n')
        self.patch = self.root / 'compat.patch'
        self.patch.write_text(
            '--- a/existing.txt\n+++ b/existing.txt\n@@ -1 +1 @@\n-before\n+after\n'
            '--- /dev/null\n+++ b/added.txt\n@@ -0,0 +1 @@\n+new compatibility\n'
        )
        self.manifest = self.root / 'manifest.json'
        digest = lambda b: hashlib.sha256(b).hexdigest()
        self.data = {
            'upstream':
                'v-fixture',
            'patch_sha256':
                digest(self.patch.read_bytes()),
            'files': [{
                'path': 'existing.txt',
                'before_sha256': digest(b'before\n'),
                'after_sha256': digest(b'after\n')
            }, {
                'path': 'added.txt',
                'before_sha256': None,
                'after_sha256': digest(b'new compatibility\n')
            }]
        }
        self.save_manifest()

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def run_patch(self, verify_only=False):
        return subprocess.run([
            shutil.which('cmake'), '-DVERIFY_ONLY=' +
            ('ON' if verify_only else 'OFF'), '-DSOURCE_DIR=' +
            str(self.source), '-DPATCH_FILE=' + str(self.patch),
            '-DMANIFEST_FILE=' + str(self.manifest), '-DGIT_EXECUTABLE=' +
            shutil.which('git'), '-DQLEMENTINE_VERSION=v-fixture', '-P',
            str(RUNNER)
        ],
                              capture_output=True,
                              text=True)

    def test_fresh_and_reused_tree_inside_repository(self):
        first = self.run_patch()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual((self.source / 'existing.txt').read_text(), 'after\n')
        self.assertEqual((self.source / 'added.txt').read_text(),
                         'new compatibility\n')
        second = self.run_patch()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)

    def test_reuse_verification_never_silently_repairs(self):
        self.assertNotEqual(self.run_patch(verify_only=True).returncode, 0)
        self.assertEqual(self.run_patch().returncode, 0)
        self.assertEqual(self.run_patch(verify_only=True).returncode, 0)
        (self.source / 'existing.txt').write_text('corrupted after build\n')
        self.assertNotEqual(self.run_patch(verify_only=True).returncode, 0)
        self.assertEqual((self.source / 'existing.txt').read_text(),
                         'corrupted after build\n')

    def test_partial_or_modified_tree_fails_closed(self):
        (self.source / 'existing.txt').write_text('after\n')
        self.assertNotEqual(self.run_patch().returncode, 0)
        (self.source / 'existing.txt').write_text('business edit\n')
        self.assertNotEqual(self.run_patch().returncode, 0)

    def test_changed_patch_requires_matching_manifest(self):
        self.patch.write_text(self.patch.read_text() + '\n')
        self.assertNotEqual(self.run_patch().returncode, 0)

    def test_post_apply_hash_mismatch_is_not_success(self):
        self.data['files'][0]['after_sha256'] = hashlib.sha256(
            b'wrong\n').hexdigest()
        self.save_manifest()
        self.assertNotEqual(self.run_patch().returncode, 0)

    def test_upstream_upgrade_requires_matching_reviewed_manifest(self):
        self.data['upstream'] = 'v-unreviewed-upgrade'
        self.save_manifest()
        self.assertNotEqual(self.run_patch().returncode, 0)

    def test_recipe_consumes_central_qt_owner(self):
        recipe = (ROOT / '3rdparty/qlementine/qlementine.cmake').read_text()
        self.assertNotIn('if(USE_QT6)', recipe)
        self.assertNotIn('find_package(${_qlementine_qt}', recipe)
        self.assertIn('set(_qlementine_qt "${QT_PREFIX}")', recipe)
        self.assertIn('INTERFACE_LINK_LIBRARIES "Qt::Core;Qt::Widgets;Qt::Svg"',
                      recipe)

    def test_manifest_cannot_escape_source(self):
        self.data['files'][0]['path'] = '../outside.txt'
        self.save_manifest()
        self.assertNotEqual(self.run_patch().returncode, 0)


if __name__ == '__main__':
    unittest.main()
