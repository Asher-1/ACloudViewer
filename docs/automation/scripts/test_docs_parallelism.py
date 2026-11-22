#!/usr/bin/env python3
"""Verify documentation worker limits and required Doxygen output contracts."""
import hashlib
import importlib.util
import json
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "make_docs",
    Path(__file__).resolve().parents[2] / "make_docs.py")
make_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_docs)


class SphinxParallelismTest(unittest.TestCase):

    def command(self, parallel, limit, cpus=32):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output/html").mkdir(parents=True)
            (root / "output/html/index.html").write_text("test",
                                                         encoding="utf-8")
            builder = make_docs.SphinxDocsBuilder(str(root),
                                                  str(root / "output"),
                                                  False,
                                                  parallel,
                                                  strict=True)
            with patch.dict(os.environ, {}, clear=True):
                if limit is not None:
                    os.environ["NPROC"] = limit
                with patch.object(make_docs.multiprocessing, "cpu_count", return_value=cpus), \
                     patch.object(make_docs.shutil, "which", return_value="sphinx-build"), \
                     patch.object(make_docs, "get_git_short_hash", return_value="test"), \
                     patch.object(make_docs.subprocess, "check_call") as call:
                    builder.run()
                    command = call.call_args.args[0]
                    self.assertIn("-W", command)
                    return command[command.index("-j") + 1]

    def test_serial_ignores_parallel_budget(self):
        self.assertEqual(self.command(False, "12"), "1")

    def test_parallel_honors_budget(self):
        self.assertEqual(self.command(True, "12"), "12")

    def test_ci_budget(self):
        self.assertEqual(self.command(True, "4"), "4")

    def test_budget_cannot_exceed_cpu_count(self):
        self.assertEqual(self.command(True, "12", cpus=4), "4")

    def test_missing_budget_retains_default(self):
        self.assertEqual(self.command(True, None), "32")

    def test_nonpositive_budget_uses_one_worker(self):
        self.assertEqual(self.command(True, "0"), "1")

    def test_invalid_budget_fails_instead_of_unbounded_build(self):
        with self.assertRaises(ValueError):
            self.command(True, "invalid")


class DoxygenThreadBudgetTest(unittest.TestCase):

    def configuration(self, budget, configured=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = ('PROJECT_NUMBER = @PROJECT_VERSION@\n'
                        'OUTPUT_DIRECTORY = @DOXYGEN_OUTPUT_DIRECTORY@\n'
                        f'DOT_NUM_THREADS = {configured}\n')
            (root / 'Doxyfile.in').write_text(template, encoding='utf-8')
            with patch.dict(os.environ, {}, clear=True):
                if budget is not None:
                    os.environ['NPROC'] = budget
                make_docs.process_doxyfile_template(str(root), '3.9.5')
            return (root / 'Doxyfile').read_text(encoding='utf-8')

    def test_dot_workers_honor_explicit_build_budget(self):
        self.assertIn('DOT_NUM_THREADS = 12', self.configuration('12'))
        self.assertIn('DOT_NUM_THREADS = 4', self.configuration('4'))
        self.assertIn('DOT_NUM_THREADS = 2', self.configuration('12', 2))

    def test_missing_budget_preserves_template_default(self):
        self.assertIn('DOT_NUM_THREADS = 0', self.configuration(None))

    def test_invalid_or_nonpositive_budget_fails(self):
        for budget in ('invalid', '0', '-2', ''):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.configuration(budget)

    def test_missing_template_cannot_reuse_old_generated_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / 'Doxyfile'
            old.write_text('DOT_NUM_THREADS = 0\n', encoding='utf-8')
            with self.assertRaises(FileNotFoundError):
                make_docs.process_doxyfile_template(str(root), '3.9.5')
            self.assertEqual(old.read_text(), 'DOT_NUM_THREADS = 0\n')

    def test_template_write_failure_propagates_before_doxygen_can_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Doxyfile.in').write_text('DOT_NUM_THREADS = 0\n')
            # A directory at the output filename deterministically denies the
            # write even when tests run with elevated filesystem permissions.
            (root / 'Doxyfile').mkdir()
            with patch.dict(os.environ, {'NPROC': '4'}), \
                    patch.object(make_docs.subprocess, 'check_call') as call:
                with self.assertRaises(IsADirectoryError):
                    make_docs.process_doxyfile_template(str(root), '3.9.5')
                call.assert_not_called()


class DoxygenOutputTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        self.builder = make_docs.DoxygenDocsBuilder("output", "doxygen")

    @staticmethod
    def formula_png():
        from PIL import Image
        stream = io.BytesIO()
        Image.new('L', (4, 4), color=255).save(stream, format='PNG')
        return stream.getvalue()

    def test_formula_failure_cannot_replace_existing_api_tree(self):
        (self.root / 'Doxyfile').write_text('OUTPUT_DIRECTORY = unused\n')
        api = self.root / 'output/html/cpp_api/api'
        api.mkdir(parents=True)
        (api / 'previous.html').write_text('last valid API')
        for image in (None, b'', b'not PNG', self.formula_png()[:-6]):
            with self.subTest(image=image):

                def incomplete_formula(*args, **kwargs):
                    root = self.root / 'doxygen/html'
                    root.mkdir()
                    (root / 'index.html').write_text(
                        '<img class="formulaDsp" src="form_0.png" alt="x^2">')
                    if image is not None:
                        (root / 'form_0.png').write_bytes(image)

                with patch.object(make_docs.subprocess,
                                  'check_call',
                                  side_effect=incomplete_formula):
                    with self.assertRaisesRegex(
                            RuntimeError,
                            'missing local images|Invalid Doxygen formula'):
                        self.builder.run()
                self.assertEqual((api / 'previous.html').read_text(),
                                 'last valid API')

    def test_local_images_are_required_but_external_images_are_not_fetched(
            self):
        root = self.root / 'images-check'
        root.mkdir()
        (root / 'index.html').write_text(
            '<img src="https://example.invalid/x.png">'
            '<img src="data:image/png;base64,AA==">'
            '<img src="image%20one.png?v=1#fragment">')
        with self.assertRaisesRegex(RuntimeError, 'image%20one.png'):
            self.builder._validate_html_images(root)
        (root / 'image one.png').write_bytes(b'image fixture')
        self.builder._validate_html_images(root)

    def test_formula_pixel_stream_must_be_complete(self):
        root = self.root / 'formula-check'
        root.mkdir()
        (root / 'index.html').write_text('<img src="form_0.png">')
        png = self.formula_png()
        (root / 'form_0.png').write_bytes(png)
        self.builder._validate_html_images(root)
        (root / 'form_0.png').write_bytes(png[:len(png) // 2])
        with self.assertRaisesRegex(RuntimeError, 'Invalid Doxygen formula'):
            self.builder._validate_html_images(root)

    def test_success_without_html_entry_is_rejected(self):
        (self.root / "Doxyfile").write_text("OUTPUT_DIRECTORY = unused\n",
                                            encoding="utf-8")
        for html_directory in (False, True):
            with self.subTest(html_directory=html_directory):

                def run_without_index(*args, **kwargs):
                    if html_directory:
                        (self.root / "doxygen/html").mkdir()

                with patch.object(make_docs.subprocess,
                                  "check_call",
                                  side_effect=run_without_index):
                    with self.assertRaisesRegex(RuntimeError,
                                                "index.html is missing"):
                        self.builder.run()
                self.assertFalse(
                    (self.root / "output/html/cpp_api/api").exists())

    def test_missing_requested_configuration_is_rejected(self):
        with patch.object(make_docs.subprocess, "check_call") as call:
            with self.assertRaisesRegex(RuntimeError, "Doxyfile not found"):
                self.builder.run()
        call.assert_not_called()

    def test_doxygen_failure_is_propagated(self):
        (self.root / "Doxyfile").write_text("OUTPUT_DIRECTORY = unused\n",
                                            encoding="utf-8")
        with patch.object(make_docs.subprocess,
                          "check_call",
                          side_effect=subprocess.CalledProcessError(
                              2, "doxygen")):
            with self.assertRaises(subprocess.CalledProcessError):
                self.builder.run()

    def test_valid_output_is_integrated_without_overwriting_sphinx_pages(self):
        (self.root / "Doxyfile").write_text(
            "OUTPUT_DIRECTORY = unused\nEXTRACT_ALL = YES\n", encoding="utf-8")
        api = self.root / "output/html/cpp_api/api"
        api.mkdir(parents=True)
        (api / "stale.html").write_text("old", encoding="utf-8")
        overview = api.parent / "overview.html"
        overview.write_text("Sphinx landing", encoding="utf-8")
        images = self.root / "images"
        images.mkdir()
        (images / "MainUI.png").write_bytes(b"image fixture")

        def generate_html(command, **kwargs):
            config = Path(command[1]).read_text(encoding="utf-8")
            self.assertIn(f"OUTPUT_DIRECTORY       = {self.root / 'doxygen'}",
                          config)
            self.assertIn("EXTRACT_ALL = YES", config)
            html = self.root / "doxygen/html"
            html.mkdir()
            (html / "index.html").write_text("C++ API", encoding="utf-8")
            (html / "classMesh.html").write_text(
                '<object data="classMesh__coll__graph.svg"></object>'
                '<map name="inline"><area href="classMesh.html"></map>'
                '<img class="formulaInl" src="form_0.png" alt="x^2">',
                encoding="utf-8")
            (html / "classMesh_8h_source.html").write_text("header",
                                                           encoding="utf-8")
            (html / "search.js").write_text("search", encoding="utf-8")
            (html / "form_0.png").write_bytes(self.formula_png())
            (html / "classMesh__coll__graph.svg").write_text(
                '<svg><a href="classMesh.html">Mesh</a></svg>',
                encoding="utf-8")
            (html / "classMesh__coll__graph.map").write_text(
                '<map id="graph"><area href="$classMesh.html"></map>',
                encoding="utf-8")
            (html / "classMesh__coll__graph.md5").write_text('a' * 32,
                                                             encoding="utf-8")
            # Legal downloads/source maps, including a paired SVG with a
            # non-Graphviz source map, must not be blanket-excluded.
            (html / "search.js.map").write_text('{"version":3}',
                                                encoding="utf-8")
            (html / "download.md5").write_text('b' * 32, encoding="utf-8")
            (html / "drawing.svg").write_text('<svg/>', encoding="utf-8")
            (html / "drawing.map").write_text('{"version":3}', encoding="utf-8")
            (html / "binary.svg").write_text('<svg/>', encoding="utf-8")
            (html / "binary.map").write_bytes(b'\xff\xfe user download')
            (html / "binary.md5").write_bytes(b'\xff\xfe checksum download')
            (html / "png_graph.map").write_text('<map id="png"/>',
                                                encoding="utf-8")
            (html / "assets").mkdir()
            (html / "assets/user.svg").write_text('<svg/>', encoding="utf-8")
            (html / "assets/user.map").write_text('<map id="user"/>',
                                                  encoding="utf-8")

        with patch.object(make_docs.subprocess,
                          "check_call",
                          side_effect=generate_html) as call:
            self.builder.run()
        self.assertEqual(call.call_args.args[0][0], "doxygen")
        self.assertEqual((api / "index.html").read_text(encoding="utf-8"),
                         "C++ API")
        self.assertTrue((api / "classMesh.html").is_file())
        source = self.root / "doxygen/html"
        for name in [
                'classMesh__coll__graph.map', 'classMesh__coll__graph.md5'
        ]:
            self.assertTrue((source / name).is_file())  # Build cache retained.
            self.assertFalse((api / name).exists())
        for name in [
                'classMesh.html', 'classMesh_8h_source.html', 'search.js',
                'form_0.png', 'classMesh__coll__graph.svg', 'search.js.map',
                'download.md5', 'drawing.map', 'binary.map', 'binary.md5',
                'png_graph.map', 'assets/user.map'
        ]:
            self.assertEqual((api / name).read_bytes(),
                             (source / name).read_bytes())
        self.assertEqual((api / "MainUI.png").read_bytes(), b"image fixture")
        self.assertFalse((api / "stale.html").exists())
        self.assertEqual(overview.read_text(encoding="utf-8"), "Sphinx landing")

        # A successful producer is insufficient if integration omits its tree.
        with patch.object(make_docs.subprocess, "check_call", side_effect=generate_html), \
             patch.object(make_docs.shutil, "copytree"):
            with self.assertRaisesRegex(RuntimeError,
                                        "integration has no entry page"):
                self.builder.run()


class OriginalIllustrationPublicationTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.api = root / 'api'
        self.assets = root / 'assets'
        self.api.mkdir()
        self.assets.mkdir()
        self.png = DoxygenOutputTest.formula_png()
        for name in ['QCPGraph.png', 'QCPBars.png']:
            (self.assets / name).write_bytes(self.png)
        (self.assets / 'manifest.json').write_text(
            json.dumps({
                'assets': [{
                    'name': name,
                    'sha256': hashlib.sha256(self.png).hexdigest()
                } for name in ['QCPGraph.png', 'QCPBars.png']]
            }))
        for name in ['README.md', 'LICENSE-NOTICE.md', 'LICENSE']:
            (self.assets / name).write_text('Original example attribution')
        spec = importlib.util.spec_from_file_location(
            'install_qcp_doc_images',
            Path(__file__).with_name('install_qcp_doc_images.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.install = module.install

    def test_only_referenced_missing_image_is_copied_with_attribution(self):
        page = self.api / 'classQCPGraph.html'
        page.write_text('<img src="QCPGraph.png">')
        before = page.read_bytes()
        self.assertEqual(self.install(self.api, self.assets), 1)
        self.assertEqual((self.api / 'QCPGraph.png').read_bytes(), self.png)
        self.assertFalse((self.api / 'QCPBars.png').exists())
        self.assertEqual(page.read_bytes(), before)
        self.assertTrue((self.api / 'QCustomPlot-original-README.md').is_file())

    def test_existing_historical_image_is_not_replaced(self):
        (self.api / 'index.html').write_text('<img src="QCPGraph.png">')
        (self.api / 'QCPGraph.png').write_bytes(b'historical own image')
        self.assertEqual(self.install(self.api, self.assets), 0)
        self.assertEqual((self.api / 'QCPGraph.png').read_bytes(),
                         b'historical own image')

    def test_corrupt_asset_cannot_create_partial_repair(self):
        (self.api / 'index.html'
        ).write_text('<img src="QCPGraph.png"><img src="QCPBars.png">')
        (self.assets / 'QCPBars.png').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum/format'):
            self.install(self.api, self.assets)
        self.assertFalse((self.api / 'QCPGraph.png').exists())

    def test_escape_and_unknown_images_fail_instead_of_fabricated_fallback(
            self):
        for source in ['../QCPGraph.png', 'QCPUnknown.png']:
            with self.subTest(source=source):
                (self.api / 'index.html').write_text(f'<img src="{source}">')
                with self.assertRaises(ValueError):
                    self.install(self.api, self.assets)
        self.assertFalse((self.api.parent / 'QCPGraph.png').exists())

    def test_attribution_symlink_cannot_write_outside_api(self):
        (self.api / 'index.html').write_text('<img src="QCPGraph.png">')
        outside = self.api.parent / 'outside-notice'
        (self.api / 'QCustomPlot-original-LICENSE').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'attribution target'):
            self.install(self.api, self.assets)
        self.assertFalse(outside.exists())
        self.assertFalse((self.api / 'QCPGraph.png').exists())

    def test_unrelated_api_does_not_require_or_copy_examples(self):
        (self.api / 'index.html').write_text('<img src="other.png">')
        self.assertEqual(self.install(self.api, self.assets / 'absent'), 0)


if __name__ == "__main__":
    unittest.main()
