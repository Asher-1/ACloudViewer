#!/usr/bin/env python3
"""Regression coverage for autodoc event ordering and pybind directive repair."""
import ast
import contextlib
import glob
import os
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
import importlib.util
import io
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    "check_docstring_rst", Path(__file__).with_name("check_docstring_rst.py"))
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class DocstringPipelineTests(unittest.TestCase):
    def test_inline_note_wrap_is_one_directive_paragraph(self):
        doc = """Register a callback.

.. note:: Ordering is not guaranteed, since
some messages take longer than others. Use a unique message id
and include it in the reply.

.. code:: python

    def callback(data):
        print(data)

.. code:: js

    this.dataChannel.send('hello')
"""
        lines = checker.parse_like_sphinx(doc, name="cloudViewer.webrtc.callback")
        self.assertEqual(checker.diagnostics("\n".join(lines)), [])
        note = lines.index(".. note:: Ordering is not guaranteed, since")
        self.assertEqual(lines[note + 1], "   some messages take longer than others. Use a unique message id")
        self.assertIn("    def callback(data):", lines)
        self.assertIn("        print(data)", lines)
        self.assertIn("    this.dataChannel.send('hello')", lines)

    def test_empty_note_is_repaired_and_valid_note_unchanged(self):
        pre, _ = checker._load_transforms()
        repair = pre.__globals__["_indent_directive_bodies"]
        bad = [".. note::", "Body without indentation.", ""]
        repair(bad)
        self.assertEqual(bad, [".. note::", "", "   Body without indentation.", ""])
        valid = [".. note:: Inline text.", "   Correct continuation.", ""]
        before = valid[:]
        repair(valid)
        self.assertEqual(valid, before)

    def test_exact_prepare_400_500_600_order(self):
        events = []
        target = object()
        name = "cloudViewer.geometry.example"

        def pre(app, what, actual_name, obj, options, lines):
            self.assertEqual(actual_name, name)
            self.assertIs(obj, target)
            self.assertEqual(lines, ["Summary.", "", "Body.", ""])
            events.append("400")
            lines.append("pre")

        def google(lines):
            self.assertEqual(lines[-1], "pre")
            events.append("google500")
            return "\n".join(lines + ["google"])

        def numpy(lines):
            self.assertEqual(lines[-1], "google")
            events.append("numpy500")
            return "\n".join(lines + ["numpy"])

        def post(app, what, actual_name, obj, options, lines):
            self.assertEqual(lines[-1], "numpy")
            events.append("600")
            lines.append("post")

        with patch.object(checker, "_load_transforms", return_value=(pre, post)), \
             patch("sphinx.ext.napoleon.docstring.GoogleDocstring", side_effect=google), \
             patch("sphinx.ext.napoleon.docstring.NumpyDocstring", side_effect=numpy):
            result = checker.parse_like_sphinx("Summary.\n\n    Body.", name=name, obj=target)
        self.assertEqual(events, ["400", "google500", "numpy500", "600"])
        self.assertEqual(result[-1], "post")

    def test_main_does_not_apply_first_pass_again(self):
        pre, post = Mock(), Mock()
        name, doc = "cloudViewer.geometry.example", "Example."
        with patch.object(checker, "_load_transforms", return_value=(pre, post)), \
             patch.object(checker, "_documented_roots", return_value=[]), \
             patch.object(checker, "collect_docstrings", return_value=[(name, doc)]), \
             patch.object(checker, "parse_like_sphinx", return_value=["Example."]) as parse, \
             patch.object(checker, "diagnostics", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checker.main(), 0)
        parse.assert_called_once_with(doc, name=name)
        pre.assert_not_called()
        post.assert_not_called()

    def test_post_napoleon_fields_are_separated(self):
        lines = checker.parse_like_sphinx(
            "Summary.\n:param x: A value.\n    A continuation.\n",
            name="cloudViewer.geometry.example")
        field = next(i for i, line in enumerate(lines) if line.startswith(":param x:"))
        self.assertEqual(lines[field - 1], "")
        self.assertEqual(checker.diagnostics("\n".join(lines)), [])

    def test_math_body_and_namespaced_label_survive(self):
        doc = r"""A robust loss.

.. math::
\begin{equation}
x^{*} = \frac{1}{|X|}
\end{equation}
:label: robust_loss

See :eq:`robust_loss`.
"""
        lines = checker.parse_like_sphinx(doc, name="cloudViewer.geometry.loss")
        text = "\n".join(lines)
        self.assertIn(r"x^{*} = \frac{1}{|X|}", text)
        self.assertNotIn(r"x^{\*}", text)
        self.assertIn(":label: robust_loss_cloudViewer_geometry_loss", text)
        self.assertIn(":eq:`robust_loss_cloudViewer_geometry_loss`", text)
        self.assertLess(text.index(":label:"), text.index(r"\begin{equation}"))

    def test_both_handlers_leave_third_party_docstrings_alone(self):
        target = types.SimpleNamespace(__module__="torch")
        original = ["Summary.", ":param x: A value."]
        for handler in checker._load_transforms():
            lines = original[:]
            handler(None, None, "cloudViewer.exported", target, None, lines)
            self.assertEqual(lines, original)

    def test_third_party_alias_is_not_traversed(self):
        class Foreign(types.ModuleType):
            def __dir__(self):
                raise AssertionError("must not inspect third-party module aliases")

        root = types.ModuleType("cloudViewer")
        root.alias = Foreign("torch")
        def own():
            """Our own docstring."""
        own.__module__ = "cloudViewer.geometry"
        root.own = own
        self.assertEqual(list(checker.collect_docstrings([(root, "cloudViewer")])),
                         [("cloudViewer.own", "Our own docstring.")])


class PluginImagePublicationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source'
        self.output = self.root / 'html'
        self.plugin = self.src / 'plugins/qExample'
        (self.plugin / 'images').mkdir(parents=True)
        (self.plugin / 'README.md').write_text('Plugin documentation')
        self.page = self.output / 'plugins/qExample/README.html'
        self.page.parent.mkdir(parents=True)
        self.app = types.SimpleNamespace(
            srcdir=str(self.src),
            outdir=str(self.output),
            builder=types.SimpleNamespace(
                format='html',
                get_outfilename=lambda name: str(self.output /
                                                 (name + '.html'))))
        conf = Path(__file__).resolve().parents[2] / 'source/conf.py'
        tree = ast.parse(conf.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and
                        node.name == '_cleanup_plugin_docs')
        namespace = dict(os=os,
                         re=re,
                         glob=glob,
                         shutil=shutil,
                         unquote=unquote,
                         urlsplit=urlsplit)
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(conf),
                    'exec'), namespace)
        self.finish = namespace['_cleanup_plugin_docs']

    def test_raw_html_image_published_without_copying_unreferenced_images(self):
        (self.plugin / 'images/raw image.png').write_bytes(b'raw fixture')
        (self.plugin / 'images/unused.png').write_bytes(b'unused fixture')
        (self.output / '_images').mkdir()
        (self.output / '_images/markdown.png').write_bytes(b'already published')
        self.page.write_text(
            '''<img width="180" src="images/raw%20image.png?v=1">
<img src="../../_images/markdown.png"><img src="https://example.invalid/external.png">'''
        )
        self.finish(self.app, None)
        self.assertEqual(
            (self.page.parent / 'images/raw image.png').read_bytes(),
            b'raw fixture')
        self.assertFalse((self.page.parent / 'images/unused.png').exists())
        self.assertEqual((self.output / '_images/markdown.png').read_bytes(),
                         b'already published')
        self.assertFalse(self.plugin.exists())

    def test_missing_raw_image_is_not_a_successful_build(self):
        self.page.write_text('<img src="images/missing.png">')
        with self.assertRaisesRegex(FileNotFoundError,
                                    'Plugin image referenced'):
            self.finish(self.app, None)
        self.assertFalse((self.page.parent / 'images/missing.png').exists())

    def test_updated_plugin_source_replaces_its_previous_output(self):
        (self.plugin / 'images/raw.png').write_bytes(b'source fixture')
        (self.page.parent / 'images').mkdir()
        target = self.page.parent / 'images/raw.png'
        target.write_bytes(b'existing output')
        self.page.write_text('<img src="images/raw.png">')
        self.finish(self.app, None)
        self.assertEqual(target.read_bytes(), b'source fixture')

    def test_removed_source_cannot_be_hidden_by_old_output(self):
        (self.page.parent / 'images').mkdir()
        target = self.page.parent / 'images/removed.png'
        target.write_bytes(b'stale output')
        self.page.write_text('<img src="images/removed.png">')
        with self.assertRaisesRegex(FileNotFoundError,
                                    'Plugin image referenced'):
            self.finish(self.app, None)
        self.assertEqual(target.read_bytes(), b'stale output')

    def test_failed_build_only_cleans_sources(self):
        self.finish(self.app, RuntimeError('original Sphinx failure'))
        self.assertFalse(self.plugin.exists())
        self.assertFalse(self.page.exists())


if __name__ == "__main__":
    unittest.main()
