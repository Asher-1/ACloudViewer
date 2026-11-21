#!/usr/bin/env python3
"""Regression coverage for autodoc event ordering and pybind directive repair."""
import contextlib
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


if __name__ == "__main__":
    unittest.main()
