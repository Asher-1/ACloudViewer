#!/usr/bin/env python3
"""Regression checks for migration conflicts and shared-model ownership."""

import json
from pathlib import Path
import tempfile
import unittest

import audit_llama_migration as audit


class MigrationAuditTests(unittest.TestCase):
    def test_conflicts_preserve_both_sides_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, patched, new = [root / x for x in ("old", "patched", "new")]
            for p in (old, patched, new):
                p.mkdir()
            (old / "kernel.c").write_text("int value = 1;\n")
            (patched / "kernel.c").write_text("int value = 2;\n")
            (new / "kernel.c").write_text("int value = 3;\n")
            (patched / "custom.c").write_text("custom operator\n")
            (old / "removed.metal").write_text("old kernel\n")
            (patched / "removed.metal").write_text("optimized kernel\n")
            before = [audit.tree_identity(p) for p in (old, patched, new)]
            destination = root / "candidate"
            rows = audit.merge_inventory(old, patched, new, destination)
            self.assertEqual({r["file"]: r["status"] for r in rows}, {
                "kernel.c": "conflict", "custom.c": "added",
                "removed.metal": "upstream_deleted"})
            text = (destination / "kernel.c").read_text()
            self.assertIn("int value = 2;", text)
            self.assertIn("int value = 3;", text)
            self.assertIn("<<<<<<<", text)
            self.assertEqual(before, [audit.tree_identity(p) for p in (old, patched, new)])
            with self.assertRaises(ValueError):
                audit.merge_inventory(old, patched, new, destination)

    def test_shared_catalog_consumers_and_dot_quantization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            records = [
                {"task": "yolo", "relative_path": "yolo_models/reid-f16.gguf"},
                {"task": "reid", "relative_path": "yolo_models/reid-f16.gguf"},
                {"task": "loma", "relative_path": "loma_models/matcher.q8_0.gguf"},
                {"task": "loma", "relative_path": "loma_models/detector.f32.gguf"},
                {"task": "loma", "relative_path": "loma_models/unknown.gguf"},
            ]
            catalog.write_text("\n".join(json.dumps(x) for x in records))
            manifest = {"tasks": ["yolo", "reid", "loma"], "scenarios": [
                {"task": "reid", "id": "embeddings", "for_each_model": True,
                 "model_glob": "yolo_models/reid-*.gguf"},
                {"task": "loma", "id": "matches", "for_each_model": True,
                 "model_glob": "loma_models/*q8_0.gguf"},
            ]}
            result = audit.catalog_inventory(catalog, manifest, root / "assets")
            self.assertEqual(result["selected_unique_files"], 3)
            self.assertEqual(result["all_catalog_unique_files"], 4)
            self.assertIn("loma_models/detector.f32.gguf",
                          result["all_without_consumer"])
            self.assertIn("loma_models/detector.f32.gguf",
                          result["tasks"][2]["all_without_consumer"])
            rows = result["rows"]
            self.assertEqual(rows[0]["consumers"], ["reid/embeddings"])
            self.assertEqual(rows[2]["filename_quant_hint"], "q8_0")
            self.assertFalse(rows[3]["selected"])
            self.assertTrue(rows[4]["selected"])
            self.assertEqual(rows[4]["filename_quant_hint"], "unspecified")
            self.assertFalse(result["inference_executed"])

    def test_merge_destination_cannot_recurse_into_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("old", "patched", "new"):
                (root / name).mkdir()
            with self.assertRaises(ValueError):
                audit.merge_inventory(root / "old", root / "patched", root / "new",
                                      root / "new/candidate")
            self.assertFalse((root / "new/candidate").exists())

    def test_report_cannot_overwrite_a_source_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            kernel = source / "kernel.c"
            kernel.write_text("original kernel\n")
            alias = root / "alias"
            alias.symlink_to(source, target_is_directory=True)
            for output in (kernel, alias / "kernel.c", source / "report.json"):
                with self.assertRaises(ValueError):
                    audit.validate_report_path(output, (source,))
            audit.validate_report_path(root / "report.json", (source,))
            self.assertEqual(kernel.read_text(), "original kernel\n")


if __name__ == "__main__":
    unittest.main()
