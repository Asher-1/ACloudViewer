#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("check_plugin_aicore_boundaries.py")
SPEC = importlib.util.spec_from_file_location("aicore_plugin_boundaries",
                                               MODULE_PATH)
assert SPEC and SPEC.loader
BOUNDARIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARIES)


class PluginBoundaryTests(unittest.TestCase):
    def plugin_source(self, root: Path, text: str,
                      relative: str = "src/Dialog.cpp") -> Path:
        path = root / "plugins/core/Standard/qDemo" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_protocols_escaped_text_hashing_and_comments_are_not_debt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plugin_source(root, r'''
#include "aicore/depth_capi.h"
const char* uri = "db://models/current";
const char* message = "missing:\n";
auto digest = QCryptographicHash::hash(bytes, QCryptographicHash::Sha256);
// Source: https://github.com/Asher-1/cloudViewer_downloads/releases/download/X/y
''')
            issues, warnings = BOUNDARIES.check_plugin_root(root)
        self.assertEqual(issues, [])
        self.assertEqual(warnings, [])

    def test_production_model_ownership_and_absolute_paths_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plugin_source(root, r'''
#include "aicore/depth_capi.h"
#include "aicore/asset_digests.h"
QNetworkAccessManager manager;
const char* model = "/home/developer/model.gguf";
const char* second_model = "/opt/local/model.gguf";
const char* url = "https://github.com/Asher-1/cloudViewer_downloads/releases/download/X/y";
''')
            issues, warnings = BOUNDARIES.check_plugin_root(root)
            strict_issues, strict_warnings = BOUNDARIES.check_plugin_root(
                root, strict=True)
        self.assertEqual(issues, [])
        self.assertEqual(len(warnings), 5)
        self.assertEqual(len(strict_issues), 5)
        self.assertEqual(strict_warnings, [])

    def test_test_fixture_model_literals_do_not_create_migration_debt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.plugin_source(root, '#include "aicore/depth_capi.h"\n')
            self.plugin_source(root, r'''
const char* model = "/home/fixture/model.gguf";
QNetworkAccessManager manager;
''', "tests/TestDialog.cpp")
            issues, warnings = BOUNDARIES.check_plugin_root(root)
        self.assertEqual(issues, [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
