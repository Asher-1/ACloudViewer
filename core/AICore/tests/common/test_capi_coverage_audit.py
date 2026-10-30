#!/usr/bin/env python3
"""The audit must count real exports and calls; comments or strings must not fake coverage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "capi_coverage", Path(__file__).resolve().parents[1] / "check_capi_coverage.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class CoverageAuditTests(unittest.TestCase):
    def exercise(self, header, caller):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "core/AICore/include/aicore"
            include.mkdir(parents=True)
            (include / "sample_capi.h").write_text(header)
            consumer = root / "consumer"
            consumer.mkdir()
            (consumer / "sample.cpp").write_text(caller)
            with patch.object(AUDIT, "REPO", directory), patch.object(
                    AUDIT, "CONSUMER_DIRS", ["consumer"]):
                return AUDIT.api_functions(), AUDIT.referenced_functions()

    def test_documented_result_type_is_not_an_exported_function(self):
        api, _ = self.exercise("""
/* image + optional aicore_ocr_result (boxes + layout regions) */
// aicore_future_api() is not implemented.
AICORE_CAPI const struct aicore_sample_result *
aicore_sample_run(const char *path);
typedef void (*aicore_sample_callback)(const char *);
int aicore_internal(void);  // not a public export
""", "")
        self.assertEqual(api, {"aicore_sample_run"})

    def test_comments_and_strings_cannot_cover_a_missing_call(self):
        _, calls = self.exercise("", r'''
// aicore_missing();
/* aicore_other_missing(); */
const char *help = "aicore_fake() // not a comment";
const char *escaped = "\" aicore_escaped()";
void consume() { aicore_sample_run("real"); }
''')
        self.assertEqual(calls, {"aicore_sample_run"})

    def test_comment_between_symbol_and_arguments_keeps_a_real_call(self):
        api, calls = self.exercise(
            "AICORE_CAPI int aicore_sample_run /* note */ (void);",
            "aicore_sample_run /* real invocation */ (); ")
        self.assertEqual(api, {"aicore_sample_run"})
        self.assertEqual(calls, api)


if __name__ == "__main__":
    unittest.main()
