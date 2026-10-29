"""Regression tests for the pinned Filament source patch (no downloads needed)."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

PATCH = pathlib.Path(__file__).with_name("apply_filament_patch.cmake")


class FilamentPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="filament patch ")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.memalign = self.root / "libs/utils/include/utils/memalign.h"
        self.sampler = self.root / "libs/image/src/ImageSampler.cpp"
        self.memalign.parent.mkdir(parents=True)
        self.sampler.parent.mkdir(parents=True)
        self.memalign.write_text(
            "#include <type_traits>\n#include <stddef.h>\n"
            "using Difference = std::ptrdiff_t;\n")
        self.sampler.write_text("constexpr float M_PIf = 3.1415926f;\n")

    def apply(self, source=True):
        args = ["cmake"]
        if source:
            args.append(f"-DSOURCE_DIR={self.root}")
        return subprocess.run(args + ["-P", str(PATCH)], capture_output=True,
                              text=True, check=False)

    def test_both_source_fixes(self):
        original = self.memalign.read_text()
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.memalign.read_text(), original.replace(
            "#include <type_traits>", "#include <cstddef>\n#include <type_traits>"))
        self.assertEqual(self.sampler.read_text(),
                         "constexpr float M_PI_f = 3.1415926f;\n")

    def test_repeated_application_does_not_rewrite(self):
        self.assertEqual(self.apply().returncode, 0)
        before = [(p.read_bytes(), p.stat().st_mtime_ns)
                  for p in (self.memalign, self.sampler)]
        self.assertEqual(self.apply().returncode, 0)
        after = [(p.read_bytes(), p.stat().st_mtime_ns)
                 for p in (self.memalign, self.sampler)]
        self.assertEqual(before, after)

    def test_already_fixed_header_needs_no_old_anchor(self):
        self.memalign.write_text("#include <cstddef>\nusing Difference = std::ptrdiff_t;\n")
        self.assertEqual(self.apply().returncode, 0)
        self.assertEqual(self.memalign.read_text().count("#include <cstddef>"), 1)

    def test_missing_anchor_fails_before_mutation(self):
        self.memalign.write_text("#include <stddef.h>\n")
        before = self.sampler.read_bytes()
        result = self.apply()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("patch anchor is missing", result.stderr)
        self.assertEqual(self.sampler.read_bytes(), before)

    def test_missing_source_argument_fails(self):
        result = self.apply(source=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires SOURCE_DIR", result.stderr)

    def test_missing_sampler_fails_before_mutation(self):
        self.sampler.rename(self.root / "sampler.saved")
        before = self.memalign.read_bytes()
        self.assertNotEqual(self.apply().returncode, 0)
        self.assertEqual(self.memalign.read_bytes(), before)

    @unittest.skipUnless(shutil.which("c++"), "C++ compiler is unavailable")
    def test_missing_std_declaration_is_fixed(self):
        # Isolate the missing declaration: older type_traits headers can hide
        # the defect by transitively including cstddef, unlike the failing SDK.
        flags = ["c++", "-std=c++17", "-x", "c++", "-fsyntax-only", "-"]
        source = f'#include "{self.memalign.as_posix()}"\n'
        # C++ libraries differ in C-header namespace exports; only the fixed
        # header compilation is portable enough to be a mandatory assertion.
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(flags, input=source,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
