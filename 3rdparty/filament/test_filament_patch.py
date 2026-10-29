"""Regression tests for the pinned Filament source patch (no downloads needed)."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest

PATCH = pathlib.Path(__file__).with_name("apply_filament_patch.cmake")
ZUTIL_OLD = "#if defined(MACOS) || defined(TARGET_OS_MAC)"
ZUTIL_NEW = "#if (defined(MACOS) || defined(TARGET_OS_MAC)) && !defined(__APPLE__)"
# The relevant platform branches from the pinned zutil.h, with NULL available.
ZUTIL_SOURCE = """#include <stddef.h>
#if defined(MACOS) || defined(TARGET_OS_MAC)
#  define OS_CODE  7
#  ifndef Z_SOLO
#    if defined(__MWERKS__) && __dest_os != __be_os && __dest_os != __win32_os
#      include <unix.h> /* for fdopen */
#    else
#      ifndef fdopen
#        define fdopen(fd,mode) NULL /* No fdopen() */
#      endif
#    endif
#  endif
#endif
#ifdef __APPLE__
#  define OS_CODE 19
#endif
#ifndef OS_CODE
#  define OS_CODE  3
#endif
"""


class FilamentPatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="filament patch ")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.memalign = self.root / "libs/utils/include/utils/memalign.h"
        self.sampler = self.root / "libs/image/src/ImageSampler.cpp"
        self.zutil = self.root / "third_party/libz/zutil.h"
        self.memalign.parent.mkdir(parents=True)
        self.sampler.parent.mkdir(parents=True)
        self.zutil.parent.mkdir(parents=True)
        self.memalign.write_text(
            "#include <type_traits>\n#include <stddef.h>\n"
            "using Difference = std::ptrdiff_t;\n")
        self.sampler.write_text("constexpr float M_PIf = 3.1415926f;\n")
        self.zutil.write_text(ZUTIL_SOURCE)

    def apply(self, source=True):
        args = ["cmake"]
        if source:
            args.append(f"-DSOURCE_DIR={self.root}")
        return subprocess.run(args + ["-P", str(PATCH)], capture_output=True,
                              text=True, check=False)

    def test_all_source_fixes(self):
        original = self.memalign.read_text()
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.memalign.read_text(), original.replace(
            "#include <type_traits>", "#include <cstddef>\n#include <type_traits>"))
        self.assertEqual(self.sampler.read_text(),
                         "constexpr float M_PI_f = 3.1415926f;\n")
        self.assertEqual(self.zutil.read_text(),
                         ZUTIL_SOURCE.replace(ZUTIL_OLD, ZUTIL_NEW))

    def test_repeated_application_does_not_rewrite(self):
        self.assertEqual(self.apply().returncode, 0)
        before = [(p.read_bytes(), p.stat().st_mtime_ns)
                  for p in (self.memalign, self.sampler, self.zutil)]
        self.assertEqual(self.apply().returncode, 0)
        after = [(p.read_bytes(), p.stat().st_mtime_ns)
                 for p in (self.memalign, self.sampler, self.zutil)]
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

    def test_missing_zutil_fails_before_mutation(self):
        self.zutil.rename(self.root / "zutil.saved")
        before = [p.read_bytes() for p in (self.memalign, self.sampler)]
        self.assertNotEqual(self.apply().returncode, 0)
        self.assertEqual([p.read_bytes() for p in (self.memalign, self.sampler)], before)

    def test_missing_zutil_anchor_fails_before_mutation(self):
        self.zutil.write_text("#define OS_CODE 19\n")
        paths = (self.memalign, self.sampler, self.zutil)
        before = [p.read_bytes() for p in paths]
        result = self.apply()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("zutil.h patch anchor is missing", result.stderr)
        self.assertEqual([p.read_bytes() for p in paths], before)

    def test_already_fixed_zutil_needs_no_old_anchor(self):
        self.zutil.write_text(ZUTIL_SOURCE.replace(ZUTIL_OLD, ZUTIL_NEW))
        before = (self.zutil.read_bytes(), self.zutil.stat().st_mtime_ns)
        self.assertEqual(self.apply().returncode, 0)
        self.assertEqual((self.zutil.read_bytes(), self.zutil.stat().st_mtime_ns), before)

    def compile_zutil(self, defines, mode, declaration=""):
        # Undefine host platform macros before selecting a simulated target.
        args = ["cc", "-std=c11", "-x", "c", "-U__APPLE__", "-UMACOS",
                "-UTARGET_OS_MAC", *defines, *mode, "-"]
        source = f'#include "{self.zutil.as_posix()}"\n' + declaration
        return subprocess.run(args, input=source, capture_output=True,
                              text=True, check=False)

    @unittest.skipUnless(shutil.which("cc"), "C compiler is unavailable")
    def test_modern_apple_fdopen_declaration_is_fixed(self):
        # Isolate the failing include order without requiring Apple's SDK.
        declaration = "void *fdopen(int, const char *);\n"
        for target in ("0", "1"):
            with self.subTest(target=target):
                defines = ["-D__APPLE__=1", f"-DTARGET_OS_MAC={target}"]
                result = self.compile_zutil(defines, ["-fsyntax-only"], declaration)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("fdopen", result.stderr)
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        for target in ("0", "1"):
            with self.subTest(patched_target=target):
                result = self.compile_zutil(
                    ["-D__APPLE__=1", f"-DTARGET_OS_MAC={target}"],
                    ["-fsyntax-only"], declaration)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("cc"), "C compiler is unavailable")
    def test_zutil_platform_branches_are_preserved(self):
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = (
            (["-D__APPLE__=1"], "19", False),
            (["-D__APPLE__=1", "-DTARGET_OS_MAC=1"], "19", False),
            (["-D__APPLE__=1", "-DMACOS=1"], "19", False),
            (["-DMACOS=1"], "7", True),
            (["-DTARGET_OS_MAC=1"], "7", True),
            ([], "3", False),
        )
        for defines, os_code, fallback in cases:
            with self.subTest(defines=defines):
                result = self.compile_zutil(defines, ["-E", "-dM"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertRegex(result.stdout, rf"#define OS_CODE\s+{os_code}\b")
                self.assertEqual("#define fdopen(" in result.stdout, fallback)

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
