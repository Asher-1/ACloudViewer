"""Regression tests for the macOS wheel bundler (no macOS tools needed).

Covers the libomp.dylib conflict: conda's copy (minimum macOS 11.0, linked
by our libraries) and torch/lib's copy (minimum 14.0, reachable from the
bundled libtorch_cpu.dylib) were both copied into lib/, the string hash
seed picked the survivor, and the wheel tag drifted to macosx_14_0.
"""
import ast
import io
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import lib_bundle_wheel as bundler  # noqa: E402


def encode_version(major, minor=0, patch=0):
    return (major << 16) | (minor << 8) | patch


def thin_macho(minos, legacy=False, platform=1, wide=True):
    """A little-endian Mach-O header with one version load command."""
    if legacy:
        cmd = struct.pack("<IIII", 0x24, 16, encode_version(*minos), 0)
    else:
        cmd = struct.pack("<IIIIII", 0x32, 24, platform, encode_version(*minos),
                          encode_version(14, 5), 0)
    uuid = struct.pack("<II", 0x1B, 24) + bytes(16)
    cmds = uuid + cmd
    if wide:
        header = struct.pack("<IiiIIIII", 0xFEEDFACF, 0x0100000C, 0, 6, 2, len(cmds), 0, 0)
    else:
        header = struct.pack("<IiiIIII", 0xFEEDFACE, 7, 0, 6, 2, len(cmds), 0)
    return header + cmds


def fat_macho(slices, wide=False):
    """A fat binary whose slices start on 4 KiB boundaries."""
    table = b""
    body = b""
    offset = 4096
    for data in slices:
        if wide:
            table += struct.pack(">iiQQII", 0x0100000C, 0, offset, len(data), 12, 0)
        else:
            table += struct.pack(">iiIII", 0x0100000C, 0, offset, len(data), 12)
        padded = data + bytes(-len(data) % 4096)
        body += padded
        offset += len(padded)
    head = struct.pack(">II", 0xCAFEBABF if wide else 0xCAFEBABE, len(slices)) + table
    return head + bytes(4096 - len(head)) + body


class SelectEmbeddedLibrariesTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.conda = root / "conda" / "lib" / "libomp.dylib"
        self.torch = root / "site-packages" / "torch" / "lib" / "libomp.dylib"
        for path, minos in ((self.conda, (11, 0)), (self.torch, (14, 0))):
            path.parent.mkdir(parents=True)
            path.write_bytes(thin_macho(minos))

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_linked_libomp_wins_in_every_order(self):
        preferred = {"libomp.dylib": self.conda}
        for order in ([self.conda, self.torch], [self.torch, self.conda]):
            selected = bundler.select_embedded_libraries(order, preferred)
            self.assertEqual(selected, {"libomp.dylib": self.conda})

    def test_build_linked_libomp_wins_under_every_hash_seed(self):
        # The original defect: set iteration order follows PYTHONHASHSEED.
        script = (
            "import pathlib, sys; sys.path.insert(0, sys.argv[1]);"
            "import lib_bundle_wheel as b;"
            "c, t = map(pathlib.Path, sys.argv[2:4]);"
            "print(b.select_embedded_libraries({c, t}, {c.name: c})[c.name])")
        for seed in range(1, 9):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            out = subprocess.run(
                [sys.executable, "-c", script, str(HERE), str(self.conda), str(self.torch)],
                env=env, capture_output=True, text=True, check=True).stdout.strip()
            self.assertEqual(out, str(self.conda), f"PYTHONHASHSEED={seed}")

    def test_build_linked_symlink_selects_the_same_file(self):
        alias = self.conda.with_name("libomp.5.dylib")
        alias.symlink_to(self.conda.name)
        selected = bundler.select_embedded_libraries(
            [self.conda, self.torch], {"libomp.dylib": alias})
        self.assertEqual(selected, {"libomp.dylib": self.conda})

    def test_unknown_library_conflict_fails(self):
        left = self.conda.with_name("libfoo.dylib")
        right = self.torch.with_name("libfoo.dylib")
        left.write_bytes(b"left")
        right.write_bytes(b"right")
        with self.assertRaisesRegex(RuntimeError, "different files resolve to lib/libfoo.dylib"):
            bundler.select_embedded_libraries([left, right], {})

    def test_unresolved_conflict_fails(self):
        with self.assertRaisesRegex(RuntimeError, "different files resolve to lib/libomp.dylib"):
            bundler.select_embedded_libraries([self.conda, self.torch], {})

    def test_preference_outside_candidates_fails(self):
        other = pathlib.Path(self.tmp.name) / "brew" / "libomp.dylib"
        other.parent.mkdir()
        other.write_bytes(b"brew")
        with self.assertRaises(RuntimeError):
            bundler.select_embedded_libraries([self.conda, self.torch],
                                              {"libomp.dylib": other})

    def test_identical_copies_are_interchangeable(self):
        twin = pathlib.Path(self.tmp.name) / "twin" / "libomp.dylib"
        twin.parent.mkdir()
        twin.write_bytes(self.conda.read_bytes())
        selected = bundler.select_embedded_libraries([twin, self.conda], {})
        self.assertEqual(selected, {"libomp.dylib": min(twin, self.conda)})

    def test_distinct_names_are_kept(self):
        libfoo = self.conda.with_name("libfoo.dylib")
        libfoo.write_bytes(b"foo")
        selected = bundler.select_embedded_libraries([libfoo, self.torch], {})
        self.assertEqual(selected, {"libfoo.dylib": libfoo, "libomp.dylib": self.torch})


class MachOVersionTest(unittest.TestCase):

    def versions(self, data):
        return bundler.macho_min_macos_versions(io.BytesIO(data))

    def test_build_version(self):
        self.assertEqual(self.versions(thin_macho((14, 0))), [(14, 0, 0)])

    def test_legacy_version_min(self):
        self.assertEqual(self.versions(thin_macho((10, 15), legacy=True)), [(10, 15, 0)])

    def test_non_macos_platform_is_ignored(self):
        self.assertEqual(self.versions(thin_macho((17, 0), platform=2)), [])

    def test_fat_binary(self):
        data = fat_macho([thin_macho((11, 0)), thin_macho((14, 0))])
        self.assertEqual(self.versions(data), [(11, 0, 0), (14, 0, 0)])

    def test_fat64_binary(self):
        data = fat_macho([thin_macho((11, 0)), thin_macho((14, 0))], wide=True)
        self.assertEqual(self.versions(data), [(11, 0, 0), (14, 0, 0)])

    def test_thin32_binary(self):
        self.assertEqual(self.versions(thin_macho((10, 15), legacy=True, wide=False)),
                         [(10, 15, 0)])

    def test_non_macho_content(self):
        java_class = struct.pack(">IHH", 0xCAFEBABE, 0, 52) + bytes(32)
        for data in (b"", b"#!/bin/sh\n", b"\x7fELF" + bytes(60), java_class):
            self.assertEqual(self.versions(data), [])


class CheckWheelTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def wheel(self, tag, libomp_minos, pybind_minos=(11, 0)):
        path = self.root / f"cloudviewer-3.9.5-cp312-cp312-{tag}.whl"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("cloudViewer/__init__.py", "")
            archive.writestr("cloudViewer/lib/libomp.dylib", thin_macho(libomp_minos))
            archive.writestr("cloudViewer/cpu/pybind.cpython-312-darwin.so",
                             fat_macho([thin_macho(pybind_minos)]))
        return path

    def test_conforming_wheel_passes(self):
        self.assertEqual(bundler.check_wheel(self.wheel("macosx_11_0_arm64", (11, 0)), "11.0"), [])

    def test_torch_libomp_is_reported(self):
        errors = bundler.check_wheel(self.wheel("macosx_14_0_arm64", (14, 0)), "11.0")
        self.assertEqual(len(errors), 2, errors)
        self.assertIn("macosx_14_0_arm64 is not macosx_11_0", errors[0])
        self.assertIn("cloudViewer/lib/libomp.dylib requires macOS 14.0.0 > 11.0", errors[1])

    def test_wrong_tag_alone_is_reported(self):
        errors = bundler.check_wheel(self.wheel("macosx_12_0_arm64", (11, 0)), "11.0")
        self.assertEqual(len(errors), 1, errors)

    def test_legacy_target_keeps_minor(self):
        wheel = self.wheel("macosx_10_15_x86_64", (10, 15), pybind_minos=(10, 15))
        self.assertEqual(bundler.check_wheel(wheel, "10.15"), [])
        errors = bundler.check_wheel(self.wheel("macosx_10_15_x86_64", (10, 15)), "10.15")
        self.assertEqual(errors, ["cloudViewer/cpu/pybind.cpython-312-darwin.so "
                                  "requires macOS 11.0.0 > 10.15"])

    def test_cli_checks_newest_wheel_in_directory(self):
        self.wheel("macosx_14_0_arm64", (14, 0))
        env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET="11.0")
        result = subprocess.run(
            [sys.executable, str(HERE / "lib_bundle_wheel.py"), "--check_wheel", str(self.root)],
            env=env, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("requires macOS 14.0.0", result.stderr)


@unittest.skipUnless(shutil.which("cmake") and shutil.which("make"), "CMake and make required")
class CMakePackagingTest(unittest.TestCase):
    """Exercise the production CMake command blocks without a macOS build."""

    def test_bundle_arguments_and_failure_propagation(self):
        script = (HERE.parents[3] / "libs/Python/pybind/make_python_package.cmake").read_text()
        block = script[script.index("    set(_mac_wheel_bundle_args"):
                       script.index("elseif (UNIX)", script.index("    set(_mac_wheel_bundle_args"))]
        with tempfile.TemporaryDirectory(prefix="wheel cmake ") as tmp:
            root = pathlib.Path(tmp)
            tool = root / "capture args.py"
            capture = root / "args.txt"
            tool.write_text("import pathlib, sys; "
                            f"pathlib.Path({str(capture)!r}).write_text(repr(sys.argv[1:])); "
                            "sys.exit(int(sys.argv[1]))")
            driver = root / "args.cmake"
            driver.write_text(f'set(PACKAGE_TOOL "{tool.as_posix()}")\n'
                              f'set(PYTHON_PACKAGE_DST_DIR "{root.as_posix()}")\n' + block)
            env = dict(os.environ, PATH=str(pathlib.Path(sys.executable).parent) +
                       os.pathsep + os.environ.get("PATH", ""))
            for value in ("", "OpenMP_libomp_LIBRARY-NOTFOUND", str(root / "conda lib/libomp.dylib")):
                result = subprocess.run(
                    ["cmake", "-DDEST_PATH=0", f"-DOpenMP_libomp_LIBRARY={value}", "-P", str(driver)],
                    env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = ["0"] + (["--openmp_library", value] if "libomp.dylib" in value else [])
                self.assertEqual(ast.literal_eval(capture.read_text()), expected)
            result = subprocess.run(["cmake", "-DDEST_PATH=7", "-P", str(driver)],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("macOS wheel bundling failed (exit 7)", result.stderr)

    def test_apple_wheel_gate_runs_after_bdist_and_is_platform_scoped(self):
        script = (HERE.parents[3] / "libs/Python/pybind/CMakeLists.txt").read_text()
        block = script[script.index("set(PIP_PACKAGE_CHECKS)"):
                       script.index("# Use `make install-pip-package`")]
        with tempfile.TemporaryDirectory(prefix="wheel cmake ") as tmp:
            root = pathlib.Path(tmp)
            # The real checker is used through a source path containing spaces.
            tool = root / "scripts/platforms/mac/bundle/lib_bundle_wheel.py"
            tool.parent.mkdir(parents=True)
            shutil.copy2(HERE / tool.name, tool)
            shutil.copy2(HERE / "bundle_slim.py", tool.parent)
            setup = root / "setup.py"
            setup.write_text("import os, pathlib, zipfile; "
                             "p=pathlib.Path('pip_package'); p.mkdir(exist_ok=True); "
                             "tag=os.environ['TEST_PLATFORM_TAG']; "
                             "w=p/f'cloudviewer-3.9.5-cp312-cp312-{tag}.whl'; "
                             "zipfile.ZipFile(w, 'w').close()")
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\nproject(WheelGate NONE)\n"
                f'set(Python3_EXECUTABLE "{sys.executable}")\n'
                f'set(PYTHON_PACKAGE_DST_DIR "{root.as_posix()}")\n'
                "add_custom_target(python-package)\n" + block)
            # project() derives APPLE from the target system, overriding -DAPPLE.
            for system in ("Darwin", "Linux"):
                build = root / ("build " + system)
                subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                                "-G", "Unix Makefiles", f"-DCMAKE_SYSTEM_NAME={system}"],
                               check=True, capture_output=True, text=True)
                for tag in ("macosx_11_0_arm64", "macosx_14_0_arm64"):
                    env = dict(os.environ, TEST_PLATFORM_TAG=tag, MACOSX_DEPLOYMENT_TARGET="11.0")
                    result = subprocess.run(["cmake", "--build", str(build), "--target", "pip-package"],
                                            env=env, capture_output=True, text=True)
                    should_fail = system == "Darwin" and tag == "macosx_14_0_arm64"
                    self.assertEqual(result.returncode != 0, should_fail,
                                     result.stdout + result.stderr)
                    if should_fail:
                        self.assertNotIn("pip wheel created at", result.stdout)
                    else:
                        self.assertIn("pip wheel created at", result.stdout)
                        self.assertEqual("satisfies macOS" in result.stderr, system == "Darwin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
