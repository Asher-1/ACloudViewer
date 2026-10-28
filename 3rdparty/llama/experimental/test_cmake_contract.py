"""Configure the real dependency entries without fetching/building sources."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
LLAMA = ROOT / "3rdparty/llama/llama_cpp.cmake"
# The 0.21.0 standalone ggml entry was retired with the 0.27 re-anchoring;
# only its absence is contracted now.
GGML = ROOT / "3rdparty/ggml/ggml.cmake"
PREAMBLE = '''cmake_minimum_required(VERSION 3.19)
project(SourceOwnerContract LANGUAGES C CXX)
set(CloudViewer_INSTALL_LIB_DIR lib)
set(CLOUDVIEWER_EXTERNAL_INSTALL_DIR "${CMAKE_BINARY_DIR}/install")
set(CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set(GGML_ENABLED ON CACHE BOOL "")
set(GGML_BUILD_SHARED ON CACHE BOOL "")
set(GGML_USE_SAM3D_OPS OFF CACHE BOOL "")
foreach(backend IN ITEMS CUDA VULKAN METAL SYCL OPENCL)
    set(GGML_USE_${backend} OFF CACHE BOOL "")
endforeach()
'''


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class SourceOwnerTests(unittest.TestCase):
    def test_application_uses_llama_as_its_production_source_owner(self):
        dependencies = (ROOT / "3rdparty/find_dependencies.cmake").read_text()
        self.assertIn("include(${CloudViewer_3RDPARTY_DIR}/llama/llama_cpp.cmake)",
                      dependencies)
        self.assertNotIn("include(${CloudViewer_3RDPARTY_DIR}/ggml/ggml.cmake)",
                         dependencies)

    def test_validation_driver_forwards_production_visibility_to_external_build(self):
        with tempfile.TemporaryDirectory(prefix="llama-driver-contract-") as directory:
            build = Path(directory) / "build"
            result = subprocess.run(
                ["cmake", "-S", str(LLAMA.parent / "experimental"), "-B", str(build),
                 "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Inspect the command passed to ExternalProject, not just the
            # outer cache: otherwise these flags can silently stop at the driver.
            command = (build / "llama/tmp/ext_llama-cfgcmd.txt").read_text()
            for option in ("CMAKE_POLICY_DEFAULT_CMP0063:STRING=NEW",
                           "CMAKE_C_VISIBILITY_PRESET=hidden",
                           "CMAKE_CXX_VISIBILITY_PRESET=hidden",
                           "CMAKE_CUDA_VISIBILITY_PRESET=hidden",
                           "CMAKE_VISIBILITY_INLINES_HIDDEN=ON",
                           "GGML_CPU_ALL_VARIANTS=ON"):
                self.assertIn("-D" + option, command)
            cache = (build / "CMakeCache.txt").read_text()
            self.assertIn("GGML_CPU_ALL_VARIANTS:BOOL=ON", cache)
            self.assertFalse(any(file.is_file() for file in (build / "downloads").rglob("*")))

    def test_validation_driver_rejects_stale_cpu_variant_off_cache(self):
        with tempfile.TemporaryDirectory(prefix="llama-driver-variants-") as directory:
            build = Path(directory) / "build"
            result = subprocess.run(
                ["cmake", "-S", str(LLAMA.parent / "experimental"), "-B", str(build),
                 "-DGGML_CPU_ALL_VARIANTS=OFF", "-DBUILD_TESTING=OFF"],
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("requires GGML_CPU_ALL_VARIANTS=ON", result.stdout + result.stderr)
            self.assertFalse((build / "llama").exists())
            self.assertFalse(any(file.is_file() for file in (build / "downloads").rglob("*")))

    def configure(self, text, *options):
        directory = tempfile.TemporaryDirectory(prefix="llama-owner-contract-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "CMakeLists.txt").write_text(PREAMBLE + text)
        result = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"),
                                 *options], capture_output=True, text=True)
        return result, root / "build"

    def test_llama_owns_the_only_ggml_and_preserves_configure_stamp(self):
        result, build = self.configure(f'include("{LLAMA}")\n'
            'if(NOT TARGET ext_llama OR TARGET 3rdparty_ggml OR '
            'NOT TARGET 3rdparty_llama OR TARGET ext_ggml OR '
            'NOT GGML_DEPENDENCY_TARGET STREQUAL "3rdparty_llama" OR '
            'NOT GGML_EXTERNAL_TARGET STREQUAL "ext_llama")\n'
            'message(FATAL_ERROR "wrong source ownership")\nendif()\n')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        signature = build / "llama-patch-signature.txt"
        identity = (signature.read_bytes(), signature.stat().st_mtime_ns)
        repeat = subprocess.run(["cmake", "-S", str(build.parent), "-B", str(build)],
                                capture_output=True, text=True)
        self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
        self.assertEqual(identity, (signature.read_bytes(), signature.stat().st_mtime_ns))
        self.assertFalse(any(path.is_file() for path in (build / "downloads").rglob("*")))

    def test_incomplete_accelerators_fail_before_fetch(self):
        for backend in ("METAL", "SYCL", "OPENCL"):
            with self.subTest(backend=backend):
                result, build = self.configure(f'include("{LLAMA}")\n',
                                               f"-DGGML_USE_{backend}=ON")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"{backend} patch migration/validation is incomplete",
                              result.stdout + result.stderr)
                self.assertFalse((build / "llama").exists())

    def test_metal_validation_requires_apple_and_an_exclusive_backend(self):
        for apple, cuda, expected in (
                (False, False, "requires an Apple platform and SDK"),
                (True, True, "Validate Metal separately from CUDA/Vulkan")):
            with self.subTest(apple=apple, cuda=cuda):
                # This exercises configure guards only, without pretending
                # that setting APPLE supplies an SDK or validates a shader.
                result, build = self.configure(
                    f'set(APPLE {"TRUE" if apple else "FALSE"})\n'
                    f'include("{LLAMA}")\n',
                    "-DGGML_USE_METAL=ON", "-DLLAMA_METAL_MIGRATION_VALIDATION=ON",
                    f'-DGGML_USE_CUDA={"ON" if cuda else "OFF"}')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout + result.stderr)
                self.assertFalse((build / "llama").exists())

    def test_metal_validation_selects_its_manifest_and_external_option(self):
        result, build = self.configure(
            'set(APPLE TRUE)\n'
            f'include("{LLAMA}")\n'
            'if(NOT _GGML_OWNER_PATCH_MANIFEST MATCHES "/metal-manifest.yaml$" '
            'OR TARGET ext_ggml OR TARGET 3rdparty_ggml)\n'
            'message(FATAL_ERROR "wrong Metal source or manifest")\nendif()\n',
            "-DGGML_USE_METAL=ON", "-DLLAMA_METAL_MIGRATION_VALIDATION=ON")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = (build / "llama/tmp/ext_llama-cfgcmd.txt").read_text()
        self.assertIn("-DGGML_METAL=ON", command)
        self.assertFalse(any(path.is_file() for path in (build / "downloads").rglob("*")))

    @unittest.skipUnless(shutil.which("nvcc"), "CUDA toolkit is required")
    def test_cuda_selects_its_operator_manifest_without_a_second_owner(self):
        result, build = self.configure(f'include("{LLAMA}")\n'
            'if(TARGET ext_ggml OR TARGET 3rdparty_ggml OR '
            'NOT TARGET ext_llama OR NOT TARGET 3rdparty_llama)\n'
            'message(FATAL_ERROR "CUDA created a second source owner")\nendif()\n'
            'if(NOT _GGML_OWNER_PATCH_MANIFEST MATCHES "/cuda-manifest.yaml$")\n'
            'message(FATAL_ERROR "CUDA operator manifest not selected")\nendif()\n',
            "-DGGML_USE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        command = (build / "llama/tmp/ext_llama-cfgcmd.txt").read_text()
        self.assertRegex(command, r"-DCMAKE_CUDA_COMPILER(:FILEPATH|:STRING)?=[^ ;]*nvcc")
        self.assertFalse(any(path.is_file() for path in (build / "downloads").rglob("*")))

    @unittest.skipUnless(shutil.which("glslc"), "Vulkan shader compiler is required")
    def test_vulkan_and_combined_manifests_preserve_one_source_owner(self):
        modes = [(False, "vulkan-manifest.yaml")]
        if shutil.which("nvcc"):
            modes.append((True, "cuda-vulkan-manifest.yaml"))
        for cuda, manifest in modes:
            with self.subTest(cuda=cuda):
                result, build = self.configure(f'include("{LLAMA}")\n'
                    'if(TARGET ext_ggml OR TARGET 3rdparty_ggml OR '
                    'NOT TARGET ext_llama OR NOT TARGET 3rdparty_llama)\n'
                    'message(FATAL_ERROR "Vulkan created a second source owner")\nendif()\n'
                    f'if(NOT _GGML_OWNER_PATCH_MANIFEST MATCHES "/{manifest}$")\n'
                    'message(FATAL_ERROR "wrong GPU operator manifest")\nendif()\n',
                    "-DGGML_USE_VULKAN=ON", f"-DGGML_USE_CUDA={'ON' if cuda else 'OFF'}",
                    "-DCMAKE_CUDA_ARCHITECTURES=86")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(any(p.is_file() for p in (build / "downloads").rglob("*")))

    def test_cuda_manifests_allow_sam3d_attention_scratch_allocation(self):
        patch_dir = LLAMA.parent / "llama-patches"
        for name in ("cuda-manifest.yaml", "cuda-vulkan-manifest.yaml"):
            with self.subTest(manifest=name):
                manifest = (patch_dir / name).read_text()
                self.assertIn("0002-acv-cuda-operators.patch", manifest)
                self.assertIn("0007-acv-sam3d-allocation-size.patch", manifest)
                self.assertLess(manifest.index("0002-acv-cuda-operators.patch"),
                                manifest.index("0007-acv-sam3d-allocation-size.patch"))

    def test_vulkan_manifests_route_q4_k_quantized_copy(self):
        patch_dir = LLAMA.parent / "llama-patches"
        for name in ("vulkan-manifest.yaml", "cuda-vulkan-manifest.yaml"):
            with self.subTest(manifest=name):
                manifest = (patch_dir / name).read_text()
                self.assertIn("0005-acv-vulkan-operators.patch", manifest)
                self.assertIn("0008-acv-vulkan-q4-k-copy-route.patch", manifest)
                self.assertLess(manifest.index("0005-acv-vulkan-operators.patch"),
                                manifest.index("0008-acv-vulkan-q4-k-copy-route.patch"))

    def test_patch_content_refreshes_only_its_generated_owner(self):
        with tempfile.TemporaryDirectory(prefix="llama-patch-input-") as directory:
            thirdparty = Path(directory)
            entry = thirdparty / "llama"
            shutil.copytree(LLAMA.parent, entry)
            result, build = self.configure(f'include("{entry / "llama_cpp.cmake"}")\n')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            signature = (build / "llama-patch-signature.txt").read_bytes()
            generated = [build / "llama/src" / name / "stale-marker" for name in
                         ("ext_llama", "ext_llama-build", "ext_llama-stamp")]
            preserved = [build / "ggml/src/ext_ggml/keep",
                         build / "downloads/llama/keep.tar.gz",
                         build / "install/lib/keep.so"]
            for file in generated + preserved:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("retain unrelated data")
            patch = next((entry / "llama-patches").glob("*.patch"))
            patch.write_bytes(patch.read_bytes() + b"\n")
            repeat = subprocess.run(["cmake", "-S", str(build.parent), "-B", str(build)],
                                    capture_output=True, text=True)
            self.assertEqual(repeat.returncode, 0, repeat.stdout + repeat.stderr)
            self.assertNotEqual(signature, (build / "llama-patch-signature.txt").read_bytes())
            self.assertTrue(all(not file.exists() for file in generated))
            self.assertTrue(all(file.exists() for file in preserved))

    def test_requested_vulkan_cannot_silently_become_cpu_only(self):
        result, build = self.configure(f'include("{LLAMA}")\n',
            "-DGGML_USE_VULKAN=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_Vulkan=TRUE")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requested Vulkan backend dependencies are missing",
                      result.stdout + result.stderr)
        self.assertFalse((build / "llama").exists())
        self.assertFalse(any(p.is_file() for p in (build / "downloads").rglob("*")))

    def test_retired_ggml_entry_stays_absent(self):
        self.assertFalse(GGML.exists(),
                         "the 0.21.0 ggml entry must stay retired (0.27 anchoring)")
        dependencies = (ROOT / "3rdparty/find_dependencies.cmake").read_text()
        self.assertNotIn("ggml/ggml.cmake", dependencies)

    def test_llama_does_not_adopt_unidentified_ggml_installation(self):
        for relative in ("lib/libggml.so.0", "bin/ggml-cpu.dll", "include/ggml.h"):
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory(prefix="llama-old-install-") as directory:
                    prefix = Path(directory)
                    stale = prefix / relative
                    stale.parent.mkdir(parents=True, exist_ok=True)
                    stale.write_bytes(b"previous installation")
                    result, build = self.configure(
                        f'set(CLOUDVIEWER_EXTERNAL_INSTALL_DIR "{prefix}")\n'
                        f'include("{LLAMA}")\n')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("unidentified ggml installation", result.stdout + result.stderr)
                    self.assertEqual(stale.read_bytes(), b"previous installation")
                    self.assertFalse((prefix / ".acloudviewer-ggml-source-owner").exists())
                    self.assertFalse((build / "llama").exists())

    def test_changed_archive_cannot_reuse_installed_source_identity(self):
        with tempfile.TemporaryDirectory(prefix="llama-other-archive-") as directory:
            thirdparty = Path(directory)
            entry = thirdparty / "llama"
            shutil.copytree(LLAMA.parent, entry)
            pin = entry / LLAMA.name
            result, build = self.configure(f'include("{entry / LLAMA.name}")\n'
                                           'add_custom_target(source_identity_probe)\n')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            marker = build / "install/.acloudviewer-ggml-source-owner"
            identity = marker.read_bytes()
            signature = (build / "llama-patch-signature.txt").read_bytes()
            pin.write_text(re.sub(r'set\(LLAMA_SHA256 "[a-f0-9]{64}"\)',
                                  'set(LLAMA_SHA256 "' + "0" * 64 + '")',
                                  pin.read_text()))
            # Build a target with no archive dependency. The pin change must
            # trigger regeneration and rejection without an explicit configure.
            repeat = subprocess.run(["cmake", "--build", str(build), "--target",
                                     "source_identity_probe"], capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertIn("ggml source identity differs", repeat.stdout + repeat.stderr)
            self.assertEqual(identity, marker.read_bytes())
            self.assertEqual(signature, (build / "llama-patch-signature.txt").read_bytes())

    def test_disabling_inference_creates_no_source_owner(self):
        result, build = self.configure(f'include("{LLAMA}")\n'
            'if(TARGET ext_llama OR TARGET 3rdparty_ggml OR TARGET 3rdparty_llama)\n'
            'message(FATAL_ERROR "disabled dependency was created")\nendif()\n',
            "-DGGML_ENABLED=OFF")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((build / "llama-patch-signature.txt").exists())
        self.assertFalse((build / "install/.acloudviewer-ggml-source-owner").exists())


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class RuntimeCopyTests(unittest.TestCase):
    """Execute the production copy script; filenames do not simulate an SDK."""

    SCRIPT = ROOT / "core/AICore/cmake/CopyGgmlBackends.cmake"

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="llama runtime copy ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source with spaces"
        self.dest = self.root / "package with spaces"
        self.source.mkdir()
        self.dest.mkdir()

    def run_copy(self, prefix, suffix, **options):
        return subprocess.run(
            ["cmake", f"-DGGML_BACKEND_DIR={self.source}",
             f"-DDEST_DIR={self.dest}", f"-DLIB_PREFIX={prefix}",
             f"-DLIB_SUFFIX={suffix}",
             *[f"-D{key}={value}" for key, value in options.items()],
             "-P", str(self.SCRIPT)], capture_output=True, text=True)

    def create_core(self, prefix, suffix):
        for name in ("ggml", "ggml-base", "llama", "mtmd"):
            link = f"{prefix}{name}{suffix}"
            if suffix == ".dll":
                (self.source / link).write_bytes(name.encode())
            else:
                current = (f"{prefix}{name}.0.27.0{suffix}" if suffix == ".dylib"
                           else f"{link}.0.27.0")
                stale = (f"{prefix}{name}.0.21.0{suffix}" if suffix == ".dylib"
                         else f"{link}.0.21.0")
                (self.source / current).write_bytes(name.encode())
                (self.source / stale).write_bytes(b"stale")
                (self.source / link).symlink_to(current)

    def test_core_copy_preserves_current_aliases_on_all_filename_layouts(self):
        # Windows DLLs have no aliases; POSIX hosts can exercise both alias
        # layouts without requiring Windows developer-mode symlink privileges.
        layouts = (("", ".dll"),) if os.name == "nt" else (
            ("lib", ".so"), ("lib", ".dylib"), ("", ".dll"))
        for prefix, suffix in layouts:
            with self.subTest(suffix=suffix):
                self.create_core(prefix, suffix)
                result = self.run_copy(prefix, suffix, COPY_CORE_SHARED="ON")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for name in ("ggml", "ggml-base", "llama", "mtmd"):
                    link = self.dest / f"{prefix}{name}{suffix}"
                    self.assertEqual(link.read_bytes(), name.encode())
                    if suffix != ".dll":
                        self.assertTrue(link.is_symlink())
                        self.assertEqual(link.resolve().parent, self.dest)
                self.assertFalse(any("0.21.0" in p.name for p in self.dest.iterdir()))

    def test_missing_mtmd_is_a_hard_failure(self):
        for name in ("ggml", "ggml-base", "llama"):
            (self.source / f"{name}.dll").write_bytes(name.encode())
        result = self.run_copy("", ".dll", COPY_CORE_SHARED="ON")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mtmd.dll", result.stderr)

    def test_cpu_variants_and_requested_gpu_modules_are_copied(self):
        for name in ("ggml-cpu-x64.dll", "ggml-cpu-avx2.dll", "ggml-vulkan.dll"):
            (self.source / name).write_bytes(name.encode())
        result = self.run_copy("", ".dll", COPY_BACKEND_MODULES="ON",
                               CPU_ALL_VARIANTS="ON", GPU_FILES_PIPE="ggml-vulkan.dll")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({p.name for p in self.dest.iterdir()},
                         {p.name for p in self.source.iterdir()})

    def test_configured_missing_gpu_is_a_hard_failure(self):
        (self.source / "ggml-cpu.dll").write_bytes(b"cpu")
        result = self.run_copy("", ".dll", COPY_BACKEND_MODULES="ON",
                               GPU_FILES_PIPE="ggml-vulkan.dll")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configured GPU backend is missing", result.stderr)

    def test_install_template_runs_both_modes_with_space_containing_paths(self):
        self.create_core("", ".dll")
        for name in ("ggml-cpu-x64.dll", "ggml-cpu-avx2.dll", "ggml-vulkan.dll"):
            (self.source / name).write_bytes(name.encode())
        template = ROOT / "core/AICore/cmake/InstallGgmlBackends.cmake.in"
        values = {
            "_AICORE_COPY_GGML_SCRIPT": self.SCRIPT.as_posix(),
            "_AICORE_GGML_INSTALL_SUBDIR": "bin",
            "GGML_RUNTIME_LIB_DIR": self.source.as_posix(),
            "GGML_BACKEND_DIR": self.source.as_posix(),
            "CMAKE_SHARED_LIBRARY_PREFIX": "",
            "CMAKE_SHARED_LIBRARY_SUFFIX": ".dll",
            "GGML_MODULE_SUFFIX": ".dll",
            "_AICORE_GPU_PIPE": "ggml-vulkan.dll",
            "_AICORE_CPU_ALL_VARIANTS": "ON",
            "CMAKE_COMMAND": Path(shutil.which("cmake")).as_posix(),
        }
        script = self.root / "install runtime.cmake"
        script.write_text(re.sub(r"@(\w+)@", lambda m: values[m[1]],
                                 template.read_text()))
        result = subprocess.run(
            ["cmake", f"-DCMAKE_INSTALL_PREFIX={self.dest}", "-P", str(script)],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual({p.name for p in (self.dest / "bin").iterdir()},
                         {p.name for p in self.source.iterdir()})
        for source in self.source.iterdir():
            self.assertEqual((self.dest / "bin" / source.name).read_bytes(),
                             source.read_bytes())

    @unittest.skipUnless(os.name == "posix" and os.geteuid() != 0,
                         "Requires POSIX non-root write permission checks")
    def test_failed_copy_cannot_report_a_successful_package(self):
        self.create_core("", ".dll")
        (self.source / "ggml-cpu.dll").write_bytes(b"cpu")
        self.dest.chmod(0o555)
        try:
            for mode in ("COPY_CORE_SHARED", "COPY_BACKEND_MODULES"):
                with self.subTest(mode=mode):
                    result = self.run_copy("", ".dll", **{mode: "ON"})
                    self.assertNotEqual(result.returncode, 0,
                                        result.stdout + result.stderr)
        finally:
            self.dest.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
