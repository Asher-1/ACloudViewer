"""Build-artifact discovery and dynamic-loader lifecycle contracts."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import runtime_environment as runtime


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_linked_dependencies_preserve_spaces_and_skip_unresolved_libraries(self):
        output = ("\tlibAICore.so => /opt/Cloud Viewer/lib/libAICore.so (0x00007f00)\n"
                  "\tlibggml.so => /opt/lib/libggml.so (0x00007f10)\n"
                  "\tlibmissing.so => not found\n")
        with mock.patch.object(runtime.sys, "platform", "linux"), \
                mock.patch.object(runtime.subprocess, "check_output", return_value=output):
            self.assertEqual(runtime.linked_library_paths(Path("probe"), {}),
                             [Path("/opt/Cloud Viewer/lib/libAICore.so"),
                              Path("/opt/lib/libggml.so")])

    def test_macho_rpaths_skip_cmdsize_and_ignore_other_load_commands(self):
        output = """Load command 0
          cmd LC_RPATH
      cmdsize 48
         path @loader_path/../Frameworks (offset 12)
Load command 1
          cmd LC_LOAD_DYLIB
      cmdsize 64
         path /must/not/be/an/rpath (offset 24)
Load command 2
          cmd LC_RPATH
      cmdsize 80
         path /opt/Qt 6/lib (offset 12)
"""
        with mock.patch.object(runtime.shutil, "which", return_value="otool"), \
                mock.patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout=output)):
            self.assertEqual(runtime._macho_runtime_dirs(Path("/app/bin/probe")),
                             [Path("/app/bin/../Frameworks"), Path("/opt/Qt 6/lib")])

    def test_windows_configuration_selection_applies_to_dlls_and_executables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config in ("Release", "Debug"):
                target = root / "bin" / config
                target.mkdir(parents=True)
                (target / "AICore.dll").touch()
                (target / "probe.exe").touch()
            (root / "CMakeCache.txt").write_text(
                "CMAKE_CONFIGURATION_TYPES:STRING=Debug;Release\n")
            # A stale flat artifact must not override the selected config.
            (root / "bin/AICore.dll").touch()
            with mock.patch.object(runtime, "is_windows", return_value=True), \
                    mock.patch.dict(os.environ, {"AICORE_BUILD_CONFIG": "Debug"}):
                self.assertEqual(runtime.resolve_aicore_library(root),
                                 root / "bin/Debug/AICore.dll")
                self.assertEqual(runtime.find_binary(root, "probe"),
                                 root / "bin/Debug/probe.exe")
                self.assertEqual(runtime.find_binary(root, "probe.exe"),
                                 root / "bin/Debug/probe.exe")
                self.assertEqual(runtime.find_build_tool(root, "aicore_tests/probe"),
                                 root / "bin/Debug/probe.exe")
                self.assertEqual(runtime.venv_python(root), root / "Scripts/python.exe")
            with mock.patch.object(runtime, "is_windows", return_value=True), \
                    mock.patch.dict(os.environ, {"AICORE_BUILD_CONFIG": ""}):
                self.assertEqual(runtime.resolve_aicore_library(root),
                                 root / "bin/Release/AICore.dll")

    def test_windows_loader_retains_handles_and_cleans_up_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handle = mock.Mock()
            library = SimpleNamespace()
            with mock.patch.object(runtime, "is_windows", return_value=True), \
                    mock.patch.object(runtime.os, "add_dll_directory", create=True,
                                      return_value=handle) as register, \
                    mock.patch.object(runtime.ctypes, "CDLL", return_value=library) as load, \
                    mock.patch.dict(os.environ, {"PATH": str(root)}):
                loaded = runtime.load_shared_library(root / "AICore.dll")
                self.assertEqual(loaded._aicore_dll_directories, [handle])
                register.assert_called_once_with(str(root))
                handle.close.assert_not_called()
                load.side_effect = OSError("missing dependency")
                with self.assertRaisesRegex(OSError, "missing dependency"):
                    runtime.load_shared_library(root / "AICore.dll")
                handle.close.assert_called_once()

    def test_loader_environment_preserves_inherited_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {runtime.loader_variable(): "inherited"}
            runtime.prepend_runtime_paths(env, [root, root / "missing"])
            self.assertEqual(env[runtime.loader_variable()],
                             str(root) + os.pathsep + "inherited")

    def test_windows_reference_filters_both_native_and_lib_prefixed_dlls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "clean"
            clean.mkdir()
            candidates = []
            for name in ("AICore.dll", "libAICore.dll", "ggml-cpu.dll", "libggml.dll"):
                path = root / name.replace(".dll", "")
                path.mkdir()
                (path / name).touch()
                candidates.append(path)
            path = os.pathsep.join(map(str, [clean, *candidates]))
            self.assertEqual(runtime.windows_reference_path(path), [str(clean)])

    def test_reference_isolation_preserves_system_path_and_original_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "system"
            candidate = root / "candidate"
            system.mkdir()
            candidate.mkdir()
            (candidate / "ggml-cpu.dll").touch()
            environment = {"PATH": os.pathsep.join(map(str, (system, candidate))),
                           "GGML_BACKEND_PATH": "candidate", "LD_PRELOAD": "foreign.so",
                           "DYLD_INSERT_LIBRARIES": "foreign.dylib", "LD_LIBRARY_PATH": "foreign"}
            original = environment.copy()
            with mock.patch.object(runtime, "is_windows", return_value=True):
                reference = runtime.reference_runtime_environment(environment)
                self.assertEqual(reference["PATH"], str(system))
                self.assertNotIn("GGML_BACKEND_PATH", reference)
                self.assertNotIn("LD_PRELOAD", reference)
                self.assertNotIn("DYLD_INSERT_LIBRARIES", reference)
            with mock.patch.object(runtime, "is_windows", return_value=False), \
                    mock.patch.object(runtime.sys, "platform", "linux"):
                reference = runtime.reference_runtime_environment(environment)
                self.assertNotIn("LD_LIBRARY_PATH", reference)
                self.assertEqual(reference["PATH"], original["PATH"])
            self.assertEqual(environment, original)

    def test_preloads_use_host_library_names_and_keep_dependency_order(self):
        root = Path("torch-runtime")
        with mock.patch.object(runtime, "is_windows", return_value=False), \
                mock.patch.object(runtime.sys, "platform", "darwin"), \
                mock.patch.object(runtime, "load_shared_library", side_effect=["c10", "torch"]) as load:
            self.assertEqual(runtime.preload_shared_libraries(root, ("c10", "torch")),
                             ["c10", "torch"])
            self.assertEqual(load.call_args_list, [
                mock.call(root / "libc10.dylib", global_symbols=True),
                mock.call(root / "libtorch.dylib", global_symbols=True)])
        with mock.patch.object(runtime, "is_windows", return_value=True), \
                mock.patch.object(runtime, "load_shared_library") as load:
            self.assertEqual(runtime.preload_shared_libraries(root, ("c10",)), [])
            load.assert_not_called()

    def test_lookup_and_load_preserve_symlink_backend_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            binary = staging / runtime.aicore_library_names()[0]
            binary.touch()
            deployed = root / "bin"
            deployed.mkdir()
            link = deployed / binary.name
            try:
                link.symlink_to(binary)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            self.assertEqual(runtime.resolve_aicore_library(root), link)
            with mock.patch.object(runtime, "is_windows", return_value=False), \
                    mock.patch.object(runtime.ctypes, "CDLL", return_value=SimpleNamespace()) as load:
                runtime.load_shared_library(link)
                self.assertEqual(load.call_args.args[0], str(link))


if __name__ == "__main__":
    unittest.main()
