"""Exercise real Linux CUDA bundling with small ELF fixtures, not model inference."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/platforms/linux/bundle_cuda_runtime.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux ELF bundler")


def compile_library(directory, name, symbol, dependencies=(), rpath=None):
    compiler = shutil.which("cc")
    assert compiler, "Linux bundler regression requires a C compiler"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / (name + ".c")
    body = "\n".join("extern int " + entry[2] + "(void);" for entry in dependencies)
    calls = " + ".join(entry[2] + "()" for entry in dependencies) or "1"
    source.write_text(body + "\nint " + symbol + "(void) { return " + calls + "; }\n")
    library = directory / name
    command = [compiler, "-shared", "-fPIC", str(source), "-Wl,-soname," + name, "-o", str(library)]
    for parent, dependency, _symbol in dependencies:
        command.extend(["-L" + str(parent), "-l:" + dependency])
    if rpath:
        command.append("-Wl,-rpath," + str(rpath))
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library


def bundle(module, destination, extra=""):
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)
    return subprocess.run(["bash", str(SCRIPT), str(module), str(destination), str(extra)],
                          env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("directory_name", ["runtime", "runtime with spaces"])
def test_resolved_ldd_address_is_not_part_of_library_path(tmp_path, directory_name):
    runtime = tmp_path / directory_name
    cuda = compile_library(runtime, "libcudart.so.99", "cuda_fixture")
    driver = compile_library(runtime, "libcuda.so.99", "driver_fixture")
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(runtime, cuda.name, "cuda_fixture"),
                              (runtime, driver.name, "driver_fixture")], rpath=runtime)
    destination = tmp_path / "bundle"
    result = bundle(module, destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in destination.iterdir()} == {"libcudart.so.99"}
    assert hashlib.sha256((destination / cuda.name).read_bytes()).digest() == hashlib.sha256(cuda.read_bytes()).digest()


def test_extra_search_paths_close_unresolved_transitive_dependencies(tmp_path):
    runtime = tmp_path / "runtime"
    leaf = compile_library(runtime, "libcublasLt.so.99", "leaf_fixture")
    parent = compile_library(runtime, "libcublas.so.99", "parent_fixture",
                             [(runtime, leaf.name, "leaf_fixture")])
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(runtime, parent.name, "parent_fixture")])
    destination = tmp_path / "bundle"
    result = bundle(module, destination, runtime)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in destination.iterdir()} == {"libcublas.so.99", "libcublasLt.so.99"}
    for original in (parent, leaf):
        assert (destination / original.name).read_bytes() == original.read_bytes()


def test_one_copied_runtime_does_not_hide_a_missing_dependency(tmp_path):
    runtime = tmp_path / "runtime"
    leaf = compile_library(tmp_path / "unavailable", "libcublasLt.so.99", "leaf_fixture")
    parent = compile_library(runtime, "libcublas.so.99", "parent_fixture",
                             [(leaf.parent, leaf.name, "leaf_fixture")])
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(runtime, parent.name, "parent_fixture")])
    destination = tmp_path / "bundle"
    result = bundle(module, destination, runtime)
    assert result.returncode != 0, "An incomplete runtime bundle must not be reported as successful"
    assert "libcublasLt.so.99" in result.stderr


def test_same_library_referenced_twice_is_copied_once(tmp_path):
    runtime = tmp_path / "runtime"
    leaf = compile_library(runtime, "libcublasLt.so.99", "leaf_fixture")
    parent = compile_library(runtime, "libcublas.so.99", "parent_fixture",
                             [(runtime, leaf.name, "leaf_fixture")])
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(runtime, parent.name, "parent_fixture"),
                              (runtime, leaf.name, "leaf_fixture")])
    destination = tmp_path / "bundle"
    result = bundle(module, destination, runtime)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("Bundling CUDA runtime:") == 2
    assert {path.name for path in destination.iterdir()} == {"libcublas.so.99", "libcublasLt.so.99"}


def test_reinstall_reuses_a_runtime_already_in_the_destination(tmp_path):
    destination = tmp_path / "bundle with spaces"
    runtime = compile_library(destination, "libcudart.so.99", "runtime_fixture")
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(destination, runtime.name, "runtime_fixture")],
                             rpath=destination)
    original = runtime.read_bytes()
    result = bundle(module, destination)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "are the same file" not in result.stderr
    assert runtime.read_bytes() == original


def test_copy_failure_is_not_reported_as_a_successful_bundle(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    runtime = compile_library(runtime_dir, "libcudart.so.99", "runtime_fixture")
    module = compile_library(tmp_path / "module", "libggml-cuda.so", "module_fixture",
                             [(runtime_dir, runtime.name, "runtime_fixture")],
                             rpath=runtime_dir)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    failing_copy = fake_bin / "cp"
    failing_copy.write_text("#!/bin/sh\nexit 1\n")
    failing_copy.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    result = bundle(module, tmp_path / "bundle")
    assert result.returncode != 0, "A failed copy must fail runtime deployment"


@pytest.mark.parametrize("scan_directory", [False, True])
def test_general_packager_preserves_paths_with_spaces(tmp_path, scan_directory):
    runtime = tmp_path / "torch library directory"
    leaf = compile_library(runtime, "libfixture_leaf.so.1", "leaf_fixture")
    module = compile_library(tmp_path / "module directory", "libaicore_cumesh.so",
                             "module_fixture", [(runtime, leaf.name, "leaf_fixture")],
                             rpath=runtime)
    destination = tmp_path / "installer libraries"
    destination.mkdir()
    target = module.parent if scan_directory else module
    result = subprocess.run(["bash", str(ROOT / "scripts/platforms/linux/pack_ubuntu.sh"),
                             str(target), str(destination), str(runtime)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / leaf.name).is_file(), result.stdout + result.stderr
    assert (destination / leaf.name).read_bytes() == leaf.read_bytes()


def test_application_launcher_preserves_spaced_prefix_and_arguments(tmp_path):
    prefix = tmp_path / "installed viewer"
    (prefix / "lib/cuda-runtime").mkdir(parents=True)
    launcher = prefix / "Viewer fixture.sh"
    shutil.copyfile(ROOT / "scripts/ACloudViewer/ACloudViewer.sh", launcher)
    binary = prefix / "Viewer fixture"
    binary.write_text('#!/bin/sh\nprintf "%s\\n" "$@" "$LD_LIBRARY_PATH"\n')
    binary.chmod(0o755)
    result = subprocess.run(["sh", str(launcher), "--fixture", "input with spaces"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["--fixture", "input with spaces"]
    assert lines[2].split(":")[:3] == [str(prefix / "lib/cuda-runtime"),
                                       str(prefix), str(prefix / "lib")]
