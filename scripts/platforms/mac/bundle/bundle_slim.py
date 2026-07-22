"""Shared slimming rules for macOS bundle scripts.

Aligned with:
  - scripts/platforms/linux/pack_ubuntu.sh (CUDA runtime exclusion)
  - scripts/platforms/windows/pack_windows.ps1 (Should-Filter)
  - plugins/core/Standard/qPythonRuntime/cmake/Helpers.cmake (minimal python env)
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# NVIDIA CUDA toolkit runtimes — do not embed (large, version-locked).
# GPU features require a matching CUDA install on the target Mac (if used).
CUDA_RUNTIME_GLOBS = (
    "libcuda.dylib",
    "libcuda.*.dylib",
    "libcudart*.dylib",
    "libcublas*.dylib",
    "libcufft*.dylib",
    "libcurand*.dylib",
    "libcusolver*.dylib",
    "libcusparse*.dylib",
    "libnpp*.dylib",
    "libnvrtc*.dylib",
    "libcudnn*.dylib",
    "libculibos*.dylib",
    "libnvjpeg*.dylib",
    "libnvToolsExt*.dylib",
)

# Dev-site-packages bloat (torch, jupyter, …) — skip when copying full env.
PYTHON_BLOAT_PACKAGES = frozenset(
    {
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "tensorboard",
        "keras",
        "jax",
        "jupyter",
        "jupyterlab",
        "notebook",
        "ipython",
        "scipy",
        "pandas",
        "matplotlib",
        "sklearn",
        "scikit_learn",
        "cv2",
        "opencv_python",
        "cloudViewer",
        "cloudviewer",
        "nvidia",
        "triton",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REQUIREMENTS_RELEASE = (
    _REPO_ROOT / "plugins/core/Standard/qPythonRuntime/requirements-release.txt"
)


def should_skip_cuda_runtime_lib(path: Path) -> bool:
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in CUDA_RUNTIME_GLOBS)


def _python_tree_ignore(_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".tox", "htmlcov"):
            ignored.add(name)
        if name.endswith((".pyc", ".pyo")):
            ignored.add(name)
    return ignored


def load_release_packages() -> list[str]:
    packages: list[str] = []
    if not _REQUIREMENTS_RELEASE.is_file():
        logger.warning("requirements-release.txt not found: %s", _REQUIREMENTS_RELEASE)
        return packages
    for line in _REQUIREMENTS_RELEASE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split("[", 1)[0].strip()
        name = token.split(">=", 1)[0].split("==", 1)[0].split("<", 1)[0].strip()
        if name:
            packages.append(name)
    return packages


def required_runtime_imports() -> tuple[str, ...]:
    return ("numpy", "pip", "setuptools", "tqdm", "invoke", "typing_extensions")


def _package_paths(src: Path, pkg: str) -> list[Path]:
    paths: list[Path] = []
    pkg_dir = src / pkg
    pkg_py = src / f"{pkg}.py"
    if pkg_dir.is_dir():
        paths.append(pkg_dir)
    elif pkg_py.is_file():
        paths.append(pkg_py)
    paths.extend(sorted(src.glob(f"{pkg}-*.dist-info")))
    return paths


def verify_release_site_packages(src: Path) -> None:
    missing: list[str] = []
    for pkg in load_release_packages():
        if pkg == "pybind11":
            continue
        if pkg == "wheel":
            continue
        if not _package_paths(src, pkg):
            missing.append(pkg)
    if missing:
        raise RuntimeError(
            "Missing site-packages for minimal python bundle: "
            f"{', '.join(missing)}. Install:\n"
            "  python -m pip install -r "
            "plugins/core/Standard/qPythonRuntime/requirements-release.txt"
        )


def copy_libpython_into_bundle(base_prefix: Path, dest_libpath: Path) -> list[Path]:
    """Copy libpython shared library next to the embedded python lib tree."""
    copied: list[Path] = []
    lib_dir = base_prefix / "lib"
    if not lib_dir.is_dir():
        logger.warning("No lib/ under Python prefix: %s", base_prefix)
        return copied
    dest_libpath.mkdir(parents=True, exist_ok=True)
    for pattern in ("libpython*.dylib", "libpython*.so*"):
        for lib in sorted(lib_dir.glob(pattern)):
            target = dest_libpath / lib.name
            shutil.copy2(lib, target)
            logger.info("Copied libpython: %s -> %s", lib, target)
            copied.append(target)
    if not copied:
        logger.warning("libpython not found under %s", lib_dir)
    return copied


def copy_release_site_packages(src: Path, dest: Path) -> None:
    verify_release_site_packages(src)
    dest.mkdir(parents=True, exist_ok=True)
    for pkg in load_release_packages():
        if pkg == "pybind11":
            continue
        for pkg_src in _package_paths(src, pkg):
            target = dest / pkg_src.name
            if pkg_src.is_dir():
                logger.info("Minimal python: copy package %s", pkg_src.name)
                shutil.copytree(
                    pkg_src,
                    target,
                    ignore=_python_tree_ignore,
                    dirs_exist_ok=True,
                )
            else:
                logger.info("Minimal python: copy module %s", pkg_src.name)
                shutil.copy2(pkg_src, target)
    numpy_libs = src / "numpy.libs"
    if numpy_libs.is_dir():
        shutil.copytree(numpy_libs, dest / "numpy.libs", dirs_exist_ok=True)


def copy_python_stdlib(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    skip_dirs = {"site-packages", "test", "idle_test", "ensurepip"}
    for item in src.iterdir():
        if item.name in skip_dirs:
            continue
        if item.name.startswith("config-"):
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=_python_tree_ignore, dirs_exist_ok=True)
        elif item.suffix != ".a":
            shutil.copy2(item, target)


def copy_python_env_minimal(
    base_python_libs: Path,
    base_python_binary: Path,
    embedded_python_lib: Path,
    embedded_python_binary: Path,
    embedded_python_libpath: Path,
    python_version_name: str,
) -> None:
    """Copy stdlib + requirements-release.txt packages only."""
    logger.info(
        "Minimal python env: stdlib from %s + release packages from site-packages",
        base_python_libs,
    )
    site_packages_src = base_python_libs / "site-packages"
    site_packages_dest = embedded_python_lib / "site-packages"

    copy_python_stdlib(base_python_libs, embedded_python_lib)
    copy_release_site_packages(site_packages_src, site_packages_dest)

    copy_libpython_into_bundle(base_python_libs.parent.parent, embedded_python_libpath)

    embedded_python_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_python_binary, embedded_python_binary)

    _create_symlink(
        f"{python_version_name}/site-packages",
        "site-packages",
        embedded_python_libpath,
    )


def _is_bloat_site_package(name: str) -> bool:
    for bloat in PYTHON_BLOAT_PACKAGES:
        if name == bloat or name.startswith(f"{bloat}-") or name.startswith(f"{bloat}."):
            return True
    return False


def copy_python_env_filtered(
    base_python_libs: Path,
    base_python_binary: Path,
    embedded_python_lib: Path,
    embedded_python_binary: Path,
    embedded_python_libpath: Path,
    python_version_name: str,
) -> None:
    """Copy stdlib + site-packages minus known bloat packages."""
    logger.info("Filtered python env: stdlib + site-packages (bloat excluded) from %s", base_python_libs)
    site_packages_src = base_python_libs / "site-packages"
    site_packages_dest = embedded_python_lib / "site-packages"

    copy_python_stdlib(base_python_libs, embedded_python_lib)

    site_packages_dest.mkdir(parents=True, exist_ok=True)
    if site_packages_src.is_dir():
        verify_release_site_packages(site_packages_src)
        for item in site_packages_src.iterdir():
            if _is_bloat_site_package(item.name):
                logger.info("Skip bloat site-package: %s", item.name)
                continue
            target = site_packages_dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, ignore=_python_tree_ignore, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)

    embedded_python_binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_python_binary, embedded_python_binary)
    copy_libpython_into_bundle(base_python_libs.parent.parent, embedded_python_libpath)
    _create_symlink(f"{python_version_name}/site-packages", "site-packages", embedded_python_libpath)


def _create_symlink(source: str, target: str, working_directory: Path) -> None:
    link_path = working_directory / target
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    original_directory = os.getcwd()
    try:
        os.chdir(working_directory)
        os.symlink(source, target)
        logger.info("symlink created: %s -> %s", target, source)
    except OSError as exc:
        logger.error("Failed to create symlink %s -> %s: %s", target, source, exc)
        raise
    finally:
        os.chdir(original_directory)
