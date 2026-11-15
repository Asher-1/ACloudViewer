"""Build artifacts and native loader services shared by AICore Python tools.

Callers select reference isolation and preload sets; host mechanics live here.
"""

from __future__ import annotations

import os
import sys
import ctypes
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def is_windows() -> bool:
    return os.name == "nt"


def executable_name(name: str) -> str:
    return name + ".exe" if is_windows() and not name.lower().endswith(".exe") else name


def venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if is_windows() else "bin/python")


def build_configurations(build: Path) -> list[str]:
    """List configurations in selection order, preferring an explicit choice."""
    configured: list[str] = []
    single_config: list[str] = []
    try:
        lines = (build / "CMakeCache.txt").read_text(
            encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        variable = key.split(":", 1)[0]
        if variable in ("CMAKE_BUILD_TYPE", "CMAKE_DEFAULT_BUILD_TYPE"):
            if value.strip():
                single_config.append(value.strip())
        elif variable == "CMAKE_CONFIGURATION_TYPES":
            configured.extend(item.strip() for item in value.split(";") if item.strip())
    ordered: list[str] = []
    requested = os.environ.get("AICORE_BUILD_CONFIG", "").strip()
    for config in ([requested] if requested else []) + single_config + [
            "Release", "RelWithDebInfo", "Debug", "MinSizeRel"] + configured:
        if config and config.lower() not in {item.lower() for item in ordered}:
            ordered.append(config)
    return ordered


def runtime_directories(build: Path) -> list[Path]:
    """Candidate directories for flat and CMake multi-config build outputs."""
    bin_dir = build.resolve() / "bin"
    directories: list[Path] = []
    for config in build_configurations(build):
        for path in (bin_dir / config,
                     *(bin_dir / config / group for group in
                       ("aicore_tests", "plugin_test", "colmap_test", "cloudviewer_test", "examples")),
                     *(bin_dir / group / config for group in
                       ("aicore_tests", "plugin_test", "colmap_test", "cloudviewer_test", "examples"))):
            if path not in directories:
                directories.append(path)
    directories.extend([bin_dir / "aicore_tests", bin_dir / "plugin_test", bin_dir])
    return directories


def find_binary(build: Path, name: str) -> Path:
    names = tuple(dict.fromkeys((executable_name(name), name)))
    for directory in runtime_directories(build):
        for filename in names:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()
    return (build / "bin" / "aicore_tests" / names[0]).resolve()


def find_build_tool(build: Path, relative_name: str) -> Path:
    """Find an explicitly grouped tool, accepting legacy and config layouts."""
    relative = Path(relative_name)
    names = tuple(dict.fromkeys((executable_name(relative.name), relative.name)))
    directories = []
    for config in build_configurations(build):
        directories.extend((build / "bin" / config / relative.parent,
                            build / "bin" / relative.parent / config))
    directories.append(build / "bin" / relative.parent)
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    return find_binary(build, relative.name)


def shared_library_names(stem: str) -> tuple[str, ...]:
    if is_windows():
        return (f"{stem}.dll", f"{stem}_d.dll", f"lib{stem}.dll")
    if sys.platform == "darwin":
        return (f"lib{stem}.dylib",)
    return (f"lib{stem}.so",)


def aicore_library_names() -> tuple[str, ...]:
    return shared_library_names("AICore")


def resolve_aicore_library(path: Path) -> Path:
    """Locate a runtime without resolving symlinks used for backend discovery."""
    path = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    if path.is_file():
        return path
    configurations = build_configurations(path)
    directories = [path / "bin" / config for config in configurations]
    directories.extend(path / config for config in configurations)
    directories.extend((path / "bin", path / "lib", path))
    for directory in directories:
        for name in aicore_library_names():
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"AICore library unavailable: {path}")


def loader_variable() -> str:
    """Return the environment variable used by the host dynamic loader."""
    if is_windows():
        return "PATH"
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def load_shared_library(path: Path, *, global_symbols: bool = False,
                        runtime_dirs: Iterable[Path] = ()) -> ctypes.CDLL:
    """Keep Windows DLL search registrations alive with the loaded library.

    CPython's secure Windows loader does not use PATH for ctypes dependencies.
    Register existing PATH entries and explicit directories for that process;
    Unix retains the caller's requested symbol visibility.
    """
    # The lexical load path owns ggml's adjacent-backend search. Resolving a
    # symlink may redirect that search into a staging directory without ggml.
    path = Path(os.path.abspath(os.fspath(path)))
    handles = []
    try:
        if is_windows():
            directories = [path.parent, *runtime_dirs]
            directories.extend(Path(entry) for entry in
                               os.environ.get("PATH", "").split(os.pathsep) if entry)
            seen = set()
            for directory in directories:
                directory = Path(directory).resolve()
                if directory.is_dir() and directory not in seen:
                    handles.append(os.add_dll_directory(str(directory)))
                    seen.add(directory)
        mode = ctypes.RTLD_GLOBAL if global_symbols and not is_windows() else ctypes.DEFAULT_MODE
        library = ctypes.CDLL(str(path), mode=mode)
        library._aicore_dll_directories = handles
        return library
    except BaseException:
        for handle in handles:
            handle.close()
        raise


def windows_reference_path(path: str) -> list[str]:
    """Preserve framework/system paths while excluding candidate runtime DLLs."""
    result = []
    for directory in path.split(os.pathsep):
        if not directory:
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            result.append(directory)
            continue
        if any(name.lower().endswith(".dll") and
               name.lower().startswith(("aicore", "libaicore", "ggml", "libggml"))
               for name in names):
            continue
        result.append(directory)
    return result


def reference_runtime_environment(environment: dict[str, str]) -> dict[str, str]:
    """Exclude the candidate's loader overrides from an independent reference.

    Windows must retain system/framework PATH entries to launch the interpreter;
    Unix can remove its separate library-path override entirely.
    """
    result = environment.copy()
    for variable in ("GGML_BACKEND_PATH", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        result.pop(variable, None)
    variable = loader_variable()
    if is_windows():
        result[variable] = os.pathsep.join(windows_reference_path(result.get(variable, "")))
    else:
        result.pop(variable, None)
    return result


def linked_library_paths(binary: Path, environment: dict[str, str]) -> list[Path]:
    """Inspect resolved ELF dependencies; other hosts have no ldd evidence."""
    if not sys.platform.startswith("linux"):
        return []
    output = subprocess.check_output(["ldd", str(binary)], env=environment, text=True)
    return [Path(path) for path in re.findall(
        r"=>\s+(.+?)\s+\(0x[0-9a-fA-F]+\)", output)]


def preload_shared_libraries(directory: Path, stems: Iterable[str]) -> list[ctypes.CDLL]:
    """Publish a caller-selected Unix dependency chain in the global namespace.

    Windows resolves imports through load_shared_library's DLL directories;
    callers still pass this directory there and retain the returned libraries.
    """
    if is_windows():
        return []
    return [load_shared_library(directory / shared_library_names(stem)[0],
                                global_symbols=True) for stem in stems]


def prepend_runtime_paths(environment: dict[str, str], directories: Iterable[Path]) -> None:
    """Prepend existing runtime directories with the native list separator."""
    variable = loader_variable()
    paths = [str(directory) for directory in directories
             if directory.is_dir()]
    existing = environment.get(variable)
    if existing:
        paths.append(existing)
    environment[variable] = os.pathsep.join(paths)


def configured_qt_library_dir(build: Path) -> Path | None:
    """Recover the configured Qt runtime directory for direct CLI runs.

    CTest injects AICORE_QT_LIBRARY_DIR, but users also invoke this script
    directly. Reading CMakeCache keeps the spawned catalog/probe binaries on
    the same Qt major/minor build instead of mixing the host Qt with the one
    linked into AICore.
    """
    cache = build / "CMakeCache.txt"
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    cmake_dirs: list[Path] = []
    prefixes: list[Path] = []
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        variable = key.split(":", 1)[0]
        if variable in ("Qt5Core_DIR", "Qt6Core_DIR") and value:
            cmake_dirs.append(Path(value))
        elif variable == "CMAKE_PREFIX_PATH":
            prefixes.extend(Path(item) for item in value.split(";") if item)
    candidates: list[Path] = []
    for directory in cmake_dirs:
        if len(directory.parents) >= 2:
            qt_lib = directory.parents[1]
            candidates.extend(([qt_lib.parent / "bin", qt_lib]
                               if is_windows() else [qt_lib]))
    for prefix in prefixes:
        candidates.append(prefix / ("bin" if is_windows() else "lib"))
    return next((path.resolve() for path in candidates if path.is_dir()), None)


def _elf_runtime_dirs(binary: Path) -> list[Path]:
    """Extract DT_RUNPATH/DT_RPATH entries from an ELF binary (Linux)."""
    readelf = shutil.which("readelf")
    if not readelf:
        return []
    try:
        result = subprocess.run([readelf, "-d", str(binary)],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    dirs: list[Path] = []
    for line in result.stdout.splitlines():
        if "RPATH" not in line and "RUNPATH" not in line:
            continue
        match = re.search(r"\[(.*)\]", line)
        if not match:
            continue
        for entry in match.group(1).split(":"):
            entry = entry.replace("$ORIGIN", str(binary.parent))
            entry = entry.replace("${ORIGIN}", str(binary.parent))
            if entry:
                dirs.append(Path(entry))
    return dirs


def _macho_runtime_dirs(binary: Path) -> list[Path]:
    """Extract LC_RPATH entries from a Mach-O binary (macOS)."""
    otool = shutil.which("otool")
    if not otool:
        return []
    try:
        result = subprocess.run([otool, "-l", str(binary)],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    dirs: list[Path] = []
    in_rpath = False
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("cmd "):
            in_rpath = stripped == "cmd LC_RPATH"
        elif in_rpath:
            # LC_RPATH contains cmdsize between cmd and path in otool -l.
            match = re.fullmatch(r"path\s+(.+?)\s+\(offset\s+\d+\)", stripped)
            if match:
                entry = match.group(1).replace("@loader_path", str(binary.parent))
                if entry:
                    dirs.append(Path(entry))
                in_rpath = False
    return dirs


def _qt_core_names() -> tuple[str, ...]:
    if is_windows():
        return ("Qt6Core.dll", "Qt5Core.dll")
    if sys.platform == "darwin":
        return ("libQt6Core.6.dylib", "libQt5Core.5.dylib", "libQt6Core.dylib",
                "libQt5Core.dylib")
    return ("libQt6Core.so.6", "libQt5Core.so.5")


def embedded_qt_library_dir(build: Path) -> Path | None:
    """Resolve the Qt runtime directory from the binaries' embedded paths.

    The loader resolves a shared library's own DT_NEEDED entries with that
    library's DT_RUNPATH, not the executable's, so fronting only the probe
    path still lets libAICore's Qt dependency fall through to a host Qt of a
    different minor version (Qt_5.15 symbol-version failure). The build
    bakes the absolute Qt runtime directory into every linked binary's
    RUNPATH/LC_RPATH, so reading it back from disk is machine-independent
    for in-place builds and needs no hardcoded path anywhere.
    """
    core_names = _qt_core_names()
    bin_dir = build / "bin"
    candidates: list[Path] = []
    if is_windows():
        # PE has no runpath; Qt runtimes are deployed next to the binaries.
        candidates.extend(runtime_directories(build))
    else:
        binaries = [bin_dir / ("libAICore.dylib" if sys.platform == "darwin"
                               else "libAICore.so")]
        tests_dir = bin_dir / "aicore_tests"
        if tests_dir.is_dir():
            for probe in sorted(tests_dir.iterdir()):
                if probe.is_file() and os.access(probe, os.X_OK):
                    binaries.append(probe)
                    break
        for binary in binaries:
            if not binary.is_file():
                continue
            dirs = (_macho_runtime_dirs(binary) if sys.platform == "darwin"
                    else _elf_runtime_dirs(binary))
            candidates.extend(dirs)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            continue
        if any((resolved / name).is_file() for name in core_names):
            return resolved
    return None
