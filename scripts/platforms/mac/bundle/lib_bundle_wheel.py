#!/usr/bin/env python3

from __future__ import annotations

import argparse
import filecmp
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import multiprocessing

_BUNDLE_DIR = Path(__file__).resolve().parent
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

from bundle_slim import should_skip_cuda_runtime_lib

logger = logging.getLogger(__name__)

# Be sure to use system codesign and not one embedded into the conda env
CODESIGN_FULL_PATH = "/usr/bin/codesign"

# Mach-O constants (mach-o/loader.h, mach-o/fat.h).
_MH_MAGIC_LE = b"\xce\xfa\xed\xfe"
_MH_MAGIC_64_LE = b"\xcf\xfa\xed\xfe"
_FAT_MAGIC = 0xCAFEBABE
_FAT_MAGIC_64 = 0xCAFEBABF
_LC_VERSION_MIN_MACOSX = 0x24
_LC_BUILD_VERSION = 0x32
_PLATFORM_MACOS = 1


def select_embedded_libraries(candidates, preferred: dict[str, Path]) -> dict[str, Path]:
    """Map each lib/ destination name to exactly one source file.

    Different rpaths can resolve the same @rpath name to different files:
    conda's libomp.dylib for our own libraries and torch/lib/libomp.dylib
    for the bundled libtorch_cpu.dylib. Copying both into lib/ let the set
    iteration order (the per-process string hash seed) pick the survivor,
    so the shipped libomp and the wheel platform tag (bdist_wheel raises it
    to the highest Mach-O minimum macOS version: torch's copy requires
    14.0) changed between builds. Byte-identical copies are interchangeable;
    a name with a build-linked file in `preferred` takes that file; any
    other conflict fails the packaging instead of overwriting silently.

    Args:
    ----
        candidates: source files that would be copied into lib/.
        preferred: destination name -> the file the build linked.

    Returns:
    -------
        dict[str, Path]: destination name -> selected source file.

    """
    by_name: dict[str, set[Path]] = {}
    for lib in candidates:
        by_name.setdefault(lib.name, set()).add(lib)
    selected = {}
    for name, paths in sorted(by_name.items()):
        # Sorted so that the choice never depends on set iteration order.
        ordered = sorted(paths)
        if all(filecmp.cmp(ordered[0], other, shallow=False) for other in ordered[1:]):
            selected[name] = ordered[0]
            continue
        linked = preferred.get(name)
        matches = [p for p in ordered if linked is not None and os.path.samefile(p, linked)]
        if not matches:
            raise RuntimeError(
                f"CCWheelBundler: {len(ordered)} different files resolve to "
                f"lib/{name}: {[str(p) for p in ordered]}; only one can be "
                f"embedded and none is the build-linked library "
                f"({linked}). Make the consumers resolve the same file.")
        selected[name] = matches[0]
        logger.warning("lib/%s: embedding build-linked %s instead of %s", name, matches[0],
                       [str(p) for p in ordered if p != matches[0]])
    return selected


def _decode_macho_version(value: int) -> tuple[int, int, int]:
    """Decode the xxxx.yy.zz nibble encoding of Mach-O versions."""
    return value >> 16, (value >> 8) & 0xFF, value & 0xFF


def _thin_min_macos_versions(stream, head: bytes) -> list[tuple[int, int, int]]:
    """Minimum macOS versions of the thin Mach-O whose first 8 bytes are head."""
    if head[:4] not in (_MH_MAGIC_LE, _MH_MAGIC_64_LE):
        return []
    header_size = 32 if head[:4] == _MH_MAGIC_64_LE else 28
    header = head + stream.read(header_size - len(head))
    if len(header) < header_size:
        return []
    ncmds, sizeofcmds = struct.unpack_from("<II", header, 16)
    cmds = stream.read(sizeofcmds)
    versions = []
    offset = 0
    for _ in range(ncmds):
        if offset + 8 > len(cmds):
            break
        cmd, size = struct.unpack_from("<II", cmds, offset)
        if cmd == _LC_BUILD_VERSION and offset + 16 <= len(cmds):
            platform, minos = struct.unpack_from("<II", cmds, offset + 8)
            if platform == _PLATFORM_MACOS:
                versions.append(_decode_macho_version(minos))
        elif cmd == _LC_VERSION_MIN_MACOSX and offset + 12 <= len(cmds):
            versions.append(_decode_macho_version(struct.unpack_from("<I", cmds, offset + 8)[0]))
        if size < 8:
            break
        offset += size
    return versions


def macho_min_macos_versions(stream) -> list[tuple[int, int, int]]:
    """Minimum macOS version of every Mach-O slice in a seekable stream.

    Empty for non-Mach-O content and for slices that declare none. Only the
    headers and load commands are read (a few KB per slice), so multi-
    hundred-MB wheel members are never loaded into memory.
    """
    head = stream.read(8)
    if len(head) < 8:
        return []
    magic, nfat = struct.unpack(">II", head)
    if magic not in (_FAT_MAGIC, _FAT_MAGIC_64):
        return _thin_min_macos_versions(stream, head)
    # Java class files share 0xCAFEBABE; their "slice count" is the class
    # file version (>= 45), far above any real number of architectures.
    if not 0 < nfat <= 20:
        return []
    entry, offset_format = (20, ">I") if magic == _FAT_MAGIC else (32, ">Q")
    table = stream.read(nfat * entry)
    if len(table) < nfat * entry:
        return []
    versions = []
    # Ascending offsets: a compressed zip member seeks backward by re-reading.
    for offset in sorted(struct.unpack_from(offset_format, table, i * entry + 8)[0] for i in range(nfat)):
        stream.seek(offset)
        versions += _thin_min_macos_versions(stream, stream.read(8))
    return versions


def check_wheel(wheel: Path, deployment_target: str) -> list[str]:
    """Return the violations of the macOS deployment-target contract.

    bdist_wheel silently raises the platform tag to the highest Mach-O
    minimum version it finds, and the CI test runner uses the newest macOS,
    so a wheel that cannot install on the declared target only shows up as
    a warning. Fail on both the cause (a Mach-O above the target) and the
    effect (a tag other than the target's).
    """
    major, minor = (int(part) for part in (deployment_target.split(".") + ["0"])[:2])
    target = (major, minor, 0)
    # The wheel tag drops the minor version from macOS 11 on.
    expected = f"macosx_{major}_{minor if major == 10 else 0}"
    errors = []
    platform_tag = wheel.name[:-len(".whl")].split("-")[-1]
    if not re.fullmatch(rf"{expected}_(arm64|x86_64|universal2)", platform_tag):
        errors.append(f"wheel platform tag {platform_tag} is not {expected}_<arch> "
                      f"(MACOSX_DEPLOYMENT_TARGET={deployment_target})")
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            with archive.open(info) as member:
                for version in macho_min_macos_versions(member):
                    if version > target:
                        errors.append(f"{info.filename} requires macOS "
                                      f"{'.'.join(map(str, version))} > {deployment_target}")
    return errors


def _default_deployment_target() -> str:
    """The base macOS version bdist_wheel starts from (same precedence)."""
    return os.environ.get("MACOSX_DEPLOYMENT_TARGET") or sysconfig.get_platform().split("-")[1]


class CCWheelBundleConfig:
    output_dependencies: bool

    cpu_bin_path: Path
    cuda_bin_path: Path
    extra_pathlib: Path
    lib_path: Path
    install_path: Path
    bin_path_list: list
    bin_abs_path_list: list
    bin_list: list
    cpu_bin_list: list
    cuda_bin_list: list
    bin_name_list: list

    signature: str

    def __init__(
            self,
            install_path: Path,
            extra_pathlib: Path,
            signature: str,
            output_dependencies: bool,
            preferred_libs: dict[str, Path] | None = None,
    ) -> None:
        """Construct a configuration.

        Args:
        ----
            install_path (str): Path where CC is "installed".
            extra_pathlib (str): A Path where additional libs can be found.
            output_dependencies (bool): boolean that control the level of debug. If true some extra
            files will be created (macos_bundle_warnings.json macos_bundle_dependencies.json).
            preferred_libs (dict[str, Path]): lib/ name -> build-linked file that wins a
            same-name conflict (see select_embedded_libraries).

        """
        self.signature = signature
        self.preferred_libs = preferred_libs or {}
        self.install_path = install_path
        self.extra_pathlib = extra_pathlib
        self.output_dependencies = output_dependencies
        self.lib_path = install_path / "lib"
        self.cpu_bin_path = install_path / "cpu"
        self.cuda_bin_path = install_path / "cuda"
        self.bin_path_list = []
        self.bin_abs_path_list = []

        if self.cpu_bin_path.exists():
            self.cpu_bin_list = list(self.cpu_bin_path.iterdir())
            self.bin_path_list.append(self.cpu_bin_path)
        else:
            self.cpu_bin_list = []
        if self.cuda_bin_path.exists():
            self.cuda_bin_list = list(self.cuda_bin_path.iterdir())
            self.bin_path_list.append(self.cuda_bin_path)
        else:
            self.cuda_bin_list = []
            
        for bin_path in self.cpu_bin_list:
            self.bin_abs_path_list.append(bin_path)
        for bin_path in self.cuda_bin_list:
            self.bin_abs_path_list.append(bin_path)
        self.bin_name_list = [os.path.basename(bin_path) for bin_path in self.bin_abs_path_list]

    def __str__(self) -> str:
        """Return a string representation of the class."""
        res = (
            f"--- lib path: {self.lib_path} \n"
            f" --- cpu path: {self.cpu_bin_path} \n"
            f" --- cuda path: {self.cuda_bin_path} \n"
        )
        return res


class CCWheelBundler:
    config: CCWheelBundleConfig

    # dictionary of lib dependencies : key depends on (list of libs) (not recursive)
    dependencies: dict[str, list[str]] = dict()
    warnings: dict[str, list[str]] = dict()

    def __init__(self, config: CCWheelBundleConfig) -> None:
        """Construct a CCWheelBundler object"""
        self.config = config
        
    @staticmethod
    def _remove_signature(path: Path) -> None:
        """Remove signature of a binary file.

        Call `codesign` utility via a subprocess

        Args:
        ----
            path (Path): The path to the file to sign.

        """
        subprocess.run([CODESIGN_FULL_PATH, "--remove-signature", str(path)], stdout=subprocess.PIPE, check=True)

    def _add_signature(self, path: Path) -> None:
        """Sign a binary file.

        Call `codesign` utility via a subprocess. The signature stored
        into the `config` object is used to sign the binary.

        Args:
        ----
            path (Path): The path to the file to sign.

        """
        dummy_signature = "-"
        if len(self.config.signature) != 0:
             dummy_signature=self.config.signature
        subprocess.run(
            [
                CODESIGN_FULL_PATH,
                "--force",
                "-s",
                dummy_signature,
                "--timestamp",
                str(path),
            ],
            stdout=subprocess.PIPE,
            check=True,
        )

    def sign(self) -> int:
        logger.info("Collect libs in the bundle")
        so_generator = self.config.install_path.rglob("*.so")
        dylib_generator = self.config.install_path.rglob("*.dylib")
        all_libs = set(list(so_generator) + list(dylib_generator))
        # codesign cannot act through symlinked names, and a concurrent worker
        # rewriting the link target (libggml.dylib -> libggml.0.dylib -> real,
        # created by CopyGgmlBackends.cmake COPY_CORE_SHARED) makes the link
        # transiently dangling, failing remove-signature with "No such file
        # or directory" (run 39601603941 macos-14 Build wheel). The real file
        # behind each link is collected separately, so filtering symlinks
        # here loses no signature; the SOVERSION links stay in the wheel for
        # runtime @rpath resolution.
        all_libs = {lib for lib in all_libs if not lib.is_symlink()}
        # create the process pool
        process_pool = multiprocessing.Pool()

        # Remove signature in all embedded libs
        logger.info("Remove cloudViewer old signatures")
        process_pool.map(CCWheelBundler._remove_signature, all_libs)

        logger.info("Sign cloudViewer dynamic libraries")
        process_pool.map(self._add_signature, all_libs)
        return 0

    def bundle(self) -> None:
        """Bundle the dependencies into the .app"""
        libs_found, libs_ex_found, libs_in_plugins = self._collect_dependencies()
        self._embed_libraries(libs_found, libs_ex_found, libs_in_plugins)

        # output debug files if needed
        if self.config.output_dependencies:
            logger.info("write debug files (macos_bundle_dependencies.json and macos_bundle_warnings.json)")
            with open(
                    Path.cwd() / "macos_bundle_dependencies.json",
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(self.dependencies, f, sort_keys=True, indent=4)

            with open(
                    Path.cwd() / "macos_bundle_warnings.json",
                    "w",
                    encoding="utf-8",
            ) as f:
                json.dump(self.warnings, f, sort_keys=True, indent=4)

    @staticmethod
    def _get_install_name(mainlib: Path) -> str | None:
        """Return the LC_ID_DYLIB install name of a dylib (otool -D).

        Executables carry no install name; otool -D then prints only the
        '<path>:' header and None is returned.

        Args:
        ----
            mainlib (Path): Path to a binary (lib, executable)

        Returns:
        -------
            str | None: install name (e.g. '@rpath/libfoo.dylib') or None

        """
        with subprocess.Popen(["otool", "-D", str(mainlib)], stdout=subprocess.PIPE) as proc:
            lines = proc.stdout.readlines()
        if len(lines) < 2:
            return None
        vals = lines[1].split()
        if not vals:
            return None
        return vals[0].decode()

    def _get_lib_dependencies(self, mainlib: Path) -> tuple[list[str], list[str]]:
        """List dependencies of mainlib (using otool -L).

        We only look for dependencies with @rpath and @executable_path.
        We consider @executable_path being relative to the ACloudViewer executable.
        We keep record and debug /usr and /System for debug purposes.

        Args:
        ----
            mainlib (Path): Path to a binary (lib, executable)

        Returns:
        -------
            libs (list[Path]): lib @rpath or @executable_path
            lib_ex (list[(Path, Path)]): lib @executable_path

        """
        libs: list[Path] = []
        lib_ex: list[Path] = []
        warning_libs = []
        install_name = self._get_install_name(mainlib)
        with subprocess.Popen(["otool", "-L", str(mainlib)], stdout=subprocess.PIPE) as proc:
            lines = proc.stdout.readlines()
            logger.debug(mainlib)
            lines.pop(0)  # Drop the first line as it contains the name of the lib / binary
            # The first remaining dylib record is LC_ID_DYLIB: the library's
            # OWN install name repeated by otool -L, not a dependency. Every
            # @rpath-installed dylib therefore used to sprout a fake
            # self-dependency here, silently ignored until the missing-
            # dependency guard started rejecting unresolvable single-component
            # names (observed as 'cannot resolve cloudViewer_torch_ops.dylib'
            # on the macOS wheel CI). Filter it out with otool -D's answer.
            for line in lines:
                vals = line.split()
                if len(vals) < 2:
                    continue
                pathlib = vals[0].decode()
                if install_name is not None and pathlib == install_name:
                    logger.debug("%s: skipping own install name %s", mainlib, pathlib)
                    continue
                logger.debug("->pathlib: %s", pathlib)
                if pathlib == self.config.extra_pathlib:
                    logger.info("%s lib from additional extra pathlib", mainlib)
                    libs.append(Path(pathlib))
                    continue
                dirs = pathlib.split("/")
                # TODO: should be better with startswith
                # we are likely to have only @rpath values
                if dirs[0] == "@rpath":
                    libs.append(Path(dirs[1]))
                elif dirs[0] == "@loader_path":
                    logger.warning("%s declares a dependencies with @loader_path, this won't be resolved", mainlib)
                elif dirs[0] == "@executable_path":
                    logger.warning("%s declares a dependencies with @executable_path", mainlib)
                    # TODO: check if mainlib is in the bundle in order to be sure that
                    # the executable path is relative to the application

                    lib_ex.append(
                        (
                            mainlib.name,
                            Path(pathlib[len("@executable_path/"):]),
                        ),
                    )
                elif (dirs[1] != "usr") and (dirs[1] != "System"):
                    logger.warning("%s depends on undeclared pathlib: %s", mainlib, pathlib)
            self.warnings[str(mainlib)] = str(warning_libs)
            self.dependencies[mainlib.name] = str(libs)
        return libs, lib_ex

    @staticmethod
    def _get_rpath(binary_path: Path) -> list[str]:
        """Retrieve paths stored in LC_RPATH part of the binary.

        Paths are expected to be in the form @loader_path/xxx, @executable_path/xxx, or abs/relative paths

        Args:
        ----
            binary_path (Path): Path to a binary (lib, executable)

        Returns:
        -------
        list[str]: rpath list (string representation)

        """
        rpaths = []
        with subprocess.Popen(["otool", "-l", str(binary_path)], stdout=subprocess.PIPE) as proc:
            lines = proc.stdout.readlines()
            for line in lines:
                res = line.decode()
                vals = res.split()
                if len(vals) > 1 and vals[0] == "path":
                    rpaths.append(vals[1])
        return rpaths

    @staticmethod
    def _convert_rpaths(binary_path: Path, rpaths: list[str]) -> list[Path]:
        """Convert rpaths to absolute paths.

        Given a path to a binary (lib, executable) and a list of rpaths, resolve rpaths
        and append binary_path to them in order to create putative absolute path to this binary

        Args:
        ----
            binary_path (Path): string representation of the path to a binary (lib, executable)

            rpaths (list[str]): List of string representation of rpaths

        Returns:
        -------
            list[Path]: list of putative full / absolute path to the binary

        """
        dirname_binary = binary_path.parent
        abs_paths = []
        for rpath in rpaths:
            if "@loader_path" in rpath:
                vals = rpath.split("/")
                abs_path = dirname_binary
                if len(vals) > 1:
                    abs_path = (abs_path / Path("/".join(vals[1:]))).resolve()
            else:
                # TODO: test if it's an absolute path
                abs_path = Path(rpath).resolve()
            abs_paths.append(abs_path)
        return abs_paths

    def _collect_dependencies(self):
        """Collect dependencies of ACloudViewer binary and QT libs

        Returns
        -------
            set[Path]: Libs and binaries found in the collect process.
            set[(Path, Path)]: Libs and binaries found with an @executable_path dependency.
            set[Path]: Libs and binaries found in the plugin dir.

        """
        # Searching for CC dependencies
        libs_to_check = []

        # results
        libs_found = set()  # Abs path of libs/binaries already checked, candidate for embedding in the bundle
        lib_ex_found = set()
        libs_in_plugins = set()

        logger.info("Adding cpu or cuda so to the libs to check")
        for bin_path in self.config.bin_path_list:
            for file_path in bin_path.iterdir():
                libs_to_check.append(file_path)

        logger.info("Adding libs or plugins already available in lib to the libsToCheck")
        for lib_dir in self.config.lib_path.iterdir():
            if lib_dir.is_dir():
                for file in lib_dir.iterdir():
                    if file.is_file() and file.suffix in (".dylib", ".so"):
                        libs_to_check.append(file)
                        libs_in_plugins.add(file)
            elif lib_dir.is_file() and (lib_dir.suffix == ".so" or lib_dir.suffix == ".dylib"):
                libs_to_check.append(lib_dir)

        logger.info("number of libs in PlugIns directory: %i", len(libs_in_plugins))
        logger.info("number of libs already in Frameworks directory: %i", len(libs_to_check))

        logger.info("searching for dependencies...")
        while len(libs_to_check):
            # --- Unstack a binary/lib from the libs_to_check array
            lib2check = libs_to_check.pop(0)

            # If the lib was already processed we continue, of course
            if lib2check in libs_found:
                continue

            # Add the current lib to the already processed libs
            libs_found.add(lib2check)

            # search for @rpath and @executable_path dependencies in the current lib
            lib_deps, lib_ex = self._get_lib_dependencies(lib2check)

            # @executable_path are handled in a seperate set
            lib_ex_found.update(lib_ex)

            # TODO: group these two functions since we do not need
            # get all rpath for the current lib
            rpaths_str = CCWheelBundler._get_rpath(lib2check)
            # get absolute path from found rpath
            abs_search_paths = CCWheelBundler._convert_rpaths(lib2check, rpaths_str)

            # If the extra_pathlib is not already added, we ad it
            # TODO:: there is no way it can be False
            # maybe we should prefer to check for authorized lib_dir
            # TODO: if rpath is @loader_path, LIB is either in frameworks (already embedded) or in extra_pathlib
            # we can take advantage of that...
            if self.config.extra_pathlib not in abs_search_paths:
                abs_search_paths.append(self.config.extra_pathlib)
            # Libraries already staged in the wheel payload (e.g. the
            # source-built OIIO dylibs pre-copied by make_python_package.cmake,
            # like the ggml backend modules) are legitimate resolution
            # targets for @rpath dependencies of other payload members.
            if self.config.lib_path not in abs_search_paths:
                abs_search_paths.append(self.config.lib_path)

            # TODO: check if exists, else throw and exception
            for dependency in lib_deps:
                abslib_path = None
                for abs_rp in abs_search_paths:
                    candidate = abs_rp / dependency
                    if candidate.is_file():
                        abslib_path = candidate
                        break
                # dependency comes from _get_lib_dependencies as a Path;
                # `in` on a Path raises TypeError, so test its string form.
                if abslib_path is None and "/" not in str(dependency):
                    # An unresolvable @rpath dependency would be silently
                    # missing from the wheel and crash at import time
                    # ("Library not loaded: @rpath/..."). Fail the
                    # packaging here instead, matching the packaging guards
                    # in make_python_package.cmake. Multi-component @rpath
                    # paths (e.g. Foo.framework/Versions/A/Foo) keep the
                    # legacy silent skip.
                    raise RuntimeError(
                        f"CCWheelBundler: cannot resolve @rpath dependency "
                        f"'{dependency}' of {lib2check} from its rpaths "
                        f"{rpaths_str} plus extra pathlib "
                        f"{self.config.extra_pathlib}; the built wheel "
                        f"would fail at import. Make sure the dependency is "
                        f"built or installed into a directory listed on the "
                        f"consuming binary's build rpath.")
                if abslib_path is not None:
                    if should_skip_cuda_runtime_lib(abslib_path):
                        logger.info("Skip NVIDIA CUDA runtime dependency: %s", abslib_path)
                    elif abslib_path not in libs_to_check and abslib_path not in libs_found:
                        # if this lib was not checked for dependencies yet, we append it to the list of lib to check
                        libs_to_check.append(abslib_path)

            # TODO: handle lib_ex here
            # for dependency in lib_ex:...
            # TODO: add to libTOcheck executable_path/dep

        # ggml backend modules are dlopen'd at runtime and NOT in the otool
        # dependency chain.  They are pre-copied into lib/ by CMake's
        # CopyGgmlBackends.cmake, but we still need to traverse their
        # dependencies so they get bundled.
        for lib in list(libs_found):
            if lib.name.startswith("libAICore"):
                src_dir = lib.parent
                for ggml_mod in list(src_dir.glob("libggml-*.so")) + list(src_dir.glob("libggml-*.dylib")):
                    if ggml_mod not in libs_found:
                        if should_skip_cuda_runtime_lib(ggml_mod):
                            continue
                        logger.info("Adding ggml backend module: %s", ggml_mod)
                        libs_found.add(ggml_mod)
                break

        return libs_found, lib_ex_found, libs_in_plugins

    def _embed_libraries(
            self,
            libs_found: set[Path],
            lib_ex_found: set[(Path, Path)],
            libs_in_plugins: set[Path],
    ) -> None:
        """Embed collected libraries into the `.app` bundle.

        rpath of embedded libs is modified to match their new location

        Args:
        ----
            libs_found (set[Path]): libs and binaries found in the collect process.
            libs_ex_found (set[(Path, Path)]): libs and binaries found with an @executable_path dependency.
            libs_found (set[Path]): libs and binaries found in the plugin dir.

        """
        logger.info("Copying libraries")
        logger.info("lib_ex_found to add to Frameworks: %i", len(lib_ex_found))
        logger.info("libs_found to add to Frameworks: %i", len(libs_found))

        libs_in_frameworks = set(self.config.lib_path.iterdir())

        libs_to_copy = []
        for lib in libs_found:
            if lib in self.config.cpu_bin_list or lib in self.config.cuda_bin_list:
                continue
            if should_skip_cuda_runtime_lib(lib):
                logger.info("Skip NVIDIA CUDA runtime in wheel lib/: %s", lib)
                continue
            base = self.config.lib_path / lib.name
            if base not in libs_in_frameworks and (lib not in libs_in_plugins):
                libs_to_copy.append(lib)
        selected = select_embedded_libraries(libs_to_copy, self.config.preferred_libs)
        for lib in selected.values():
            shutil.copy2(lib, self.config.lib_path)
        logger.info("number of libs added to lib: %i", len(selected))

        logger.info(" --- Qt PlugIns libs: add rpath to lib, number of qt libs: %i", len(libs_in_plugins))
        for file in libs_in_plugins:
            if file.is_file():
                subprocess.run(
                    ["install_name_tool", "-add_rpath", "@loader_path/../../lib", str(file)],
                    stdout=subprocess.PIPE,
                    check=False,
                )

        logger.info(" --- cuda or cpu libs: add rpath to lib, number of libs: %i", len(self.config.bin_abs_path_list))
        for file in self.config.bin_abs_path_list:
            if file.is_file() and file.suffix in (".so", ".dylib"):
                subprocess.run(
                    ["install_name_tool", "-add_rpath", "@loader_path/../lib", str(file)],
                    stdout=subprocess.PIPE,
                    check=False,
                )

        # --- ajout des rpath pour les libraries
        logger.info(" --- Frameworks libs: add rpath to Frameworks")
        nb_frameworks_libs = 0
        # TODO: purge old rpath
        for file in self.config.lib_path.iterdir():
            if file.is_file() and file.suffix in (".so", ".dylib"):
                nb_frameworks_libs += 1
                subprocess.run(
                    ["install_name_tool", "-add_rpath", "@loader_path", str(file)],
                    stdout=subprocess.PIPE,
                    check=False,
                )

        # TODO: make a function for this
        # Embed libs with an @executable_path dependencies
        for lib_ex in lib_ex_found:
            base = lib_ex[0]
            target = lib_ex[1]
            if base in self.config.bin_name_list:
                continue

            framework_path = self.config.lib_path / base

            if framework_path.is_file():
                base_path = framework_path
            else:
                # This should not be possible
                raise Exception("no base path")
                sys.exit(1)

            logger.info("modify : @executable_path -> @rpath: %s", base_path)

            subprocess.run(
                [
                    "install_name_tool",
                    "-change",
                    "@executable_path/" + str(target),
                    "@rpath/" + str(target),
                    str(base_path),
                ],
                stdout=subprocess.PIPE,
                check=False,
            )


if __name__ == "__main__":
    # configure logger
    formatter = " CCWheelBundler::%(levelname)-8s:: %(message)s"
    logging.basicConfig(level=logging.INFO, format=formatter)
    std_handler = logging.StreamHandler()

    # CLI parser
    parser = argparse.ArgumentParser("CCWheelBundler")
    parser.add_argument(
        "install_path",
        help="Path where the cloudViewer python package is installed (CMake install dir)",
        type=Path,
        nargs="?",
    )
    parser.add_argument(
        "--openmp_library",
        help="libomp the build linked; embedded when other same-named copies are reachable",
        type=Path,
    )
    parser.add_argument(
        "--check_wheel",
        help="Only verify a built wheel (or the newest wheel in a directory) against "
        "MACOSX_DEPLOYMENT_TARGET and exit",
        type=Path,
    )
    parser.add_argument(
        "--extra_pathlib",
        help="Extra path to find libraries (default to $CONDA_PREFIX/lib)",
        type=Path,
    )
    parser.add_argument(
        "--output_dependencies",
        help="Output a json files in order to debug dependency graph",
        action="store_true",
    )
    parser.add_argument(
        "--signature",
        help="Signature to use for code signing (or will use ACLOUDVIEWER_BUNDLE_SIGN var)",
        type=str,
        default="",
    )

    arguments = parser.parse_args()

    if arguments.check_wheel is not None:
        wheel = arguments.check_wheel
        if wheel.is_dir():
            wheels = sorted(wheel.glob("*.whl"), key=lambda p: p.stat().st_mtime)
            if not wheels:
                logger.error("no wheel found in %s", wheel)
                sys.exit(1)
            wheel = wheels[-1]
        deployment_target = _default_deployment_target()
        violations = check_wheel(wheel, deployment_target)
        for violation in violations:
            logger.error("%s: %s", wheel.name, violation)
        if violations:
            sys.exit(1)
        logger.info("%s satisfies macOS %s", wheel.name, deployment_target)
        sys.exit(0)
    if arguments.install_path is None:
        parser.error("install_path is required unless --check_wheel is given")

    signature = os.environ.get("ACLOUDVIEWER_BUNDLE_SIGN")
    if signature is None:
        logger.warning(
            "ACLOUDVIEWER_BUNDLE_SIGN variable is undefined. Please define it or use the `signature` argument.",
        )
        signature = arguments.signature

    logger.debug("Signature: %s", signature)  # Could be dangerous to display this

    # convert extra_pathlib to absolute paths
    if arguments.extra_pathlib is not None:
        extra_pathlib = arguments.extra_pathlib.resolve()
    else:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix is not None:
            extra_pathlib = (Path(conda_prefix) / "lib").resolve()
        else:
            logger.error(
                "Unable to find CONDA_PREFIX system variable, please run this script inside a conda environment or use the extra_pathlib option.",
            )
            sys.exit(1)

    preferred_libs = {}
    if arguments.openmp_library is not None:
        preferred_libs[arguments.openmp_library.name] = arguments.openmp_library

    config = CCWheelBundleConfig(
        arguments.install_path,
        extra_pathlib,
        signature,
        arguments.output_dependencies,
        preferred_libs,
    )

    bundler = CCWheelBundler(config)
    bundler.bundle()
    if Path(CODESIGN_FULL_PATH).exists():
        sys.exit(bundler.sign())
