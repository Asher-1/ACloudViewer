#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import multiprocessing

logger = logging.getLogger(__name__)

# Be sure to use system codesign and not one embedded into the conda env
CODESIGN_FULL_PATH = "/usr/bin/codesign"


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
    ) -> None:
        """Construct a configuration.

        Args:
        ----
            install_path (str): Path where CC is "installed".
            extra_pathlib (str): A Path where additional libs can be found.
            output_dependencies (bool): boolean that control the level of debug. If true some extra
            files will be created (macos_bundle_warnings.json macos_bundle_dependencies.json).

        """
        self.signature = signature
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
        with subprocess.Popen(["otool", "-L", str(mainlib)], stdout=subprocess.PIPE) as proc:
            lines = proc.stdout.readlines()
            logger.debug(mainlib)
            lines.pop(0)  # Drop the first line as it contains the name of the lib / binary
            # now first line is LC_ID_DYLIB (should be @rpath/libname)
            for line in lines:
                vals = line.split()
                if len(vals) < 2:
                    continue
                pathlib = vals[0].decode()
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

            # TODO: check if exists, else throw and exception
            for dependency in lib_deps:
                for abs_rp in abs_search_paths:
                    abslib_path = abs_rp / dependency
                    if abslib_path.is_file():
                        if abslib_path not in libs_to_check and abslib_path not in libs_found:
                            # if this lib was not checked for dependencies yet, we append it to the list of lib to check
                            libs_to_check.append(abslib_path)
                        break

            # TODO: handle lib_ex here
            # for dependency in lib_ex:...
            # TODO: add to libTOcheck executable_path/dep

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

        nb_libs_added = 0
        for lib in libs_found:
            if lib in self.config.cpu_bin_list or lib in self.config.cuda_bin_list:
                continue
            base = self.config.lib_path / lib.name
            if base not in libs_in_frameworks and (lib not in libs_in_plugins):
                shutil.copy2(lib, self.config.lib_path)
                nb_libs_added += 1
        logger.info("number of libs added to lib: %i", nb_libs_added)

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

    config = CCWheelBundleConfig(
        arguments.install_path,
        extra_pathlib,
        signature,
        arguments.output_dependencies,
    )

    bundler = CCWheelBundler(config)
    bundler.bundle()
    if Path(CODESIGN_FULL_PATH).exists():
        sys.exit(bundler.sign())
