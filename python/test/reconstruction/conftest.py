# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Shared bootstrap for the pycolmap-parity reconstruction test suite.

Loads the freshly built ``pybind`` extension directly from the build tree so
the suite runs in developer environments without an installed package. The
build-tree path mirrors the AICore tooling convention of resolving relative
to the repository root.
"""
import importlib.util
import os
from pathlib import Path

import pytest

# The engine links glog; under pytest's default fd-capture, CHECK failures
# that write their stack trace through the captured fd can double-fault and
# abort the whole run before the real result is reported. Force glog onto
# the real stderr and drop INFO/WARNING chatter BEFORE the extension (and
# thus glog's static initialization) is loaded.
os.environ.setdefault("GLOG_logtostderr", "1")
os.environ.setdefault("GLOG_minloglevel", "3")


def pytest_configure(config):
    # The engine runs RANSAC/ceres work on worker threads while glog owns
    # fd 2; under pytest's fd capture those writes race and can abort the
    # process before the real test result is reported. Run this suite
    # without capture (harmless for the other suites).
    config.option.capture = "no"


import glob as _glob

_REPO = Path(__file__).resolve().parents[3]


def _find_so():
    """Locate the built pybind extension: env override, then the build tree
    (any cpython ABI tag), CUDA and non-CUDA layouts, and Windows .pyd."""
    override = __import__("os").environ.get("ACV_PYBIND_SO")
    if override:
        return Path(override)
    patterns = [
        str(_REPO / "build_app/lib/*/Python/*/pybind*.so"),
        str(_REPO / "build_app/lib/*/Python/*/pybind*.pyd"),
        str(_REPO / "build_app/lib/*/Python/pybind*.so"),
        str(_REPO / "build_app/lib/*/Python/pybind*.pyd"),
        str(_REPO / "build_app/lib/pybind*.so"),
        str(_REPO / "build_app/lib/pybind*.pyd"),
    ]
    for pattern in patterns:
        hits = sorted(_glob.glob(pattern))
        if hits:
            return Path(hits[-1])
    return _REPO / "build_app/lib/Release/Python/cuda/"
    "pybind.cpython-312-x86_64-linux-gnu.so"


_SO = _find_so()


@pytest.fixture(scope="session")
def pybind():
    """The root ``pybind`` extension module (loaded at most once per
    process: pybind11 generic_type registrations are not re-runnable).

    Resolution order:
      1. an already-imported installed package extension
         (``cloudViewer.cuda.pybind`` / ``cloudViewer.cpu.pybind`` - the
         wheel builds carry BUILD_RECONSTRUCTION=ON, so CI lanes that
         install the wheel pick the reconstruction bindings up without
         any special-casing);
      2. the freshly built extension in the build tree (developer
         environments).
    """
    import sys

    # 1) Installed package: the extension is already loaded under its
    # package-qualified name. NEVER spec-load the package-internal file
    # again - a second PyInit on the same .so aborts with a
    # generic_type re-registration error.
    if "cloudViewer" in sys.modules:
        for name in ("cloudViewer.cuda.pybind", "cloudViewer.cpu.pybind"):
            module = sys.modules.get(name)
            if module is not None and hasattr(module, "reconstruction"):
                return module
    else:
        try:
            import cloudViewer  # noqa: F401
        except ImportError:
            pass
        else:
            for name in ("cloudViewer.cuda.pybind", "cloudViewer.cpu.pybind"):
                module = sys.modules.get(name)
                if module is not None and hasattr(module, "reconstruction"):
                    return module
    # 2) Build-tree bootstrap.
    if "pybind" in sys.modules:
        return sys.modules["pybind"]
    if not _SO.exists():
        pytest.skip(f"pybind extension not built: {_SO}")
    spec = importlib.util.spec_from_file_location("pybind", _SO)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pybind"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def recon(pybind):
    """The ``reconstruction`` submodule (parity surface root)."""
    return pybind.reconstruction


@pytest.fixture(scope="session")
def scene(recon):
    return recon.scene


@pytest.fixture(scope="session")
def geometry(recon):
    return recon.geometry


@pytest.fixture(scope="session")
def estimators(recon):
    return recon.estimators


@pytest.fixture(scope="session")
def mvs(recon):
    return recon.mvs


@pytest.fixture(scope="session")
def pipeline(recon):
    return recon.pipeline


@pytest.fixture(scope="session")
def sensor(recon):
    return recon.sensor


@pytest.fixture(scope="session")
def image(recon):
    return recon.image


@pytest.fixture(scope="session")
def feature(recon):
    return recon.feature


@pytest.fixture(scope="session")
def retrieval(recon):
    # Renamed to the upstream pycolmap module name (retrieval/).
    return recon.retrieval


@pytest.fixture(scope="session")
def database(recon):
    return recon.database


@pytest.fixture(scope="session")
def sfm(recon):
    return recon.sfm


@pytest.fixture(scope="session")
def utility(pybind):
    # The root-module utility surface (DoubleVector and friends) used by
    # database writers that take opaque vectors.
    return pybind.utility


@pytest.fixture(scope="session")
def tmp_db_path(tmp_path_factory):
    """A fresh database file path."""
    return str(tmp_path_factory.mktemp("db") / "test.db")
