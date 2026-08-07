# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Validate the packaged AICore backend contract without importing pybind."""

from __future__ import annotations

import argparse
import ctypes
import re
from pathlib import Path

# Single source of truth for the expected ABI version is the macro
# AICORE_BACKEND_ABI_VERSION in core/AICore/include/aicore/backend_capi.h.
# This script reads that header instead of hardcoding the number, so bumping
# the version in the header can never silently desync this checker.
_DEFAULT_ABI_HEADER = (Path(__file__).resolve().parent.parent /
                       "core/AICore/include/aicore/backend_capi.h")
_ABI_VERSION_RE = re.compile(
    r"^\s*#define\s+AICORE_BACKEND_ABI_VERSION\s+(\d+)\s*$", re.MULTILINE)


class Device(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char_p),
        ("label", ctypes.c_char_p),
        ("is_default", ctypes.c_int),
    ]


def expected_abi_version(header: Path) -> int:
    """Parse AICORE_BACKEND_ABI_VERSION from the C header."""
    match = _ABI_VERSION_RE.search(header.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"AICORE_BACKEND_ABI_VERSION not found in {header}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("aicore", type=Path)
    parser.add_argument("--abi-header", type=Path, default=_DEFAULT_ABI_HEADER)
    parser.add_argument("--expect-device", action="append", default=[])
    args = parser.parse_args()

    library = ctypes.CDLL(str(args.aicore.resolve()))
    library.aicore_backend_abi_version.restype = ctypes.c_int
    library.aicore_device_count.restype = ctypes.c_size_t
    library.aicore_device_at.argtypes = [ctypes.c_size_t]
    library.aicore_device_at.restype = ctypes.POINTER(Device)
    library.aicore_device_available.argtypes = [ctypes.c_char_p]
    library.aicore_device_available.restype = ctypes.c_int

    expected = expected_abi_version(args.abi_header)
    if library.aicore_backend_abi_version() != expected:
        raise RuntimeError(
            f"unexpected AICore backend ABI: got "
            f"{library.aicore_backend_abi_version()}, expected {expected} "
            f"(see {args.abi_header})")

    devices: list[tuple[str, str]] = []
    for index in range(library.aicore_device_count()):
        device = library.aicore_device_at(index)
        if not device:
            raise RuntimeError(f"null device entry at index {index}")
        devices.append((
            device.contents.id.decode("utf-8"),
            device.contents.label.decode("utf-8"),
        ))

    ids = {device_id.split(":", 1)[0] for device_id, _ in devices}
    if "cpu" not in ids or "blas" in ids:
        raise RuntimeError(f"invalid baseline devices: {devices}")
    for expected in args.expect_device:
        if library.aicore_device_available(expected.encode("utf-8")) != 1:
            raise RuntimeError(
                f"required device {expected!r} is unavailable: {devices}")

    print("AICore devices:")
    for device_id, label in devices:
        print(f"  {device_id}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
