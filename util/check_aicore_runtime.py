#!/usr/bin/env python3
"""Validate the packaged AICore backend contract without importing pybind."""

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path


class Device(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char_p),
        ("label", ctypes.c_char_p),
        ("is_default", ctypes.c_int),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("aicore", type=Path)
    parser.add_argument("--expect-device", action="append", default=[])
    args = parser.parse_args()

    library = ctypes.CDLL(str(args.aicore.resolve()))
    library.aicore_backend_abi_version.restype = ctypes.c_int
    library.aicore_device_count.restype = ctypes.c_size_t
    library.aicore_device_at.argtypes = [ctypes.c_size_t]
    library.aicore_device_at.restype = ctypes.POINTER(Device)
    library.aicore_device_available.argtypes = [ctypes.c_char_p]
    library.aicore_device_available.restype = ctypes.c_int

    if library.aicore_backend_abi_version() != 1:
        raise RuntimeError("unexpected AICore backend ABI")

    devices: list[tuple[str, str]] = []
    for index in range(library.aicore_device_count()):
        device = library.aicore_device_at(index)
        if not device:
            raise RuntimeError(f"null device entry at index {index}")
        devices.append(
            (
                device.contents.id.decode("utf-8"),
                device.contents.label.decode("utf-8"),
            )
        )

    ids = {device_id.split(":", 1)[0] for device_id, _ in devices}
    if "cpu" not in ids or "blas" in ids:
        raise RuntimeError(f"invalid baseline devices: {devices}")
    for expected in args.expect_device:
        if library.aicore_device_available(expected.encode("utf-8")) != 1:
            raise RuntimeError(f"required device {expected!r} is unavailable: {devices}")

    print("AICore devices:")
    for device_id, label in devices:
        print(f"  {device_id}: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
