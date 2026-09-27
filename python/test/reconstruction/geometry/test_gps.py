# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap geometry/gps_test.py (Open3D style).

The fork owns the full GPSTransform conversion surface (ellipsoid/ENU/ECEF)
plus the UTM entry points added for upstream parity. The upstream tests
parametrize over all ellipsoids; the subset below keeps the same numeric
invariants (round trips) for the default ellipsoid (WGS84) and one metric
datum to bound the runtime.
"""
import numpy as np
import pytest

ELLIPSOIDS = [None, None]  # default ctor is fixed to WGS84 in the fork


def _gps(geometry):
    return geometry.GPSTransform()


def _v3(pybind, rows):
    """Wrap rows in the repo Vector3dVector container expected by the
    GPS conversion surface."""
    return pybind.utility.Vector3dVector(rows)


def test_ellipsoid_to_enu_roundtrip(geometry, pybind):
    gps = _gps(geometry)
    lla = [(47.5, 8.5, 500.0), (48.0, 9.0, 600.0)]
    enu = np.asarray(gps.ellipsoid_to_enu(_v3(pybind, lla), 47.5, 8.5, 500.0))
    assert enu.shape[0] == 2
    back = np.asarray(
        gps.enu_to_ellipsoid(_v3(pybind, [tuple(row) for row in enu]), 47.5,
                             8.5, 500.0))
    np.testing.assert_allclose(back, lla, atol=1e-6)


def test_ellipsoid_to_ecef_roundtrip(geometry, pybind):
    gps = _gps(geometry)
    lla = [(47.5, 8.5, 500.0)]
    ecef = np.asarray(gps.ellipsoid_to_ecef(_v3(pybind, lla)))
    assert ecef.shape == (1, 3)
    back = np.asarray(gps.ecef_to_ellipsoid(_v3(pybind, [tuple(ecef[0])])))
    np.testing.assert_allclose(back, lla, atol=1e-6)


def test_ecef_to_enu(geometry, pybind):
    gps = _gps(geometry)
    lla = [(47.5, 8.5, 500.0)]
    ecef = np.asarray(gps.ellipsoid_to_ecef(_v3(pybind, lla)))
    enu = np.asarray(
        gps.ecef_to_enu(_v3(pybind, [tuple(ecef[0])]), ecef[0].reshape(3, 1)))
    assert enu.shape == (1, 3)
    # The reference point itself is at the ENU origin.
    np.testing.assert_allclose(enu[0], [0, 0, 0], atol=1e-6)


def test_ellipsoid_to_utm_roundtrip(geometry):
    gps = _gps(geometry)
    lla = [(47.5, 8.5, 500.0)]
    utm = geometry.ellipsoid_to_utm(gps, np.array(lla))
    assert "xyz_in_utm" in utm and "zone" in utm
    assert utm["zone"] == 32
    back = geometry.utm_to_ellipsoid(gps, utm["xyz_in_utm"], utm["zone"], True)
    np.testing.assert_allclose(back, lla, atol=1e-6)
