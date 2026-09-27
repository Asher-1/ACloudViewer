# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap util/types_test.py (Open3D style)."""


def test_sensor_t_surface(scene):
    sensor = scene.sensor_t(scene.SensorType.CAMERA, 3)
    assert sensor.type == scene.SensorType.CAMERA
    assert sensor.id == 3


def test_data_t_surface(scene):
    data = scene.data_t(scene.sensor_t(scene.SensorType.CAMERA, 1), 42)
    assert data.sensor_id.id == 1
    assert data.id == 42
