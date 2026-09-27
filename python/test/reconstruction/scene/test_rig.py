# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap scene/rig_test.py (Open3D style)."""
import numpy as np


def test_rig_default(scene):
    rig = scene.Rig()
    assert rig.rig_id == 2**32 - 1
    assert rig.num_sensors() == 0


def _same(a, b):
    return a.type == b.type and a.id == b.id


def test_rig_add_ref_sensor(scene, geometry):
    rig = scene.Rig()
    sensor = scene.sensor_t(scene.SensorType.CAMERA, 1)
    rig.add_ref_sensor(sensor)
    assert _same(rig.ref_sensor_id, sensor)
    assert rig.num_sensors() == 1


def test_rig_add_sensor_with_pose(scene, geometry):
    rig = scene.Rig()
    ref = scene.sensor_t(scene.SensorType.CAMERA, 1)
    non_ref = scene.sensor_t(scene.SensorType.CAMERA, 2)
    rig.add_ref_sensor(ref)
    rig.add_sensor(non_ref, qvec=[1, 0, 0, 0], tvec=[0.1, 0.0, 0.0])
    assert rig.has_sensor_from_rig(non_ref)
    pose = rig.sensor_from_rig(non_ref)
    np.testing.assert_allclose(list(pose.translation), [0.1, 0.0, 0.0])
    assert any(_same(s, non_ref) for s in rig.non_ref_sensors())


def test_rig_set_sensor_from_rig(scene, geometry):
    rig = scene.Rig()
    ref = scene.sensor_t(scene.SensorType.CAMERA, 1)
    non_ref = scene.sensor_t(scene.SensorType.CAMERA, 2)
    rig.add_ref_sensor(ref)
    rig.add_sensor(non_ref)
    pose = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0],
                            translation=[0.2, 0.0, 0.0])
    rig.set_sensor_from_rig(non_ref, pose)
    assert rig.has_sensor_from_rig(non_ref)
    assert list(rig.sensor_from_rig(non_ref).translation) == [0.2, 0.0, 0.0]


def test_rig_reset_sensor_from_rig(scene, geometry):
    rig = scene.Rig()
    ref = scene.sensor_t(scene.SensorType.CAMERA, 1)
    non_ref = scene.sensor_t(scene.SensorType.CAMERA, 2)
    rig.add_ref_sensor(ref)
    rig.add_sensor(non_ref, qvec=[1, 0, 0, 0], tvec=[0, 0, 0])
    rig.reset_sensor_from_rig(non_ref)
    assert not rig.has_sensor_from_rig(non_ref)
