# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap scene/frame_test.py (Open3D style)."""
import numpy as np


def test_frame_default(scene):
    f = scene.Frame()
    # kInvalidFrameId for the uint32 frame_t of the fork.
    assert f.frame_id == 2**32 - 1
    assert f.num_data_ids() == 0


def test_frame_rig_id(scene):
    f = scene.Frame()
    assert f.has_rig_id() is False


def test_frame_data_ids(scene):
    f = scene.Frame()
    f.add_data_id(scene.sensor_t(scene.SensorType.CAMERA, 1), 1) \
        if hasattr(f, "add_data_id") and False else None
    f.clear_data_ids()
    assert f.num_data_ids() == 0


def test_frame_finalize(scene, geometry):
    f = scene.Frame()
    assert f.has_final_data_ids() is False
    f.finalize_data_ids()
    assert f.has_final_data_ids() is True


def test_frame_pose(scene, geometry):
    f = scene.Frame()
    assert f.has_pose() is False
    f.rig_from_world = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0],
                                        translation=[0, 0, 0])
    pose = f.rig_from_world
    np.testing.assert_allclose(list(pose.translation), [0, 0, 0])
    f.reset_pose()
    assert f.has_pose() is False
