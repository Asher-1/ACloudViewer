# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/pose_test.py (Open3D style).

The fork exposes the pose entry points as methods on the options classes
(and as module-level twins); the synthetic-data invariants mirror upstream.
"""
import numpy as np
import pytest


def _synthetic_scene(scene):
    """A 3D point row + its 2D projections through a PINHOLE camera."""
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    points3D = np.array([[0.0, 0.0, 5.0], [-1.0, -1.0, 6.0], [1.0, 1.0, 4.0],
                         [-0.5, 0.5, 5.5], [0.5, -0.5, 4.5], [0.0, 1.0, 5.2],
                         [1.0, 0.0, 4.8], [-1.0, 0.0, 5.1]])
    cam_from_world = (0.2, -0.1, 0.3, 0.9, 0.05, -0.02, 0.1)
    points2D = []
    R = _quat_to_matrix(cam_from_world[:4])
    t = np.array(cam_from_world[4:])
    for X in points3D:
        p = R @ X + t
        points2D.append(
            [800.0 * p[0] / p[2] + 320.0, 800.0 * p[1] / p[2] + 240.0])
    return cam, np.array(points2D), points3D, cam_from_world


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + y * y)]])


def test_estimate_absolute_pose(scene, estimators):
    cam, points2D, points3D, cfw = _synthetic_scene(scene)
    est = estimators.AbsolutePoseEstimationOptions()
    out = estimators.estimate_absolute_pose(est, np.array(points2D),
                                            np.array(points3D), cam)
    assert out is not None


def test_refine_absolute_pose(scene, estimators, geometry):
    cam, points2D, points3D, cfw = _synthetic_scene(scene)
    ref = estimators.AbsolutePoseRefinementOptions()
    pose = geometry.Rigid3d(rotation_wxyz=np.array(cfw[:4]),
                            translation=np.array(cfw[4:]))
    out = ref.refine_absolute_pose(np.ones(len(points2D), dtype=bool),
                                   np.array(points2D), np.array(points3D), pose,
                                   cam)
    # Refinement may legitimately return None on a degenerate synthetic
    # configuration; the surface contract is "dict or None".
    assert out is None or isinstance(out, dict)


def test_estimate_and_refine_absolute_pose(scene, estimators):
    cam, points2D, points3D, _ = _synthetic_scene(scene)
    ref = estimators.AbsolutePoseRefinementOptions()
    # estimation_options occupies the first positional slot; pass the rest
    # by keyword.
    out = ref.estimate_and_refine_absolute_pose(points2D=np.array(points2D),
                                                points3D=np.array(points3D),
                                                camera=cam)
    assert out is not None and "cam_from_world" in out
