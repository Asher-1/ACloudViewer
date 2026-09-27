"""Port of upstream pycolmap estimators/generalized_pose_test.py (Open3D
style)."""
import numpy as np
import pytest


@pytest.mark.parametrize(
    "name",
    [
        "estimate_generalized_absolute_pose",
        "refine_generalized_absolute_pose",
        "estimate_and_refine_generalized_absolute_pose",
        "estimate_scaled_generalized_absolute_pose",
        "refine_scaled_generalized_absolute_pose",
        "estimate_and_refine_scaled_generalized_absolute_pose",
        "estimate_generalized_relative_pose",
    ],
)
def test_public_api_callable(estimators, name):
    assert callable(getattr(estimators, name))


def build_tiny_scaled_problem(scene, geometry):
    """Two-camera rig with exact correspondences for smoke testing."""
    camera = scene.Camera.create_from_model_name(
        1, "SIMPLE_PINHOLE", 500.0, 640, 480)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    gt_rig_from_world = geometry.Sim3d(1.5, identity, np.zeros(3))
    cams_from_rig = [
        geometry.Rigid3d(),
        geometry.Rigid3d(identity, np.array([1.0, 0.0, 0.0])),
    ]
    points2D = []
    points3D = []
    camera_idxs = []
    for i in range(8):
        point3D = np.array([0.1 * i - 0.35, 0.05 * (i % 3) - 0.05, 5.0])
        point3D_in_cam = cams_from_rig[i % 2] * (gt_rig_from_world * point3D)
        points2D.append(camera.img_from_cam(point3D_in_cam))
        points3D.append(point3D)
        camera_idxs.append(i % 2)
    return dict(
        gt_rig_from_world=gt_rig_from_world,
        points2D=np.array(points2D),
        points3D=np.array(points3D),
        camera_idxs=np.array(camera_idxs),
        cams_from_rig=cams_from_rig,
        cameras=[camera, camera],
    )


def test_scaled_generalized_absolute_pose_smoke(estimators, scene, geometry):
    problem = build_tiny_scaled_problem(scene, geometry)
    ransac_options = estimators.RANSACOptions()
    ransac_options.max_error = 4.0
    ransac_options.random_seed = 0
    estimation = estimators.estimate_scaled_generalized_absolute_pose(
        points2D=problem["points2D"],
        points3D=problem["points3D"],
        camera_idxs=problem["camera_idxs"],
        cams_from_rig=problem["cams_from_rig"],
        cameras=problem["cameras"],
        estimation_options=ransac_options,
    )
    assert estimation is not None
    assert {"rig_from_world", "num_inliers", "inlier_mask"} <= set(
        estimation.keys())
