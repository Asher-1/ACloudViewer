"""Port of upstream pycolmap geometry/homography_matrix_test.py (Open3D
style)."""


def test_pose_from_homography_matrix_exists(geometry):
    assert hasattr(geometry, "pose_from_homography_matrix")
    assert callable(geometry.pose_from_homography_matrix)
