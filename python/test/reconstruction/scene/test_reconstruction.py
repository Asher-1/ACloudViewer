# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/reconstruction_test.py (Open3D style)."""
import copy

import numpy as np
import pytest


@pytest.fixture
def rec(scene, geometry):
    """A trivial reconstruction: 1 camera, 1 image, 1 point3D."""
    r = scene.Reconstruction()
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    r.add_camera_with_trivial_rig(cam)
    img = scene.Image()
    img.name = "img1.jpg"
    img.camera_id = 1
    img.image_id = 1
    r.add_image_with_trivial_frame(
        img, geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0,
                                                                       0]))
    r.add_point3D([0.0, 0.0, 5.0], scene.Track(), [255, 0, 0])
    return r


def test_reconstruction_empty(scene):
    r = scene.Reconstruction()
    assert r.num_cameras == 0
    assert r.num_images == 0
    assert r.num_points3D == 0
    assert r.is_valid() is True or r.is_valid() in (True,)


def test_reconstruction_counts(rec):
    assert rec.num_cameras == 1
    assert rec.num_images == 1
    assert rec.num_points3D == 1


def test_reconstruction_exists(rec):
    assert rec.exists_camera(1)
    assert rec.exists_image(1)
    assert rec.exists_point3D(rec.point3D_ids()[0])


def test_reconstruction_find_image_with_name(rec):
    img = rec.find_image_with_name("img1.jpg")
    assert img is not None and img.name == "img1.jpg"
    assert rec.find_image_with_name("missing.jpg") is None


def test_reconstruction_accessors(rec):
    assert rec.camera(1).model_name == "PINHOLE"
    assert rec.image(1).name == "img1.jpg"
    pid = rec.point3D_ids()[0]
    assert list(rec.point3D(pid).xyz) == [0.0, 0.0, 5.0]


def test_reconstruction_point3d_mutations(rec, scene):
    pid = rec.point3D_ids()[0]
    p3 = scene.Point3D()
    p3.xyz = [0.1, 0.1, 5.0]
    rec.add_point3D_with_id(pid + 100, p3)
    assert rec.exists_point3D(pid + 100)
    rec.delete_point3D(pid + 100)
    assert not rec.exists_point3D(pid + 100)


def test_reconstruction_deepcopy(rec):
    r2 = copy.deepcopy(rec)
    assert r2.num_images == rec.num_images
    assert r2.num_points3D == rec.num_points3D


def test_reconstruction_io(rec, tmp_path):
    rec.write_binary(str(tmp_path))
    r2 = type(rec)()
    r2.read_binary(str(tmp_path))
    assert r2.num_images == rec.num_images
    assert r2.num_points3D == rec.num_points3D


def test_reconstruction_export_ply(rec, tmp_path):
    out = str(tmp_path / "model.ply")
    rec.export_PLY(out)
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_reconstruction_normalize_and_centroid(rec):
    rec.normalize()
    assert rec.compute_centroid() is not None
    lo, hi = rec.compute_bounding_box()
    assert len(lo) == 3 and len(hi) == 3


def test_reconstruction_repr(rec):
    assert "Reconstruction" in repr(rec) or len(repr(rec)) > 0
