# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import cloudViewer.core as cv3c
import numpy as np
import pytest
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from cloudViewer_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_constructor_and_accessors(device):
    dtype = cv3c.float32

    # Constructor.
    pcd = cv3d.t.geometry.PointCloud(device)
    assert "positions" not in pcd.point
    assert "colors" not in pcd.point
    assert isinstance(pcd.point, cv3d.t.geometry.TensorMap)

    # Assignment.
    pcd.point.positions = cv3c.Tensor.ones((0, 3), dtype, device)
    pcd.point.colors = cv3c.Tensor.ones((0, 3), dtype, device)
    assert len(pcd.point.positions) == 0
    assert len(pcd.point.colors) == 0

    pcd.point.positions = cv3c.Tensor.ones((1, 3), dtype, device)
    pcd.point.colors = cv3c.Tensor.ones((1, 3), dtype, device)
    assert len(pcd.point.positions) == 1
    assert len(pcd.point.colors) == 1

    # Edit and access values.
    points = pcd.point.positions
    points[0] = cv3c.Tensor([1, 2, 3], dtype, device)
    assert pcd.point.positions.allclose(cv3c.Tensor([[1, 2, 3]], dtype, device))


@pytest.mark.parametrize("device", list_devices())
def test_from_legacy(device):
    dtype = cv3c.float32

    legacy_pcd = cv3d.geometry.ccPointCloud()
    legacy_pcd.set_points(
        cv3d.utility.Vector3dVector(np.array([[0, 1, 2], [3, 4, 5]])))
    # Use normalized color values in [0, 1] range for legacy point clouds
    legacy_pcd.set_colors(
        cv3d.utility.Vector3dVector(np.array([[0.2, 0.3, 0.4], [0.5, 0.6,
                                                                0.7]])))

    pcd = cv3d.t.geometry.PointCloud.from_legacy(legacy_pcd, dtype, device)
    assert pcd.point.positions.allclose(
        cv3c.Tensor([[0, 1, 2], [3, 4, 5]], dtype, device))
    # Legacy point clouds store colors as uint8 internally, causing quantization.
    # Allow tolerance for 8-bit quantization error (max ~1/255 ≈ 0.004)
    assert pcd.point.colors.allclose(cv3c.Tensor(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype, device),
                                     rtol=1e-2,
                                     atol=2.5e-3)


@pytest.mark.parametrize("device", list_devices())
def test_to_legacy(device):
    dtype = cv3c.float32

    pcd = cv3d.t.geometry.PointCloud(device)
    pcd.point.positions = cv3c.Tensor([[0, 1, 2], [3, 4, 5]], dtype, device)
    # Use normalized color values in [0, 1] range for legacy point cloud compatibility
    pcd.point.colors = cv3c.Tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], dtype,
                                   device)

    legacy_pcd = pcd.to_legacy()
    np.testing.assert_allclose(np.asarray(legacy_pcd.points()),
                               np.array([[0, 1, 2], [3, 4, 5]]))
    # Legacy point clouds store colors as uint8 internally, causing quantization.
    # Allow tolerance for 8-bit quantization error (max ~1/255 ≈ 0.004)
    np.testing.assert_allclose(np.asarray(legacy_pcd.colors()),
                               np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]),
                               rtol=1e-2,
                               atol=2.5e-3)


@pytest.mark.parametrize("device", list_devices())
def test_member_functions(device):
    dtype = cv3c.float32

    # get_min_bound, get_max_bound, get_center.
    pcd = cv3d.t.geometry.PointCloud(device)
    pcd.point.positions = cv3c.Tensor([[1, 10, 20], [30, 2, 40], [50, 60, 3]],
                                      dtype, device)
    assert pcd.get_min_bound().allclose(cv3c.Tensor([1, 2, 3], dtype, device))
    assert pcd.get_max_bound().allclose(cv3c.Tensor([50, 60, 40], dtype,
                                                    device))
    assert pcd.get_center().allclose(cv3c.Tensor([27, 24, 21], dtype, device))

    # append.
    pcd = cv3d.t.geometry.PointCloud(device)
    pcd.point.positions = cv3c.Tensor.ones((2, 3), dtype, device)
    pcd.point.normals = cv3c.Tensor.ones((2, 3), dtype, device)

    pcd2 = cv3d.t.geometry.PointCloud(device)
    pcd2.point.positions = cv3c.Tensor.ones((2, 3), dtype, device)
    pcd2.point.normals = cv3c.Tensor.ones((2, 3), dtype, device)
    pcd2.point.labels = cv3c.Tensor.ones((2, 3), dtype, device)

    pcd3 = cv3d.t.geometry.PointCloud(device)
    pcd3 = pcd + pcd2

    assert pcd3.point.positions.allclose(cv3c.Tensor.ones((4, 3), dtype,
                                                          device))
    assert pcd3.point.normals.allclose(cv3c.Tensor.ones((4, 3), dtype, device))

    with pytest.raises(RuntimeError) as excinfo:
        pcd3 = pcd2 + pcd
        assert 'The pointcloud is missing attribute' in str(excinfo.value)

    # transform.
    pcd = cv3d.t.geometry.PointCloud(device)
    transform_t = cv3c.Tensor(
        [[1, 1, 0, 1], [0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]], dtype, device)
    pcd.point.positions = cv3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.point.normals = cv3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.transform(transform_t)
    assert pcd.point.positions.allclose(cv3c.Tensor([[3, 3, 2]], dtype, device))
    assert pcd.point.normals.allclose(cv3c.Tensor([[2, 2, 1]], dtype, device))

    # translate.
    pcd = cv3d.t.geometry.PointCloud(device)
    transloation = cv3c.Tensor([10, 20, 30], dtype, device)

    pcd.point.positions = cv3c.Tensor([[0, 1, 2], [6, 7, 8]], dtype, device)
    pcd.translate(transloation, True)
    assert pcd.point.positions.allclose(
        cv3c.Tensor([[10, 21, 32], [16, 27, 38]], dtype, device))

    pcd.point.positions = cv3c.Tensor([[0, 1, 2], [6, 7, 8]], dtype, device)
    pcd.translate(transloation, False)
    assert pcd.point.positions.allclose(
        cv3c.Tensor([[7, 17, 27], [13, 23, 33]], dtype, device))

    # scale
    pcd = cv3d.t.geometry.PointCloud(device)
    pcd.point.positions = cv3c.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype,
                                      device)
    center = cv3c.Tensor([1, 1, 1], dtype, device)
    pcd.scale(4, center)
    assert pcd.point.positions.allclose(
        cv3c.Tensor([[-3, -3, -3], [1, 1, 1], [5, 5, 5]], dtype, device))

    # rotate.
    pcd = cv3d.t.geometry.PointCloud(device)
    rotation = cv3c.Tensor([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype, device)
    center = cv3c.Tensor([1, 1, 1], dtype, device)
    pcd.point.positions = cv3c.Tensor([[2, 2, 2]], dtype, device)
    pcd.point.normals = cv3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.rotate(rotation, center)
    assert pcd.point.positions.allclose(cv3c.Tensor([[3, 3, 2]], dtype, device))
    assert pcd.point.normals.allclose(cv3c.Tensor([[2, 2, 1]], dtype, device))

    # voxel_down_sample
    pcd = cv3d.t.geometry.PointCloud(device)
    pcd.point.positions = cv3c.Tensor(
        [[0.1, 0.3, 0.9], [0.9, 0.2, 0.4], [0.3, 0.6, 0.8], [0.2, 0.4, 0.2]],
        dtype, device)

    pcd_small_down = pcd.voxel_down_sample(1)
    assert pcd_small_down.point.positions.allclose(
        cv3c.Tensor([[0.375, 0.375, 0.575]], dtype, device))


def test_extrude_rotation():
    pcd = cv3d.t.geometry.PointCloud([[1.0, 0, 0]])
    ans = pcd.extrude_rotation(3 * 360, [0, 1, 0],
                               resolution=3 * 16,
                               translation=2)
    assert ans.point.positions.shape == (49, 3)
    assert ans.line.indices.shape == (48, 2)


def test_extrude_linear():
    pcd = cv3d.t.geometry.PointCloud([[1.0, 0, 0]])
    ans = pcd.extrude_linear([0, 0, 1])
    assert ans.point.positions.shape == (2, 3)
    assert ans.line.indices.shape == (1, 2)


@pytest.mark.parametrize("device", list_devices())
def test_pickle(device):
    pcd = cv3d.t.geometry.PointCloud(device)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/pcd.pkl"
        pcd.point.positions = cv3c.Tensor.ones((10, 3),
                                               cv3c.float32,
                                               device=device)
        pickle.dump(pcd, open(file_name, "wb"))
        pcd_load = pickle.load(open(file_name, "rb"))
        assert pcd_load.point.positions.device == device and pcd_load.point.positions.dtype == cv3c.float32
        np.testing.assert_equal(pcd.point.positions.cpu().numpy(),
                                pcd_load.point.positions.cpu().numpy())


def test_metrics():

    from cloudViewer.t.geometry import TriangleMesh, PointCloud, Metric, MetricParameters
    # box is a cube with one vertex at the origin and a side length 1
    pos = TriangleMesh.create_box().vertex.positions
    pcd1 = PointCloud(pos.clone())
    pcd2 = PointCloud(pos * 1.1)

    # (1, 3, 3, 1) vertices are shifted by (0, 0.1, 0.1*sqrt(2), 0.1*sqrt(3))
    # respectively
    metric_params = MetricParameters(fscore_radius=(0.01, 0.11, 0.15, 0.18))
    metrics = pcd1.compute_metrics(
        pcd2, (Metric.ChamferDistance, Metric.HausdorffDistance, Metric.FScore),
        metric_params)

    np.testing.assert_allclose(
        metrics.cpu().numpy(),
        (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
        rtol=1e-6)
