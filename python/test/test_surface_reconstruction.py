# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import pytest


@pytest.mark.parametrize("with_colors", [False, True])
def test_poisson_triangle_indices(with_colors):
    """Both output streams must emit zero-based indices, not vertex counts."""
    # Deterministic oriented sphere; no data downloads or GL context required.
    count = 512
    z = 1.0 - 2.0 * (np.arange(count) + 0.5) / count
    phi = np.arange(count) * (np.pi * (3.0 - np.sqrt(5.0)))
    radius = np.sqrt(1.0 - z * z)
    points = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    cloud = cv3d.geometry.ccPointCloud()
    cloud.set_points(cv3d.utility.Vector3dVector(points))
    cloud.set_normals(cv3d.utility.Vector3dVector(points))
    if with_colors:
        cloud.set_colors(cv3d.utility.Vector3dVector((points + 1.0) / 2.0))

    mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(
        cloud, depth=5, n_threads=1)
    vertices = np.asarray(mesh.get_vertices())
    triangles = np.asarray(mesh.get_triangles())
    assert len(vertices) > 0
    assert len(triangles) > 0
    assert len(densities) == len(vertices)
    assert np.isfinite(vertices).all()
    assert np.isfinite(densities).all()
    assert triangles.max() < len(vertices)
    assert triangles.min() >= 0
    if with_colors:
        assert len(mesh.get_vertex_colors()) == len(vertices)
    # Follow the notebook's pruning path before computing any normals. Merely
    # reserving an empty normal table must not make the mesh report normals.
    assert not mesh.has_normals()
    assert len(mesh.get_vertex_normals()) == 0
    mask = np.asarray(densities) < np.quantile(densities, 0.01)
    expected_vertices = np.count_nonzero(~mask)
    mesh.remove_vertices_by_mask(mask)
    assert len(mesh.get_vertices()) == expected_vertices
    assert np.asarray(mesh.get_triangles()).min() >= 0
    assert np.asarray(mesh.get_triangles()).max() < expected_vertices
    if with_colors:
        assert len(mesh.get_vertex_colors()) == expected_vertices
    # Exercise a downstream consumer without allowing invalid indices to reach
    # the renderer (which would otherwise fail later during notebook capture).
    mesh.compute_vertex_normals()
    assert len(mesh.get_vertex_normals()) == expected_vertices
    assert np.isfinite(np.asarray(mesh.get_vertex_normals())).all()
