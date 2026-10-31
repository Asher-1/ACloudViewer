# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap pipeline/meshing_test.py (Open3D style).

Fork adaptation: the meshing surface lives on recon.mvs."""
import pytest


def test_poisson_meshing_options_init(mvs):
    assert mvs.PoissonMeshingOptions() is not None


def test_poisson_meshing_options_depth(mvs):
    options = mvs.PoissonMeshingOptions()
    options.depth = 10
    assert options.depth == 10


def test_poisson_meshing_options_num_threads(mvs):
    options = mvs.PoissonMeshingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_mesh_simplification_options_init(mvs):
    assert mvs.MeshSimplificationOptions() is not None


def test_delaunay_meshing_options_init(mvs):
    assert mvs.DelaunayMeshingOptions() is not None


@pytest.mark.parametrize(
    "name",
    [
        "poisson_mesh",
        "mesh_delaunay",
    ],
)
def test_public_api_callable(mvs, name):
    assert callable(getattr(mvs, name))
