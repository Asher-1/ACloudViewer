# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/pose_graph_test.py (Open3D style)."""
import pytest


def _pair_id(database, a, b):
    return database.Database.image_pair_to_pair_id(a, b)


def test_pose_graph_empty(scene):
    pg = scene.PoseGraph()
    assert pg is not None
    assert isinstance(pg.edges, dict) and len(pg.edges) == 0


def test_pose_graph_edge_validity(scene, database):
    pg = scene.PoseGraph()
    edge = scene.PoseGraphEdge(cam2_from_cam1=None) if False \
        else scene.PoseGraphEdge()
    pg.add_edge(1, 2, edge)
    pair_id = _pair_id(database, 1, 2)
    pg.set_valid_edge(pair_id)
    assert pg.is_valid(pair_id)
    pg.set_invalid_edge(pair_id)
    assert not pg.is_valid(pair_id)
    assert pair_id in pg.edges


def test_pose_graph_get_update_delete_edge(scene, geometry, database):
    pg = scene.PoseGraph()
    pg.add_edge(1, 2, scene.PoseGraphEdge())
    edge = pg.get_edge(1, 2)
    assert edge is not None
    edge.cam2_from_cam1 = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0],
                                           translation=[1, 0, 0])
    pg.update_edge(1, 2, edge)
    pg.delete_edge(1, 2)
    assert _pair_id(database, 1, 2) not in pg.edges
