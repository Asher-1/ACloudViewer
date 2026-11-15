# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/correspondence_graph_test.py (Open3D style)."""


def test_correspondence_graph_default(scene):
    graph = scene.CorrespondenceGraph()
    assert graph is not None
    assert graph.num_images == 0


def test_correspondence_value_type(scene):
    corr = scene.Correspondence(image_id=1, point2D_idx=2)
    assert corr.image_id == 1
    assert corr.point2D_idx == 2


def test_finalize_and_counts(scene):
    graph = scene.CorrespondenceGraph()
    graph.finalize()
    assert graph.num_image_pairs == 0
