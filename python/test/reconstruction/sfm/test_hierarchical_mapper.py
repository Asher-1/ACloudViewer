# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap sfm/hierarchical_mapper_test.py (Open3D
style). The options classes live on recon.sfm."""


def test_scene_clustering_options_init(sfm):
    assert sfm.SceneClusteringOptions() is not None


def test_scene_clustering_options_check(sfm):
    assert sfm.SceneClusteringOptions().check()


def test_scene_clustering_options_leaf_max_num_images(sfm):
    options = sfm.SceneClusteringOptions()
    options.leaf_max_num_images = 100
    assert options.leaf_max_num_images == 100


def test_hierarchical_pipeline_options_init(sfm):
    assert sfm.HierarchicalPipelineOptions() is not None


def test_hierarchical_pipeline_options_check(sfm):
    assert sfm.HierarchicalPipelineOptions().check()


def test_hierarchical_mapper_callable(sfm):
    assert callable(sfm.hierarchical_mapper)
