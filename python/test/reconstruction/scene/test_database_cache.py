# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/database_cache_test.py (Open3D style)."""


def test_database_cache_options_surface(scene):
    opts = scene.DatabaseCacheOptions()
    opts.min_num_matches = 5
    assert opts.min_num_matches == 5
    opts.ignore_watermarks = True
    assert opts.ignore_watermarks
    opts.load_all_images = True
    assert opts.load_all_images


def test_database_cache_construct_and_load(scene, database, tmp_db_path):
    import os
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)
    db = database.Database.open(tmp_db_path)
    cache = scene.DatabaseCache()
    cache.load(db, scene.DatabaseCacheOptions())
    assert cache.num_rigs == 0
    assert cache.num_cameras == 0
    assert cache.num_images == 0
    db.close()


def test_database_cache_create_factory(scene, database, tmp_db_path):
    import os
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)
    db = database.Database.open(tmp_db_path)
    cache = scene.DatabaseCache.create(db, scene.DatabaseCacheOptions())
    assert cache is not None
    db.close()


def test_database_cache_management_surface(scene):
    cache = scene.DatabaseCache()
    for method in ("add_rig", "add_camera", "add_frame", "add_image",
                   "add_pose_prior", "exists_rig", "exists_camera",
                   "exists_frame", "exists_image", "find_image_with_name",
                   "rig", "frame", "image"):
        assert hasattr(cache, method), method
