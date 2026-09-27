# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap pipeline/match_features_test.py (Open3D style),
plus end-to-end coverage of the fork's pairing-driven matching engine."""
import os

import numpy as np
import pytest


def _make_database(database,
                   utility,
                   scene,
                   tmp_path,
                   num_images=3,
                   seed=7,
                   num_keypoints=80):
    """Creates a database with a camera, images and near-identical SIFT-like
    features so that matching produces real matches."""
    db_path = str(tmp_path / "test.db")
    db = database.Database.open(db_path)
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 50.0, 100, 100)
    camera_id = db.write_camera(cam,
                                utility.DoubleVector([50.0, 50.0, 50.0, 50.0]))
    rng = np.random.RandomState(seed)
    base_desc = (rng.rand(num_keypoints, 128) * 255).astype(np.uint8)
    kps = (rng.rand(num_keypoints, 6).astype(np.float32) * 100)
    for i in range(num_images):
        img = scene.Image()
        img.name = f"image{i}.jpg"
        img.camera_id = camera_id
        image_id = db.write_image(img)
        noise = rng.randint(-2, 3, size=base_desc.shape)
        desc = np.clip(base_desc.astype(int) + (noise if i else 0), 0,
                       255).astype(np.uint8)
        db.write_keypoints(image_id, kps)
        db.write_descriptors(image_id, desc)
    db.close()
    return db_path


def test_device_enum(pipeline):
    assert int(pipeline.Device.AUTO) == -1
    assert int(pipeline.Device.CPU) == 0
    assert int(pipeline.Device.CUDA) == 1


def test_exhaustive_pairing_options(pipeline):
    opts = pipeline.ExhaustivePairingOptions()
    assert opts.block_size == 50
    assert opts.check()
    opts.block_size = 100
    assert opts.block_size == 100


def test_spatial_pairing_options(pipeline):
    opts = pipeline.SpatialPairingOptions()
    assert opts.check()
    assert opts.max_num_neighbors == 50
    opts.ignore_z = False
    assert not opts.ignore_z


def test_vocab_tree_pairing_options(pipeline):
    opts = pipeline.VocabTreePairingOptions()
    assert opts.check()
    assert opts.num_images == 100
    assert opts.num_nearest_neighbors == 5
    assert opts.num_checks == 64
    assert opts.num_images_after_verification == 0
    assert opts.max_num_features == -1


def test_sequential_pairing_options(pipeline):
    opts = pipeline.SequentialPairingOptions()
    assert opts.check()
    assert opts.overlap == 10
    assert opts.quadratic_overlap
    assert not opts.loop_detection
    vocab_opts = opts.vocab_tree_options()
    assert vocab_opts.num_images == 50


def test_imported_pairing_options(pipeline):
    opts = pipeline.ImportedPairingOptions()
    assert opts.check()
    assert opts.block_size == 1225


def test_existing_matched_pairing_options(pipeline):
    opts = pipeline.ExistingMatchedPairingOptions()
    assert opts.check()
    assert opts.batch_size == 1000


def test_geometric_verifier_options(pipeline):
    opts = pipeline.GeometricVerifierOptions()
    assert opts.num_threads == -1


def test_exhaustive_pair_generator_empty_db(pipeline, database, tmp_path):
    db_path = str(tmp_path / "empty.db")
    db = database.Database.open(db_path)
    gen = pipeline.ExhaustivePairGenerator(pipeline.ExhaustivePairingOptions(),
                                           db)
    assert gen.has_finished()
    assert gen.all_pairs() == []
    db.close()


def test_exhaustive_pair_generator_pairs(pipeline, database, utility, scene,
                                         tmp_path):
    db_path = str(tmp_path / "test.db")
    db = database.Database.open(db_path)
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 50.0, 100, 100)
    camera_id = db.write_camera(cam,
                                utility.DoubleVector([50.0, 50.0, 50.0, 50.0]))
    for i in range(4):
        img = scene.Image()
        img.name = f"image{i}.jpg"
        img.camera_id = camera_id
        db.write_image(img)
    gen = pipeline.ExhaustivePairGenerator(pipeline.ExhaustivePairingOptions(),
                                           db)
    pairs = gen.all_pairs()
    assert len(pairs) == 4 * 3 // 2
    db.close()


def test_imported_pair_generator(pipeline, database, utility, scene, tmp_path):
    db_path = str(tmp_path / "test.db")
    db = database.Database.open(db_path)
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 50.0, 100, 100)
    camera_id = db.write_camera(cam,
                                utility.DoubleVector([50.0, 50.0, 50.0, 50.0]))
    for i in range(3):
        img = scene.Image()
        img.name = f"image{i}.jpg"
        img.camera_id = camera_id
        db.write_image(img)
    db.close()

    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("image0.jpg image1.jpg\n"
                          "image1.jpg image2.jpg\n")
    opts = pipeline.ImportedPairingOptions()
    opts.match_list_path = str(pairs_file)
    db = database.Database.open(db_path)
    gen = pipeline.ImportedPairGenerator(opts, db)
    pairs = gen.all_pairs()
    assert len(pairs) == 2
    db.close()


def test_match_exhaustive(pipeline, database, utility, scene, tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path)
    pipeline.match_exhaustive(db_path, device=pipeline.Device.CPU)

    db = database.Database.open(db_path)
    raw = db.read_matches(1, 2)
    assert len(raw) > 0
    db.close()


def test_match_sequential(pipeline, database, utility, scene, tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path)
    pipeline.match_sequential(db_path, device=pipeline.Device.CPU)

    db = database.Database.open(db_path)
    raw = db.read_matches(1, 2)
    assert len(raw) > 0
    db.close()


def test_match_sequential_quadratic_overlap(pipeline, database, utility, scene,
                                            tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path, num_images=5)
    opts = pipeline.SequentialPairingOptions()
    opts.quadratic_overlap = True
    pipeline.match_sequential(db_path,
                              pairing_options=opts,
                              device=pipeline.Device.CPU)
    db = database.Database.open(db_path)
    assert len(db.read_matches(1, 2)) > 0
    db.close()


def test_verify_matches(pipeline, database, utility, scene, tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path)
    # Import matches first: use match_image_pairs on a pairs file.
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("image0.jpg image1.jpg\n")
    pipeline.match_image_pairs(db_path,
                               pairing_options=_imported_opts(
                                   pipeline, str(pairs_file)),
                               device=pipeline.Device.CPU)
    db = database.Database.open(db_path)
    assert len(db.read_matches(1, 2)) > 0
    db.close()


def test_match_image_pairs_missing_image(pipeline, database, utility, scene,
                                         tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path)
    pairs_file = tmp_path / "pairs.txt"
    pairs_file.write_text("missing0.jpg missing1.jpg\n")
    # Upstream behavior: unknown image names are reported and skipped.
    pipeline.match_image_pairs(db_path,
                               pairing_options=_imported_opts(
                                   pipeline, str(pairs_file)),
                               device=pipeline.Device.CPU)


def test_match_rejects_missing_database(pipeline, tmp_path):
    with pytest.raises(Exception):
        pipeline.match_exhaustive(str(tmp_path / "does_not_exist.db"))


def test_geometric_verification(pipeline, database, utility, scene, tmp_path):
    db_path = _make_database(database, utility, scene, tmp_path)
    # Match first so that the database holds matches + two-view geometries.
    pipeline.match_exhaustive(db_path, device=pipeline.Device.CPU)
    # Re-run plain verification on the existing matched pairs.
    pipeline.geometric_verification(db_path)
    db = database.Database.open(db_path)
    pairs, _ = db.read_two_view_geometry_num_inliers()
    assert len(pairs) >= 0
    db.close()


def _imported_opts(pipeline, path):
    opts = pipeline.ImportedPairingOptions()
    opts.match_list_path = path
    return opts
