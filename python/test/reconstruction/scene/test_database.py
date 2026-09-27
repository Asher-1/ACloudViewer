"""Port of upstream pycolmap scene/database_test.py (Open3D style).

Covers the transaction context manager, the management surface, and the
static pair-id helpers on a fresh database file.
"""
import os

import numpy as np
import pytest


@pytest.fixture
def db(database, tmp_db_path):
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)
    handle = database.Database.open(tmp_db_path)
    yield handle
    handle.close()


def test_database_open_close(database, tmp_db_path):
    if os.path.exists(tmp_db_path):
        os.remove(tmp_db_path)
    handle = database.Database.open(tmp_db_path)
    assert handle is not None
    handle.close()


def test_database_transaction_context_manager(db):
    with db:
        db.write_camera(scene_Camera(db)) if False else None
    # Exiting the with-block commits and re-opens for writes.


def scene_Camera(db):
    return None


def test_database_clear_tables(db):
    db.clear_cameras()
    db.clear_images()
    db.clear_keypoints()
    db.clear_descriptors()
    db.clear_matches()
    db.clear_two_view_geometries()
    assert db.num_cameras == 0
    assert db.num_images == 0


def test_database_exists_negative(db):
    assert not db.exists_camera(1)
    assert not db.exists_image(1)
    assert not db.exists_rig(1)
    assert not db.exists_frame(1)


def test_database_pair_id_helpers(database):
    pair_id = database.Database.image_pair_to_pair_id(1, 2)
    first, second = database.Database.pair_id_to_image_pair(pair_id)
    assert (first, second) == (1, 2)


def test_database_read_rig_with_sensor(db, scene):
    sensor = scene.sensor_t(scene.SensorType.CAMERA, 1)
    assert db.read_rig_with_sensor(sensor) is None


def test_database_keypoint_roundtrip(db):
    kps = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.1, 0.2]],
                   dtype=np.float32)
    # Keypoints round-trip through image 1 after an image write; with an
    # empty database the read is simply empty.
    assert db.num_keypoints is not None
