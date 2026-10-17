# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap sensor/bitmap_test.py (Open3D style)."""
import numpy as np
from PIL import Image as PILImage


def test_bitmap_read_to_array(sensor, tmp_path):
    p = tmp_path / "t.png"
    PILImage.new("RGB", (4, 2), (10, 20, 30)).save(str(p))
    bm = sensor.Bitmap()
    assert bm.read(str(p))
    arr = bm.to_array()
    assert arr.shape == (2, 4, 3)
    assert arr[0][0][0] == 10 and arr[0][0][2] == 30


def test_bitmap_rot90(sensor, tmp_path):
    p = tmp_path / "t.png"
    PILImage.new("RGB", (4, 2), (0, 0, 0)).save(str(p))
    bm = sensor.Bitmap()
    assert bm.read(str(p))
    bm.rot90()
    assert bm.width == 2 and bm.height == 4


def test_bitmap_thumbnail(sensor, tmp_path):
    p = tmp_path / "t.png"
    PILImage.new("RGB", (16, 16), (0, 0, 0)).save(str(p))
    bm = sensor.Bitmap()
    assert bm.read(str(p))
    bm.thumbnail(4, 4)
    assert bm.width <= 4 and bm.height <= 4


def test_bitmap_jpeg_quality(sensor, tmp_path):
    p = tmp_path / "t.jpg"
    PILImage.new("RGB", (4, 4)).save(str(p))
    bm = sensor.Bitmap()
    assert bm.read(str(p))
    bm.set_jpeg_quality(85)
    assert bm.jpeg_quality() == 85


def test_bitmap_write_roundtrip(sensor, tmp_path):
    p_in = tmp_path / "in.png"
    PILImage.new("RGB", (4, 4), (5, 5, 5)).save(str(p_in))
    bm = sensor.Bitmap()
    assert bm.read(str(p_in))
    p_out = tmp_path / "out.png"
    assert bm.write(str(p_out))
    bm2 = sensor.Bitmap()
    assert bm2.read(str(p_out))
    assert bm2.width == 4
