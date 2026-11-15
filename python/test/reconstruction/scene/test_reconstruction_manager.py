# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/reconstruction_manager_test.py."""
import pytest


def test_reconstruction_manager_add_get_delete(scene):
    manager = scene.ReconstructionManager()
    assert manager.size == 0
    idx = manager.add()
    assert manager.size == 1
    manager.get(idx)
    manager.delete(idx)
    assert manager.size == 0


def test_reconstruction_manager_clear(scene):
    manager = scene.ReconstructionManager()
    manager.add()
    manager.add()
    manager.clear()
    assert manager.size == 0
