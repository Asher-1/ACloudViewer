# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap util/logging_test.py (Open3D style).

Fork adaptation: only the members the fork binds on recon.util.logging are
asserted."""
import pytest


def test_logging_module_exists(recon):
    assert hasattr(recon.util, "logging")


def test_logging_level_members(recon):
    logging = recon.util.logging
    for name in ["INFO", "WARNING", "ERROR", "FATAL"]:
        assert hasattr(logging, name), name
