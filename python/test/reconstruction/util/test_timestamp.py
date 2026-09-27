# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap util/timestamp_test.py (skipped with reason).

Fork adjudication (batch 24): the nanosecond timestamp helpers
(timestamp_from_seconds / seconds_from_timestamp / timestamp_diff_seconds)
are not bound on recon.util; re-enable when the fork binds them.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="fork does not bind the nanosecond timestamp helpers")


def test_timestamp_helpers_placeholder(recon):
    pytest.skip("fork does not bind the nanosecond timestamp helpers")
