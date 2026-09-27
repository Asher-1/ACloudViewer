# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap dataclass_test.py (skipped with reason).

Fork adjudication (batch 24): the upstream dataclass protocol surface
(summary()/todict()/mergedict()/pickle round-trips with qualified enum
reprs) is a pybind11 dataclass-protocol layer the fork has not adopted; the
option value semantics are covered by the C++ gate tests instead. Re-enable
when the fork binds the dataclass protocol.
"""
import pytest

pytestmark = pytest.mark.skip(
    reason="fork does not bind the upstream dataclass protocol surface")


def test_dataclass_protocol_placeholder(recon):
    pytest.skip("fork does not bind the upstream dataclass protocol surface")
