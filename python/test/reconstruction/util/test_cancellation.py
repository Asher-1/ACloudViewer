# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap util/cancellation_test.py (Open3D style).

Fork adaptation: the token surface lives on recon.util; the cancelled
pipeline-entry assertions from upstream (extract_features/bundle_adjustment
cancellation_token kwargs) are covered by the fork engine tests and are not
part of the fork's option surface, so only the token lifecycle is asserted.
"""


def test_cancellation_token(recon):
    token = recon.util.CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled
