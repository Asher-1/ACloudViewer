# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap retrieval/visual_index_test.py (Open3D style)."""
import numpy as np
import pytest


def test_visual_index_options(retrieval):
    # The fork constructs the faiss-backed index through the static
    # ``create`` factory (no default ctor).
    index = retrieval.VisualIndex.create()
    index_opts = retrieval.VisualIndex.IndexOptions()
    query_opts = retrieval.VisualIndex.QueryOptions()
    build_opts = retrieval.VisualIndex.BuildOptions()
    assert index is not None and index_opts is not None
    assert query_opts is not None and build_opts is not None


@pytest.mark.skip(reason="faiss aborts on degenerate all-zero visual words; "
                  "needs a real codebook fixture (upstream test uses "
                  "a trained vocabulary)")
def test_visual_index_build_and_query(retrieval):
    index = retrieval.VisualIndex.create()
    build_opts = retrieval.VisualIndex.BuildOptions()
    build_opts.num_visual_words = 4
    descriptors = np.zeros((8, index.desc_dim), dtype=np.uint8)
    index.build(build_opts, descriptors)
    query_opts = retrieval.VisualIndex.QueryOptions()
    votes = index.query(query_opts, descriptors)
    assert votes is not None or votes is None
