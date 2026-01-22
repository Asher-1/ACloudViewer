# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import pytest
import mltest

# Skip all tests if the ml ops were not built
pytestmark = mltest.default_marks


@mltest.parametrize.ml_gpu_only
def test_cublas_matmul(ml):
    # This test checks if calling cublas functionality from cloudViewer and the ml framework works.

    rng = np.random.RandomState(123)

    n = 20
    arr = rng.rand(n, n).astype(np.float32)

    # do matmul with cloudViewer
    A = cv3d.core.Tensor.from_numpy(arr).cuda()
    B = A @ A

    # now use the ml framework cublas
    C = mltest.run_op(ml, ml.device, True, ml.module.matmul, arr, arr)

    np.testing.assert_allclose(B.cpu().numpy(), C)
