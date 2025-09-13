# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if cv3d.__DEVICE_API__ == 'cuda':
    from cloudViewer.cuda.pybind.core import *
else:
    from cloudViewer.cpu.pybind.core import *


def _numpy_dtype_to_dtype(numpy_dtype):
    if numpy_dtype == np.float32:
        return Dtype.Float32
    elif numpy_dtype == np.float64:
        return Dtype.Float64
    elif numpy_dtype == np.int32:
        return Dtype.Int32
    elif numpy_dtype == np.int64:
        return Dtype.Int64
    elif numpy_dtype == np.uint8:
        return Dtype.UInt8
    elif numpy_dtype == np.uint16:
        return Dtype.UInt16
    elif numpy_dtype == np.bool:
        return Dtype.Bool
    else:
        raise ValueError("Unsupported numpy dtype:", numpy_dtype)


class TensorList(cv3d.pybind.core.TensorList):
    """
    CloudViewer TensorList class. A TensorList is a view of list of Tensor.
    """

    def __init__(self, shape, dtype=None, device=None):
        # input shape is regarded as a single Tensor
        if isinstance(shape, cv3d.pybind.core.Tensor):
            shape = [shape]

        # input shape is regarded as a TensorList
        if isinstance(shape, TensorList):
            return super(TensorList, self).__init__(shape)

        # input shape is regarded as a list, tuple or np.ndarray of tensors
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()
        if isinstance(shape, (tuple, list)) and len(shape) > 0:
            if isinstance(shape[0], cv3d.pybind.core.Tensor):
                return super(TensorList, self).__init__(shape)

        # input shape is regarded as the element shape of Tensorlist
        shape = self._reduction_shape_to_size_vector(shape)
        if dtype is None:
            dtype = _numpy_dtype_to_dtype(np.float32)
        if device is None:
            device = Device("CPU:0")
        super(TensorList, self).__init__(shape, dtype, device)

    def _reduction_shape_to_size_vector(self, shape):
        if shape is None:
            return SizeVector(list(range(self.ndim)))
        elif isinstance(shape, SizeVector):
            return shape
        elif isinstance(shape, int):
            return SizeVector([shape])
        elif isinstance(shape, list) or isinstance(shape, tuple):
            return SizeVector(shape)
        else:
            raise TypeError(
                "shape must be int, list or tuple, but was {}.".format(
                    type(shape)))
