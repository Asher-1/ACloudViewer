# ----------------------------------------------------------------------------
# -                        CloudViewer: www.erow.cn                          -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.erow.cn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if cv3d.__DEVICE_API__ == 'cuda':
    from cloudViewer.cuda.pybind.core import (Tensor, Hashmap, Dtype, DtypeCode,
                                              Device, cuda, nns, SizeVector,
                                              DynamicSizeVector, matmul,
                                              lstsq, solve, inv, svd, TensorList)
else:
    from cloudViewer.cpu.pybind.core import (Tensor, Hashmap, Dtype, DtypeCode,
                                             Device, cuda, nns, SizeVector,
                                             DynamicSizeVector, matmul,
                                             lstsq, solve, inv, svd, TensorList)


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

    @staticmethod
    def from_tensors(tensors):
        return TensorList(tensors)

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
                "shape must be int, list or tuple, but was {}.".format(type(shape)))
