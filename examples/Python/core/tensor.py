#!/usr/bin/env python
# coding: utf-8


import cloudViewer.core as cv3c
import numpy as np


# # Tensor
# 
# Tensor is a "view" of a data Blob with shape, stride, and a data pointer. I
# t is a multidimensional and homogeneous matrix containing elements of single data type.
# It is used in CloudViewer to perform numerical operations. It supports GPU operations as well.
# 
# ## Tensor creation
# 
# Tensor can be created from list, numpy array, another tensor.
# A tensor of specific data type and device can be constructed by passing a ```cv3c.Dtype``` and/or ```cv3c.Device```
# to a constructor. If not passed, the default data type is inferred from the data, and the default device is CPU.
# Note that while creating tensor from a list or numpy array, the underlying memory is not shared and a copy is created.


# Tensor from list.
a = cv3c.Tensor([0, 1, 2])
print("Created from list:\n{}".format(a))

# Tensor from Numpy.
a = cv3c.Tensor(np.array([0, 1, 2]))
print("\nCreated from numpy array:\n{}".format(a))

# Dtype and inferred from list.
a_float = cv3c.Tensor([0.0, 1.0, 2.0])
print("\nDefault dtype and device:\n{}".format(a_float))

# Specify dtype.
a = cv3c.Tensor(np.array([0, 1, 2]), dtype=cv3c.Dtype.Float64)
print("\nSpecified data type:\n{}".format(a))

# Specify device.
a = cv3c.Tensor(np.array([0, 1, 2]), device=cv3c.Device("CUDA:0"))
print("\nSpecified device:\n{}".format(a))


#    Tensor can also be created from another tensor by invoking the copy constructor.
#    This is a shallow copy, the data_ptr will be copied but the memory it points to will not be copied.


# Shallow copy constructor.
vals = np.array([1, 2, 3])
src = cv3c.Tensor(vals)
dst = src
src[0] += 10
print("\n")
# Changes in one will get reflected in other.
print("Source tensor:\n{}".format(src))
print("\nTarget tensor:\n{}".format(dst))


# ## Properties of a tensor


vals = np.array((range(24))).reshape(2, 3, 4)
a = cv3c.Tensor(vals, dtype=cv3c.Dtype.Float64, device=cv3c.Device("CUDA:0"))
print("\n")
print(f"a.shape: {a.shape}")
print(f"a.strides: {a.strides}")
print(f"a.dtype: {a.dtype}")
print(f"a.device: {a.device}")
print(f"a.ndim: {a.ndim}")


# ## Copy & device transfer
# We can transfer tensors across host and multiple devices.

print("\n")
# Host -> Device.
a_cpu = cv3c.Tensor([0, 1, 2])
a_gpu = a_cpu.cuda(0)
print(a_gpu)

# Device -> Host.
a_gpu = cv3c.Tensor([0, 1, 2], device=cv3c.Device("CUDA:0"))
a_cpu = a_gpu.cpu()
print(a_cpu)

# Device -> another Device.
a_gpu_0 = cv3c.Tensor([0, 1, 2], device=cv3c.Device("CUDA:0"))
a_gpu_1 = a_gpu_0.cuda(1)
print(a_gpu_1)


# ## Data Types
# 
# CloudViewer defines seven tensor data types.
# 
# | Data type                | dtype               | byte_size  |
# |--------------------------|---------------------|------------|
# | Uninitialized Tensor     | cv3c.Dtype.Undefined | -          |
# | 32-bit floating point    | cv3c.Dtype.Float32   | 4          |
# | 64-bit floating point    | cv3c.Dtype.Float64   | 8          |
# | 32-bit integer (signed)  | cv3c.Dtype.Int32     | 4          |
# | 64-bit integer (signed)  | cv3c.Dtype.Int64     | 8          |
# | 8-bit integer (unsigned) | cv3c.Dtype.UInt8     | 1          |
# | Boolean                  | cv3c.Dtype.Bool      | 1          |
# 
# ### Type casting
# We can cast tensor's data type. Forced casting might result in data loss.

print("\n")

# E.g. float -> int
a = cv3c.Tensor([0.1, 1.5, 2.7])
b = a.to(cv3c.Dtype.Int32)
print(a)
print(b)


# E.g. int -> float
a = cv3c.Tensor([1, 2, 3])
b = a.to(cv3c.Dtype.Float32)
print(a)
print(b)


# ## Numpy I/O with direct memory map
# 
# Tensors created by passing numpy array to the constructor(```cv3c.Tensor(np.array(...)```)
# do not share memory with the numpy aray. To have shared memory,
# you can use ```cv3c.Tensor.from_numpy(...)``` and ```cv3c.Tensor.numpy(...)```.
# Changes in either of them will get reflected in other.

# Using constructor.
print("\n")
np_a = np.ones((5,), dtype=np.int32)
o3_a = cv3c.Tensor(np_a)
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to numpy array will not reflect as memory is not shared.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")


# From numpy.
print("\n")
np_a = np.ones((5,), dtype=np.int32)
o3_a = cv3c.Tensor.from_numpy(np_a)

# Changes to numpy array reflects on cloudViewer Tensor and vice versa.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")


# To numpy.
print("\n")
o3_a = cv3c.Tensor([1, 1, 1, 1, 1], dtype=cv3c.Dtype.Int32)
np_a = o3_a.numpy()

# Changes to numpy array reflects on cloudViewer Tensor and vice versa.
np_a[0] += 100
o3_a[1] += 200
print(f"np_a: {np_a}")
print(f"o3_a: {o3_a}")

# For CUDA Tensor, call cpu() before calling numpy().
o3_a = cv3c.Tensor([1, 1, 1, 1, 1], device=cv3c.Device("CUDA:0"))
print(f"\no3_a.cpu().numpy(): {o3_a.cpu().numpy()}")


# ## PyTorch I/O with DLPack memory map
# We can convert tensors from/to DLManagedTensor.


import torch
import torch.utils.dlpack

print("\n")
# From PyTorch
th_a = torch.ones((5,)).cuda(0)
o3_a = cv3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to PyTorch array reflects on cloudViewer Tensor and vice versa
th_a[0] = 100
o3_a[1] = 200
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")


print("\n")
# To PyTorch
o3_a = cv3c.Tensor([1, 1, 1, 1, 1], device=cv3c.Device("CUDA:0"))
th_a = torch.utils.dlpack.from_dlpack(o3_a.to_dlpack())
o3_a = cv3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(th_a))
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")
print("")

# Changes to PyTorch array reflects on cloudViewer Tensor and vice versa
th_a[0] = 100
o3_a[1] = 200
print(f"th_a: {th_a}")
print(f"o3_a: {o3_a}")


# ## Binary element-wise operation:
# 
# Supported element-wise binary operations are:
# 1. `Add(+)`
# 2. `Sub(-)`
# 3. `Mul(*)`
# 4. `Div(/)`
# 5. `Add_(+=)`
# 6. `Sub_(-=)`
# 7. `Mul_(*=)`
# 8. `Div_(/=)`
# 
# Note that the operands have to be of same Device, dtype and Broadcast compatible.

print("\n")
a = cv3c.Tensor([1, 1, 1], dtype=cv3c.Dtype.Float32)
b = cv3c.Tensor([2, 2, 2], dtype=cv3c.Dtype.Float32)
print("a + b = {}".format(a + b))
print("a - b = {}".format(a - b))
print("a * b = {}".format(a * b))
print("a / b = {}".format(a / b))


# Broadcasting follows the same numpy broadcasting rule as given
# [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).<br>
# Automatic type casting is done in a way to avoid data loss.


# Automatic broadcasting.
print("\n")
a = cv3c.Tensor.ones((2, 3), dtype=cv3c.Dtype.Float32)
b = cv3c.Tensor.ones((3,), dtype=cv3c.Dtype.Float32)
print("a + b = \n{}\n".format(a + b))

# Automatic type casting.
a = a[0]
print("a + 1 = {}".format(a + 1))  # Float + Int -> Float.
print("a + True = {}".format(a + True))  # Float + Bool -> Float.

# Inplace.
a -= True
print("a = {}".format(a))


# ## Unary element-wise operation:
# Supported unary element-wise operations are:
# 1. `sqrt`, `sqrt_`(inplace))
# 2. `sin`, `sin_`
# 3. `cos`, `cos_`
# 4. `neg`, `neg_`
# 5. `exp`, `exp_`
# 6. `abs`, `abs_`
# 

print("\n")
a = cv3c.Tensor([4, 9, 16], dtype=cv3c.Dtype.Float32)
print("a = {}\n".format(a))
print("a.sqrt = {}\n".format(a.sqrt()))
print("a.sin = {}\n".format(a.sin()))
print("a.cos = {}\n".format(a.cos()))

# Inplace operation
a.sqrt_()
print(a)


# ## Reduction:
# 
# CloudViewer supports following reduction operations.
# 1. `sum` - returns a tensor with sum of values over a given axis.
# 2. `mean` - returns a tensor with mean of values over a given axis.
# 3. `prod` - returns a tensor with product of values over a given axis.
# 4. `min` - returns a tensor of minimum values along a given axis.
# 5. `max` - returns a tensor of maximum values along a given axis.
# 6. `argmin` - returns a tensor of minimum value indices over a given axis.
# 7. `argmax` - returns a tensor of maximum value indices over a given axis.

print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)
print("a.sum = {}\n".format(a.sum()))
print("a.min = {}\n".format(a.min()))
print("a.ArgMax = {}\n".format(a.argmax()))


# With specified dimension.
print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)

print("Along dim=0\n{}".format(a.sum(dim=(0))))
print("Along dim=(0, 2)\n{}\n".format(a.sum(dim=(0, 2))))

# Retention of reduced dimension.
print("Shape without retention : {}".format(a.sum(dim=(0, 2)).shape))
print("Shape with retention : {}".format(a.sum(dim=(0, 2), keepdim=True).shape))


# ## Slicing, indexing, getitem, and setitem
# 
# Basic slicing is done by passing an integer, slice object(```start:stop:step```),
# index array or boolean array. Slicing and indexing produce a view of the tensor.
# Hence any change in it will also get reflected in the original tensor.

print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)
print("a = \n{}\n".format(a))

# Indexing __getitem__.
print("a[1, 2] = {}\n".format(a[1, 2]))

# Slicing __getitem__.
print("a[1:] = \n{}\n".format(a[1:]))

# slice object.
print("a[:, 0:3:2, :] = \n{}\n".format(a[:, 0:3:2, :]))

# Combined __getitem__
print("a[:-1, 0:3:2, 2] = \n{}\n".format(a[:-1, 0:3:2, 2]))


print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)

# Changes get reflected.
b = a[:-1, 0:3:2, 2]
b[0] += 100
print("b = {}\n".format(b))
print("a = \n{}".format(a))


print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)

# Example __setitem__
a[:, :, 2] += 100
print(a)


# ## Advanced indexing
# 
# Advanced indexing is triggered while passing an index array or a boolean array or their
# combination with integer/slice object. Note that advanced indexing always returns a copy of the data
# (contrast with basic slicing that returns a view).
# ### Integer array indexing
# Integer array indexing allows selection of arbitrary items in the tensor based on their
# dimensional index. Indexes passed should be broadcast compatible.

print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)

# Along each dimension, a specific element is selected.
print("a[[0, 1], [1, 2], [1, 0]] = {}\n".format(a[[0, 1], [1, 2], [1, 0]]))

# Changes not reflected as it is a copy.
b = a[[0, 0], [0, 1], [1, 1]]
b[0] += 100
print("b = {}\n".format(b))
print("a[[0, 0], [0, 1], [1, 1]] = {}".format(a[[0, 0], [0, 1], [1, 1]]))


# ### Combining advanced and basic indexing
# When there is at least one slice(```:```), ellipse(```...```), or newaxis in the index,
# then the behaviour can be more complicated. It is like concatenating the indexing result for
# each advanced index element. Under the advanced indexing mode, some preprocessing is done before
# sending to the advanced indexing engine.
# 1. Specific index positions are converted to a Indextensor with the specified index.
# 2. If slice is non-full slice, then we slice the tensor first, then use full slice for advanced indexing engine.
# 
# ```dst = src[1, 0:2, [1, 2]]``` is done in two steps:<br>
# ```temp = src[:, 0:2, :]```<br>
# ```dst = temp[[1], :, [1, 2]]```
# 
# There are two parts to the indexing operation, the subspace defined by the basic indexing,
# and the subspace from the advanced indexing part.
# 
# 1. The advanced indexes are separated by a slice, Ellipse, or newaxis. For example ```x[arr1, :, arr2]```.
# 2. The advanced indexes are all next to each other. For example ```x[..., arr1, arr2, :]```,
# but not ```x[arr1, :, 1]``` since ```1``` is an advanced index here.
# 
# In the first case, the dimensions resulting from the advanced indexing operation come first in the result array,
# and the subspace dimensions after that. In the second case, the dimensions from the advanced indexing operations
# are inserted into the result array at the same spot as they were in the initial array.

print("\n")
vals = np.array(range(24)).reshape((2, 3, 4))
a = cv3c.Tensor(vals)

print("a[1, 0:2, [1, 2]] = \n{}\n".format(a[1, 0:2, [1, 2]]))

# Subtle difference in selection and advanced indexing.
print("a[(0, 1)] = {}\n".format(a[(0, 1)]))
print("a[[0, 1]] = \n{}\n".format(a[[0, 1]]))

a = cv3c.Tensor(np.array(range(120)).reshape((2, 3, 4, 5)))

# Interleaving slice and advanced indexing.
print("a[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]] = \n{}\n".format(
    a[1, [[1, 2], [2, 1]], 0:4:2, [3, 4]]))


# ### Boolean array indexing
# Advanced indexing gets triggered when we pass a boolean array as an index, or it is returned
# from comparision operators. Boolean array should have exactly as many dimensions as it is supposed to work with.

print("\n")
a = cv3c.Tensor(np.array([1, -1, -2, 3]))
print("a = {}\n".format(a))

# Add constant to all negative numbers.
a[a < 0] += 20
print("a = {}\n".format(a))


# ## Logical operations
# 
# CloudViewer supports following logical operators:
# 1. `logical_and` - returns tensor with element wise logical AND.
# 2. `logical_or`  - returns tensor with element wise logical OR.
# 3. `logical_xor` - returns tensor with element wise logical XOR.
# 4. `logical_not` - returns tensor with element wise logical NOT.
# 5. `all`         - returns true if all elements in the tensor are true.
# 6. `any`         - returns true if any element in the tensor is true.
# 7. `allclose`    - returns true if two tensors are element wise equal within a tolerance.
# 8. `isclose`     - returns tensor with element wise ```allclose``` operation.
# 9. `issame`      - returns true if and only if two tensors are same(even same underlying memory).
# 

print("\n")
a = cv3c.Tensor(np.array([True, False, True, False]))
b = cv3c.Tensor(np.array([True, True, False, False]))

print("a AND b = {}".format(a.logical_and(b)))
print("a OR b = {}".format(a.logical_or(b)))
print("a XOR b = {}".format(a.logical_xor(b)))
print("NOT a = {}\n".format(a.logical_not()))

# Only works for boolean tensors.
print("a.any = {}".format(a.any()))
print("a.all = {}\n".format(a.all()))

# If tensor is not boolean, 0 will be treated as False, while non-zero as true.
# The tensor will be filled with 0 or 1 casted to tensor's dtype.
c = cv3c.Tensor(np.array([2.0, 0.0, 3.5, 0.0]))
d = cv3c.Tensor(np.array([0.0, 3.0, 1.5, 0.0]))
print("c AND d = {}".format(c.logical_and(d)))


a = cv3c.Tensor(np.array([1, 2, 3, 4]), dtype=cv3c.Dtype.Float64)
b = cv3c.Tensor(np.array([1, 1.99999, 3, 4]))

# Throws exception if the device/dtype is not same.
# Returns false if the shape is not same.
print("allclose : {}".format(a.allclose(b)))

# Throws exception if the device/dtype/shape is not same.
print("isclose : {}".format(a.isclose(b)))

# Returns false if the device/dtype/shape/ is not same.
print("issame : {}".format(a.issame(b)))


# ## Comparision Operations

print("\n")
a = cv3c.Tensor([0, 1, -1])
b = cv3c.Tensor([0, 0, 0])

print("a > b = {}".format(a > b))
print("a >= b = {}".format(a >= b))
print("a < b = {}".format(a < b))
print("a <= b = {}".format(a <= b))
print("a == b = {}".format(a == b))
print("a != b = {}".format(a != b))

# Throws exception if device/dtype is not shape.
# If shape is not same, then tensors should be broadcast compatible.
print("a > b[0] = {}".format(a > b[0]))


# ## Nonzero operations
# 1. When ```as_tuple``` is ```False```(default), it returns a tensor indices of the elements that are non-zero.
# Each row in the result contains the indices of a non-zero element in the input. If the input has $n$ dimensions,
# then the resulting tensor is of size $(z x n)$,
# where $z$ is the total number of non-zero elements in the input tensor.
# 2. When ```as_tuple``` is ```True```, it returns a tuple of 1D tensors,
# one for each dimension in input, each containing the indices of all non-zero elements of input.
# If the input has $n$ dimension, then the resulting tuple contains $n$ tensors of size $z$,
# where $z$ is the total number of non-zero elements in the input tensor.
# 

print("\n")
a = cv3c.Tensor([[3, 0, 0], [0, 4, 0], [5, 6, 0]])

print("a = \n{}\n".format(a))
print("a.nonzero() = \n{}\n".format(a.nonzero()))
print("a.nonzero(as_tuple = 1) = \n{}".format(a.nonzero(as_tuple=1)))


# ## TensorList
# A tensorlist is a list of tensors of the same shape, similar to ```std::vector<Tensor>```.
# Internally, a tensorlist stores the tensors in one big internal tensor,
# where the begin dimension of the internal tensor is extendable.
# This enables storing of 3D points, colours in a contiguous manner.

print("\n")
vals = np.array(range(24), dtype=np.float32).reshape((2, 3, 4))
t = cv3c.Tensor(vals)

# TensorList with single Tensor.
b = cv3c.TensorList(t)
print("b = {}".format(b))

# Empty TensorList.
a = cv3c.TensorList(shape=[2, 3, 4])
print("a = {}".format(a))
print("a.size = {}".format(a.size))
a.resize(3)
print("a = {}".format(a))
print("a.size = {}".format(a.size))

a.push_back(t)

print("a = {}".format(a))
print("a.size = {}".format(a.size))

print("a.is_resizable = {}".format(a.is_resizable))


# ### from_tensor
# We can create tensorlist from a single tensor where we breaking first dimension into multiple tensors.
# The first dimension of the tensor will be used as the `size` dimension of the tensorlist.
# Remaining dimensions will be used as the element shape of the tensor list. For example,
# if the input tensor has shape `(2, 3, 4)`, the resulting tensorlist will have size 2 and element shape `(3, 4)`.
# Here the memory will be copied by default.
# If `inplace == true`, the tensorlist will share the same memory with the input tensor.
# The input tensor must be contiguous. The resulting tensorlist will not be resizable,
# and hence we cannot do certain operations like resize, push_back, extend, concatenate, and clear.
# 
# ### from_tensors
# Tensorlist can also be created from a list of tensors. The tensors must have the same shape, dtype and device.
# Here the values will be copied.


vals = np.array(range(24), dtype=np.float32).reshape((2, 3, 4))

# TensorList from tensor.
c = cv3c.TensorList.from_tensor(cv3c.Tensor(vals))
print("from tensor = {}\n".format(c))

# TensorList from multiple tensors.
b = cv3c.TensorList([cv3c.Tensor(vals[0]), cv3c.Tensor(vals[1])])
print("tensors = {}\n".format(b))
b = cv3c.TensorList.from_tensors([cv3c.Tensor(vals[0]), cv3c.Tensor(vals[1])])
print("from tensors = {}\n".format(b))

d = cv3c.TensorList(b)
print("from tensorlist = {}\n".format(d))

# Below operations are only valid for resizable tensorlist.
# Concatenate TensorLists.
print("b + c = {}".format(b + c))
print("concat(b, c) = {}\n".format(cv3c.TensorList.concat(b, c)))

# Append a Tensor to TensorList.
d.push_back(cv3c.Tensor(vals[0]))
print("d = {}\n".format(d))

# Append a TensorList to another TensorList.
d.extend(c)
print("extended d = {}".format(d))
