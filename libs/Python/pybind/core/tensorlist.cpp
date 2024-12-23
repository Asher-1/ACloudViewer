// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "core/TensorList.h"

#include <vector>

#include "core/Blob.h"
#include "core/CUDAUtils.h"
#include "core/Device.h"
#include "core/Dispatch.h"
#include "core/Dtype.h"
#include "core/SizeVector.h"
#include "core/Tensor.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"
#include "pybind/pybind_utils.h"

namespace cloudViewer {
namespace core {

void pybind_core_tensorlist(py::module& m) {
    py::class_<TensorList> tensorlist(
            m, "TensorList",
            "A TensorList is an extendable tensor at the 0-th dimension.");

    // Constructors.
    tensorlist.def(py::init([](const SizeVector& element_shape,
                               const Dtype& dtype, const Device& device) {
                       return new TensorList(element_shape, dtype, device);
                   }),
                   "element_shape"_a, "dtype"_a, "device"_a);
    tensorlist.def(py::init([](const std::vector<Tensor>& tensors) {
                       return new TensorList(tensors);
                   }),
                   "tensors"_a);
    tensorlist.def(py::init([](const TensorList& other) {
                       return new TensorList(other);
                   }),
                   "other"_a);

    // Factory function.
    tensorlist.def_static("from_tensor", &TensorList::FromTensor, "tensor"_a,
                          "inplace"_a = false);

    // Copiers.
    tensorlist.def("copy_from", &TensorList::CopyFrom);
    tensorlist.def("clone", &TensorList::Clone);

    // Accessors.
    tensorlist.def("__getitem__",
                   [](TensorList& tl, int64_t index) { return tl[index]; });
    tensorlist.def("__setitem__",
                   [](TensorList& tl, int64_t index, const Tensor& value) {
                       tl[index] = value;
                   });
    tensorlist.def("as_tensor",
                   [](const TensorList& tl) { return tl.AsTensor(); });
    tensorlist.def("__repr__",
                   [](const TensorList& tl) { return tl.ToString(); });
    tensorlist.def("__str__",
                   [](const TensorList& tl) { return tl.ToString(); });

    // Manipulations.
    tensorlist.def("push_back", &TensorList::PushBack);
    tensorlist.def("resize", &TensorList::Resize);
    tensorlist.def("extend", &TensorList::Extend);
    tensorlist.def("__iadd__", &TensorList::operator+=);
    tensorlist.def("__add__", &TensorList::operator+);
    tensorlist.def_static("concat", &TensorList::Concatenate);

    // Python list properties.
    // TODO: make TensorList behave more like regular python list, see
    // std_bind.h.
    tensorlist.def("__len__", &TensorList::GetSize);

    // Properties.
    tensorlist.def_property_readonly("size", &TensorList::GetSize);
    tensorlist.def_property_readonly("element_shape",
                                     &TensorList::GetElementShape);
    tensorlist.def_property_readonly("dtype", &TensorList::GetDtype);
    tensorlist.def_property_readonly("device", &TensorList::GetDevice);
    tensorlist.def_property_readonly("is_resizable", &TensorList::IsResizable);
}

}  // namespace core
}  // namespace cloudViewer
