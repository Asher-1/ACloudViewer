// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "core/Tensor.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace core {

void pybind_core(py::module& m);
void pybind_cuda_utils(py::module& m);
void pybind_core_blob(py::module& m);
void pybind_core_dtype(py::module& m);
void pybind_core_device(py::module& m);
void pybind_core_size_vector(py::module& m);
void pybind_core_tensorlist(py::module& m);
void pybind_core_tensor(py::module& m);
void pybind_core_tensor_accessor(py::class_<Tensor>& t);
void pybind_core_linalg(py::module& m);
void pybind_core_kernel(py::module& m);
void pybind_core_hashmap(py::module& m);
void pybind_core_hashset(py::module& m);
void pybind_core_scalar(py::module& m);
void pybind_core_tensor_function(py::module& m);

}  // namespace core
}  // namespace cloudViewer
