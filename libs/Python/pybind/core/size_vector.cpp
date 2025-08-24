// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/SizeVector.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"
#include "pybind/cloudViewer_pybind.h"

namespace cloudViewer {
namespace core {

void pybind_core_size_vector(py::module& m) {
    // In Python, We want (3), (3,), [3] and [3,] to represent the same thing.
    // The following are all equivalent to core::SizeVector({3}):
    // - cv3d.core.SizeVector(3)     # int
    // - cv3d.core.SizeVector((3))   # int, not tuple
    // - cv3d.core.SizeVector((3,))  # tuple
    // - cv3d.core.SizeVector([3])   # list
    // - cv3d.core.SizeVector([3,])  # list
    //
    // Difference between C++ and Python:
    // - cv3d.core.SizeVector(3) creates a 1-D SizeVector: {3}.
    // - core::SizeVector(3) creates a 3-D SizeVector: {0, 0, 0}.
    //
    // The API difference is due to the NumPy convention which allows integer to
    // represent a 1-element tuple, and the C++ constructor for vectors.
    auto sv = py::bind_vector<SizeVector>(
            m, "SizeVector",
            "A vector of integers for specifying shape, strides, etc.");
    sv.def(py::init([](int64_t i) { return SizeVector({i}); }));
    py::implicitly_convertible<py::int_, SizeVector>();

    // Allows tuple and list implicit conversions to SizeVector.
    py::implicitly_convertible<py::tuple, SizeVector>();
    py::implicitly_convertible<py::list, SizeVector>();
    auto dsv = py::bind_vector<DynamicSizeVector>(
                m, "DynamicSizeVector",
                "A vector of integers for specifying shape, strides, etc. Some "
                "elements can be None.");
    dsv.def("__repr__",
            [](const DynamicSizeVector& dsv) { return dsv.ToString(); });
}

}  // namespace core
}  // namespace cloudViewer
