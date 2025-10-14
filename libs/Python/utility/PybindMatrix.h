// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#undef slots
#include <pybind11/numpy.h>

#include "utility/Matrix.h"

namespace py = pybind11;
using namespace cloudViewer;

// type caster: utility::Matrix <-> NumPy-array
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<utility::Matrix<T>> {
public:
    PYBIND11_TYPE_CASTER(utility::Matrix<T>, _("Matrix<T>"));

    // Conversion part 1 (Python -> C++)
    bool load(py::handle src, bool convert) {
        if (!convert && !py::array_t<T>::check_(src)) return false;

        auto buf = py::array_t<T, py::array::c_style |
                                          py::array::forcecast>::ensure(src);
        if (!buf) return false;

        auto dims = buf.ndim();
        if (dims < 1 || dims > 3) return false;

        std::vector<size_t> shape(buf.ndim());

        for (int i = 0; i < buf.ndim(); i++) shape[i] = buf.shape()[i];

        value = utility::Matrix<T>(shape, buf.data());

        return true;
    }

    // Conversion part 2 (C++ -> Python)
    static py::handle cast(const utility::Matrix<T>& src,
                           py::return_value_policy policy,
                           py::handle parent) {
        py::array a(std::move(src.shape()), std::move(src.strides(true)),
                    src.data());
        return a.release();
    }
};

}  // namespace detail
}  // namespace pybind11
