// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#pragma once

// PYBIND_11
#undef slots
#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// CV_CORE_LIB
#include <Eigen.h>
#include <CVGeom.h>
#include <Optional.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvGLMatrix.h>
#include <ecvPointCloud.h>

// OPENGL_ENGIEN_LIB
#include <pipelines/registration/PoseGraph.h>

// QT
#include <QString>


namespace py = pybind11;
using namespace py::literals;

typedef std::vector<Eigen::Matrix4d, cloudViewer::utility::Matrix4d_allocator>
        temp_eigen_matrix4d;
typedef std::vector<Eigen::Vector4i, cloudViewer::utility::Vector4i_allocator>
        temp_eigen_vector4i;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);
PYBIND11_MAKE_OPAQUE(temp_eigen_matrix4d);
PYBIND11_MAKE_OPAQUE(temp_eigen_vector4i);
PYBIND11_MAKE_OPAQUE(std::vector<cloudViewer::pipelines::registration::PoseGraphEdge>);
PYBIND11_MAKE_OPAQUE(std::vector<cloudViewer::pipelines::registration::PoseGraphNode>);

PYBIND11_MAKE_OPAQUE(std::vector<CCVector3>);
PYBIND11_MAKE_OPAQUE(std::vector<CCVector3d>);
PYBIND11_MAKE_OPAQUE(ccGLMatrixd);
PYBIND11_MAKE_OPAQUE(QString);
PYBIND11_MAKE_OPAQUE(std::vector<QString>);

// some helper functions
namespace pybind11 {
namespace detail {

template <typename T, typename Class_>
void bind_default_constructor(Class_ &cl) {
    cl.def(py::init([]() { return new T(); }), "Default constructor");
}

template <typename T, typename Class_>
void bind_copy_functions(Class_ &cl) {
    cl.def(py::init([](const T &cp) { return new T(cp); }), "Copy constructor");
    cl.def("__copy__", [](T &v) { return T(v); });
    cl.def("__deepcopy__", [](T &v, py::dict &memo) { return T(v); });
}

/// Custom pybind11 type caster for cloudViewer::utility::optional, which backports
/// C++17's `std::optional`. We need compiler supporting C++14 or newer.
/// Typically, this can be used to handle "None" parameters from Python.
///
/// Example python function:
///
/// ```python
/// def add(a, b=None):
///     # Assuming a, b are int.
///     if b is None:
///         return a
///     else:
///         return a + b
/// ```
///
/// Here's the equivalent C++ implementation:
///
/// ```cpp
/// m.def("add",
///     [](int a, cloudViewer::utility::optional<int> b) {
///         if (!b.has_value()) {
///             return a;
///         } else {
///             return a + b.value();
///         }
///     },
///     py::arg("a"), py::arg("b") = py::none()
/// );
/// ```
///
/// Then this function can be called with:
///
/// ```python
/// add(1)
/// add(1, 2)
/// add(1, b=2)
/// add(1, b=None)
/// ```
template <typename T>
struct cloudViewer_optional_caster {
    using value_conv = make_caster<typename T::value_type>;

    template <typename T_>
    static handle cast(T_ &&src, return_value_policy policy, handle parent) {
        if (!src) return none().inc_ref();
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<T>::policy(policy);
        }
        return value_conv::cast(*std::forward<T_>(src), policy, parent);
    }

    bool load(handle src, bool convert) {
        if (!src) {
            return false;
        } else if (src.is_none()) {
            return true;  // default-constructed value is already empty
        }
        value_conv inner_caster;
        if (!inner_caster.load(src, convert)) return false;

        value.emplace(
                cast_op<typename T::value_type &&>(std::move(inner_caster)));
        return true;
    }

    PYBIND11_TYPE_CASTER(T, _("Optional[") + value_conv::name + _("]"));
};

template <typename T>
struct type_caster<cloudViewer::utility::optional<T>>
    : public cloudViewer_optional_caster<cloudViewer::utility::optional<T>> {};

template <>
struct type_caster<cloudViewer::utility::nullopt_t>
    : public void_caster<cloudViewer::utility::nullopt_t> {};

}  // namespace detail
}  // namespace pybind11

