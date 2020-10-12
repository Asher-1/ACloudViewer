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

#ifndef CV_CLOUDVIEWER_PYBIND_HEADER
#define CV_CLOUDVIEWER_PYBIND_HEADER

// CV_CORE_LIB
#include <Eigen.h>
#include <CVGeom.h>

// ECV_DB_LIB
#include <ecvGLMatrix.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

// OPENGL_ENGIEN_LIB
#include <Registration/PoseGraph.h>

// QT
#include <QString>

// PYBIND_11
#include <pybind11/detail/internals.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>  // Include first to suppress compiler warnings
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace py::literals;

typedef std::vector<Eigen::Matrix4d, CVLib::utility::Matrix4d_allocator>
        temp_eigen_matrix4d;
typedef std::vector<Eigen::Vector4i, CVLib::utility::Vector4i_allocator>
        temp_eigen_vector4i;

PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);
PYBIND11_MAKE_OPAQUE(temp_eigen_matrix4d);
PYBIND11_MAKE_OPAQUE(temp_eigen_vector4i);
PYBIND11_MAKE_OPAQUE(std::vector<cloudViewer::registration::PoseGraphEdge>);
PYBIND11_MAKE_OPAQUE(std::vector<cloudViewer::registration::PoseGraphNode>);

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

}  // namespace detail
}  // namespace pybind11

#endif // CV_CLOUDVIEWER_PYBIND_HEADER