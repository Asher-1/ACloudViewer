// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io -
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

#include "t/geometry/Geometry.h"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <deque>
#include <vector>

#include "pybind/cloudViewer_pybind.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace cloudViewer {
namespace t {
namespace geometry {

void pybind_geometry_class(py::module& m) {
    // cloudViewer.t.geometry.Geometry
    py::class_<Geometry, PyGeometry<Geometry>, std::shared_ptr<Geometry>>
            geometry(m, "Geometry", "The base geometry class.");

    geometry.def("clear", &Geometry::Clear,
                 "Clear all elements in the geometry.")
            .def("is_empty", &Geometry::IsEmpty,
                 "Returns ``True`` iff the geometry is empty.");
    geometry.def_property_readonly("device", &Geometry::GetDevice,
                                   "Returns the device of the geometry.");
    geometry.def_property_readonly("is_cpu", &Geometry::IsCPU,
                                   "Returns true if the geometry is on CPU.");
    geometry.def_property_readonly("is_cuda", &Geometry::IsCUDA,
                                   "Returns true if the geometry is on CUDA.");
    docstring::ClassMethodDocInject(m, "Geometry", "clear");
    docstring::ClassMethodDocInject(m, "Geometry", "is_empty");
}

void pybind_geometry(py::module& m) {
    py::module m_submodule = m.def_submodule(
            "geometry", "Tensor-based geometry defining module.");
    py::bind_vector<std::vector<Metric>>(m_submodule, "VectorMetric");
    
    py::native_enum<Metric>(
            m_submodule, "Metric", "enum.Enum",
            "Enum for metrics for comparing point clouds and triangle meshes.")
            .value("ChamferDistance", Metric::ChamferDistance,
                   "Chamfer Distance")
            .value("HausdorffDistance", Metric::HausdorffDistance,
                   "Hausdorff Distance")
            .value("FScore", Metric::FScore, "F-Score")
            .export_values()
            .finalize();
    py::class_<MetricParameters> metric_parameters(
            m_submodule, "MetricParameters",
            "Holder for various parameters required by metrics.");

    py::native_enum<MethodOBBCreate>(
            m_submodule, "MethodOBBCreate", "enum.Enum",
            "Methods for creating oriented bounding boxes.")
            .value("PCA", MethodOBBCreate::PCA)
            .value("MINIMAL_APPROX", MethodOBBCreate::MINIMAL_APPROX)
            .value("MINIMAL_JYLANKI", MethodOBBCreate::MINIMAL_JYLANKI)
            .export_values()
            .finalize();

    // Use std::deque instead of std::vector to enable automatic Python list /
    // tuple conversion. FIXME: Ideally this should work for std::vector.
    metric_parameters
            .def(py::init([](const std::deque<float>& fsr, size_t nsp) {
                     std::vector<float> fsrvec{fsr.begin(), fsr.end()};
                     return MetricParameters{fsrvec, nsp};
                 }),
                 "fscore_radius"_a = std::deque<float>{0.01f},
                 "n_sampled_points"_a = 1000)
            .def_property(
                    "fscore_radius",
                    [](const MetricParameters& self) {  // getter
                        return std::deque<float>(self.fscore_radius.begin(),
                                                 self.fscore_radius.end());
                    },
                    [](MetricParameters& self,
                       const std::deque<float>& fsr) {  // setter
                        self.fscore_radius =
                                std::vector<float>(fsr.begin(), fsr.end());
                    },
                    "Radius for computing the F-Score. A match between "
                    "a point and its nearest neighbor is sucessful if "
                    "it is within this radius.")
            .def_readwrite("n_sampled_points",
                           &MetricParameters::n_sampled_points,
                           "Points are sampled uniformly from the surface of "
                           "triangle meshes before distance computation. This "
                           "specifies the number of points sampled. No "
                           "sampling is done for point clouds.")
            .def("__repr__", &MetricParameters::ToString);

    pybind_geometry_class(m_submodule);
    pybind_drawable_geometry(m_submodule);
    pybind_tensormap(m_submodule);
    pybind_pointcloud(m_submodule);
    pybind_lineset(m_submodule);
    pybind_trianglemesh(m_submodule);
    pybind_image(m_submodule);
    pybind_boundingvolume(m_submodule);
    pybind_voxel_block_grid(m_submodule);
    pybind_raycasting_scene(m_submodule);
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
