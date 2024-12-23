// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
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

#include <LineSet.h>
#include <camera/PinholeCameraIntrinsic.h>
#include <ecvPointCloud.h>
#include <ecvTetraMesh.h>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace cloudViewer {
namespace geometry {

void pybind_lineset(py::module &m) {
    py::class_<LineSet, PyGeometry<LineSet>, std::shared_ptr<LineSet>,
               ccHObject>
            lineset(m, "LineSet",
                    "LineSet define a sets of lines in 3D. A typical "
                    "application is to display the point cloud correspondence "
                    "pairs.");
    py::detail::bind_default_constructor<LineSet>(lineset);
    py::detail::bind_copy_functions<LineSet>(lineset);
    lineset.def(py::init<const std::vector<Eigen::Vector3d> &,
                         const std::vector<Eigen::Vector2i> &>(),
                "Create a LineSet from given points and line indices",
                "points"_a, "lines"_a)
            .def("__repr__",
                 [](const LineSet &lineset) {
                     return std::string("LineSet with ") +
                            std::to_string(lineset.lines_.size()) + " lines.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_points", &LineSet::hasPoints,
                 "Returns ``True`` if the object contains points.")
            .def("has_lines", &LineSet::hasLines,
                 "Returns ``True`` if the object contains lines.")
            .def("has_colors", &LineSet::hasColors,
                 "Returns ``True`` if the object's lines contain "
                 "colors.")
            .def("get_line_coordinate", &LineSet::getLineCoordinate,
                 "line_index"_a)
            .def("paint_uniform_color", &LineSet::paintUniformColor,
                 "Assigns each line in the line set the same color.", "color"_a)
            .def_static("create_from_point_cloud_correspondences",
                        &LineSet::CreateFromPointCloudCorrespondences,
                        "Factory function to create a LineSet from two "
                        "pointclouds and a correspondence set.",
                        "cloud0"_a, "cloud1"_a, "correspondences"_a)
            .def_static("create_from_oriented_bounding_box",
                        &LineSet::CreateFromOrientedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "OrientedBoundingBox.",
                        "box"_a)
            .def_static("create_from_axis_aligned_bounding_box",
                        &LineSet::CreateFromAxisAlignedBoundingBox,
                        "Factory function to create a LineSet from an "
                        "AxisAlignedBoundingBox.",
                        "box"_a)
            .def_static("create_from_triangle_mesh",
                        &LineSet::CreateFromTriangleMesh,
                        "Factory function to create a LineSet from edges of a "
                        "triangle mesh.",
                        "mesh"_a)
            .def_static("create_from_tetra_mesh", &LineSet::CreateFromTetraMesh,
                        "Factory function to create a LineSet from edges of a "
                        "tetra mesh.",
                        "mesh"_a)
            .def_static("create_camera_visualization",
                        &LineSet::CreateCameraVisualization,
                        "Factory function to create a LineSet from intrinsic "
                        "and extrinsic camera matrices",
                        "view_width_px"_a, "view_height_px"_a, "intrinsic"_a,
                        "extrinsic"_a, "scale"_a = 1.0)
            .def_static(
                    "create_camera_visualization",
                    [](const camera::PinholeCameraIntrinsic &intrinsic,
                       const Eigen::Matrix4d &extrinsic, double scale) {
                        return LineSet::CreateCameraVisualization(
                                intrinsic.width_, intrinsic.height_,
                                intrinsic.intrinsic_matrix_, extrinsic, scale);
                    },
                    "Factory function to create a LineSet from intrinsic "
                    "and extrinsic camera matrices",
                    "intrinsic"_a, "extrinsic"_a, "scale"_a = 1.0)
            .def_readwrite("points", &LineSet::points_,
                           "``float64`` array of shape ``(num_points, 3)``, "
                           "use ``numpy.asarray()`` to access data: Points "
                           "coordinates.")
            .def_readwrite("lines", &LineSet::lines_,
                           "``int`` array of shape ``(num_lines, 2)``, use "
                           "``numpy.asarray()`` to access data: Lines denoted "
                           "by the index of points forming the line.")
            .def_readwrite(
                    "colors", &LineSet::colors_,
                    "``float64`` array of shape ``(num_lines, 3)``, "
                    "range ``[0, 1]`` , use ``numpy.asarray()`` to access "
                    "data: RGB colors of lines.");
    docstring::ClassMethodDocInject(m, "LineSet", "has_colors");
    docstring::ClassMethodDocInject(m, "LineSet", "has_lines");
    docstring::ClassMethodDocInject(m, "LineSet", "has_points");
    docstring::ClassMethodDocInject(m, "LineSet", "get_line_coordinate",
                                    {{"line_index", "Index of the line."}});
    docstring::ClassMethodDocInject(m, "LineSet", "paint_uniform_color",
                                    {{"color", "Color for the LineSet."}});
    docstring::ClassMethodDocInject(
            m, "LineSet", "create_from_point_cloud_correspondences",
            {{"cloud0", "First point cloud."},
             {"cloud1", "Second point cloud."},
             {"correspondences", "Set of correspondences."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_oriented_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet",
                                    "create_from_axis_aligned_bounding_box",
                                    {{"box", "The input bounding box."}});
    docstring::ClassMethodDocInject(m, "LineSet", "create_from_triangle_mesh",
                                    {{"mesh", "The input triangle mesh."}});
    docstring::ClassMethodDocInject(m, "LineSet", "create_from_tetra_mesh",
                                    {{"mesh", "The input tetra mesh."}});
}

void pybind_lineset_methods(py::module &m) {}

}  // namespace geometry
}  // namespace cloudViewer