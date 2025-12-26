// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CV_CORE_LIB
#include <Logging.h>
#include <ReferenceCloud.h>

// ECV_DB_LIB
#include <ecvCircle.h>
#include <ecvFacet.h>
#include <ecvMesh.h>
#include <ecvObject.h>
#include <ecvPlanarEntityInterface.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// LOCAL
#include "pybind/docstring.h"
#include "pybind/geometry/cloudbase.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

using namespace cloudViewer;
namespace cloudViewer {
namespace geometry {

void pybind_polyline(py::module& m) {
    py::class_<cloudViewer::Polyline,
               PyGenericReferenceCloud<cloudViewer::Polyline>,
               std::shared_ptr<cloudViewer::Polyline>, ReferenceCloud>
            polyline(m, "Polyline",
                     "The polyline is considered as a cloud of points "
                     "(in a specific order) with a open  closed state "
                     "information..");
    polyline.def(py::init([](std::shared_ptr<GenericIndexedCloudPersist>
                                     associated_cloud) {
                     return new cloudViewer::Polyline(associated_cloud.get());
                 }),
                 "Polyline constructor", "associated_cloud"_a)
            .def("__repr__",
                 [](const cloudViewer::Polyline& poly) {
                     std::string info = fmt::format(
                             "cloudViewer::Polyline with {} points and is "
                             "closed {}",
                             poly.size(), poly.isClosed() ? "True" : "False");
                     return info;
                 })
            .def("is_closed", &cloudViewer::Polyline::isClosed,
                 "Returns whether the polyline is closed or not.")
            .def("set_closed", &cloudViewer::Polyline::setClosed,
                 "Sets whether the polyline is closed or not.", "state"_a);
    docstring::ClassMethodDocInject(m, "Polyline", "is_closed");
    docstring::ClassMethodDocInject(m, "Polyline", "set_closed");

    // cloudViewer.geometry.ccPolyline
    py::class_<ccPolyline, PyGeometry<ccPolyline>, std::shared_ptr<ccPolyline>,
               cloudViewer::Polyline, ccHObject>
            pyply(m, "ccPolyline", py::multiple_inheritance(),
                  "Colored polyline, Extends the cloudViewer::Polyline class.");
    py::detail::bind_copy_functions<ccPolyline>(pyply);
    // pyply.def(py::init([](std::shared_ptr<cloudViewer::GenericIndexedCloudPersist>
    // cloud) { 	return new ccPolyline(cloud.get());
    // }), "cloud -> the associated point cloud (i.e. the vertices)", "cloud"_a
    // = nullptr)
    pyply.def(py::init([](std::shared_ptr<ccPointCloud> cloud) {
                  if (cloud) {
                      return new ccPolyline(*cloud);
                  } else {
                      return new ccPolyline(nullptr);
                  }
              }),
              "cloud -> the associated point cloud (i.e. the vertices)",
              "cloud"_a = nullptr)
            .def("__repr__",
                 [](const ccPolyline& polyline) {
                     bool is2D = polyline.is2DMode();
                     PointCoordinateType length = polyline.computeLength();
                     std::string info = fmt::format(
                             "ccPolyline with segment count {}, length {} and "
                             "2D {}",
                             polyline.segmentCount(), length,
                             is2D ? "True" : "False");
                     return info;
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("append", &ccPolyline::add, "Append another point cloud.",
                 "cloud"_a)
            .def("set_2d_mode", &ccPolyline::set2DMode,
                 "Defines if the polyline is considered as 2D or 3D.",
                 "state"_a)
            .def("is_2d_mode", &ccPolyline::is2DMode,
                 "Returns whether the polyline is considered as 2D or 3D.")
            .def("set_transform_flag", &ccPolyline::setTransformFlag,
                 "Defines if the polyline is considered as processed polyline.",
                 "state"_a)
            .def("need_transform", &ccPolyline::needTransform,
                 "Returns whether the polyline is considered as 2D or 3D.")
            .def("set_width", &ccPolyline::setWidth,
                 "Sets the width of the line.", "width"_a)
            .def("get_width", &ccPolyline::getWidth,
                 "Returns the width of the line.")
            .def("compute_length", &ccPolyline::computeLength,
                 "Computes the polyline length.")
            .def("segment_count", &ccPolyline::segmentCount,
                 " Returns the number of segments.")
            .def("paint_uniform_color", &ccPolyline::PaintUniformColor,
                 " Assigns each line in the polyline the same color.",
                 "color"_a)
            .def(
                    "set_color",
                    [](ccPolyline& polyline, const Eigen::Vector3d& color) {
                        polyline.setColor(ecvColor::Rgb::FromEigen(color));
                    },
                    "Sets the polyline color.", "color"_a)
            .def(
                    "get_color",
                    [](const ccPolyline& polyline) {
                        return ecvColor::Rgb::ToEigen(polyline.getColor());
                    },
                    "Returns the polyline color.")
            .def(
                    "split",
                    [](ccPolyline& polyline,
                       PointCoordinateType max_edge_length) {
                        std::vector<ccPolyline*> parts;
                        bool success = polyline.split(max_edge_length, parts);
                        std::vector<std::shared_ptr<ccPolyline>> outParts;
                        for (size_t i = 0; i < parts.size(); ++i) {
                            outParts.push_back(
                                    std::shared_ptr<ccPolyline>(parts[i]));
                        }
                        return std::make_tuple(outParts, success);
                    },
                    "Splits the polyline into several parts based on a maximum "
                    "edge length",
                    "max_edge_length"_a)
            .def("get_arrow_index", &ccPolyline::getArrowIndex,
                 "Returns arrow index.")
            .def("get_arrow_length", &ccPolyline::getArrowLength,
                 "Returns arrow length.")
            .def("vertices_shown", &ccPolyline::verticesShown,
                 "Whether the polyline vertices should be displayed or not.")
            .def("show_vertices", &ccPolyline::showVertices,
                 "Sets whether to display or hide the polyline vertices.",
                 "state"_a)
            .def("get_vertex_marker_width", &ccPolyline::getVertexMarkerWidth,
                 "Returns the width of vertex markers.")
            .def("set_vertex_marker_width", &ccPolyline::setVertexMarkerWidth,
                 "Sets the width of vertex markers.", "width"_a)
            .def(
                    "init_with",
                    [](ccPolyline& polyline, ccPointCloud& vertices,
                       const ccPolyline& other_polyline) {
                        ccPointCloud* cloud = &vertices;
                        return polyline.initWith(cloud, other_polyline);
                    },
                    "Initializes the polyline with a given set of vertices and "
                    "the parameters of another polyline",
                    "vertices"_a, "other_polyline"_a)
            .def(
                    "import_parameters_from",
                    [](ccPolyline& polyline, const ccPolyline& other_polyline) {
                        polyline.importParametersFrom(other_polyline);
                    },
                    "Copy the parameters from another polyline",
                    "other_polyline"_a)
            .def("show_arrow", &ccPolyline::showArrow,
                 "Shows an arrow in place of a given vertex.", "state"_a,
                 "vertex_index"_a, "length"_a)
            .def(
                    "sample_points",
                    [](ccPolyline& polyline, bool density_based,
                       double sampling_parameter, bool with_rgb) {
                        ccPointCloud* sampledCloud = polyline.samplePoints(
                                density_based, sampling_parameter, with_rgb);
                        return std::shared_ptr<ccPointCloud>(sampledCloud);
                    },
                    "Samples points on the polyline", "density_based"_a,
                    "sampling_parameter"_a, "with_rgb"_a);

    docstring::ClassMethodDocInject(m, "ccPolyline", "segment_count");
    docstring::ClassMethodDocInject(m, "ccPolyline", "compute_length");
    docstring::ClassMethodDocInject(m, "ccPolyline", "append");
    docstring::ClassMethodDocInject(m, "ccPolyline", "set_2d_mode");
    docstring::ClassMethodDocInject(m, "ccPolyline", "is_2d_mode");
    docstring::ClassMethodDocInject(m, "ccPolyline", "set_transform_flag");
    docstring::ClassMethodDocInject(m, "ccPolyline", "need_transform");
    docstring::ClassMethodDocInject(m, "ccPolyline", "paint_uniform_color");
    docstring::ClassMethodDocInject(m, "ccPolyline", "get_color");
    docstring::ClassMethodDocInject(m, "ccPolyline", "set_color",
                                    {{"color", "The polyline rgb color."}});
    docstring::ClassMethodDocInject(m, "ccPolyline", "get_width");
    docstring::ClassMethodDocInject(m, "ccPolyline", "set_width",
                                    {{"width", "The polyline width."}});
    docstring::ClassMethodDocInject(
            m, "ccPolyline", "split",
            {{"max_edge_length",
              "maximum edge length "
              "(warning output polylines set (parts) may be "
              "empty if all the vertices are too far from each other!)"}});
    docstring::ClassMethodDocInject(m, "ccPolyline", "get_arrow_index");
    docstring::ClassMethodDocInject(m, "ccPolyline", "get_arrow_length");
    docstring::ClassMethodDocInject(m, "ccPolyline", "vertices_shown");
    docstring::ClassMethodDocInject(m, "ccPolyline", "show_vertices");
    docstring::ClassMethodDocInject(m, "ccPolyline", "get_vertex_marker_width");
    docstring::ClassMethodDocInject(m, "ccPolyline", "set_vertex_marker_width");
    docstring::ClassMethodDocInject(
            m, "ccPolyline", "init_with",
            {{"vertices",
              "set of vertices (can be null, in which case the polyline "
              "vertices will be cloned)."},
             {"other_polyline", "The other polyline."}});
    docstring::ClassMethodDocInject(
            m, "ccPolyline", "import_parameters_from",
            {{"other_polyline", "The other polyline."}});
    docstring::ClassMethodDocInject(m, "ccPolyline", "show_arrow",
                                    {{"state", "The state."},
                                     {"vertex_index", "The vertex index."},
                                     {"length", "The length."}});
    docstring::ClassMethodDocInject(
            m, "ccPolyline", "sample_points",
            {{"density_based", "The density based."},
             {"sampling_parameter", "The sampling parameter."},
             {"with_rgb", "with rgb."}});

    // cloudViewer.geometry.ccCircle
    py::class_<ccCircle, PyGeometry<ccCircle>, std::shared_ptr<ccCircle>,
               ccPolyline>
            pycircle(m, "ccCircle", py::multiple_inheritance(),
                     "A 3D circle represented as a polyline.");
    py::detail::bind_copy_functions<ccCircle>(pycircle);
    pycircle.def(py::init([](double radius, unsigned resolution,
                             unsigned uniqueID) {
                     return new ccCircle(radius, resolution, uniqueID);
                 }),
                 "Circle constructor", "radius"_a = 0.0, "resolution"_a = 48,
                 "uniqueID"_a = ccUniqueIDGenerator::InvalidUniqueID)
            .def("__repr__",
                 [](const ccCircle& circle) {
                     std::string info = fmt::format(
                             "ccCircle with radius {}, resolution {} and {} "
                             "segments",
                             circle.getRadius(), circle.getResolution(),
                             circle.segmentCount());
                     return info;
                 })
            .def("get_radius", &ccCircle::getRadius,
                 "Returns the radius of the circle.")
            .def("set_radius", &ccCircle::setRadius,
                 "Sets the radius of the circle.", "radius"_a)
            .def("get_resolution", &ccCircle::getResolution,
                 "Returns the resolution of the displayed circle.")
            .def("set_resolution", &ccCircle::setResolution,
                 "Sets the resolution of the displayed circle.", "resolution"_a)
            .def(
                    "clone",
                    [](const ccCircle& circle) {
                        return std::shared_ptr<ccCircle>(circle.clone());
                    },
                    "Clones this circle.");

    docstring::ClassMethodDocInject(m, "ccCircle", "get_radius");
    docstring::ClassMethodDocInject(m, "ccCircle", "set_radius",
                                    {{"radius", "The desired radius."}});
    docstring::ClassMethodDocInject(m, "ccCircle", "get_resolution");
    docstring::ClassMethodDocInject(
            m, "ccCircle", "set_resolution",
            {{"resolution", "The displayed resolution (>= 4)."}});
    docstring::ClassMethodDocInject(m, "ccCircle", "clone");
}

void pybind_polyline_methods(py::module& m) {}

}  // namespace geometry
}  // namespace cloudViewer