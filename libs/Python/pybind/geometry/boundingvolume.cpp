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

#include <ecvBBox.h>
#include <ecvHObject.h>
#include <ecvOrientedBBox.h>

#include <sstream>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace cloudViewer {
namespace geometry {

void pybind_boundingvolume(py::module &m) {
	py::class_<cloudViewer::OrientedBoundingBox, 
		PyOrientedBBoxBase<cloudViewer::OrientedBoundingBox>,
		std::shared_ptr<cloudViewer::OrientedBoundingBox> >
		oriented_bounding_box_base(m, "OrientedBoundingBox", 
			"The base OrientedBoundingBox class.");
	py::detail::bind_default_constructor<cloudViewer::OrientedBoundingBox>(oriented_bounding_box_base);
	py::detail::bind_copy_functions<cloudViewer::OrientedBoundingBox>(oriented_bounding_box_base);
	oriented_bounding_box_base.def(
		py::init<const Eigen::Vector3d &, const Eigen::Matrix3d &,
			const Eigen::Vector3d &>(),
			"Create OrientedBoundingBox from center, rotation R and extent "
			"in x, y and z direction",
			"center"_a, "R"_a, "extent"_a)
		.def("__repr__",
			[](const cloudViewer::OrientedBoundingBox &box) {
			return std::string("OrientedBoundingBox"); })
		.def("volume", &cloudViewer::OrientedBoundingBox::volume,
			"Returns the volume of the bounding box.")
		.def("get_extent", &cloudViewer::OrientedBoundingBox::getExtent,
			"Get the extent/length of the bounding box in x, y, and z dimension "
			"in its frame of reference")
		.def("get_half_extent", &cloudViewer::OrientedBoundingBox::getHalfExtent,
			"Returns the half extent of the bounding box in its frame of reference.")
		.def("get_max_extent", &cloudViewer::OrientedBoundingBox::getMaxExtent,
			"Returns the max extent of the bounding box in its frame of reference")
		.def("set_color", &cloudViewer::OrientedBoundingBox::setColor,
			"``float64`` array of shape ``(3, )``",
			"color"_a)
		.def("get_color", &cloudViewer::OrientedBoundingBox::getColor,
			"``float64`` array of shape ``(3, )``")
		.def("get_box_points", &cloudViewer::OrientedBoundingBox::getBoxPoints,
			"Returns the eight points that define the bounding box.")
		.def("clear", &cloudViewer::OrientedBoundingBox::Clear,
			"Clear all elements in the geometry..")
		.def("get_point_indices_within_bounding_box",
			py::overload_cast<const std::vector<Eigen::Vector3d> &>(
				&cloudViewer::OrientedBoundingBox::getPointIndicesWithinBoundingBox, py::const_),
			"Return indices to points that are within the bounding box.",
			"points"_a)
		.def_static("create_from_axis_aligned_bounding_box",
			&cloudViewer::OrientedBoundingBox::CreateFromAxisAlignedBoundingBox,
			"Returns an oriented bounding box from the BoundingBox.",
			"aabox"_a)
		.def_readwrite("center", &cloudViewer::OrientedBoundingBox::center_,
		"``float64`` array of shape ``(3, )``")
		.def_readwrite("R", &cloudViewer::OrientedBoundingBox::R_,
			"``float64`` array of shape ``(3,3 )``")
		.def_readwrite("extent", &cloudViewer::OrientedBoundingBox::extent_,
			"``float64`` array of shape ``(3, )``")
		.def_readwrite("color", &cloudViewer::OrientedBoundingBox::color_,
			"``float64`` array of shape ``(3, )``");

	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "clear");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "volume");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_extent");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_half_extent");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_max_extent");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "set_color");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_color");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox", "get_box_points");
	docstring::ClassMethodDocInject(m, "OrientedBoundingBox",
		"get_point_indices_within_bounding_box",
		{ {"points", "A list of points."} });
	docstring::ClassMethodDocInject(
		m, "OrientedBoundingBox", "create_from_axis_aligned_bounding_box",
		{ {"aabox",
		  "BoundingBox object from which OrientedBoundingBox is "
		  "created."} });

    py::class_<ecvOrientedBBox, PyGeometry<ecvOrientedBBox>,
               std::shared_ptr<ecvOrientedBBox>, ccHObject, cloudViewer::OrientedBoundingBox>
            oriented_bounding_box(m, "ecvOrientedBBox",
                                  "Class that defines an oriented box that can "
                                  "be computed from 3D geometries.");
    py::detail::bind_default_constructor<ecvOrientedBBox>(oriented_bounding_box);
    py::detail::bind_copy_functions<ecvOrientedBBox>(oriented_bounding_box);
    oriented_bounding_box
        .def(py::init<const Eigen::Vector3d &, const Eigen::Matrix3d &,
                        const Eigen::Vector3d &, const std::string &>(),
                "Create ecvOrientedBBox from center, rotation R and extent "
                "in x, y and z "
                "direction",
                "center"_a, "R"_a, "extent"_a, "name"_a = "ecvOrientedBBox")
        .def("__repr__", [](const ecvOrientedBBox &box) {
                 std::stringstream s;
                 auto c = box.center_;
                 auto e = box.extent_;
                 s << "ecvOrientedBBox: center: (" << c.x() << ", "
                   << c.y() << ", " << c.z() << "), extent: " << e.x()
                   << ", " << e.y() << ", " << e.z() << ")";
                 return s.str();
             })
		.def_static(
			"create_from_points",
			py::overload_cast<const std::vector<Eigen::Vector3d> &>(
				&ecvOrientedBBox::CreateFromPoints),
			"Creates the bounding box that encloses the set of points.",
			"points"_a)
		.def_static("create_from_axis_aligned_bounding_box",
			&ecvOrientedBBox::CreateFromAxisAlignedBoundingBox,
			"Returns an oriented bounding box from the ccBBox.",
			"aabox"_a);

    docstring::ClassMethodDocInject(m, "ecvOrientedBBox",
                                    "create_from_points",
                                    {{"points", "A list of points."}});
	docstring::ClassMethodDocInject(
		m, "ecvOrientedBBox", "create_from_axis_aligned_bounding_box",
		{ {"aabox",
		  "BoundingBox object from which OrientedBoundingBox is "
		  "created."} });

    py::class_<cloudViewer::BoundingBox, std::shared_ptr<cloudViewer::BoundingBox>>
		axis_bounding_box_base(m, "BoundingBox",
			"Class that defines an axis_aligned box "
			"that can be computed from 3D "
            "geometries, The axis aligned bounding "
			"box uses the coordinate axes for "
			"bounding box generation.");
    py::detail::bind_default_constructor<cloudViewer::BoundingBox>(axis_bounding_box_base);
    py::detail::bind_copy_functions<cloudViewer::BoundingBox>(axis_bounding_box_base);
    axis_bounding_box_base
		.def(py::init<const Eigen::Vector3d &, const Eigen::Vector3d &>(),
			"Create an BoundingBox from min bounds and max "
			"bounds in x, y and z",
			"bbMinCorner"_a, "bbMaxCorner"_a)
		.def("__repr__",
            [](const cloudViewer::BoundingBox &box) {
			return std::string("BoundingBox"); })

        .def("clear", &cloudViewer::BoundingBox::clear,
			"Resets the bounding box.")
            .def("is_valid", &cloudViewer::BoundingBox::isValid,
				"Returns whether bounding box is valid or not")
            .def("set_validity", &cloudViewer::BoundingBox::setValidity,
				"Sets bonding box validity.",
				"state"_a)
            .def("volume", &cloudViewer::BoundingBox::volume,
				"Returns the bounding box volume.")
            .def("add", &cloudViewer::BoundingBox::addEigen,
				"'Enlarges' the bounding box with a point.",
				"point"_a)
            .def("get_x_percentage", &cloudViewer::BoundingBox::getXPercentage,
				"Returns x Percentage.",
				"x"_a)
            .def("get_y_percentage", &cloudViewer::BoundingBox::getYPercentage,
				"Returns y Percentage.",
				"y"_a)
            .def("get_z_percentage", &cloudViewer::BoundingBox::getZPercentage,
				"Returns z Percentage.",
				"z"_a)
            .def("get_diag_norm", &cloudViewer::BoundingBox::getDiagNormd,
				"Returns diagonal length (double precision)")
            .def("get_min_box_dim", &cloudViewer::BoundingBox::getMinBoxDim,
				"Returns minimal box dimension")
            .def("get_max_box_dim", &cloudViewer::BoundingBox::getMaxBoxDim,
				"Returns maximal box dimension")
            .def("compute_volume", &cloudViewer::BoundingBox::computeVolume,
				"Returns the bounding-box volume")
            .def("get_bounds", &cloudViewer::BoundingBox::getBounds,
				"Returns the bounding-box bounds",
				"bounds"_a)
            .def("min_distance_to", &cloudViewer::BoundingBox::minDistTo,
				"Computes min gap (absolute distance) between this "
				"bounding-box and another one; return min gap (>=0) "
				"or -1 if at least one of the box is not valid",
				"box"_a)
            .def("contains", &cloudViewer::BoundingBox::containsEigen,
				"Returns whether a points is inside the box or not",
				"point"_a)
			.def("get_point_indices_within_boundingbox",
				py::overload_cast<const std::vector<Eigen::Vector3d> &>(
                    &cloudViewer::BoundingBox::getPointIndicesWithinBoundingBox, py::const_),
				"Returns point indices Within bounding box.",
				"points"_a);

		docstring::ClassMethodDocInject(m, "BoundingBox", "clear");
		docstring::ClassMethodDocInject(m, "BoundingBox", "is_valid");
		docstring::ClassMethodDocInject(m, "BoundingBox", "set_validity");
		docstring::ClassMethodDocInject(m, "BoundingBox", "volume");
		docstring::ClassMethodDocInject(m, "BoundingBox", "add");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_x_percentage");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_y_percentage");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_z_percentage");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_diag_norm");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_min_box_dim");
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_max_box_dim");
		docstring::ClassMethodDocInject(m, "BoundingBox", "compute_volume");
		docstring::ClassMethodDocInject(m, "BoundingBox", "min_distance_to");
		docstring::ClassMethodDocInject(m, "BoundingBox", "contains");
		docstring::ClassMethodDocInject(m, "BoundingBox",
			"get_point_indices_within_boundingbox",
			{ {"points", "A list of points."} });
		docstring::ClassMethodDocInject(m, "BoundingBox", "get_bounds",
			{ {"bounds", "the output bounds with six double values."} });

    py::class_<ccBBox, PyGeometry<ccBBox>,
               std::shared_ptr<ccBBox>, ccHObject, cloudViewer::BoundingBox>
            axis_aligned_bounding_box(m, "ccBBox",
                                      "Class that defines an axis_aligned box "
                                      "that can be computed from 3D "
                                      "geometries, The axis aligned bounding "
                                      "box uses the coordinate axes for "
                                      "bounding box generation.");
    py::detail::bind_default_constructor<ccBBox>(axis_aligned_bounding_box);
    py::detail::bind_copy_functions<ccBBox>(axis_aligned_bounding_box);
    axis_aligned_bounding_box
		.def(py::init<const Eigen::Vector3d &,
			const Eigen::Vector3d &,
			const std::string &>(),
			"Create an ccBBox from min bounds and max "
			"bounds in x, y and z",
			"min_bound"_a, "max_bound"_a, "name"_a = "ccBBox")
        .def("__repr__", [](const ccBBox &box) {
                 std::stringstream s;
                 auto mn = box.getMinBound();
                 auto mx = box.getMaxBound();
                 s << "ccBBox: min: (" << mn.x() << ", "
                   << mn.y() << ", " << mn.z() << "), max: (" << mx.x()
                   << ", " << mx.y() << ", " << mx.z() << ")";
                 return s.str();
             })
        .def(py::self += py::self)
		.def("get_box_points", &ccBBox::getBoxPoints,
		"Returns the eight points that define the bounding box.")
		.def("get_extent", &ccBBox::getExtent,
			"Get the extent/length of the bounding box in x, y, and z dimension.")
		.def("get_half_extent", &ccBBox::getHalfExtent,
			"Returns the half extent of the bounding box.")
		.def("get_max_extent", &ccBBox::getMaxExtent,
			"Returns the maximum extent, i.e. the maximum of X, Y and Z axis")
		.def("set_min_bound", &ccBBox::setMinBounds,
			"``float64`` array of shape ``(3, )``",
			"minBound"_a)
		.def("set_max_bound", &ccBBox::setMaxBounds,
			"``float64`` array of shape ``(3, )``",
			"maxBound"_a)
		.def("set_color", &ccBBox::setColor,
			"``float64`` array of shape ``(3, )``",
			"color"_a)
		.def("get_color", &ccBBox::getColor,
			"``float64`` array of shape ``(3, )``")
		.def("get_print_info", &ccBBox::getPrintInfo,
			 "Returns the 3D dimensions of the bounding box in string format.")
		.def_static("create_from_points", 
			py::overload_cast<const std::vector<Eigen::Vector3d> &>(
				&ccBBox::CreateFromPoints),
			"Creates the bounding box that encloses the set of points.",
			"points"_a);
    docstring::ClassMethodDocInject(m, "ccBBox", "get_box_points");
    docstring::ClassMethodDocInject(m, "ccBBox", "get_extent");
    docstring::ClassMethodDocInject(m, "ccBBox", "get_half_extent");
    docstring::ClassMethodDocInject(m, "ccBBox", "get_max_extent");
    docstring::ClassMethodDocInject(m, "ccBBox", "set_color");
    docstring::ClassMethodDocInject(m, "ccBBox", "get_color");
	docstring::ClassMethodDocInject(m, "ccBBox", "get_print_info");
	docstring::ClassMethodDocInject(m, "ccBBox", "set_min_bound",
		{ {"minBound", "The minimum corner coordinate." } });
	docstring::ClassMethodDocInject(m, "ccBBox", "set_max_bound",
		{ { "maxBound", "The maximum corner coordinate."}});
	docstring::ClassMethodDocInject(m, "ccBBox", "create_from_points",
		{ {"points", "A list of points."} });
}

}  // namespace geometry
}  // namespace cloudViewer
