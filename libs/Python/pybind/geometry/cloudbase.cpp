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

// LOCAL
#include "cloudbase.h"

// CV_CORE_LIB
#include <Polyline.h>
#include <BoundingBox.h>

// ECV_DB_LIB
#include <ecvIndexedTransformationBuffer.h>

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

#pragma warning(disable:4715)

using namespace cloudViewer;
namespace cloudViewer {
namespace geometry {

void pybind_cloudbase(py::module &m) {
	// cloudViewer.geometry.GenericCloud functions
	py::class_<GenericCloud, PyGenericCloud<GenericCloud>, std::shared_ptr<GenericCloud>>
		pygc(m, "GenericCloud", "A generic 3D point cloud interface for data communication"
			" between library and client applications.");
	pygc.def("size", &GenericCloud::size, "Returns the number of points.")
		.def("has_points", &GenericCloud::hasPoints, "Returns whether has points.")
		.def("get_bounding_box", [](GenericCloud& obj) {
				CCVector3 bbMin, bbMax;
				obj.getBoundingBox(bbMin, bbMax);
				return std::make_tuple(CCVector3d::fromArray(bbMin), CCVector3d::fromArray(bbMax));
			}, "Returns the cloud bounding box.")
		.def("test_visibility", [](const GenericCloud& obj, const Eigen::Vector3d& point) {
				return obj.testVisibility(CCVector3::fromArray(point)); 
			}, "Returns a given point visibility state (relatively to a sensor for instance).", "point"_a)
		.def("place_iterator_at_beginning", &GenericCloud::placeIteratorAtBeginning,
			"Sets the cloud iterator at the beginning.")
		.def("get_next_point", [](GenericCloud& obj) {
				return CCVector3d::fromArray(*obj.getNextPoint());
			}, "Returns the next point (relatively to the global iterator position).")
		.def("enable_scalar_field", &GenericCloud::enableScalarField,
			"Enables the scalar field associated to the cloud.")
		.def("is_scalar_field_enabled", &GenericCloud::isScalarFieldEnabled,
			"Returns true if the scalar field is enabled, false otherwise.")
		.def("set_scalar_value", &GenericCloud::setPointScalarValue,
			"Sets the ith point associated scalar value.", "point_index"_a, "value"_a)
		.def("get_scalar_value", &GenericCloud::getPointScalarValue,
			"Returns the ith point associated scalar value.", "point_index"_a);
		docstring::ClassMethodDocInject(m, "GenericCloud", "size");
		docstring::ClassMethodDocInject(m, "GenericCloud", "has_points");
		docstring::ClassMethodDocInject(m, "GenericCloud", "get_bounding_box");
		docstring::ClassMethodDocInject(m, "GenericCloud", "test_visibility",
			{ {"point", "the 3D point to test, "
				"return visibility (default: POINT_VISIBLE)"
				"Generic method to request a point visibility (should be overloaded if this functionality is required)."
				"The point visibility is such as defined in Daniel Girardeau - Montaut's PhD manuscript (see Chapter 2, "
				"section 2 - 3 - 3).In this case, a ground based laser sensor model should be used to determine it."
				"This method is called before performing any point - to - cloud comparison.If the result is not"
				"POINT_VISIBLE, then the comparison won't be performed and the scalar field value associated"
				"to this point will be this visibility value."} });
		docstring::ClassMethodDocInject(m, "GenericCloud", "place_iterator_at_beginning");
		docstring::ClassMethodDocInject(m, "GenericCloud", "get_next_point");
		docstring::ClassMethodDocInject(m, "GenericCloud", "enable_scalar_field");
		docstring::ClassMethodDocInject(m, "GenericCloud", "is_scalar_field_enabled");
		docstring::ClassMethodDocInject(m, "GenericCloud", "set_scalar_value");
		docstring::ClassMethodDocInject(m, "GenericCloud", "get_scalar_value");
	
	// cloudViewer.geometry.GenericIndexedCloud functions
	py::class_<GenericIndexedCloud, PyGenericIndexedCloud<GenericIndexedCloud>,
		std::shared_ptr<GenericIndexedCloud>, GenericCloud>
	pygic(m, "GenericIndexedCloud", " A generic 3D point cloud with index-based point access.");
	pygic.def("get_point", [](const GenericIndexedCloud& obj, unsigned index) {
			return CCVector3d::fromArray(*obj.getPoint(index));
		}, "Returns the ith point(virtual method to request a point with a specific index).", "index"_a);
	docstring::ClassMethodDocInject(m, "GenericIndexedCloud", "get_point");

	// cloudViewer.geometry.GenericIndexedCloudPersist functions
	py::class_<GenericIndexedCloudPersist, PyGenericIndexedCloudPersist<GenericIndexedCloudPersist>,
		std::shared_ptr<GenericIndexedCloudPersist>, GenericIndexedCloud>
	pygicp(m, "GenericIndexedCloudPersist", "A generic 3D point cloud with index-based point access.");
	pygicp.def("get_point_persistent", [](GenericIndexedCloudPersist& obj, unsigned index) {
			return CCVector3d::fromArray(*obj.getPointPersistentPtr(index));
		}, "Returns the ith point as a persistent point).", "index"_a);
	docstring::ClassMethodDocInject(m, "GenericIndexedCloudPersist", "get_point_persistent");

	// cloudViewer.geometry.ReferenceCloud functions
	py::class_<ReferenceCloud, PyGenericReferenceCloud<ReferenceCloud>,
		std::shared_ptr<ReferenceCloud>, GenericIndexedCloudPersist>
		pyrefcloud(m, "ReferenceCloud", "The polyline is considered as a cloud of points "
			"(in a specific order) with a open  closed state information.");
	py::detail::bind_copy_functions<ReferenceCloud>(pyrefcloud);
	pyrefcloud.def(py::init([](std::shared_ptr<GenericIndexedCloudPersist> associated_cloud) {
			return new ReferenceCloud(associated_cloud.get());
		}), "ReferenceCloud constructor", "associated_cloud"_a)
	.def("__repr__", [](const cloudViewer::Polyline &poly) {
			return fmt::format("ReferenceCloud with {} points", poly.size());
		})
	.def("get_point_global_index", &ReferenceCloud::getPointGlobalIndex,
		"Returns global index (i.e. relative to the associated cloud) of a given element.",
		"local_index"_a)
	.def("get_cur_point_global_index", &ReferenceCloud::getCurrentPointGlobalIndex,
		"Returns the global index of the point pointed by the current element.")
	.def("get_cur_point_scalar", &ReferenceCloud::getCurrentPointScalarValue,
		"Returns the current point associated scalar value.")
	.def("set_cur_point_scalar", &ReferenceCloud::setCurrentPointScalarValue,
		"Sets the current point associated scalar value.", "value"_a)
	.def("forward_iterator", &ReferenceCloud::forwardIterator,
		"Forwards the local element iterator.")
	.def("get_cur_point_coordinates", [](const ReferenceCloud& obj) {
			return CCVector3d::fromArray(*obj.getCurrentPointCoordinates());
		}, "Returns the coordinates of the point pointed by the current element.")
	.def("get_associated_cloud", [](ReferenceCloud& obj) {
			if (obj.getAssociatedCloud()) {
				return std::ref(*obj.getAssociatedCloud());
			}
			else {
				cloudViewer::utility::LogWarning("[cloudViewer::ReferenceCloud] does not have associated cloud!");
			}
		}, "Returns the associated (source) cloud.")
	.def("get_associated_cloud", [](const ReferenceCloud& obj) {
			if (obj.getAssociatedCloud()) {
				return std::ref(*obj.getAssociatedCloud());
			} else {
				cloudViewer::utility::LogWarning("[cloudViewer::ReferenceCloud] does not have associated cloud!");
			}
		}, "Returns the associated (source) cloud (const version).")
	.def("set_associated_cloud", [](ReferenceCloud& obj, 
			std::shared_ptr<GenericIndexedCloudPersist> cloud) {
			obj.setAssociatedCloud(cloud.get());
		}, "Sets the associated (source) cloud.")
	.def("add", &ReferenceCloud::add, "Add another reference cloud.", "cloud"_a)
	.def("clear", &ReferenceCloud::clear, "Clears the cloud.", "release_memory"_a = false)
	.def("swap", &ReferenceCloud::swap,
		"Swaps two point references.", "first_index"_a, "second_index"_a)
	.def("reserve", &ReferenceCloud::reserve,
		"Reserves some memory for hosting the point references.", "n"_a)
	.def("resize", &ReferenceCloud::resize,
		"Presets the size of the vector used to store point references.", "n"_a)
	.def("capacity", &ReferenceCloud::capacity,
		"Reserves some memory for hosting the point references.")
	.def("invalidate_boundingBox", &ReferenceCloud::invalidateBoundingBox, "Invalidates the bounding-box")
	.def("add_point_index", py::overload_cast<unsigned>(&ReferenceCloud::addPointIndex),
		"Point global index insertion mechanism.", "global_index"_a)
	.def("add_point_index", py::overload_cast<unsigned, unsigned>(&ReferenceCloud::addPointIndex),
		"Point global index insertion mechanism (range).", "first_index"_a, "last_index"_a)
	.def("set_point_index", &ReferenceCloud::setPointIndex,
		"Sets global index for a given element.", "local_index"_a, "global_index"_a)
	.def("remove_point_global_index", &ReferenceCloud::removePointGlobalIndex,
		"Removes a given element.", "local_ndex"_a)
	.def("remove_cur_point_global_index", &ReferenceCloud::removeCurrentPointGlobalIndex,
		"Removes current global element, WARNING: this method changes the cloud size!.");

	docstring::ClassMethodDocInject(m, "ReferenceCloud", "add");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "swap");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "clear");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "resize");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "reserve");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "capacity");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "get_point_global_index");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "get_cur_point_global_index");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "get_cur_point_scalar");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "set_cur_point_scalar");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "forward_iterator");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "get_cur_point_coordinates");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "get_associated_cloud");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "set_associated_cloud");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "invalidate_boundingBox");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "add_point_index");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "set_point_index");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "remove_point_global_index");
	docstring::ClassMethodDocInject(m, "ReferenceCloud", "remove_cur_point_global_index");

	// cloudViewer.geometry.ccGenericPointCloud functions
	py::class_<ccGenericPointCloud, PyGeometry<ccGenericPointCloud>,
		std::shared_ptr<ccGenericPointCloud>, cloudViewer::GenericIndexedCloudPersist, ccHObject>
		genericPointCloud(m, "ccGenericPointCloud", py::multiple_inheritance(),
			"A 3D cloud interface with associated features (color, normals, octree, etc.).");
	genericPointCloud.def("__repr__", [](const ccGenericPointCloud &cloud) {
		return fmt::format("ccGenericPointCloud with {} points", cloud.size());
	})
	.def("clone", [](ccGenericPointCloud& cloud, bool ignore_children) {
		return std::shared_ptr<ccGenericPointCloud>(cloud.clone(nullptr, ignore_children));
		}, "Clones this entity.", "ignore_children"_a = false)
	.def("clear", &ccGenericPointCloud::clear, "Clears the entity from all its points and features.")
	.def("get_scalar_value_color", [](const ccGenericPointCloud& cloud, ScalarType value) {
			return ecvColor::Rgb::ToEigen(*cloud.getScalarValueColor(value));
		}, "Returns color corresponding to a given scalar value.", "value"_a)
	.def("get_point_scalar_value_color", [](const ccGenericPointCloud& cloud, unsigned point_index) {
			return ecvColor::Rgb::ToEigen(*cloud.getPointScalarValueColor(point_index));
		}, "Returns color corresponding to a given point associated scalar value.", "point_index"_a)
	.def("get_point_color", [](const ccGenericPointCloud& cloud, unsigned point_index) {
			return ecvColor::Rgb::ToEigen(cloud.getPointColor(point_index));
		}, "Returns color corresponding to a given point.", "point_index"_a)
	.def("get_point_displayed_distance", &ccGenericPointCloud::getPointDisplayedDistance,
			"Returns scalar value associated to a given point.", "point_index"_a)
	.def("get_point_normal_index", &ccGenericPointCloud::getPointNormalIndex,
			"Returns compressed normal corresponding to a given point.", "point_index"_a)
	.def("get_point_normal", [](const ccGenericPointCloud& cloud, unsigned point_index) {
			return CCVector3d::fromArray(cloud.getPointNormal(point_index));
		}, "Returns normal corresponding to a given point.", "point_index"_a)
	.def("get_visibility_array", [](const ccGenericPointCloud& cloud) {
		return cloud.getTheVisibilityArray();
	}, "Returns associated visibility array.")
	.def("get_visibility_array", [](ccGenericPointCloud& cloud) {
		return cloud.getTheVisibilityArray();
	}, "Returns associated visibility array (const version).")
	.def("get_visible_points", [](const ccGenericPointCloud& cloud, 
								const std::vector<unsigned char>& vis_table,
								bool silent) {
        const std::vector<unsigned char>* visTable = nullptr;
		if (!vis_table.empty())
		{
			visTable = &vis_table;
		}
		return std::ref(*cloud.getTheVisiblePoints(visTable, silent));
    },
    "Returns a ReferenceCloud equivalent to the visibility "
    "array.",
    "vis_table"_a,
    "silent"_a = false)
	.def("is_visibility_table_instantiated", &ccGenericPointCloud::isVisibilityTableInstantiated, 
		"Returns whether the visibility array is allocated or not")
	.def("reset_visibility_array", &ccGenericPointCloud::resetVisibilityArray, "Resets the associated visibility array")
	.def("invert_visibility_array", &ccGenericPointCloud::invertVisibilityArray, "Inverts the visibility array")
	.def("unallocate_visibility_array", &ccGenericPointCloud::unallocateVisibilityArray, "Erases the points visibility information")
	.def("refresh_bbox", &ccGenericPointCloud::refreshBB, "Forces bounding-box update.")
	.def("create_cloud_from_visibility_selection", [](ccGenericPointCloud& cloud, bool remove_selected_points,
		ccGenericPointCloud::VisibilityTableType& vis_table, bool silent) {
		ccGenericPointCloud::VisibilityTableType* visTable = nullptr;
		if (!vis_table.empty())
		{
			visTable = &vis_table;
		}
		return std::shared_ptr<ccGenericPointCloud>(
			cloud.createNewCloudFromVisibilitySelection(remove_selected_points, visTable, silent));
		}, "Creates a new point cloud with only the 'visible' points (as defined by the visibility array).", 
		"remove_selected_points"_a = false, "vis_table"_a = nullptr, "silent"_a = false)
	.def("crop", [](ccGenericPointCloud& cloud, const ccBBox& box, bool inside) {
			return std::shared_ptr<ReferenceCloud>(cloud.crop(box, inside));
		}, "Crops the cloud inside (or outside) a bounding box.", "box"_a, "inside"_a = true)
	.def("crop", [](ccGenericPointCloud& cloud, const ecvOrientedBBox& box) {
			return std::shared_ptr<ReferenceCloud>(cloud.crop(box));
		}, "Crops the cloud inside a oriented bounding box.", "box"_a)
	.def("remove_points", &ccGenericPointCloud::removePoints, "Remove points.", "index"_a)
	.def("set_point_size", &ccGenericPointCloud::setPointSize, "Sets point size", "size"_a)
	.def("get_point_size", &ccGenericPointCloud::getPointSize, "Returns current point size")
	.def("import_parameters_from", [](ccGenericPointCloud& cloud, const ccGenericPointCloud& source) {
			cloud.importParametersFrom(&source);
		}, "Imports the parameters from another cloud", "source"_a)
	.def("compute_mean_and_covariance", &ccGenericPointCloud::computeMeanAndCovariance,
			"Function to compute the mean and covariance matrix of a point cloud.");

	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "clone");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "clear");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_scalar_value_color");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_scalar_value_color");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_color");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_displayed_distance");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_normal_index");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_normal");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_visibility_array");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_visible_points");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "is_visibility_table_instantiated");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "reset_visibility_array");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "invert_visibility_array");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "unallocate_visibility_array");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "refresh_bbox");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "create_cloud_from_visibility_selection");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "crop",
		{ {"box", "Axis Aligned BoundingBox"}, {"inside", "whether selected points are inside or outside the box"} });
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "crop",
		{ {"box", "Oriented BoundingBox"} });
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "remove_points");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "set_point_size");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "get_point_size");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "import_parameters_from");
	docstring::ClassMethodDocInject(m, "ccGenericPointCloud", "compute_mean_and_covariance");
}

void pybind_cloudbase_methods(py::module &m) {}

}  // namespace geometry
}  // namespace cloudViewer
