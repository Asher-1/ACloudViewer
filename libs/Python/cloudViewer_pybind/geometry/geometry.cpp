// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

// CV_CORE_LIB
#include <BoundingBox.h>

// ECV_DB_LIB
#include <ecvHObject.h>

#include "cloudViewer_pybind/docstring.h"
#include "cloudViewer_pybind/geometry/geometry.h"
#include "cloudViewer_pybind/geometry/geometry_trampoline.h"

using namespace cloudViewer;

void pybind_geometry_classes(py::module &m) {
	// open3d.geometry functions
	m.def("get_rotation_matrix_from_xyz",
		&ccHObject::GetRotationMatrixFromXYZ, "rotation"_a);
	m.def("get_rotation_matrix_from_yzx",
		&ccHObject::GetRotationMatrixFromYZX, "rotation"_a);
	m.def("get_rotation_matrix_from_zxy",
		&ccHObject::GetRotationMatrixFromZXY, "rotation"_a);
	m.def("get_rotation_matrix_from_xzy",
		&ccHObject::GetRotationMatrixFromXZY, "rotation"_a);
	m.def("get_rotation_matrix_from_zyx",
		&ccHObject::GetRotationMatrixFromZYX, "rotation"_a);
	m.def("get_rotation_matrix_from_yxz",
		&ccHObject::GetRotationMatrixFromYXZ, "rotation"_a);
	m.def("get_rotation_matrix_from_axis_angle",
		&ccHObject::GetRotationMatrixFromAxisAngle, "rotation"_a);
	m.def("get_rotation_matrix_from_euler_angle",
		&ccHObject::GetRotationMatrixFromEulerAngle, "rotation"_a);
	m.def("get_rotation_matrix_from_quaternion",
		&ccHObject::GetRotationMatrixFromQuaternion, "rotation"_a);

	py::class_<ccObject, PyObjectBase<ccObject>, std::shared_ptr<ccObject>>
		geometry(m, "ccObject", "The base geometry class.");
	geometry.def("get_uniqueID", &ccObject::getUniqueID,
		"Returns object unique ID.")
		.def("set_uniqueID", &ccObject::setUniqueID,
			"Changes unique ID.")
		.def("get_name", [](const ccObject& obj) {
			return obj.getName().toStdString(); }, "Returns object name.")
		.def("set_name", [](ccObject& obj, const std::string& name) {
				obj.setName(name.c_str()); }, "Sets object name.")
		.def("is_leaf", &ccObject::isLeaf,
			"Returns whether the geometry is leaf.")
		.def("is_custom", &ccObject::isCustom,
			"Returns whether the geometry is custom.")
		.def("is_hierarchy", &ccObject::isHierarchy,
			"Returns whether the geometry is hierarchy.")
		.def("is_kindOf", &ccObject::isKindOf,
			"Returns whether the geometry is kind of the type pointed.",
			"type"_a)
		.def("is_a", &ccObject::isA,
			"Returns whether the geometry is a type pointed.",
			"type"_a);
	docstring::ClassMethodDocInject(m, "ccObject", "get_uniqueID");
	docstring::ClassMethodDocInject(m, "ccObject", "set_uniqueID");
	docstring::ClassMethodDocInject(m, "ccObject", "get_name");
	docstring::ClassMethodDocInject(m, "ccObject", "set_name");
	docstring::ClassMethodDocInject(m, "ccObject", "is_leaf");
	docstring::ClassMethodDocInject(m, "ccObject", "is_custom");
	docstring::ClassMethodDocInject(m, "ccObject", "is_hierarchy");
	docstring::ClassMethodDocInject(m, "ccObject", "is_kindOf");
	docstring::ClassMethodDocInject(m, "ccObject", "is_a");

	// cloudViewer.geometry.Geometry.Type
	py::enum_<CV_TYPES::GeometryType> geometry_type(geometry, "Type", py::arithmetic());
	// Trick to write docs without listing the members in the enum class again.
	geometry_type.attr("__doc__") = docstring::static_property(
		py::cpp_function([](py::handle arg) -> std::string {
		return "Enum class for Geometry types.";
	}),
		py::none(), py::none(), "");

	geometry_type
		.value("CUSTOM_H_OBJECT", CV_TYPES::CUSTOM_H_OBJECT)
		.value("POINT_CLOUD", CV_TYPES::POINT_CLOUD)
		.value("VOXEL_GRID", CV_TYPES::VOXEL_GRID)
		.value("LINESET", CV_TYPES::LINESET)
		.value("MESH", CV_TYPES::MESH)
		.value("IMAGE2", CV_TYPES::IMAGE2)
		.value("RGBD_IMAGE", CV_TYPES::RGBD_IMAGE)
		.value("TETRA_MESH", CV_TYPES::TETRA_MESH)
		.value("BBOX", CV_TYPES::BBOX)
		.value("ORIENTED_BBOX", CV_TYPES::ORIENTED_BBOX)
		.export_values();

	// cloudViewer.geometry
	py::class_<ccHObject, PyGeometry<ccHObject>, 
		std::shared_ptr<ccHObject>, ccObject>
		geometry3d(m, "ccHObject", py::multiple_inheritance(), "The geometry 3D class.");
	geometry3d.def("get_class_id", &ccHObject::getClassID,
		"Returns class ID.")
		.def("is_group", &ccHObject::isGroup,
			"Returns whether the instance is a group.")
		.def("get_parent", &ccHObject::getParent,
			"Returns parent object.")
		.def("add_child", &ccHObject::addChild,
			"Adds a child.", "child"_a, "dependencyFlags"_a = 24, "insertIndex"_a = -1)
		.def("get_child", &ccHObject::getChild,
			"Returns the ith child.", "childPos"_a)
		.def("remove_child", py::overload_cast<ccHObject*>(&ccHObject::removeChild),
			"Removes a specific child.", "child"_a)
		.def("remove_child", py::overload_cast<int>(&ccHObject::removeChild),
			"Removes a specific child given its index.", "pos"_a)
		.def("remove_all_children", &ccHObject::removeAllChildren,
			"Clear all children in the geometry.")
		.def("set_pointSize_recursive", &ccHObject::setPointSizeRecursive,
			"Sets the point size.", "pSize"_a)
		.def("set_lineWidth_recursive", &ccHObject::setLineWidthRecursive,
			"Sets the line width.", "width"_a)
		.def("is_serializable", &ccHObject::isSerializable,
			"Returns whether the instance is a serializable.")
		.def("is_empty", &ccHObject::isEmpty,
			"Returns ``True`` if the geometry is empty.")
		.def("get_min_bound", &ccHObject::getMinBound,
			"Returns min bounds for geometry coordinates.")
		.def("get_max_bound", &ccHObject::getMaxBound,
			"Returns max bounds for geometry coordinates.")
		.def("get_min_2Dbound", &ccHObject::getMin2DBound,
			"Returns min 2d bounds for geometry coordinates.")
		.def("get_max_2Dbound", &ccHObject::getMax2DBound,
			"Returns max 2d bounds for geometry coordinates.")
		.def("get_center", &ccHObject::getGeometryCenter,
			"Returns the center of the geometry coordinates.")
		.def("get_own_bounding_box", &ccHObject::getOwnBB,
			"Returns an axis-aligned bounding box of the geometry.",
			"withGLFeatures"_a = false)
		.def("get_axis_aligned_bounding_box",
			&ccHObject::getAxisAlignedBoundingBox,
			"Returns an axis-aligned bounding box of the geometry.")
		.def("get_oriented_bounding_box",
			&ccHObject::getOrientedBoundingBox,
			"Returns an oriented bounding box of the geometry.")
		.def("transform", &ccHObject::transform,
			"Apply transformation (4x4 matrix) to the geometry "
			"coordinates.")
		.def("translate", &ccHObject::translate,
			"Apply translation to the geometry coordinates.",
			"translation"_a, "relative"_a = true)
		.def("scale", py::overload_cast<double>(&ccHObject::scale),
			"Apply scaling to the geometry coordinates.", "s"_a)
		.def("scale", py::overload_cast<double, const Eigen::Vector3d &>(&ccHObject::scale),
			"Apply scaling to the geometry coordinates.", "s"_a,
			"center"_a)
		.def("rotate",
			py::overload_cast<const Eigen::Matrix3d &>(&ccHObject::rotate),
			"Apply rotation to the geometry coordinates and normals.",
			"R"_a)
		.def("rotate",
			py::overload_cast<const Eigen::Matrix3d &,
							  const Eigen::Vector3d &>(
				&ccHObject::rotate),
			"Apply rotation to the geometry coordinates and normals.",
			"R"_a, "center"_a)
		.def("to_file", &ccHObject::toFile,
			"Saves data to binary stream.", "out"_a)
		.def("from_file", &ccHObject::fromFile,
			"Loads data from binary stream.", "in"_a, "dataVersion"_a, "flags"_a)
		.def("get_glTransformation_history", &ccHObject::getGLTransformationHistory,
			"Returns the transformation 'history' matrix.")
		.def("set_glTransformation_history", &ccHObject::setGLTransformationHistory,
			"Sets the transformation 'history' matrix (handle with care!).", "mat"_a)
		.def("reset_glTransformation_history", &ccHObject::resetGLTransformationHistory,
			"Resets the transformation 'history' matrix.")
		.def("find", &ccHObject::find,
			"Finds an entity in this object hierarchy.", "uniqueID"_a)
		.def("get_children_number", &ccHObject::getChildrenNumber,
			"Returns the number of children.")
		.def("filter_children", &ccHObject::filterChildren,
			"Collects the children corresponding to a certain pattern.",
			"filteredChildren"_a, 
			"recursive"_a = false, 
			"filter"_a = CV_TYPES::OBJECT, 
			"strict"_a = false)
		.def_static("New",
			py::overload_cast<CV_CLASS_ENUM, const char*>(&ccHObject::New),
			"objectType"_a, "name"_a = nullptr)
		.def_static("New",
			py::overload_cast<const QString&, const QString&,
			const char*>(&ccHObject::New),
			"pluginId"_a, "classId"_a, "name"_a = nullptr)
		.def_static("get_rotation_matrix_from_xyz",
			&ccHObject::GetRotationMatrixFromXYZ,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_yzx",
			&ccHObject::GetRotationMatrixFromYZX,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_zxy",
			&ccHObject::GetRotationMatrixFromZXY,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_xzy",
			&ccHObject::GetRotationMatrixFromXZY,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_zyx",
			&ccHObject::GetRotationMatrixFromZYX,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_yxz",
			&ccHObject::GetRotationMatrixFromYXZ,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_axis_angle",
			&ccHObject::GetRotationMatrixFromAxisAngle,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_euler_angle",
			&ccHObject::GetRotationMatrixFromEulerAngle,
			"rotation"_a)
		.def_static("get_rotation_matrix_from_quaternion",
			&ccHObject::GetRotationMatrixFromQuaternion,
			"rotation"_a);
	docstring::ClassMethodDocInject(m, "ccHObject", "get_class_id");
	docstring::ClassMethodDocInject(m, "ccHObject", "is_group");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_parent");
	docstring::ClassMethodDocInject(m, "ccHObject", "remove_all_children");
	docstring::ClassMethodDocInject(
		m, "ccHObject", "add_child",
		{ {"child", "child instance geometry"},
		 {"dependencyFlags", "dependency flags"},
		 {"insertIndex", "insertion index "
		"(if <0, child is simply appended to the children list)"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "get_child",
		{ {"childPos", "child position"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "set_pointSize_recursive",
		{ {"pSize", "point size"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "set_lineWidth_recursive",
		{ {"width", "line width"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "is_serializable");
	docstring::ClassMethodDocInject(m, "ccHObject", "is_empty");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_min_bound");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_max_bound");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_min_2Dbound");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_max_2Dbound");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_center");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_own_bounding_box",
		{ {"withGLFeatures", "whether use openGL features."} });
	docstring::ClassMethodDocInject(m, "ccHObject",
		"get_axis_aligned_bounding_box");
	docstring::ClassMethodDocInject(m, "ccHObject",
		"get_oriented_bounding_box");
	docstring::ClassMethodDocInject(m, "ccHObject", "transform");
	docstring::ClassMethodDocInject(
		m, "ccHObject", "translate",
		{ {"translation", "A 3D vector to transform the geometry"},
		 {"relative",
		  "If true, the translation vector is directly added to the "
		  "geometry "
		  "coordinates. Otherwise, the center is moved to the translation "
		  "vector."} });
	docstring::ClassMethodDocInject(
		m, "ccHObject", "scale",
		{ {"s",
		  "The scale parameter that is multiplied to the points/vertices "
		  "of the geometry"},
		 {"center", "Scale center used for transformation"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "rotate",
		{ {"R", "The rotation matrix"},
		 {"center", "Rotation center used for transformation"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "to_file",
		{ {"out", "output file (already opened)"} });
	docstring::ClassMethodDocInject(
		m, "ccHObject", "from_file",
		{ {"in", "input file (already opened)"},
		{"dataVersion", "file version"},
		{"flags", "deserialization flags (see ccSerializableObject::DeserializationFlags)"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "get_glTransformation_history");
	docstring::ClassMethodDocInject(
		m, "ccHObject", "set_glTransformation_history",
		{ {"mat", "transformation 'history' matrix"} });
	docstring::ClassMethodDocInject(m, "ccHObject", "reset_glTransformation_history");
	docstring::ClassMethodDocInject(m, "ccHObject", "find");
	docstring::ClassMethodDocInject(m, "ccHObject", "get_children_number");
	docstring::ClassMethodDocInject(
		m, "ccHObject", "filter_children",
		{ {"filteredChildren", "result container"},
		{"recursive", "specifies if the search should be recursive"},
		{"filter", "pattern for children selection"},
		{"strict", "whether the search is strict on the type "
		"(i.e 'isA') or not (i.e. 'isKindOf')"} });
}

void pybind_geometry(py::module &m) {
	py::module m_submodule = m.def_submodule("geometry");
	pybind_geometry_classes(m_submodule);
	pybind_kdtreeflann(m_submodule);
	pybind_pointcloud(m_submodule);
	pybind_voxelgrid(m_submodule);
	pybind_lineset(m_submodule);
	pybind_meshbase(m_submodule);
	pybind_trianglemesh(m_submodule);
	//pybind_halfedgetrianglemesh(m_submodule);
	pybind_image(m_submodule);
	pybind_tetramesh(m_submodule);
	pybind_pointcloud_methods(m_submodule);
	pybind_voxelgrid_methods(m_submodule);
	pybind_meshbase_methods(m_submodule);
	pybind_lineset_methods(m_submodule);
	pybind_image_methods(m_submodule);
	pybind_octree_methods(m_submodule);
	pybind_octree(m_submodule);
	pybind_boundingvolume(m_submodule);
}
