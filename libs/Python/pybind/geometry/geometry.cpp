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

// CV_CORE_LIB
#include <BoundingBox.h>

// ECV_DB_LIB
#include <ecv2DLabel.h>
#include <ecv2DViewportLabel.h>
#include <ecv2DViewportObject.h>
#include <ecvCameraSensor.h>
#include <ecvCone.h>
#include <ecvCylinder.h>
#include <ecvDish.h>
#include <ecvExtru.h>
#include <ecvFacet.h>
#include <ecvGBLSensor.h>
#include <ecvHObject.h>
#include <ecvImage.h>
#include <ecvKdTree.h>
#include <ecvMesh.h>
#include <ecvOctreeProxy.h>
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvShiftedObject.h>
#include <ecvSphere.h>
#include <ecvSubMesh.h>
#include <ecvTorus.h>
#include <ecvBox.h>
#include <ecvQuadric.h>
#include <Image.h>
#include <Octree.h>
#include <LineSet.h>
#include <RGBDImage.h>
#include <VoxelGrid.h>
#include <ecvSensor.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include "ecvIndexedTransformationBuffer.h"

#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

#pragma warning(disable:4715)

namespace cloudViewer {
namespace geometry {

void pybind_geometry_classes(py::module &m) {
	// cloudViewer.geometry functions
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

	m.def("ToGenericPointCloud", [](ccHObject& entity) {
		if (ccHObjectCaster::ToGenericPointCloud(&entity)) {
			return std::ref(*ccHObjectCaster::ToGenericPointCloud(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccGenericPointCloud (if possible)", "entity"_a);
	m.def("ToPointCloud", [](ccHObject& entity) {
		if (ccHObjectCaster::ToPointCloud(&entity)) {
			return std::ref(*ccHObjectCaster::ToPointCloud(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to 'equivalent' ccPointCloud", "entity"_a);
	m.def("ToShifted", [](ccHObject& entity) {
		if (ccHObjectCaster::ToShifted(&entity)) {
			return std::ref(*ccHObjectCaster::ToShifted(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to 'equivalent' ccShiftedObject", "entity"_a);
	m.def("ToPolyline", [](ccHObject& entity) {
		if (ccHObjectCaster::ToPolyline(&entity)) {
			return std::ref(*ccHObjectCaster::ToPolyline(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccPolyline (if possible)", "entity"_a);
	m.def("ToFacet", [](ccHObject& entity) {
		if (ccHObjectCaster::ToFacet(&entity)) {
			return std::ref(*ccHObjectCaster::ToFacet(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccFacet (if possible)", "entity"_a);
	m.def("ToGenericMesh", [](ccHObject& entity) {
		if (ccHObjectCaster::ToGenericMesh(&entity)) {
			return std::ref(*ccHObjectCaster::ToGenericMesh(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccGenericMesh (if possible)", "entity"_a);
	m.def("ToMesh", [](ccHObject& entity) {
		if (ccHObjectCaster::ToMesh(&entity)) {
			return std::ref(*ccHObjectCaster::ToMesh(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccMesh (if possible)", "entity"_a);
	m.def("ToSubMesh", [](ccHObject& entity) {
		if (ccHObjectCaster::ToSubMesh(&entity)) {
			return std::ref(*ccHObjectCaster::ToSubMesh(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccSubMesh (if possible)", "entity"_a);
	m.def("ToPlanarEntity", [](ccHObject& entity) {
		if (ccHObjectCaster::ToPlanarEntity(&entity)) {
			return std::ref(*ccHObjectCaster::ToPlanarEntity(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccPlanarEntityInterface (if possible)", "entity"_a);
	m.def("ToQuadric", [](ccHObject& entity) {
		if (ccHObjectCaster::ToQuadric(&entity)) {
			return std::ref(*ccHObjectCaster::ToQuadric(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccQuadric (if possible)", "entity"_a);
	m.def("ToBox", [](ccHObject& entity) {
		if (ccHObjectCaster::ToBox(&entity)) {
			return std::ref(*ccHObjectCaster::ToBox(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccBox (if possible)", "entity"_a);
	m.def("ToSphere", [](ccHObject& entity) {
		if (ccHObjectCaster::ToSphere(&entity)) {
			return std::ref(*ccHObjectCaster::ToSphere(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccSphere (if possible)", "entity"_a);
	m.def("ToCylinder", [](ccHObject& entity) {
		if (ccHObjectCaster::ToCylinder(&entity)) {
			return std::ref(*ccHObjectCaster::ToCylinder(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccCylinder (if possible)", "entity"_a);
	m.def("ToCone", [](ccHObject& entity) {
		if (ccHObjectCaster::ToCone(&entity)) {
			return std::ref(*ccHObjectCaster::ToCone(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccCone (if possible)", "entity"_a);
	m.def("ToPlane", [](ccHObject& entity) {
		if (ccHObjectCaster::ToPlane(&entity)) {
			return std::ref(*ccHObjectCaster::ToPlane(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccPlane (if possible)", "entity"_a);
	m.def("ToDish", [](ccHObject& entity) {
		if (ccHObjectCaster::ToDish(&entity)) {
			return std::ref(*ccHObjectCaster::ToDish(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccDish (if possible)", "entity"_a);
	m.def("ToExtru", [](ccHObject& entity) {
		if (ccHObjectCaster::ToExtru(&entity)) {
			return std::ref(*ccHObjectCaster::ToExtru(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccExtru (if possible)", "entity"_a);
	m.def("ToTorus", [](ccHObject& entity) {
		if (ccHObjectCaster::ToTorus(&entity)) {
			return std::ref(*ccHObjectCaster::ToTorus(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccTorus (if possible)", "entity"_a);
	
	m.def("ToOctreeProxy", [](ccHObject& entity) {
		if (ccHObjectCaster::ToOctreeProxy(&entity)) {
			return std::ref(*ccHObjectCaster::ToOctreeProxy(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccOctreeProxy (if possible)", "entity"_a);
	m.def("ToKdTree", [](ccHObject& entity) {
		if (ccHObjectCaster::ToKdTree(&entity)) {
			return std::ref(*ccHObjectCaster::ToKdTree(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccKdTree (if possible)", "entity"_a);
	m.def("ToSensor", [](ccHObject& entity) {
		if (ccHObjectCaster::ToSensor(&entity)) {
			return std::ref(*ccHObjectCaster::ToSensor(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccSensor (if possible)", "entity"_a);
	m.def("ToGBLSensor", [](ccHObject& entity) {
		if (ccHObjectCaster::ToGBLSensor(&entity)) {
			return std::ref(*ccHObjectCaster::ToGBLSensor(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccGBLSensor (if possible)", "entity"_a);
	m.def("ToCameraSensor", [](ccHObject& entity) {
		if (ccHObjectCaster::ToCameraSensor(&entity)) {
			return std::ref(*ccHObjectCaster::ToCameraSensor(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccCameraSensor (if possible)", "entity"_a);
	m.def("ToImage", [](ccHObject& entity) {
		if (ccHObjectCaster::ToImage(&entity)) {
			return std::ref(*ccHObjectCaster::ToImage(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccImage (if possible)", "entity"_a);
	
	m.def("To2DLabel", [](ccHObject& entity) {
		if (ccHObjectCaster::To2DLabel(&entity)) {
			return std::ref(*ccHObjectCaster::To2DLabel(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to cc2DLabel (if possible)", "entity"_a);
	m.def("To2DViewportLabel", [](ccHObject& entity) {
		if (ccHObjectCaster::To2DViewportLabel(&entity)) {
			return std::ref(*ccHObjectCaster::To2DViewportLabel(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to cc2DViewportLabel (if possible)", "entity"_a);
	m.def("To2DViewportObject", [](ccHObject& entity) {
		if (ccHObjectCaster::To2DViewportObject(&entity)) {
			return std::ref(*ccHObjectCaster::To2DViewportObject(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to cc2DViewportObject (if possible)", "entity"_a);
	m.def("ToTransBuffer", [](ccHObject& entity) {
		if (ccHObjectCaster::ToTransBuffer(&entity)) {
			return std::ref(*ccHObjectCaster::ToTransBuffer(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccIndexedTransformationBuffer (if possible)", "entity"_a);
	m.def("ToImage2", [](ccHObject& entity) {
		if (ccHObjectCaster::ToImage2(&entity)) {
			return std::ref(*ccHObjectCaster::ToImage2(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to Image (if possible)", "entity"_a);
	m.def("ToRGBDImage", [](ccHObject& entity) {
		if (ccHObjectCaster::ToRGBDImage(&entity)) {
			return std::ref(*ccHObjectCaster::ToRGBDImage(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to RGBDImage (if possible)", "entity"_a);
	m.def("ToVoxelGrid", [](ccHObject& entity) {
		if (ccHObjectCaster::ToVoxelGrid(&entity)) {
			return std::ref(*ccHObjectCaster::ToVoxelGrid(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to VoxelGrid (if possible)", "entity"_a);
	m.def("ToLineSet", [](ccHObject& entity) {
		if (ccHObjectCaster::ToLineSet(&entity)) {
			return std::ref(*ccHObjectCaster::ToLineSet(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to LineSet (if possible)", "entity"_a);
	m.def("ToOctree2", [](ccHObject& entity) {
		if (ccHObjectCaster::ToOctree2(&entity)) {
			return std::ref(*ccHObjectCaster::ToOctree2(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to Octree (if possible)", "entity"_a);
	m.def("ToBBox", [](ccHObject& entity) {
		if (ccHObjectCaster::ToBBox(&entity)) {
			return std::ref(*ccHObjectCaster::ToBBox(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ccBBox (if possible)", "entity"_a);
	m.def("ToOrientedBBox", [](ccHObject& entity) {
		if (ccHObjectCaster::ToOrientedBBox(&entity)) {
			return std::ref(*ccHObjectCaster::ToOrientedBBox(&entity));
		} else {
			CVLib::utility::LogWarning("[ccHObjectCaster] converting failed!");
		}
	}, "Converts current object to ecvOrientedBBox (if possible)", "entity"_a);

	py::class_<ccObject, PyObjectBase<ccObject>, std::shared_ptr<ccObject>>
		geometry(m, "ccObject", "The base geometry class.");
	geometry.def("get_unique_id", &ccObject::getUniqueID,
		"Returns object unique ID.")
		.def("set_unique_id", &ccObject::setUniqueID,
			"Changes unique ID.", "ID"_a)
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
		.def("is_kind_of", &ccObject::isKindOf,
			"Returns whether the geometry is kind of the type pointed.",
			"type"_a)
		.def("is_a", &ccObject::isA,
			"Returns whether the geometry is a type pointed.",
			"type"_a);
	docstring::ClassMethodDocInject(m, "ccObject", "get_unique_id");
	docstring::ClassMethodDocInject(m, "ccObject", "set_unique_id");
	docstring::ClassMethodDocInject(m, "ccObject", "get_name");
	docstring::ClassMethodDocInject(m, "ccObject", "set_name");
	docstring::ClassMethodDocInject(m, "ccObject", "is_leaf");
	docstring::ClassMethodDocInject(m, "ccObject", "is_custom");
	docstring::ClassMethodDocInject(m, "ccObject", "is_hierarchy");
	docstring::ClassMethodDocInject(m, "ccObject", "is_kind_of");
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
		.value("MESH", CV_TYPES::MESH)
		.value("LINESET", CV_TYPES::LINESET)
		.value("FACET", CV_TYPES::FACET)
		.value("SUB_MESH", CV_TYPES::SUB_MESH)
		.value("MESH_GROUP", CV_TYPES::MESH_GROUP)
		.value("POINT_CLOUD", CV_TYPES::POINT_CLOUD)
		.value("VOXEL_GRID", CV_TYPES::VOXEL_GRID)
		.value("POLY_LINE", CV_TYPES::POLY_LINE)
		.value("IMAGE", CV_TYPES::IMAGE)
		.value("IMAGE2", CV_TYPES::IMAGE2)
		.value("RGBD_IMAGE", CV_TYPES::RGBD_IMAGE)
		.value("TETRA_MESH", CV_TYPES::TETRA_MESH)
		.value("POINT_KDTREE", CV_TYPES::POINT_KDTREE)
		.value("POINT_OCTREE", CV_TYPES::POINT_OCTREE)
		.value("POINT_OCTREE2", CV_TYPES::POINT_OCTREE2)
		.value("BBOX", CV_TYPES::BBOX)
		.value("ORIENTED_BBOX", CV_TYPES::ORIENTED_BBOX)
		.value("PRIMITIVE", CV_TYPES::PRIMITIVE)
		.value("PLANE", CV_TYPES::PLANE)
		.value("SPHERE", CV_TYPES::SPHERE)
		.value("TORUS", CV_TYPES::TORUS)
		.value("CONE", CV_TYPES::CONE)
		.value("OLD_CYLINDER_ID", CV_TYPES::OLD_CYLINDER_ID)
		.value("CYLINDER", CV_TYPES::CYLINDER)
		.value("BOX", CV_TYPES::BOX)
		.value("DISH", CV_TYPES::DISH)
		.value("EXTRU", CV_TYPES::EXTRU)
		.value("QUADRIC", CV_TYPES::QUADRIC)
		.value("LABEL_2D", CV_TYPES::LABEL_2D)
		.value("SENSOR", CV_TYPES::SENSOR)
		.value("GBL_SENSOR", CV_TYPES::GBL_SENSOR)
		.value("CAMERA_SENSOR", CV_TYPES::CAMERA_SENSOR)
		.value("MATERIAL_SET", CV_TYPES::MATERIAL_SET)
		.value("VIEWPORT_2D_LABEL", CV_TYPES::VIEWPORT_2D_LABEL)
		.value("VIEWPORT_2D_OBJECT", CV_TYPES::VIEWPORT_2D_OBJECT)
		.export_values();

	// cloudViewer.geometry.ccDrawableObject
	py::class_<ccDrawableObject, PyDrawableObjectBase<ccDrawableObject>, std::shared_ptr<ccDrawableObject>>
		drawableObject(m, "ccDrawableObject", "The Generic interface for (3D) drawable entities.");
	drawableObject.def("__repr__", [](const ccDrawableObject &painter) {
		return std::string("Generic interface for (3D) drawable entities");
	})
	.def("is_visible",			&ccDrawableObject::isVisible, "Returns whether entity is visible or not.")
	.def("set_visible",			&ccDrawableObject::setVisible, "Sets entity visibility.", "state"_a)
	.def("toggle_visibility",	&ccDrawableObject::toggleVisibility, "Toggles visibility.")
	.def("is_visiblity_locked", &ccDrawableObject::isVisiblityLocked, "Returns whether visibility is locked or not.")
	.def("lock_visibility",		&ccDrawableObject::lockVisibility, "Locks/unlocks visibility.", "state"_a)
	.def("is_selected",			&ccDrawableObject::isSelected, "Returns whether entity is selected or not.")
	.def("set_selected",		&ccDrawableObject::setSelected, "Selects/Unselects entity.", "state"_a)
	.def("has_colors",			&ccDrawableObject::hasColors, "Returns whether colors are enabled or not.")
	.def("colors_shown",		&ccDrawableObject::colorsShown, "Returns whether colors are shown or not.")
	.def("show_colors",			&ccDrawableObject::showColors, "Sets colors visibility.", "state"_a)
	.def("toggle_colors",		&ccDrawableObject::toggleColors, "Toggles colors display state.")
	.def("has_normals",			&ccDrawableObject::hasNormals, "Returns whether normals are enabled or not.")
	.def("normals_shown",		&ccDrawableObject::normalsShown, "Returns whether normals are shown or not.")
	.def("show_normals",		&ccDrawableObject::showNormals, "Sets normals visibility.", "state"_a)
	.def("toggle_normals",		&ccDrawableObject::toggleNormals, "Toggles normals display state.")
	.def("has_displayed_scalar_field", &ccDrawableObject::hasDisplayedScalarField, "Returns whether an active scalar field is available or not.")
	.def("has_scalar_fields",	&ccDrawableObject::hasScalarFields, "Returns whether one or more scalar fields are instantiated.")
	.def("show_sf",				&ccDrawableObject::showSF, "Sets active scalar field visibility.", "state"_a)
	.def("toggle_sf",			&ccDrawableObject::toggleSF, "Toggles SF display state.")
	.def("sf_shown",			&ccDrawableObject::sfShown, "Returns whether active scalar field is visible.")
	.def("toggle_materials",	&ccDrawableObject::toggleMaterials, "Toggles material display state.")
	.def("show_3d_name",		&ccDrawableObject::showNameIn3D, "Sets whether name should be displayed in 3D.", "state"_a)
	.def("name_3d_shown",		&ccDrawableObject::nameShownIn3D, "Returns whether name is displayed in 3D or not.")
	.def("toggle_show_name",	&ccDrawableObject::toggleShowName, "Toggles name in 3D display state.")
	.def("get_opacity",			&ccDrawableObject::getOpacity, "Get opacity.")
	.def("set_opacity",			&ccDrawableObject::setOpacity, "Set opacity activation state.", "opacity"_a)
	.def("is_color_overriden",	&ccDrawableObject::isColorOverriden, "Returns whether colors are currently overridden by a temporary (unique) color.")
	.def("get_temp_color", [](ccDrawableObject& painter) {
			return ecvColor::Rgb::ToEigen(painter.getTempColor());
		}, "Returns current temporary (unique) color.")
	.def("set_temp_color", [](ccDrawableObject& painter, const Eigen::Vector3d& color, bool auto_activate) {
			painter.setTempColor(ecvColor::Rgb::FromEigen(color), auto_activate);
		}, "Sets current temporary (unique).", "color"_a, "auto_activate"_a = true)
	.def("enable_temp_color", &ccDrawableObject::enableTempColor,
		"Set temporary color activation state.", "state"_a)
	.def("set_gl_transformation", [](ccDrawableObject& painter, const Eigen::Matrix4d& transformation) {
			painter.setGLTransformation(ccGLMatrix::FromEigenMatrix(transformation));
		}, "Associates entity with a GL transformation (rotation + translation).", "transformation"_a)
	.def("enable_gl_transformation", &ccDrawableObject::enableGLTransformation, 
		"Enables/disables associated GL transformation.", "state"_a)
	.def("is_gl_trans_enabled", &ccDrawableObject::isGLTransEnabled, 
		"Returns whether a GL transformation is enabled or not.")
	.def("get_gl_transformation", [](const ccDrawableObject& painter) {
			return ccGLMatrix::ToEigenMatrix4(painter.getGLTransformation());
		}, "Returns associated GL transformation.")
	.def("reset_gl_transformation", &ccDrawableObject::resetGLTransformation, "Resets associated GL transformation.")
	.def("rotate_gl", [](ccDrawableObject& painter, const Eigen::Matrix4d& rotation) {
			painter.rotateGL(ccGLMatrix::FromEigenMatrix(rotation));
		}, "Multiplies (left) current GL transformation by a rotation matrix.", "rotation"_a)
	.def("translate_gl", [](ccDrawableObject& painter, const Eigen::Vector3d& translation) {
			painter.translateGL(CCVector3::fromArray(translation));
		}, "Translates current GL transformation by a rotation matrix.", "translation"_a);
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "is_visible");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "set_visible");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_visibility");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "is_visiblity_locked");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "lock_visibility");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "is_selected");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "set_selected");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "has_colors");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "colors_shown");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "show_colors");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_colors");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "has_normals");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "normals_shown");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "show_normals");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_normals");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "has_displayed_scalar_field");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "has_scalar_fields");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "show_sf");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_sf");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "sf_shown");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_materials");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "show_3d_name");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "name_3d_shown");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "toggle_show_name");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "get_opacity");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "set_opacity");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "is_color_overriden");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "get_temp_color");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "set_temp_color",
		{ {"color", "rgb color"}, {"auto_activate", "auto activates temporary color"} });
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "enable_temp_color");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "set_gl_transformation");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "get_gl_transformation");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "enable_gl_transformation");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "is_gl_trans_enabled");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "reset_gl_transformation");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "rotate_gl");
	docstring::ClassMethodDocInject(m, "ccDrawableObject", "translate_gl");

	// cloudViewer.geometry
	py::class_<ccHObject, PyGeometry<ccHObject>, 
		std::shared_ptr<ccHObject>, ccObject, ccDrawableObject>
		geometry3d(m, "ccHObject", py::multiple_inheritance(), "The geometry 3D class.");
		geometry3d.def("__repr__", [](const ccHObject &geometry) {
			return std::string("The ccHObject base class for 3D Geometries.");
		})
		.def("get_class_id", &ccHObject::getClassID, "Returns class ID.")
		.def("is_group", &ccHObject::isGroup, "Returns whether the instance is a group.")
		.def("get_parent", &ccHObject::getParent, "Returns parent object.")
		.def("add_child", &ccHObject::addChild,
			"Adds a child.", "child"_a, "dependencyFlags"_a = 24, "insertIndex"_a = -1)
		.def("get_child", &ccHObject::getChild, "Returns the ith child.", "childPos"_a)
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
		.def("rotate", py::overload_cast<const Eigen::Matrix3d &,
						const Eigen::Vector3d &>(&ccHObject::rotate),
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
			"Finds an entity in this object hierarchy.", "unique_id"_a)
		.def("get_children_number", &ccHObject::getChildrenNumber,
			"Returns the number of children.")
		.def("filter_children", [](const ccHObject& entity, bool recursive/*=false*/,
						CV_CLASS_ENUM filter/*=CV_TYPES::OBJECT*/, bool strict/*=false*/) {
				ccHObject::Container filteredChildren;
				entity.filterChildren(filteredChildren, recursive, filter, strict);
				std::vector<std::shared_ptr<ccHObject>> container;
				for (auto child : filteredChildren)
				{
					const_cast<ccHObject&>(entity).detachChild(child);
					container.push_back(std::shared_ptr<ccHObject>(child));
				}
				
				return container;
			},
			"Collects the children corresponding to a certain pattern.", 
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
		{ {"recursive", "specifies if the search should be recursive"},
		{"filter", "pattern for children selection"},
		{"strict", "whether the search is strict on the type "
		"(i.e 'isA') or not (i.e. 'isKindOf')"} });

	// cloudViewer.geometry.ccPlanarEntityInterface
	py::class_<ccPlanarEntityInterface, 
		PyPlanarEntityInterface<ccPlanarEntityInterface>,
		std::shared_ptr<ccPlanarEntityInterface>>
		planarEntityInterface(m, "ccPlanarEntityInterface", "The Interface for a planar entity.");
	planarEntityInterface.def("__repr__", [](const ccPlanarEntityInterface &planarEI) {
		CCVector3 normal = planarEI.getNormal();
		std::string info = fmt::format(
			"ccPlanarEntityInterface with normal ({}, {}, {}), show normal vector {}",
			normal.x, normal.y, normal.z,
			planarEI.normalVectorIsShown() ? "True" : "False");
	})
	.def("get_normal", [](const ccPlanarEntityInterface &planarEI) {
		return CCVector3d::fromArray(planarEI.getNormal());
		}, "Returns the entity normal")
	.def("show_normal_vector", &ccPlanarEntityInterface::showNormalVector,
		"Show normal vector.", "state"_a)
	.def("normal_vector_is_shown", &ccPlanarEntityInterface::normalVectorIsShown,
		"Whether normal vector is shown or not.");
	docstring::ClassMethodDocInject(m, "ccPlanarEntityInterface", "get_normal");
	docstring::ClassMethodDocInject(m, "ccPlanarEntityInterface", "normal_vector_is_shown");
	docstring::ClassMethodDocInject(m, "ccPlanarEntityInterface", "show_normal_vector",
		{ {"state", "normal vector shown flag."} });
}

void pybind_geometry(py::module &m) {
	py::module m_submodule = m.def_submodule("geometry");
	pybind_geometry_classes(m_submodule);
	pybind_kdtreeflann(m_submodule);
	pybind_cloudbase(m_submodule);
	pybind_pointcloud(m_submodule);
	pybind_keypoint(m_submodule);
	pybind_voxelgrid(m_submodule);
	pybind_lineset(m_submodule);
	pybind_meshbase(m_submodule);
	pybind_trianglemesh(m_submodule);
	pybind_primitives(m_submodule);
	pybind_facet(m_submodule);
	pybind_polyline(m_submodule);
	//pybind_halfedgetrianglemesh(m_submodule);
	pybind_image(m_submodule);
	pybind_tetramesh(m_submodule);
	pybind_cloudbase_methods(m_submodule);
	pybind_pointcloud_methods(m_submodule);
	pybind_voxelgrid_methods(m_submodule);
	pybind_meshbase_methods(m_submodule);
	pybind_primitives_methods(m_submodule);
	pybind_facet_methods(m_submodule);
	pybind_polyline_methods(m_submodule);
	pybind_lineset_methods(m_submodule);
	pybind_image_methods(m_submodule);
	pybind_octree_methods(m_submodule);
	pybind_octree(m_submodule);
	pybind_boundingvolume(m_submodule);
}

}  // namespace geometry
}  // namespace cloudViewer
