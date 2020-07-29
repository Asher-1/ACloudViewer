//##########################################################################
//#                                                                        #
//#                               CVLIB                                    #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef CV_TYPES_HEADER
#define CV_TYPES_HEADER

#include "CVConst.h"

//! Type of the coordinates of a (N-D) point
using PointCoordinateType = float;

//! Type of a single scalar field value
using ScalarType = float;

//! Object state flag
enum CV_OBJECT_FLAG {	//CC_UNUSED			= 1, //DGM: not used anymore (former CC_FATHER_DEPENDENT)
	CC_ENABLED = 2,
	CC_LOCKED = 4,
};

//! Type of object type flags (64 bits)
using CV_CLASS_ENUM = int64_t;

//! CVLib  object type flags
namespace CV_TYPES
{
	typedef enum : CV_CLASS_ENUM {
		OBJECT = 0,
		HIERARCHY_OBJECT	=	CC_HIERARCH_BIT,
		POINT_CLOUD			=	HIERARCHY_OBJECT	| CC_CLOUD_BIT,
		MESH				=	HIERARCHY_OBJECT	| CC_MESH_BIT,
		TETRA_MESH			=	HIERARCHY_OBJECT	| CC_TETRA_MESH_BIT,
		VOXEL_GRID			=	HIERARCHY_OBJECT	| CC_VOXEL_GRID_BIT,
		SUB_MESH			=	HIERARCHY_OBJECT	| CC_MESH_BIT				| CC_LEAF_BIT,
		MESH_GROUP			=	MESH				| CC_GROUP_BIT,								//DEPRECATED; DEFINITION REMAINS FOR BACKWARD COMPATIBILITY ONLY
		FACET				=	HIERARCHY_OBJECT	| CC_FACET_BIT,
		POINT_OCTREE2		=	HIERARCHY_OBJECT	| CC_OCTREE2_BIT			| CC_LEAF_BIT,
		POINT_OCTREE		=	HIERARCHY_OBJECT	| CC_OCTREE_BIT				| CC_LEAF_BIT,
		POINT_KDTREE		=	HIERARCHY_OBJECT	| CC_KDTREE_BIT				| CC_LEAF_BIT,
		POLY_LINE			=	HIERARCHY_OBJECT	| CC_POLYLINE_BIT,
		LINESET				=	HIERARCHY_OBJECT	| CC_LINESET_BIT,
		BBOX				=	HIERARCHY_OBJECT	| CC_BBOX_BIT,
		ORIENTED_BBOX		=	HIERARCHY_OBJECT	| CC_ORIENTED_BBOX_BIT,
		IMAGE				=	CC_HIERARCH_BIT		| CC_IMAGE_BIT,
		IMAGE2				=	CC_HIERARCH_BIT		| CC_IMAGE2_BIT,
		RGBD_IMAGE			=	CC_HIERARCH_BIT		| CC_RGBD_IMAGE_BIT,
		CALIBRATED_IMAGE	=	IMAGE				| CC_LEAF_BIT,
		SENSOR				=	CC_HIERARCH_BIT		| CC_SENSOR_BIT,
		GBL_SENSOR			=	SENSOR				| CC_GROUND_BASED_BIT,
		CAMERA_SENSOR		=	SENSOR				| CC_CAMERA_BIT,
		PRIMITIVE			=	MESH				| CC_PRIMITIVE_BIT,							//primitives are meshes
		PLANE				=	PRIMITIVE			| CC_PLANE_BIT,
		SPHERE				=	PRIMITIVE			| CC_SPHERE_BIT,
		TORUS				=	PRIMITIVE			| CC_TORUS_BIT,
		CONE				=	PRIMITIVE			| CC_CONE_BIT,
		OLD_CYLINDER_ID		=	PRIMITIVE			| CC_CYLINDER_BIT,							//for backward compatibility
		CYLINDER			=	PRIMITIVE			| CC_CYLINDER_BIT			| CC_CONE_BIT,	//cylinders are cones
		BOX					=	PRIMITIVE			| CC_BOX_BIT,
		DISH				=	PRIMITIVE			| CC_DISH_BIT,
		EXTRU				=	PRIMITIVE			| CC_EXTRU_BIT,
		QUADRIC				=	PRIMITIVE			| CC_QUADRIC_BIT,
		MATERIAL_SET		=	CC_MATERIAL_BIT		| CC_GROUP_BIT				| CC_LEAF_BIT,
		ARRAY				=	CC_ARRAY_BIT,
		NORMALS_ARRAY		=	CC_ARRAY_BIT		| CC_NORMAL_BIT				| CC_LEAF_BIT,
		NORMAL_INDEXES_ARRAY=	CC_ARRAY_BIT		| CC_COMPRESSED_NORMAL_BIT	| CC_LEAF_BIT,
		RGB_COLOR_ARRAY		=	CC_ARRAY_BIT		| CC_RGB_COLOR_BIT			| CC_LEAF_BIT,
		TEX_COORDS_ARRAY	=	CC_ARRAY_BIT		| CC_TEX_COORDS_BIT			| CC_LEAF_BIT,
		LABEL_2D			=	HIERARCHY_OBJECT	| CC_LABEL_BIT				| CC_LEAF_BIT,
		VIEWPORT_2D_OBJECT	=	HIERARCHY_OBJECT	| CC_VIEWPORT_BIT			| CC_LEAF_BIT,
		VIEWPORT_2D_LABEL	=	VIEWPORT_2D_OBJECT	| CC_LABEL_BIT,
		CLIPPING_BOX		=	CC_CLIP_BOX_BIT		| CC_LEAF_BIT,
		TRANS_BUFFER		=	HIERARCHY_OBJECT	| CC_TRANS_BUFFER_BIT		| CC_LEAF_BIT,
		
		//  Custom types
		/** Custom objects are typically defined by plugins. They can be inserted in an object
			hierarchy or displayed in an OpenGL context like any other ccHObject.
			To differentiate custom objects, use the meta-data mechanism (see ccObject::getMetaData
			and ccObject::setMetaData). You can also define a custom icon (see ccHObject::getIcon).
	
			It is highly advised to use the ccCustomHObject and ccCustomLeafObject interfaces to
			define a custom types. Carefully read the ccCustomHObject::isDeserialized method's
			description and the warning below!
	
			Warning: custom objects can't be 'fully' serialized. Don't overload the
			'ccSerializableObject::toFile' method for them as this would break the deserialization mechanism!
			They can only be serialized as plain ccHObject instances (CV_TYPES::HIERARCHY_OBJECT).
			Hierarchical custom objects (CV_TYPES::CUSTOM_H_OBJECT) will be deserialized as ccCustomHObject
			instances. Leaf custom objects (CV_TYPES::CUSTOM_LEAF_OBJECT) will be deserialized as
			ccCustomLeafObject instances.
		**/
		CUSTOM_H_OBJECT		=	HIERARCHY_OBJECT | CC_CUSTOM_BIT,
		CUSTOM_LEAF_OBJECT	=	CUSTOM_H_OBJECT | CC_LEAF_BIT,
	} GeometryType;
}


#endif //CV_TYPES_HEADER
