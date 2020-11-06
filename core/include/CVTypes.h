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

#include <stdint.h>

//! Type of the coordinates of a (N-D) point
using PointCoordinateType = float;

//! Type of a single scalar field value
#if defined CV_CORE_LIB_USES_DOUBLE
using ScalarType = double;
#elif defined CV_CORE_LIB_USES_FLOAT
using ScalarType = float;
#else
static_assert(false, "type for ScalarType has not been declared");
#endif //SCALAR_TYPE_DOUBLE

//! Object state flag
enum CV_OBJECT_FLAG {	//CC_UNUSED			= 1, //DGM: not used anymore (former CC_FATHER_DEPENDENT)
	CC_ENABLED = 2,
	CC_LOCKED = 4,
};

//Bits for object type flags (64 bits)
#define CC_HIERARCH_BIT					0x00000000000001	//Hierarchical object
#define CC_LEAF_BIT						0x00000000000002	//Tree leaf (no children)
#define CC_GROUP_BIT					0x00000000000004	//Group (no data, aggregation only)
#define CC_PRIMITIVE_BIT				0x00000000000008	//Primitive (sphere, plane, torus, cylinder, etc.)
#define CC_ARRAY_BIT					0x00000000000010	//Array
#define CC_LABEL_BIT					0x00000000000020	//2D label
#define CC_VIEWPORT_BIT					0x00000000000040	//2D viewport
#define CC_CUSTOM_BIT					0x00000000000080	//For custom (plugin defined) objects
#define CC_CLOUD_BIT					0x00000000000100	//Point Cloud
#define CC_MESH_BIT						0x00000000000200	//Mesh
#define CC_OCTREE_BIT					0x00000000000400	//Octree
#define CC_POLYLINE_BIT					0x00000000000800	//Polyline
#define CC_IMAGE_BIT					0x00000000001000	//Picture
#define CC_SENSOR_BIT					0x00000000002000	//Sensor def.
#define CC_PLANE_BIT					0x00000000004000	//Plane (primitive)
#define CC_SPHERE_BIT					0x00000000008000	//Sphere (primitive)
#define CC_TORUS_BIT					0x00000000010000	//Torus (primitive)
#define CC_CYLINDER_BIT					0x00000000020000	//Cylinder (primitive)
#define CC_CONE_BIT						0x00000000040000	//Cone (primitive)
#define CC_BOX_BIT						0x00000000080000	//Box (primitive)
#define CC_DISH_BIT						0x00000000100000	//Dish (primitive)
#define CC_EXTRU_BIT					0x00000000200000	//Extrusion (primitive)
#define CC_KDTREE_BIT					0x00000000400000	//Kd-tree
#define CC_FACET_BIT					0x00000000800000	//Facet (composite object: cloud + 2D1/2 mesh + 2D1/2 polyline)
#define CC_MATERIAL_BIT					0x00000001000000	//Material
#define CC_CLIP_BOX_BIT					0x00000002000000	//Clipping box
#define CC_TRANS_BUFFER_BIT				0x00000004000000	//Indexed transformation buffer
#define CC_GROUND_BASED_BIT				0x00000008000000	//For Ground Based Lidar Sensors
#define CC_RGB_COLOR_BIT				0x00000010000000	//Color (R,G,B)
#define CC_NORMAL_BIT					0x00000020000000	//Normal (Nx,Ny,Nz)
#define CC_COMPRESSED_NORMAL_BIT		0x00000040000000	//Compressed normal (index)
#define CC_TEX_COORDS_BIT				0x00000080000000	//Texture coordinates (u,v)
#define CC_CAMERA_BIT					0x00000100000000	//For camera sensors (projective sensors)
#define CC_QUADRIC_BIT					0x00000200000000	//Quadric (primitive)
#define CC_OCTREE2_BIT					0x00000400000000
#define CC_IMAGE2_BIT					0x00001000000000
#define CC_RGBD_IMAGE_BIT				0x00002000000000
#define CC_LINE3D_BIT					0x40000000000000
#define CC_MESH_BASE_BIT				0x04000000000000
#define CC_TETRA_MESH_BIT				0x80000000000000
#define CC_HALF_EDGE_MESH_BIT			0x08000000000000
#define CC_BBOX_BIT						0x00800000000000
#define CC_ORIENTED_BBOX_BIT			0x00080000000000
#define CC_LINESET_BIT					0x00008000000000
#define CC_VOXEL_GRID_BIT				0x00000800000000
//#define CC_FREE_BIT					0x00008000000000
//#define CC_FREE_BIT					0x00008000000000
//#define CC_FREE_BIT					...

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
		MESH_BASE			=	HIERARCHY_OBJECT	| CC_MESH_BASE_BIT,
		TETRA_MESH			=	HIERARCHY_OBJECT	| CC_TETRA_MESH_BIT,
		HALF_EDGE_MESH		=	HIERARCHY_OBJECT	| CC_HALF_EDGE_MESH_BIT,
		VOXEL_GRID			=	HIERARCHY_OBJECT	| CC_VOXEL_GRID_BIT,
		SUB_MESH			=	HIERARCHY_OBJECT	| CC_MESH_BIT				| CC_LEAF_BIT,
		MESH_GROUP			=	MESH				| CC_GROUP_BIT,								//DEPRECATED; DEFINITION REMAINS FOR BACKWARD COMPATIBILITY ONLY
		FACET				=	HIERARCHY_OBJECT	| CC_FACET_BIT,
		POINT_OCTREE2		=	HIERARCHY_OBJECT	| CC_OCTREE2_BIT			| CC_LEAF_BIT,
		POINT_OCTREE		=	HIERARCHY_OBJECT	| CC_OCTREE_BIT				| CC_LEAF_BIT,
		POINT_KDTREE		=	HIERARCHY_OBJECT	| CC_KDTREE_BIT				| CC_LEAF_BIT,
		POLY_LINE			=	HIERARCHY_OBJECT	| CC_POLYLINE_BIT,
		LINESET				=	HIERARCHY_OBJECT	| CC_LINESET_BIT,
		LINE3D				=	HIERARCHY_OBJECT	| CC_LINE3D_BIT,
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
