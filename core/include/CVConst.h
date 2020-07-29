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

#ifndef CV_CONST_HEADER
#define CV_CONST_HEADER

//system
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdint.h>

//! Pi
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//! Pi/2
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif

//! Square root of 3
#ifndef SQRT_3
#define SQRT_3 1.7320508075688772935274463415059
#endif

//! Conversion factor from radians to degrees
#ifndef CV_RAD_TO_DEG
#define CV_RAD_TO_DEG (180.0/M_PI)
#endif

//! Conversion factor from degrees to radians
#ifndef CV_DEG_TO_RAD
#define CV_DEG_TO_RAD (M_PI/180.0)
#endif

//! Numerical threshold for considering a value as "zero"
#ifndef ZERO_TOLERANCE
#define ZERO_TOLERANCE static_cast<double>(FLT_EPSILON)
#endif

using PointCoordinateType = float;
using ScalarType = float;

//! '1' as a PointCoordinateType value
/** To avoid compiler warnings about 'possible loss of data' **/
const PointCoordinateType PC_ONE = static_cast<PointCoordinateType>(1.0);

//! 'NaN' as a PointCoordinateType value
/** \warning: handle with care! **/
const PointCoordinateType PC_NAN = std::numeric_limits<PointCoordinateType>::quiet_NaN();

//! NaN as a ScalarType value
/** \warning: handle with care! **/
const ScalarType NAN_VALUE = std::numeric_limits<ScalarType>::quiet_NaN();

const float MIN_POINT_SIZE_F = 1.0f;
const float MAX_POINT_SIZE_F = 16.0f;
const float MIN_LINE_WIDTH_F = 1.0f;
const float MAX_LINE_WIDTH_F = 16.0f;

//Min and max zoom ratio (relative)
static const float CC_GL_MAX_ZOOM_RATIO = 1.0e6f;
static const float CC_GL_MIN_ZOOM_RATIO = 1.0e-6f;

const double EPSILON_VALUE = 1.0e-5;

// Point visibility states
// By default visibility is expressed relatively to the sensor point of view.
// Warning: 'visible' value must always be the lowest!
const unsigned char POINT_VISIBLE				=	 0;				/**< Point visibility state: visible **/
const unsigned char POINT_HIDDEN				=	 1;				/**< Point visibility state: hidden (e.g. behind other points) **/
const unsigned char POINT_OUT_OF_RANGE			=	 2;				/**< Point visibility state: out of range **/
const unsigned char POINT_OUT_OF_FOV			=	 4;				/**< Point visibility state: out of field of view **/


	//! View orientation
enum CC_VIEW_ORIENTATION {
	CC_TOP_VIEW,	/**< Top view (eye: +Z) **/
	CC_BOTTOM_VIEW,	/**< Bottom view **/
	CC_FRONT_VIEW,	/**< Front view **/
	CC_BACK_VIEW,	/**< Back view **/
	CC_LEFT_VIEW,	/**< Left view **/
	CC_RIGHT_VIEW,	/**< Right view **/
	CC_ISO_VIEW_1,	/**< Isometric view 1: front, right and top **/
	CC_ISO_VIEW_2,	/**< Isometric view 2: back, left and top **/
};


//! Chamfer distances types
enum CV_CHAMFER_DISTANCE_TYPE { CHAMFER_111		=	0,				/**< Chamfer distance <1-1-1> **/
								CHAMFER_345		=	1				/**< Chamfer distance <3-4-5> **/
};

//! Types of local models (no model, least square best fitting plan, Delaunay 2D1/2 triangulation, height function)
enum CV_LOCAL_MODEL_TYPES {NO_MODEL				=	0,				/**< No local model **/
							LS					=	1,				/**< Least Square best fitting plane **/
							TRI					=	2,				/**< 2.5D Delaunay triangulation **/
							QUADRIC				=	3				/**< 2.5D quadric function **/
};

//! Min number of points to compute local models (see CV_LOCAL_MODEL_TYPES)
const unsigned CV_LOCAL_MODEL_MIN_SIZE[] = {		1,				/**< for single point model (i.e. no model ;) **/
													3,				/**< for least Square best fitting plane **/
													3,				/**< for Delaunay triangulation (2.5D) **/
													6,				/**< for Quadratic 'height' function **/
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
#define CC_VOXEL_GRID_BIT				0x00000800000000
#define CC_OCTREE2_BIT					0x00000400000000
#define CC_IMAGE2_BIT					0x00001000000000
#define CC_RGBD_IMAGE_BIT				0x00002000000000
#define CC_TETRA_MESH_BIT				0x00004000000000
#define CC_LINESET_BIT					0x00008000000000
#define CC_BBOX_BIT						0x00800000000000
#define CC_ORIENTED_BBOX_BIT			0x00080000000000
//#define CC_FREE_BIT					0x00008000000000
//#define CC_FREE_BIT					...

#endif // CV_CONST_HEADER
