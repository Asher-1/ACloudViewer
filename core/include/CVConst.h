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

#include "CVTypes.h"

//system
#include <cfloat>
#include <cmath>
#include <limits>

//! Pi
#ifndef M_PI
constexpr double M_PI = 3.14159265358979323846;
#endif

//! Pi/2
#ifndef M_PI_2
constexpr double M_PI_2 = 1.57079632679489661923;
#endif

//! Square root of 3
#ifndef SQRT_3
constexpr double SQRT_3 = 1.7320508075688772935274463415059;
#endif

//! Conversion factor from radians to degrees
#ifndef CV_RAD_TO_DEG
//! Conversion factor from radians to degrees
[[deprecated("Use CVLib::RadiansToDegrees()")]]
constexpr double CV_RAD_TO_DEG = (180.0 / M_PI);
#endif

//! Conversion factor from degrees to radians
#ifndef CV_DEG_TO_RAD
constexpr double CV_DEG_TO_RAD = (M_PI / 180.0);
#endif

//! Numerical threshold for considering a value as "zero"
#ifndef ZERO_TOLERANCE
[[deprecated( "Use ZERO_TOLERANCE_F/ZERO_TOLERANCE_D or CVLib::LessThanEpsilon()/CVLib::GreaterThanEpsilon()" )]]
constexpr double ZERO_TOLERANCE = std::numeric_limits<float>::epsilon();
#endif

//! ZERO_TOLERANCE_F is used to set or compare a float variable to "close to zero".
#ifndef ZERO_TOLERANCE_F
[[deprecated( "Use ZERO_TOLERANCE_F/ZERO_TOLERANCE_D or CVLib::LessThanEpsilon()/CVLib::GreaterThanEpsilon()" )]]
constexpr float ZERO_TOLERANCE_F = std::numeric_limits<float>::epsilon();
#endif

//! ZERO_TOLERANCE_D is used to set or compare a double variable to "close to zero".
//! It is defined as std::numeric_limits<float>::epsilon() because using
//! std::numeric_limits<double>::epsilon() results in numbers that are too small for our purposes.
constexpr double ZERO_TOLERANCE_D = static_cast<double>(ZERO_TOLERANCE_F);

//! ZERO_TOLERANCE_SCALAR is used to set or compare a ScalarType variable to "close to zero".
constexpr ScalarType ZERO_TOLERANCE_SCALAR = std::numeric_limits<ScalarType>::epsilon();

//! ZERO_TOLERANCE_POINT_COORDINATE is used to set or compare a PointCoordinateType variable to "close to zero".
constexpr ScalarType ZERO_TOLERANCE_POINT_COORDINATE = std::numeric_limits<PointCoordinateType>::epsilon();

//! '1' as a PointCoordinateType value
/** To avoid compiler warnings about 'possible loss of data' **/
constexpr PointCoordinateType PC_ONE = static_cast<PointCoordinateType>(1.0);

//! 'NaN' as a PointCoordinateType value
/** \warning: handle with care! **/
constexpr PointCoordinateType PC_NAN = std::numeric_limits<PointCoordinateType>::quiet_NaN();

//! NaN as a ScalarType value
/** \warning: handle with care! **/
constexpr ScalarType NAN_VALUE = std::numeric_limits<ScalarType>::quiet_NaN();

constexpr float MIN_POINT_SIZE_F = 1.0f;
constexpr float MAX_POINT_SIZE_F = 16.0f;
constexpr float MIN_LINE_WIDTH_F = 1.0f;
constexpr float MAX_LINE_WIDTH_F = 16.0f;

//Min and max zoom ratio (relative)
static constexpr float CC_GL_MAX_ZOOM_RATIO = 1.0e6f;
static constexpr float CC_GL_MIN_ZOOM_RATIO = 1.0e-6f;

constexpr double EPSILON_VALUE = 1.0e-5;

// Point visibility states
// By default visibility is expressed relatively to the sensor point of view.
// Warning: 'visible' value must always be the lowest!
constexpr unsigned char POINT_VISIBLE				=	 0;				/**< Point visibility state: visible **/
constexpr unsigned char POINT_HIDDEN				=	 1;				/**< Point visibility state: hidden (e.g. behind other points) **/
constexpr unsigned char POINT_OUT_OF_RANGE			=	 2;				/**< Point visibility state: out of range **/
constexpr unsigned char POINT_OUT_OF_FOV			=	 4;				/**< Point visibility state: out of field of view **/

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
enum CHAMFER_DISTANCE_TYPE	{ CHAMFER_111		=	0,				/**< Chamfer distance <1-1-1> **/
							  CHAMFER_345		=	1				/**< Chamfer distance <3-4-5> **/
};

//! Types of local models (no model, least square best fitting plan, Delaunay 2D1/2 triangulation, height function)
enum CV_LOCAL_MODEL_TYPES {NO_MODEL				=	0,				/**< No local model **/
							LS					=	1,				/**< Least Square best fitting plane **/
							TRI					=	2,				/**< 2.5D Delaunay triangulation **/
							QUADRIC				=	3				/**< 2.5D quadric function **/
};

//! Min number of points to compute local models (see CV_LOCAL_MODEL_TYPES)
constexpr unsigned CV_LOCAL_MODEL_MIN_SIZE[] = {	1,				/**< for single point model (i.e. no model ;) **/
													3,				/**< for least Square best fitting plane **/
													3,				/**< for Delaunay triangulation (2.5D) **/
													6,				/**< for Quadratic 'height' function **/
};

#endif // CV_CONST_HEADER
