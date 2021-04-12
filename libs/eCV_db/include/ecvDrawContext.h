//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_GL_DRAW_CONTEXT_HEADER
#define ECV_GL_DRAW_CONTEXT_HEADER

#include <CVGeom.h>

#include <QFont>
#include <QRect>
#include <QPoint>
#include <QImage>

#include "ecvGLMatrix.h"
#include "ecvMaterial.h"
#include "ecvColorTypes.h"

class ccHObject;
class ccScalarField;

//! Display parameters of a 3D entity
struct glDrawParams
{
	//! Display scalar field (prioritary on colors)
	bool showSF;
	//! Display colors
	bool showColors;
	//! Display normals
	bool showNorms;
};

enum PROPERTY_MODE
{
	ECV_POINTSSIZE_PROPERTY,
	ECV_LINEWITH_PROPERTY,
	ECV_COLOR_PROPERTY,
	ECV_OPACITY_PROPERTY,
};

enum MESH_RENDERING_MODE
{
	ECV_POINTS_MODE,
	ECV_WIREFRAME_MODE,
	ECV_SURFACE_MODE,
};

enum ENTITY_TYPE
{
	ECV_HIERARCHY_OBJECT,
	ECV_POINT_CLOUD,
	ECV_MARK_POINT,
	ECV_MESH,
	ECV_SHAPE,
	ECV_OCTREE,
	ECV_KDTREE,
	ECV_FACET,
	ECV_LINES_3D,
	ECV_LINES_2D,
	ECV_TRIANGLE_2D,
	ECV_RECTANGLE_2D,
	ECV_POLYLINE_2D,
	ECV_CIRCLE_2D,
	ECV_POLYGON,
	ECV_2DLABLE,
	ECV_2DLABLE_VIEWPORT,
    ECV_CAPTION,
	ECV_SCALAR_BAR,
	ECV_TEXT3D,
	ECV_TEXT2D,
	ECV_IMAGE,
	ECV_SENSOR,
	ECV_ALL,
	ECV_NONE,
};

enum WIDGETS_TYPE
{
	WIDGET_BBOX,
	WIDGET_IMAGE,
	WIDGET_LINE_2D,
	WIDGET_TRIANGLE_2D,
	WIDGET_POINTS_2D,
	WIDGET_CIRCLE_2D,
	WIDGET_RECTANGLE_2D,
	WIDGET_POLYGONMESH,
	WIDGET_COORDINATE,
	WIDGET_POLYLINE,
	WIDGET_POLYLINE_2D,
	WIDGET_LINE_3D,
	WIDGET_SPHERE,
	WIDGET_CAPTION,
	WIDGET_SCALAR_BAR,
	WIDGET_T3D,
	WIDGET_T2D,
};

struct ECV_DB_LIB_API PROPERTY_PARAM
{
	//! Display scalar field (prioritary on colors)
	QString viewId;
    int viewport = 0;
	PointCoordinateType lineWidth = 2;
	unsigned char pointSize = 1;
    ccHObject* entity;
    ecvColor::Rgb color;

    double opacity = 1.0;
	PROPERTY_MODE property;
	ENTITY_TYPE entityType;

	PROPERTY_PARAM(ccHObject* obj, const ecvColor::Rgb& col)
		: entity(obj)
		, color(col)
		, property(PROPERTY_MODE::ECV_COLOR_PROPERTY)
	{}

	PROPERTY_PARAM(ccHObject* obj, double opacity)
		: entity(obj)
		, opacity(opacity)
		, property(PROPERTY_MODE::ECV_OPACITY_PROPERTY)
	{}

	PROPERTY_PARAM(ccHObject* obj, unsigned char pointSize)
		: entity(obj)
		, property(PROPERTY_MODE::ECV_POINTSSIZE_PROPERTY)
	{
		if (pointSize > 1)
		{
			this->pointSize = pointSize;
		}
	}

	inline void setProperty(PROPERTY_MODE mode)
	{
		property = mode;
	}

	inline void setColor(const ecvColor::Rgb& col)
	{
		color = const_cast<ecvColor::Rgb&>(col);
	}

	inline void setPointSize(unsigned char size)
	{
		pointSize = size;
	}

	inline void setOpacity(double op)
	{
		opacity = op;
	}

	inline void setLineWith(PointCoordinateType with)
	{
		lineWidth = with;
	}

};

struct ECV_DB_LIB_API LineWidget {
	CCVector3 lineSt;
	CCVector3 lineEd;
	float lineWidth;
	ecvColor::Rgb lineColor;
    bool valid;
	LineWidget(float width = 2.0f, const ecvColor::Rgb& color = ecvColor::red)
		: lineWidth(width)
		, lineColor(color)
		, valid(false)
	{}

	LineWidget(
		const CCVector3& st, 
		const CCVector3& ed, 
		float width = 2, 
		const ecvColor::Rgb& color = ecvColor::red)
		: lineSt(st)
		, lineEd(ed)
		, lineWidth(width)
		, lineColor(color)
		, valid(true)
	{}
};

//! to be removed structure
struct ECV_DB_LIB_API removeInfo {
	//! Remove type
	ENTITY_TYPE removeType;
	//! Remove viewId
	QString removeId;

	friend bool operator == (const removeInfo& first, const removeInfo& second)
	{
		return (first.removeType == second.removeType && first.removeId == second.removeId) ? true : false;
	}
};

struct ECV_DB_LIB_API hideInfo {
	//! Hide type
	ENTITY_TYPE hideType;
	//! Hide viewId
	QString hideId;

	friend bool operator == (const hideInfo& first, const hideInfo& second)
	{
		return (first.hideType == second.hideType && first.hideId == second.hideId) ? true : false;
	}
};

struct ECV_DB_LIB_API TransformInfo {
	struct RotateParam {
		double angle;
		CCVector3 rotAxis;
	};

	RotateParam rotateParam;
	bool isRotate = false;
    bool applyEuler = true;
	bool isScale = false;
	bool isTranslate = false;
	bool isPositionChanged = false;

    double quaternion[4];

    CCVector3d eulerZYX;

	//! the x y z scale
	CCVector3 scaleXYZ;

	CCVector3 transVecStart;
	CCVector3 transVecEnd;

	CCVector3 position;

	void setPostion(const CCVector3& pos)
	{
		position = pos;
		isPositionChanged = true;
	}

    bool isApplyTransform() const { return isRotate || isScale || isTranslate || isPositionChanged; }

    void setTransformation(const ccGLMatrixd& transform, bool updateTranslation = true, bool useEuler = true) {
        if (useEuler) {
//            double phi_rad, theta_rad, psi_rad;
//            CCVector3d t3D;
//            transform.getParameters(phi_rad, theta_rad, psi_rad, t3D);
//            setRotation(cloudViewer::RadiansToDegrees(psi_rad),
//                        cloudViewer::RadiansToDegrees(theta_rad),
//                        cloudViewer::RadiansToDegrees(phi_rad));
//            if (updateTranslation)
//            {
//                setTranslationStart(CCVector3::fromArray(transform.getTranslationAsVec3D().u));
//            }

            double rz, ry, rx;
            transform.toEulerAngle(rz, ry, rx);
            setRotation(cloudViewer::RadiansToDegrees(rz),
                        cloudViewer::RadiansToDegrees(ry),
                        cloudViewer::RadiansToDegrees(rx));
            if (updateTranslation)
            {
                setTranslationStart(CCVector3::fromArray(transform.getTranslationAsVec3D().u));
            }

        } else {
            double angle_rad;
            CCVector3d axis, trans;
            transform.getParameters(angle_rad, axis, trans);
            double angle_deg = cloudViewer::RadiansToDegrees(angle_rad);
            setRotation(angle_deg, axis);
            if (updateTranslation)
            {
                setTranslationStart(CCVector3::fromArray(trans.u));
            }
        }
    }
    void setTransformation(const ccGLMatrix& transform,
                           bool updateTranslation = true,
                           bool useEuler = true) {
        setTransformation(ccGLMatrixd(transform.data()),
                          updateTranslation, useEuler);
    }

    void setRotation(double zAngle, double yAngle, double xAngle) {
        isRotate = true;
        applyEuler = true;
        eulerZYX[0] = zAngle;
        eulerZYX[1] = yAngle;
        eulerZYX[2] = xAngle;
    }
    void setRotation(double zyxAngle[3]) {
        setRotation(zyxAngle[0], zyxAngle[1], zyxAngle[2]);
    }

    void setRotation(double angle, double x, double y, double z) { setRotation(angle, CCVector3d(x, y, z)); }
	void setRotation(double angle, const CCVector3& axis) {
        isRotate = true;
        applyEuler = false;
		rotateParam.angle = angle; 
		rotateParam.rotAxis = axis; 
	}
	void setRotation(double angle, const CCVector3d& axis) {
		isRotate = true; 
        applyEuler = false;
		rotateParam.angle = angle;
		rotateParam.rotAxis = CCVector3::fromArray(axis.u);
	}

	void setScale(const CCVector3& scale) { isScale = true; scaleXYZ = scale; }
	void setTranslationStart(const CCVector3& trans) { isTranslate = true; transVecStart = trans; }
	void setTranslationEnd(const CCVector3& trans) { isTranslate = true; transVecEnd = trans; }
};

struct ECV_DB_LIB_API ecvTextParam {
	bool display3D = false;
	double textScale = 1.0;
	double opacity = 1.0;
	CCVector3d textPos = CCVector3d(0.0, 0.0, 0.0);
	QString text = "";
	QFont font = QFont();
};

// Drawing flags (type: short)
enum CC_DRAWING_FLAGS
{
	CC_DRAW_2D								= 0x0001,
	CC_DRAW_3D								= 0x0002,
	CC_DRAW_FOREGROUND						= 0x0004,
	CC_LIGHT_ENABLED						= 0x0008,
	CC_SKIP_UNSELECTED						= 0x0010,
	CC_SKIP_SELECTED						= 0x0020,
	CC_SKIP_ALL								= 0x0030,		// = CC_SKIP_UNSELECTED | CC_SKIP_SELECTED
	CC_DRAW_ENTITY_NAMES					= 0x0040,
	//CC_FREE_FLAG							= 0x0080,		// UNUSED (formerly CC_DRAW_POINT_NAMES)
	//CC_FREE_FLAG							= 0x0100,		// UNUSED (formerly CC_DRAW_TRI_NAMES)
	CC_DRAW_FAST_NAMES_ONLY					= 0x0200,
	//CC_FREE_FLAG							= 0x03C0,		// UNUSED (formerly CC_DRAW_ANY_NAMES = CC_DRAW_ENTITY_NAMES | CC_DRAW_POINT_NAMES | CC_DRAW_TRI_NAMES)
	CC_LOD_ACTIVATED						= 0x0400,
	CC_VIRTUAL_TRANS_ENABLED				= 0x0800
};

// Drawing flags testing macros (see ccDrawableObject)
#define MACRO_Draw2D(context)              (context.drawingFlags & CC_DRAW_2D)
#define MACRO_Draw3D(context)              (context.drawingFlags & CC_DRAW_3D)
#define MACRO_DrawEntityNames(context)     (context.drawingFlags & CC_DRAW_ENTITY_NAMES)
#define MACRO_DrawFastNamesOnly(context)   (context.drawingFlags & CC_DRAW_FAST_NAMES_ONLY)
#define MACRO_SkipUnselected(context)      (context.drawingFlags & CC_SKIP_UNSELECTED)
#define MACRO_SkipSelected(context)        (context.drawingFlags & CC_SKIP_SELECTED)
#define MACRO_LightIsEnabled(context)      (context.drawingFlags & CC_LIGHT_ENABLED)
#define MACRO_Foreground(context)          (context.drawingFlags & CC_DRAW_FOREGROUND)
#define MACRO_LODActivated(context)        (context.drawingFlags & CC_LOD_ACTIVATED)
#define MACRO_VirtualTransEnabled(context) (context.drawingFlags & CC_VIRTUAL_TRANS_ENABLED)

//! Display context
struct ccGLDrawContext
{
	//! Drawing options (see below)
	int drawingFlags;
	bool forceRedraw;
	bool visFiltering;
	TransformInfo transformInfo;
	ecvTextParam textParam;

	QString viewID;
	int defaultViewPort;
	int normalDensity;
	float normalScale;

	float opacity;
	
	bool visible;
	unsigned char defaultLineWidth;
	unsigned char currentLineWidth;
	unsigned char defaultPointSize;
	glDrawParams drawParam;
	MESH_RENDERING_MODE meshRenderingMode;
	ENTITY_TYPE hideShowEntityType;

	QString removeViewID;
	ENTITY_TYPE removeEntityType;

	bool clearDepthLayer;
	bool clearColorLayer;
	
	//! GL screen widthecvColor
	int glW;
	//! GL screen height
	int glH;
	//! Device pixel ratio (general 1, 2 on HD displays)
	float devicePixelRatio;

	//! Current zoom (screen to file rendering mode)
	float renderZoom;

	//! Default material
	ccMaterial::Shared defaultMat;

	//! Default color for mesh (front side)
	ecvColor::Rgbaf defaultMeshFrontDiff;
	//! Default color for mesh (back side)
	ecvColor::Rgbaf defaultMeshBackDiff;	
	//! Default color for mesh
	ecvColor::Rgb defaultMeshColor;

	//! Default color for polyline 
	ecvColor::Rgb defaultPolylineColor;

	//! Default point color
	ecvColor::Rgbub pointsDefaultCol;
	//! Current point color
	ecvColor::Rgbub pointsCurrentCol;
	//! Default text color
	ecvColor::Rgbub textDefaultCol;
	//! Default label background color
	ecvColor::Rgbub labelDefaultBkgCol;

	ecvColor::Rgbf viewDefaultBkgCol;

	//! Default label marker color
	ecvColor::Rgbub labelDefaultMarkerCol;
	//! Default bounding-box color
	ecvColor::Rgbub bbDefaultCol;

    ecvColor::Rgbub backgroundCol;
    ecvColor::Rgbub backgroundCol2;
    bool drawBackgroundGradient;

	//! Whether to decimate big clouds when updating the 3D view
	bool decimateCloudOnMove;
	//! Minimum number of points for activating LOD display
	unsigned minLODPointCount;
	//! Current level for LOD display
	unsigned char currentLODLevel;
	//! Wheter more points are available or not at the current level
	bool moreLODPointsAvailable;
	//! Wheter higher levels are available or not
	bool higherLODLevelsAvailable;

	//! Whether to decimate big meshes when rotating the camera
	bool decimateMeshOnMove;
	//! Minimum number of triangles for activating LOD display
	unsigned minLODTriangleCount;

	//! Currently displayed color scale (the corresponding scalar field in fact)
	ccScalarField* sfColorScaleToDisplay;
	
	//! Use VBOs for faster display
	bool useVBOs;

	//! Label marker size (radius)
	float labelMarkerSize;
	//! Shift for 3D label marker display (around the marker, in pixels)
	float labelMarkerTextShift_pix;

	//! Numerical precision (for displaying text)
	unsigned dispNumberPrecision;

	//! Label background opacity
	unsigned labelOpacity;

	//! Stereo pass index
	unsigned stereoPassIndex;

	//! Whether to draw rounded points (instead of sqaures)
	bool drawRoundedPoints;

	//Default constructor
	ccGLDrawContext()
		: drawingFlags(0)
		, forceRedraw(true)
        , visFiltering(false)
        , viewID("unnamed")
        , defaultViewPort(0)
        , normalDensity(100)
        , normalScale(0.02f)
        , opacity(1.0)
        , visible(true)
        , defaultLineWidth(2)
        , currentLineWidth(defaultLineWidth)
        , defaultPointSize(1)
        , meshRenderingMode(MESH_RENDERING_MODE::ECV_SURFACE_MODE)
        , removeViewID("unnamed")
        , removeEntityType(ENTITY_TYPE::ECV_POINT_CLOUD)
        , clearDepthLayer(true)
        , clearColorLayer(true)
		, glW(0)
		, glH(0)
		, devicePixelRatio(1.0f)
		, renderZoom(1.0f)
		, defaultMat(new ccMaterial("default"))
		, defaultMeshFrontDiff(ecvColor::defaultMeshFrontDiff)
		, defaultMeshBackDiff(ecvColor::defaultMeshBackDiff)
		, defaultMeshColor(ecvColor::lightGrey)
        , defaultPolylineColor(ecvColor::green)
        , pointsDefaultCol(ecvColor::defaultColor)
        , pointsCurrentCol(ecvColor::defaultColor)
        , textDefaultCol(ecvColor::defaultColor)
        , labelDefaultBkgCol(ecvColor::defaultLabelBkgColor)
        , viewDefaultBkgCol(ecvColor::defaultViewBkgColor)
        , labelDefaultMarkerCol(ecvColor::defaultLabelMarkerColor)
        , bbDefaultCol(ecvColor::yellow)
        , backgroundCol(ecvColor::defaultBkgColor)
        , backgroundCol2(ecvColor::defaultLabelBkgColor)
		, drawBackgroundGradient(true)
        , decimateCloudOnMove(true)
        , minLODPointCount(10000000)
        , currentLODLevel(0)
        , moreLODPointsAvailable(false)
        , higherLODLevelsAvailable(false)
        , decimateMeshOnMove(true)
        , minLODTriangleCount(2500000)
        , sfColorScaleToDisplay(nullptr)
        , useVBOs(true)
        , labelMarkerSize(5)
        , labelMarkerTextShift_pix(5)
        , dispNumberPrecision(6)
        , labelOpacity(100)
        , stereoPassIndex(0)
        , drawRoundedPoints(false)
	{}
};

using CC_DRAW_CONTEXT = ccGLDrawContext;

struct ECV_DB_LIB_API WIDGETS_PARAMETER {
public:
	/*for general*/
    ccHObject* entity;
    WIDGETS_TYPE type;
	QString viewID;
    int viewport = 0;
	ecvColor::Rgbaf color;

	CC_DRAW_CONTEXT context;

	int fontSize = 10;

	/*for image*/
	QImage image;
	double opacity = 1.0;

	/*for text*/
	QString text;

	/*for rectangle*/
	bool filled = true;
	QRect rect;

	/*for 2D line or triangle*/
	QPoint p1;
	QPoint p2;
	QPoint p3;
	QPoint p4 = QPoint(-1, -1);

	/*for circle, sphere or mark point*/
	float radius;
	CCVector3 center;
	CCVector2 pos;
	bool handleEnabled = false;

	/*for 3D line*/
	LineWidget lineWidget;

	//Default constructor
	WIDGETS_PARAMETER(WIDGETS_TYPE t, QString id = "id", int port = 0)
        : type(t)
        , viewID(id)
        , viewport(port)
	{
		context.viewID = viewID;
	}

	WIDGETS_PARAMETER(ccHObject* obj, WIDGETS_TYPE t, QString id = "id", int port = 0)
		: entity(obj)
        , type(t)
        , viewID(id)
        , viewport(port)
	{
		context.viewID = viewID;
	}

	void setLineWidget(const LineWidget& line)
	{
		lineWidget = line;
	}
};

#endif // ECV_GL_DRAW_CONTEXT_HEADER
