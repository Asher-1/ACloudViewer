// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <CVGeom.h>

#include <QFont>
#include <QImage>
#include <QPoint>
#include <QRect>

#include "ecvColorTypes.h"
#include "ecvGLMatrix.h"
#include "ecvMaterial.h"

class ccHObject;
class ccScalarField;

/**
 * @brief Display parameters of a 3D entity
 *
 * This structure defines what visual attributes should be displayed
 * when rendering a 3D entity.
 */
struct glDrawParams {
    //! Display scalar field (prioritary on colors)
    bool showSF;
    //! Display colors
    bool showColors;
    //! Display normals
    bool showNorms;
};

/**
 * @brief Property modes for entity rendering parameters
 *
 * Defines which property is being set or modified on a drawable entity.
 */
enum PROPERTY_MODE {
    ECV_POINTSSIZE_PROPERTY,  ///< Point size property
    ECV_LINEWITH_PROPERTY,    ///< Line width property
    ECV_COLOR_PROPERTY,       ///< Color property
    ECV_OPACITY_PROPERTY,     ///< Opacity/transparency property
    ECV_SHADING_PROPERTY,     ///< Shading mode property
};

/**
 * @brief Shading modes for mesh rendering
 *
 * Defines the shading algorithm used for rendering surfaces.
 */
enum SHADING_MODE {
    ECV_SHADING_FLAT,     ///< Flat shading (constant per face)
    ECV_SHADING_GOURAUD,  ///< Gouraud shading (interpolated per vertex)
    ECV_SHADING_PHONG     ///< Phong shading (per-pixel lighting)
};

/**
 * @brief Mesh rendering modes
 *
 * Defines how a mesh should be rendered.
 */
enum MESH_RENDERING_MODE {
    ECV_POINTS_MODE,     ///< Render as points (vertices only)
    ECV_WIREFRAME_MODE,  ///< Render as wireframe (edges)
    ECV_SURFACE_MODE,    ///< Render as filled surface
};

/**
 * @brief Entity type enumeration
 *
 * Defines all types of entities that can be rendered in the scene.
 */
enum ENTITY_TYPE {
    ECV_HIERARCHY_OBJECT,  ///< Hierarchical object container
    ECV_POINT_CLOUD,       ///< Point cloud entity
    ECV_MARK_POINT,        ///< Marker point
    ECV_MESH,              ///< Mesh/polygon mesh entity
    ECV_SHAPE,             ///< Generic shape
    ECV_OCTREE,            ///< Octree structure
    ECV_KDTREE,            ///< KD-tree structure
    ECV_FACET,             ///< Planar facet
    ECV_LINES_3D,          ///< 3D lines
    ECV_LINES_2D,          ///< 2D lines
    ECV_TRIANGLE_2D,       ///< 2D triangle
    ECV_RECTANGLE_2D,      ///< 2D rectangle
    ECV_POLYLINE_2D,       ///< 2D polyline
    ECV_CIRCLE_2D,         ///< 2D circle
    ECV_POLYGON,           ///< Polygon
    ECV_2DLABLE,           ///< 2D label
    ECV_2DLABLE_VIEWPORT,  ///< Viewport-anchored 2D label
    ECV_CAPTION,           ///< Caption text
    ECV_SCALAR_BAR,        ///< Scalar/color bar
    ECV_TEXT3D,            ///< 3D text
    ECV_TEXT2D,            ///< 2D text
    ECV_IMAGE,             ///< Image entity
    ECV_SENSOR,            ///< Sensor entity
    ECV_ALL,               ///< All entities
    ECV_NONE,              ///< No entity
};

/**
 * @brief Widget type enumeration
 *
 * Defines all types of widgets (visual aids) that can be displayed.
 */
enum WIDGETS_TYPE {
    WIDGET_BBOX,          ///< Bounding box widget
    WIDGET_IMAGE,         ///< Image widget
    WIDGET_LINE_2D,       ///< 2D line widget
    WIDGET_TRIANGLE_2D,   ///< 2D triangle widget
    WIDGET_POINTS_2D,     ///< 2D points widget
    WIDGET_CIRCLE_2D,     ///< 2D circle widget
    WIDGET_RECTANGLE_2D,  ///< 2D rectangle widget
    WIDGET_POLYGONMESH,   ///< Polygon mesh widget
    WIDGET_COORDINATE,    ///< Coordinate system widget
    WIDGET_POLYLINE,      ///< 3D polyline widget
    WIDGET_POLYLINE_2D,   ///< 2D polyline widget
    WIDGET_LINE_3D,       ///< 3D line widget
    WIDGET_SPHERE,        ///< Sphere widget
    WIDGET_CAPTION,       ///< Caption widget
    WIDGET_SCALAR_BAR,    ///< Scalar bar widget
    WIDGET_T3D,           ///< 3D text widget
    WIDGET_T2D,           ///< 2D text widget
};

/**
 * @brief Property parameters for entity rendering
 *
 * This structure encapsulates various rendering properties that can be
 * applied to a drawable entity, such as color, opacity, point size, etc.
 */
struct CV_DB_LIB_API PROPERTY_PARAM {
    QString viewId;                     ///< View identifier
    int viewport = 0;                   ///< Viewport index
    PointCoordinateType lineWidth = 2;  ///< Line width for rendering
    unsigned char pointSize = 1;        ///< Point size for rendering
    ccHObject* entity;                  ///< Target entity
    ecvColor::Rgb color;                ///< Color property

    double opacity = 1.0;      ///< Opacity value [0, 1]
    PROPERTY_MODE property;    ///< Property mode
    SHADING_MODE shadingMode;  ///< Shading mode
    ENTITY_TYPE entityType;    ///< Entity type

    /**
     * @brief Constructor for color property
     * @param obj Target entity
     * @param col Color value
     */
    PROPERTY_PARAM(ccHObject* obj, const ecvColor::Rgb& col)
        : entity(obj),
          color(col),
          property(PROPERTY_MODE::ECV_COLOR_PROPERTY) {}

    /**
     * @brief Constructor for opacity property
     * @param obj Target entity
     * @param opacity Opacity value
     */
    PROPERTY_PARAM(ccHObject* obj, double opacity)
        : entity(obj),
          opacity(opacity),
          property(PROPERTY_MODE::ECV_OPACITY_PROPERTY) {}

    /**
     * @brief Constructor for point size property
     * @param obj Target entity
     * @param pointSize Point size value (minimum 1)
     */
    PROPERTY_PARAM(ccHObject* obj, unsigned char pointSize)
        : entity(obj), property(PROPERTY_MODE::ECV_POINTSSIZE_PROPERTY) {
        if (pointSize > 1) {
            this->pointSize = pointSize;
        }
    }

    /**
     * @brief Constructor for shading mode property
     * @param obj Target entity
     * @param shadingMode Shading mode
     */
    PROPERTY_PARAM(ccHObject* obj, SHADING_MODE shadingMode)
        : entity(obj), property(PROPERTY_MODE::ECV_SHADING_PROPERTY) {
        this->shadingMode = shadingMode;
    }

    /**
     * @brief Set the property mode
     * @param mode Property mode to set
     */
    inline void setProperty(PROPERTY_MODE mode) { property = mode; }

    /**
     * @brief Set the shading mode
     * @param shadingMode Shading mode to set
     */
    inline void setShadingMode(SHADING_MODE shadingMode) {
        this->shadingMode = shadingMode;
    }

    /**
     * @brief Set the color
     * @param col Color to set
     */
    inline void setColor(const ecvColor::Rgb& col) {
        color = const_cast<ecvColor::Rgb&>(col);
    }

    /**
     * @brief Set the point size
     * @param size Point size to set
     */
    inline void setPointSize(unsigned char size) { pointSize = size; }

    /**
     * @brief Set the opacity
     * @param op Opacity value [0, 1]
     */
    inline void setOpacity(double op) { opacity = op; }

    /**
     * @brief Set the line width
     * @param with Line width to set
     */
    inline void setLineWith(PointCoordinateType with) { lineWidth = with; }
};

/**
 * @brief Line widget structure
 *
 * Represents a 3D line widget with start/end points, width, and color.
 */
struct CV_DB_LIB_API LineWidget {
    CCVector3 lineSt;         ///< Line start point
    CCVector3 lineEd;         ///< Line end point
    float lineWidth;          ///< Line width
    ecvColor::Rgb lineColor;  ///< Line color
    bool valid;               ///< Whether the line widget is valid

    /**
     * @brief Default constructor
     * @param width Line width (default: 2.0)
     * @param color Line color (default: red)
     */
    LineWidget(float width = 2.0f, const ecvColor::Rgb& color = ecvColor::red)
        : lineWidth(width), lineColor(color), valid(false) {}

    /**
     * @brief Constructor with endpoints
     * @param st Start point
     * @param ed End point
     * @param width Line width (default: 2)
     * @param color Line color (default: red)
     */
    LineWidget(const CCVector3& st,
               const CCVector3& ed,
               float width = 2,
               const ecvColor::Rgb& color = ecvColor::red)
        : lineSt(st),
          lineEd(ed),
          lineWidth(width),
          lineColor(color),
          valid(true) {}
};

/**
 * @brief Information structure for entity removal
 *
 * Stores information about entities to be removed from the view.
 */
struct CV_DB_LIB_API removeInfo {
    ENTITY_TYPE removeType;  ///< Type of entity to remove
    QString removeId;        ///< View ID of entity to remove

    /**
     * @brief Equality operator
     * @param first First removeInfo
     * @param second Second removeInfo
     * @return true if both structures are equal
     */
    friend bool operator==(const removeInfo& first, const removeInfo& second) {
        return (first.removeType == second.removeType &&
                first.removeId == second.removeId)
                       ? true
                       : false;
    }
};

/**
 * @brief Information structure for entity visibility
 *
 * Stores information about entities to be hidden in the view.
 */
struct CV_DB_LIB_API hideInfo {
    ENTITY_TYPE hideType;  ///< Type of entity to hide
    QString hideId;        ///< View ID of entity to hide

    /**
     * @brief Equality operator
     * @param first First hideInfo
     * @param second Second hideInfo
     * @return true if both structures are equal
     */
    friend bool operator==(const hideInfo& first, const hideInfo& second) {
        return (first.hideType == second.hideType &&
                first.hideId == second.hideId)
                       ? true
                       : false;
    }
};

/**
 * @brief Transformation information structure
 *
 * Encapsulates all transformation parameters including rotation, scale,
 * and translation for entity manipulation.
 */
struct CV_DB_LIB_API TransformInfo {
    /**
     * @brief Rotation parameters using angle-axis representation
     */
    struct RotateParam {
        double angle;       ///< Rotation angle in degrees
        CCVector3 rotAxis;  ///< Rotation axis
    };

    RotateParam rotateParam;         ///< Angle-axis rotation parameters
    bool isRotate = false;           ///< Whether rotation is applied
    bool applyEuler = true;          ///< Whether to use Euler angles
    bool isScale = false;            ///< Whether scale is applied
    bool isTranslate = false;        ///< Whether translation is applied
    bool isPositionChanged = false;  ///< Whether position has changed

    double quaternion[4];  ///< Quaternion representation

    CCVector3d eulerZYX;  ///< Euler angles in ZYX order (degrees)

    CCVector3 scaleXYZ;  ///< Scale factors for X, Y, Z axes

    CCVector3 transVecStart;  ///< Translation start vector
    CCVector3 transVecEnd;    ///< Translation end vector

    CCVector3 position;  ///< Position vector

    /**
     * @brief Set the position
     * @param pos New position vector
     */
    void setPostion(const CCVector3& pos) {
        position = pos;
        isPositionChanged = true;
    }

    /**
     * @brief Check if any transformation is applied
     * @return true if rotation, scale, translation, or position is applied
     */
    bool isApplyTransform() const {
        return isRotate || isScale || isTranslate || isPositionChanged;
    }

    /**
     * @brief Set transformation from a matrix
     * @param transform Transformation matrix
     * @param updateTranslation Whether to update translation (default: true)
     * @param useEuler Whether to use Euler angles (default: false)
     */
    void setTransformation(const ccGLMatrixd& transform,
                           bool updateTranslation = true,
                           bool useEuler = false) {
        if (useEuler) {
            //            double phi_rad, theta_rad, psi_rad;
            //            CCVector3d t3D;
            //            transform.getParameters(phi_rad, theta_rad, psi_rad,
            //            t3D);
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
            if (updateTranslation) {
                setTranslationStart(CCVector3::fromArray(
                        transform.getTranslationAsVec3D().u));
            }

        } else {
            double angle_rad;
            CCVector3d axis, trans;
            transform.getParameters(angle_rad, axis, trans);
            double angle_deg = cloudViewer::RadiansToDegrees(angle_rad);
            setRotation(angle_deg, axis);
            if (updateTranslation) {
                setTranslationStart(CCVector3::fromArray(trans.u));
            }
        }
    }
    /**
     * @brief Set transformation from a matrix (float version)
     * @param transform Transformation matrix
     * @param updateTranslation Whether to update translation (default: true)
     * @param useEuler Whether to use Euler angles (default: true)
     */
    void setTransformation(const ccGLMatrix& transform,
                           bool updateTranslation = true,
                           bool useEuler = true) {
        setTransformation(ccGLMatrixd(transform.data()), updateTranslation,
                          useEuler);
    }

    /**
     * @brief Set rotation using Euler angles (ZYX order)
     * @param zAngle Rotation around Z axis (degrees)
     * @param yAngle Rotation around Y axis (degrees)
     * @param xAngle Rotation around X axis (degrees)
     */
    void setRotation(double zAngle, double yAngle, double xAngle) {
        isRotate = true;
        applyEuler = true;
        eulerZYX[0] = zAngle;
        eulerZYX[1] = yAngle;
        eulerZYX[2] = xAngle;
    }

    /**
     * @brief Set rotation using Euler angles array
     * @param zyxAngle Array of Euler angles [Z, Y, X] (degrees)
     */
    void setRotation(double zyxAngle[3]) {
        setRotation(zyxAngle[0], zyxAngle[1], zyxAngle[2]);
    }

    /**
     * @brief Set rotation using angle-axis representation
     * @param angle Rotation angle (degrees)
     * @param x X component of rotation axis
     * @param y Y component of rotation axis
     * @param z Z component of rotation axis
     */
    void setRotation(double angle, double x, double y, double z) {
        setRotation(angle, CCVector3d(x, y, z));
    }

    /**
     * @brief Set rotation using angle-axis representation
     * @param angle Rotation angle (degrees)
     * @param axis Rotation axis
     */
    void setRotation(double angle, const CCVector3& axis) {
        isRotate = true;
        applyEuler = false;
        rotateParam.angle = angle;
        rotateParam.rotAxis = axis;
    }

    /**
     * @brief Set rotation using angle-axis representation (double precision)
     * @param angle Rotation angle (degrees)
     * @param axis Rotation axis
     */
    void setRotation(double angle, const CCVector3d& axis) {
        isRotate = true;
        applyEuler = false;
        rotateParam.angle = angle;
        rotateParam.rotAxis = CCVector3::fromArray(axis.u);
    }

    /**
     * @brief Set scale factors
     * @param scale Scale vector for X, Y, Z axes
     */
    void setScale(const CCVector3& scale) {
        isScale = true;
        scaleXYZ = scale;
    }

    /**
     * @brief Set translation start vector
     * @param trans Translation start vector
     */
    void setTranslationStart(const CCVector3& trans) {
        isTranslate = true;
        transVecStart = trans;
    }

    /**
     * @brief Set translation end vector
     * @param trans Translation end vector
     */
    void setTranslationEnd(const CCVector3& trans) {
        isTranslate = true;
        transVecEnd = trans;
    }
};

/**
 * @brief Text rendering parameters
 *
 * Parameters for rendering text in 2D or 3D space.
 */
struct CV_DB_LIB_API ecvTextParam {
    bool display3D = false;                          ///< Display as 3D text
    double textScale = 1.0;                          ///< Text scale factor
    double opacity = 1.0;                            ///< Text opacity [0, 1]
    CCVector3d textPos = CCVector3d(0.0, 0.0, 0.0);  ///< Text position
    QString text = "";                               ///< Text content
    QFont font = QFont();                            ///< Font settings
};

/**
 * @brief Drawing flags for controlling render behavior
 *
 * Bitflags that control various aspects of rendering such as 2D/3D mode,
 * entity picking, LOD, etc.
 */
enum CC_DRAWING_FLAGS {
    CC_DRAW_2D = 0x0001,          ///< Draw in 2D mode
    CC_DRAW_3D = 0x0002,          ///< Draw in 3D mode
    CC_DRAW_FOREGROUND = 0x0004,  ///< Draw in foreground
    CC_LIGHT_ENABLED = 0x0008,    ///< Enable lighting
    CC_SKIP_UNSELECTED = 0x0010,  ///< Skip unselected entities
    CC_SKIP_SELECTED = 0x0020,    ///< Skip selected entities
    CC_SKIP_ALL = 0x0030,         ///< Skip all entities (UNSELECTED | SELECTED)
    CC_ENTITY_PICKING = 0x0040,   ///< Enable entity picking mode
    // CC_FREE_FLAG	  = 0x0080,		// UNUSED (formerly
    // CC_DRAW_POINT_NAMES) CC_FREE_FLAG	  = 0x0100,		//
    // UNUSED (formerly CC_DRAW_TRI_NAMES)
    CC_FAST_ENTITY_PICKING = 0x0200,  ///< Fast entity picking mode
    // UNUSED (formerly CC_DRAW_ANY_NAMES = CC_DRAW_ENTITY_NAMES |
    // CC_DRAW_POINT_NAMES | CC_DRAW_TRI_NAMES) CC_FREE_FLAG	  = 0x03C0,
    CC_LOD_ACTIVATED = 0x0400,         ///< Level of detail (LOD) activated
    CC_VIRTUAL_TRANS_ENABLED = 0x0800  ///< Virtual transformation enabled
};

// Drawing flags testing macros (see ccDrawableObject)
#define MACRO_Draw2D(context) (context.drawingFlags & CC_DRAW_2D)
#define MACRO_Draw3D(context) (context.drawingFlags & CC_DRAW_3D)
#define MACRO_EntityPicking(context) (context.drawingFlags & CC_ENTITY_PICKING)
#define MACRO_FastEntityPicking(context) \
    (context.drawingFlags & CC_FAST_ENTITY_PICKING)
#define MACRO_SkipUnselected(context) \
    (context.drawingFlags & CC_SKIP_UNSELECTED)
#define MACRO_SkipSelected(context) (context.drawingFlags & CC_SKIP_SELECTED)
#define MACRO_LightIsEnabled(context) (context.drawingFlags & CC_LIGHT_ENABLED)
#define MACRO_Foreground(context) (context.drawingFlags & CC_DRAW_FOREGROUND)
#define MACRO_LODActivated(context) (context.drawingFlags & CC_LOD_ACTIVATED)
#define MACRO_VirtualTransEnabled(context) \
    (context.drawingFlags & CC_VIRTUAL_TRANS_ENABLED)

/**
 * @brief OpenGL drawing context
 *
 * Comprehensive structure containing all parameters needed for rendering
 * entities in the 3D/2D views, including colors, materials, flags, LOD
 * settings, and transformation information.
 */
struct ccGLDrawContext {
    int drawingFlags;             ///< Drawing flags (see CC_DRAWING_FLAGS)
    bool forceRedraw;             ///< Force redraw
    bool visFiltering;            ///< Visibility filtering enabled
    TransformInfo transformInfo;  ///< Transformation information
    ecvTextParam textParam;       ///< Text rendering parameters

    QString viewID;       ///< View identifier
    int defaultViewPort;  ///< Default viewport index
    int normalDensity;    ///< Normal vector display density (%)
    float normalScale;    ///< Normal vector scale factor

    float opacity;  ///< Global opacity

    bool visible;                           ///< Whether entities are visible
    unsigned char defaultLineWidth;         ///< Default line width
    unsigned char currentLineWidth;         ///< Current line width
    unsigned char defaultPointSize;         ///< Default point size
    glDrawParams drawParam;                 ///< Draw parameters
    MESH_RENDERING_MODE meshRenderingMode;  ///< Mesh rendering mode
    ENTITY_TYPE hideShowEntityType;         ///< Entity type for hide/show

    QString removeViewID;          ///< View ID for entity removal
    ENTITY_TYPE removeEntityType;  ///< Entity type for removal

    bool clearDepthLayer;  ///< Clear depth buffer
    bool clearColorLayer;  ///< Clear color buffer

    int glW;                 ///< OpenGL screen width
    int glH;                 ///< OpenGL screen height
    float devicePixelRatio;  ///< Device pixel ratio (1 or 2 for HD)

    float renderZoom;  ///< Render zoom factor

    ccMaterial::Shared defaultMat;  ///< Default material

    ecvColor::Rgbaf defaultMeshFrontDiff;  ///< Default mesh color (front)
    ecvColor::Rgbaf defaultMeshBackDiff;   ///< Default mesh color (back)
    ecvColor::Rgb defaultMeshColor;        ///< Default mesh color

    ecvColor::Rgb defaultPolylineColor;  ///< Default polyline color

    ecvColor::Rgbub pointsDefaultCol;    ///< Default point color
    ecvColor::Rgbub pointsCurrentCol;    ///< Current point color
    ecvColor::Rgbub textDefaultCol;      ///< Default text color
    ecvColor::Rgbub labelDefaultBkgCol;  ///< Default label background color

    ecvColor::Rgbf viewDefaultBkgCol;  ///< Default view background color

    ecvColor::Rgbub labelDefaultMarkerCol;  ///< Default label marker color
    ecvColor::Rgbub bbDefaultCol;           ///< Default bounding box color

    ecvColor::Rgbub backgroundCol;   ///< Background color (primary)
    ecvColor::Rgbub backgroundCol2;  ///< Background color (secondary)
    bool drawBackgroundGradient;     ///< Draw background gradient

    bool decimateCloudOnMove;       ///< Decimate clouds during camera movement
    unsigned minLODPointCount;      ///< Min points for LOD activation
    unsigned char currentLODLevel;  ///< Current LOD level
    bool moreLODPointsAvailable;    ///< More LOD points available
    bool higherLODLevelsAvailable;  ///< Higher LOD levels available

    bool decimateMeshOnMove;       ///< Decimate meshes during camera movement
    unsigned minLODTriangleCount;  ///< Min triangles for LOD activation

    ccScalarField*
            sfColorScaleToDisplay;  ///< Scalar field color scale to display

    bool useVBOs;  ///< Use VBOs for rendering

    float labelMarkerSize;           ///< Label marker size (radius)
    float labelMarkerTextShift_pix;  ///< Label marker text shift (pixels)

    unsigned dispNumberPrecision;  ///< Numerical display precision

    unsigned labelOpacity;  ///< Label background opacity

    unsigned stereoPassIndex;  ///< Stereo rendering pass index

    bool drawRoundedPoints;  ///< Draw rounded points

    /**
     * @brief Default constructor
     *
     * Initializes all drawing context parameters to their default values.
     */
    ccGLDrawContext()
        : drawingFlags(0),
          forceRedraw(true),
          visFiltering(false),
          viewID("unnamed"),
          defaultViewPort(0),
          normalDensity(100),
          normalScale(0.02f),
          opacity(1.0),
          visible(true),
          defaultLineWidth(2),
          currentLineWidth(defaultLineWidth),
          defaultPointSize(1),
          meshRenderingMode(MESH_RENDERING_MODE::ECV_SURFACE_MODE),
          removeViewID("unnamed"),
          removeEntityType(ENTITY_TYPE::ECV_POINT_CLOUD),
          clearDepthLayer(true),
          clearColorLayer(true),
          glW(0),
          glH(0),
          devicePixelRatio(1.0f),
          renderZoom(1.0f),
          defaultMat(new ccMaterial("default")),
          defaultMeshFrontDiff(ecvColor::defaultMeshFrontDiff),
          defaultMeshBackDiff(ecvColor::defaultMeshBackDiff),
          defaultMeshColor(ecvColor::lightGrey),
          defaultPolylineColor(ecvColor::green),
          pointsDefaultCol(ecvColor::defaultColor),
          pointsCurrentCol(ecvColor::defaultColor),
          textDefaultCol(ecvColor::defaultColor),
          labelDefaultBkgCol(ecvColor::defaultLabelBkgColor),
          viewDefaultBkgCol(ecvColor::defaultViewBkgColor),
          labelDefaultMarkerCol(ecvColor::defaultLabelMarkerColor),
          bbDefaultCol(ecvColor::yellow),
          backgroundCol(ecvColor::defaultBkgColor),
          backgroundCol2(ecvColor::defaultLabelBkgColor),
          drawBackgroundGradient(true),
          decimateCloudOnMove(true),
          minLODPointCount(10000000),
          currentLODLevel(0),
          moreLODPointsAvailable(false),
          higherLODLevelsAvailable(false),
          decimateMeshOnMove(true),
          minLODTriangleCount(2500000),
          sfColorScaleToDisplay(nullptr),
          useVBOs(true),
          labelMarkerSize(5),
          labelMarkerTextShift_pix(5),
          dispNumberPrecision(6),
          labelOpacity(100),
          stereoPassIndex(0),
          drawRoundedPoints(false) {}
};

/**
 * @brief Type alias for ccGLDrawContext
 */
using CC_DRAW_CONTEXT = ccGLDrawContext;

/**
 * @brief Widget parameters structure
 *
 * Comprehensive parameters for rendering various types of widgets
 * including images, text, shapes, and 3D objects.
 */
struct CV_DB_LIB_API WIDGETS_PARAMETER {
public:
    // General parameters
    ccHObject* entity;      ///< Associated entity
    WIDGETS_TYPE type;      ///< Widget type
    QString viewID;         ///< View identifier
    int viewport = 0;       ///< Viewport index
    ecvColor::Rgbaf color;  ///< Widget color

    CC_DRAW_CONTEXT context;  ///< Drawing context

    int fontSize = 10;  ///< Font size for text widgets

    // Image widget parameters
    QImage image;          ///< Image data
    double opacity = 1.0;  ///< Image opacity

    // Text widget parameters
    QString text;  ///< Text content

    // Rectangle widget parameters
    bool filled = true;  ///< Whether rectangle is filled
    QRect rect;          ///< Rectangle dimensions

    // 2D shape parameters (line, triangle, quad)
    QPoint p1;                   ///< First point
    QPoint p2;                   ///< Second point
    QPoint p3;                   ///< Third point
    QPoint p4 = QPoint(-1, -1);  ///< Fourth point (optional)

    // Circle/sphere parameters
    float radius;                ///< Radius
    CCVector3 center;            ///< 3D center
    CCVector2 pos;               ///< 2D position
    bool handleEnabled = false;  ///< Enable interactive handle

    // 3D line parameters
    LineWidget lineWidget;  ///< Line widget data

    /**
     * @brief Constructor with widget type
     * @param t Widget type
     * @param id View ID (default: "id")
     * @param port Viewport index (default: 0)
     */
    WIDGETS_PARAMETER(WIDGETS_TYPE t, QString id = "id", int port = 0)
        : type(t), viewID(id), viewport(port) {
        context.viewID = viewID;
    }

    /**
     * @brief Constructor with entity and widget type
     * @param obj Associated entity
     * @param t Widget type
     * @param id View ID (default: "id")
     * @param port Viewport index (default: 0)
     */
    WIDGETS_PARAMETER(ccHObject* obj,
                      WIDGETS_TYPE t,
                      QString id = "id",
                      int port = 0)
        : entity(obj), type(t), viewID(id), viewport(port) {
        context.viewID = viewID;
    }

    /**
     * @brief Set line widget parameters
     * @param line Line widget data
     */
    void setLineWidget(const LineWidget& line) { lineWidget = line; }
};
