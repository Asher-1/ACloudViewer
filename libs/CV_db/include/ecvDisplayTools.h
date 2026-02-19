// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvGLMatrix.h"
#include "ecvGenericDisplayTools.h"
#include "ecvGuiParameters.h"
#include "ecvHObject.h"
#include "ecvViewportParameters.h"

// QT
#include <QApplication>  // Added for QApplication::primaryScreen()
#include <QElapsedTimer>
#include <QMainWindow>
#include <QObject>
#include <QRect>
#include <QScreen>  // Added for QScreen
#include <QTimer>

// System
#include <list>
#include <unordered_set>
#include <vector>

namespace cloudViewer {
namespace geometry {
class LineSet;
}
}  // namespace cloudViewer

class ccPolyline;
class ccInteractor;
class ccGenericMesh;
class ecvOrientedBBox;

class ecvGenericVisualizer;
class ecvGenericVisualizer2D;
class ecvGenericVisualizer3D;

// ============================================================================
// Axes Grid Properties Structure (ParaView-compatible)
// ============================================================================

/**
 * @struct AxesGridProperties
 * @brief Data Axes Grid properties structure (ParaView-compatible)
 * 
 * Encapsulates all properties for vtkCubeAxesActor configuration,
 * providing ParaView-style axis grid visualization with customizable
 * labels, bounds, and appearance.
 */
struct CV_DB_LIB_API AxesGridProperties {
    // Basic properties
    bool visible = false;                            ///< Axes grid visibility
    CCVector3 color = CCVector3(255, 255, 255);      ///< Color (RGB 0-255, default: white)
    double lineWidth = 1.0;                          ///< Line width in pixels
    double spacing = 1.0;                            ///< Grid spacing
    int subdivisions = 10;                           ///< Number of subdivisions
    bool showLabels = true;                          ///< Show axis labels
    double opacity = 1.0;                            ///< Opacity (0.0-1.0)

    // Extended properties (ParaView-style)
    bool showGrid = false;                           ///< Show grid lines (default: OFF)
    QString xTitle = "X-Axis";                       ///< X-axis title
    QString yTitle = "Y-Axis";                       ///< Y-axis title
    QString zTitle = "Z-Axis";                       ///< Z-axis title
    bool xUseCustomLabels = false;                   ///< Use custom X labels
    bool yUseCustomLabels = false;                   ///< Use custom Y labels
    bool zUseCustomLabels = false;                   ///< Use custom Z labels
    bool useCustomBounds = false;                    ///< Use custom axis bounds

    // Custom labels (ParaView-style: value -> label string)
    QList<QPair<double, QString>> xCustomLabels;     ///< X-axis custom labels
    QList<QPair<double, QString>> yCustomLabels;     ///< Y-axis custom labels
    QList<QPair<double, QString>> zCustomLabels;     ///< Z-axis custom labels

    // Custom bounds (ParaView-style: explicit min/max for each axis)
    double xMin = 0.0, xMax = 1.0;                   ///< X-axis bounds
    double yMin = 0.0, yMax = 1.0;                   ///< Y-axis bounds
    double zMin = 0.0, zMax = 1.0;                   ///< Z-axis bounds

    /**
     * @brief Default constructor
     */
    AxesGridProperties() = default;
};

/**
 * @class ecvDisplayTools
 * @brief Main display and rendering management class
 * 
 * Central singleton class for managing all visualization, rendering, and user
 * interaction in CloudViewer. Provides a comprehensive interface for:
 * 
 * - 3D/2D rendering and display control
 * - Camera manipulation (position, orientation, projection)
 * - Entity picking (point, triangle, object selection)
 * - Mouse/keyboard interaction handling
 * - Viewport management
 * - Visual overlays (axes, grids, labels, widgets)
 * - Screenshot and rendering to file
 * - Perspective/orthographic projection switching
 * 
 * This class follows the singleton pattern and integrates with both Qt (for UI)
 * and VTK/OpenGL (for 3D rendering).
 * 
 * @see ecvGenericDisplayTools
 * @see ecvGenericVisualizer3D
 * @see ecvGenericVisualizer2D
 */
class CV_DB_LIB_API ecvDisplayTools : public QObject,
                                      public ecvGenericDisplayTools {
    Q_OBJECT
public:
    /**
     * @brief Initialize the display tools singleton
     * @param displayTools Display tools instance
     * @param win Main window widget
     * @param stereoMode Enable stereo rendering mode (default: false)
     */
    static void Init(ecvDisplayTools* displayTools,
                     QMainWindow* win,
                     bool stereoMode = false);
    
    /**
     * @brief Get the singleton instance
     * @return Pointer to the singleton instance
     */
    static ecvDisplayTools* TheInstance();

    /**
     * @brief Release and destroy the singleton instance
     */
    static void ReleaseInstance();

    /**
     * @brief Virtual destructor
     */
    virtual ~ecvDisplayTools() override;

    /**
     * @brief Schedule a full redraw
     * 
     * Schedules a complete redraw (no LOD) after a specified delay.
     * Any previously scheduled redraw will be cancelled.
     * @param maxDelay_ms Maximum delay before redraw (milliseconds)
     * @warning Cancelled if redraw/update is called before delay expires
     */
    void scheduleFullRedraw(unsigned maxDelay_ms);

    /**
     * @brief Cancel any scheduled redraw
     */
    void cancelScheduledRedraw();

public:
    /**
     * @brief Picking mode enumeration
     * 
     * Defines the type of picking operation for user selection.
     */
    enum PICKING_MODE {
        NO_PICKING,                          ///< No picking enabled
        ENTITY_PICKING,                      ///< Pick entire entities
        ENTITY_RECT_PICKING,                 ///< Rectangular entity selection
        FAST_PICKING,                        ///< Fast picking (optimized)
        POINT_PICKING,                       ///< Pick individual points
        TRIANGLE_PICKING,                    ///< Pick mesh triangles
        POINT_OR_TRIANGLE_PICKING,           ///< Pick points or triangles
        POINT_OR_TRIANGLE_OR_LABEL_PICKING,  ///< Pick points, triangles, or labels
        LABEL_PICKING,                       ///< Pick labels only
        DEFAULT_PICKING,                     ///< Default picking mode
    };

    /**
     * @brief Interaction flags for mouse/keyboard handling
     * 
     * Bitflags defining enabled interaction types and signal emissions.
     * Can be combined to create custom interaction modes.
     */
    enum INTERACTION_FLAG {
        // No interaction
        INTERACT_NONE = 0,                      ///< No interactions enabled

        // Camera interactions
        INTERACT_ROTATE = 1,                    ///< Enable camera rotation
        INTERACT_PAN = 2,                       ///< Enable camera panning
        INTERACT_CTRL_PAN = 4,                  ///< Enable Ctrl+pan
        INTERACT_ZOOM_CAMERA = 8,               ///< Enable camera zoom
        INTERACT_2D_ITEMS = 16,                 ///< Enable 2D item interaction (labels, etc.)
        INTERACT_CLICKABLE_ITEMS = 32,          ///< Enable hot zone interaction

        // Options / modifiers
        INTERACT_TRANSFORM_ENTITIES = 64,       ///< Enable entity transformation

        // Signals
        INTERACT_SIG_RB_CLICKED = 128,          ///< Emit right button clicked signal
        INTERACT_SIG_LB_CLICKED = 256,          ///< Emit left button clicked signal
        INTERACT_SIG_MOUSE_MOVED = 512,         ///< Emit mouse moved signal (when button pressed)
        INTERACT_SIG_BUTTON_RELEASED = 1024,    ///< Emit button released signal
        INTERACT_SIG_MB_CLICKED = 2048,         ///< Emit middle button clicked signal
        INTERACT_SEND_ALL_SIGNALS =             ///< Emit all signals
                INTERACT_SIG_RB_CLICKED | INTERACT_SIG_LB_CLICKED |
                INTERACT_SIG_MB_CLICKED | INTERACT_SIG_MOUSE_MOVED |
                INTERACT_SIG_BUTTON_RELEASED,

        // Default interaction modes
        MODE_PAN_ONLY =                         ///< Pan and zoom only
                INTERACT_PAN | INTERACT_ZOOM_CAMERA |
                INTERACT_2D_ITEMS | INTERACT_CLICKABLE_ITEMS,
        MODE_TRANSFORM_CAMERA =                 ///< Camera transformation mode
                INTERACT_ROTATE | MODE_PAN_ONLY,
        MODE_TRANSFORM_ENTITIES =               ///< Entity transformation mode
                INTERACT_ROTATE | INTERACT_PAN | INTERACT_ZOOM_CAMERA |
                INTERACT_TRANSFORM_ENTITIES | INTERACT_CLICKABLE_ITEMS,
    };
    Q_DECLARE_FLAGS(INTERACTION_FLAGS, INTERACTION_FLAG)

    /**
     * @brief Get pan-only interaction mode
     * @return Pan-only interaction flags
     */
    static INTERACTION_FLAGS PAN_ONLY();
    
    /**
     * @brief Get camera transformation mode
     * @return Camera transformation interaction flags
     */
    static INTERACTION_FLAGS TRANSFORM_CAMERA();
    
    /**
     * @brief Get entity transformation mode
     * @return Entity transformation interaction flags
     */
    static INTERACTION_FLAGS TRANSFORM_ENTITIES();

    /**
     * @brief Message display positions on screen
     */
    enum MessagePosition {
        LOWER_LEFT_MESSAGE,      ///< Lower-left corner
        UPPER_CENTER_MESSAGE,    ///< Upper-center
        SCREEN_CENTER_MESSAGE,   ///< Screen center
    };

    /**
     * @brief Message type enumeration
     * 
     * Defines the type of on-screen message, allowing only one
     * message of each type to be displayed simultaneously.
     */
    enum MessageType {
        CUSTOM_MESSAGE,                   ///< Custom user message
        SCREEN_SIZE_MESSAGE,              ///< Screen size info
        PERSPECTIVE_STATE_MESSAGE,        ///< Perspective mode state
        SUN_LIGHT_STATE_MESSAGE,          ///< Sun light state
        CUSTOM_LIGHT_STATE_MESSAGE,       ///< Custom light state
        MANUAL_TRANSFORMATION_MESSAGE,    ///< Manual transformation mode
        MANUAL_SEGMENTATION_MESSAGE,      ///< Manual segmentation mode
        ROTAION_LOCK_MESSAGE,             ///< Rotation lock state
        FULL_SCREEN_MESSAGE,              ///< Full screen mode state
    };

    /**
     * @brief Pivot symbol visibility modes
     */
    enum PivotVisibility {
        PIVOT_HIDE,              ///< Always hide pivot
        PIVOT_SHOW_ON_MOVE,      ///< Show pivot only during camera movement
        PIVOT_ALWAYS_SHOW,       ///< Always show pivot
    };

    /**
     * @struct MessageToDisplay
     * @brief Temporary on-screen message descriptor
     * 
     * Contains all information needed to display a temporary message
     * on screen, including content, position, validity time, and type.
     */
    struct CV_DB_LIB_API MessageToDisplay {
        /**
         * @brief Default constructor
         */
        MessageToDisplay()
            : messageValidity_sec(0),
              position(LOWER_LEFT_MESSAGE),
              type(CUSTOM_MESSAGE) {}

        QString message;                    ///< Message text
        qint64 messageValidity_sec;         ///< Message expiration time (seconds)
        MessagePosition position;           ///< Display position on screen
        MessageType type;                   ///< Message type
    };

    /**
     * @brief List of currently displayed messages
     */
    std::list<MessageToDisplay> m_messagesToDisplay;

    /**
     * @struct ProjectionMetrics
     * @brief Optional output metrics from projection matrix computation
     * 
     * Contains computed camera and scene metrics useful for advanced
     * rendering operations and debugging.
     */
    struct CV_DB_LIB_API ProjectionMetrics {
        /**
         * @brief Default constructor
         */
        ProjectionMetrics()
            : zNear(0.0),
              zFar(0.0),
              cameraToBBCenterDist(0.0),
              bbHalfDiag(0.0) {}

        double zNear;                       ///< Near clipping plane distance
        double zFar;                        ///< Far clipping plane distance
        double cameraToBBCenterDist;        ///< Camera to bounding box center distance
        double bbHalfDiag;                  ///< Half diagonal of bounding box
    };

    /**
     * @struct HotZone
     * @brief Hot zone (interactive UI overlay) properties
     * 
     * Manages the "hot zone" - an interactive overlay in the upper-right corner
     * displaying clickable controls for point size, line width, bubble-view mode,
     * and fullscreen mode. Includes pre-computed layout metrics for efficient rendering.
     */
    struct CV_DB_LIB_API HotZone {
        // display font
        QFont font;
        // text height
        int textHeight;
        // text shift
        int yTextBottomLineShift;
        // default color
        unsigned char color[3];

        // bubble-view label rect.
        QString bbv_label;
        // bubble-view label rect.
        QRect bbv_labelRect;
        // bubble-view row width
        int bbv_totalWidth;

        // fullscreen label rect.
        QString fs_label;
        // fullscreen label rect.
        QRect fs_labelRect;
        // fullscreen row width
        int fs_totalWidth;

        // point size label
        QString psi_label;
        // point size label rect.
        QRect psi_labelRect;
        // point size row width
        int psi_totalWidth;

        // line size label
        QString lsi_label;
        // line size label rect.
        QRect lsi_labelRect;
        // line size row width
        int lsi_totalWidth;

        int margin;
        int iconSize;
        QPoint topCorner;
        qreal pixelDeviceRatio;

        explicit HotZone(QWidget* win)
            : textHeight(0),
              yTextBottomLineShift(0),
              bbv_label("bubble-view mode"),
              fs_label("fullscreen mode"),
              psi_label("default point size"),
              lsi_label("default line width"),
              margin(16)  // 16|10
              ,
              iconSize(16)  // 16
              ,
              topCorner(0, 0),
              pixelDeviceRatio(1.0) {
            color[0] = ecvColor::defaultLabelBkgColor.r;
            color[1] = ecvColor::defaultLabelBkgColor.g;
            color[2] = ecvColor::defaultLabelBkgColor.b;

            updateInternalVariables(win);
        }

        void updateInternalVariables(QWidget* win) {
            if (win) {
                font = win->font();
                pixelDeviceRatio = GetPlatformAwareDPIScale();
                int fontSize = GetOptimizedFontSize(12);
                if (fontSize != pixelDeviceRatio) {
                    font.setPointSize(fontSize);
                } else {
                    font.setPointSize(12 * pixelDeviceRatio);
                }
                CVLog::Print(QString("pixelDeviceRatio: %1 and fontSize %2")
                                     .arg(pixelDeviceRatio)
                                     .arg(fontSize));
                margin *= pixelDeviceRatio;
                iconSize *= pixelDeviceRatio;
                font.setBold(true);
            }

            QFontMetrics metrics(font);
            bbv_labelRect = metrics.boundingRect(bbv_label);
            fs_labelRect = metrics.boundingRect(fs_label);
            psi_labelRect = metrics.boundingRect(psi_label);
            lsi_labelRect = metrics.boundingRect(lsi_label);

            psi_totalWidth = /*margin() + */ psi_labelRect.width() + margin +
                             iconSize + margin + iconSize /* + margin*/;
            lsi_totalWidth = /*margin() + */ lsi_labelRect.width() + margin +
                             iconSize + margin + iconSize /* + margin*/;
            bbv_totalWidth = /*margin() + */ bbv_labelRect.width() + margin +
                             iconSize /* + margin*/;
            fs_totalWidth = /*margin() + */ fs_labelRect.width() + margin +
                            iconSize /* + margin*/;

            textHeight =
                    std::max(psi_labelRect.height(), bbv_labelRect.height());
            textHeight = std::max(lsi_labelRect.height(), textHeight);
            textHeight = std::max(fs_labelRect.height(), textHeight);
            textHeight = (3 * textHeight) /
                         4;  // --> factor: to recenter the baseline a little
            yTextBottomLineShift = (iconSize / 2) + (textHeight / 2);
        }

        QRect rect(bool clickableItemsVisible,
                   bool bubbleViewModeEnabled,
                   bool fullScreenEnabled) const {
            // total hot zone area size (without margin)
            int totalWidth = 0;
            if (clickableItemsVisible)
                totalWidth = std::max(psi_totalWidth, lsi_totalWidth);
            if (bubbleViewModeEnabled)
                totalWidth = std::max(totalWidth, bbv_totalWidth);
            if (fullScreenEnabled)
                totalWidth = std::max(totalWidth, fs_totalWidth);

#ifdef Q_OS_MAC
            // fix the hot zone width on mac
            totalWidth = totalWidth + 3 * (margin + iconSize);
#endif
            QPoint minAreaCorner(
                    0, std::min(0, yTextBottomLineShift - textHeight));
            QPoint maxAreaCorner(totalWidth,
                                 std::max(iconSize, yTextBottomLineShift));
            int rowCount = clickableItemsVisible ? 2 : 0;
            rowCount += bubbleViewModeEnabled ? 1 : 0;
            rowCount += fullScreenEnabled ? 1 : 0;
            maxAreaCorner.setY(maxAreaCorner.y() +
                               (iconSize + margin) * (rowCount - 1));

            QRect areaRect(minAreaCorner - QPoint(margin, margin) / 2,
                           maxAreaCorner + QPoint(margin, margin) / 2);

            return areaRect;
        }
    };

    /**
     * @brief Display text at 2D screen position
     * 
     * Renders text during 2D pass. Coordinates are relative to viewport
     * with y=0 at the top.
     * 
     * @param text Text string to display
     * @param x Horizontal position (pixels)
     * @param y Vertical position (pixels, 0 = top)
     * @param align Alignment flags (default: ALIGN_DEFAULT)
     * @param bkgAlpha Background transparency (0.0-1.0, default: 0)
     * @param rgbColor Text color RGB (optional, uses default if nullptr)
     * @param font Font to use (optional, uses default if nullptr)
     * @param id Unique identifier for the text (optional)
     * @note Call only during 2D rendering pass
     */
    static void DisplayText(const QString& text,
                            int x,
                            int y,
                            unsigned char align = ALIGN_DEFAULT,
                            float bkgAlpha = 0.0f,
                            const unsigned char* rgbColor = nullptr,
                            const QFont* font = nullptr,
                            const QString& id = "");

    /**
     * @brief Display text with draw context
     * @param CONTEXT Drawing context containing text parameters
     */
    static void DisplayText(const CC_DRAW_CONTEXT& CONTEXT) {
        TheInstance()->displayText(CONTEXT);
    }
    
    /**
     * @brief Virtual interface for displaying text (to be overridden)
     * @param CONTEXT Drawing context
     */
    inline virtual void displayText(
            const CC_DRAW_CONTEXT& CONTEXT) { /* do nothing */ }

    /**
     * @brief Display 3D label at world position
     * 
     * Renders text label at a 3D position during 3D rendering pass.
     * @param str Label text
     * @param pos3D 3D world position
     * @param color Text color (optional, uses default if nullptr)
     * @param font Font to use (optional)
     * @note Call only during 3D rendering pass
     */
    static void Display3DLabel(const QString& str,
                               const CCVector3& pos3D,
                               const ecvColor::Rgbub* color = nullptr,
                               const QFont& font = QFont());

public:  // Main 3D layer drawing methods
    /**
     * @brief Set focus to the screen widget
     */
    static void SetFocusToScreen();
    
    /**
     * @brief Mark display for refresh
     */
    static void ToBeRefreshed();
    
    /**
     * @brief Refresh the display
     * @param only2D Refresh only 2D elements (default: false)
     * @param forceRedraw Force complete redraw (default: true)
     */
    static void RefreshDisplay(bool only2D = false, bool forceRedraw = true);
    
    /**
     * @brief Redraw the display
     * @param only2D Redraw only 2D elements (default: false)
     * @param forceRedraw Force complete redraw (default: true)
     */
    static void RedrawDisplay(bool only2D = false, bool forceRedraw = true);
    
    /**
     * @brief Check if entities should be removed
     */
    static void CheckIfRemove();
    
    /**
     * @brief Draw an object
     * @param context Drawing context
     * @param obj Object to draw
     */
    inline static void Draw(const CC_DRAW_CONTEXT& context,
                            const ccHObject* obj) {
        TheInstance()->draw(context, obj);
    }
    
    /**
     * @brief Virtual draw interface (to be overridden)
     * @param context Drawing context
     * @param obj Object to draw
     */
    inline virtual void draw(const CC_DRAW_CONTEXT& context,
                             const ccHObject* obj) { /* do nothing */ }

    /**
     * @brief Update mesh textures
     * @param context Drawing context
     * @param mesh Mesh with textures to update
     */
    inline static void UpdateMeshTextures(const CC_DRAW_CONTEXT& context,
                                          const ccGenericMesh* mesh) {
        TheInstance()->updateMeshTextures(context, mesh);
    }
    
    /**
     * @brief Virtual interface for updating mesh textures
     * @param context Drawing context
     * @param mesh Mesh to update
     */
    inline virtual void updateMeshTextures(
            const CC_DRAW_CONTEXT& context,
            const ccGenericMesh* mesh) { /* do nothing */ }

    /**
     * @brief Draw axis-aligned bounding box
     * @param context Drawing context
     * @param bbox Bounding box to draw
     */
    inline static void DrawBBox(const CC_DRAW_CONTEXT& context,
                                const ccBBox* bbox) {
        TheInstance()->drawBBox(context, bbox);
    }
    
    /**
     * @brief Virtual interface for drawing bounding box
     * @param context Drawing context
     * @param bbox Bounding box
     */
    inline virtual void drawBBox(const CC_DRAW_CONTEXT& context,
                                 const ccBBox* bbox) { /* do nothing */ }

    /**
     * @brief Draw oriented bounding box
     * @param context Drawing context
     * @param obb Oriented bounding box to draw
     */
    inline static void DrawOrientedBBox(const CC_DRAW_CONTEXT& context,
                                        const ecvOrientedBBox* obb) {
        TheInstance()->drawOrientedBBox(context, obb);
    }
    
    /**
     * @brief Virtual interface for drawing oriented bounding box
     * @param context Drawing context
     * @param obb Oriented bounding box
     */
    inline virtual void drawOrientedBBox(
            const CC_DRAW_CONTEXT& context,
            const ecvOrientedBBox* obb) { /* do nothing */ }

    /**
     * @brief Remove bounding box from view
     * @param context Drawing context
     */
    static void RemoveBB(CC_DRAW_CONTEXT context);
    
    /**
     * @brief Remove bounding box by view ID
     * @param viewId View identifier
     */
    static void RemoveBB(const QString& viewId);
    
    /**
     * @brief Change entity properties
     * @param propertyParam Property parameters to apply
     * @param autoUpdate Auto-update display (default: true)
     */
    static void ChangeEntityProperties(PROPERTY_PARAM& propertyParam,
                                       bool autoUpdate = true);
    
    /**
     * @brief Virtual interface for changing entity properties
     * @param propertyParam Property parameters
     */
    inline virtual void changeEntityProperties(
            PROPERTY_PARAM& propertyParam) { /* do nothing */ }
    
    /**
     * @brief Draw widgets (2D/3D overlays)
     * @param param Widget parameters
     * @param update Update display immediately (default: false)
     */
    static void DrawWidgets(const WIDGETS_PARAMETER& param,
                            bool update = false);
    
    /**
     * @brief Virtual interface for drawing widgets
     * @param param Widget parameters
     */
    inline virtual void drawWidgets(
            const WIDGETS_PARAMETER& param) { /* do nothing */ }
    
    /**
     * @brief Remove widgets by parameters
     * @param param Widget parameters identifying widgets to remove
     * @param update Update display immediately (default: false)
     */
    static void RemoveWidgets(const WIDGETS_PARAMETER& param,
                              bool update = false);
    
    /**
     * @brief Remove all widgets from display
     * @param update Update display immediately (default: true)
     */
    static void RemoveAllWidgets(bool update = true);
    
    /**
     * @brief Remove 3D label by view ID
     * @param view_id View identifier of the label
     */
    static void Remove3DLabel(const QString& view_id);

    /**
     * @brief Draw coordinate system axes
     * @param scale Scale factor for axes size (default: 1.0)
     * @param id Unique identifier (default: "reference")
     * @param viewport Viewport index (default: 0)
     */
    inline static void DrawCoordinates(double scale = 1.0,
                                       const std::string& id = "reference",
                                       int viewport = 0) {
        TheInstance()->drawCoordinates(scale, id, viewport);
    }

    /**
     * @brief Virtual interface for drawing coordinate axes
     * @param scale Scale factor
     * @param id Identifier
     * @param viewport Viewport index
     */
    inline virtual void drawCoordinates(double scale = 1.0,
                                        const std::string& id = "reference",
                                        int viewport = 0) { /* do nothing */ }

    /**
     * @brief Rotate camera around arbitrary axis
     * 
     * Rotates the camera about a specified axis by a given angle.
     * @param pos Mouse position on screen
     * @param axis Rotation axis in world coordinates
     * @param angle Rotation angle in degrees
     * @param viewport Viewport index (default: 0)
     */
    inline static void RotateWithAxis(const CCVector2i& pos,
                                      const CCVector3d& axis,
                                      double angle,
                                      int viewport = 0) {
        TheInstance()->rotateWithAxis(pos, axis, angle, viewport);
    }

    /**
     * @brief Virtual interface for axis rotation
     * @param pos Mouse position
     * @param axis Rotation axis
     * @param angle Rotation angle (degrees)
     * @param viewport Viewport index
     */
    inline virtual void rotateWithAxis(const CCVector2i& pos,
                                       const CCVector3d& axis,
                                       double angle,
                                       int viewport = 0) { /* do nothing */ }

    /**
     * @brief Toggle orientation marker visibility
     * @param state Show marker (default: true)
     */
    inline static void ToggleOrientationMarker(bool state = true) {
        TheInstance()->toggleOrientationMarker(state);
        UpdateScreen();
    }
    
    /**
     * @brief Virtual interface for toggling orientation marker
     * @param state Visibility state
     */
    inline virtual void toggleOrientationMarker(
            bool state = true) { /* do nothing */ }

    /**
     * @brief Check if orientation marker is shown
     * @return true if marker is visible
     */
    inline static bool OrientationMarkerShown() {
        return TheInstance()->orientationMarkerShown();
    }
    
    /**
     * @brief Virtual interface for checking marker visibility
     * @return Marker visibility state
     */
    inline virtual bool orientationMarkerShown() {
        return false; /* do nothing */
    }

    // ========================================================================
    // ParaView-style Axes Grid Support (Unified Interface with
    // AxesGridProperties)
    // ========================================================================

    /**
     * @brief Set Data Axes Grid properties (Unified Interface)
     * Each ccHObject has its own Data Axes Grid bound to its viewID.
     *
     * @param viewID The view ID of the ccHObject (empty string for current
     * selected object)
     * @param props All axes grid properties encapsulated in AxesGridProperties
     * struct
     * @param viewport Viewport ID (default: 0)
     */
    inline static void SetDataAxesGridProperties(
            const QString& viewID,
            const AxesGridProperties& props,
            int viewport = 0) {
        TheInstance()->setDataAxesGridProperties(viewID, props, viewport);
        UpdateScreen();
    }

    /**
     * @brief Set Data Axes Grid properties (Virtual interface for derived
     * classes)
     */
    inline virtual void setDataAxesGridProperties(
            const QString& viewID,
            const AxesGridProperties& props,
            int viewport = 0) { /* do nothing */ }

    /**
     * @brief Get Data Axes Grid properties (Unified Interface)
     *
     * @param viewID The view ID of the ccHObject
     * @param props Output: current axes grid properties
     * @param viewport Viewport ID (default: 0)
     */
    inline static void GetDataAxesGridProperties(const QString& viewID,
                                                 AxesGridProperties& props,
                                                 int viewport = 0) {
        TheInstance()->getDataAxesGridProperties(viewID, props, viewport);
    }

    /**
     * @brief Get Data Axes Grid properties (Virtual interface for derived
     * classes)
     */
    inline virtual void getDataAxesGridProperties(const QString& viewID,
                                                  AxesGridProperties& props,
                                                  int viewport = 0) const {
        // Default implementation
        props = AxesGridProperties();
    }

    /// Enable/disable view axes grid (aligned with camera/view)
    inline static void SetViewAxesGridVisible(bool visible, int viewport = 0) {
        TheInstance()->setViewAxesGridVisible(visible, viewport);
        UpdateScreen();
    }
    inline virtual void setViewAxesGridVisible(
            bool visible, int viewport = 0) { /* do nothing */ }

    /// Configure view axes grid properties
    inline static void SetViewAxesGridProperties(bool visible,
                                                 const CCVector3& color,
                                                 double lineWidth,
                                                 double spacing,
                                                 int subdivisions,
                                                 bool showLabels,
                                                 double opacity,
                                                 int viewport = 0) {
        TheInstance()->setViewAxesGridProperties(visible, color, lineWidth,
                                                 spacing, subdivisions,
                                                 showLabels, opacity, viewport);
        UpdateScreen();
    }
    inline virtual void setViewAxesGridProperties(
            bool visible,
            const CCVector3& color,
            double lineWidth,
            double spacing,
            int subdivisions,
            bool showLabels,
            double opacity,
            int viewport = 0) { /* do nothing */ }

    /// Get view axes grid properties
    inline virtual void getViewAxesGridProperties(bool& visible,
                                                  CCVector3& color,
                                                  double& lineWidth,
                                                  double& spacing,
                                                  int& subdivisions,
                                                  bool& showLabels,
                                                  double& opacity,
                                                  int viewport = 0) const {
        // Default values
        visible = false;
        color = CCVector3(179, 179, 179);
        lineWidth = 1.0;
        spacing = 1.0;
        subdivisions = 10;
        showLabels = true;
        opacity = 0.3;
    }

    /// Enable/disable center axes visualization
    inline static void SetCenterAxesVisible(bool visible, int viewport = 0) {
        TheInstance()->setCenterAxesVisible(visible, viewport);
        UpdateScreen();
    }
    inline virtual void setCenterAxesVisible(
            bool visible, int viewport = 0) { /* do nothing */ }

    /// Toggle Camera Orientation Widget visibility (ParaView-style interactive
    /// widget)
    inline static void ToggleCameraOrientationWidget(bool show) {
        TheInstance()->toggleCameraOrientationWidget(show);
        UpdateScreen();
    }
    inline virtual void toggleCameraOrientationWidget(
            bool show) { /* do nothing */ }

    /// Check if Camera Orientation Widget is shown
    inline static bool IsCameraOrientationWidgetShown() {
        return TheInstance()->isCameraOrientationWidgetShown();
    }
    inline virtual bool isCameraOrientationWidgetShown() const { return false; }

    /// Set global default light intensity (ParaView-style)
    /// Modifies the renderer's headlight intensity for the entire scene.
    /// @param intensity Light intensity (0.0-1.0, default 1.0)
    inline static void SetLightIntensity(double intensity) {
        TheInstance()->setLightIntensity(intensity);
        UpdateScreen();
    }
    inline virtual void setLightIntensity(double intensity) { /* do nothing */ }

    /// Get current global default light intensity
    /// @return Current light intensity (0.0-1.0)
    inline static double GetLightIntensity() {
        return TheInstance()->getLightIntensity();
    }
    inline virtual double getLightIntensity() const { return 1.0; }

    /// Set light intensity for a specific object (per-object)
    /// @param viewID The view ID of the target object
    /// @param intensity Light intensity (0.0-1.0)
    inline static void SetObjectLightIntensity(const QString& viewID,
                                                double intensity) {
        TheInstance()->setObjectLightIntensity(viewID, intensity);
        UpdateScreen();
    }
    inline virtual void setObjectLightIntensity(const QString& /*viewID*/,
                                                 double /*intensity*/) {}

    /// Get light intensity for a specific object
    /// @param viewID The view ID of the target object
    /// @return Object's light intensity (falls back to global default)
    inline static double GetObjectLightIntensity(const QString& viewID) {
        return TheInstance()->getObjectLightIntensity(viewID);
    }
    inline virtual double getObjectLightIntensity(
            const QString& /*viewID*/) const {
        return 1.0;
    }

private:
    /**
     * @brief Internal 3D drawing method
     * @param CONTEXT Drawing context
     */
    static void Draw3D(CC_DRAW_CONTEXT& CONTEXT);

public:  // Main interface accessors
    /**
     * @brief Get 3D visualizer instance
     * @return Pointer to 3D visualizer
     */
    inline static ecvGenericVisualizer3D* GetVisualizer3D() {
        return TheInstance()->getVisualizer3D();
    }
    
    /**
     * @brief Virtual interface for getting 3D visualizer
     * @return 3D visualizer pointer
     */
    inline virtual ecvGenericVisualizer3D* getVisualizer3D() {
        return nullptr; /* do nothing */
    }
    
    /**
     * @brief Get 2D visualizer instance
     * @return Pointer to 2D visualizer
     */
    inline static ecvGenericVisualizer2D* GetVisualizer2D() {
        return TheInstance()->getVisualizer2D();
    }
    
    /**
     * @brief Virtual interface for getting 2D visualizer
     * @return 2D visualizer pointer
     */
    inline virtual ecvGenericVisualizer2D* getVisualizer2D() {
        return nullptr; /* do nothing */
    }

    /**
     * @brief Get current screen widget
     * @return Current screen widget pointer
     */
    inline static QWidget* GetCurrentScreen() {
        if (!TheInstance()) return nullptr;
        return TheInstance()->m_currentScreen;
    }
    
    /**
     * @brief Set current screen widget
     * @param widget Screen widget to set as current
     */
    static void SetCurrentScreen(QWidget* widget);
    
    /**
     * @brief Get main screen widget
     * @return Main screen widget pointer
     */
    inline static QWidget* GetMainScreen() {
        if (!TheInstance()) return nullptr;
        return TheInstance()->m_mainScreen;
    }
    
    /**
     * @brief Set main screen widget
     * @param widget Screen widget to set as main
     */
    inline static void SetMainScreen(QWidget* widget) {
        TheInstance()->m_mainScreen = widget;
    }

    /**
     * @brief Get main window
     * @return Main window pointer
     */
    inline static QMainWindow* GetMainWindow() { return TheInstance()->m_win; }
    
    /**
     * @brief Set main window
     * @param win Main window to set
     */
    inline static void SetMainWindow(QMainWindow* win) {
        TheInstance()->m_win = win;
    }

    /**
     * @brief Convert screen coordinates to centered GL coordinates
     * @param x Screen X coordinate
     * @param y Screen Y coordinate
     * @return Centered GL coordinates
     */
    static QPointF ToCenteredGLCoordinates(int x, int y);
    
    /**
     * @brief Convert screen coordinates to VTK coordinates
     * @param x Screen X coordinate
     * @param y Screen Y coordinate
     * @param z Screen Z coordinate (default: 0)
     * @return VTK 3D coordinates
     */
    static CCVector3d ToVtkCoordinates(int x, int y, int z = 0);
    
    /**
     * @brief Convert point to VTK coordinates (in-place)
     * @param sP Point to convert
     */
    static void ToVtkCoordinates(CCVector3d& sP);
    
    /**
     * @brief Convert 2D point to VTK coordinates (in-place)
     * @param sP Point to convert
     */
    static void ToVtkCoordinates(CCVector2i& sP);

    /**
     * @brief Get window's own database root
     * @return Window database root object
     */
    inline static ccHObject* GetOwnDB() { return TheInstance()->m_winDBRoot; }
    
    /**
     * @brief Add entity to window's own database
     * 
     * By default, no dependency link is established between the entity
     * and the window database.
     * @param obj Object to add
     * @param noDependency No dependency link (default: true)
     */
    static void AddToOwnDB(ccHObject* obj, bool noDependency = true);

    /**
     * @brief Remove entity from window's own database
     * @param obj Object to remove
     */
    static void RemoveFromOwnDB(ccHObject* obj);

    /**
     * @brief Set global scene database root
     * @param root Scene database root object
     */
    static void SetSceneDB(ccHObject* root);
    
    /**
     * @brief Get global scene database root
     * @return Scene database root object
     */
    inline static ccHObject* GetSceneDB() {
        return TheInstance()->m_globalDBRoot;
    }

    /**
     * @brief Update name and pose recursively for all objects
     */
    static void UpdateNamePoseRecursive();

    /**
     * @brief Set redraw flag recursively for all objects
     * @param redraw Redraw flag state (default: false)
     */
    static void SetRedrawRecursive(bool redraw = false);
    
    /**
     * @brief Set redraw flag recursively for object hierarchy
     * @param obj Root object for recursion
     * @param redraw Redraw flag state (default: false)
     */
    static void SetRedrawRecursive(ccHObject* obj, bool redraw = false);

    /**
     * @brief Get bounding box of all visible objects
     * @param box Output bounding box
     */
    static void GetVisibleObjectsBB(ccBBox& box);

    /**
     * @brief Rotate the base view matrix
     * 
     * The 'base view' matrix represents:
     * - In object-centered mode: rotation around the object
     * - In viewer-centered mode: rotation around the camera center
     * @param rotMat Rotation matrix to apply
     * @see setPerspectiveState
     */
    static void RotateBaseViewMat(const ccGLMatrixd& rotMat);

    /**
     * @brief Get base view matrix
     * @return Reference to base view matrix
     */
    inline static ccGLMatrixd& GetBaseViewMat() {
        return TheInstance()->m_viewportParams.viewMat;
    }
    
    /**
     * @brief Set base view matrix
     * @param mat View matrix to set
     */
    static void SetBaseViewMat(ccGLMatrixd& mat);

    /**
     * @brief Set view IDs to be removed
     * @param removeinfos List of removal information
     */
    static void SetRemoveViewIDs(std::vector<removeInfo>& removeinfos);
    
    /**
     * @brief Set remove-all flag
     * @param state Remove-all flag state
     */
    inline static void SetRemoveAllFlag(bool state) {
        TheInstance()->m_removeAllFlag = state;
    }

    /**
     * @brief Transform camera view matrix
     * @param viewMat View transformation matrix
     */
    inline static void TransformCameraView(const ccGLMatrixd& viewMat) {
        TheInstance()->transformCameraView(viewMat);
    }
    
    /**
     * @brief Virtual interface for view transformation
     * @param viewMat View matrix
     */
    inline virtual void transformCameraView(
            const ccGLMatrixd& viewMat) { /* do nothing */ }
    
    /**
     * @brief Transform camera projection matrix
     * @param projMat Projection transformation matrix
     */
    inline static void TransformCameraProjection(const ccGLMatrixd& projMat) {
        TheInstance()->transformCameraProjection(projMat);
    }
    
    /**
     * @brief Virtual interface for projection transformation
     * @param projMat Projection matrix
     */
    inline virtual void transformCameraProjection(
            const ccGLMatrixd& projMat) { /* do nothing */ }

    static inline int GetDevicePixelRatio() {
        // return TheInstance()->getDevicePixelRatio();
        return GetMainWindow()->devicePixelRatio();
    }

    // New: Cross-platform font size optimization function
    static inline int GetOptimizedFontSize(int baseFontSize = 12) {
        QWidget* win = GetMainWindow();
        if (!win) {
            return baseFontSize;
        }

        int dpiScale = win->devicePixelRatio();
        QScreen* screen = QApplication::primaryScreen();
        if (!screen) {
            return baseFontSize;
        }

        // Get screen resolution information
        QSize screenSize = screen->size();
        int screenWidth = screenSize.width();
        int screenHeight = screenSize.height();
        int screenDPI = screen->physicalDotsPerInch();

        // Platform-specific base font size adjustment
        int platformBaseSize = baseFontSize;
#ifdef Q_OS_MAC
        // macOS: Default font is slightly larger, but need to consider Retina
        // display over-scaling
        platformBaseSize = baseFontSize;
        if (dpiScale > 1) {
            // Retina display: Use smaller font to avoid over-scaling
            platformBaseSize = std::max(8, baseFontSize - (dpiScale - 1) * 2);
        }
#elif defined(Q_OS_WIN)
        // Windows: Adjust font size according to DPI
        if (screenDPI > 120) {
            // High DPI display
            platformBaseSize = std::max(8, baseFontSize - 1);
        } else if (screenDPI < 96) {
            // Low DPI display
            platformBaseSize = baseFontSize + 1;
        }
#elif defined(Q_OS_LINUX)
        // Linux: Adjust according to screen resolution
        if (screenWidth >= 1920 && screenHeight >= 1080) {
            // High resolution display
            platformBaseSize = std::max(8, baseFontSize - 1);
        } else if (screenWidth < 1366) {
            // Low resolution display
            platformBaseSize = baseFontSize + 1;
        }
#endif

        // Resolution-specific adjustment
        int resolutionFactor = 1;
        if (screenWidth >= 2560 && screenHeight >= 1440) {
            // 2K and above resolution
            resolutionFactor = 0;
        } else if (screenWidth >= 1920 && screenHeight >= 1080) {
            // 1080p resolution
            resolutionFactor = 0;
        } else if (screenWidth < 1366) {
            // Low resolution
            resolutionFactor = 1;
        }

        // Final font size calculation
        int finalSize = platformBaseSize + resolutionFactor;

        // Ensure font size is within reasonable range
        finalSize = std::max(6, std::min(24, finalSize));

        return finalSize;
    }

    // New: Cross-platform DPI scaling handling function
    static inline double GetPlatformAwareDPIScale() {
        QWidget* win = GetMainWindow();
        if (!win) {
            return 1.0;
        }

        int dpiScale = win->devicePixelRatio();
        QScreen* screen = QApplication::primaryScreen();
        if (!screen) {
            return static_cast<double>(dpiScale);
        }

        // Get screen information
        QSize screenSize = screen->size();
        int screenWidth = screenSize.width();
        int screenHeight = screenSize.height();
        int screenDPI = screen->physicalDotsPerInch();

        // Platform-specific DPI scaling adjustment
        double adjustedScale = static_cast<double>(dpiScale);

#ifdef Q_OS_MAC
        // macOS: Retina displays need special handling
        if (dpiScale > 1) {
            // For UI elements, use smaller scaling to avoid over-scaling
            adjustedScale = 1.0 + (dpiScale - 1.0) * 0.5;
        }
#elif defined(Q_OS_WIN)
        // Windows: Adjust according to DPI settings
        if (screenDPI > 120) {
            // High DPI display, appropriately reduce scaling
            adjustedScale = std::min(adjustedScale, 1.5);
        } else if (screenDPI < 96) {
            // Low DPI display, appropriately increase scaling
            adjustedScale = std::max(adjustedScale, 1.0);
        }
#elif defined(Q_OS_LINUX)
        // Linux: Adjust according to resolution
        if (screenWidth >= 2560 && screenHeight >= 1440) {
            // Ultra-high resolution, reduce scaling
            adjustedScale = std::min(adjustedScale, 1.3);
        } else if (screenWidth < 1366) {
            // Low resolution, increase scaling
            adjustedScale = std::max(adjustedScale, 1.0);
        }
#endif

        // Ensure scaling is within reasonable range
        adjustedScale = std::max(0.5, std::min(2.0, adjustedScale));

        return adjustedScale;
    }

    inline static QRect GetScreenRect() {
        QRect screenRect = GetCurrentScreen()->geometry();
        QPoint globalPosition =
                GetCurrentScreen()->mapToGlobal(screenRect.topLeft());
        screenRect.setTopLeft(globalPosition);
        return screenRect;
    }
    inline static void SetScreenSize(int xw, int yw) {
        GetCurrentScreen()->resize(QSize(xw, yw));
    }
    inline static void DoResize(int xw, int yw) { SetScreenSize(xw, yw); }
    inline static void DoResize(const QSize& size) {
        SetScreenSize(size.width(), size.height());
    }
    inline static QSize GetScreenSize() { return GetCurrentScreen()->size(); }

    inline static void SetRenderWindowSize(int xw, int yw) {
        TheInstance()->setRenderWindowSize(xw, yw);
    }
    inline virtual void setRenderWindowSize(int xw, int yw) { /* do nothing */ }

    inline static void FullScreen(bool state) {
        TheInstance()->fullScreen(state);
    }
    inline virtual void fullScreen(bool state) { /* do nothing */ }

    /**
     * @brief Zoom camera by factor
     * @param zoomFactor Zoom multiplier
     * @param viewport Viewport index (default: 0)
     */
    static void ZoomCamera(double zoomFactor, int viewport = 0);
    
    /**
     * @brief Virtual interface for camera zoom
     * @param zoomFactor Zoom multiplier
     * @param viewport Viewport index
     */
    inline virtual void zoomCamera(double zoomFactor, int viewport = 0) {}
    
    /**
     * @brief Get camera focal distance
     * @param viewport Viewport index (default: 0)
     * @return Focal distance
     */
    inline static double GetCameraFocalDistance(int viewport = 0) {
        return TheInstance()->getCameraFocalDistance(viewport);
    }
    
    /**
     * @brief Virtual interface for getting focal distance
     * @param viewport Viewport index
     * @return Focal distance
     */
    inline virtual double getCameraFocalDistance(int viewport = 0) {
        return 100.0; /* do nothing */
    }

    /**
     * @brief Set camera focal distance
     * @param focal_distance Focal distance to set
     * @param viewport Viewport index (default: 0)
     */
    inline static void SetCameraFocalDistance(double focal_distance,
                                              int viewport = 0) {
        TheInstance()->setCameraFocalDistance(focal_distance, viewport);
    }
    
    /**
     * @brief Virtual interface for setting focal distance
     * @param focal_distance Focal distance
     * @param viewport Viewport index
     */
    inline virtual void setCameraFocalDistance(
            double focal_distance, int viewport = 0) { /* do nothing */ }

    inline static void GetCameraPos(double* pos, int viewport = 0) {
        TheInstance()->getCameraPos(pos, viewport);
    }
    inline virtual void getCameraPos(double* pos,
                                     int viewport = 0) { /* do nothing */ }
    inline static void GetCameraFocal(double* focal, int viewport = 0) {
        TheInstance()->getCameraFocal(focal, viewport);
    }
    inline virtual void getCameraFocal(double* focal,
                                       int viewport = 0) { /* do nothing */ }
    inline static void GetCameraUp(double* up, int viewport = 0) {
        TheInstance()->getCameraUp(up, viewport);
    }
    virtual void getCameraUp(double* up, int viewport = 0) { /* do nothing */ }

    inline static void SetCameraPosition(const CCVector3d& pos,
                                         int viewport = 0) {
        TheInstance()->setCameraPosition(pos, viewport);
    }
    inline virtual void setCameraPosition(const CCVector3d& pos,
                                          int viewport = 0) { /* do nothing */ }
    inline static void SetCameraPosition(const double* pos,
                                         const double* focal,
                                         const double* up,
                                         int viewport = 0) {
        TheInstance()->setCameraPosition(pos, focal, up, viewport);
    }
    inline virtual void setCameraPosition(const double* pos,
                                          const double* focal,
                                          const double* up,
                                          int viewport = 0) { /* do nothing */ }
    inline static void SetCameraPosition(const double* pos,
                                         const double* up,
                                         int viewport = 0) {
        TheInstance()->setCameraPosition(pos, up, viewport);
    }
    inline virtual void setCameraPosition(const double* pos,
                                          const double* up,
                                          int viewport = 0) { /* do nothing */ }
    inline static void SetCameraPosition(double pos_x,
                                         double pos_y,
                                         double pos_z,
                                         double view_x,
                                         double view_y,
                                         double view_z,
                                         double up_x,
                                         double up_y,
                                         double up_z,
                                         int viewport = 0) {
        TheInstance()->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y,
                                         view_z, up_x, up_y, up_z, viewport);
    }
    inline virtual void setCameraPosition(double pos_x,
                                          double pos_y,
                                          double pos_z,
                                          double view_x,
                                          double view_y,
                                          double view_z,
                                          double up_x,
                                          double up_y,
                                          double up_z,
                                          int viewport = 0) { /* do nothing */ }

    // set and get clip distances (near and far)
    inline static void GetCameraClip(double* clipPlanes, int viewport = 0) {
        TheInstance()->getCameraClip(clipPlanes, viewport);
    }
    virtual void getCameraClip(double* clipPlanes,
                               int viewport = 0) { /* do nothing */ }
    inline static void SetCameraClip(double znear,
                                     double zfar,
                                     int viewport = 0) {
        TheInstance()->m_viewportParams.zNear = znear;
        TheInstance()->m_viewportParams.zFar = zfar;
        TheInstance()->setCameraClip(znear, zfar, viewport);
    }
    virtual void setCameraClip(double znear,
                               double zfar,
                               int viewport = 0) { /* do nothing */ }

    inline static void ResetCameraClippingRange(int viewport = 0) {
        TheInstance()->resetCameraClippingRange(viewport);
    }
    inline virtual void resetCameraClippingRange(
            int viewport = 0) { /* do nothing */ }

    // set and get view angle in y direction
    inline static double GetCameraFovy(int viewport = 0) {
        return TheInstance()->getCameraFovy(viewport);
    }
    inline virtual double getCameraFovy(int viewport = 0) {
        return 0; /* do nothing */
    }
    inline static void SetCameraFovy(double fovy, int viewport = 0) {
        TheInstance()->m_viewportParams.fov_deg = static_cast<float>(fovy);
        TheInstance()->setCameraFovy(cloudViewer::DegreesToRadians(fovy),
                                     viewport);
    }
    inline virtual void setCameraFovy(double fovy,
                                      int viewport = 0) { /* do nothing */ }

    inline static void GetViewerPos(int* viewPos, int viewport = 0) {
        viewPos[0] = 0;
        viewPos[1] = 0;
        viewPos[2] = Width();
        viewPos[3] = Height();
    }

    /** \brief Save the current rendered image to disk, as a PNG screenshot.
     * \param[in] file the name of the PNG file
     */
    inline static void SaveScreenshot(const std::string& file) {
        TheInstance()->saveScreenshot(file);
    }
    inline virtual void saveScreenshot(
            const std::string& file) { /* do nothing */ }

    /** \brief Save or Load the current rendered camera parameters to disk or
     * current camera. \param[in] file the name of the param file
     */
    inline static void SaveCameraParameters(const std::string& file) {
        TheInstance()->saveCameraParameters(file);
    }
    inline virtual void saveCameraParameters(
            const std::string& file) { /* do nothing */ }

    inline static void LoadCameraParameters(const std::string& file) {
        TheInstance()->loadCameraParameters(file);
    }
    inline virtual void loadCameraParameters(
            const std::string& file) { /* do nothing */ }

    inline static void ShowOrientationMarker() {
        TheInstance()->showOrientationMarker();
        UpdateScreen();
    }
    inline virtual void showOrientationMarker() { /* do nothing */ }

    inline static void SetOrthoProjection(int viewport = 0) {
        TheInstance()->setOrthoProjection(viewport);
        UpdateScreen();
    }
    inline virtual void setOrthoProjection(int viewport = 0) { /* do nothing */
    }
    inline static void SetPerspectiveProjection(int viewport = 0) {
        TheInstance()->setPerspectiveProjection(viewport);
        UpdateScreen();
    }
    inline virtual void setPerspectiveProjection(
            int viewport = 0) { /* do nothing */ }

    /** \brief Use Vertex Buffer Objects renderers.
     * This is an optimization for the obsolete OpenGL backend. Modern OpenGL2
     * backend (VTK \A1\DD 6.3) uses vertex buffer objects by default,
     * transparently for the user. \param[in] use_vbos set to true to use VBOs
     */
    inline static void SetUseVbos(bool useVbos) {
        TheInstance()->setUseVbos(useVbos);
    }
    inline virtual void setUseVbos(bool useVbos) { /* do nothing */ }

    /** \brief Set the ID of a cloud or shape to be used for LUT display
     * \param[in] id The id of the cloud/shape look up table to be displayed
     * The look up table is displayed by pressing 'u' in the PCLVisualizer */
    inline static void SetLookUpTableID(const std::string& viewID) {
        TheInstance()->setLookUpTableID(viewID);
    }
    inline virtual void setLookUpTableID(
            const std::string& viewID) { /* do nothing */ }

    inline static void GetProjectionMatrix(double* projArray,
                                           int viewport = 0) {
        TheInstance()->getProjectionMatrix(projArray, viewport);
    }
    inline virtual void getProjectionMatrix(double* projArray,
                                            int viewport = 0) { /* do nothing */
    }
    inline static void GetViewMatrix(double* viewArray, int viewport = 0) {
        TheInstance()->getViewMatrix(viewArray, viewport);
    }
    inline virtual void getViewMatrix(double* viewArray,
                                      int viewport = 0) { /* do nothing */ }

    inline static void SetViewMatrix(const ccGLMatrixd& viewMat,
                                     int viewport = 0) {
        TheInstance()->setViewMatrix(viewMat, viewport);
    }
    inline virtual void setViewMatrix(const ccGLMatrixd& viewMat,
                                      int viewport = 0) { /* do nothing */ }

    static bool HideShowEntities(const ccHObject* obj, bool visible);
    inline static bool HideShowEntities(const CC_DRAW_CONTEXT& CONTEXT) {
        return TheInstance()->hideShowEntities(CONTEXT);
    }
    inline virtual bool hideShowEntities(const CC_DRAW_CONTEXT& CONTEXT) {
        return true; /* do nothing */
    }
    static void HideShowEntities(const QStringList& viewIDs,
                                 ENTITY_TYPE hideShowEntityType,
                                 bool visibility = false);

    static void RemoveEntities(const ccHObject* obj);
    static void RemoveEntities(const QStringList& viewIDs,
                               ENTITY_TYPE removeEntityType);
    inline static void RemoveEntities(const CC_DRAW_CONTEXT& CONTEXT) {
        TheInstance()->removeEntities(CONTEXT);
    }
    inline virtual void removeEntities(
            const CC_DRAW_CONTEXT& CONTEXT) { /* do nothing */ }

    static void DrawBackground(CC_DRAW_CONTEXT& CONTEXT);
    static void DrawForeground(CC_DRAW_CONTEXT& CONTEXT);
    static void Update2DLabel(bool immediateUpdate = false);
    static void Pick2DLabel(int x, int y);
    virtual QString pick2DLabel(int x, int y) {
        return QString(); /* do nothing */
    }
    static void Redraw2DLabel();

    static QString Pick3DItem(int x = -1, int y = -1) {
        return TheInstance()->pick3DItem(x, y);
    }
    virtual QString pick3DItem(int x = -1, int y = -1) {
        return QString(); /* do nothing */
    }
    static QString PickObject(double x = -1, double y = -1) {
        return TheInstance()->pickObject(x, y);
    }
    virtual QString pickObject(double x = -1, double y = -1) {
        return QString(); /* do nothing */
    }

    static void FilterByEntityType(ccHObject::Container& labels,
                                   CV_CLASS_ENUM type);

    inline virtual void setBackgroundColor(
            const CC_DRAW_CONTEXT& CONTEXT) { /* do nothing */ }

    /** \brief Create a new viewport from [xmin,ymin] -> [xmax,ymax].
     * \param[in] xmin the minimum X coordinate for the viewport (0.0 <= 1.0)
     * \param[in] ymin the minimum Y coordinate for the viewport (0.0 <= 1.0)
     * \param[in] xmax the maximum X coordinate for the viewport (0.0 <= 1.0)
     * \param[in] ymax the maximum Y coordinate for the viewport (0.0 <= 1.0)
     * \param[in] viewport the id of the new viewport
     *
     * \note If no renderer for the current window exists, one will be created,
     * and the viewport will be set to 0 ('all'). In case one or multiple
     * renderers do exist, the viewport ID will be set to the total number of
     * renderers - 1.
     */
    inline static void CreateViewPort(
            double xmin, double ymin, double xmax, double ymax, int& viewport) {
        TheInstance()->createViewPort(xmin, ymin, xmax, ymax, viewport);
    }
    inline virtual void createViewPort(double xmin,
                                       double ymin,
                                       double xmax,
                                       double ymax,
                                       int& viewport) { /* do nothing */ }

    inline static void ResetCameraViewpoint(const std::string& viewID) {
        TheInstance()->resetCameraViewpoint(viewID);
    }
    inline virtual void resetCameraViewpoint(
            const std::string& viewID) { /* do nothing */ }

    static void SetPointSize(float size, bool silent = false, int viewport = 0);
    static void SetPointSizeRecursive(int size);

    //! Sets line width
    /** \param width lines width (between MIN_LINE_WIDTH_F and MAX_LINE_WIDTH_F)
     **/
    static void SetLineWidth(float width,
                             bool silent = false,
                             int viewport = 0);
    static void SetLineWithRecursive(PointCoordinateType with);

    inline static void Toggle2Dviewer(bool state) {
        TheInstance()->toggle2Dviewer(state);
    }
    inline virtual void toggle2Dviewer(bool state) { /* do nothing */ }

public:  // visualization matrix transformation
    //! Displays a status message in the bottom-left corner
    /** WARNING: currently, 'append' is not supported for SCREEN_CENTER_MESSAGE
            \param message message (if message is empty and append is 'false',
    all messages will be cleared) \param pos message position on screen \param
    append whether to append the message or to replace existing one(s) (only
    messages of the same type are impacted) \param displayMaxDelay_sec minimum
    display duration \param type message type (if not custom, only one message
    of this type at a time is accepted)
    **/
    static void DisplayNewMessage(const QString& message,
                                  MessagePosition pos,
                                  bool append = false,
                                  int displayMaxDelay_sec = 2,
                                  MessageType type = CUSTOM_MESSAGE);

    //! Returns current parameters for this display (const version)
    /** Warning: may return overridden parameters!
     **/
    static const ecvGui::ParamStruct& GetDisplayParameters();

    //! Sets current parameters for this display
    static void SetDisplayParameters(const ecvGui::ParamStruct& params);

    static void UpdateDisplayParameters();

    static void SetupProjectiveViewport(const ccGLMatrixd& cameraMatrix,
                                        float fov_deg = 0.0f,
                                        float ar = 1.0f,
                                        bool viewerBasedPerspective = true,
                                        bool bubbleViewMode = false);

    //! Sets current camera aspect ratio (width/height)
    /** AR is only used in perspective mode.
     **/
    static void SetAspectRatio(float ar);

    //! Sets current zoom
    /** Warning: has no effect in viewer-centered perspective mode
     **/
    static void ResizeGL(int w, int h);
    static void UpdateScreenSize();
    inline static void Update() {
        GetCurrentScreen()->update();
        UpdateCamera();
    }
    inline static void UpdateScreen() {
        GetCurrentScreen()->update();
        UpdateScene();
    }
    inline static void ResetCamera(const ccBBox* bbox) {
        TheInstance()->resetCamera(bbox);
        UpdateScreen();
    }
    inline virtual void resetCamera(const ccBBox* bbox) { /* do nothing */ }
    inline static void ResetCamera() {
        TheInstance()->resetCamera();
        UpdateScreen();
    }
    inline virtual void resetCamera() { /* do nothing */ }
    inline static void UpdateCamera() {
        TheInstance()->updateCamera();
        UpdateScreen();
    }
    inline virtual void updateCamera() { /* do nothing */ }

    inline static void UpdateScene() { TheInstance()->updateScene(); }
    inline virtual void updateScene() { /* do nothing */ }

    inline static void SetAutoUpateCameraPos(bool state) {
        TheInstance()->setAutoUpateCameraPos(state);
    }
    inline virtual void setAutoUpateCameraPos(bool state) { /* do nothing */ }

    /**
     * Get the current center of rotation
     */
    inline static void GetCenterOfRotation(double center[3]) {
        TheInstance()->getCenterOfRotation(center);
    }
    inline static void GetCenterOfRotation(CCVector3d& center) {
        TheInstance()->getCenterOfRotation(center.u);
    }
    inline virtual void getCenterOfRotation(double center[3]) { /* do nothing */
    }

    /**
     * Resets the center of rotation to the focal point.
     */
    inline static void ResetCenterOfRotation(int viewport = 0) {
        TheInstance()->resetCenterOfRotation(viewport);
        UpdateScreen();
    }
    inline virtual void resetCenterOfRotation(
            int viewport = 0) { /* do nothing */ }

    /**
     * Set the center of rotation. For this to work,
     * one should have appropriate interaction style
     * and camera manipulators that use the center of rotation
     * They are setup correctly by default
     */
    inline static void SetCenterOfRotation(double x, double y, double z) {
        TheInstance()->setCenterOfRotation(x, y, z);
    }
    inline virtual void setCenterOfRotation(double x,
                                            double y,
                                            double z) { /* do nothing */ }

    inline static void SetCenterOfRotation(const double xyz[3]) {
        SetCenterOfRotation(xyz[0], xyz[1], xyz[2]);
    }

    inline static void SetCenterOfRotation(const CCVector3d& center) {
        SetCenterOfRotation(center.u);
    }

    inline static double GetGLDepth(int x, int y) {
        return TheInstance()->getGLDepth(x, y);
    }
    inline virtual double getGLDepth(int x, int y) {
        return 1.0; /* do nothing */
    }

    inline static void ChangeOpacity(double opacity,
                                     const std::string& viewID,
                                     int viewport = 0) {
        TheInstance()->changeOpacity(opacity, viewID, viewport);
        UpdateScreen();
    }
    inline virtual void changeOpacity(double opacity,
                                      const std::string& viewID,
                                      int viewport = 0) {
        /* do nothing */
    }

    //! Converts a given (mouse) position in pixels to an orientation
    /** The orientation vector origin is the current pivot point!
     **/
    static CCVector3d ConvertMousePositionToOrientation(int x, int y);

    //! Updates currently active items list (m_activeItems)
    /** The items must be currently displayed in this context
            AND at least one of them must be under the mouse cursor.
    **/
    static void UpdateActiveItemsList(int x,
                                      int y,
                                      bool extendToSelectedLabels = false);

    //! Sets the OpenGL viewport (shortut)
    inline static void SetGLViewport(int x, int y, int w, int h) {
        SetGLViewport(QRect(x, y, w, h));
    }
    //! Sets the OpenGL viewport
    static void SetGLViewport(const QRect& rect);

    // Graphical features controls
    static void drawCross();
    static void drawTrihedron();

    //! Renders screen to a file
    static bool RenderToFile(QString filename,
                             float zoomFactor = 1.0f,
                             bool dontScaleFeatures = false,
                             bool renderOverlayItems = false);

    inline static QImage RenderToImage(int zoomFactor = 1,
                                       bool renderOverlayItems = false,
                                       bool silent = false,
                                       int viewport = 0) {
        return TheInstance()->renderToImage(zoomFactor, renderOverlayItems,
                                            silent, viewport);
    }
    inline virtual QImage renderToImage(int zoomFactor = 1,
                                        bool renderOverlayItems = false,
                                        bool silent = false,
                                        int viewport = 0) {
        return QImage(); /* do nothing */
    }

    inline static void SetScaleBarVisible(bool visible) {
        return TheInstance()->setScaleBarVisible(visible);
    }
    inline virtual void setScaleBarVisible(bool visible) { /* do nothing */ }

    static void DisplayTexture2DPosition(QImage image,
                                         const QString& id,
                                         int x,
                                         int y,
                                         int w,
                                         int h,
                                         unsigned char alpha = 255);
    //! Draws the 'hot zone' (+/- icons for point size), 'leave bubble-view'
    //! button, etc.
    static void DrawClickableItems(int xStart, int& yStart);
    static void RenderText(
            int x,
            int y,
            const QString& str,
            const QFont& font = QFont(),
            const ecvColor::Rgbub& color = ecvColor::defaultLabelBkgColor,
            const QString& id = "");
    static void RenderText(
            double x,
            double y,
            double z,
            const QString& str,
            const QFont& font = QFont(),
            const ecvColor::Rgbub& color = ecvColor::defaultLabelBkgColor,
            const QString& id = "");

    //! Toggles (exclusive) full-screen mode
    inline static void ToggleExclusiveFullScreen(bool state) {
        TheInstance()->toggleExclusiveFullScreen(state);
    }
    inline virtual void toggleExclusiveFullScreen(
            bool state) { /* in this do nothing */ }

    //! Returns whether the window is in exclusive full screen mode or not
    inline static bool ExclusiveFullScreen() {
        return TheInstance()->m_exclusiveFullscreen;
    }
    inline static void SetExclusiveFullScreenFlage(bool state) {
        TheInstance()->m_exclusiveFullscreen = state;
    }

    //! Sets pixel size (i.e. zoom base)
    /** Emits the 'pixelSizeChanged' signal.
     **/
    static void SetPixelSize(float pixelSize);

    //! Center and zoom on a given bounding box
    /** If no bounding box is defined, the current displayed 'scene graph'
            bounding box is taken.
    **/
    static void UpdateConstellationCenterAndZoom(const ccBBox* aBox = nullptr,
                                                 bool redraw = true);

    //! Returns context information
    static void GetContext(CC_DRAW_CONTEXT& CONTEXT);

    //! Returns the current OpenGL camera parameters
    static void GetGLCameraParameters(ccGLCameraParameters& params);

    static void SetInteractionMode(INTERACTION_FLAGS flags);
    //! Returns the current interaction flags
    static INTERACTION_FLAGS GetInteractionMode();

    static void SetView(CC_VIEW_ORIENTATION orientation, ccBBox* bbox);
    static void SetView(CC_VIEW_ORIENTATION orientation,
                        bool forceRedraw = false);

    //! Returns a 4x4 'OpenGL' matrix corresponding to a default 'view'
    //! orientation
    /** \param orientation view orientation
            \return corresponding GL matrix
    **/
    static ccGLMatrixd GenerateViewMat(CC_VIEW_ORIENTATION orientation);

    //! Set perspective state/mode
    /** Persepctive mode can be:
            - object-centered (moving the mouse make the object rotate)
            - viewer-centered (moving the mouse make the camera move)
            \warning Disables bubble-view mode automatically

            \param state whether perspective mode is enabled or not
            \param objectCenteredView whether view is object- or viewer-centered
    (forced to true in ortho. mode)
    **/
    static void SetPerspectiveState(bool state, bool objectCenteredView);

    static bool GetPerspectiveState(int viewport = 0) {
        return TheInstance()->getPerspectiveState(viewport);
    }
    inline virtual bool getPerspectiveState(int viewport = 0) const override {
        return TheInstance()->m_viewportParams.perspectiveView;
    }

    //! Shortcut: returns whether object-based perspective mode is enabled
    static bool ObjectPerspectiveEnabled();
    //! Shortcut: returns whether viewer-based perspective mode is enabled
    static bool ViewerPerspectiveEnabled();

    //! Returns the zoom value equivalent to the current camera position
    //! (perspective only)
    static float ComputePerspectiveZoom();

    //! Sets current camera f.o.v. (field of view) in degrees
    /** FOV is only used in perspective mode.
     **/
    static void SetFov(float fov);
    //! Returns the current f.o.v. (field of view) in degrees
    static float GetFov();

    inline static void ZoomGlobal() { UpdateConstellationCenterAndZoom(); }

    //! Returns current viewing direction
    /** This is the direction normal to the screen
            (pointing 'inside') in world base.
    **/
    static CCVector3d GetCurrentViewDir();

    //! Returns current up direction
    /** This is the vertical direction of the screen
            (pointing 'upward') in world base.
    **/
    static CCVector3d GetCurrentUpDir();

    //! Sets camera position
    /** Emits the 'cameraPosChanged' signal.
     **/
    static void SetCameraPos(const CCVector3d& P);

    //! Displaces camera
    /** Values are given in objects world along the current camera
            viewing directions (we use the right hand rule):
            * X: horizontal axis (right)
            * Y: vertical axis (up)
            * Z: depth axis (pointing out of the screen)
    **/
    static void MoveCamera(float dx, float dy, float dz);
    //! Displaces camera
    /** Values are given in objects world along the current camera
            viewing directions (we use the right hand rule):
            * X: horizontal axis (right)
            * Y: vertical axis (up)
            * Z: depth axis (pointing out of the screen)
            \param v displacement vector
    **/
    inline static void MoveCamera(const CCVector3d& v) {
        MoveCamera(v.x, v.y, v.z);
    }

    static void SetPickingMode(PICKING_MODE mode = DEFAULT_PICKING);
    static PICKING_MODE GetPickingMode();

    //! Locks picking mode
    /** \warning Bes sure to unlock it at some point ;)
     **/
    static void LockPickingMode(bool state);

    //! Returns whether picking mode is locked or not
    static bool IsPickingModeLocked();

    //! Sets current zoom
    /** Warning: has no effect in viewer-centered perspective mode
     **/
    static void SetZoom(float value);

    //! Updates current zoom
    /** Warning: has no effect in viewer-centered perspective mode
     **/
    static void UpdateZoom(float zoomFactor);

    //! Sets pivot point
    /** Emits the 'pivotPointChanged' signal.
     **/
    static void SetPivotPoint(const CCVector3d& P,
                              bool autoUpdateCameraPos = false,
                              bool verbose = false);

    //! Sets pivot visibility
    static void SetPivotVisibility(PivotVisibility vis);
    static void SetPivotVisibility(bool state) {
        TheInstance()->setPivotVisibility(state);
    }
    virtual void setPivotVisibility(bool state) { /*do nothing here*/ }

    //! Returns pivot visibility
    inline static PivotVisibility GetPivotVisibility() {
        return TheInstance()->m_pivotVisibility;
    }

    //! Shows or hide the pivot symbol
    /** Warnings:
            - not to be mistaken with setPivotVisibility
            - only taken into account if pivot visibility is set to
    PIVOT_SHOW_ON_MOVE
    **/
    static void ShowPivotSymbol(bool state);

    static void SetViewportParameters(const ecvViewportParameters& params);
    static const ecvViewportParameters& GetViewportParameters();

    // return value (in rad)
    inline static double GetParallelScale(int viewport = 0) {
        return TheInstance()->getParallelScale(viewport);
    }
    inline virtual double getParallelScale(int viewport = 0) { return -1.0; }

    // scale (in rad)
    inline static void SetParallelScale(double scale, int viewport = 0) {
        TheInstance()->setParallelScale(scale, viewport);
    }
    inline virtual void setParallelScale(double scale,
                                         int viewport = 0) { /*do nothing here*/
    }

    static ccGLMatrixd& GetModelViewMatrix();
    static ccGLMatrixd& GetProjectionMatrix();

    static void UpdateModelViewMatrix();
    static void UpdateProjectionMatrix();

    //! Computes the model view matrix
    static ccGLMatrixd ComputeModelViewMatrix();
    //! Computes the projection matrix
    /** \param[in]  withGLfeatures whether to take additional elements (pivot
    symbol, custom light, etc.) into account or not \param[out] metrics
    [optional] output other metrics (Znear and Zfar, etc.) \param[out] eyeOffset
    [optional] eye offset (for stereo display)
    **/
    static ccGLMatrixd ComputeProjectionMatrix(
            bool withGLfeatures,
            ProjectionMetrics* metrics = nullptr,
            double* eyeOffset = nullptr);

    inline static void Deprecate3DLayer() { TheInstance()->m_updateFBO = true; }
    inline static void InvalidateViewport() {
        TheInstance()->m_validProjectionMatrix = false;
    }
    inline static void InvalidateVisualization() {
        TheInstance()->m_validModelviewMatrix = false;
    }

    static CCVector3d GetRealCameraCenter();

    //! Returns the actual pixel size on screen (taking zoom or perspective
    //! parameters into account)
    /** In perspective mode, this value is approximate.
     **/
    static double ComputeActualPixelSize();

    //! Returns whether rectangular picking is allowed or not
    static bool IsRectangularPickingAllowed();

    //! Sets whether rectangular picking is allowed or not
    static void SetRectangularPickingAllowed(bool state);

    //! Sets bubble-view mode state
    /** Bubble-view is a kind of viewer-based perspective mode where
            the user can't displace the camera (apart from up-down or
            left-right rotations). The f.o.v. is also maximized.

            \warning Any call to a method that changes the perpsective will
            automatically disable this mode.
    **/
    static void SetBubbleViewMode(bool state);
    //! Returns whether bubble-view mode is enabled or no
    inline static bool BubbleViewModeEnabled() {
        return TheInstance()->m_bubbleViewModeEnabled;
    }
    //! Set bubble-view f.o.v. (in degrees)
    static void SetBubbleViewFov(float fov_deg);

    //! Sets whether to display the coordinates of the point below the cursor
    //! position
    inline static void ShowCursorCoordinates(bool state) {
        TheInstance()->m_showCursorCoordinates = state;
    }
    //! Whether the coordinates of the point below the cursor position are
    //! displayed or not
    inline static bool CursorCoordinatesShown() {
        return TheInstance()->m_showCursorCoordinates;
    }

    //! Toggles the automatic setting of the pivot point at the center of the
    //! screen
    static void SetAutoPickPivotAtCenter(bool state);
    static void SendAutoPickPivotAtCenter(bool state) {
        emit TheInstance() -> autoPickPivot(state);
    }
    //! Whether the pivot point is automatically set at the center of the screen
    inline static bool AutoPickPivotAtCenter() {
        return TheInstance()->m_autoPickPivotAtCenter;
    }

    //! Lock the rotation axis
    static void LockRotationAxis(bool state, const CCVector3d& axis);

    //! Returns whether the rotation axis is locaked or not
    inline static bool IsRotationAxisLocked() {
        return TheInstance()->m_rotationAxisLocked;
    }

    //! Returns the approximate 3D position of the clicked pixel
    static bool GetClick3DPos(int x, int y, CCVector3d& P3D);

    static void DrawPivot();

    // debug traces on screen
    //! Shows debug info on screen
    inline static void EnableDebugTrace(bool state) {
        TheInstance()->m_showDebugTraces = state;
    }

    //! Toggles debug info on screen
    inline static void ToggleDebugTrace() {
        TheInstance()->m_showDebugTraces = !TheInstance()->m_showDebugTraces;
    }

    /**
     * @struct PickingParameters
     * @brief Parameters for picking operations
     * 
     * Encapsulates all parameters needed for entity/point/triangle picking,
     * including picking mode, screen region, and database search scope.
     */
    struct PickingParameters {
        /**
         * @brief Constructor with parameters
         * @param _mode Picking mode (default: NO_PICKING)
         * @param _centerX Center X coordinate (default: 0)
         * @param _centerY Center Y coordinate (default: 0)
         * @param _pickWidth Pick region width (default: 5)
         * @param _pickHeight Pick region height (default: 5)
         * @param _pickInSceneDB Search in scene DB (default: true)
         * @param _pickInLocalDB Search in local DB (default: true)
         */
        PickingParameters(PICKING_MODE _mode = NO_PICKING,
                          int _centerX = 0,
                          int _centerY = 0,
                          int _pickWidth = 5,
                          int _pickHeight = 5,
                          bool _pickInSceneDB = true,
                          bool _pickInLocalDB = true)
            : mode(_mode),
              centerX(_centerX),
              centerY(_centerY),
              pickWidth(_pickWidth),
              pickHeight(_pickHeight),
              pickInSceneDB(_pickInSceneDB),
              pickInLocalDB(_pickInLocalDB) {}

        PICKING_MODE mode;          ///< Picking mode
        int centerX;                ///< Pick region center X
        int centerY;                ///< Pick region center Y
        int pickWidth;              ///< Pick region width (pixels)
        int pickHeight;             ///< Pick region height (pixels)
        bool pickInSceneDB;         ///< Search in scene database
        bool pickInLocalDB;         ///< Search in local database
    };

    //! Processes the clickable items
    /** \return true if an item has been clicked
     **/
    static bool ProcessClickableItems(int x, int y);

    //! Sets current camera 'zNear' coefficient
    /** zNear coef. is only used in perspective mode.
     **/
    static void SetZNearCoef(double coef);

    //! Starts picking process
    /** \param params picking parameters
     **/
    static void StartPicking(PickingParameters& params);

    static ccHObject* GetPickedEntity(const PickingParameters& params);

    //! Performs the picking with OpenGL
    static void StartOpenGLPicking(const PickingParameters& params);

    //! Starts OpenGL picking process
    static void StartCPUBasedPointPicking(const PickingParameters& params);

    //! Processes the picking process result and sends the corresponding signal
    static void ProcessPickingResult(
            const PickingParameters& params,
            ccHObject* pickedEntity,
            int pickedItemIndex,
            const CCVector3* nearestPoint = nullptr,
            const std::unordered_set<int>* selectedIDs = nullptr);

    //! Sets current font size
    /** Warning: only used internally.
            Change 'defaultFontSize' with setDisplayParameters instead!
    **/
    inline static void SetFontPointSize(int pixelSize) {
        TheInstance()->m_font.setPointSize(pixelSize);
    }
    //! Returns current font size
    static int GetFontPointSize();
    //! Returns current font size for labels
    static int GetLabelFontPointSize();

    static void SetClickableItemsVisible(bool state) {
        TheInstance()->m_clickableItemsVisible = state;
    }
    static bool GetClickableItemsVisible() {
        return TheInstance()->m_clickableItemsVisible;
    }

    // takes rendering zoom into account!
    static QFont GetLabelDisplayFont();
    // takes rendering zoom into account!
    inline static QFont GetTextDisplayFont() { return TheInstance()->m_font; }

    static ENTITY_TYPE ConvertToEntityType(const CV_CLASS_ENUM& type);

    //! Default picking radius value
    static const int DefaultPickRadius = 5;

    //! Sets picking radius
    inline static void SetPickingRadius(int radius) {
        TheInstance()->m_pickRadius = radius;
    }
    //! Returns the current picking radius
    inline static int GetPickingRadius() { return TheInstance()->m_pickRadius; }

    //! Sets whether overlay entities (scale, tetrahedron, etc.) should be
    //! displayed or not
    static void DisplayOverlayEntities(bool state);

    //! Returns whether overlay entities (scale, tetrahedron, etc.) are
    //! displayed or not
    inline static bool OverlayEntitiesAreDisplayed() {
        return TheInstance()->m_displayOverlayEntities;
    }

    //! Currently active items
    /** Active items can be moved with mouse, etc.
     **/
    std::list<ccInteractor*> m_activeItems;

protected:
    ecvDisplayTools() = default;
    //! register visualizer callback function
    virtual void registerVisualizer(QMainWindow* win,
                                    bool stereoMode = false) = 0;

    QWidget* m_currentScreen;
    QWidget* m_mainScreen;
    QMainWindow* m_win;

public:
    //! Viewport parameters (zoom, etc.)
    ecvViewportParameters m_viewportParams;

    /**
     * @struct ClickableItem
     * @brief Clickable UI item in hot zone
     * 
     * Represents an interactive item in the hot zone overlay,
     * with an associated role and screen area.
     */
    struct ClickableItem {
        /**
         * @brief Clickable item roles
         */
        enum Role {
            NO_ROLE,                    ///< No role assigned
            INCREASE_POINT_SIZE,        ///< Increase point size button
            DECREASE_POINT_SIZE,        ///< Decrease point size button
            INCREASE_LINE_WIDTH,        ///< Increase line width button
            DECREASE_LINE_WIDTH,        ///< Decrease line width button
            LEAVE_BUBBLE_VIEW_MODE,     ///< Exit bubble-view mode button
            LEAVE_FULLSCREEN_MODE,      ///< Exit fullscreen mode button
        };

        /**
         * @brief Default constructor
         */
        ClickableItem() : role(NO_ROLE) {}
        
        /**
         * @brief Constructor with role and area
         * @param _role Item role
         * @param _area Screen area rectangle
         */
        ClickableItem(Role _role, QRect _area) : role(_role), area(_area) {}

        Role role;      ///< Item role/function
        QRect area;     ///< Screen area (pixels)
    };

    //! Currently displayed clickable items
    std::vector<ClickableItem> m_clickableItems;

    //! Whether clickable items are visible (= mouse over) or not
    bool m_clickableItemsVisible;

    //! Current intercation flags
    INTERACTION_FLAGS m_interactionFlags;

    PICKING_MODE m_pickingMode;
    //! Whether picking mode is locked or not
    bool m_pickingModeLocked;

    //! Internal timer
    QElapsedTimer m_timer;

    //! Touch event in progress
    bool m_touchInProgress;
    //! Touch gesture initial distance
    qreal m_touchBaseDist;

    //! Scheduler timer
    QTimer m_scheduleTimer;
    //! Scheduled full redraw (no LOD)
    qint64 m_scheduledFullRedrawTime;

    //! Overridden display parameter
    ecvGui::ParamStruct m_overridenDisplayParameters;

    //! Whether to display overlay entities or not (scale, tetrahedron, etc.)
    bool m_displayOverlayEntities;

    //! Whether display parameters are overidden for this window
    bool m_overridenDisplayParametersEnabled;

    //! Wether exclusive full screen is enabled or not
    bool m_exclusiveFullscreen;

    //! Whether to display the coordinates of the point below the cursor
    //! position
    bool m_showCursorCoordinates;

    //! Whether the pivot point is automatically picked at the center of the
    //! screen (when possible)
    bool m_autoPickPivotAtCenter;

    //! Whether the display should be refreshed on next call to 'refresh'
    bool m_shouldBeRefreshed;

    //! Candidate pivot point (will be used when the mouse is released)
    CCVector3d m_autoPivotCandidate;

    //! viewport
    QRect m_glViewport;

    //! Debug traces visibility
    bool m_showDebugTraces;

    //! Picking radius (pixels)
    int m_pickRadius;

    //! Auto-refresh mode
    bool m_autoRefresh;

    //! Wheter the rotation axis is locked or not
    bool m_rotationAxisLocked;
    //! Locked rotation axis
    CCVector3d m_lockedRotationAxis;

    //! Complete visualization matrix (GL style - double version)
    ccGLMatrixd m_viewMatd;
    //! Whether the model veiw matrix is valid (or need to be recomputed)
    bool m_validModelviewMatrix;
    //! Projection matrix (GL style - double version)
    ccGLMatrixd m_projMatd;
    //! Whether the projection matrix is valid (or need to be recomputed)
    bool m_validProjectionMatrix;
    //! Distance between the camera and the displayed objects bounding-box
    double m_cameraToBBCenterDist;
    //! Half size of the displayed objects bounding-box
    double m_bbHalfDiag;

    //! Pivot symbol visibility
    PivotVisibility m_pivotVisibility;

    //! Whether pivot symbol should be shown or not
    bool m_pivotSymbolShown;

    //! Whether rectangular picking is allowed or not
    bool m_allowRectangularEntityPicking;

    //! Rectangular picking polyline
    ccPolyline* m_rectPickingPoly;

    cloudViewer::geometry::LineSet* m_scale_lineset;

    //! Window own DB
    ccHObject* m_winDBRoot;

    //! CV main DB
    ccHObject* m_globalDBRoot;

    bool m_removeFlag;
    bool m_removeAllFlag;
    std::vector<removeInfo> m_removeInfos;

    //! Whether to always use FBO or only for GL filters
    bool m_alwaysUseFBO;

    //! Whether FBO should be updated (or simply displayed as a texture =
    //! faster!)
    bool m_updateFBO;

    //! Sun light position
    /** Relative to screen.
     **/
    float m_sunLightPos[4];

    //! Whether sun light is enabled or not
    bool m_sunLightEnabled;

    //! Custom light position
    /** Relative to object.
     **/
    float m_customLightPos[4];

    //! Whether custom light is enabled or not
    bool m_customLightEnabled;

    //! Bubble-view mode state
    bool m_bubbleViewModeEnabled;

    //! Bubble-view mode f.o.v. (degrees)
    float m_bubbleViewFov_deg;

    //! Pre-bubble-view camera parameters (backup)
    ecvViewportParameters m_preBubbleViewParameters;

    //! Unique ID
    int m_uniqueID;

    //! Default font
    QFont m_font;

    //! Display capturing mode options
    struct CaptureModeOptions {
        //! Default constructor
        CaptureModeOptions()
            : enabled(false), zoomFactor(1.0f), renderOverlayItems(false) {}

        bool enabled;
        float zoomFactor;
        bool renderOverlayItems;
    };

    //! Display capturing mode options
    CaptureModeOptions m_captureMode;

    //! Deferred picking
    QTimer m_deferredPickingTimer;

public:  // event representation
    static bool USE_2D;
    static bool USE_VTK_PICK;

    CCVector3 m_last_picked_point;
    int m_last_point_index = -1;
    QString m_last_picked_id = QString();

    //! Last click time (msec)
    qint64 m_lastClickTime_ticks;

    //! Hot zone
    HotZone* m_hotZone;

    //! Last mouse position
    QPoint m_lastMousePos;

    QPoint m_lastMouseMovePos;

    QStringList m_diagStrings;

    //! Whether the mouse (cursor) has moved after being pressed or not
    bool m_mouseMoved;
    //! Whether the mouse is currently pressed or not
    bool m_mouseButtonPressed;

    //! Ignore next mouse release event
    bool m_ignoreMouseReleaseEvent;

    //! Flag to indicate that a VTK widget was clicked (to prevent deferred
    //! picking)
    bool m_widgetClicked;

    static int Width() { return size().width(); }
    static int Height() { return size().height(); }
    static QSize size() { return GetScreenSize(); }

    //! Returns the OpenGL context width
    static int GlWidth() { return TheInstance()->m_glViewport.width(); }
    //! Returns the OpenGL context height
    static int GlHeight() { return TheInstance()->m_glViewport.height(); }
    //! Returns the OpenGL context size
    static QSize GlSize() { return TheInstance()->m_glViewport.size(); }

    static void ClearBubbleView();

public slots:

    //! Reacts to the itemPickedFast signal
    void onItemPickedFast(ccHObject* pickedEntity,
                          int pickedItemIndex,
                          int x,
                          int y);

    void onPointPicking(const CCVector3& p, int index, const std::string& id);

    //! Checks for scheduled redraw
    void checkScheduledRedraw();

    //! Performs standard picking at the last clicked mouse position (see
    //! m_lastMousePos)
    void doPicking();

    // called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

signals:

    /**
     * @brief Signal emitted when entity selection changes
     * @param entity Selected entity (nullptr if deselected)
     */
    void entitySelectionChanged(ccHObject* entity);
    
    /**
     * @brief Signal emitted when multiple entities are selected
     * @param entIDs Set of selected entity IDs
     */
    void entitiesSelectionChanged(std::unordered_set<int> entIDs);

    /**
     * @brief Signal emitted when point or triangle is picked
     * @param entity Picked entity
     * @param subEntityID Point or triangle index within entity
     * @param x Mouse cursor X position (pixels)
     * @param y Mouse cursor Y position (pixels)
     * @param P 3D coordinates of picked point
     */
    void itemPicked(ccHObject* entity,
                    unsigned subEntityID,
                    int x,
                    int y,
                    const CCVector3& P);

    /**
     * @brief Signal emitted during fast picking (FAST_PICKING mode)
     * @param entity Picked entity
     * @param subEntityID Point or triangle index
     * @param x Mouse cursor X position (pixels)
     * @param y Mouse cursor Y position (pixels)
     */
    void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);

    /**
     * @brief Signal emitted when fast picking completes
     */
    void fastPickingFinished();

    /*** Camera link mode (interactive modifications of the view/camera are
     * echoed to other windows) ***/

    //! Signal emitted when the window 'model view' matrix is interactively
    //! changed
    void viewMatRotated(const ccGLMatrixd& rotMat);
    //! Signal emitted when the camera is interactively displaced
    void cameraDisplaced(float ddx, float ddy);
    //! Signal emitted when the mouse wheel is rotated
    void mouseWheelRotated(float wheelDelta_deg);
    void mouseWheelChanged(QWheelEvent* event);

    //! Signal emitted when the perspective state changes (see
    //! setPerspectiveState)
    void perspectiveStateChanged();

    //! Signal emitted when the window 'base view' matrix is changed
    void baseViewMatChanged(const ccGLMatrixd& newViewMat);

    //! Signal emitted when the pixel size is changed
    void pixelSizeChanged(float pixelSize);

    //! Signal emitted when the f.o.v. changes
    void fovChanged(float fov);

    //! Signal emitted when the zNear coef changes
    void zNearCoefChanged(float coef);

    //! Signal emitted when the pivot point is changed
    void pivotPointChanged(const CCVector3d&);

    //! Signal emitted when the camera position is changed
    void cameraPosChanged(const CCVector3d&);

    //! Signal emitted when the selected object is translated by the user
    void translation(const CCVector3d& t);

    //! Signal emitted when the selected object is rotated by the user
    /** \param rotMat rotation applied to current viewport (4x4 OpenGL matrix)
     **/
    void rotation(const ccGLMatrixd& rotMat);

    //! Signal emitted when the left mouse button is cliked on the window
    /** See INTERACT_SIG_LB_CLICKED.
            Arguments correspond to the clicked point coordinates (x,y) in
            pixels relative to the window corner!
    **/
    void leftButtonClicked(int x, int y);

    //! Signal emitted when the right mouse button is cliked on the window
    /** See INTERACT_SIG_RB_CLICKED.
            Arguments correspond to the clicked point coordinates (x,y) in
            pixels relative to the window corner!
    **/
    void rightButtonClicked(int x, int y);

    //! Signal emitted when the double mouse button is cliked on the window
    /** See INTERACT_SIG_LB_CLICKED.
            Arguments correspond to the clicked point coordinates (x,y) in
            pixels relative to the window corner!
    **/
    void doubleButtonClicked(int x, int y);

    //! Signal emitted when the mouse is moved
    /** See INTERACT_SIG_MOUSE_MOVED.
            The two first arguments correspond to the current cursor coordinates
    (x,y) relative to the window corner!
    **/
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);

    //! Signal emitted when a mouse button is released (cursor on the window)
    /** See INTERACT_SIG_BUTTON_RELEASED.
     **/
    void buttonReleased();

    //! Signal emitted during 3D pass of OpenGL display process
    /** Any object connected to this slot can draw additional stuff in 3D.
            Depth buffering, lights and shaders are enabled by default.
    **/
    void drawing3D();

    //! Signal emitted when files are dropped on the window
    void filesDropped(const QStringList& filenames, bool displayDialog);

    //! Signal emitted when a new label is created
    void newLabel(ccHObject* obj);

    //! Signal emitted when the exclusive fullscreen is toggled
    void exclusiveFullScreenToggled(bool exclusive);
    void autoPickPivot(bool state);

    void labelmove2D(int x, int y, int dx, int dy);

    void mousePosChanged(const QPoint& pos);

    void pointPicked(double x, double y, double z);

    void cameraParamChanged();
};

Q_DECLARE_OPERATORS_FOR_FLAGS(ecvDisplayTools::INTERACTION_FLAGS);
