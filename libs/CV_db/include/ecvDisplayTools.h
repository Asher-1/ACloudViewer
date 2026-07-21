// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "ecvDisplayTypes.h"
#include "ecvGLMatrix.h"
#include "ecvGenericDisplayTools.h"
#include "ecvGenericGLDisplay.h"
#include "ecvGuiParameters.h"
#include "ecvHObject.h"
#include "ecvViewContext.h"
#include "ecvViewManager.h"
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
#include <initializer_list>
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

// AxesGridProperties is now in ecvDisplayTypes.h (included above)

/**
 * @class ecvDisplayTools
 * @brief Main display and rendering management class
 *
 * Central class for managing visualization, rendering, and user interaction in
 * CloudViewer. Provides a comprehensive interface for:
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
 * Lifecycle is owned by ecvViewManager (displayTools()). Integrates with Qt
 * (for UI) and VTK/OpenGL (for 3D rendering).
 *
 * @see ecvGenericDisplayTools
 * @see ecvGenericVisualizer3D
 * @see ecvGenericVisualizer2D
 */
class CV_DB_LIB_API ecvDisplayTools : public QObject,
                                      public ecvGenericDisplayTools,
                                      public ecvGenericGLDisplay {
    Q_OBJECT
public:
    // ================================================================
    // Shared display-tools lifecycle (managed by ecvViewManager)
    //
    // Use ecvViewManager::initDisplayTools() / displayTools() /
    // releaseDisplayTools().
    // ================================================================

    /// One-time setup after construction; called from
    /// ecvViewManager::initDisplayTools().
    void initializeEngine(QMainWindow* win, bool stereoMode = false);

private:
    friend class ecvViewManager;

public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ecvDisplayTools() override;

    // Phase M4: beginPrimaryRender/endPrimaryRender removed. Each vtkGLView
    // now does its own full rendering pipeline without ecvViewManager-wide
    // primary display-tools context swap.

    // ================================================================
    // Per-view context
    //
    // VtkDisplayTools is an engine, not a view; per-view state lives on
    // vtkGLView. Parameter-less static wrappers use
    // ecvViewManager::instance().resolveViewContext().
    // ================================================================

    /// ecvDisplayTools (VTK engine) does not own a view context.
    ecvViewContext* viewContext() override { return nullptr; }
    const ecvViewContext* viewContext() const override { return nullptr; }

    ecvHotZone*& hotZonePtrRef() override { return m_hotZone; }

    // ================================================================
    // Context-aware static API overloads  (Phase A)
    //
    // These overloads accept an explicit ecvViewContext so callers can
    // operate on a specific view's state without relying on
    // ecvViewManager::resolveViewContext() or activeSecondaryView
    // delegation.  The original
    // overloads (no ecvViewContext param) still work as before.
    // ================================================================

    /// Fill a draw context from explicit per-view state.
    static void GetContext(CC_DRAW_CONTEXT& CONTEXT,
                           const ecvViewContext& viewCtx);

    /// Read camera params from explicit per-view state.
    static void GetGLCameraParameters(ccGLCameraParameters& params,
                                      const ecvViewContext& viewCtx);

    /// Modify point size inside a context (no shared-state side-effects).
    static void SetPointSize(ecvViewContext& ctx, float size);

    /// Modify line width inside a context (no shared-state side-effects).
    static void SetLineWidth(ecvViewContext& ctx, float width);

    /// Modify near/far clip inside a context (no shared-state side-effects).
    static void SetCameraClip(ecvViewContext& ctx, double znear, double zfar);

    /// Modify vertical FOV inside a context (no shared-state side-effects).
    static void SetCameraFovy(ecvViewContext& ctx, double fovy);
    static PivotVisibility GetPivotVisibility(const ecvViewContext& ctx);
    static void SetInteractionMode(ecvViewContext& ctx,
                                   INTERACTION_FLAGS flags);
    static INTERACTION_FLAGS GetInteractionMode(const ecvViewContext& ctx);
    static void SetPickingMode(ecvViewContext& ctx, PICKING_MODE mode);
    static PICKING_MODE GetPickingMode(const ecvViewContext& ctx);

    // ================================================================
    // Phase N1: Per-view context parameterized overloads
    // Each has a corresponding parameterless overload on display tools
    // (managed by ecvViewManager) that delegates to resolveViewContext()
    // (or ctx-parameterized overloads).
    // ================================================================

    static bool IsRectangularPickingAllowed(const ecvViewContext& ctx);
    static void SetRectangularPickingAllowed(ecvViewContext& ctx, bool state);
    static void LockPickingMode(ecvViewContext& ctx, bool state);
    static bool IsPickingModeLocked(const ecvViewContext& ctx);
    static void DisplayOverlayEntities(ecvViewContext& ctx, bool state);
    static void SetViewportDefaultPointSize(ecvViewContext& ctx, float size);
    static void SetViewportDefaultLineWidth(ecvViewContext& ctx, float width);
    static bool ObjectPerspectiveEnabled(const ecvViewContext& ctx);
    static bool ViewerPerspectiveEnabled(const ecvViewContext& ctx);
    static void SetGLViewport(ecvViewContext& ctx, const QRect& rect);
    static CCVector3d GetCurrentViewDir(const ecvViewContext& ctx);
    static CCVector3d GetCurrentUpDir(const ecvViewContext& ctx);
    static const ecvViewportParameters& GetViewportParameters(
            const ecvViewContext& ctx);
    static bool GetClick3DPos(const ecvViewContext& ctx,
                              int x,
                              int y,
                              CCVector3d& P3D);

    // ================================================================
    // Phase N2: Per-view context parameterized overloads (continued)
    // N2a: Functions with side effects (emit, QSettings, VTK callbacks)
    // N2b: State setters/getters delegating to resolveViewContext()
    // ================================================================

    // N2b: Simple state accessors
    static void SetPixelSize(ecvViewContext& ctx, float pixelSize);
    static void SetZoom(ecvViewContext& ctx, float value);
    static void UpdateZoom(ecvViewContext& ctx, float zoomFactor);
    static void SetAspectRatio(ecvViewContext& ctx, float ar);
    static void UpdateModelViewMatrix(ecvViewContext& ctx);
    static void SetBaseViewMat(ecvViewContext& ctx, ccGLMatrixd& mat);
    static ccGLMatrixd& GetModelViewMatrix(ecvViewContext& ctx);
    static ccGLMatrixd& GetProjectionMatrix(ecvViewContext& ctx);
    static CCVector3d GetRealCameraCenter(const ecvViewContext& ctx);
    static void SetCameraPos(ecvViewContext& ctx, const CCVector3d& P);
    static float ComputePerspectiveZoom(const ecvViewContext& ctx);
    static void LockRotationAxis(ecvViewContext& ctx,
                                 bool state,
                                 const CCVector3d& axis);
    static void ShowPivotSymbol(ecvViewContext& ctx, bool state);

    // N2 remaining: ctx-parameterized overloads
    static ccGLMatrixd ComputeModelViewMatrix(const ecvViewContext& ctx);
    static void MoveCamera(ecvViewContext& ctx, float dx, float dy, float dz);
    static ccHObject* GetPickedEntity(const ecvViewContext& ctx,
                                      const ecvPickingParameters& params);
    static void SetBubbleViewFov(ecvViewContext& ctx, float fov_deg);
    static void SetFov(ecvViewContext& ctx, float fov_deg);
    static void UpdateProjectionMatrix(ecvViewContext& ctx);
    static void SetView(ecvViewContext& ctx,
                        CC_VIEW_ORIENTATION orientation,
                        bool forceRedraw = false);
    static void StartOpenGLPicking(ecvViewContext& ctx,
                                   const ecvPickingParameters& params);
    static void SetupProjectiveViewport(ecvViewContext& ctx,
                                        const ccGLMatrixd& cameraMatrix,
                                        float fov_deg = 0.0f,
                                        float ar = 1.0f,
                                        bool viewerBasedPerspective = true,
                                        bool bubbleViewMode = false);
    static bool ProcessClickableItems(ecvViewContext& ctx, int x, int y);

    // N2a: Functions with side effects (emit, QSettings, VTK callbacks)
    static void SetPivotVisibility(ecvViewContext& ctx, PivotVisibility vis);
    static void ResizeGL(ecvViewContext& ctx, int w, int h);
    static void RotateBaseViewMat(ecvViewContext& ctx,
                                  const ccGLMatrixd& rotMat);
    static CCVector3d ConvertMousePositionToOrientation(
            const ecvViewContext& ctx, int x, int y);
    static void RedrawDisplay(ecvViewContext& ctx, bool only2D = false);
    static void Draw3D(ecvViewContext& ctx, CC_DRAW_CONTEXT& CONTEXT);

    // N3: Heavy State Mutators
    static void SetZNearCoef(ecvViewContext& ctx, double coef);
    static double ComputeActualPixelSize(const ecvViewContext& ctx);
    static void SetViewportParameters(ecvViewContext& ctx,
                                      const ecvViewportParameters& params);
    static void SetBubbleViewMode(ecvViewContext& ctx, bool state);
    static void UpdateDisplayParameters(ecvViewContext& ctx);

    // N4: Core Projection/Camera Engine
    static ccGLMatrixd ComputeProjectionMatrix(
            const ecvViewContext& ctx,
            bool withGLfeatures,
            ecvProjectionMetrics* metrics = nullptr,
            double* eyeOffset = nullptr);
    static void SetPerspectiveState(ecvViewContext& ctx,
                                    bool state,
                                    bool objectCenteredView);

    // N5: Picking Pipeline
    static void StartCPUBasedPointPicking(ecvViewContext& ctx,
                                          const ecvPickingParameters& params);
    static void DrawPivot(const ecvViewContext& ctx);

    // -- ecvGenericGLDisplay implementation (shared display tools, managed by
    // ecvViewManager) --

    int getUniqueID() const override { return m_uniqueID; }
    QString getTitle() const override { return QStringLiteral("RenderView1"); }
    void redraw(bool only2D = false, bool forceRedraw = true) override;
    void refresh(bool only2D = false) override;
    void toBeRefreshed() override;
    const ecvViewportParameters& getViewportParameters() const override;
    void setViewportParameters(const ecvViewportParameters& params) override;
    void setPerspectiveState(bool state,
                             bool objectCenteredView,
                             bool persistDefault = true) override;
    bool perspectiveView() const override;
    bool objectCenteredView() const override;
    void setSceneDB(ccHObject* root) override;
    ccHObject* getSceneDB() override;
    ccHObject* getOwnDB() override;
    void addToOwnDB(ccHObject* obj, bool noDependency = true) override;
    void removeFromOwnDB(ccHObject* obj) override;
    void updateConstellationCenterAndZoom(
            const ccBBox* aBox = nullptr) override;
    QWidget* asWidget() override;
    const QWidget* asWidget() const override;
    bool hasOverriddenDisplayParameters() const override;
    QFont textDisplayFont() const override {
        return ecvViewManager::instance().defaultFont();
    }

public:
    // PICKING_MODE, INTERACTION_FLAG/FLAGS, MessagePosition, MessageType,
    // PivotVisibility are now defined in the base class ecvGenericGLDisplay
    // and inherited here.  Existing code using ecvDisplayTools::NO_PICKING,
    // ecvDisplayTools::INTERACT_ROTATE, etc. still compiles.

    static INTERACTION_FLAGS PAN_ONLY();
    static INTERACTION_FLAGS TRANSFORM_CAMERA();
    static INTERACTION_FLAGS TRANSFORM_ENTITIES();

    using MessageToDisplay = ecvMessageToDisplay;

    using ProjectionMetrics = ecvProjectionMetrics;

    using HotZone = ecvHotZone;

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
                            const QString& id = "",
                            ecvGenericGLDisplay* display = nullptr);

    /**
     * @brief Display text with draw context
     * @param CONTEXT Drawing context containing text parameters
     */
    static void DisplayText(const CC_DRAW_CONTEXT& CONTEXT) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->displayText(CONTEXT);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->draw(context, obj);
    }

    /**
     * @brief Virtual draw interface (to be overridden)
     * @param context Drawing context
     * @param obj Object to draw
     */
    inline void draw(const CC_DRAW_CONTEXT& context,
                     const ccHObject* obj) override { /* do nothing */ }

    /**
     * @brief Update mesh textures
     * @param context Drawing context
     * @param mesh Mesh with textures to update
     */
    inline static void UpdateMeshTextures(const CC_DRAW_CONTEXT& context,
                                          const ccGenericMesh* mesh) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->updateMeshTextures(context, mesh);
    }

    /**
     * @brief Virtual interface for updating mesh textures
     * @param context Drawing context
     * @param mesh Mesh to update
     */
    inline void updateMeshTextures(
            const CC_DRAW_CONTEXT& context,
            const ccGenericMesh* mesh) override { /* do nothing */ }

    /**
     * @brief Draw axis-aligned bounding box
     * @param context Drawing context
     * @param bbox Bounding box to draw
     */
    inline static void DrawBBox(const CC_DRAW_CONTEXT& context,
                                const ccBBox* bbox) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->drawBBox(context, bbox);
    }

    inline static void DrawBBoxBatch(const CC_DRAW_CONTEXT& context,
                                     const std::vector<ccBBox>& boxes) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->drawBBoxBatch(context, boxes);
    }

    /**
     * @brief Virtual interface for drawing bounding box
     * @param context Drawing context
     * @param bbox Bounding box
     */
    inline void drawBBox(const CC_DRAW_CONTEXT& context,
                         const ccBBox* bbox) override { /* do nothing */ }

    /**
     * @brief Draw oriented bounding box
     * @param context Drawing context
     * @param obb Oriented bounding box to draw
     */
    inline static void DrawOrientedBBox(const CC_DRAW_CONTEXT& context,
                                        const ecvOrientedBBox* obb) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->drawOrientedBBox(context, obb);
    }

    /**
     * @brief Virtual interface for drawing oriented bounding box
     * @param context Drawing context
     * @param obb Oriented bounding box
     */
    inline void drawOrientedBBox(
            const CC_DRAW_CONTEXT& context,
            const ecvOrientedBBox* obb) override { /* do nothing */ }

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
    inline void changeEntityProperties(
            PROPERTY_PARAM& propertyParam) override { /* do nothing */ }

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
    inline void drawWidgets(
            const WIDGETS_PARAMETER& param) override { /* do nothing */ }

    /**
     * @brief Remove widgets by parameters
     * @param param Widget parameters identifying widgets to remove
     * @param update Update display immediately (default: false)
     */
    void removeWidgets(const WIDGETS_PARAMETER& param) override;

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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->drawCoordinates(scale, id, viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->rotateWithAxis(pos, axis, angle, viewport);
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
                                       int viewport) { /* do nothing */ }

    /**
     * @brief Toggle orientation marker visibility
     * @param state Show marker (default: true)
     */
    inline static void ToggleOrientationMarker(bool state = true) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->toggleOrientationMarker(state);
        UpdateScreen();
    }

    /**
     * @brief Virtual interface for toggling orientation marker
     * @param state Visibility state
     */
    inline void toggleOrientationMarker(
            bool state = true) override { /* do nothing */ }

    /**
     * @brief Check if orientation marker is shown
     * @return true if marker is visible
     */
    inline static bool OrientationMarkerShown() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->orientationMarkerShown();
        return false;
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setDataAxesGridProperties(viewID, props, viewport);
        UpdateScreen();
    }

    /**
     * @brief Set Data Axes Grid properties (Virtual interface for derived
     * classes)
     */
    inline void setDataAxesGridProperties(
            const QString& viewID,
            const AxesGridProperties& props,
            int viewport = 0) override { /* do nothing */ }

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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getDataAxesGridProperties(viewID, props, viewport);
    }

    /**
     * @brief Get Data Axes Grid properties (Virtual interface for derived
     * classes)
     */
    inline void getDataAxesGridProperties(const QString& viewID,
                                          AxesGridProperties& props,
                                          int viewport = 0) const override {
        props = AxesGridProperties();
    }

    /// Enable/disable view axes grid (aligned with camera/view)
    inline static void SetViewAxesGridVisible(bool visible, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setViewAxesGridVisible(visible, viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setViewAxesGridProperties(visible, color, lineWidth, spacing,
                                          subdivisions, showLabels, opacity,
                                          viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCenterAxesVisible(visible, viewport);
        UpdateScreen();
    }
    inline virtual void setCenterAxesVisible(
            bool visible, int viewport = 0) { /* do nothing */ }

    /// Toggle Camera Orientation Widget visibility (ParaView-style interactive
    /// widget)
    inline static void ToggleCameraOrientationWidget(bool show) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->toggleCameraOrientationWidget(show);
        UpdateScreen();
    }
    inline void toggleCameraOrientationWidget(
            bool show) override { /* do nothing */ }

    /// Check if Camera Orientation Widget is shown
    inline static bool IsCameraOrientationWidgetShown() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->isCameraOrientationWidgetShown();
        return false;
    }
    inline virtual bool isCameraOrientationWidgetShown() const { return false; }

    /// Set global default light intensity (ParaView-style)
    /// Modifies the renderer's headlight intensity for the entire scene.
    /// @param intensity Light intensity (0.0-1.0, default 1.0)
    inline static void SetLightIntensity(double intensity) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setLightIntensity(intensity);
        UpdateScreen();
    }
    inline void setLightIntensity(double intensity) override { /* do nothing */
    }

    /// Get current global default light intensity
    /// @return Current light intensity (0.0-1.0)
    inline static double GetLightIntensity() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getLightIntensity();
        return 1.0;
    }
    inline double getLightIntensity() const override { return 1.0; }

    /// Set light intensity for a specific object (per-object)
    /// @param viewID The view ID of the target object
    /// @param intensity Light intensity (0.0-1.0)
    inline static void SetObjectLightIntensity(const QString& viewID,
                                               double intensity) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setObjectLightIntensity(viewID, intensity);
        UpdateScreen();
    }
    inline void setObjectLightIntensity(
            const QString& /*viewID*/,
            double /*intensity*/,
            bool /*triggerRender*/ = true) override {}

    /// Get light intensity for a specific object
    /// @param viewID The view ID of the target object
    /// @return Object's light intensity (falls back to global default)
    inline static double GetObjectLightIntensity(const QString& viewID) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getObjectLightIntensity(viewID);
        return 1.0;
    }
    inline double getObjectLightIntensity(
            const QString& /*viewID*/) const override {
        return 1.0;
    }

private:
    /**
     * @brief Internal 3D drawing method
     * @param CONTEXT Drawing context
     */
    [[deprecated("Phase B: use per-view vtkGLView::redraw()")]]
    static void Draw3D(CC_DRAW_CONTEXT& CONTEXT);

public:  // Main interface accessors
    /**
     * @brief Get 3D visualizer instance
     * @return Pointer to 3D visualizer
     */
    inline static ecvGenericVisualizer3D* GetVisualizer3D() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getVisualizer3D();
        return nullptr;
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getVisualizer2D();
        return nullptr;
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
        auto* av = ecvViewManager::instance().getEffectiveView();
        if (av) {
            QWidget* w = av->asWidget();
            if (w) return w;
        }
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            return dt->m_currentScreen;
        }
        return nullptr;
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
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            return dt->m_mainScreen;
        }
        return nullptr;
    }

    /**
     * @brief Set main screen widget
     * @param widget Screen widget to set as main
     */
    inline static void SetMainScreen(QWidget* widget) {
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            dt->m_mainScreen = widget;
        }
    }

    /**
     * @brief Get main window
     * @return Main window pointer
     */
    inline static QMainWindow* GetMainWindow() {
        return ecvViewManager::instance().mainWindow();
    }

    /**
     * @brief Set main window
     * @param win Main window to set
     */
    inline static void SetMainWindow(QMainWindow* win) {
        ecvViewManager::instance().setMainWindow(win);
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            dt->m_win = win;
        }
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
    inline static ccHObject* GetOwnDB() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (!dt) return nullptr;
        if (av && av != dt) return av->getOwnDB();
        return dt->m_winDBRoot;
    }

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
        return ecvViewManager::instance().globalDBRoot();
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
     * @brief Selectively redraw a single object (resets all flags first)
     * @param obj Object to redraw
     * @param only2D Redraw only 2D elements (default: false)
     * @param forceRedraw Force complete redraw (default: true)
     */
    static void RedrawObject(ccHObject* obj,
                             bool only2D = false,
                             bool forceRedraw = true);

    /**
     * @brief Selectively redraw multiple objects (resets all flags first)
     * @param objects Objects to redraw
     * @param only2D Redraw only 2D elements (default: false)
     * @param forceRedraw Force complete redraw (default: true)
     */
    static void RedrawObjects(std::initializer_list<ccHObject*> objects,
                              bool only2D = false,
                              bool forceRedraw = true);

    /**
     * @brief Get bounding box of all visible objects
     * @param box Output bounding box
     */
    static void GetVisibleObjectsBB(
            ccBBox& box, const ecvGenericGLDisplay* display = nullptr);

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
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) {
            return const_cast<ecvViewportParameters&>(
                           av->getViewportParameters())
                    .viewMat;
        }
        return ecvViewManager::instance()
                .resolveViewContext()
                .viewportParams.viewMat;
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
        ecvViewManager::instance().setRemoveAllFlag(state);
    }

    /**
     * @brief Transform camera view matrix
     * @param viewMat View transformation matrix
     */
    inline static void TransformCameraView(const ccGLMatrixd& viewMat) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->transformCameraView(viewMat);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->transformCameraProjection(projMat);
    }

    /**
     * @brief Virtual interface for projection transformation
     * @param projMat Projection matrix
     */
    inline virtual void transformCameraProjection(
            const ccGLMatrixd& projMat) { /* do nothing */ }

    static inline int GetDevicePixelRatio() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) {
            return av->getDevicePixelRatio();
        }
        QWidget* screen = GetCurrentScreen();
        if (screen) return static_cast<int>(screen->devicePixelRatio());
        QMainWindow* mw = GetMainWindow();
        return mw ? static_cast<int>(mw->devicePixelRatio()) : 1;
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
        QWidget* w = GetCurrentScreen();
        if (!w) return QRect();
        QRect screenRect = w->geometry();
        QPoint globalPosition = w->mapToGlobal(screenRect.topLeft());
        screenRect.setTopLeft(globalPosition);
        return screenRect;
    }
    inline static void SetScreenSize(int xw, int yw) {
        if (QWidget* w = GetCurrentScreen()) w->resize(QSize(xw, yw));
    }
    inline static void DoResize(int xw, int yw) { SetScreenSize(xw, yw); }
    inline static void DoResize(const QSize& size) {
        SetScreenSize(size.width(), size.height());
    }
    inline static QSize GetScreenSize() {
        QWidget* w = GetCurrentScreen();
        return w ? w->size() : QSize();
    }

    inline static void SetRenderWindowSize(int xw, int yw) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setRenderWindowSize(xw, yw);
    }
    inline virtual void setRenderWindowSize(int xw, int yw) { /* do nothing */ }

    inline static void FullScreen(bool state) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->fullScreen(state);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getCameraFocalDistance(viewport);
        return 100.0;
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCameraFocalDistance(focal_distance, viewport);
    }

    /**
     * @brief Virtual interface for setting focal distance
     * @param focal_distance Focal distance
     * @param viewport Viewport index
     */
    inline virtual void setCameraFocalDistance(
            double focal_distance, int viewport = 0) { /* do nothing */ }

    inline static void GetCameraPos(double* pos, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCameraPos(pos, viewport);
    }
    inline virtual void getCameraPos(double* pos,
                                     int viewport = 0) { /* do nothing */ }
    inline static void GetCameraFocal(double* focal, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCameraFocal(focal, viewport);
    }
    inline virtual void getCameraFocal(double* focal,
                                       int viewport = 0) { /* do nothing */ }
    inline static void GetCameraUp(double* up, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCameraUp(up, viewport);
    }
    virtual void getCameraUp(double* up, int viewport = 0) { /* do nothing */ }

    inline static void SetCameraPosition(const CCVector3d& pos,
                                         int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCameraPosition(pos, viewport);
    }
    inline virtual void setCameraPosition(const CCVector3d& pos,
                                          int viewport = 0) { /* do nothing */ }
    inline static void SetCameraPosition(const double* pos,
                                         const double* focal,
                                         const double* up,
                                         int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCameraPosition(pos, focal, up, viewport);
    }
    inline virtual void setCameraPosition(const double* pos,
                                          const double* focal,
                                          const double* up,
                                          int viewport = 0) { /* do nothing */ }
    inline static void SetCameraPosition(const double* pos,
                                         const double* up,
                                         int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCameraPosition(pos, up, viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCameraPosition(pos_x, pos_y, pos_z, view_x, view_y, view_z,
                                  up_x, up_y, up_z, viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCameraClip(clipPlanes, viewport);
    }
    virtual void getCameraClip(double* clipPlanes,
                               int viewport = 0) { /* do nothing */ }
    inline static void SetCameraClip(double znear,
                                     double zfar,
                                     int viewport = 0) {
        auto& ctx = ecvViewManager::instance().resolveViewContext();
        ctx.viewportParams.zNear = znear;
        ctx.viewportParams.zFar = zfar;
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) {
            auto vp = av->getViewportParameters();
            vp.zNear = znear;
            vp.zFar = zfar;
            av->setViewportParameters(vp);
        }
        if (dt) dt->setCameraClip(znear, zfar, viewport);
    }
    virtual void setCameraClip(double znear,
                               double zfar,
                               int viewport = 0) { /* do nothing */ }

    inline static void ResetCameraClippingRange(int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->resetCameraClippingRange(viewport);
    }
    inline virtual void resetCameraClippingRange(
            int viewport = 0) { /* do nothing */ }

    // set and get view angle in y direction
    inline static double GetCameraFovy(int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getCameraFovy(viewport);
        return 0;
    }
    inline virtual double getCameraFovy(int viewport = 0) {
        return 0; /* do nothing */
    }
    inline static void SetCameraFovy(double fovy, int viewport = 0) {
        auto& ctx = ecvViewManager::instance().resolveViewContext();
        ctx.viewportParams.fov_deg = static_cast<float>(fovy);
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) {
            auto vp = av->getViewportParameters();
            vp.fov_deg = static_cast<float>(fovy);
            av->setViewportParameters(vp);
        }
        if (dt) {
            dt->setCameraFovy(cloudViewer::DegreesToRadians(fovy), viewport);
        }
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->saveScreenshot(file);
    }
    inline virtual void saveScreenshot(
            const std::string& file) { /* do nothing */ }

    /** \brief Save or Load the current rendered camera parameters to disk or
     * current camera. \param[in] file the name of the param file
     */
    inline static void SaveCameraParameters(const std::string& file) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->saveCameraParameters(file);
    }
    inline virtual void saveCameraParameters(
            const std::string& file) { /* do nothing */ }

    inline static void LoadCameraParameters(const std::string& file) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->loadCameraParameters(file);
    }
    inline virtual void loadCameraParameters(
            const std::string& file) { /* do nothing */ }

    inline static void ShowOrientationMarker() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->showOrientationMarker();
        UpdateScreen();
    }
    inline virtual void showOrientationMarker() { /* do nothing */ }

    inline static void SetOrthoProjection(int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setOrthoProjection(viewport);
        UpdateScreen();
    }
    inline virtual void setOrthoProjection(int viewport = 0) { /* do nothing */
    }
    inline static void SetPerspectiveProjection(int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setPerspectiveProjection(viewport);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setUseVbos(useVbos);
    }
    inline virtual void setUseVbos(bool useVbos) { /* do nothing */ }

    /** \brief Set the ID of a cloud or shape to be used for LUT display
     * \param[in] id The id of the cloud/shape look up table to be displayed
     * The look up table is displayed by pressing 'u' in the PCLVisualizer */
    inline static void SetLookUpTableID(const std::string& viewID) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setLookUpTableID(viewID);
    }
    inline virtual void setLookUpTableID(
            const std::string& viewID) { /* do nothing */ }

    inline static void GetProjectionMatrix(double* projArray,
                                           int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getProjectionMatrix(projArray, viewport);
    }
    inline virtual void getProjectionMatrix(double* projArray,
                                            int viewport = 0) { /* do nothing */
    }
    inline static void GetViewMatrix(double* viewArray, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getViewMatrix(viewArray, viewport);
    }
    inline virtual void getViewMatrix(double* viewArray,
                                      int viewport = 0) { /* do nothing */ }

    inline static void SetViewMatrix(const ccGLMatrixd& viewMat,
                                     int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setViewMatrix(viewMat, viewport);
    }
    inline virtual void setViewMatrix(const ccGLMatrixd& viewMat,
                                      int viewport = 0) { /* do nothing */ }

    static bool HideShowEntities(const ccHObject* obj, bool visible);
    inline static bool HideShowEntities(const CC_DRAW_CONTEXT& CONTEXT) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->hideShowEntities(CONTEXT);
        return true;
    }
    inline bool hideShowEntities(const CC_DRAW_CONTEXT& CONTEXT) override {
        return true; /* do nothing */
    }
    static void HideShowEntities(const QStringList& viewIDs,
                                 ENTITY_TYPE hideShowEntityType,
                                 bool visibility = false);

    static void RemoveEntities(const ccHObject* obj);
    static void RemoveEntities(const QStringList& viewIDs,
                               ENTITY_TYPE removeEntityType);
    static void RemoveEntities(const CC_DRAW_CONTEXT& CONTEXT);
    inline void removeEntities(
            const CC_DRAW_CONTEXT& CONTEXT) override { /* do nothing */ }

    [[deprecated("Phase B: use per-view vtkGLView::redraw()")]]
    static void DrawBackground(CC_DRAW_CONTEXT& CONTEXT);
    static void Update2DLabel(bool immediateUpdate = false);
    QString pick2DLabel(int x, int y) override { return QString(); }
    static void Redraw2DLabel();

    static QString Pick3DItem(int x = -1, int y = -1) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->pick3DItem(x, y);
        return QString();
    }
    QString pick3DItem(int x = -1, int y = -1) override { return QString(); }
    static QString PickObject(double x = -1, double y = -1) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->pickObject(x, y);
        return QString();
    }
    QString pickObject(double x = -1, double y = -1) override {
        return QString();
    }

    static void FilterByEntityType(ccHObject::Container& labels,
                                   CV_CLASS_ENUM type);

    inline void setBackgroundColor(
            const CC_DRAW_CONTEXT& CONTEXT) override { /* do nothing */ }

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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->createViewPort(xmin, ymin, xmax, ymax, viewport);
    }
    inline virtual void createViewPort(double xmin,
                                       double ymin,
                                       double xmax,
                                       double ymax,
                                       int& viewport) { /* do nothing */ }

    inline static void ResetCameraViewpoint(const std::string& viewID) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->resetCameraViewpoint(viewID);
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

    /// Direct setters for viewport default point size / line width,
    /// without recursive entity updates or user messages.
    /// Used by per-view HotZone synchronization.
    static void SetViewportDefaultPointSize(float size);
    static void SetViewportDefaultLineWidth(float width);

    inline static void Toggle2Dviewer(bool state) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->toggle2Dviewer(state);
    }
    inline void toggle2Dviewer(bool state) override { /* do nothing */ }

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
        if (QWidget* w = GetCurrentScreen()) w->update();
        UpdateCamera();
    }
    static void UpdateScreen();
    inline static void ResetCamera(const ccBBox* bbox) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->resetCamera(bbox);
        UpdateScreen();
    }
    inline void resetCamera(const ccBBox* bbox) override { /* do nothing */ }
    inline static void ResetCamera() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->resetCamera();
        UpdateScreen();
    }
    inline void resetCamera() override { /* do nothing */ }
    inline static void UpdateCamera() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->updateCamera();
        UpdateScreen();
    }
    inline void updateCamera() override { /* do nothing */ }

    inline static void UpdateScene() {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->updateScene();
    }
    inline void updateScene() override { /* do nothing */ }

    inline static void SetAutoUpateCameraPos(bool state) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setAutoUpateCameraPos(state);
    }
    inline virtual void setAutoUpateCameraPos(bool state) { /* do nothing */ }

    /**
     * Get the current center of rotation
     */
    inline static void GetCenterOfRotation(double center[3]) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCenterOfRotation(center);
    }
    inline static void GetCenterOfRotation(CCVector3d& center) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->getCenterOfRotation(center.u);
    }
    inline virtual void getCenterOfRotation(double center[3]) { /* do nothing */
    }

    /**
     * Resets the center of rotation to the focal point.
     */
    inline static void ResetCenterOfRotation(int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->resetCenterOfRotation(viewport);
        UpdateScreen();
    }
    inline void resetCenterOfRotation(
            int viewport = 0) override { /* do nothing */ }

    /**
     * Set the center of rotation. For this to work,
     * one should have appropriate interaction style
     * and camera manipulators that use the center of rotation
     * They are setup correctly by default
     */
    inline static void SetCenterOfRotation(double x, double y, double z) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setCenterOfRotation(x, y, z);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getGLDepth(x, y);
        return 1.0;
    }
    inline virtual double getGLDepth(int x, int y) {
        return 1.0; /* do nothing */
    }

    inline static void ChangeOpacity(double opacity,
                                     const std::string& viewID,
                                     int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->changeOpacity(opacity, viewID, viewport);
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

    //! Updates the per-view active items list via activeItemsRef().
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->renderToImage(zoomFactor, renderOverlayItems, silent,
                                     viewport);
        return QImage();
    }
    inline virtual QImage renderToImage(int zoomFactor = 1,
                                        bool renderOverlayItems = false,
                                        bool silent = false,
                                        int viewport = 0) {
        return QImage(); /* do nothing */
    }

    inline static void SetScaleBarVisible(bool visible) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setScaleBarVisible(visible);
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
    //! button, etc.  Legacy wrapper using shared state (ecvViewManager).
    static void DrawClickableItems(int xStart, int& yStart);
    //! Phase M4 parameterized overload: accepts explicit per-view state so
    //! callers (vtkGLView) can bypass ScopedHotZoneRender.
    static void DrawClickableItems(
            int xStart,
            int& yStart,
            HotZone*& hotZone,
            std::vector<ecvClickableItem>& clickableItems,
            ecvGenericGLDisplay* display);
    static void RenderText(
            int x,
            int y,
            const QString& str,
            const QFont& font = QFont(),
            const ecvColor::Rgbub& color = ecvColor::defaultLabelBkgColor,
            const QString& id = "",
            ecvGenericGLDisplay* display = nullptr,
            double bkgAlpha = 0.0,
            const double* bkgColor = nullptr);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->toggleExclusiveFullScreen(state);
    }
    inline void toggleExclusiveFullScreen(
            bool state) override { /* do nothing */ }

    //! Returns whether the window is in exclusive full screen mode or not
    inline static bool ExclusiveFullScreen() {
        return ecvViewManager::instance()
                .resolveViewContext()
                .exclusiveFullscreen;
    }
    inline static bool ExclusiveFullScreen(ecvGenericGLDisplay* view) {
        if (view && view->viewContext())
            return view->viewContext()->exclusiveFullscreen;
        return ExclusiveFullScreen();
    }
    inline static void SetExclusiveFullScreenFlag(bool state) {
        ecvViewManager::instance().resolveViewContext().exclusiveFullscreen =
                state;
    }
    inline static void SetExclusiveFullScreenFlag(bool state,
                                                  ecvGenericGLDisplay* view) {
        if (view && view->viewContext()) {
            view->viewContext()->exclusiveFullscreen = state;
            return;
        }
        SetExclusiveFullScreenFlag(state);
    }
    // Legacy typo-compat alias
    inline static void SetExclusiveFullScreenFlage(bool state) {
        SetExclusiveFullScreenFlag(state);
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getPerspectiveState(viewport);
        return GetViewportParameters().perspectiveView;
    }
    inline virtual bool getPerspectiveState(int viewport = 0) const override {
        return GetViewportParameters().perspectiveView;
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setPivotVisibility(state);
    }
    virtual void setPivotVisibility(bool state) { /*do nothing here*/ }

    //! Returns pivot visibility
    inline static PivotVisibility GetPivotVisibility() {
        return GetPivotVisibility(
                ecvViewManager::instance().resolveViewContext());
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
        if (auto* dt = ecvViewManager::instance().displayTools())
            return dt->getParallelScale(viewport);
        return -1.0;
    }
    inline virtual double getParallelScale(int viewport = 0) { return -1.0; }

    // scale (in rad)
    inline static void SetParallelScale(double scale, int viewport = 0) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setParallelScale(scale, viewport);
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

    inline static void Deprecate3DLayer() {
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            dt->m_updateFBO = true;
        }
    }
    inline static void InvalidateViewport() {
        ecvViewManager::instance().resolveViewContext().validProjectionMatrix =
                false;
    }
    inline static void InvalidateVisualization() {
        ecvViewManager::instance().resolveViewContext().validModelviewMatrix =
                false;
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
        return ecvViewManager::instance()
                .resolveViewContext()
                .bubbleViewModeEnabled;
    }
    //! Set bubble-view f.o.v. (in degrees)
    static void SetBubbleViewFov(float fov_deg);

    //! Sets whether to display the coordinates of the point below the cursor
    //! position
    inline static void ShowCursorCoordinates(bool state) {
        ecvViewManager::instance().resolveViewContext().showCursorCoordinates =
                state;
    }
    //! Whether the coordinates of the point below the cursor position are
    //! displayed or not
    inline static bool CursorCoordinatesShown() {
        return ecvViewManager::instance()
                .resolveViewContext()
                .showCursorCoordinates;
    }

    //! Lock the rotation axis
    static void LockRotationAxis(bool state, const CCVector3d& axis);

    //! Returns whether the rotation axis is locaked or not
    inline static bool IsRotationAxisLocked() {
        return ecvViewManager::instance()
                .resolveViewContext()
                .rotationAxisLocked;
    }

    //! Returns the approximate 3D position of the clicked pixel
    static bool GetClick3DPos(int x, int y, CCVector3d& P3D);

    static void DrawPivot();

    // debug traces on screen
    //! Shows debug info on screen
    inline static void EnableDebugTrace(bool state) {
        ecvViewManager::instance().resolveViewContext().showDebugTraces = state;
    }

    //! Toggles debug info on screen
    inline static void ToggleDebugTrace() {
        auto& ctx = ecvViewManager::instance().resolveViewContext();
        ctx.showDebugTraces = !ctx.showDebugTraces;
    }

    using PickingParameters = ecvPickingParameters;

    //! Processes the clickable items
    /** \return true if an item has been clicked
     **/
    static bool ProcessClickableItems(int x, int y);

    //! Processes clickable items for a specific view (per-view point size /
    //! line width)
    /** \param localPointSize  per-view point size to modify (instead of global)
     *  \param localLineWidth  per-view line width to modify (instead of global)
     *  \return true if an item has been clicked
     **/
    static bool ProcessClickableItems(int x,
                                      int y,
                                      float* localPointSize,
                                      float* localLineWidth);

    //! Sets current camera 'zNear' coefficient
    /** zNear coef. is only used in perspective mode.
     **/
    static void SetZNearCoef(double coef);

    //! Starts picking process
    /** \param params picking parameters
     **/
    static void StartPicking(PickingParameters& params);
    static void StartPicking(ecvViewContext& ctx, PickingParameters& params);

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
        if (auto* dt = ecvViewManager::instance().displayTools()) {
            QFont f = ecvViewManager::instance().defaultFont();
            f.setPointSize(pixelSize);
            ecvViewManager::instance().setDefaultFont(f);
            dt->m_font = f;
        }
    }
    //! Returns current font size
    static int GetFontPointSize();
    //! Returns current font size for labels
    static int GetLabelFontPointSize();

    static void SetClickableItemsVisible(bool state) {
        ecvViewManager::instance().resolveViewContext().clickableItemsVisible =
                state;
    }
    static bool GetClickableItemsVisible() {
        return ecvViewManager::instance()
                .resolveViewContext()
                .clickableItemsVisible;
    }

    // takes rendering zoom into account!
    static QFont GetLabelDisplayFont();
    // takes rendering zoom into account!
    inline static QFont GetTextDisplayFont() {
        return ecvViewManager::instance().defaultFont();
    }

    static ENTITY_TYPE ConvertToEntityType(const CV_CLASS_ENUM& type);

    //! Default picking radius value
    static const int DefaultPickRadius = 5;

    //! Sets picking radius
    inline static void SetPickingRadius(int radius) {
        ecvViewManager::instance().resolveViewContext().pickRadius = radius;
    }
    //! Returns the current picking radius
    inline static int GetPickingRadius() {
        return ecvViewManager::instance().resolveViewContext().pickRadius;
    }

    //! Sets whether overlay entities (scale, tetrahedron, etc.) should be
    //! displayed or not
    static void DisplayOverlayEntities(bool state);

    //! Returns whether overlay entities (scale, tetrahedron, etc.) are
    //! displayed or not
    inline static bool OverlayEntitiesAreDisplayed() {
        return ecvViewManager::instance()
                .resolveViewContext()
                .displayOverlayEntities;
    }

protected:
    ecvDisplayTools() = default;
    //! register visualizer callback function
    virtual void registerVisualizer(QMainWindow* win,
                                    bool stereoMode = false) = 0;

    QWidget* m_currentScreen;
    QWidget* m_mainScreen;
    QMainWindow* m_win;

public:
    using ClickableItem = ecvClickableItem;

    //! Internal timer
    QElapsedTimer m_timer;

    //! Overridden display parameter
    ecvGui::ParamStruct m_overridenDisplayParameters;

    //! Whether display parameters are overidden for this window
    bool m_overridenDisplayParametersEnabled;

    //! Whether the display should be refreshed on next call to 'refresh'
    bool m_shouldBeRefreshed;

    //! Auto-refresh mode
    bool m_autoRefresh;

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

    //! View that triggered the current deferred pick (nullptr =
    //! resolve via active view from ecvViewManager).
    ecvGenericGLDisplay* m_pickingTargetView = nullptr;

public:  // event representation
    static bool USE_2D;
    static bool USE_VTK_PICK;

    //! Hot zone (may point to a per-widget HotZone or a fallback created here)
    HotZone* m_hotZone;

    QStringList m_diagStrings;

    static int Width() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        if (av) {
            QWidget* w = av->asWidget();
            return w ? w->width() : 0;
        }
        return 0;
    }
    static int Height() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        if (av) {
            QWidget* w = av->asWidget();
            return w ? w->height() : 0;
        }
        return 0;
    }
    static QSize size() { return QSize(Width(), Height()); }

    static int GlWidth() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) return av->glWidth();
        return ecvViewManager::instance()
                .resolveViewContext()
                .glViewport.width();
    }
    static int GlHeight() {
        auto* av = ecvViewManager::instance().getEffectiveView();
        auto* dt = ecvViewManager::instance().displayTools();
        if (av && av != dt) return av->glHeight();
        return ecvViewManager::instance()
                .resolveViewContext()
                .glViewport.height();
    }
    static QSize GlSize() { return QSize(GlWidth(), GlHeight()); }

    static void ClearBubbleView(ecvGenericGLDisplay* display = nullptr);

public slots:

    //! Reacts to the itemPickedFast signal
    void onItemPickedFast(ccHObject* pickedEntity,
                          int pickedItemIndex,
                          int x,
                          int y);

    void onPointPicking(const CCVector3& p, int index, const std::string& id);

    //! Performs standard picking at the last clicked mouse position.
    //! Uses m_pickingTargetView's context if set, otherwise shared state
    //! (ecvViewManager).
    void doPicking();

    //! Set the view that should provide state for the next deferred pick.
    void setPickingTargetView(ecvGenericGLDisplay* view) {
        m_pickingTargetView = view;
    }

    // called when receiving mouse wheel is rotated
    void onWheelEvent(float wheelDelta_deg);

public:
    // ================================================================
    // Phase 7a wave 2: Virtual overrides matching ecvGenericGLDisplay
    // ================================================================

    CCVector3d toVtkCoordinates(int x, int y, int z = 0) override;
    bool getClick3DPos(int x, int y, CCVector3d& pos) override;
    void setView(CC_VIEW_ORIENTATION orientation) override;
    CCVector3d getCurrentViewDir() const override;
    void setPivotPoint(const CCVector3d& P,
                       bool autoRedraw = true,
                       bool verbose = false) override;
    void setPivotVisibility(PivotVisibility vis) override;
    bool isRotationAxisLocked() const override;
    void lockRotationAxis(bool state, const CCVector3d& axis) override;
    void toggleDebugTrace() override;
    void update2DLabels(bool immediateUpdate = false) override;
    bool renderToFile(const QString& filename,
                      float zoomFactor = 1.0f,
                      bool dontScale = false) override;
    void removeBB(const QString& viewId) override;
    void removeBB(const ccGLDrawContext& context) override;
    void setExclusiveFullScreenFlag(bool state) override;

    // Phase 2.6 overrides
    void filterByEntityType(std::vector<ccHObject*>& entities,
                            CV_CLASS_ENUM type) override;
    void updateActiveItemsList(int x, int y, bool centerItems) override;
    double computeActualPixelSize() const override;
    void updateNamePoseRecursive() override;
    void showPivotSymbol(bool state) override;
    bool exclusiveFullScreen() const override;
    CCVector3d convertMousePositionToOrientation(int x, int y) override;
    bool processClickableItems(int x, int y) override;
    void updateZoom(float zoomFactor) override;
    void resizeGL(int w, int h) override;
    void setViewportDefaultPointSize(float size) override;
    void setViewportDefaultLineWidth(float width) override;
    void setZNearCoef(double coef) override;
    void setFov(float fov_deg) override;
    void setPointSizeOnView(float size) override;
    void rotateWithAxis(const CCVector2i& mousePos,
                        const CCVector3d& axis,
                        double angle_deg) override;
    void startPicking(
            PICKING_MODE mode, int x, int y, int w = 0, int h = 0) override;
    void redraw2DLabel() override;

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
    void pickCenterOfRotation();

    void labelmove2D(int x, int y, int dx, int dy);

    void mousePosChanged(const QPoint& pos);

    void pointPicked(double x, double y, double z);

    void cameraParamChanged();
};

// Q_DECLARE_OPERATORS_FOR_FLAGS for INTERACTION_FLAGS is now in
// ecvGenericGLDisplay.h (base class).  The type is the same through
// inheritance so no duplicate declaration is needed.
