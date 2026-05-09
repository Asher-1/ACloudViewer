// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvDisplayTools.h>
#include <ecvDisplayTypes.h>
#include <ecvGenericGLDisplay.h>
#include <ecvGuiParameters.h>
#include <ecvViewContext.h>
#include <ecvViewportParameters.h>

#include <QElapsedTimer>
#include <QFont>
#include <QObject>
#include <QPoint>

#include <vtkSmartPointer.h>

class vtkImplicitPlaneWidget2;
#include <QPointer>
#include <QTimer>
#include <list>
#include <memory>
#include <unordered_set>
#include <vector>

#include "qVTK.h"

class ccHObject;
class ccPolyline;
class QMainWindow;
class QVTKWidgetCustom;

namespace Visualization {
class VtkVis;
class VtkDisplayTools;
class ImageVis;
using VtkVisPtr = std::shared_ptr<VtkVis>;
using ImageVisPtr = std::shared_ptr<ImageVis>;
}  // namespace Visualization

/// Per-window 3D view — the **complete per-view state container**.
///
/// After the Phase-A refactoring this class holds all display state
/// that was formerly global in ecvDisplayTools inside `ecvViewContext m_ctx`:
///   - Viewport / camera / projection matrices
///   - Interaction & picking mode
///   - Mouse & touch state
///   - HotZone / clickable items
///   - Display parameters & overlay messages
///   - Refresh / schedule timers
///   - VTK pipeline (QVTKWidgetCustom, VtkVis, ImageVis)
///
/// Design references:
///   CloudCompare ccGLWindow / ccGLWindowInterface — per-window state
///   ParaView     pqRenderView                     — per-view vtkRenderWindow
///   MeshLab      GLArea                           — independent paintGL
class QVTK_ENGINE_LIB_API vtkGLView : public QObject,
                                      public ecvGenericGLDisplay {
    Q_OBJECT

public:
    static vtkGLView* Create(QMainWindow* parent, bool stereoMode = false);
    ~vtkGLView() override;

    // ================================================================
    // ecvGenericGLDisplay — core overrides
    // ================================================================

    int getUniqueID() const override { return m_uniqueID; }
    QString getTitle() const override { return m_title; }
    void redraw(bool only2D = false, bool forceRedraw = true) override;
    void refresh(bool only2D = false) override;
    void toBeRefreshed() override;
    const ecvViewportParameters& getViewportParameters() const override;
    void setViewportParameters(const ecvViewportParameters& params) override;
    void setPerspectiveState(bool state, bool objectCenteredView) override;
    bool perspectiveView() const override;
    bool objectCenteredView() const override;
    void setSceneDB(ccHObject* root) override;
    ccHObject* getSceneDB() override;
    ccHObject* getOwnDB() override;
    void addToOwnDB(ccHObject* obj, bool noDependency = true) override;
    void removeFromOwnDB(ccHObject* obj) override;
    QWidget* asWidget() override;
    const QWidget* asWidget() const override;
    bool hasOverriddenDisplayParameters() const override;
    void aboutToBeRemoved(ccDrawableObject* obj) override;

    // ================================================================
    // ecvGenericGLDisplay — new per-view overrides (Phase 1)
    // ================================================================

    void getGLCameraParameters(ccGLCameraParameters& params) const override;
    void getVisibleObjectsBB(ccBBox& box) const override;
    void updateConstellationCenterAndZoom(const ccBBox* box = nullptr) override;
    QRect getGLViewport() const override;
    int glWidth() const override;
    int glHeight() const override;
    int getDevicePixelRatio() const override;

    void setInteractionMode(INTERACTION_FLAGS flags) override;
    INTERACTION_FLAGS getInteractionMode() const override;
    void setPickingMode(PICKING_MODE mode) override;
    PICKING_MODE getPickingMode() const override;

    void getContext(ccGLDrawContext& context) const override;
    const ecvGui::ParamStruct& getDisplayParameters() const override;
    void setDisplayParameters(const ecvGui::ParamStruct& params,
                              bool thisWindowOnly = false) override;
    void drawClickableItems(int xStart, int& yStart) override;
    void invalidateViewport() override;
    void deprecate3DLayer() override;
    void displayNewMessage(const QString& message,
                           MessagePosition pos,
                           bool append = false,
                           int displayMaxDelay_sec = 2,
                           MessageType type = CUSTOM_MESSAGE) override;
    QFont textDisplayFont() const override { return m_font; }

    // ================================================================
    // Phase 7a: Per-view VTK operation overrides
    //
    // Delegate to VtkDisplayTools (which resolves per-view VtkVis
    // via resolveVisualizer(context.display)).  When Phase 7b
    // removes the singleton, these will use m_visualizer3D directly.
    // ================================================================

    void draw(const ccGLDrawContext& context, const ccHObject* obj) override;
    void drawBBox(const ccGLDrawContext& context, const ccBBox* bbox) override;
    void drawBBoxBatch(const ccGLDrawContext& context,
                       const std::vector<ccBBox>& boxes) override;
    void drawOrientedBBox(const ccGLDrawContext& context,
                          const ecvOrientedBBox* obb) override;
    void updateMeshTextures(const ccGLDrawContext& context,
                            const ccGenericMesh* mesh) override;
    void drawWidgets(const WIDGETS_PARAMETER& param) override;
    void removeWidgets(const WIDGETS_PARAMETER& param) override;
    bool hideShowEntities(const ccGLDrawContext& context) override;
    void removeEntities(const ccGLDrawContext& context) override;
    void changeEntityProperties(PROPERTY_PARAM& param) override;
    void updateCamera() override;
    void updateScene() override;
    void resetCamera(const ccBBox* bbox) override;
    void resetCamera() override;
    void toggle2Dviewer(bool state) override;

    // ================================================================
    // Phase 7a wave 2: Additional per-view virtual overrides
    // ================================================================

    CCVector3d toVtkCoordinates(int x, int y, int z = 0) override;
    bool getClick3DPos(int x, int y, CCVector3d& pos) override;
    void setView(CC_VIEW_ORIENTATION orientation) override;
    CCVector3d getCurrentViewDir() const override;
    void setPivotPoint(const CCVector3d& P,
                       bool autoRedraw = true,
                       bool verbose = false) override;
    void setPivotVisibility(PivotVisibility vis) override;
    void setAutoPickPivotAtCenter(bool state) override;
    void resetCenterOfRotation(int viewport = 0) override;
    bool isRotationAxisLocked() const override;
    void lockRotationAxis(bool state, const CCVector3d& axis) override;
    void toggleCameraOrientationWidget(bool state) override;
    void toggleOrientationMarker(bool state) override;
    void toggleDebugTrace() override;
    void update2DLabels(bool immediateUpdate = false) override;
    bool renderToFile(const QString& filename,
                      float zoomFactor = 1.0f,
                      bool dontScale = false) override;
    void removeBB(const QString& viewId) override;
    void removeBB(const ccGLDrawContext& context) override;
    void setExclusiveFullScreenFlag(bool state) override;
    double getObjectLightIntensity(const QString& viewID) const override;
    void setObjectLightIntensity(const QString& viewID,
                                 double intensity) override;
    double getLightIntensity() const override;
    void setLightIntensity(double intensity) override;
    void getDataAxesGridProperties(const QString& viewID,
                                   AxesGridProperties& props,
                                   int viewport = 0) const override;
    void setDataAxesGridProperties(const QString& viewID,
                                   const AxesGridProperties& props,
                                   int viewport = 0) override;

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

    // -- Per-view picking (Phase M1.3) --
    QString pick2DLabel(int x, int y) override;
    QString pick3DItem(int x = -1, int y = -1) override;
    QString pickObject(double x = -1, double y = -1) override;

    // -- Per-view rendering (Phase M1.3) --
    QImage renderToImage(int zoomFactor = 1,
                         bool renderOverlayItems = false,
                         bool silent = false,
                         int viewport = 0);

    // ================================================================
    // VTK-specific accessors
    // ================================================================

    void enableEDL(bool enable = true);
    bool isEDLEnabled() const { return m_edlEnabled; }

    void enableSliceMode(bool enable = true);
    bool isSliceModeEnabled() const { return m_sliceMode; }

    enum OrthoAxis { AXIS_XY, AXIS_XZ, AXIS_YZ };
    void setOrthoSliceCamera(OrthoAxis axis);

    QVTKWidgetCustom* getVtkWidget() const;
    Visualization::VtkVis* getVisualizer3D() const;
    QJsonObject saveLayoutCameraState() const override;
    void loadLayoutCameraState(const QJsonObject& cameraJson) override;
    Visualization::VtkVisPtr getVisualizer3DSP() const {
        return m_visualizer3D;
    }
    Visualization::ImageVisPtr getImageVis() const { return m_visualizer2D; }

    void zoomGlobal();

    // ================================================================
    // Per-view timer & scheduling (replaces singleton m_timer)
    // ================================================================

    qint64 elapsedMs() const override { return m_timer.elapsed(); }
    void scheduleFullRedraw(int delayMs) override;
    void startDeferredPicking();
    void syncVtkCameraToContext();
    QTimer& deferredPickingTimer() { return m_deferredPickingTimer; }
    void stopDeferredPicking() override { m_deferredPickingTimer.stop(); }
    void startDeferredPickingFor(ecvGenericGLDisplay* targetView) override {
        Q_UNUSED(targetView);
        startDeferredPicking();
    }

    // ================================================================
    // Per-view context — the single source of truth for view state.
    //
    // All per-view display state (viewport, camera matrices,
    // interaction mode, picking, mouse, display flags, light, pivot)
    // lives in m_ctx.  Accessors below are thin wrappers.
    // ================================================================

    const ecvViewContext& context() const { return m_ctx; }
    ecvViewContext& context() { return m_ctx; }

    ecvViewContext* viewContext() override { return &m_ctx; }
    const ecvViewContext* viewContext() const override { return &m_ctx; }

    // ================================================================
    // Per-view state accessors (delegate to m_ctx)
    // ================================================================

    const QPoint& lastMousePos() const { return m_ctx.lastMousePos; }
    void setLastMousePos(const QPoint& p) { m_ctx.lastMousePos = p; }

    bool mouseMoved() const { return m_ctx.mouseMoved; }
    void setMouseMoved(bool v) { m_ctx.mouseMoved = v; }

    bool mouseButtonPressed() const { return m_ctx.mouseButtonPressed; }
    void setMouseButtonPressed(bool v) { m_ctx.mouseButtonPressed = v; }

    bool clickableItemsVisible() const { return m_ctx.clickableItemsVisible; }
    void setClickableItemsVisible(bool v) { m_ctx.clickableItemsVisible = v; }

    ecvHotZone* hotZone() const { return m_hotZone; }
    ecvHotZone*& hotZoneRef() { return m_hotZone; }
    void setHotZone(ecvHotZone* hz) { m_hotZone = hz; }
    ccPolyline*& rectPickingPolyRef() { return m_rectPickingPoly; }
    std::list<ccInteractor*>& activeItemsRef() override {
        return m_activeItems;
    }

    ecvHotZone*& hotZonePtrRef() override { return m_hotZone; }

    std::vector<ecvClickableItem>& clickableItemsRef() override {
        return m_clickableItems;
    }

    float defaultPointSize() const {
        return m_ctx.viewportParams.defaultPointSize;
    }
    void setDefaultPointSize(float s) {
        m_ctx.viewportParams.defaultPointSize = s;
    }

    float defaultLineWidth() const {
        return m_ctx.viewportParams.defaultLineWidth;
    }
    void setDefaultLineWidth(float w) {
        m_ctx.viewportParams.defaultLineWidth = w;
    }

    bool bubbleViewModeEnabled() const { return m_ctx.bubbleViewModeEnabled; }

signals:
    void aboutToClose(vtkGLView* self);
    void viewActivated(vtkGLView* self);

    // -- Per-view picking signals (ParaView pqView pattern) --
    void itemPicked(ccHObject* entity,
                    unsigned subEntityID,
                    int x,
                    int y,
                    const CCVector3& P);
    void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);
    void fastPickingFinished();
    void pointPicked(double x, double y, double z);

    // -- Per-view camera signals --
    void viewMatRotated(const ccGLMatrixd& rotMat);
    void cameraDisplaced(float ddx, float ddy);
    void mouseWheelChanged(QWheelEvent* event);
    void mouseWheelRotated(float wheelDelta_deg);
    void perspectiveStateChanged();
    void baseViewMatChanged(const ccGLMatrixd& newViewMat);
    void pixelSizeChanged(float pixelSize);
    void fovChanged(float fov);
    void zNearCoefChanged(float coef);
    void pivotPointChanged(const CCVector3d&);
    void cameraPosChanged(const CCVector3d&);
    void cameraParamChanged();

    // -- Per-view transform signals --
    void translation(const CCVector3d& t);
    void rotation(const ccGLMatrixd& rotMat);

    // -- Per-view mouse signals --
    void leftButtonClicked(int x, int y);
    void rightButtonClicked(int x, int y);
    void doubleButtonClicked(int x, int y);
    void mouseMoved(int x, int y, Qt::MouseButtons buttons);
    void buttonReleased();
    void mousePosChanged(const QPoint& pos);

    // -- Per-view misc signals --
    void drawing3D();
    void filesDropped(const QStringList& filenames, bool displayDialog);
    void newLabel(ccHObject* obj);
    void exclusiveFullScreenToggled(bool exclusive);
    void autoPickPivot(bool state);
    void labelmove2D(int x, int y, int dx, int dy);

    // -- Selection (per-view emission, relayed to global via ecvViewManager) --
    void entitySelectionChanged(ccHObject* entity);
    void entitiesSelectionChanged(std::unordered_set<int> entIDs);

protected:
    explicit vtkGLView(QMainWindow* parent);

private:
    void initVtkPipeline(QMainWindow* parent, bool stereoMode);

    // -- Identification --
    int m_uniqueID;
    QString m_title;

    // -- VTK pipeline (per-view, not part of context) --
    QPointer<QVTKWidgetCustom> m_vtkWidget;
    Visualization::VtkVisPtr m_visualizer3D;
    Visualization::ImageVisPtr m_visualizer2D;
    Visualization::VtkDisplayTools* m_displayTools = nullptr;

    // -- Scene DB (not part of context) --
    ccHObject* m_globalDBRoot = nullptr;
    ccHObject* m_winDBRoot = nullptr;

    // ================================================================
    // Per-view context: ALL push/pull-able value-type state.
    // ================================================================
    ecvViewContext m_ctx;

    // ================================================================
    // UI artifacts that depend on ecvDisplayTools nested types.
    // These are pushed/pulled individually alongside m_ctx.
    // ================================================================

    ccPolyline* m_rectPickingPoly = nullptr;
    QTimer m_deferredPickingTimer;
    ecvHotZone* m_hotZone = nullptr;
    std::vector<ecvClickableItem> m_clickableItems;
    std::list<ccInteractor*> m_activeItems;
    std::list<ecvMessageToDisplay> m_messagesToDisplay;

    // -- Display (not pushed/pulled via context) --
    QFont m_font;
    ecvGui::ParamStruct m_overriddenDisplayParameters;
    bool m_overriddenDisplayParametersEnabled = false;

    struct CaptureModeOptions {
        bool enabled = false;
        float zoomFactor = 1.0f;
        bool renderOverlayItems = false;
    };
    CaptureModeOptions m_captureMode;

    // -- Refresh / Timer (not pushed/pulled) --
    bool m_shouldBeRefreshed = false;
    bool m_insideRedraw = false;
    bool m_edlEnabled = false;
    bool m_sliceMode = false;
    QTimer m_scheduleTimer;
    qint64 m_scheduledFullRedrawTime = 0;
    bool m_autoRefresh = false;
    QElapsedTimer m_timer;
    vtkSmartPointer<vtkImplicitPlaneWidget2> m_slicePlaneWidget;

    static int s_nextWindowID;
};
