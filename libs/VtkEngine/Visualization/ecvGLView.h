// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvGuiParameters.h>
#include <ecvViewContext.h>
#include <ecvViewportParameters.h>

#include <QElapsedTimer>
#include <QFont>
#include <QObject>
#include <QPoint>
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
class QVTK_ENGINE_LIB_API ecvGLView : public QObject,
                                      public ecvGenericGLDisplay {
    Q_OBJECT

public:
    static ecvGLView* Create(QMainWindow* parent, bool stereoMode = false);
    ~ecvGLView() override;

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

    // ================================================================
    // VTK-specific accessors
    // ================================================================

    QVTKWidgetCustom* getVtkWidget() const;
    Visualization::VtkVis* getVisualizer3D() const;
    Visualization::VtkVisPtr getVisualizer3DSP() const {
        return m_visualizer3D;
    }
    Visualization::ImageVisPtr getImageVis() const { return m_visualizer2D; }

    void zoomGlobal();

    // ================================================================
    // Per-view timer & scheduling (replaces singleton m_timer)
    // ================================================================

    qint64 elapsedMs() const { return m_timer.elapsed(); }
    void scheduleFullRedraw(int delayMs);
    void startDeferredPicking();
    QTimer& deferredPickingTimer() { return m_deferredPickingTimer; }

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

    bool clickableItemsVisible() const {
        return m_ctx.clickableItemsVisible;
    }
    void setClickableItemsVisible(bool v) {
        m_ctx.clickableItemsVisible = v;
    }

    ecvDisplayTools::HotZone* hotZone() const { return m_hotZone; }
    ecvDisplayTools::HotZone*& hotZoneRef() { return m_hotZone; }
    void setHotZone(ecvDisplayTools::HotZone* hz) { m_hotZone = hz; }
    ccPolyline*& rectPickingPolyRef() { return m_rectPickingPoly; }
    std::list<ccInteractor*>& activeItemsRef() override { return m_activeItems; }

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

    bool bubbleViewModeEnabled() const {
        return m_ctx.bubbleViewModeEnabled;
    }

signals:
    void aboutToClose(ecvGLView* self);
    void viewActivated(ecvGLView* self);

    // -- Per-view picking signals (ParaView pqView pattern) --
    void itemPicked(ccHObject* entity, unsigned subEntityID, int x, int y, const CCVector3& P);
    void itemPickedFast(ccHObject* entity, int subEntityID, int x, int y);
    void fastPickingFinished();
    void pointPicked(double x, double y, double z);

    // -- Per-view camera signals --
    void viewMatRotated(const ccGLMatrixd& rotMat);
    void cameraDisplaced(float ddx, float ddy);
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
    explicit ecvGLView(QMainWindow* parent);

private:
    void initVtkPipeline(QMainWindow* parent, bool stereoMode);

    // -- Identification --
    int m_uniqueID;
    QString m_title;

    // -- VTK pipeline (per-view, not part of context) --
    QPointer<QVTKWidgetCustom> m_vtkWidget;
    Visualization::VtkVisPtr m_visualizer3D;
    Visualization::ImageVisPtr m_visualizer2D;

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
    ecvDisplayTools::HotZone* m_hotZone = nullptr;
    std::vector<ecvDisplayTools::ClickableItem> m_clickableItems;
    std::list<ccInteractor*> m_activeItems;
    std::list<ecvDisplayTools::MessageToDisplay> m_messagesToDisplay;

    // -- Display (not pushed/pulled via context) --
    QFont m_font;
    ecvGui::ParamStruct m_overriddenDisplayParameters;
    bool m_overriddenDisplayParametersEnabled = false;

    // -- Refresh / Timer (not pushed/pulled) --
    bool m_shouldBeRefreshed = false;
    QTimer m_scheduleTimer;
    qint64 m_scheduledFullRedrawTime = 0;
    bool m_autoRefresh = false;
    QElapsedTimer m_timer;

    static int s_nextWindowID;
};
