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
#include <ecvViewportParameters.h>

#include <QElapsedTimer>
#include <QFont>
#include <QObject>
#include <QPoint>
#include <QTimer>
#include <list>
#include <memory>
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
/// After the Phase-1 refactoring this class holds all display state
/// that was formerly global in ecvDisplayTools:
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

    // ================================================================
    // VTK-specific accessors
    // ================================================================

    QVTKWidgetCustom* getVtkWidget() const { return m_vtkWidget; }
    Visualization::VtkVis* getVisualizer3D() const;
    Visualization::VtkVisPtr getVisualizer3DSP() const {
        return m_visualizer3D;
    }
    Visualization::ImageVisPtr getImageVis() const { return m_visualizer2D; }

    void zoomGlobal();

    // ================================================================
    // Per-view state <-> singleton synchronization
    //
    // When this view becomes the active view, pushStateToSingleton()
    // copies per-view state INTO the ecvDisplayTools singleton so that
    // all existing m_tools-> code sees the correct values.
    // When this view loses active status, pullStateFromSingleton()
    // saves any changes made via the singleton back into this view.
    // ================================================================

    void pushStateToSingleton() override;
    void pullStateFromSingleton() override;

    // ================================================================
    // Per-view state accessors
    // ================================================================

    const QPoint& lastMousePos() const { return m_lastMousePos; }
    void setLastMousePos(const QPoint& p) { m_lastMousePos = p; }

    bool mouseMoved() const { return m_mouseMoved; }
    void setMouseMoved(bool v) { m_mouseMoved = v; }

    bool mouseButtonPressed() const { return m_mouseButtonPressed; }
    void setMouseButtonPressed(bool v) { m_mouseButtonPressed = v; }

    bool clickableItemsVisible() const { return m_clickableItemsVisible; }
    void setClickableItemsVisible(bool v) { m_clickableItemsVisible = v; }

    ecvDisplayTools::HotZone* hotZone() const { return m_hotZone; }
    void setHotZone(ecvDisplayTools::HotZone* hz) { m_hotZone = hz; }

    float defaultPointSize() const {
        return m_viewportParams.defaultPointSize;
    }
    void setDefaultPointSize(float s) {
        m_viewportParams.defaultPointSize = s;
    }

    float defaultLineWidth() const {
        return m_viewportParams.defaultLineWidth;
    }
    void setDefaultLineWidth(float w) {
        m_viewportParams.defaultLineWidth = w;
    }

    bool bubbleViewModeEnabled() const { return m_bubbleViewModeEnabled; }

signals:
    void aboutToClose(ecvGLView* self);
    void viewActivated(ecvGLView* self);

protected:
    explicit ecvGLView(QMainWindow* parent);

private:
    void initVtkPipeline(QMainWindow* parent, bool stereoMode);

    // -- Identification --
    int m_uniqueID;
    QString m_title;

    // -- VTK pipeline (per-view) --
    QVTKWidgetCustom* m_vtkWidget = nullptr;
    Visualization::VtkVisPtr m_visualizer3D;
    Visualization::ImageVisPtr m_visualizer2D;

    // -- Scene DB --
    ccHObject* m_globalDBRoot = nullptr;
    ccHObject* m_winDBRoot = nullptr;

    // -- Viewport / Camera group --
    ecvViewportParameters m_viewportParams;
    QRect m_glViewport;
    ccGLMatrixd m_viewMatd;
    ccGLMatrixd m_projMatd;
    bool m_validModelviewMatrix = false;
    bool m_validProjectionMatrix = false;
    double m_cameraToBBCenterDist = 1.0;
    double m_bbHalfDiag = 1.0;
    bool m_bubbleViewModeEnabled = false;
    float m_bubbleViewFov_deg = 90.0f;
    ecvViewportParameters m_preBubbleViewParameters;

    // -- Interaction / Picking group --
    INTERACTION_FLAGS m_interactionFlags = MODE_TRANSFORM_CAMERA;
    PICKING_MODE m_pickingMode = DEFAULT_PICKING;
    bool m_pickingModeLocked = false;
    int m_pickRadius = 3;
    ccPolyline* m_rectPickingPoly = nullptr;
    bool m_allowRectangularEntityPicking = true;
    QTimer m_deferredPickingTimer;
    CCVector3 m_lastPickedPoint{0, 0, 0};
    int m_lastPointIndex = -1;
    QString m_lastPickedId;
    bool m_widgetClicked = false;
    bool m_ignoreMouseReleaseEvent = false;

    // -- Mouse / Touch group --
    QPoint m_lastMousePos;
    QPoint m_lastMouseMovePos;
    bool m_mouseMoved = false;
    bool m_mouseButtonPressed = false;
    bool m_touchInProgress = false;
    qreal m_touchBaseDist = 1.0;

    // -- HotZone / Clickable items group --
    ecvDisplayTools::HotZone* m_hotZone = nullptr;
    std::vector<ecvDisplayTools::ClickableItem> m_clickableItems;
    bool m_clickableItemsVisible = false;

    // -- Display group --
    std::list<ecvDisplayTools::MessageToDisplay> m_messagesToDisplay;
    QFont m_font;
    ecvGui::ParamStruct m_overriddenDisplayParameters;
    bool m_overriddenDisplayParametersEnabled = false;
    bool m_displayOverlayEntities = true;
    bool m_exclusiveFullscreen = false;
    bool m_showCursorCoordinates = false;
    bool m_showDebugTraces = false;

    // -- Refresh / Timer group --
    bool m_shouldBeRefreshed = false;
    QTimer m_scheduleTimer;
    qint64 m_scheduledFullRedrawTime = 0;
    bool m_autoRefresh = false;
    qint64 m_lastClickTime_ticks = 0;

    // -- Misc per-view --
    PivotVisibility m_pivotVisibility = PIVOT_SHOW_ON_MOVE;
    bool m_pivotSymbolShown = false;
    bool m_rotationAxisLocked = false;
    CCVector3d m_lockedRotationAxis{0, 0, 1};
    bool m_autoPickPivotAtCenter = true;
    CCVector3d m_autoPivotCandidate{0, 0, 0};

    // -- Custom light per-view --
    bool m_sunLightEnabled = true;
    bool m_customLightEnabled = false;
    float m_customLightPos[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    static int s_nextWindowID;
};
