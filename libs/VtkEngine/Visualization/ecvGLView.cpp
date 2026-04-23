// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGLView.h"

#include <cstring>

#include <ecvBBox.h>
#include <ecvDisplayTools.h>
#include <ecvDrawContext.h>
#include <ecvGenericDisplayTools.h>
#include <ecvHObject.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "VTKExtensions/Widgets/QVTKWidgetCustom.h"
#include "VtkDisplayTools.h"
#include "VtkVis.h"

int ecvGLView::s_nextWindowID = 1000;

ecvGLView::ecvGLView(QMainWindow* parent) : QObject(parent) {
    m_uniqueID = ++s_nextWindowID;
    m_title = QString("RenderView%1").arg(m_uniqueID);
    m_viewportParams.viewMat.toIdentity();
    m_viewportParams.setCameraCenter(CCVector3d(0.0, 0.0, 1.0));

    m_viewMatd.toIdentity();
    m_projMatd.toIdentity();
}

ecvGLView::~ecvGLView() {
    emit aboutToClose(this);

    if (m_globalDBRoot) {
        m_globalDBRoot->removeFromDisplay_recursive(this);
    }

    ecvRepresentationManager::instance().removeRepresentationsForView(this);
    ecvViewManager::instance().unregisterView(this);

    if (m_vtkWidget) {
        ecvGenericGLDisplay::UnregisterGLDisplay(m_vtkWidget);
    }

    delete m_hotZone;
    m_hotZone = nullptr;

    delete m_rectPickingPoly;
    m_rectPickingPoly = nullptr;

    if (m_winDBRoot) {
        delete m_winDBRoot;
        m_winDBRoot = nullptr;
    }
}

ecvGLView* ecvGLView::Create(QMainWindow* parent, bool stereoMode) {
    auto* view = new ecvGLView(parent);
    view->initVtkPipeline(parent, stereoMode);

    view->m_winDBRoot =
            new ccHObject(QString("DB.GLView_%1").arg(view->m_uniqueID));

    ecvGenericGLDisplay::RegisterGLDisplay(view->m_vtkWidget, view);
    ecvViewManager::instance().registerView(view);

    return view;
}

void ecvGLView::initVtkPipeline(QMainWindow* parent, bool stereoMode) {
    m_vtkWidget = new QVTKWidgetCustom(parent, ecvDisplayTools::TheInstance(),
                                       stereoMode);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    auto interactorStyle =
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();

    m_visualizer3D = std::make_shared<Visualization::VtkVis>(
            renderer, renderWindow, interactorStyle, m_title.toStdString(),
            false);

    m_vtkWidget->SetRenderWindow(renderWindow);
    m_visualizer3D->setupInteractor(m_vtkWidget->GetInteractor(),
                                    m_vtkWidget->GetRenderWindow());
    m_vtkWidget->initVtk(m_visualizer3D->getRenderWindowInteractor(), false);
    m_vtkWidget->setCustomInteractorStyle(
            m_visualizer3D->get3DInteractorStyle());
    m_visualizer3D->initialize();
}

// ================================================================
// Core overrides (existing)
// ================================================================

void ecvGLView::redraw(bool only2D, bool forceRedraw) {
    if (!m_visualizer3D || !m_vtkWidget) return;

    auto* primaryDT = static_cast<Visualization::VtkDisplayTools*>(
            ecvDisplayTools::TheInstance());
    Visualization::VtkDisplayTools::ScopedVisSwap swap(
            primaryDT, m_visualizer3D, m_vtkWidget);

    // Ensure all delegation methods (GetGLCameraParameters, etc.)
    // return THIS view's data during the draw pass, even if this view
    // is not the UI-active view (e.g. during redrawAll).
    ecvViewManager::ScopedRenderOverride renderOverride(this);

    CC_DRAW_CONTEXT context;
    ecvDisplayTools::GetContext(context);
    context.display = this;
    context.forceRedraw = forceRedraw;
    context.glW = m_vtkWidget->width();
    context.glH = m_vtkWidget->height();
    context.devicePixelRatio =
            static_cast<float>(m_vtkWidget->devicePixelRatioF());

    context.defaultPointSize =
            static_cast<unsigned char>(m_viewportParams.defaultPointSize);
    context.defaultLineWidth =
            static_cast<unsigned char>(m_viewportParams.defaultLineWidth);
    context.currentLineWidth = context.defaultLineWidth;

    ecvDisplayTools::DrawBackground(context);

    if (!only2D && m_globalDBRoot) {
        context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
        m_globalDBRoot->draw(context);
    }
    if (!only2D && m_winDBRoot) {
        context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
        m_winDBRoot->draw(context);
    }

    if (m_globalDBRoot) {
        context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
        m_globalDBRoot->draw(context);
    }
    if (m_winDBRoot) {
        context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
        m_winDBRoot->draw(context);
    }

    if (m_vtkWidget) {
        auto* savedHz = primaryDT->m_hotZone;
        bool savedVis = primaryDT->m_clickableItemsVisible;
        float savedPtSize = primaryDT->m_viewportParams.defaultPointSize;
        float savedLnWidth = primaryDT->m_viewportParams.defaultLineWidth;
        auto savedItems = primaryDT->m_clickableItems;

        if (!m_hotZone) {
            m_hotZone = new ecvDisplayTools::HotZone(m_vtkWidget);
        }

        primaryDT->m_hotZone = m_hotZone;
        primaryDT->m_clickableItemsVisible = m_clickableItemsVisible;
        primaryDT->m_viewportParams.defaultPointSize =
                m_viewportParams.defaultPointSize;
        primaryDT->m_viewportParams.defaultLineWidth =
                m_viewportParams.defaultLineWidth;

        int yStart = 0;
        ecvDisplayTools::DrawClickableItems(0, yStart);

        m_viewportParams.defaultPointSize =
                primaryDT->m_viewportParams.defaultPointSize;
        m_viewportParams.defaultLineWidth =
                primaryDT->m_viewportParams.defaultLineWidth;
        m_clickableItems = primaryDT->m_clickableItems;

        primaryDT->m_viewportParams.defaultPointSize = savedPtSize;
        primaryDT->m_viewportParams.defaultLineWidth = savedLnWidth;
        primaryDT->m_hotZone = savedHz;
        primaryDT->m_clickableItemsVisible = savedVis;
        primaryDT->m_clickableItems = savedItems;
    }

    m_visualizer3D->getRenderWindow()->Render();
    m_shouldBeRefreshed = false;
}

void ecvGLView::refresh(bool only2D) {
    if (m_shouldBeRefreshed) {
        redraw(only2D);
    }
}

void ecvGLView::toBeRefreshed() { m_shouldBeRefreshed = true; }

const ecvViewportParameters& ecvGLView::getViewportParameters() const {
    return m_viewportParams;
}

void ecvGLView::setViewportParameters(const ecvViewportParameters& params) {
    m_viewportParams = params;
}

void ecvGLView::setPerspectiveState(bool state, bool objectCenteredView) {
    m_viewportParams.perspectiveView = state;
    m_viewportParams.objectCenteredView = objectCenteredView;
}

bool ecvGLView::perspectiveView() const {
    return m_viewportParams.perspectiveView;
}

bool ecvGLView::objectCenteredView() const {
    return m_viewportParams.objectCenteredView;
}

void ecvGLView::setSceneDB(ccHObject* root) { m_globalDBRoot = root; }

ccHObject* ecvGLView::getSceneDB() { return m_globalDBRoot; }

ccHObject* ecvGLView::getOwnDB() { return m_winDBRoot; }

void ecvGLView::addToOwnDB(ccHObject* obj, bool noDependency) {
    if (!obj || !m_winDBRoot) return;
    if (noDependency) {
        m_winDBRoot->addChild(obj, ccHObject::DP_NONE);
    } else {
        m_winDBRoot->addChild(obj);
    }
}

void ecvGLView::removeFromOwnDB(ccHObject* obj) {
    if (m_winDBRoot) {
        m_winDBRoot->removeChild(obj);
    }
}

QWidget* ecvGLView::asWidget() { return m_vtkWidget; }

const QWidget* ecvGLView::asWidget() const { return m_vtkWidget; }

bool ecvGLView::hasOverriddenDisplayParameters() const {
    return m_overriddenDisplayParametersEnabled;
}

void ecvGLView::aboutToBeRemoved(ccDrawableObject* obj) { Q_UNUSED(obj); }

Visualization::VtkVis* ecvGLView::getVisualizer3D() const {
    return m_visualizer3D.get();
}

// ================================================================
// New per-view overrides (Phase 1)
// ================================================================

void ecvGLView::getGLCameraParameters(ccGLCameraParameters& params) const {
    if (!m_vtkWidget) return;
    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = m_vtkWidget->width();
    params.viewport[3] = m_vtkWidget->height();
    params.perspective = m_viewportParams.perspectiveView;
    params.fov_deg = m_viewportParams.fov_deg;
    params.pixelSize = m_viewportParams.pixelSize;
    // Model-view and projection matrices — use cached per-view values
    params.modelViewMat = m_viewMatd;
    params.projectionMat = m_projMatd;
}

void ecvGLView::getVisibleObjectsBB(ccBBox& box) const {
    if (m_globalDBRoot) {
        box = m_globalDBRoot->getDisplayBB_recursive(false, this);
    }
    if (m_winDBRoot) {
        ccBBox ownBox = m_winDBRoot->getDisplayBB_recursive(false, this);
        if (ownBox.isValid()) {
            box += ownBox;
        }
    }
}

void ecvGLView::updateConstellationCenterAndZoom(const ccBBox* box) {
    if (box && box->isValid()) {
        CCVector3d center = CCVector3d::fromArray(box->getCenter().u);
        m_viewportParams.setPivotPoint(center, true);
        if (m_visualizer3D) {
            m_visualizer3D->resetCamera(box);
        }
    } else {
        zoomGlobal();
    }
}

QRect ecvGLView::getGLViewport() const {
    return m_glViewport.isValid()
                   ? m_glViewport
                   : (m_vtkWidget
                              ? QRect(0, 0, m_vtkWidget->width(),
                                      m_vtkWidget->height())
                              : QRect());
}

int ecvGLView::glWidth() const { return getGLViewport().width(); }

int ecvGLView::glHeight() const { return getGLViewport().height(); }

int ecvGLView::getDevicePixelRatio() const {
    return m_vtkWidget ? static_cast<int>(m_vtkWidget->devicePixelRatio()) : 1;
}

void ecvGLView::setInteractionMode(INTERACTION_FLAGS flags) {
    m_interactionFlags = flags;
}

ecvGLView::INTERACTION_FLAGS ecvGLView::getInteractionMode() const {
    return m_interactionFlags;
}

void ecvGLView::setPickingMode(PICKING_MODE mode) {
    if (!m_pickingModeLocked) {
        m_pickingMode = mode;
    }
}

ecvGLView::PICKING_MODE ecvGLView::getPickingMode() const {
    return m_pickingMode;
}

void ecvGLView::getContext(ccGLDrawContext& context) const {
    ecvDisplayTools::GetContext(context);
    context.display = const_cast<ecvGLView*>(this);
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio =
                static_cast<float>(m_vtkWidget->devicePixelRatioF());
    }
    context.defaultPointSize =
            static_cast<unsigned char>(m_viewportParams.defaultPointSize);
    context.defaultLineWidth =
            static_cast<unsigned char>(m_viewportParams.defaultLineWidth);
    context.currentLineWidth = context.defaultLineWidth;
}

const ecvGui::ParamStruct& ecvGLView::getDisplayParameters() const {
    if (m_overriddenDisplayParametersEnabled) {
        return m_overriddenDisplayParameters;
    }
    return ecvGui::Parameters();
}

void ecvGLView::setDisplayParameters(const ecvGui::ParamStruct& params,
                                     bool thisWindowOnly) {
    if (thisWindowOnly) {
        m_overriddenDisplayParameters = params;
        m_overriddenDisplayParametersEnabled = true;
    } else {
        ecvGui::Set(params);
    }
}

void ecvGLView::drawClickableItems(int xStart, int& yStart) {
    Q_UNUSED(xStart);
    Q_UNUSED(yStart);
}

void ecvGLView::zoomGlobal() {
    if (!m_visualizer3D) return;

    ccBBox bbox;
    getVisibleObjectsBB(bbox);

    if (bbox.isValid()) {
        m_visualizer3D->resetCamera(&bbox);
        CCVector3d center = CCVector3d::fromArray(bbox.getCenter().u);
        m_viewportParams.setPivotPoint(center, true);
    } else {
        m_visualizer3D->resetCamera();
    }

    m_visualizer3D->getRenderWindow()->Render();
}

// ================================================================
// State synchronization with the singleton
// ================================================================

void ecvGLView::pushStateToSingleton() {
    auto* dt = ecvDisplayTools::TheInstance();
    if (!dt) return;

    // Interaction / picking
    dt->m_interactionFlags = m_interactionFlags;
    dt->m_pickingMode = m_pickingMode;
    dt->m_pickingModeLocked = m_pickingModeLocked;
    dt->m_pickRadius = m_pickRadius;

    // Mouse state
    dt->m_lastMousePos = m_lastMousePos;
    dt->m_lastMouseMovePos = m_lastMouseMovePos;
    dt->m_mouseMoved = m_mouseMoved;
    dt->m_mouseButtonPressed = m_mouseButtonPressed;
    dt->m_ignoreMouseReleaseEvent = m_ignoreMouseReleaseEvent;
    dt->m_widgetClicked = m_widgetClicked;

    // Touch
    dt->m_touchInProgress = m_touchInProgress;
    dt->m_touchBaseDist = m_touchBaseDist;

    // HotZone / clickable
    dt->m_hotZone = m_hotZone;
    dt->m_clickableItemsVisible = m_clickableItemsVisible;
    dt->m_clickableItems = m_clickableItems;

    // Display
    dt->m_displayOverlayEntities = m_displayOverlayEntities;
    dt->m_exclusiveFullscreen = m_exclusiveFullscreen;
    dt->m_showCursorCoordinates = m_showCursorCoordinates;
    dt->m_showDebugTraces = m_showDebugTraces;

    // Bubble view
    dt->m_bubbleViewModeEnabled = m_bubbleViewModeEnabled;
    dt->m_bubbleViewFov_deg = m_bubbleViewFov_deg;

    // Pivot
    dt->m_pivotVisibility = m_pivotVisibility;
    dt->m_autoPickPivotAtCenter = m_autoPickPivotAtCenter;

    // Viewport params — sync all fields so m_tools-> access is consistent.
    dt->m_viewportParams = m_viewportParams;

    // Rotation lock
    dt->m_rotationAxisLocked = m_rotationAxisLocked;
    dt->m_lockedRotationAxis = m_lockedRotationAxis;

    // Picking aux
    dt->m_last_point_index = m_lastPointIndex;
    dt->m_last_picked_id = m_lastPickedId;
    dt->m_rectPickingPoly = m_rectPickingPoly;
    dt->m_allowRectangularEntityPicking = m_allowRectangularEntityPicking;

    // Light
    dt->m_sunLightEnabled = m_sunLightEnabled;
    dt->m_customLightEnabled = m_customLightEnabled;
    memcpy(dt->m_customLightPos, m_customLightPos, sizeof(m_customLightPos));

    // Pivot shown
    dt->m_pivotSymbolShown = m_pivotSymbolShown;

    // Timer
    dt->m_lastClickTime_ticks = m_lastClickTime_ticks;
}

void ecvGLView::pullStateFromSingleton() {
    auto* dt = ecvDisplayTools::TheInstance();
    if (!dt) return;

    // Interaction / picking
    m_interactionFlags = dt->m_interactionFlags;
    m_pickingMode = dt->m_pickingMode;
    m_pickingModeLocked = dt->m_pickingModeLocked;
    m_pickRadius = dt->m_pickRadius;

    // Mouse state
    m_lastMousePos = dt->m_lastMousePos;
    m_lastMouseMovePos = dt->m_lastMouseMovePos;
    m_mouseMoved = dt->m_mouseMoved;
    m_mouseButtonPressed = dt->m_mouseButtonPressed;
    m_ignoreMouseReleaseEvent = dt->m_ignoreMouseReleaseEvent;
    m_widgetClicked = dt->m_widgetClicked;

    // Touch
    m_touchInProgress = dt->m_touchInProgress;
    m_touchBaseDist = dt->m_touchBaseDist;

    // HotZone / clickable
    m_clickableItemsVisible = dt->m_clickableItemsVisible;
    m_clickableItems = dt->m_clickableItems;

    // Display
    m_displayOverlayEntities = dt->m_displayOverlayEntities;
    m_exclusiveFullscreen = dt->m_exclusiveFullscreen;
    m_showCursorCoordinates = dt->m_showCursorCoordinates;
    m_showDebugTraces = dt->m_showDebugTraces;

    // Bubble view
    m_bubbleViewModeEnabled = dt->m_bubbleViewModeEnabled;
    m_bubbleViewFov_deg = dt->m_bubbleViewFov_deg;

    // Pivot
    m_pivotVisibility = dt->m_pivotVisibility;
    m_autoPickPivotAtCenter = dt->m_autoPickPivotAtCenter;

    // Viewport params — sync all fields back.
    m_viewportParams = dt->m_viewportParams;

    // Rotation lock
    m_rotationAxisLocked = dt->m_rotationAxisLocked;
    m_lockedRotationAxis = dt->m_lockedRotationAxis;

    // Picking aux
    m_lastPointIndex = dt->m_last_point_index;
    m_lastPickedId = dt->m_last_picked_id;
    m_rectPickingPoly = dt->m_rectPickingPoly;
    m_allowRectangularEntityPicking = dt->m_allowRectangularEntityPicking;

    // Light
    m_sunLightEnabled = dt->m_sunLightEnabled;
    m_customLightEnabled = dt->m_customLightEnabled;
    memcpy(m_customLightPos, dt->m_customLightPos, sizeof(m_customLightPos));

    // Pivot shown
    m_pivotSymbolShown = dt->m_pivotSymbolShown;

    // Timer
    m_lastClickTime_ticks = dt->m_lastClickTime_ticks;
}
