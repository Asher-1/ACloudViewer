// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGLView.h"

#include <CVLog.h>
#include <ecvBBox.h>
#include <ecvDisplayTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericDisplayTools.h>
#include <ecvHObject.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <vtkActor.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "Tools/Common/ecvTools.h"
#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "VTKExtensions/Widgets/QVTKWidgetCustom.h"
#include "VtkDisplayTools.h"
#include "VtkVis.h"

int ecvGLView::s_nextWindowID = 1000;

ecvGLView::ecvGLView(QMainWindow* parent) : QObject(parent) {
    m_uniqueID = ++s_nextWindowID;
    m_title = QString("RenderView%1").arg(m_uniqueID);
    m_ctx.viewportParams.viewMat.toIdentity();
    m_ctx.viewportParams.setCameraCenter(CCVector3d(0.0, 0.0, 1.0));

    m_ctx.viewMatd.toIdentity();
    m_ctx.projMatd.toIdentity();
    m_timer.start();
}

ecvGLView::~ecvGLView() {
    emit aboutToClose(this);

    if (m_globalDBRoot) {
        m_globalDBRoot->removeFromDisplay_recursive(this);
    }

    ecvRepresentationManager::instance().removeRepresentationsForView(this);
    // Safety net: idempotent if prepareViewClose() already called
    // unregisterView
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
    m_displayTools = static_cast<Visualization::VtkDisplayTools*>(
            ecvViewManager::instance().displayTools());
    m_vtkWidget = new QVTKWidgetCustom(parent, m_displayTools, stereoMode);
    m_vtkWidget->setOwnerView(this);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    renderWindow->AddRenderer(renderer);

    auto interactorStyle =
            vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();

    m_visualizer3D = std::make_shared<Visualization::VtkVis>(
            renderer, renderWindow, interactorStyle, m_title.toStdString(),
            false);
    m_visualizer3D->setOwnerDisplay(this);

    m_vtkWidget->SetRenderWindow(renderWindow);
    m_visualizer3D->setupInteractor(m_vtkWidget->GetInteractor(),
                                    m_vtkWidget->GetRenderWindow());
    m_vtkWidget->initVtk(m_visualizer3D->getRenderWindowInteractor(), false);
    m_vtkWidget->setCustomInteractorStyle(
            m_visualizer3D->get3DInteractorStyle());
    m_visualizer3D->initialize();

    m_deferredPickingTimer.setSingleShot(true);
    m_deferredPickingTimer.setInterval(100);
    connect(&m_deferredPickingTimer, &QTimer::timeout, this, [this]() {
        m_displayTools->setPickingTargetView(this);
        m_displayTools->doPicking();
    });
}

// ================================================================
// Core overrides (existing)
// ================================================================

void ecvGLView::redraw(bool only2D, bool forceRedraw) {
    if (!m_visualizer3D || !m_vtkWidget) return;

    // Ensure effectiveCtx() resolves to THIS view's context for the
    // entire duration of drawing. Without this guard, any code calling
    // effectiveCtx() during a direct redraw (Qt resize, interaction)
    // would get the primary view's context, causing wrong viewport
    // dimensions for widgets/overlays in split windows.
    ecvViewManager::ScopedRenderOverride viewGuard(this);

    // Sync per-view glViewport from the actual widget so that
    // ComputeActualPixelSize() (which reads effectiveCtx().glViewport)
    // returns correct dimensions for this sub-window.
    const int dpr = static_cast<int>(m_vtkWidget->devicePixelRatioF());
    m_ctx.glViewport = QRect(0, 0, m_vtkWidget->width() * dpr,
                             m_vtkWidget->height() * dpr);

    // --- Build draw context from per-view state ---
    CC_DRAW_CONTEXT context;
    getContext(context);
    context.forceRedraw = forceRedraw;

    // --- Background ---
    context.drawingFlags = CC_DRAW_2D;
    if (m_ctx.interactionFlags &
        ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
        context.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }

    const ecvGui::ParamStruct& displayParams = getDisplayParameters();
    if (displayParams.drawBackgroundGradient) {
        const ecvColor::Rgbub& bkgCol2 = displayParams.backgroundCol;
        ecvColor::Rgbub bkgCol1;
        bkgCol1.r = 255 - displayParams.textDefaultCol.r;
        bkgCol1.g = 255 - displayParams.textDefaultCol.g;
        bkgCol1.b = 255 - displayParams.textDefaultCol.b;
        context.backgroundCol = bkgCol1;
        context.backgroundCol2 = bkgCol2;
        context.drawBackgroundGradient = true;
    } else {
        const ecvColor::Rgbub& bkgCol = displayParams.backgroundCol;
        context.backgroundCol = bkgCol;
        context.backgroundCol2 = bkgCol;
        context.drawBackgroundGradient = false;
    }

    m_vtkWidget->setBackgroundColor(
            ecvTools::TransFormRGB(context.backgroundCol),
            ecvTools::TransFormRGB(context.backgroundCol2),
            context.drawBackgroundGradient);

    // --- 3D pass ---
    if (!only2D && m_globalDBRoot) {
        context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
        m_globalDBRoot->draw(context);
    }
    if (!only2D && m_winDBRoot) {
        context.drawingFlags = CC_DRAW_3D | CC_DRAW_FOREGROUND;
        m_winDBRoot->draw(context);
    }

    // --- 2D foreground pass ---
    if (m_globalDBRoot) {
        context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
        m_globalDBRoot->draw(context);
    }
    if (m_winDBRoot) {
        context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
        m_winDBRoot->draw(context);
    }

    // --- Hot zone / clickable items ---
    // Phase B: the per-view hot zone is rendered by temporarily routing
    // DrawClickableItems through this view's VtkVis pipeline.
    {
        if (m_displayTools) {
            Visualization::VtkDisplayTools::ScopedHotZoneRender hzRender(
                    m_displayTools, m_visualizer3D, m_vtkWidget, m_hotZone,
                    m_ctx, m_clickableItems);
            hzRender.draw();
        }
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
    return m_ctx.viewportParams;
}

void ecvGLView::setViewportParameters(const ecvViewportParameters& params) {
    m_ctx.viewportParams = params;
}

void ecvGLView::setPerspectiveState(bool state, bool objectCenteredView) {
    m_ctx.viewportParams.perspectiveView = state;
    m_ctx.viewportParams.objectCenteredView = objectCenteredView;
}

bool ecvGLView::perspectiveView() const {
    return m_ctx.viewportParams.perspectiveView;
}

bool ecvGLView::objectCenteredView() const {
    return m_ctx.viewportParams.objectCenteredView;
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

QVTKWidgetCustom* ecvGLView::getVtkWidget() const { return m_vtkWidget.data(); }

Visualization::VtkVis* ecvGLView::getVisualizer3D() const {
    return m_visualizer3D.get();
}

// ================================================================
// New per-view overrides (Phase 1)
// ================================================================

void ecvGLView::getGLCameraParameters(ccGLCameraParameters& params) const {
    if (!m_vtkWidget) return;
    const int dpr = static_cast<int>(m_vtkWidget->devicePixelRatioF());
    params.viewport[0] = 0;
    params.viewport[1] = 0;
    params.viewport[2] = m_vtkWidget->width() * dpr;
    params.viewport[3] = m_vtkWidget->height() * dpr;
    params.perspective = m_ctx.viewportParams.perspectiveView;
    params.fov_deg = m_ctx.viewportParams.fov_deg;
    params.pixelSize = m_ctx.viewportParams.pixelSize;
    params.modelViewMat = m_ctx.viewMatd;
    params.projectionMat = m_ctx.projMatd;
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
        m_ctx.viewportParams.setPivotPoint(center, true);
        if (m_visualizer3D) {
            m_visualizer3D->resetCamera(box);
            m_visualizer3D->getRenderWindow()->Render();
        }
    } else {
        zoomGlobal();
    }
}

QRect ecvGLView::getGLViewport() const {
    return m_ctx.glViewport.isValid()
                   ? m_ctx.glViewport
                   : (m_vtkWidget ? QRect(0, 0, m_vtkWidget->width(),
                                          m_vtkWidget->height())
                                  : QRect());
}

int ecvGLView::glWidth() const { return getGLViewport().width(); }

int ecvGLView::glHeight() const { return getGLViewport().height(); }

int ecvGLView::getDevicePixelRatio() const {
    return m_vtkWidget ? static_cast<int>(m_vtkWidget->devicePixelRatio()) : 1;
}

void ecvGLView::setInteractionMode(INTERACTION_FLAGS flags) {
    m_ctx.interactionFlags = flags;
}

ecvGLView::INTERACTION_FLAGS ecvGLView::getInteractionMode() const {
    return m_ctx.interactionFlags;
}

void ecvGLView::setPickingMode(PICKING_MODE mode) {
    if (!m_ctx.pickingModeLocked) {
        m_ctx.pickingMode = mode;
    }
}

ecvGLView::PICKING_MODE ecvGLView::getPickingMode() const {
    return m_ctx.pickingMode;
}

void ecvGLView::getContext(ccGLDrawContext& context) const {
    ecvViewManager::instance().sharedGetContext(context, m_ctx);
    context.display = const_cast<ecvGLView*>(this);
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio =
                static_cast<float>(m_vtkWidget->devicePixelRatioF());
    }
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

void ecvGLView::invalidateViewport() { m_ctx.validProjectionMatrix = false; }

void ecvGLView::deprecate3DLayer() { m_shouldBeRefreshed = true; }

void ecvGLView::displayNewMessage(const QString& message,
                                  MessagePosition pos,
                                  bool append,
                                  int displayMaxDelay_sec,
                                  MessageType type) {
    if (!append) {
        m_messagesToDisplay.remove_if([type](const ecvMessageToDisplay& msg) {
            return msg.type == type;
        });
    }
    if (!message.isEmpty()) {
        ecvMessageToDisplay msg;
        msg.message = message;
        msg.messageValidity_sec = displayMaxDelay_sec;
        msg.position = pos;
        msg.type = type;
        m_messagesToDisplay.push_back(msg);
    }
    toBeRefreshed();
}

void ecvGLView::zoomGlobal() {
    if (!m_visualizer3D) return;

    ccBBox bbox;
    getVisibleObjectsBB(bbox);

    if (bbox.isValid()) {
        m_visualizer3D->resetCamera(&bbox);
        CCVector3d center = CCVector3d::fromArray(bbox.getCenter().u);
        m_ctx.viewportParams.setPivotPoint(center, true);
    } else {
        m_visualizer3D->resetCamera();
    }

    m_visualizer3D->getRenderWindow()->Render();
}

// ================================================================
// Phase 7a: Per-view VTK operation overrides
//
// All delegates route through the singleton's VtkDisplayTools which
// internally calls resolveVisualizer(context.display) to target the
// correct per-view VtkVis.  When Phase 7b removes the singleton,
// these will call m_visualizer3D methods directly.
// ================================================================

void ecvGLView::draw(const ccGLDrawContext& context, const ccHObject* obj) {
    if (m_displayTools) m_displayTools->draw(context, obj);
}

void ecvGLView::drawBBox(const ccGLDrawContext& context, const ccBBox* bbox) {
    if (m_displayTools) m_displayTools->drawBBox(context, bbox);
}

void ecvGLView::drawOrientedBBox(const ccGLDrawContext& context,
                                 const ecvOrientedBBox* obb) {
    if (m_displayTools) m_displayTools->drawOrientedBBox(context, obb);
}

void ecvGLView::updateMeshTextures(const ccGLDrawContext& context,
                                   const ccGenericMesh* mesh) {
    if (m_displayTools) m_displayTools->updateMeshTextures(context, mesh);
}

void ecvGLView::drawWidgets(const WIDGETS_PARAMETER& param) {
    if (m_displayTools) m_displayTools->drawWidgets(param);
}

void ecvGLView::removeWidgets(const WIDGETS_PARAMETER& param) {
    if (m_displayTools) m_displayTools->removeWidgets(param);
}

bool ecvGLView::hideShowEntities(const ccGLDrawContext& context) {
    if (m_displayTools) return m_displayTools->hideShowEntities(context);
    return true;
}

void ecvGLView::removeEntities(const ccGLDrawContext& context) {
    if (m_displayTools) m_displayTools->removeEntities(context);
}

void ecvGLView::changeEntityProperties(PROPERTY_PARAM& param) {
    if (m_displayTools) m_displayTools->changeEntityProperties(param);
}

void ecvGLView::updateCamera() {
    if (m_visualizer3D) m_visualizer3D->getRenderWindow()->Render();
}

void ecvGLView::updateScene() {
    if (m_visualizer3D) m_visualizer3D->getRenderWindow()->Render();
}

// -- Phase M1.3: Per-view picking and rendering --

QString ecvGLView::pick2DLabel(int x, int y) {
    if (m_visualizer2D) {
        return m_visualizer2D->pickItem(x, y).c_str();
    }
    return {};
}

QString ecvGLView::pick3DItem(int x, int y) {
    if (m_visualizer3D) {
        return m_visualizer3D->pickItem(x, y).c_str();
    }
    return {};
}

QString ecvGLView::pickObject(double x, double y) {
    if (m_visualizer3D) {
        vtkActor* pickedActor = m_visualizer3D->pickActor(x, y);
        if (pickedActor) {
            return m_visualizer3D->getIdByActor(pickedActor).c_str();
        }
    }
    return QStringLiteral("-1");
}

QImage ecvGLView::renderToImage(int zoomFactor,
                                bool renderOverlayItems,
                                bool silent,
                                int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->renderToImage(zoomFactor, renderOverlayItems,
                                             silent, viewport);
    }
    if (!silent)
        CVLog::Error("[ecvGLView::renderToImage] No 3D visualizer");
    return {};
}

void ecvGLView::resetCamera(const ccBBox* bbox) {
    if (m_visualizer3D) {
        m_visualizer3D->resetCamera(bbox);
        m_visualizer3D->getRenderWindow()->Render();
    }
}

void ecvGLView::resetCamera() {
    if (m_visualizer3D) {
        m_visualizer3D->resetCamera();
        m_visualizer3D->getRenderWindow()->Render();
    }
}

void ecvGLView::toggle2Dviewer(bool state) {
    if (m_visualizer3D) {
        m_visualizer3D->setInteractionMode(
                state ? Visualization::VtkVis::INTERACTION_MODE_2D
                      : Visualization::VtkVis::INTERACTION_MODE_3D);
    }
}

// ================================================================
// Phase 7a wave 2: Additional per-view virtual overrides
// ================================================================

CCVector3d ecvGLView::toVtkCoordinates(int x, int y, int z) {
    CCVector3d p(x * 1.0, y * 1.0, z * 1.0);
    p.y = glHeight() - p.y;
    p *= getDevicePixelRatio();
    return p;
}

bool ecvGLView::getClick3DPos(int x, int y, CCVector3d& pos) {
    return m_displayTools ? m_displayTools->getClick3DPos(x, y, pos) : false;
}

void ecvGLView::setView(CC_VIEW_ORIENTATION orientation) {
    if (m_displayTools) m_displayTools->setView(orientation);
}

CCVector3d ecvGLView::getCurrentViewDir() const {
    const double* M = m_ctx.viewportParams.viewMat.data();
    CCVector3d axis(-M[2], -M[6], -M[10]);
    axis.normalize();
    return axis;
}

void ecvGLView::setPivotPoint(const CCVector3d& P,
                              bool autoRedraw,
                              bool verbose) {
    if (m_displayTools) m_displayTools->setPivotPoint(P, autoRedraw, verbose);
}

void ecvGLView::setPivotVisibility(ecvGenericGLDisplay::PivotVisibility vis) {
    if (m_displayTools) m_displayTools->setPivotVisibility(vis);
}

void ecvGLView::setAutoPickPivotAtCenter(bool state) {
    if (m_ctx.autoPickPivotAtCenter != state) {
        m_ctx.autoPickPivotAtCenter = state;
        if (state) {
            m_ctx.autoPivotCandidate = CCVector3d(0, 0, 0);
        }
    }
}

void ecvGLView::resetCenterOfRotation(int viewport) {
    if (m_displayTools) m_displayTools->resetCenterOfRotation(viewport);
}

bool ecvGLView::isRotationAxisLocked() const {
    return m_ctx.rotationAxisLocked;
}

void ecvGLView::lockRotationAxis(bool state, const CCVector3d& axis) {
    m_ctx.rotationAxisLocked = state;
    m_ctx.lockedRotationAxis = axis;
    m_ctx.lockedRotationAxis.normalize();
}

void ecvGLView::toggleCameraOrientationWidget(bool state) {
    if (m_displayTools) m_displayTools->toggleCameraOrientationWidget(state);
}

void ecvGLView::toggleOrientationMarker(bool state) {
    if (m_displayTools) m_displayTools->toggleOrientationMarker(state);
}

void ecvGLView::toggleDebugTrace() {
    m_ctx.showDebugTraces = !m_ctx.showDebugTraces;
}

void ecvGLView::update2DLabels(bool immediateUpdate) {
    if (m_displayTools) m_displayTools->update2DLabels(immediateUpdate);
}

bool ecvGLView::renderToFile(const QString& filename,
                             float zoomFactor,
                             bool dontScale) {
    return m_displayTools ? m_displayTools->renderToFile(filename, zoomFactor,
                                                         dontScale)
                          : false;
}

void ecvGLView::removeBB(const QString& viewId) {
    if (m_displayTools) m_displayTools->removeBB(viewId);
}

void ecvGLView::removeBB(const ccGLDrawContext& context) {
    if (m_displayTools) m_displayTools->removeBB(context);
}

void ecvGLView::setExclusiveFullScreenFlag(bool state) {
    m_ctx.exclusiveFullscreen = state;
}

double ecvGLView::getObjectLightIntensity(const QString& viewID) const {
    return m_displayTools ? m_displayTools->getObjectLightIntensity(viewID)
                          : 1.0;
}

void ecvGLView::setObjectLightIntensity(const QString& viewID,
                                        double intensity) {
    if (m_displayTools)
        m_displayTools->setObjectLightIntensity(viewID, intensity);
}

double ecvGLView::getLightIntensity() const {
    return m_displayTools ? m_displayTools->getLightIntensity() : 1.0;
}

void ecvGLView::setLightIntensity(double intensity) {
    if (m_displayTools) m_displayTools->setLightIntensity(intensity);
}

void ecvGLView::getDataAxesGridProperties(const QString& viewID,
                                          AxesGridProperties& props,
                                          int viewport) const {
    if (m_displayTools)
        m_displayTools->getDataAxesGridProperties(viewID, props, viewport);
}

void ecvGLView::setDataAxesGridProperties(const QString& viewID,
                                          const AxesGridProperties& props,
                                          int viewport) {
    if (m_displayTools)
        m_displayTools->setDataAxesGridProperties(viewID, props, viewport);
}

void ecvGLView::filterByEntityType(std::vector<ccHObject*>& entities,
                                   CV_CLASS_ENUM type) {
    if (m_displayTools) m_displayTools->filterByEntityType(entities, type);
}

void ecvGLView::updateActiveItemsList(int x, int y, bool centerItems) {
    if (m_displayTools)
        m_displayTools->updateActiveItemsList(x, y, centerItems);
}

double ecvGLView::computeActualPixelSize() const {
    if (!m_ctx.viewportParams.perspectiveView) {
        return static_cast<double>(m_ctx.viewportParams.pixelSize /
                                   m_ctx.viewportParams.zoom);
    }

    int minScreenDim =
            std::min(m_ctx.glViewport.width(), m_ctx.glViewport.height());
    if (minScreenDim <= 0) return 1.0;

    double zoomEquivalentDist = (m_ctx.viewportParams.getCameraCenter() -
                                 m_ctx.viewportParams.getPivotPoint())
                                        .norm();

    float fov_deg = m_ctx.bubbleViewModeEnabled ? m_ctx.bubbleViewFov_deg
                                                : m_ctx.viewportParams.fov_deg;
    return zoomEquivalentDist *
           std::tan(cloudViewer::DegreesToRadians(
                   std::min(static_cast<double>(fov_deg), 75.0))) /
           minScreenDim;
}

void ecvGLView::updateNamePoseRecursive() {
    if (m_displayTools) m_displayTools->updateNamePoseRecursive();
}

void ecvGLView::showPivotSymbol(bool state) {
    if (state && !m_ctx.pivotSymbolShown &&
        m_ctx.viewportParams.objectCenteredView &&
        m_ctx.pivotVisibility != PIVOT_HIDE) {
        invalidateViewport();
        deprecate3DLayer();
    }
    m_ctx.pivotSymbolShown = state;
}

bool ecvGLView::exclusiveFullScreen() const {
    return m_ctx.exclusiveFullscreen;
}

CCVector3d ecvGLView::convertMousePositionToOrientation(int x, int y) {
    return m_displayTools
                   ? m_displayTools->convertMousePositionToOrientation(x, y)
                   : CCVector3d(0, 0, 0);
}

bool ecvGLView::processClickableItems(int x, int y) {
    return m_displayTools ? m_displayTools->processClickableItems(x, y) : false;
}

void ecvGLView::updateZoom(float zoomFactor) {
    if (m_displayTools) m_displayTools->updateZoom(zoomFactor);
}

void ecvGLView::resizeGL(int w, int h) {
    if (m_displayTools) m_displayTools->resizeGL(w, h);
}

void ecvGLView::setViewportDefaultPointSize(float size) {
    m_ctx.viewportParams.defaultPointSize = size;
}

void ecvGLView::setViewportDefaultLineWidth(float width) {
    m_ctx.viewportParams.defaultLineWidth = width;
}

void ecvGLView::setZNearCoef(double coef) {
    if (m_displayTools) m_displayTools->setZNearCoef(coef);
}

void ecvGLView::setFov(float fov_deg) {
    if (m_displayTools) m_displayTools->setFov(fov_deg);
}

void ecvGLView::setPointSizeOnView(float size) {
    if (m_displayTools) m_displayTools->setPointSizeOnView(size);
}

void ecvGLView::rotateWithAxis(const CCVector2i& mousePos,
                               const CCVector3d& axis,
                               double angle_deg) {
    if (m_displayTools)
        m_displayTools->rotateWithAxis(mousePos, axis, angle_deg, 0);
}

void ecvGLView::startPicking(PICKING_MODE mode, int x, int y, int w, int h) {
    if (m_displayTools) m_displayTools->startPicking(mode, x, y, w, h);
}

void ecvGLView::redraw2DLabel() {
    if (m_displayTools) m_displayTools->redraw2DLabel();
}

void ecvGLView::scheduleFullRedraw(int delayMs) {
    m_scheduledFullRedrawTime = m_timer.elapsed() + delayMs;
    if (!m_scheduleTimer.isActive()) {
        m_scheduleTimer.start(delayMs);
    }
}

void ecvGLView::startDeferredPicking() { m_deferredPickingTimer.start(); }
