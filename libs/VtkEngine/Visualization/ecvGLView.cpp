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
    m_vtkWidget->setOwnerView(this);

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

    // Phase B: self-contained draw — no ScopedVisSwap, no ScopedRenderOverride.
    // All state comes from m_ctx; rendering goes directly to this view's
    // VtkVis and QVTKWidgetCustom.

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
        auto* primaryDT = static_cast<Visualization::VtkDisplayTools*>(
                ecvDisplayTools::TheInstance());
        if (primaryDT) {
            Visualization::VtkDisplayTools::ScopedHotZoneRender hzRender(
                    primaryDT, m_visualizer3D, m_vtkWidget,
                    m_hotZone, m_ctx, m_clickableItems);
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
        }
    } else {
        zoomGlobal();
    }
}

QRect ecvGLView::getGLViewport() const {
    return m_ctx.glViewport.isValid()
                   ? m_ctx.glViewport
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
    ecvDisplayTools::GetContext(context, m_ctx);
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
// State synchronization with the singleton
// ================================================================

void ecvGLView::pushStateToSingleton() {
    // Phase E: no-op — views own their state; effectiveCtx() routes reads.
}

void ecvGLView::pullStateFromSingleton() {
    // Phase E: no-op — views own their state; effectiveCtx() routes reads.
}
