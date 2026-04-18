// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvGLView.h"

#include <ecvBBox.h>
#include <ecvDisplayTools.h>
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

void ecvGLView::redraw(bool only2D, bool forceRedraw) {
    if (!m_visualizer3D || !m_vtkWidget) return;

    // Temporarily swap the primary VtkDisplayTools' pipeline to this view's
    // VtkVis + QVTKWidget so all existing draw code renders into this view.
    auto* primaryDT = static_cast<Visualization::VtkDisplayTools*>(
            ecvDisplayTools::TheInstance());
    Visualization::VtkDisplayTools::ScopedVisSwap swap(
            primaryDT, m_visualizer3D, m_vtkWidget);

    CC_DRAW_CONTEXT context;
    ecvDisplayTools::GetContext(context);
    context.display = this;
    context.forceRedraw = forceRedraw;
    context.glW = m_vtkWidget->width();
    context.glH = m_vtkWidget->height();
    context.devicePixelRatio =
            static_cast<float>(m_vtkWidget->devicePixelRatioF());

    // Set background to match primary view
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

    // Draw per-view HotZone (point size / line width / bubble-view overlay).
    // Temporarily sync the per-widget state to the singleton so
    // DrawClickableItems uses this widget's HotZone.
    if (m_vtkWidget) {
        auto* savedHz = primaryDT->m_hotZone;
        bool savedVis = primaryDT->m_clickableItemsVisible;
        if (m_vtkWidget->localHotZone()) {
            primaryDT->m_hotZone = m_vtkWidget->localHotZone();
            primaryDT->m_clickableItemsVisible =
                    m_vtkWidget->localClickableItemsVisible();
        }
        int yStart = 0;
        ecvDisplayTools::DrawClickableItems(0, yStart);
        primaryDT->m_hotZone = savedHz;
        primaryDT->m_clickableItemsVisible = savedVis;
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

bool ecvGLView::hasOverriddenDisplayParameters() const { return false; }

void ecvGLView::aboutToBeRemoved(ccDrawableObject* obj) { Q_UNUSED(obj); }

Visualization::VtkVis* ecvGLView::getVisualizer3D() const {
    return m_visualizer3D.get();
}

void ecvGLView::zoomGlobal() {
    if (!m_visualizer3D) return;

    ccBBox bbox;
    if (m_globalDBRoot) {
        bbox += m_globalDBRoot->getBB_recursive();
    }
    if (m_winDBRoot) {
        bbox += m_winDBRoot->getBB_recursive();
    }

    if (bbox.isValid()) {
        m_visualizer3D->resetCamera(&bbox);
    } else {
        m_visualizer3D->resetCamera();
    }

    m_visualizer3D->getRenderWindow()->Render();
}
