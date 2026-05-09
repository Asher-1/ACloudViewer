// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkGLView.h"

#include <CVLog.h>
#include <ecvBBox.h>
#include <ecvDisplayTypes.h>
#include <ecvDrawContext.h>
#include <ecvGenericDisplayTools.h>
#include <ecvGenericPointCloud.h>
#include <ecvGenericVisualizer3D.h>
#include <ecvHObject.h>
#include <ecvRenderingTools.h>
#include <ecvRepresentationManager.h>
#include <ecvUndoManager.h>
#include <ecvViewManager.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCameraPass.h>
#include <vtkDefaultPass.h>
#include <vtkEDLShading.h>
#include <vtkImplicitPlaneRepresentation.h>
#include <vtkImplicitPlaneWidget2.h>
#include <vtkActorCollection.h>
#include <vtkCallbackCommand.h>
#include <vtkMapper.h>
#include <vtkNew.h>
#include <vtkPlane.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkMatrix4x4.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "ImageVis.h"
#include "Tools/Common/ecvTools.h"
#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "VTKExtensions/Widgets/QVTKWidgetCustom.h"
#include "VtkDisplayTools.h"
#include "VtkVis.h"

int vtkGLView::s_nextWindowID = 1000;

vtkGLView::vtkGLView(QMainWindow* parent) : QObject(parent) {
    m_uniqueID = ++s_nextWindowID;
    m_title = QString("RenderView%1").arg(m_uniqueID);
    m_ctx.viewportParams.viewMat.toIdentity();
    m_ctx.viewportParams.setCameraCenter(CCVector3d(0.0, 0.0, 1.0));

    m_ctx.viewMatd.toIdentity();
    m_ctx.projMatd.toIdentity();
    m_timer.start();
}

vtkGLView::~vtkGLView() {
    emit aboutToClose(this);

    if (m_vtkWidget) {
        m_vtkWidget->setOwnerView(nullptr);
    }

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

vtkGLView* vtkGLView::Create(QMainWindow* parent, bool stereoMode) {
    auto* view = new vtkGLView(parent);
    view->initVtkPipeline(parent, stereoMode);

    view->m_winDBRoot =
            new ccHObject(QString("DB.GLView_%1").arg(view->m_uniqueID));

    ecvGenericGLDisplay::RegisterGLDisplay(view->m_vtkWidget, view);
    ecvViewManager::instance().registerView(view);

    if (Visualization::VtkVis* vis = view->getVisualizer3D()) {
        vis->setUndoManager(ecvViewManager::instance().undoManager());
    }

    return view;
}

void vtkGLView::initVtkPipeline(QMainWindow* parent, bool stereoMode) {
    m_displayTools = static_cast<Visualization::VtkDisplayTools*>(
            ecvViewManager::instance().displayTools());
    m_vtkWidget = new QVTKWidgetCustom(parent, m_displayTools, stereoMode);
    m_vtkWidget->setOwnerView(this);
    m_vtkWidget->connectSignalsTo(m_displayTools);

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

    if (ecvDisplayTools::USE_2D) {
        m_visualizer2D = std::make_shared<Visualization::ImageVis>(
                m_title.toStdString() + "_2D", false);
        m_visualizer2D->setRender(m_vtkWidget->getVtkRender());
        m_visualizer2D->setupInteractor(m_vtkWidget->GetInteractor(),
                                        m_vtkWidget->GetRenderWindow());
    }

    connect(m_visualizer3D.get(),
            &ecvGenericVisualizer3D::interactorPointPickedEvent, m_displayTools,
            &ecvDisplayTools::onPointPicking, Qt::UniqueConnection);

    if (!m_hotZone && m_vtkWidget) {
        m_hotZone = new ecvHotZone(m_vtkWidget);
    }

    m_deferredPickingTimer.setSingleShot(true);
    m_deferredPickingTimer.setInterval(100);
    connect(&m_deferredPickingTimer, &QTimer::timeout, this, [this]() {
        syncVtkCameraToContext();

        // Diagnostic: compare VTK's projection with ours
        static int s_vtkProjLogCount = 0;
        if (s_vtkProjLogCount < 3 && m_visualizer3D && m_globalDBRoot) {
            vtkRenderer* ren = m_visualizer3D->getCurrentRenderer();
            if (ren) {
                const int* renSz = ren->GetSize();
                CVLog::PrintDebug(
                        QString("[VtkProjTest] RendererSize=(%1,%2) "
                                "WidgetSize=(%3,%4) DPR=%5 "
                                "glViewport=(%6,%7)")
                                .arg(renSz[0])
                                .arg(renSz[1])
                                .arg(m_vtkWidget ? m_vtkWidget->width() : -1)
                                .arg(m_vtkWidget ? m_vtkWidget->height() : -1)
                                .arg(m_vtkWidget
                                             ? m_vtkWidget->devicePixelRatioF()
                                             : -1)
                                .arg(m_ctx.glViewport.width())
                                .arg(m_ctx.glViewport.height()));
                ccHObject::Container clouds;
                m_globalDBRoot->filterChildren(clouds, true,
                                               CV_TYPES::POINT_CLOUD, true);
                if (!clouds.empty()) {
                    auto* cloud = static_cast<ccGenericPointCloud*>(clouds[0]);
                    if (cloud->size() > 0) {
                        const CCVector3* pt = cloud->getPoint(0);
                        ren->SetWorldPoint(pt->x, pt->y, pt->z, 1.0);
                        ren->WorldToDisplay();
                        double* disp = ren->GetDisplayPoint();
                        ccGLCameraParameters cam;
                        getGLCameraParameters(cam);
                        CCVector3d ourProj;
                        cam.project(*pt, ourProj);
                        CVLog::PrintDebug(QString("[VtkProjTest] pt=(%1,%2,%3) "
                                                  "VTK_disp=(%4,%5,%6) "
                                                  "Our_proj=(%7,%8,%9)")
                                                  .arg(pt->x, 0, 'g', 6)
                                                  .arg(pt->y, 0, 'g', 6)
                                                  .arg(pt->z, 0, 'g', 6)
                                                  .arg(disp[0], 0, 'g', 6)
                                                  .arg(disp[1], 0, 'g', 6)
                                                  .arg(disp[2], 0, 'g', 6)
                                                  .arg(ourProj.x, 0, 'g', 6)
                                                  .arg(ourProj.y, 0, 'g', 6)
                                                  .arg(ourProj.z, 0, 'g', 6));
                        ++s_vtkProjLogCount;
                    }
                }
            }
        }

        m_displayTools->setPickingTargetView(this);
        m_displayTools->doPicking();
    });

    connect(&ecvRepresentationManager::instance(),
            &ecvRepresentationManager::representationChanged, this,
            [this](ecvViewRepresentation* rep) {
                if (rep && rep->getView() == this) {
                    redraw(false, true);
                }
            },
            Qt::QueuedConnection);
}

// ================================================================
// Core overrides (existing)
// ================================================================

void vtkGLView::redraw(bool only2D, bool forceRedraw) {
    if (!m_visualizer3D || !m_vtkWidget) return;

    // Prevent re-entrant redraws.  VTK's vtkRenderWindow::Render() is
    // not re-entrant; calling it while already inside a render causes
    // deadlocks or corrupted state.  This can happen when:
    //   - ensureRepresentation() emits representationChanged during draw
    //   - ecvRedrawScope destructor triggers redraw inside updateLabel()
    //   - any signal handler calls redraw() during the draw pipeline
    if (m_insideRedraw) {
        CVLog::Warning("[vtkGLView::redraw] RE-ENTRANT redraw blocked (view %d)", m_uniqueID);
        m_shouldBeRefreshed = true;
        return;
    }
    m_insideRedraw = true;

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

    // Sync VTK camera → m_ctx matrices so getGLCameraParameters()
    // returns valid projection/modelview for 2D label positioning.
    syncVtkCameraToContext();

    // --- 3D pass ---
    // CC_LIGHT_ENABLED is always set for VTK views because VTK handles
    // lighting in its own pipeline. Without this flag, normals display
    // is blocked by the MACRO_LightIsEnabled check in drawMeOnly.
    if (!only2D && m_globalDBRoot) {
        context.drawingFlags =
                CC_DRAW_3D | CC_DRAW_FOREGROUND | CC_LIGHT_ENABLED;
        context.visible = true;
        m_globalDBRoot->draw(context);
    }
    if (!only2D && m_winDBRoot) {
        context.drawingFlags =
                CC_DRAW_3D | CC_DRAW_FOREGROUND | CC_LIGHT_ENABLED;
        context.visible = true;
        m_winDBRoot->draw(context);
    }

    // --- 2D foreground pass ---
    // Reset context.visible: the 3D pass may leave it false (e.g. when the
    // last traversed entity was hidden), which would cascade through the
    // ancestorVisible logic and hide actors that were just shown in 3D.
    context.visible = true;
    context.drawingFlags = CC_DRAW_2D | CC_DRAW_FOREGROUND;
    if (m_ctx.interactionFlags &
        ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
        context.drawingFlags |= CC_VIRTUAL_TRANS_ENABLED;
    }
    if (m_globalDBRoot) m_globalDBRoot->draw(context);
    if (m_winDBRoot) m_winDBRoot->draw(context);

    // --- Color ramp ---
    ccRenderingTools::DrawColorRamp(context);

    // --- Scale bar ---
    if (m_ctx.displayOverlayEntities && m_vtkWidget) {
        m_vtkWidget->setScaleBarVisible(!m_ctx.viewportParams.perspectiveView);
    }

    // --- Messages overlay (per-view) ---
    if (m_ctx.displayOverlayEntities && !m_messagesToDisplay.empty()) {
        int currentTime = m_timer.elapsed() / 1000;

        // Remove expired messages and clean up their VTK text actors
        for (auto it = m_messagesToDisplay.begin();
             it != m_messagesToDisplay.end();) {
            if (currentTime > it->messageValidity_sec) {
                WIDGETS_PARAMETER rmParam(WIDGETS_TYPE::WIDGET_T2D,
                                          it->message);
                rmParam.context.display = this;
                removeWidgets(rmParam);
                it = m_messagesToDisplay.erase(it);
            } else {
                ++it;
            }
        }

        if (!m_messagesToDisplay.empty()) {
            QFont font = m_font;
            QFontMetrics fm(font);
            int margin = fm.height() / 4;
            int ll_currentHeight = m_ctx.glViewport.height() - 10;
            int uc_currentHeight = 10;

            for (const auto& message : m_messagesToDisplay) {
                switch (message.position) {
                    case ecvGenericGLDisplay::LOWER_LEFT_MESSAGE: {
                        ecvDisplayTools::RenderText(
                                10, ll_currentHeight, message.message, font,
                                ecvColor::defaultLabelBkgColor, "", this);
                        int messageHeight = fm.height();
                        ll_currentHeight -= (messageHeight + margin);
                    } break;
                    case ecvGenericGLDisplay::UPPER_CENTER_MESSAGE: {
                        QRect rect = fm.boundingRect(message.message);
                        int x = (m_ctx.glViewport.width() - rect.width()) / 2;
                        int y = uc_currentHeight + rect.height();
                        ecvDisplayTools::RenderText(
                                x, y, message.message, font,
                                ecvColor::defaultLabelBkgColor, "", this);
                        uc_currentHeight += (rect.height() + margin);
                    } break;
                    case ecvGenericGLDisplay::SCREEN_CENTER_MESSAGE: {
                        QFont newFont(font);
                        int fontSize =
                                ecvDisplayTools::GetOptimizedFontSize(12);
                        newFont.setPointSize(fontSize);
                        QRect rect = QFontMetrics(newFont).boundingRect(
                                message.message);
                        ecvDisplayTools::RenderText(
                                (m_ctx.glViewport.width() - rect.width()) / 2,
                                (m_ctx.glViewport.height() - rect.height()) / 2,
                                message.message, newFont,
                                ecvColor::defaultLabelBkgColor, "", this);
                    } break;
                }
            }
        }
    }

    // --- Hot zone / clickable items (Phase M4: direct parameterized call) ---
    if (m_ctx.displayOverlayEntities) {
        m_clickableItems.clear();
        int yStart = 0;
        ecvDisplayTools::DrawClickableItems(0, yStart, m_hotZone,
                                            m_clickableItems, this);
    }

    m_visualizer3D->getRenderWindow()->Render();
    if (m_vtkWidget) m_vtkWidget->update();

    m_insideRedraw = false;

    // If a redraw was requested while we were inside this redraw (e.g.
    // from representationChanged or ecvRedrawScope during the draw pipeline),
    // schedule it now via a queued single-shot so VTK finishes cleanly first.
    if (m_shouldBeRefreshed) {
        m_shouldBeRefreshed = false;
        QTimer::singleShot(0, this, [this]() { redraw(false, true); });
    } else {
        m_shouldBeRefreshed = false;
    }
}

void vtkGLView::refresh(bool only2D) {
    if (m_shouldBeRefreshed) {
        redraw(only2D);
    }
}

void vtkGLView::toBeRefreshed() { m_shouldBeRefreshed = true; }

const ecvViewportParameters& vtkGLView::getViewportParameters() const {
    return m_ctx.viewportParams;
}

void vtkGLView::setViewportParameters(const ecvViewportParameters& params) {
    m_ctx.viewportParams = params;
}

void vtkGLView::enableEDL(bool enable) {
    if (m_edlEnabled == enable) return;
    m_edlEnabled = enable;

    if (!m_visualizer3D) return;
    auto* renderer = m_visualizer3D->getCurrentRenderer();
    if (!renderer) return;

    if (enable) {
        vtkNew<vtkEDLShading> edlPass;
        vtkNew<vtkCameraPass> cameraPass;
        vtkNew<vtkDefaultPass> defaultPass;
        cameraPass->SetDelegatePass(defaultPass);
        edlPass->SetDelegatePass(cameraPass);
        renderer->SetPass(edlPass);
    } else {
        renderer->SetPass(nullptr);
    }
    updateScene();
}

static void SlicePlaneCallback(vtkObject* caller, unsigned long,
                               void* clientData, void*) {
    auto* widget = static_cast<vtkImplicitPlaneWidget2*>(caller);
    auto* rep = static_cast<vtkImplicitPlaneRepresentation*>(
            widget->GetRepresentation());
    auto* renderer = static_cast<vtkRenderer*>(clientData);
    if (!rep || !renderer) return;

    vtkNew<vtkPlane> plane;
    rep->GetPlane(plane);

    auto* actors = renderer->GetActors();
    actors->InitTraversal();
    while (auto* actor = actors->GetNextActor()) {
        if (auto* mapper = actor->GetMapper()) {
            mapper->RemoveAllClippingPlanes();
            mapper->AddClippingPlane(plane);
        }
    }
    renderer->GetRenderWindow()->Render();
}

void vtkGLView::enableSliceMode(bool enable) {
    if (m_sliceMode == enable) return;
    m_sliceMode = enable;

    if (!m_visualizer3D) return;
    auto* renderer = m_visualizer3D->getCurrentRenderer();
    if (!renderer) return;

    if (enable) {
        m_slicePlaneWidget = vtkSmartPointer<vtkImplicitPlaneWidget2>::New();
        vtkNew<vtkImplicitPlaneRepresentation> rep;
        rep->SetPlaceFactor(1.25);
        rep->SetOutlineTranslation(false);

        double bounds[6] = {-1, 1, -1, 1, -1, 1};
        renderer->ComputeVisiblePropBounds(bounds);
        if (bounds[0] > bounds[1]) {
            bounds[0] = -1; bounds[1] = 1;
            bounds[2] = -1; bounds[3] = 1;
            bounds[4] = -1; bounds[5] = 1;
        }
        rep->PlaceWidget(bounds);
        rep->SetNormal(0, 0, 1);

        double origin[3] = {(bounds[0] + bounds[1]) / 2.0,
                            (bounds[2] + bounds[3]) / 2.0,
                            (bounds[4] + bounds[5]) / 2.0};
        rep->SetOrigin(origin);

        m_slicePlaneWidget->SetInteractor(
                m_visualizer3D->getRenderWindow()->GetInteractor());
        m_slicePlaneWidget->SetRepresentation(rep);

        vtkNew<vtkCallbackCommand> callback;
        callback->SetCallback(SlicePlaneCallback);
        callback->SetClientData(renderer);
        m_slicePlaneWidget->AddObserver(vtkCommand::InteractionEvent,
                                        callback);
        m_slicePlaneWidget->On();
    } else {
        if (m_slicePlaneWidget) {
            m_slicePlaneWidget->Off();
            m_slicePlaneWidget = nullptr;
        }
        auto* actors = renderer->GetActors();
        actors->InitTraversal();
        while (auto* actor = actors->GetNextActor()) {
            if (auto* mapper = actor->GetMapper()) {
                mapper->RemoveAllClippingPlanes();
            }
        }
    }
    updateScene();
}

void vtkGLView::setOrthoSliceCamera(OrthoAxis axis) {
    if (!m_visualizer3D) return;
    auto* camera = m_visualizer3D->getCurrentRenderer()
                           ? m_visualizer3D->getCurrentRenderer()->GetActiveCamera()
                           : nullptr;
    if (!camera) return;

    setPerspectiveState(false, true);

    double bounds[6] = {-1, 1, -1, 1, -1, 1};
    m_visualizer3D->getCurrentRenderer()->ComputeVisiblePropBounds(bounds);
    double cx = (bounds[0] + bounds[1]) / 2.0;
    double cy = (bounds[2] + bounds[3]) / 2.0;
    double cz = (bounds[4] + bounds[5]) / 2.0;
    double dx = bounds[1] - bounds[0];
    double dy = bounds[3] - bounds[2];
    double dz = bounds[5] - bounds[4];
    double maxDim = std::max({dx, dy, dz});
    if (maxDim < 1e-6) maxDim = 2.0;

    camera->SetFocalPoint(cx, cy, cz);
    switch (axis) {
        case AXIS_XY:
            camera->SetPosition(cx, cy, cz + maxDim * 2);
            camera->SetViewUp(0, 1, 0);
            break;
        case AXIS_XZ:
            camera->SetPosition(cx, cy + maxDim * 2, cz);
            camera->SetViewUp(0, 0, 1);
            break;
        case AXIS_YZ:
            camera->SetPosition(cx + maxDim * 2, cy, cz);
            camera->SetViewUp(0, 0, 1);
            break;
    }
    camera->SetParallelScale(maxDim * 0.6);
    m_visualizer3D->getCurrentRenderer()->ResetCameraClippingRange();
    updateScene();
}

void vtkGLView::setPerspectiveState(bool state, bool objectCenteredView) {
    m_ctx.viewportParams.perspectiveView = state;
    m_ctx.viewportParams.objectCenteredView = objectCenteredView;
}

bool vtkGLView::perspectiveView() const {
    return m_ctx.viewportParams.perspectiveView;
}

bool vtkGLView::objectCenteredView() const {
    return m_ctx.viewportParams.objectCenteredView;
}

void vtkGLView::setSceneDB(ccHObject* root) {
    m_globalDBRoot = root;
    auto& vm = ecvViewManager::instance();
    if (!vm.globalDBRoot() && root) {
        vm.setGlobalDBRoot(root);
    }
}

ccHObject* vtkGLView::getSceneDB() { return m_globalDBRoot; }

ccHObject* vtkGLView::getOwnDB() { return m_winDBRoot; }

void vtkGLView::addToOwnDB(ccHObject* obj, bool noDependency) {
    if (!obj || !m_winDBRoot) return;
    if (noDependency) {
        m_winDBRoot->addChild(obj, ccHObject::DP_NONE);
    } else {
        m_winDBRoot->addChild(obj);
    }
    ecvViewManager::instance().invalidateLabelCache();
}

void vtkGLView::removeFromOwnDB(ccHObject* obj) {
    if (m_winDBRoot) {
        m_winDBRoot->removeChild(obj);
        ecvViewManager::instance().invalidateLabelCache();
    }
}

QWidget* vtkGLView::asWidget() { return m_vtkWidget; }

const QWidget* vtkGLView::asWidget() const { return m_vtkWidget; }

bool vtkGLView::hasOverriddenDisplayParameters() const {
    return m_overriddenDisplayParametersEnabled;
}

void vtkGLView::aboutToBeRemoved(ccDrawableObject* obj) { Q_UNUSED(obj); }

QVTKWidgetCustom* vtkGLView::getVtkWidget() const { return m_vtkWidget.data(); }

Visualization::VtkVis* vtkGLView::getVisualizer3D() const {
    return m_visualizer3D.get();
}

QJsonObject vtkGLView::saveLayoutCameraState() const {
    if (!m_visualizer3D) return {};
    return m_visualizer3D->saveCameraToJson(0);
}

void vtkGLView::loadLayoutCameraState(const QJsonObject& cameraJson) {
    if (!m_visualizer3D || cameraJson.isEmpty()) return;
    m_visualizer3D->loadCameraFromJson(cameraJson, 0);
}

// ================================================================
// New per-view overrides (Phase 1)
// ================================================================

void vtkGLView::syncVtkCameraToContext() {
    if (!m_visualizer3D) return;
    vtkCamera* cam = m_visualizer3D->getVtkCamera();
    if (!cam) return;
    vtkRenderer* ren = m_visualizer3D->getCurrentRenderer();
    if (!ren) return;

    // Use VTK's composite projection transform matrix with normalized depth
    // range (-1, 1), which is exactly what VTK's WorldToDisplay() uses
    // internally. This guarantees our projection matches VTK's rendering.
    double aspect = ren->GetTiledAspectRatio();
    vtkMatrix4x4* compositeMat =
            cam->GetCompositeProjectionTransformMatrix(aspect, -1, 1);

    // VTK stores row-major. Eigen interprets as column-major → transpose.
    // Then transposeInPlace → back to VTK's row-major form.
    // Our Project() treats matrices as column-major (OpenGL style), accessing
    // m[col*4+row]. After the transpose dance, the Eigen data layout matches
    // what VTK would produce for column-vector math.
    Eigen::Matrix4d composite = Eigen::Matrix4d(compositeMat->GetData());
    composite.transposeInPlace();

    // Store composite as projection, identity as modelview.
    // Project() computes: Proj * MV * P. With MV=I, result = Composite * P.
    ccGLMatrixd identity;
    identity.toIdentity();
    std::memcpy(m_ctx.viewMatd.data(), identity.data(), 16 * sizeof(double));
    std::memcpy(m_ctx.projMatd.data(), composite.data(), 16 * sizeof(double));
    m_ctx.validModelviewMatrix = true;
    m_ctx.validProjectionMatrix = true;

    // Sync perspective state
    bool vtkParallel = (cam->GetParallelProjection() != 0);
    m_ctx.viewportParams.perspectiveView = !vtkParallel;
    if (vtkParallel) {
        double ps = cam->GetParallelScale();
        int h = ren->GetSize()[1];
        if (h > 0) {
            m_ctx.viewportParams.pixelSize = static_cast<float>(2.0 * ps / h);
        }
    } else {
        m_ctx.viewportParams.fov_deg = static_cast<float>(cam->GetViewAngle());
    }

    // Viewport from renderer's actual size
    const int* sz = ren->GetSize();
    m_ctx.glViewport = QRect(0, 0, sz[0], sz[1]);
}

void vtkGLView::getGLCameraParameters(ccGLCameraParameters& params) const {
    if (!m_vtkWidget) return;

    // If matrices haven't been synced yet, do a live sync from VTK camera.
    if (!m_ctx.validModelviewMatrix || !m_ctx.validProjectionMatrix) {
        const_cast<vtkGLView*>(this)->syncVtkCameraToContext();
    }

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

void vtkGLView::getVisibleObjectsBB(ccBBox& box) const {
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

void vtkGLView::updateConstellationCenterAndZoom(const ccBBox* box) {
    if (box && box->isValid()) {
        CCVector3d center = CCVector3d::fromArray(box->getCenter().u);
        m_ctx.viewportParams.setPivotPoint(center, true);
        m_ctx.autoPivotCandidate = center;
        if (m_visualizer3D) {
            m_visualizer3D->resetCamera(box);
            if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                        m_visualizer3D.get())) {
                vis->setCenterOfRotation(center.x, center.y, center.z);
            }
            m_visualizer3D->getRenderWindow()->Render();
            if (m_vtkWidget) m_vtkWidget->update();
        }
    } else {
        zoomGlobal();
    }
}

QRect vtkGLView::getGLViewport() const {
    return m_ctx.glViewport.isValid()
                   ? m_ctx.glViewport
                   : (m_vtkWidget ? QRect(0, 0, m_vtkWidget->width(),
                                          m_vtkWidget->height())
                                  : QRect());
}

int vtkGLView::glWidth() const { return getGLViewport().width(); }

int vtkGLView::glHeight() const { return getGLViewport().height(); }

int vtkGLView::getDevicePixelRatio() const {
    return m_vtkWidget ? static_cast<int>(m_vtkWidget->devicePixelRatio()) : 1;
}

void vtkGLView::setInteractionMode(INTERACTION_FLAGS flags) {
    m_ctx.interactionFlags = flags;
}

vtkGLView::INTERACTION_FLAGS vtkGLView::getInteractionMode() const {
    return m_ctx.interactionFlags;
}

void vtkGLView::setPickingMode(PICKING_MODE mode) {
    if (!m_ctx.pickingModeLocked) {
        m_ctx.pickingMode = mode;
    }
}

vtkGLView::PICKING_MODE vtkGLView::getPickingMode() const {
    return m_ctx.pickingMode;
}

void vtkGLView::getContext(ccGLDrawContext& context) const {
    ecvViewManager::instance().sharedGetContext(context, m_ctx);
    context.display = const_cast<vtkGLView*>(this);
    if (m_vtkWidget) {
        context.glW = m_vtkWidget->width();
        context.glH = m_vtkWidget->height();
        context.devicePixelRatio =
                static_cast<float>(m_vtkWidget->devicePixelRatioF());
    }
}

const ecvGui::ParamStruct& vtkGLView::getDisplayParameters() const {
    if (m_overriddenDisplayParametersEnabled) {
        return m_overriddenDisplayParameters;
    }
    return ecvGui::Parameters();
}

void vtkGLView::setDisplayParameters(const ecvGui::ParamStruct& params,
                                     bool thisWindowOnly) {
    if (thisWindowOnly) {
        m_overriddenDisplayParameters = params;
        m_overriddenDisplayParametersEnabled = true;
    } else {
        ecvGui::Set(params);
    }
}

void vtkGLView::drawClickableItems(int xStart, int& yStart) {
    Q_UNUSED(xStart);
    Q_UNUSED(yStart);
}

void vtkGLView::invalidateViewport() { m_ctx.validProjectionMatrix = false; }

void vtkGLView::deprecate3DLayer() { m_shouldBeRefreshed = true; }

void vtkGLView::displayNewMessage(const QString& message,
                                  MessagePosition pos,
                                  bool append,
                                  int displayMaxDelay_sec,
                                  MessageType type) {
    if (message.isEmpty()) {
        if (!append) {
            for (auto it = m_messagesToDisplay.begin();
                 it != m_messagesToDisplay.end();) {
                if (it->position == pos) {
                    WIDGETS_PARAMETER rmParam(WIDGETS_TYPE::WIDGET_T2D,
                                              it->message);
                    rmParam.context.display = this;
                    removeWidgets(rmParam);
                    it = m_messagesToDisplay.erase(it);
                } else {
                    ++it;
                }
            }
        } else {
            CVLog::Warning(
                    "[ecvDisplayTools::DisplayNewMessage] Appending an empty "
                    "message has no effect!");
        }
        toBeRefreshed();
        return;
    }

    if (!append) {
        if (type != CUSTOM_MESSAGE) {
            for (auto it = m_messagesToDisplay.begin();
                 it != m_messagesToDisplay.end();) {
                if (it->type == type) {
                    WIDGETS_PARAMETER rmParam(WIDGETS_TYPE::WIDGET_T2D,
                                              it->message);
                    rmParam.context.display = this;
                    removeWidgets(rmParam);
                    it = m_messagesToDisplay.erase(it);
                } else {
                    ++it;
                }
            }
        }
    } else if (pos == SCREEN_CENTER_MESSAGE) {
        CVLog::Warning(
                "[ecvDisplayTools::DisplayNewMessage] Append is not "
                "supported for center screen messages!");
    }

    ecvMessageToDisplay msg;
    msg.message = message;
    msg.messageValidity_sec = m_timer.elapsed() / 1000 + displayMaxDelay_sec;
    msg.position = pos;
    msg.type = type;
    m_messagesToDisplay.push_back(msg);
    toBeRefreshed();
}

void vtkGLView::zoomGlobal() {
    if (!m_visualizer3D) return;

    ccBBox bbox;
    getVisibleObjectsBB(bbox);

    if (bbox.isValid()) {
        m_visualizer3D->resetCamera(&bbox);
        CCVector3d center = CCVector3d::fromArray(bbox.getCenter().u);
        m_ctx.viewportParams.setPivotPoint(center, true);
        m_ctx.autoPivotCandidate = center;
        if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                    m_visualizer3D.get())) {
            vis->setCenterOfRotation(center.x, center.y, center.z);
        }
    } else {
        m_visualizer3D->resetCamera();
    }

    m_visualizer3D->getRenderWindow()->Render();
    if (m_vtkWidget) m_vtkWidget->update();
}

// ================================================================
// Phase 7a: Per-view VTK operation overrides
//
// All delegates route through the singleton's VtkDisplayTools which
// internally calls resolveVisualizer(context.display) to target the
// correct per-view VtkVis.  When Phase 7b removes the singleton,
// these will call m_visualizer3D methods directly.
// ================================================================

void vtkGLView::draw(const ccGLDrawContext& context, const ccHObject* obj) {
    if (m_displayTools) m_displayTools->draw(context, obj);
}

void vtkGLView::drawBBox(const ccGLDrawContext& context, const ccBBox* bbox) {
    if (m_displayTools) m_displayTools->drawBBox(context, bbox);
}

void vtkGLView::drawBBoxBatch(const ccGLDrawContext& context,
                              const std::vector<ccBBox>& boxes) {
    if (m_displayTools) m_displayTools->drawBBoxBatch(context, boxes);
}

void vtkGLView::drawOrientedBBox(const ccGLDrawContext& context,
                                 const ecvOrientedBBox* obb) {
    if (m_displayTools) m_displayTools->drawOrientedBBox(context, obb);
}

void vtkGLView::updateMeshTextures(const ccGLDrawContext& context,
                                   const ccGenericMesh* mesh) {
    if (m_displayTools) m_displayTools->updateMeshTextures(context, mesh);
}

void vtkGLView::drawWidgets(const WIDGETS_PARAMETER& param) {
    if (m_displayTools) m_displayTools->drawWidgets(param);
}

void vtkGLView::removeWidgets(const WIDGETS_PARAMETER& param) {
    if (m_displayTools) m_displayTools->removeWidgets(param);
}

bool vtkGLView::hideShowEntities(const ccGLDrawContext& context) {
    if (m_displayTools) return m_displayTools->hideShowEntities(context);
    return true;
}

void vtkGLView::removeEntities(const ccGLDrawContext& context) {
    if (m_displayTools) m_displayTools->removeEntities(context);
}

void vtkGLView::changeEntityProperties(PROPERTY_PARAM& param) {
    if (m_displayTools) m_displayTools->changeEntityProperties(param);
}

void vtkGLView::updateCamera() {
    if (m_visualizer3D) m_visualizer3D->getRenderWindow()->Render();
}

void vtkGLView::updateScene() {
    if (m_visualizer3D) m_visualizer3D->getRenderWindow()->Render();
}

// -- Phase M1.3: Per-view picking and rendering --

QString vtkGLView::pick2DLabel(int x, int y) {
    if (m_visualizer2D) {
        return m_visualizer2D->pickItem(x, y).c_str();
    }
    return {};
}

QString vtkGLView::pick3DItem(int x, int y) {
    if (m_visualizer3D) {
        return m_visualizer3D->pickItem(x, y).c_str();
    }
    return {};
}

QString vtkGLView::pickObject(double x, double y) {
    if (m_visualizer3D) {
        vtkActor* pickedActor = m_visualizer3D->pickActor(x, y);
        if (pickedActor) {
            return m_visualizer3D->getIdByActor(pickedActor).c_str();
        }
    }
    return QStringLiteral("-1");
}

QImage vtkGLView::renderToImage(int zoomFactor,
                                bool renderOverlayItems,
                                bool silent,
                                int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->renderToImage(zoomFactor, renderOverlayItems,
                                             silent, viewport);
    }
    if (!silent) CVLog::Error("[vtkGLView::renderToImage] No 3D visualizer");
    return {};
}

void vtkGLView::resetCamera(const ccBBox* bbox) {
    if (m_visualizer3D) {
        m_visualizer3D->resetCamera(bbox);
        m_visualizer3D->getRenderWindow()->Render();
    }
}

void vtkGLView::resetCamera() {
    if (m_visualizer3D) {
        m_visualizer3D->resetCamera();
        m_visualizer3D->getRenderWindow()->Render();
    }
}

void vtkGLView::toggle2Dviewer(bool state) {
    if (m_visualizer3D) {
        m_visualizer3D->setInteractionMode(
                state ? Visualization::VtkVis::INTERACTION_MODE_2D
                      : Visualization::VtkVis::INTERACTION_MODE_3D);
    }
}

// ================================================================
// Phase 7a wave 2: Additional per-view virtual overrides
// ================================================================

CCVector3d vtkGLView::toVtkCoordinates(int x, int y, int z) {
    CCVector3d p(x * 1.0, y * 1.0, z * 1.0);
    p.y = glHeight() - p.y;
    p *= getDevicePixelRatio();
    return p;
}

bool vtkGLView::getClick3DPos(int x, int y, CCVector3d& pos) {
    return m_displayTools ? m_displayTools->getClick3DPos(x, y, pos) : false;
}

void vtkGLView::setView(CC_VIEW_ORIENTATION orientation) {
    if (m_displayTools) m_displayTools->setView(orientation);
}

CCVector3d vtkGLView::getCurrentViewDir() const {
    const double* M = m_ctx.viewportParams.viewMat.data();
    CCVector3d axis(-M[2], -M[6], -M[10]);
    axis.normalize();
    return axis;
}

void vtkGLView::setPivotPoint(const CCVector3d& P,
                              bool autoRedraw,
                              bool verbose) {
    if (m_displayTools) m_displayTools->setPivotPoint(P, autoRedraw, verbose);
}

void vtkGLView::setPivotVisibility(ecvGenericGLDisplay::PivotVisibility vis) {
    m_ctx.pivotVisibility = vis;
    if (m_displayTools) m_displayTools->setPivotVisibility(vis);
}

void vtkGLView::setAutoPickPivotAtCenter(bool state) {
    if (m_ctx.autoPickPivotAtCenter != state) {
        m_ctx.autoPickPivotAtCenter = state;
        if (state) {
            m_ctx.autoPivotCandidate = CCVector3d(0, 0, 0);
        }
    }
}

void vtkGLView::resetCenterOfRotation(int viewport) {
    if (!m_displayTools) return;
    m_displayTools->resetCenterOfRotation(viewport);

    auto* vis = dynamic_cast<Visualization::VtkVis*>(m_visualizer3D.get());
    if (vis) {
        double center[3] = {0, 0, 0};
        vis->getCenterOfRotation(center);
        CCVector3d pivot(center[0], center[1], center[2]);
        m_ctx.viewportParams.setPivotPoint(pivot, true);
        m_ctx.autoPivotCandidate = pivot;
    }
}

bool vtkGLView::isRotationAxisLocked() const {
    return m_ctx.rotationAxisLocked;
}

void vtkGLView::lockRotationAxis(bool state, const CCVector3d& axis) {
    m_ctx.rotationAxisLocked = state;
    m_ctx.lockedRotationAxis = axis;
    m_ctx.lockedRotationAxis.normalize();
}

void vtkGLView::toggleCameraOrientationWidget(bool state) {
    if (m_displayTools) m_displayTools->toggleCameraOrientationWidget(state);
}

void vtkGLView::toggleOrientationMarker(bool state) {
    if (m_displayTools) m_displayTools->toggleOrientationMarker(state);
}

void vtkGLView::toggleDebugTrace() {
    m_ctx.showDebugTraces = !m_ctx.showDebugTraces;
}

void vtkGLView::update2DLabels(bool immediateUpdate) {
    if (m_displayTools) m_displayTools->update2DLabels(immediateUpdate);
}

bool vtkGLView::renderToFile(const QString& filename,
                             float zoomFactor,
                             bool dontScale) {
    return m_displayTools ? m_displayTools->renderToFile(filename, zoomFactor,
                                                         dontScale)
                          : false;
}

void vtkGLView::removeBB(const QString& viewId) {
    if (m_displayTools) m_displayTools->removeBB(viewId);
}

void vtkGLView::removeBB(const ccGLDrawContext& context) {
    if (m_displayTools) m_displayTools->removeBB(context);
}

void vtkGLView::setExclusiveFullScreenFlag(bool state) {
    m_ctx.exclusiveFullscreen = state;
}

double vtkGLView::getObjectLightIntensity(const QString& viewID) const {
    return m_displayTools ? m_displayTools->getObjectLightIntensity(viewID)
                          : 1.0;
}

void vtkGLView::setObjectLightIntensity(const QString& viewID,
                                        double intensity) {
    if (m_displayTools)
        m_displayTools->setObjectLightIntensity(viewID, intensity);
}

double vtkGLView::getLightIntensity() const {
    return m_displayTools ? m_displayTools->getLightIntensity() : 1.0;
}

void vtkGLView::setLightIntensity(double intensity) {
    if (m_displayTools) m_displayTools->setLightIntensity(intensity);
}

void vtkGLView::getDataAxesGridProperties(const QString& viewID,
                                          AxesGridProperties& props,
                                          int viewport) const {
    if (m_displayTools)
        m_displayTools->getDataAxesGridProperties(viewID, props, viewport);
}

void vtkGLView::setDataAxesGridProperties(const QString& viewID,
                                          const AxesGridProperties& props,
                                          int viewport) {
    if (m_displayTools)
        m_displayTools->setDataAxesGridProperties(viewID, props, viewport);
}

void vtkGLView::filterByEntityType(std::vector<ccHObject*>& entities,
                                   CV_CLASS_ENUM type) {
    if (m_displayTools) m_displayTools->filterByEntityType(entities, type);
}

void vtkGLView::updateActiveItemsList(int x, int y, bool centerItems) {
    if (m_displayTools)
        m_displayTools->updateActiveItemsList(x, y, centerItems);
}

double vtkGLView::computeActualPixelSize() const {
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

void vtkGLView::updateNamePoseRecursive() {
    if (m_displayTools) m_displayTools->updateNamePoseRecursive();
}

void vtkGLView::showPivotSymbol(bool state) {
    if (state && !m_ctx.pivotSymbolShown &&
        m_ctx.viewportParams.objectCenteredView &&
        m_ctx.pivotVisibility != PIVOT_HIDE) {
        invalidateViewport();
        deprecate3DLayer();
    }
    m_ctx.pivotSymbolShown = state;
}

bool vtkGLView::exclusiveFullScreen() const {
    return m_ctx.exclusiveFullscreen;
}

CCVector3d vtkGLView::convertMousePositionToOrientation(int x, int y) {
    return m_displayTools
                   ? m_displayTools->convertMousePositionToOrientation(x, y)
                   : CCVector3d(0, 0, 0);
}

bool vtkGLView::processClickableItems(int x, int y) {
    if (!m_displayTools) return false;
    return ecvDisplayTools::ProcessClickableItems(m_ctx, x, y);
}

void vtkGLView::updateZoom(float zoomFactor) {
    if (m_displayTools) m_displayTools->updateZoom(zoomFactor);
}

void vtkGLView::resizeGL(int w, int h) {
    if (m_displayTools) m_displayTools->resizeGL(w, h);
}

void vtkGLView::setViewportDefaultPointSize(float size) {
    m_ctx.viewportParams.defaultPointSize = size;
}

void vtkGLView::setViewportDefaultLineWidth(float width) {
    m_ctx.viewportParams.defaultLineWidth = width;
}

void vtkGLView::setZNearCoef(double coef) {
    if (m_displayTools) m_displayTools->setZNearCoef(coef);
}

void vtkGLView::setFov(float fov_deg) {
    if (m_displayTools) m_displayTools->setFov(fov_deg);
}

void vtkGLView::setPointSizeOnView(float size) {
    if (m_displayTools) m_displayTools->setPointSizeOnView(size);
}

void vtkGLView::rotateWithAxis(const CCVector2i& mousePos,
                               const CCVector3d& axis,
                               double angle_deg) {
    if (m_displayTools)
        m_displayTools->rotateWithAxis(mousePos, axis, angle_deg, 0);
}

void vtkGLView::startPicking(PICKING_MODE mode, int x, int y, int w, int h) {
    if (m_displayTools) m_displayTools->startPicking(mode, x, y, w, h);
}

void vtkGLView::redraw2DLabel() {
    if (m_displayTools) m_displayTools->redraw2DLabel();
}

void vtkGLView::scheduleFullRedraw(int delayMs) {
    m_scheduledFullRedrawTime = m_timer.elapsed() + delayMs;
    if (!m_scheduleTimer.isActive()) {
        m_scheduleTimer.start(delayMs);
    }
}

void vtkGLView::startDeferredPicking() { m_deferredPickingTimer.start(); }
