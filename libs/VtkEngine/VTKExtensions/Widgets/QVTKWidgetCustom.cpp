// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "QVTKWidgetCustom.h"

#include "VtkUtils/rendererslayoutalgo.h"
#include "VtkUtils/utils.h"
#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVConst.h>
#include <CVLog.h>
#include <CVTools.h>

// VTK
#include <vtkAbstractPicker.h>
#include <vtkAngleRepresentation2D.h>
#include <vtkAxesActor.h>
#include <vtkCamera.h>
#include <vtkClipPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkConeSource.h>
#include <vtkDelaunay2D.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkIdFilter.h>
#include <vtkImageData.h>
#include <vtkLogoRepresentation.h>
#include <vtkLogoWidget.h>
#include <vtkLookupTable.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkPNGReader.h>
#include <vtkProperty2D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkScalarBarRepresentation.h>
#include <vtkScalarBarWidget.h>
#include <vtkTransform.h>
#include <vtkVertexGlyphFilter.h>

#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvInteractor.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

#include "ecvGLView.h"

// QT
#include <QApplication>
#include <QCoreApplication>
#include <QHBoxLayout>
#include <QLayout>
#include <QMainWindow>
#include <QMessageBox>
#include <QMimeData>
#include <QOpenGLWidget>
#include <QPushButton>
#include <QSettings>
#include <QThread>
#include <QTimer>
#include <QTouchEvent>
#include <QWheelEvent>
#include <QWidget>

#ifdef USE_VLD
#include <vld.h>
#endif

#include <ecv2DLabel.h>
#include <ecvHObjectCaster.h>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QPainter>

#include "ScaleBarWidget.h"

// macroes
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) \
    vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif

ecvViewContext& QVTKWidgetCustom::curCtx() {
    if (m_ownerView) return m_ownerView->context();
    return m_tools->m_primaryCtx;
}

const ecvViewContext& QVTKWidgetCustom::curCtx() const {
    if (m_ownerView) return m_ownerView->context();
    return m_tools->m_primaryCtx;
}

ecvViewContext* QVTKWidgetCustom::ownerCtx() {
    return m_ownerView ? &m_ownerView->context() : nullptr;
}

ccPolyline*& QVTKWidgetCustom::curRectPickingPoly() {
    if (m_ownerView) return m_ownerView->rectPickingPolyRef();
    return m_tools->m_rectPickingPoly;
}

std::list<ccInteractor*>& QVTKWidgetCustom::curActiveItems() {
    if (m_ownerView) return m_ownerView->activeItemsRef();
    return m_tools->m_activeItems;
}

ecvHotZone*& QVTKWidgetCustom::curHotZone() {
    if (m_ownerView) return m_ownerView->hotZoneRef();
    return m_tools->m_hotZone;
}

class VtkWidgetPrivate {
public:
    VtkWidgetPrivate(QVTKWidgetCustom* q);
    ~VtkWidgetPrivate();

    void init();
    void configRenderer(vtkRenderer* renderer);
    void layoutRenderers();

    QVTKWidgetCustom* q_ptr;
    QColor backgroundColor = Qt::black;
    bool multiViewports = false;
    vtkRenderer* defaultRenderer = nullptr;
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationMarkerWidget =
            nullptr;

    QList<vtkRenderer*> renderers;
    QList<vtkProp*> actors;
    QList<vtkProp*> props;

    double bounds[6];
};

VtkWidgetPrivate::VtkWidgetPrivate(QVTKWidgetCustom* q) : q_ptr(q) { init(); }

VtkWidgetPrivate::~VtkWidgetPrivate() {}

void VtkWidgetPrivate::configRenderer(vtkRenderer* renderer) {
    if (!renderer) return;

    double bgclr[3];
    Utils::vtkColor(backgroundColor, bgclr);

    renderer->SetBackground(bgclr);
}

static int columnCount(int count) {
    int cols = 1;
    while (true) {
        if ((cols * cols) >= count) return cols;
        ++cols;
    }
    return cols;
}

void VtkWidgetPrivate::layoutRenderers() {
    switch (renderers.size()) {
        case 1:
            VtkUtils::layoutRenderers<1>(renderers);
            break;

        case 2:
            VtkUtils::layoutRenderers<2>(renderers);
            break;

        case 3:
            VtkUtils::layoutRenderers<3>(renderers);
            break;

        case 4:
            VtkUtils::layoutRenderers<4>(renderers);
            break;

        case 5:
            VtkUtils::layoutRenderers<5>(renderers);
            break;

        case 6:
            VtkUtils::layoutRenderers<6>(renderers);
            break;

        case 7:
            VtkUtils::layoutRenderers<7>(renderers);
            break;

        case 8:
            VtkUtils::layoutRenderers<8>(renderers);
            break;

        case 9:
            VtkUtils::layoutRenderers<9>(renderers);
            break;

        case 10:
            VtkUtils::layoutRenderers<10>(renderers);
            break;

        default:
            VtkUtils::layoutRenderers<-1>(renderers);
    }
}

void VtkWidgetPrivate::init() { layoutRenderers(); }

// Max click duration for enabling picking mode (in ms)
// static const int CC_MAX_PICKING_CLICK_DURATION_MS = 200;
static const int CC_MAX_PICKING_CLICK_DURATION_MS = 350;
QVTKWidgetCustom::QVTKWidgetCustom(QMainWindow* parentWindow,
                                   ecvDisplayTools* tools,
                                   bool stereoMode)
    : QVTKOpenGLNativeWidget(parentWindow),
      m_render(nullptr),
      m_win(parentWindow),
      m_tools(tools),
      m_dataObject(nullptr),
      m_modelActor(nullptr),
      m_interactor(nullptr),
      m_logoWidget(nullptr),
      m_scalarbarWidget(nullptr),
      m_axesWidget(nullptr),
      m_scaleBar(nullptr),
      m_wheelZoomUpdateTimer(nullptr) {
    this->setWindowTitle("3D View");

    // Initialize timer for delayed 2D label update after wheel zoom
    m_wheelZoomUpdateTimer = new QTimer(this);
    m_wheelZoomUpdateTimer->setSingleShot(true);
    m_wheelZoomUpdateTimer->setInterval(150);  // 150ms delay
    connect(m_wheelZoomUpdateTimer, &QTimer::timeout, this, [this]() {
        if (m_ownerView)
            m_ownerView->update2DLabels(true);
        else if (m_tools)
            m_tools->Update2DLabel(true);
    });

    QSurfaceFormat fmt = QVTKOpenGLNativeWidget::defaultFormat();
    fmt.setStereo(stereoMode);
    setFormat(fmt);

#ifdef Q_OS_WIN
    this->setEnableHiDPI(true);
#endif

    // drag & drop handling
    setAcceptDrops(true);
    setAttribute(Qt::WA_AcceptTouchEvents, true);
    // setAttribute(Qt::WA_OpaquePaintEvent, true);
    vtkObject::GlobalWarningDisplayOff();
    d_ptr = new VtkWidgetPrivate(this);
}

QVTKWidgetCustom::~QVTKWidgetCustom() {
    // Detach per-widget HotZone from the singleton if it points to ours
    if (m_tools && curHotZone() == m_localHotZone) {
        curHotZone() = nullptr;
    }
    delete m_localHotZone;
    m_localHotZone = nullptr;

    if (d_ptr) {
        delete d_ptr;
        d_ptr = nullptr;
    }
    if (m_scaleBar) {
        delete m_scaleBar;
        m_scaleBar = nullptr;
    }
    m_wheelZoomUpdateTimer = nullptr;
}

ecvGenericGLDisplay* QVTKWidgetCustom::resolveDisplay() const {
    return ecvGenericGLDisplay::FromWidget(const_cast<QVTKWidgetCustom*>(this));
}

vtkSmartPointer<vtkLookupTable> QVTKWidgetCustom::createLookupTable(
        double min, double max) {
    double hsv1[3];
    double hsv2[3];
    Utils::qColor2HSV(m_color1, hsv1);
    Utils::qColor2HSV(m_color2, hsv2);

    VTK_CREATE(vtkLookupTable, lut);
    lut->SetHueRange(hsv1[0], hsv2[0]);
    lut->SetSaturationRange(hsv1[1], hsv2[1]);
    lut->SetValueRange(hsv1[2], hsv2[2]);
    lut->SetTableRange(min, max);
    lut->Build();

    return lut;
}

void QVTKWidgetCustom::initVtk(
        vtkSmartPointer<vtkRenderWindowInteractor> interactor, bool useVBO) {
    this->m_useVBO = useVBO;
    this->m_interactor = interactor;

    this->m_render =
            this->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
    this->m_camera = m_render->GetActiveCamera();
    this->m_renders = this->GetRenderWindow()->GetRenderers();
    if (!m_scaleBar) {
        m_scaleBar = new ScaleBarWidget(m_render);
    }
}

void QVTKWidgetCustom::transformCameraView(const double* viewMat) {
    vtkSmartPointer<vtkTransform> viewTransform =
            vtkSmartPointer<vtkTransform>::New();
    viewTransform->SetMatrix(viewMat);
    vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
    cam->ApplyTransform(viewTransform);
    this->m_render->SetActiveCamera(cam);
    this->m_render->Render();
}

void QVTKWidgetCustom::transformCameraProjection(const double* projMat) {
    vtkSmartPointer<vtkMatrix4x4> ProjTransform =
            vtkSmartPointer<vtkMatrix4x4>::New();
    ProjTransform->Determinant(projMat);
    this->m_camera->SetExplicitProjectionTransformMatrix(ProjTransform);
}

bool IsCalledFromMainThread() {
    return QThread::currentThread() == QCoreApplication::instance()->thread();
}

void QVTKWidgetCustom::updateScene() {
    if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
    if (IsCalledFromMainThread() && this->GetRenderWindow()) {
        this->GetRenderWindow()->Render();
    } else {  // only core threading enabled rendering
        QMetaObject::invokeMethod(
                this, [=]() { this->GetRenderWindow()->Render(); },
                Qt::QueuedConnection);
    }
}

void QVTKWidgetCustom::toWorldPoint(const CCVector3d& input2D,
                                    CCVector3d& output3D) {
    // auto picker = GetInteractor()->GetPicker();
    // picker->Pick(input2D.x, input2D.y, 0, m_renders->GetFirstRenderer());
    // picker->GetPickPosition(output3D.u);

    m_render->SetDisplayPoint(input2D.x, input2D.y, input2D.z);
    m_render->DisplayToWorld();
    const double* world = m_render->GetWorldPoint();
    for (int i = 0; i < 3; i++) {
        output3D.u[i] = world[i] / world[3];
    }
}

void QVTKWidgetCustom::toWorldPoint(const CCVector3& input2D,
                                    CCVector3d& output3D) {
    toWorldPoint(CCVector3d::fromArray(input2D.u), output3D);
}

void QVTKWidgetCustom::toDisplayPoint(const CCVector3d& worldPos,
                                      CCVector3d& displayPos) {
    m_render->SetWorldPoint(worldPos.x, worldPos.y, worldPos.z, 1.0);
    m_render->WorldToDisplay();
    displayPos.x = (m_render->GetDisplayPoint())[0];
    displayPos.y = (m_render->GetDisplayPoint())[1];
    displayPos.z = (m_render->GetDisplayPoint())[2];
}

void QVTKWidgetCustom::toDisplayPoint(const CCVector3& worldPos,
                                      CCVector3d& displayPos) {
    toDisplayPoint(CCVector3d::fromArray(worldPos.u), displayPos);
}

void QVTKWidgetCustom::setCameraPosition(const CCVector3d& pos) {
    vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
    cam->SetPosition(pos.x, pos.y, pos.z);
    this->m_render->SetActiveCamera(cam);
    this->m_render->Render();
}

void QVTKWidgetCustom::setCameraFocalPoint(const CCVector3d& pos) {
    vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
    cam->SetFocalPoint(pos.x, pos.y, pos.z);
    this->m_render->SetActiveCamera(cam);
    this->m_render->Render();
}

void QVTKWidgetCustom::setCameraViewUp(const CCVector3d& pos) {
    vtkSmartPointer<vtkCamera> cam = this->m_render->GetActiveCamera();
    cam->SetViewUp(pos.x, pos.y, pos.z);
    this->m_render->SetActiveCamera(cam);
    this->m_render->Render();
}

void QVTKWidgetCustom::setBackgroundColor(const ecvColor::Rgbf& bkg1,
                                          const ecvColor::Rgbf& bkg2,
                                          bool gradient) {
    m_render->SetBackground2(bkg2.r, bkg2.g, bkg2.b);
    m_render->SetBackground(bkg1.r, bkg1.g, bkg1.b);
    m_render->SetGradientBackground(gradient);
}

void QVTKWidgetCustom::setMultiViewports(bool multi) {
    if (d_ptr->multiViewports != multi) {
        d_ptr->multiViewports = multi;
    }
}

bool QVTKWidgetCustom::multiViewports() const { return d_ptr->multiViewports; }

void QVTKWidgetCustom::addActor(vtkProp* actor, const QColor& clr) {
    if (!actor || d_ptr->actors.contains(actor)) return;

    double vtkClr[3];
    Utils::vtkColor(clr, vtkClr);

    d_ptr->actors.append(actor);

    if (!d_ptr->multiViewports) {
        if (d_ptr->renderers.isEmpty()) {
            vtkRenderer* renderer = vtkRenderer::New();
            renderer->SetBackground(vtkClr);
            d_ptr->configRenderer(renderer);
            renderer->AddActor(actor);
            GetRenderWindow()->AddRenderer(renderer);
            d_ptr->renderers.append(renderer);
            renderer->ResetCamera();
        } else {
            defaultRenderer()->SetBackground(vtkClr);
            defaultRenderer()->AddActor(actor);
        }
    } else {
        if (!defaultRendererTaken()) {
            defaultRenderer()->SetBackground(vtkClr);
            defaultRenderer()->AddActor(actor);
        } else {
            vtkRenderer* renderer = vtkRenderer::New();
            renderer->SetBackground(vtkClr);
            d_ptr->configRenderer(renderer);
            renderer->AddActor(actor);
            GetRenderWindow()->AddRenderer(renderer);
            d_ptr->renderers.append(renderer);
            d_ptr->layoutRenderers();
            renderer->ResetCamera();
        }
    }
}

void QVTKWidgetCustom::addViewProp(vtkProp* prop) {
    if (!prop || d_ptr->props.contains(prop)) return;

    d_ptr->props.append(prop);

    if (!d_ptr->multiViewports) {
        if (d_ptr->renderers.isEmpty()) {
            vtkRenderer* renderer = vtkRenderer::New();
            d_ptr->configRenderer(renderer);
            renderer->AddViewProp(prop);
            GetRenderWindow()->AddRenderer(renderer);
            d_ptr->renderers.append(renderer);
            renderer->ResetCamera();
        } else {
            defaultRenderer()->AddViewProp(prop);
        }
    } else {
        if (!defaultRendererTaken()) {
            defaultRenderer()->AddViewProp(prop);
        } else {
            vtkRenderer* renderer = vtkRenderer::New();
            d_ptr->configRenderer(renderer);
            renderer->AddViewProp(prop);
            GetRenderWindow()->AddRenderer(renderer);
            d_ptr->renderers.append(renderer);
            d_ptr->layoutRenderers();
            renderer->ResetCamera();
        }
    }
}

QList<vtkProp*> QVTKWidgetCustom::actors() const { return d_ptr->actors; }

void QVTKWidgetCustom::setActorsVisible(bool visible) {
    foreach (auto actor, d_ptr->actors) actor->SetVisibility(visible);
}

void QVTKWidgetCustom::setActorVisible(vtkProp* actor, bool visible) {
    actor->SetVisibility(visible);
}

bool QVTKWidgetCustom::actorVisible(vtkProp* actor) {
    return actor->GetVisibility();
}

void QVTKWidgetCustom::setBackgroundColor(const QColor& clr) {
    if (d_ptr->backgroundColor != clr) {
        d_ptr->backgroundColor = clr;

        foreach (vtkRenderer* renderer, d_ptr->renderers)
            d_ptr->configRenderer(renderer);

#if 0
		vtkRendererCollection* renderers = GetRenderWindow()->GetRenderers();
		vtkRenderer* renderer = renderers->GetFirstRenderer();
		while (renderer) {
			renderer = renderers->GetNextItem();
		}
#endif
        update();
    }
}

QColor QVTKWidgetCustom::backgroundColor() const {
    return d_ptr->backgroundColor;
}

vtkRenderer* QVTKWidgetCustom::defaultRenderer() {
    VtkUtils::vtkInitOnce(&d_ptr->defaultRenderer);
    GetRenderWindow()->AddRenderer(d_ptr->defaultRenderer);
    if (!d_ptr->renderers.contains(d_ptr->defaultRenderer))
        d_ptr->renderers.append(d_ptr->defaultRenderer);
    return d_ptr->defaultRenderer;
}

bool QVTKWidgetCustom::defaultRendererTaken() const {
    if (!d_ptr->defaultRenderer) return false;
    return d_ptr->defaultRenderer->GetActors()->GetNumberOfItems() != 0;
}

void QVTKWidgetCustom::setBounds(double* bounds) {
    Utils::ArrayAssigner<double, 6> aa;
    aa(bounds, d_ptr->bounds);
}

double QVTKWidgetCustom::xMin() const { return d_ptr->bounds[0]; }

double QVTKWidgetCustom::xMax() const { return d_ptr->bounds[1]; }

double QVTKWidgetCustom::yMin() const { return d_ptr->bounds[2]; }

double QVTKWidgetCustom::yMax() const { return d_ptr->bounds[3]; }

double QVTKWidgetCustom::zMin() const { return d_ptr->bounds[4]; }

double QVTKWidgetCustom::zMax() const { return d_ptr->bounds[5]; }

void QVTKWidgetCustom::paintGL() {
    QVTKOpenGLNativeWidget::paintGL();

    ecvGenericGLDisplay* myDisplay = ecvGenericGLDisplay::FromWidget(this);

    ccHObject::Container labels;
    if (m_ownerView)
        m_ownerView->filterByEntityType(labels, CV_TYPES::LABEL_2D);
    else
        m_tools->filterByEntityType(labels, CV_TYPES::LABEL_2D);
    if (labels.empty()) return;

    int totalLabels = 0, visibleLabels = 0, displayedLabels = 0,
        validLabels = 0;
    for (auto* obj : labels) {
        ++totalLabels;
        if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
            !obj->isVisible())
            continue;
        ++visibleLabels;
        auto* label = static_cast<cc2DLabel*>(obj);
        if (!label->isDisplayedIn(myDisplay)) continue;
        ++displayedLabels;
        if (!label->overlayValid()) continue;
        ++validLabels;
    }

    static int s_logCounter = 0;
    if ((s_logCounter++ % 120) == 0) {
        CVLog::Print(
                "[QPainter] paintGL: labels=%d visible=%d displayed=%d "
                "valid=%d myDisplay=%p",
                totalLabels, visibleLabels, displayedLabels, validLabels,
                static_cast<void*>(myDisplay));
    }

    if (validLabels == 0) return;

    if (auto* ctx = QOpenGLContext::currentContext()) {
        auto* f = ctx->functions();
        f->glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
        f->glDisable(GL_DEPTH_TEST);
        f->glDisable(GL_STENCIL_TEST);
    }

    QPainter painter(this);
    if (!painter.isActive()) {
        CVLog::Warning(
                "[QPainter] painter.isActive() = false! Cannot draw overlay.");
        return;
    }
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);

    for (auto* obj : labels) {
        if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
            !obj->isVisible())
            continue;
        auto* label = static_cast<cc2DLabel*>(obj);
        if (!label->isDisplayedIn(myDisplay)) continue;
        if (!label->overlayValid()) continue;

        const auto& od = label->overlayData();
        static int s_paintLog = 0;
        if ((s_paintLog++ % 30) == 0) {
            CVLog::Print(
                    "[QPainter] DRAW overlay '%s' panel=(%d,%d %dx%d) "
                    "segments=%d legends=%d body=%d",
                    qPrintable(label->getName()),
                    static_cast<int>(od.panelRect.x()),
                    static_cast<int>(od.panelRect.y()),
                    static_cast<int>(od.panelRect.width()),
                    static_cast<int>(od.panelRect.height()), od.segments.size(),
                    od.legends.size(), od.bodyLines.size());
        }

        label->paintOverlay(painter);
    }

    painter.end();
}

// event processing
void QVTKWidgetCustom::mousePressEvent(QMouseEvent* event) {
    // Activate the view on click (ParaView-style): the render view becomes
    // current when the user presses the mouse here, not when the cursor
    // merely passes over the widget (see mouseMoveEvent).
    auto* display = ecvGenericGLDisplay::FromWidget(this);
    if (display) {
        auto& vm = ecvViewManager::instance();
        if (vm.getActiveView() != display) {
            vm.setActiveView(display);
        }
    }

    curMouseMoved() = false;
    curMouseButtonPressed() = true;
    curIgnoreMouseReleaseEvent() = false;
    curLastMousePos() = event->pos();

    if (!ecvViewManager::useVtkPick()) {
        curLastPointIndex() = -1;
        curLastPickedId() = QString();
    }

    if ((event->buttons() & Qt::RightButton)) {
        // Check if right-click is on a cc2DLabel (collapse toggle)
        m_rightClickOnLabel = false;
        if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            ccHObject::Container labels;
            if (m_ownerView)
                m_ownerView->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            else
                m_tools->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            for (auto* obj : labels) {
                if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
                    !obj->isVisible())
                    continue;
                cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                if (!l) continue;
                QRect roi = l->getLabelROI();
                if (roi.isValid() && roi.contains(event->x(), event->y())) {
                    m_rightClickOnLabel = true;
                    break;
                }
            }
        }

        if (!m_rightClickOnLabel) {
            // right click = panning (2D translation)
            if ((curInteractionFlags() & ecvGenericGLDisplay::INTERACT_PAN) ||
                ((QApplication::keyboardModifiers() & Qt::ControlModifier) &&
                 (curInteractionFlags() &
                  ecvGenericGLDisplay::INTERACT_CTRL_PAN))) {
                QApplication::setOverrideCursor(QCursor(Qt::SizeAllCursor));
            }
        }

        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_SIG_RB_CLICKED) {
            if (m_ownerView)
                emit m_ownerView->rightButtonClicked(event->x(), event->y());
            emit m_tools->rightButtonClicked(event->x(), event->y());
        }
    } else if (event->buttons() & Qt::LeftButton) {
        curLastClickTime() = m_ownerView ? m_ownerView->elapsedMs()
                                         : m_tools->m_timer.elapsed();

        // Check if click is on a 2D label ROI BEFORE starting VTK rotation.
        // Only do direct ROI check (no FAST_PICKING fallback) to avoid
        // picking 3D objects that would block camera rotation.
        m_labelClickedOnPress = false;
        if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            ccHObject::Container labels;
            if (m_ownerView)
                m_ownerView->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            else
                m_tools->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            for (auto* obj : labels) {
                if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
                    !obj->isVisible())
                    continue;
                cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                if (!l) continue;
                QRect roi = l->getLabelROI();
                if (roi.isValid() && roi.contains(event->x(), event->y())) {
                    curActiveItems().clear();
                    curActiveItems().push_back(l);
                    m_labelClickedOnPress = true;
                    CVLog::Print(
                            "[Label] mousePressEvent: HIT label '%s' at "
                            "(%d,%d) ROI=(%d,%d %dx%d)",
                            qPrintable(l->getName()), event->x(), event->y(),
                            roi.x(), roi.y(), roi.width(), roi.height());
                    break;
                }
            }
        }

        // left click = rotation
        if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_ROTATE) {
            QApplication::setOverrideCursor(QCursor(Qt::PointingHandCursor));
        }

        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_SIG_LB_CLICKED) {
            if (m_ownerView)
                emit m_ownerView->leftButtonClicked(event->x(), event->y());
            emit m_tools->leftButtonClicked(event->x(), event->y());
        }

        // do this before drawing the pivot!
        if (curAutoPickPivotAtCenter()) {
            CCVector3d P;
            if (m_ownerView) {
                if (m_ownerView->getClick3DPos(event->x(), event->y(), P))
                    m_ownerView->setPivotPoint(P, true, false);
            } else {
                if (m_tools->getClick3DPos(event->x(), event->y(), P))
                    m_tools->setPivotPoint(P, true, false);
            }
        }
    } else {
    }

    // Skip VTK mouse processing when a cc2DLabel was clicked (left or right)
    if (m_labelClickedOnPress || m_rightClickOnLabel) {
        CVLog::Print(
                "[Label] BLOCKED VTK mousePress, labelClicked=%d "
                "rightClickOnLabel=%d",
                m_labelClickedOnPress, m_rightClickOnLabel);
        event->accept();
    } else {
        QVTKOpenGLNativeWidget::mousePressEvent(event);
    }
}

void QVTKWidgetCustom::mouseDoubleClickEvent(QMouseEvent* event) {
    // Same click-to-activate rule as mousePressEvent (some paths only
    // double-click).
    auto* display = ecvGenericGLDisplay::FromWidget(this);
    if (display) {
        auto& vm = ecvViewManager::instance();
        if (vm.getActiveView() != display) {
            vm.setActiveView(display);
        }
    }

    if (m_ownerView)
        m_ownerView->deferredPickingTimer().stop();
    else
        m_tools->m_deferredPickingTimer.stop();
    curIgnoreMouseReleaseEvent() = true;

    const int x = event->x();
    const int y = event->y();

    CCVector3d P;
    if (m_ownerView) {
        if (m_ownerView->getClick3DPos(x, y, P))
            m_ownerView->setPivotPoint(P, true, true);
    } else {
        if (m_tools->getClick3DPos(x, y, P))
            m_tools->setPivotPoint(P, true, true);
    }

    if (m_ownerView)
        emit m_ownerView->doubleButtonClicked(event->x(), event->y());
    emit m_tools->doubleButtonClicked(event->x(), event->y());

    QVTKOpenGLNativeWidget::mouseDoubleClickEvent(event);
}

void QVTKWidgetCustom::wheelEvent(QWheelEvent* event) {
    bool doRedraw = false;
    Qt::KeyboardModifiers keyboardModifiers = QApplication::keyboardModifiers();

    if (m_ownerView) emit m_ownerView->mouseWheelChanged(event);
    emit m_tools->mouseWheelChanged(event);
    double delta = qtCompatWheelEventDelta(event);

    if (keyboardModifiers & Qt::AltModifier) {
        event->accept();

        float sizeModifier = (delta < 0.0 ? -1.0f : 1.0f);
        if (m_ownerView)
            m_ownerView->setPointSizeOnView(
                    curViewportParams().defaultPointSize + sizeModifier);
        else
            m_tools->setPointSizeOnView(curViewportParams().defaultPointSize +
                                        sizeModifier);
        { ecvRedrawScope redrawScope; }
        doRedraw = true;
    } else if (keyboardModifiers & Qt::ControlModifier) {
        event->accept();
        if (curViewportParams().perspectiveView) {
            static const int MAX_INCREMENT = 150;
            int increment = ecvViewportParameters::ZNearCoefToIncrement(
                    curViewportParams().zNearCoef, MAX_INCREMENT + 1);
            int newIncrement =
                    std::min(std::max(0, increment + (delta < 0 ? -1 : 1)),
                             MAX_INCREMENT);
            if (newIncrement != increment) {
                double newCoef = ecvViewportParameters::IncrementToZNearCoef(
                        newIncrement, MAX_INCREMENT + 1);
                if (m_ownerView)
                    m_ownerView->setZNearCoef(newCoef);
                else
                    m_tools->setZNearCoef(newCoef);
                { ecvRedrawScope redrawScope; }
                doRedraw = true;
            }
        }
    } else if (keyboardModifiers & Qt::ShiftModifier) {
        event->accept();
        if (curViewportParams().perspectiveView) {
            float newFOV =
                    (curViewportParams().fov_deg + (delta < 0 ? -1.0f : 1.0f));
            newFOV = std::min(std::max(1.0f, newFOV), 180.0f);
            if (newFOV != curViewportParams().fov_deg) {
                if (m_ownerView)
                    m_ownerView->setFov(newFOV);
                else
                    m_tools->setFov(newFOV);
                { ecvRedrawScope redrawScope; }
                doRedraw = true;
            }
        }
    } else if (curInteractionFlags() &
               ecvGenericGLDisplay::INTERACT_ZOOM_CAMERA) {
        QVTKOpenGLNativeWidget::wheelEvent(event);

        float wheelDelta_deg = static_cast<float>(delta) / 8;
        m_tools->onWheelEvent(wheelDelta_deg);
        if (m_ownerView) {
            emit m_ownerView->mouseWheelRotated(wheelDelta_deg);
            emit m_ownerView->cameraParamChanged();
        }
        emit m_tools->mouseWheelRotated(wheelDelta_deg);
        emit m_tools->cameraParamChanged();

        doRedraw = true;
        event->accept();
    }

    if (doRedraw) {
        if (m_ownerView) emit m_ownerView->labelmove2D(0, 0, 0, 0);
        emit m_tools->labelmove2D(0, 0, 0, 0);
        if (m_ownerView)
            m_ownerView->updateNamePoseRecursive();
        else
            m_tools->updateNamePoseRecursive();

        if (m_wheelZoomUpdateTimer) {
            m_wheelZoomUpdateTimer->stop();
            m_wheelZoomUpdateTimer->start();
        }

        if (m_ownerView) {
            if (auto* w = m_ownerView->asWidget()) w->update();
            m_ownerView->updateCamera();
        } else {
            m_tools->updateScene();
        }
    }
}

void QVTKWidgetCustom::mouseMoveEvent(QMouseEvent* event) {
    // Do NOT call setActiveView here: ParaView-style UX activates a render view
    // on click (see mousePressEvent), not when the cursor merely moves across a
    // split window. Activating on every mouseMove caused the wrong view to
    // become "current" during passive hover.

    if (!((curInteractionFlags() & ecvGenericGLDisplay::INTERACT_ROTATE) &&
          (curInteractionFlags() &
           ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES))) {
        if ((curInteractionFlags() &
             ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA)) {
            // lock axis
            if ((curInteractionFlags() &
                 ecvGenericGLDisplay::INTERACT_ROTATE) &&
                (curInteractionFlags() &
                 ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) !=
                        ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES &&
                (event->buttons() & Qt::LeftButton) &&
                curRotationAxisLocked()) {
                event->accept();
            } else if (event->buttons() & Qt::MiddleButton) {
                if (m_ownerView) {
                    QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                } else {
                    event->accept();
                }
            } else {  // normal
                // Never forward to VTK when a cc2DLabel is being dragged
                if (!m_labelClickedOnPress && curActiveItems().empty()) {
                    CVLog::PrintDebug(
                            "[VTK-FWD] mouseMoveEvent -> VTK "
                            "labelClicked=%d activeItems=%zu btn=0x%x",
                            m_labelClickedOnPress, curActiveItems().size(),
                            (unsigned)event->buttons());
                    QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                    m_tools->UpdateDisplayParameters();
                }
            }
        }
    }

    const int x = event->x();
    const int y = event->y();
    const bool isActiveWidget =
            m_ownerView ? (ecvViewManager::instance().getEffectiveView() ==
                           m_ownerView)
                        : (ecvGenericGLDisplay::FromWidget(this) ==
                           static_cast<ecvGenericGLDisplay*>(m_tools));
    if (isActiveWidget) {
        curLastMouseMovePos() = event->pos();
        if (m_ownerView) emit m_ownerView->mousePosChanged(event->pos());
        emit m_tools->mousePosChanged(event->pos());
    }

    if ((curInteractionFlags() &
         ecvGenericGLDisplay::INTERACT_SIG_MOUSE_MOVED) &&
        isActiveWidget) {
        if (m_ownerView) emit m_ownerView->mouseMoved(x, y, event->buttons());
        emit m_tools->mouseMoved(x, y, event->buttons());
        event->accept();
    }

    // no button pressed
    if (event->buttons() == Qt::NoButton) {
        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_CLICKABLE_ITEMS) {
            // Per-widget HotZone for multi-window support
            if (!m_localHotZone) {
                m_localHotZone = new ecvHotZone(this);
            }
            if (!m_ownerView && m_tools->m_hotZoneOwnedBySingleton &&
                curHotZone() && curHotZone() != m_localHotZone) {
                delete curHotZone();
                m_tools->m_hotZoneOwnedBySingleton = false;
            }
            curHotZone() = m_localHotZone;

            QRect areaRect = m_localHotZone->rect(
                    true, curBubbleViewModeEnabled(),
                    m_ownerView ? m_ownerView->context().exclusiveFullscreen
                                : m_tools->exclusiveFullScreen());

            const int retinaScale = m_ownerView
                                            ? m_ownerView->getDevicePixelRatio()
                                            : m_tools->getDevicePixelRatio();
            bool inZone = (x * retinaScale * 3 < m_localHotZone->topCorner.x() +
                                                         areaRect.width() * 4 &&
                           y * retinaScale * 2 < m_localHotZone->topCorner.y() +
                                                         areaRect.height() * 4);

            if (inZone != m_localClickableVisible) {
                m_localClickableVisible = inZone;

                curClickableItemsVisible() = inZone;
                if (m_ownerView) {
                    m_ownerView->redraw(true, false);
                } else {
                    m_tools->redraw(true, false);
                }
            }

            event->accept();
        }

        // display the 3D coordinates of the pixel below the mouse cursor (if
        // possible)
        if (curShowCursorCoordinates()) {
            CCVector3d P;
            QString message = QString("2D (%1 ; %2)").arg(x).arg(y);
            {
                auto* ev =
                        m_ownerView
                                ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                                : static_cast<ecvGenericGLDisplay*>(m_tools);
                if (ev->getClick3DPos(x, y, P)) {
                    message += QString(" --> 3D (%1 ; %2 ; %3)")
                                       .arg(P.x)
                                       .arg(P.y)
                                       .arg(P.z);
                }
            }
            if (m_ownerView) {
                m_ownerView->displayNewMessage(
                        message, ecvGenericGLDisplay::LOWER_LEFT_MESSAGE, false,
                        5, ecvGenericGLDisplay::SCREEN_SIZE_MESSAGE);
                m_ownerView->redraw(true);
            } else {
                m_tools->displayNewMessage(
                        message, ecvGenericGLDisplay::LOWER_LEFT_MESSAGE, false,
                        5, ecvGenericGLDisplay::SCREEN_SIZE_MESSAGE);
                m_tools->redraw(true);
            }
        }

        // don't need to process any further
        return;
    }

    int dx = x - curLastMousePos().x();
    int dy = y - curLastMousePos().y();

    if ((event->buttons() & Qt::RightButton)) {
        if (abs(dx) > 0 || abs(dy) > 0) {
            if (m_ownerView) emit m_ownerView->labelmove2D(x, y, 0, 0);
            emit m_tools->labelmove2D(x, y, 0, 0);
            (m_ownerView ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                         : static_cast<ecvGenericGLDisplay*>(m_tools))
                    ->updateNamePoseRecursive();
        }
    } else if ((event->buttons() & Qt::MiddleButton)) {
        if (!m_ownerView) {
            // Primary view: CC-side pan
            if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_PAN) {
                double pixSize = m_tools->computeActualPixelSize();
                CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
                if (!curViewportParams().perspectiveView) {
                    u.y *= curViewportParams().cameraAspectRatio;
                }

                const int retinaScale = m_tools->getDevicePixelRatio();
                u *= retinaScale;

                bool entityMovingMode =
                        (curInteractionFlags() &
                         ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) ||
                        ((QApplication::keyboardModifiers() &
                          Qt::ControlModifier) &&
                         curCustomLightEnabled());
                if (entityMovingMode) {
                    curViewportParams().viewMat.transposed().applyRotation(u);

                    if (curInteractionFlags() &
                        ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
                        if (m_ownerView) emit m_ownerView->translation(u);
                        emit m_tools->translation(u);
                    } else if (curCustomLightEnabled()) {
                        curCustomLightPos()[0] += static_cast<float>(u.x);
                        curCustomLightPos()[1] += static_cast<float>(u.y);
                        curCustomLightPos()[2] += static_cast<float>(u.z);
                        if (m_ownerView) {
                            m_ownerView->invalidateViewport();
                            m_ownerView->deprecate3DLayer();
                        } else {
                            m_tools->invalidateViewport();
                            m_tools->deprecate3DLayer();
                        }
                    }
                } else {
                    if (curViewportParams().objectCenteredView) {
                        u = -u;
                    }
                    m_tools->moveCamera(static_cast<float>(u.x),
                                        static_cast<float>(u.y),
                                        static_cast<float>(u.z));
                }
            }
        }

        if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            // on the first time, let's check if the mouse is on a (selected) 2D
            // item
            if (!curMouseMoved()) {
                if (curPickingMode() != ecvGenericGLDisplay::NO_PICKING
                    // DGM: in fact we still need to move labels in those modes
                    // below (see the 'Point Picking' tool of CLOUDVIEWER  for
                    // instance)
                    //&&	m_pickingMode != POINT_PICKING
                    //&&	m_pickingMode != TRIANGLE_PICKING
                    //&&	m_pickingMode != POINT_OR_TRIANGLE_PICKING
                    && (QApplication::keyboardModifiers() == Qt::NoModifier ||
                        QApplication::keyboardModifiers() ==
                                Qt::ControlModifier)) {
                    auto* ev = m_ownerView ? static_cast<ecvGenericGLDisplay*>(
                                                     m_ownerView)
                                           : static_cast<ecvGenericGLDisplay*>(
                                                     m_tools);
                    ev->updateActiveItemsList(curLastMousePos().x(),
                                              curLastMousePos().y(), true);
                }
            }
        }

        // OPTIMIZATION: Skip 2D label updates during panning to improve
        // performance Only emit signal for label movement, but skip expensive
        // Update2DLabel during panning The label will be updated when panning
        // stops (in mouseReleaseEvent)
        if (abs(dx) > 0 || abs(dy) > 0) {
            if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
            emit m_tools->labelmove2D(x, y, dx, dy);
            (m_ownerView ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                         : static_cast<ecvGenericGLDisplay*>(m_tools))
                    ->updateNamePoseRecursive();
            if (!curActiveItems().empty()) {
                updateActivateditems(x, y, dx, dy, true);
            }
        }
    } else if (event->buttons() & Qt::LeftButton)  // rotation
    {
        if (m_labelClickedOnPress) {
            double camPos[3] = {0, 0, 0};
            if (m_camera) m_camera->GetPosition(camPos);
            CVLog::Print(
                    "[Label-Drag] LB move: labelClicked=1 "
                    "activeItems=%zu dx=%d dy=%d cam=(%.1f,%.1f,%.1f)",
                    curActiveItems().size(), dx, dy, camPos[0], camPos[1],
                    camPos[2]);
        }
        if (!m_labelClickedOnPress) {
            if (m_ownerView)
                m_ownerView->scheduleFullRedraw(1000);
            else
                m_tools->scheduleFullRedraw(1000);
        }

        if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            // Skip re-detection if a label was already activated in
            // mousePressEvent to avoid FAST_PICKING replacing the label.
            if (!curMouseMoved() && !m_labelClickedOnPress) {
                if (curPickingMode() != ecvGenericGLDisplay::NO_PICKING &&
                    (QApplication::keyboardModifiers() == Qt::NoModifier ||
                     QApplication::keyboardModifiers() ==
                             Qt::ControlModifier)) {
                    auto* ev = m_ownerView ? static_cast<ecvGenericGLDisplay*>(
                                                     m_ownerView)
                                           : static_cast<ecvGenericGLDisplay*>(
                                                     m_tools);
                    ev->updateActiveItemsList(curLastMousePos().x(),
                                              curLastMousePos().y(), true);
                }
            }
        } else if (!m_labelClickedOnPress) {
            curActiveItems().clear();
        }

        // When a cc2DLabel is being dragged, ONLY move the 2D panel.
        // CloudCompare: caption panel is fixed on screen. During drag,
        // only the panel moves - no camera, no 3D re-projection, no
        // UpdateNamePoseRecursive (which clears m_activeItems every 50ms).
        if (m_labelClickedOnPress) {
            if (abs(dx) > 0 || abs(dy) > 0) {
                updateActivateditems(x, y, dx, dy, true);
            }
        } else if (!curActiveItems().empty()) {
            if (abs(dx) > 0 || abs(dy) > 0) {
                if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
                emit m_tools->labelmove2D(x, y, dx, dy);
                (m_ownerView ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                             : static_cast<ecvGenericGLDisplay*>(m_tools))
                        ->updateNamePoseRecursive();
                updateActivateditems(x, y, dx, dy, true);
            }
        } else {
            if (abs(dx) > 0 || abs(dy) > 0) {
                if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
                emit m_tools->labelmove2D(x, y, dx, dy);
                (m_ownerView ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                             : static_cast<ecvGenericGLDisplay*>(m_tools))
                        ->updateNamePoseRecursive();
                if (!curActiveItems().empty()) {
                    updateActivateditems(x, y, dx, dy, true);
                }
                curActiveItems().clear();
            }

            // specific case: rectangular polyline drawing (for rectangular area
            // selection mode)
            if (curAllowRectangularEntityPicking() &&
                (curPickingMode() == ecvGenericGLDisplay::ENTITY_PICKING ||
                 curPickingMode() ==
                         ecvGenericGLDisplay::ENTITY_RECT_PICKING) &&
                (curRectPickingPoly() ||
                 (QApplication::keyboardModifiers() & Qt::AltModifier))) {
                auto* evRect =
                        m_ownerView
                                ? static_cast<ecvGenericGLDisplay*>(m_ownerView)
                                : static_cast<ecvGenericGLDisplay*>(m_tools);
                // first time: initialization of the rectangle
                if (!curRectPickingPoly()) {
                    ccPointCloud* vertices = new ccPointCloud("rect.vertices");
                    curRectPickingPoly() = new ccPolyline(vertices);
                    curRectPickingPoly()->addChild(vertices);
                    if (vertices->reserve(4) &&
                        curRectPickingPoly()->addPointIndex(0, 4)) {
                        curRectPickingPoly()->setForeground(true);
                        curRectPickingPoly()->setColor(ecvColor::green);
                        curRectPickingPoly()->showColors(true);
                        curRectPickingPoly()->set2DMode(true);
                        curRectPickingPoly()->setVisible(true);
                        CCVector3d pos3D = evRect->toVtkCoordinates(
                                curLastMousePos().x(), curLastMousePos().y());

                        CCVector3 A(static_cast<PointCoordinateType>(pos3D.x),
                                    static_cast<PointCoordinateType>(pos3D.y),
                                    pos3D.z);
                        // we add 4 times the same point (just to fill the
                        // cloud!)
                        vertices->addPoint(A);
                        vertices->addPoint(A);
                        vertices->addPoint(A);
                        vertices->addPoint(A);
                        curRectPickingPoly()->setClosed(true);
                        if (m_ownerView)
                            m_ownerView->addToOwnDB(curRectPickingPoly(),
                                                    false);
                        else
                            m_tools->addToOwnDB(curRectPickingPoly(), false);
                    } else {
                        CVLog::Warning(
                                "[ QVTKWidgetCustom::mouseMoveEvent] Failed to "
                                "create seleciton polyline! Not enough "
                                "memory!");
                        delete curRectPickingPoly();
                        curRectPickingPoly() = nullptr;
                        vertices = nullptr;
                    }
                }

                if (curRectPickingPoly()) {
                    cloudViewer::GenericIndexedCloudPersist* vertices =
                            curRectPickingPoly()->getAssociatedCloud();
                    assert(vertices);
                    CCVector3* B = const_cast<CCVector3*>(
                            vertices->getPointPersistentPtr(1));
                    CCVector3* C = const_cast<CCVector3*>(
                            vertices->getPointPersistentPtr(2));
                    CCVector3* D = const_cast<CCVector3*>(
                            vertices->getPointPersistentPtr(3));
                    CCVector3d pos2D =
                            evRect->toVtkCoordinates(event->x(), event->y());
                    B->x = C->x = static_cast<PointCoordinateType>(pos2D.x);
                    C->y = D->y = static_cast<PointCoordinateType>(pos2D.y);
                }
            }
            // standard rotation around the current pivot
            else if (curInteractionFlags() &
                     ecvGenericGLDisplay::INTERACT_ROTATE) {
                CVLog::Print(
                        "[Camera-Rot] CC rotation code ENTERED! "
                        "labelClicked=%d activeItems=%zu",
                        m_labelClickedOnPress, curActiveItems().size());
                // choose the right rotation mode
                enum RotationMode {
                    StandardMode,
                    BubbleViewMode,
                    LockedAxisMode
                };
                RotationMode rotationMode = StandardMode;
                if ((curInteractionFlags() &
                     ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) !=
                    ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
                    if (curBubbleViewModeEnabled())
                        rotationMode = BubbleViewMode;
                    else if (curRotationAxisLocked())
                        rotationMode = LockedAxisMode;
                }

                ccGLMatrixd rotMat;
                switch (rotationMode) {
                    case BubbleViewMode: {
                        QPoint posDelta = curLastMousePos() - event->pos();

                        if (std::abs(posDelta.x()) != 0) {
                            double delta_deg =
                                    (posDelta.x() *
                                     static_cast<double>(
                                             curBubbleViewFov_deg())) /
                                    height();
                            // rotation about the sensor Z axis
                            CCVector3d axis =
                                    curViewportParams()
                                            .viewMat.getColumnAsVec3D(2);
                            rotMat.initFromParameters(
                                    cloudViewer::DegreesToRadians(delta_deg),
                                    axis, CCVector3d(0, 0, 0));
                        }

                        if (std::abs(posDelta.y()) != 0) {
                            double delta_deg =
                                    (posDelta.y() *
                                     static_cast<double>(
                                             curBubbleViewFov_deg())) /
                                    height();
                            // rotation about the local X axis
                            ccGLMatrixd rotX;
                            rotX.initFromParameters(
                                    cloudViewer::DegreesToRadians(delta_deg),
                                    CCVector3d(1, 0, 0), CCVector3d(0, 0, 0));
                            rotMat = rotX * rotMat;
                        }
                    } break;

                    case StandardMode: {
                        static CCVector3d s_lastMouseOrientation;
                        if (!curMouseMoved()) {
                            // on the first time, we must compute the previous
                            // orientation (the camera hasn't moved yet)
                            s_lastMouseOrientation =
                                    m_ownerView
                                            ? m_ownerView
                                                      ->convertMousePositionToOrientation(
                                                              curLastMousePos()
                                                                      .x(),
                                                              curLastMousePos()
                                                                      .y())
                                            : m_tools->convertMousePositionToOrientation(
                                                      curLastMousePos().x(),
                                                      curLastMousePos().y());
                        }

                        CCVector3d currentMouseOrientation =
                                m_ownerView
                                        ? m_ownerView
                                                  ->convertMousePositionToOrientation(
                                                          x, y)
                                        : m_tools->convertMousePositionToOrientation(
                                                  x, y);
                        rotMat = ccGLMatrixd::FromToRotation(
                                s_lastMouseOrientation,
                                currentMouseOrientation);
                        s_lastMouseOrientation = currentMouseOrientation;
                    } break;

                    case LockedAxisMode: {
                        // apply rotation about the locked axis
                        CCVector3d axis = curLockedRotationAxis();
                        // curViewportParams().objectCenteredView
                        ccGLCameraParameters camera;
                        if (m_ownerView)
                            m_ownerView->getGLCameraParameters(camera);
                        else
                            m_tools->getGLCameraParameters(camera);
                        camera.modelViewMat.applyRotation(axis);

                        // determine whether we are in a side or top view
                        bool topView = (std::abs(axis.z) > 0.5);
                        double angle_rad = 0.0;
                        if (topView) {
                            // rotation origin
                            CCVector3d C2D;
                            if (curViewportParams().objectCenteredView) {
                                // project the current pivot point on screen
                                camera.project(
                                        curViewportParams().getPivotPoint(),
                                        C2D);
                                C2D.z = 0.0;
                            } else {
                                C2D = CCVector3d(width() / 2.0, height() / 2.0,
                                                 0.0);
                            }

                            CCVector3d previousMousePos(
                                    static_cast<double>(curLastMousePos().x()),
                                    static_cast<double>(height() -
                                                        curLastMousePos().y()),
                                    0.0);
                            CCVector3d currentMousePos(
                                    static_cast<double>(x),
                                    static_cast<double>(height() - y), 0.0);

                            CCVector3d a = (currentMousePos - C2D);
                            CCVector3d b = (previousMousePos - C2D);
                            CCVector3d u = a * b;
                            double u_norm = std::abs(
                                    u.z);  // a and b are in the XY plane
                            if (u_norm > 1.0e-6) {
                                double sin_angle =
                                        u_norm / (a.norm() * b.norm());

                                // determine the rotation direction
                                if (u.z * curLockedRotationAxis().z > 0) {
                                    sin_angle = -sin_angle;
                                }

                                angle_rad =
                                        asin(sin_angle);  // in [-pi/2 ; pi/2]
                                rotMat.initFromParameters(angle_rad, axis,
                                                          CCVector3d(0, 0, 0));
                            }
                        } else  // side view
                        {
                            // project the current pivot point on screen
                            CCVector3d A2D, B2D;
                            if (camera.project(
                                        curViewportParams().getPivotPoint(),
                                        A2D) &&
                                camera.project(
                                        curViewportParams().getPivotPoint() +
                                                curViewportParams().zFar *
                                                        curLockedRotationAxis(),
                                        B2D)) {
                                CCVector3d lockedRotationAxis2D = B2D - A2D;
                                lockedRotationAxis2D.z = 0;  // just in case
                                lockedRotationAxis2D.normalize();

                                CCVector3d mouseShift(static_cast<double>(dx),
                                                      -static_cast<double>(dy),
                                                      0.0);
                                mouseShift -=
                                        mouseShift.dot(lockedRotationAxis2D) *
                                        lockedRotationAxis2D;  // we only keep
                                                               // the orthogonal
                                                               // part
                                angle_rad = 2.0 * M_PI * mouseShift.norm() /
                                            (width() + height());
                                if ((lockedRotationAxis2D * mouseShift).z >
                                    0.0) {
                                    angle_rad = -angle_rad;
                                }

                                rotMat.initFromParameters(angle_rad, axis,
                                                          CCVector3d(0, 0, 0));
                            }
                        }

                        // rotate camera with axis
                        // Note: -cloudViewer::RadiansToDegrees(angle_rad):
                        // inverse direction rotation
                        if (m_ownerView)
                            m_ownerView->rotateWithAxis(
                                    CCVector2i(x, y), curLockedRotationAxis(),
                                    -cloudViewer::RadiansToDegrees(angle_rad));
                        else
                            m_tools->rotateWithAxis(
                                    CCVector2i(x, y), curLockedRotationAxis(),
                                    -cloudViewer::RadiansToDegrees(angle_rad));
                    } break;

                    default:
                        assert(false);
                        break;
                }

                if (curInteractionFlags() &
                    ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
                    rotMat = curViewportParams().viewMat.transposed() * rotMat *
                             curViewportParams().viewMat;
                    if (m_ownerView) emit m_ownerView->rotation(rotMat);
                    emit m_tools->rotation(rotMat);
                } else {
                    if (m_ownerView)
                        m_ownerView->showPivotSymbol(true);
                    else
                        m_tools->showPivotSymbol(true);
                    QApplication::changeOverrideCursor(
                            QCursor(Qt::ClosedHandCursor));

                    if (m_ownerView) emit m_ownerView->viewMatRotated(rotMat);
                    emit m_tools->viewMatRotated(rotMat);
                }
            }
        }
    }

    curMouseMoved() = true;
    curLastMousePos() = event->pos();
    if (!m_labelClickedOnPress) {
        if (m_ownerView) emit m_ownerView->cameraParamChanged();
        emit m_tools->cameraParamChanged();
        if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
    }
    event->accept();
}

void QVTKWidgetCustom::updateActivateditems(
        int x, int y, int dx, int dy, bool updatePosition) {
    CVLog::Print(
            "[Label] updateActivateditems: pos(%d,%d) delta(%d,%d) "
            "updatePos=%d activeItems=%zu labelDrag=%d",
            x, y, dx, dy, updatePosition, curActiveItems().size(),
            m_labelClickedOnPress);
    bool movedAs2D = false;
    if (updatePosition) {
        const double pixSize = m_ownerView
                                       ? m_ownerView->computeActualPixelSize()
                                       : m_tools->computeActualPixelSize();
        CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
        curViewportParams().viewMat.transposed().applyRotation(u);

        const int retinaScale = m_ownerView ? m_ownerView->getDevicePixelRatio()
                                            : m_tools->getDevicePixelRatio();
        u *= retinaScale;

        for (auto& activeItem : curActiveItems()) {
            // cc2DLabel caption should ONLY move during explicit drag.
            // During camera rotation, Update2DLabel adds labels to
            // activeItems, but we must NOT apply the rotation delta
            // to their screen position (CloudCompare: fixed on screen).
            if (!m_labelClickedOnPress &&
                dynamic_cast<cc2DLabel*>(activeItem)) {
                continue;
            }
            if (activeItem->move2D(x * retinaScale, y * retinaScale,
                                   dx * retinaScale, dy * retinaScale,
                                   m_ownerView ? m_ownerView->glWidth()
                                               : m_tools->glWidth(),
                                   m_ownerView ? m_ownerView->glHeight()
                                               : m_tools->glHeight())) {
                movedAs2D = true;
            } else if (activeItem->move3D(u)) {
                if (m_ownerView) {
                    m_ownerView->invalidateViewport();
                    m_ownerView->deprecate3DLayer();
                } else {
                    m_tools->invalidateViewport();
                    m_tools->deprecate3DLayer();
                }
            }
        }
    }

    if (m_ownerView)
        m_ownerView->redraw(true);
    else
        m_tools->redraw2DLabel();
}

void QVTKWidgetCustom::mouseReleaseEvent(QMouseEvent* event) {
    if ((curInteractionFlags() & ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA) &&
        !m_labelClickedOnPress) {
        CVLog::Print("[VTK-FWD] mouseReleaseEvent -> VTK btn=0x%x",
                     (unsigned)event->button());
        QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
    } else if (m_labelClickedOnPress) {
        CVLog::Print(
                "[Label-Drag] BLOCKED VTK mouseRelease, "
                "labelClicked=%d",
                m_labelClickedOnPress);
    }

    if (curIgnoreMouseReleaseEvent()) {
        curIgnoreMouseReleaseEvent() = false;
        return;
    }
    bool mouseHasMoved = curMouseMoved();

    // reset to default state
    curMouseButtonPressed() = false;
    curMouseMoved() = false;
    QApplication::restoreOverrideCursor();

    if (curInteractionFlags() &
        ecvGenericGLDisplay::INTERACT_SIG_BUTTON_RELEASED) {
        event->accept();
        if (m_ownerView) emit m_ownerView->buttonReleased();
        emit m_tools->buttonReleased();
    }

    if (curPivotSymbolShown()) {
        if (curPivotVisibility() == ecvGenericGLDisplay::PIVOT_SHOW_ON_MOVE) {
            if (m_ownerView)
                m_ownerView->toBeRefreshed();
            else
                m_tools->toBeRefreshed();
        }
        if (m_ownerView)
            m_ownerView->showPivotSymbol(
                    curPivotVisibility() ==
                    ecvGenericGLDisplay::PIVOT_ALWAYS_SHOW);
        else
            m_tools->showPivotSymbol(curPivotVisibility() ==
                                     ecvGenericGLDisplay::PIVOT_ALWAYS_SHOW);
    }

    if ((event->button() == Qt::MiddleButton)) {
        if (mouseHasMoved) {
            event->accept();
            curActiveItems().clear();
        } else if (curInteractionFlags() &
                   ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            // interaction with 2D item(s)
            if (m_ownerView)
                m_ownerView->updateActiveItemsList(event->x(), event->y(),
                                                   false);
            else
                m_tools->updateActiveItemsList(event->x(), event->y(), false);
            if (!curActiveItems().empty()) {
                ccInteractor* item = curActiveItems().front();
                curActiveItems().clear();
                if (item->acceptClick(event->x(), height() - 1 - event->y(),
                                      Qt::MiddleButton)) {
                    event->accept();
                }
            }
        }
    } else if (event->button() == Qt::LeftButton) {
        if (mouseHasMoved) {
            // if a rectangular picking area has been defined
            if (curRectPickingPoly()) {
                cloudViewer::GenericIndexedCloudPersist* vertices =
                        curRectPickingPoly()->getAssociatedCloud();
                assert(vertices);
                const CCVector3* A = vertices->getPointPersistentPtr(0);
                const CCVector3* C = vertices->getPointPersistentPtr(2);

                int pickX = static_cast<int>(A->x + C->x) / 2;
                int pickY = static_cast<int>(A->y + C->y) / 2;
                int pickW = static_cast<int>(std::abs(C->x - A->x));
                int pickH = static_cast<int>(std::abs(C->y - A->y));

                if (m_ownerView)
                    m_ownerView->removeFromOwnDB(curRectPickingPoly());
                else
                    m_tools->removeFromOwnDB(curRectPickingPoly());
                curRectPickingPoly() = nullptr;
                vertices = nullptr;

                const int pickRefW =
                        m_ownerView && m_ownerView->asWidget()
                                ? m_ownerView->asWidget()->width()
                                : (m_tools->asWidget()
                                           ? m_tools->asWidget()->width()
                                           : 0);
                const int pickRefH =
                        m_ownerView && m_ownerView->asWidget()
                                ? m_ownerView->asWidget()->height()
                                : (m_tools->asWidget()
                                           ? m_tools->asWidget()->height()
                                           : 0);
                if (m_ownerView)
                    m_ownerView->startPicking(
                            ecvGenericGLDisplay::ENTITY_RECT_PICKING,
                            pickX + pickRefW / 2, pickRefH / 2 - pickY, pickW,
                            pickH);
                else
                    m_tools->startPicking(
                            ecvGenericGLDisplay::ENTITY_RECT_PICKING,
                            pickX + pickRefW / 2, pickRefH / 2 - pickY, pickW,
                            pickH);
                if (m_ownerView)
                    m_ownerView->toBeRefreshed();
                else
                    m_tools->toBeRefreshed();
            }

            event->accept();
        } else {
            // picking?
            // CRITICAL: Don't start deferred picking if a VTK widget was
            // clicked This prevents doPicking() from overriding the widget
            // selection
            if (!curWidgetClicked() &&
                (m_ownerView ? m_ownerView->elapsedMs()
                             : m_tools->m_timer.elapsed()) <
                        curLastClickTime() + CC_MAX_PICKING_CLICK_DURATION_MS) {
                int x = curLastMousePos().x();
                int y = curLastMousePos().y();

                // Save global point size / line width before
                // ProcessClickableItems potentially modifies them.
                const auto& vp = m_ownerView
                                         ? m_ownerView->getViewportParameters()
                                         : m_tools->getViewportParameters();
                float savedGlobalPtSize = vp.defaultPointSize;
                float savedGlobalLnWidth = vp.defaultLineWidth;

                const bool picked =
                        m_ownerView ? m_ownerView->processClickableItems(x, y)
                                    : m_tools->processClickableItems(x, y);
                if (!picked) {
                    curLastMousePos() = event->pos();
                    if (m_ownerView) {
                        m_ownerView->startDeferredPicking();
                    } else {
                        m_tools->setPickingTargetView(
                                ecvGenericGLDisplay::FromWidget(this));
                        m_tools->m_deferredPickingTimer.start();
                    }
                } else {
                    // Sync per-widget local values from the
                    // (just-updated) global.
                    const auto& vpAfter =
                            m_ownerView ? m_ownerView->getViewportParameters()
                                        : m_tools->getViewportParameters();
                    m_localDefaultPointSize = vpAfter.defaultPointSize;
                    m_localDefaultLineWidth = vpAfter.defaultLineWidth;

                    auto* display = ecvGenericGLDisplay::FromWidget(this);
                    bool isSecondary = display && (m_ownerView != nullptr);
                    if (isSecondary) {
                        // Restore global so primary view's state
                        // is not contaminated.
                        m_tools->setViewportDefaultPointSize(savedGlobalPtSize);
                        m_tools->setViewportDefaultLineWidth(
                                savedGlobalLnWidth);
                        display->redraw();
                    } else {
                        ecvRedrawScope scope;
                    }
                }
            } else if (curWidgetClicked()) {
                CVLog::PrintVerbose(
                        "[QVTKWidgetCustom::mouseReleaseEvent] Skipping "
                        "deferred "
                        "picking because VTK widget was clicked");
                // Reset the flag after checking
                curWidgetClicked() = false;
            }
        }

        curActiveItems().clear();
        m_labelClickedOnPress = false;
    } else if (event->button() == Qt::RightButton) {
        if (m_rightClickOnLabel) {
            // Right-click on label = toggle collapse (CloudCompare behavior)
            ccHObject::Container labels;
            if (m_ownerView)
                m_ownerView->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            else
                m_tools->filterByEntityType(labels, CV_TYPES::LABEL_2D);
            for (auto* obj : labels) {
                if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isEnabled() ||
                    !obj->isVisible())
                    continue;
                cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                if (!l) continue;
                QRect roi = l->getLabelROI();
                if (roi.isValid() && roi.contains(event->x(), event->y())) {
                    if (l->acceptClick(event->x(), event->y(),
                                       Qt::RightButton)) {
                        if (m_ownerView) {
                            m_ownerView->redraw(true);
                        } else {
                            m_tools->redraw2DLabel();
                            m_tools->redraw(true);
                        }
                        event->accept();
                        CVLog::Print(
                                "[Label] Right-click toggle collapse "
                                "'%s' collapsed=%d",
                                qPrintable(l->getName()), l->isCollapsed());
                    }
                    break;
                }
            }
            m_rightClickOnLabel = false;
        } else if (mouseHasMoved) {
            if (m_ownerView)
                m_ownerView->update2DLabels(true);
            else
                m_tools->Update2DLabel(true);
        }
    }

    // CRITICAL: Always update 2D labels after any mouse interaction that moved
    // the camera (rotation, zoom, pan) to ensure labels stay aligned with their
    // 3D anchor points. This fixes the issue where labels become detached after
    // mouse release.
    if (mouseHasMoved) {
        if (m_ownerView)
            m_ownerView->update2DLabels(true);
        else
            m_tools->Update2DLabel(true);
    }

    if (m_ownerView)
        m_ownerView->refresh(true);
    else
        m_tools->refresh(true);
}

void QVTKWidgetCustom::dragEnterEvent(QDragEnterEvent* event) {
    const QMimeData* mimeData = event->mimeData();
    if (mimeData->hasFormat("text/uri-list")) event->acceptProposedAction();

    QVTKOpenGLNativeWidget::dragEnterEvent(event);
}

void QVTKWidgetCustom::dropEvent(QDropEvent* event) {
    const QMimeData* mimeData = event->mimeData();

    if (mimeData && mimeData->hasFormat("text/uri-list")) {
        QStringList fileNames;
        for (const QUrl& url : mimeData->urls()) {
            QString fileName = url.toLocalFile();
            fileNames.append(fileName);
#ifdef QT_DEBUG
            CVLog::Print(QString("File dropped: %1").arg(fileName));
#endif
        }

        if (!fileNames.empty()) {
            if (m_ownerView) emit m_ownerView->filesDropped(fileNames, true);
            emit m_tools->filesDropped(fileNames, true);
        }

        event->acceptProposedAction();
    }

    QVTKOpenGLNativeWidget::dropEvent(event);
    event->ignore();
}

static bool isVtkViewerShortcut(int key, Qt::KeyboardModifiers mods) {
    bool ctrl = mods & Qt::ControlModifier;
    bool alt = mods & Qt::AltModifier;
    bool shift = mods & Qt::ShiftModifier;

    if (ctrl && alt) {
        switch (key) {
            case Qt::Key_J:
            case Qt::Key_C:
            case Qt::Key_G:
            case Qt::Key_K:
            case Qt::Key_O:
            case Qt::Key_F:
            case Qt::Key_S:
            case Qt::Key_Plus:
            case Qt::Key_Minus:
            case Qt::Key_Equal:
                return true;
        }
    } else if (ctrl && shift && !alt) {
        switch (key) {
            case Qt::Key_P:
            case Qt::Key_W:
            case Qt::Key_S:
            case Qt::Key_Plus:
            case Qt::Key_Minus:
                return true;
        }
    } else if (ctrl && !alt && !shift) {
        switch (key) {
            case Qt::Key_S:
            case Qt::Key_R:
                return true;
        }
    }
    return false;
}

bool QVTKWidgetCustom::event(QEvent* evt) {
    switch (evt->type()) {
        case QEvent::ShortcutOverride: {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(evt);
            if (isVtkViewerShortcut(keyEvent->key(), keyEvent->modifiers())) {
                evt->accept();
                return true;
            }
        } break;

        // ── cc2DLabel VTK-interactor bypass ──────────────────────────
        // VTK's QVTKOpenGLNativeWidget::event() forwards mouse events
        // to the interactor (camera rotation) BEFORE dispatching to
        // our mousePressEvent / mouseMoveEvent / mouseReleaseEvent.
        // When a cc2DLabel is being interacted with, we must bypass
        // VTK entirely and dispatch directly through QOpenGLWidget.
        case QEvent::MouseButtonPress: {
            QMouseEvent* me = static_cast<QMouseEvent*>(evt);
            if (me->button() == Qt::LeftButton ||
                me->button() == Qt::RightButton) {
                bool onLabel = false;
                if (curInteractionFlags() &
                    ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
                    ccHObject::Container labels;
                    if (m_ownerView)
                        m_ownerView->filterByEntityType(labels,
                                                        CV_TYPES::LABEL_2D);
                    else
                        m_tools->filterByEntityType(labels, CV_TYPES::LABEL_2D);
                    for (auto* obj : labels) {
                        if (!obj->isA(CV_TYPES::LABEL_2D) ||
                            !obj->isEnabled() || !obj->isVisible())
                            continue;
                        cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                        if (!l) continue;
                        QRect roi = l->getLabelROI();
                        if (roi.isValid() && roi.contains(me->x(), me->y())) {
                            onLabel = true;
                            break;
                        }
                    }
                }
                if (onLabel) {
                    CVLog::Print(
                            "[VTK-BYPASS] MousePress on label, "
                            "skipping VTK interactor btn=0x%x",
                            (unsigned)me->button());
                    return QOpenGLWidget::event(evt);
                }
            }
        } break;

        case QEvent::MouseMove: {
            if (m_labelClickedOnPress || m_rightClickOnLabel) {
                return QOpenGLWidget::event(evt);
            }
        } break;

        case QEvent::MouseButtonRelease: {
            if (m_labelClickedOnPress || m_rightClickOnLabel) {
                CVLog::Print(
                        "[VTK-BYPASS] MouseRelease on label, "
                        "skipping VTK interactor");
                return QOpenGLWidget::event(evt);
            }
        } break;

        // Gesture start/stop
        case QEvent::TouchBegin:
        case QEvent::TouchEnd: {
            QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
            touchEvent->accept();
            curTouchInProgress() = (evt->type() == QEvent::TouchBegin);
            curTouchBaseDist() = 0.0;
            CVLog::PrintDebug(
                    QString("Touch event %1")
                            .arg(curTouchInProgress() ? "begins" : "ends"));
        } break;

        case QEvent::Close: {
            if (m_unclosable) {
                evt->ignore();
            } else {
                evt->accept();
            }
        }
            return true;

        case QEvent::DragEnter: {
            dragEnterEvent(static_cast<QDragEnterEvent*>(evt));
        }
            return true;

        case QEvent::Drop: {
            dropEvent(static_cast<QDropEvent*>(evt));
        }
            return true;

        case QEvent::TouchUpdate: {
            // Gesture update
            if (curTouchInProgress() && !curViewportParams().perspectiveView) {
                QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
                const QList<QTouchEvent::TouchPoint>& touchPoints =
                        touchEvent->touchPoints();
                if (touchPoints.size() == 2) {
                    QPointF D = (touchPoints[1].pos() - touchPoints[0].pos());
                    qreal dist = std::sqrt(D.x() * D.x() + D.y() * D.y());
                    if (curTouchBaseDist() != 0.0) {
                        float zoomFactor = dist / curTouchBaseDist();
                        if (m_ownerView)
                            m_ownerView->updateZoom(zoomFactor);
                        else
                            m_tools->updateZoom(zoomFactor);
                    }
                    curTouchBaseDist() = dist;
                    evt->accept();
                    break;
                }
            }
        } break;

        case QEvent::Resize: {
            QSize newSize = static_cast<QResizeEvent*>(evt)->size();
            if (m_ownerView)
                m_ownerView->resizeGL(newSize.width(), newSize.height());
            else
                m_tools->resizeGL(newSize.width(), newSize.height());
            if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
            evt->accept();
        } break;

        case QEvent::KeyPress: {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(evt);

            if (keyEvent->key() == Qt::Key_Escape) {
                CVLog::Print(
                        "[QVTKWidgetCustom] ESC key pressed, forwarding to "
                        "MainWindow");
                if (m_ownerView)
                    emit m_ownerView->exclusiveFullScreenToggled(false);
                emit m_tools->exclusiveFullScreenToggled(false);
                if (m_win) {
                    QKeyEvent* newEvent =
                            new QKeyEvent(QEvent::KeyPress, Qt::Key_Escape,
                                          keyEvent->modifiers());
                    QCoreApplication::postEvent(m_win, newEvent);
                }
                evt->accept();
                return true;
            }

            // Handle VTK viewer shortcuts at Qt level before VTK processes
            // them. Uses the stored m_customStyle which persists even when
            // selection tools replace the active interactor style. Passes the
            // interactor explicitly so handleShortcut works in detached state.
            if (m_customStyle && m_interactor) {
                int qkey = keyEvent->key();
                auto mods = keyEvent->modifiers();
                bool ctrl = mods & Qt::ControlModifier;
                bool alt = mods & Qt::AltModifier;
                bool shift = mods & Qt::ShiftModifier;

                if (ctrl || alt) {
                    char key = 0;
                    if (qkey >= Qt::Key_A && qkey <= Qt::Key_Z) {
                        key = 'a' + static_cast<char>(qkey - Qt::Key_A);
                    } else if (qkey == Qt::Key_Plus) {
                        key = '+';
                    } else if (qkey == Qt::Key_Minus) {
                        key = '-';
                    } else if (qkey == Qt::Key_Equal) {
                        key = '=';
                    }

                    if (key &&
                        m_customStyle->handleShortcut(key, ctrl, alt, shift,
                                                      m_interactor.Get())) {
                        evt->accept();
                        return true;
                    }
                }
            }
        } break;

        default: {
            // CVLog::Print("Unhandled event: %i", evt->type());
        } break;
    }

    return QVTKOpenGLNativeWidget::event(evt);
}

void QVTKWidgetCustom::keyPressEvent(QKeyEvent* event) {
    // ESC is already handled in event() before VTK processes it
    // Other keys are passed to the base class
    QVTKOpenGLNativeWidget::keyPressEvent(event);
}
