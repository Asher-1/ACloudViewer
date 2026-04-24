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
#include "../../Visualization/ecvGLView.h"
#include <ecvGenericGLDisplay.h>
#include <ecvInteractor.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvViewManager.h>

// QT
#include <QApplication>
#include <QCoreApplication>
#include <QHBoxLayout>
#include <QLayout>
#include <QMainWindow>
#include <QMessageBox>
#include <QMimeData>
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

#include "ScaleBarWidget.h"

// macroes
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) \
    vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif

ecvViewContext* QVTKWidgetCustom::ownerCtx() {
    return m_ownerView ? &m_ownerView->context() : nullptr;
}

ecvGenericGLDisplay::INTERACTION_FLAGS& QVTKWidgetCustom::curInteractionFlags() {
    if (m_ownerView) return m_ownerView->context().interactionFlags;
    return m_tools->m_interactionFlags;
}

ecvViewportParameters& QVTKWidgetCustom::curViewportParams() {
    if (m_ownerView) return m_ownerView->context().viewportParams;
    return m_tools->m_viewportParams;
}

const ecvViewportParameters& QVTKWidgetCustom::curViewportParams() const {
    if (m_ownerView) return m_ownerView->context().viewportParams;
    return m_tools->m_viewportParams;
}

QPoint& QVTKWidgetCustom::curLastMousePos() {
    if (m_ownerView) return m_ownerView->context().lastMousePos;
    return m_tools->m_lastMousePos;
}

QPoint& QVTKWidgetCustom::curLastMouseMovePos() {
    if (m_ownerView) return m_ownerView->context().lastMouseMovePos;
    return m_tools->m_lastMouseMovePos;
}

bool& QVTKWidgetCustom::curMouseMoved() {
    if (m_ownerView) return m_ownerView->context().mouseMoved;
    return m_tools->m_mouseMoved;
}

bool& QVTKWidgetCustom::curMouseButtonPressed() {
    if (m_ownerView) return m_ownerView->context().mouseButtonPressed;
    return m_tools->m_mouseButtonPressed;
}

bool& QVTKWidgetCustom::curIgnoreMouseReleaseEvent() {
    if (m_ownerView) return m_ownerView->context().ignoreMouseReleaseEvent;
    return m_tools->m_ignoreMouseReleaseEvent;
}

bool& QVTKWidgetCustom::curWidgetClicked() {
    if (m_ownerView) return m_ownerView->context().widgetClicked;
    return m_tools->m_widgetClicked;
}

ecvGenericGLDisplay::PICKING_MODE& QVTKWidgetCustom::curPickingMode() {
    if (m_ownerView) return m_ownerView->context().pickingMode;
    return m_tools->m_pickingMode;
}

bool& QVTKWidgetCustom::curPickingModeLocked() {
    if (m_ownerView) return m_ownerView->context().pickingModeLocked;
    return m_tools->m_pickingModeLocked;
}

int& QVTKWidgetCustom::curPickRadius() {
    if (m_ownerView) return m_ownerView->context().pickRadius;
    return m_tools->m_pickRadius;
}

bool& QVTKWidgetCustom::curAllowRectangularEntityPicking() {
    if (m_ownerView) return m_ownerView->context().allowRectangularEntityPicking;
    return m_tools->m_allowRectangularEntityPicking;
}

int& QVTKWidgetCustom::curLastPointIndex() {
    if (m_ownerView) return m_ownerView->context().lastPointIndex;
    return m_tools->m_last_point_index;
}

QString& QVTKWidgetCustom::curLastPickedId() {
    if (m_ownerView) return m_ownerView->context().lastPickedId;
    return m_tools->m_last_picked_id;
}

bool& QVTKWidgetCustom::curTouchInProgress() {
    if (m_ownerView) return m_ownerView->context().touchInProgress;
    return m_tools->m_touchInProgress;
}

qreal& QVTKWidgetCustom::curTouchBaseDist() {
    if (m_ownerView) return m_ownerView->context().touchBaseDist;
    return m_tools->m_touchBaseDist;
}

bool& QVTKWidgetCustom::curClickableItemsVisible() {
    if (m_ownerView) return m_ownerView->context().clickableItemsVisible;
    return m_tools->m_clickableItemsVisible;
}

bool& QVTKWidgetCustom::curBubbleViewModeEnabled() {
    if (m_ownerView) return m_ownerView->context().bubbleViewModeEnabled;
    return m_tools->m_bubbleViewModeEnabled;
}

float& QVTKWidgetCustom::curBubbleViewFov_deg() {
    if (m_ownerView) return m_ownerView->context().bubbleViewFov_deg;
    return m_tools->m_bubbleViewFov_deg;
}

bool& QVTKWidgetCustom::curCustomLightEnabled() {
    if (m_ownerView) return m_ownerView->context().customLightEnabled;
    return m_tools->m_customLightEnabled;
}

float* QVTKWidgetCustom::curCustomLightPos() {
    if (m_ownerView) return m_ownerView->context().customLightPos;
    return m_tools->m_customLightPos;
}

bool& QVTKWidgetCustom::curRotationAxisLocked() {
    if (m_ownerView) return m_ownerView->context().rotationAxisLocked;
    return m_tools->m_rotationAxisLocked;
}

CCVector3d& QVTKWidgetCustom::curLockedRotationAxis() {
    if (m_ownerView) return m_ownerView->context().lockedRotationAxis;
    return m_tools->m_lockedRotationAxis;
}

ecvGenericGLDisplay::PivotVisibility& QVTKWidgetCustom::curPivotVisibility() {
    if (m_ownerView) return m_ownerView->context().pivotVisibility;
    return m_tools->m_pivotVisibility;
}

bool& QVTKWidgetCustom::curPivotSymbolShown() {
    if (m_ownerView) return m_ownerView->context().pivotSymbolShown;
    return m_tools->m_pivotSymbolShown;
}

bool& QVTKWidgetCustom::curAutoPickPivotAtCenter() {
    if (m_ownerView) return m_ownerView->context().autoPickPivotAtCenter;
    return m_tools->m_autoPickPivotAtCenter;
}

bool& QVTKWidgetCustom::curShowCursorCoordinates() {
    if (m_ownerView) return m_ownerView->context().showCursorCoordinates;
    return m_tools->m_showCursorCoordinates;
}

qint64& QVTKWidgetCustom::curLastClickTime() {
    if (m_ownerView) return m_ownerView->context().lastClickTime_ticks;
    return m_tools->m_lastClickTime_ticks;
}

ccPolyline*& QVTKWidgetCustom::curRectPickingPoly() {
    return m_tools->m_rectPickingPoly;
}

std::list<ccInteractor*>& QVTKWidgetCustom::curActiveItems() {
    return m_tools->m_activeItems;
}

ecvDisplayTools::HotZone*& QVTKWidgetCustom::curHotZone() {
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
        if (m_tools) {
            m_tools->Update2DLabel(true);
        }
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
    return ecvGenericGLDisplay::FromWidget(
            const_cast<QVTKWidgetCustom*>(this));
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

    if (!ecvDisplayTools::USE_VTK_PICK) {
        curLastPointIndex() = -1;
        curLastPickedId() = QString();
    }

    if ((event->buttons() & Qt::RightButton)) {
        // right click = panning (2D translation)
        if ((curInteractionFlags() & ecvDisplayTools::INTERACT_PAN) ||
            ((QApplication::keyboardModifiers() & Qt::ControlModifier) &&
             (curInteractionFlags() &
              ecvDisplayTools::INTERACT_CTRL_PAN))) {
            QApplication::setOverrideCursor(QCursor(Qt::SizeAllCursor));
        }

        if (curInteractionFlags() &
            ecvDisplayTools::INTERACT_SIG_RB_CLICKED) {
            emit m_tools->rightButtonClicked(event->x(), event->y());
        }
    } else if (event->buttons() & Qt::LeftButton) {
        curLastClickTime() = m_tools->m_timer.elapsed();  // in msec

        // left click = rotation
        if (curInteractionFlags() & ecvDisplayTools::INTERACT_ROTATE) {
            QApplication::setOverrideCursor(QCursor(Qt::PointingHandCursor));
        }

        if (curInteractionFlags() &
            ecvDisplayTools::INTERACT_SIG_LB_CLICKED) {
            emit m_tools->leftButtonClicked(event->x(), event->y());
        }

        // do this before drawing the pivot!
        if (curAutoPickPivotAtCenter()) {
            CCVector3d P;
            // if (m_tools->GetClick3DPos(m_tools->m_glViewport.width() / 2,
            // m_tools->m_glViewport.height() / 2, P))
            if (m_tools->GetClick3DPos(event->x(), event->y(), P)) {
                ecvDisplayTools::SetPivotPoint(P, true, false);
            }
        }
    } else {
    }

    QVTKOpenGLNativeWidget::mousePressEvent(event);
}

void QVTKWidgetCustom::mouseDoubleClickEvent(QMouseEvent* event) {
    // Same click-to-activate rule as mousePressEvent (some paths only double-click).
    auto* display = ecvGenericGLDisplay::FromWidget(this);
    if (display) {
        auto& vm = ecvViewManager::instance();
        if (vm.getActiveView() != display) {
            vm.setActiveView(display);
        }
    }

    m_tools->m_deferredPickingTimer
            .stop();  // prevent the picking process from starting
    curIgnoreMouseReleaseEvent() = true;

    const int x = event->x();
    const int y = event->y();

    CCVector3d P;
    if (ecvDisplayTools::GetClick3DPos(x, y, P)) {
        ecvDisplayTools::SetPivotPoint(P, true, true);
    }

    emit m_tools->doubleButtonClicked(event->x(), event->y());

    QVTKOpenGLNativeWidget::mouseDoubleClickEvent(event);
}

void QVTKWidgetCustom::wheelEvent(QWheelEvent* event) {
    bool doRedraw = false;
    Qt::KeyboardModifiers keyboardModifiers =
            QApplication::keyboardModifiers();

    emit m_tools->mouseWheelChanged(event);

    double delta = qtCompatWheelEventDelta(event);

    if (keyboardModifiers & Qt::AltModifier) {
        event->accept();

        float sizeModifier = (delta < 0.0 ? -1.0f : 1.0f);
        ecvDisplayTools::SetPointSize(
                curViewportParams().defaultPointSize + sizeModifier);
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
                ecvDisplayTools::SetZNearCoef(newCoef);
                { ecvRedrawScope redrawScope; }
                doRedraw = true;
            }
        }
    } else if (keyboardModifiers & Qt::ShiftModifier) {
        event->accept();
        if (curViewportParams().perspectiveView) {
            float newFOV = (curViewportParams().fov_deg +
                            (delta < 0 ? -1.0f : 1.0f));
            newFOV = std::min(std::max(1.0f, newFOV), 180.0f);
            if (newFOV != curViewportParams().fov_deg) {
                ecvDisplayTools::SetFov(newFOV);
                { ecvRedrawScope redrawScope; }
                doRedraw = true;
            }
        }
    } else if (curInteractionFlags() &
               ecvDisplayTools::INTERACT_ZOOM_CAMERA) {
        QVTKOpenGLNativeWidget::wheelEvent(event);

        float wheelDelta_deg = static_cast<float>(delta) / 8;
        m_tools->onWheelEvent(wheelDelta_deg);
        emit m_tools->mouseWheelRotated(wheelDelta_deg);
        emit m_tools->cameraParamChanged();

        doRedraw = true;
        event->accept();
    }

    if (doRedraw) {
        emit m_tools->labelmove2D(0, 0, 0, 0);
        ecvDisplayTools::UpdateNamePoseRecursive();

        if (m_wheelZoomUpdateTimer) {
            m_wheelZoomUpdateTimer->stop();
            m_wheelZoomUpdateTimer->start();
        }

        ecvDisplayTools::Update();
    }
}

void QVTKWidgetCustom::mouseMoveEvent(QMouseEvent* event) {
    // Do NOT call setActiveView here: ParaView-style UX activates a render view
    // on click (see mousePressEvent), not when the cursor merely moves across a
    // split window. Activating on every mouseMove caused the wrong view to
    // become "current" during passive hover.

    if (!((curInteractionFlags() & ecvDisplayTools::INTERACT_ROTATE) &&
          (curInteractionFlags() &
           ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES))) {
        if ((curInteractionFlags() &
             ecvDisplayTools::TRANSFORM_CAMERA())) {
            // lock axis
            if ((curInteractionFlags() &
                 ecvDisplayTools::INTERACT_ROTATE) &&
                (curInteractionFlags() &
                 ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) !=
                        ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES &&
                (event->buttons() & Qt::LeftButton) &&
                curRotationAxisLocked()) {
                event->accept();
            } else {  // normal
                QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                m_tools->UpdateDisplayParameters();
            }
        }
    }

    const int x = event->x();
    const int y = event->y();
    if (ecvDisplayTools::GetCurrentScreen() == this) {
        curLastMouseMovePos() = event->pos();
        emit m_tools->mousePosChanged(event->pos());
    }

    if ((curInteractionFlags() &
         ecvDisplayTools::INTERACT_SIG_MOUSE_MOVED) &&
        (ecvDisplayTools::GetCurrentScreen() == this)) {
        emit m_tools->mouseMoved(x, y, event->buttons());
        event->accept();
    }

    // no button pressed
    if (event->buttons() == Qt::NoButton) {
        if (curInteractionFlags() &
            ecvDisplayTools::INTERACT_CLICKABLE_ITEMS) {
            // Per-widget HotZone for multi-window support
            if (!m_localHotZone) {
                m_localHotZone = new ecvDisplayTools::HotZone(this);
            }
            // Replace the singleton-owned fallback HotZone with ours
            if (m_tools->m_hotZoneOwnedBySingleton && curHotZone() &&
                curHotZone() != m_localHotZone) {
                delete curHotZone();
                m_tools->m_hotZoneOwnedBySingleton = false;
            }
            curHotZone() = m_localHotZone;

            QRect areaRect = m_localHotZone->rect(
                    true, curBubbleViewModeEnabled(),
                    ecvDisplayTools::ExclusiveFullScreen());

            const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
            bool inZone = (x * retinaScale * 3 < m_localHotZone->topCorner.x() +
                                                         areaRect.width() * 4 &&
                           y * retinaScale * 2 < m_localHotZone->topCorner.y() +
                                                         areaRect.height() * 4);

            if (inZone != m_localClickableVisible) {
                m_localClickableVisible = inZone;

                auto* display = ecvGenericGLDisplay::FromWidget(this);
                bool isSecondary =
                        display &&
                        display != ecvDisplayTools::TheInstance();
                if (isSecondary) {
                    display->redraw(true, false);
                } else {
                    curClickableItemsVisible() = inZone;
                    ecvDisplayTools::RedrawDisplay(true, false);
                }
            }

            event->accept();
        }

        // display the 3D coordinates of the pixel below the mouse cursor (if
        // possible)
        if (curShowCursorCoordinates()) {
            CCVector3d P;
            QString message = QString("2D (%1 ; %2)").arg(x).arg(y);
            if (ecvDisplayTools::GetClick3DPos(x, y, P)) {
                message += QString(" --> 3D (%1 ; %2 ; %3)")
                                   .arg(P.x)
                                   .arg(P.y)
                                   .arg(P.z);
            }
            ecvDisplayTools::DisplayNewMessage(
                    message, ecvDisplayTools::LOWER_LEFT_MESSAGE, false, 5,
                    ecvDisplayTools::SCREEN_SIZE_MESSAGE);
            ecvDisplayTools::RedrawDisplay(true);
        }

        // don't need to process any further
        return;
    }

    int dx = x - curLastMousePos().x();
    int dy = y - curLastMousePos().y();

    if ((event->buttons() & Qt::RightButton)) {
        // OPTIMIZATION: Skip 2D label updates during zoom to improve
        // performance Only emit signal for label movement, but skip expensive
        // Update2DLabel during zoom The label will be updated when zoom stops
        // (in mouseReleaseEvent)
        if (abs(dx) > 0 || abs(dy) > 0) {
            // Only emit signal for label movement, but skip expensive
            // Update2DLabel during zoom
            emit m_tools->labelmove2D(x, y, 0, 0);
            ecvDisplayTools::UpdateNamePoseRecursive();
        }
    } else if ((event->buttons() & Qt::MiddleButton)) {
        // right button = panning / translating
        if (curInteractionFlags() & ecvDisplayTools::INTERACT_PAN) {
            // displacement vector (in "3D")
            double pixSize = ecvDisplayTools::ComputeActualPixelSize();
            CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
            if (!curViewportParams().perspectiveView) {
                u.y *= curViewportParams().cameraAspectRatio;
            }

            const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
            u *= retinaScale;

            bool entityMovingMode =
                    (curInteractionFlags() &
                     ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) ||
                    ((QApplication::keyboardModifiers() &
                      Qt::ControlModifier) &&
                     curCustomLightEnabled());
            if (entityMovingMode) {
                // apply inverse view matrix
                curViewportParams().viewMat.transposed().applyRotation(u);

                if (curInteractionFlags() &
                    ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) {
                    emit m_tools->translation(u);
                } else if (curCustomLightEnabled()) {
                    // update custom light position
                    curCustomLightPos()[0] += static_cast<float>(u.x);
                    curCustomLightPos()[1] += static_cast<float>(u.y);
                    curCustomLightPos()[2] += static_cast<float>(u.z);
                    ecvDisplayTools::InvalidateViewport();
                    ecvDisplayTools::Deprecate3DLayer();
                }
            } else  // camera moving mode
            {
                if (curViewportParams().objectCenteredView) {
                    // inverse displacement in object-based mode
                    u = -u;
                }
                ecvDisplayTools::MoveCamera(static_cast<float>(u.x),
                                            static_cast<float>(u.y),
                                            static_cast<float>(u.z));
            }

        }  // if (m_interactionFlags & INTERACT_PAN)

        if (curInteractionFlags() & ecvDisplayTools::INTERACT_2D_ITEMS) {
            // on the first time, let's check if the mouse is on a (selected) 2D
            // item
            if (!curMouseMoved()) {
                if (curPickingMode() != ecvDisplayTools::NO_PICKING
                    // DGM: in fact we still need to move labels in those modes
                    // below (see the 'Point Picking' tool of CLOUDVIEWER  for
                    // instance)
                    //&&	m_pickingMode != POINT_PICKING
                    //&&	m_pickingMode != TRIANGLE_PICKING
                    //&&	m_pickingMode != POINT_OR_TRIANGLE_PICKING
                    && (QApplication::keyboardModifiers() == Qt::NoModifier ||
                        QApplication::keyboardModifiers() ==
                                Qt::ControlModifier)) {
                    ecvDisplayTools::UpdateActiveItemsList(
                            curLastMousePos().x(),
                            curLastMousePos().y(), true);
                }
            }
        }

        // OPTIMIZATION: Skip 2D label updates during panning to improve
        // performance Only emit signal for label movement, but skip expensive
        // Update2DLabel during panning The label will be updated when panning
        // stops (in mouseReleaseEvent)
        if (abs(dx) > 0 || abs(dy) > 0) {
            // Only emit signal for label movement, but skip expensive
            // Update2DLabel during panning
            emit m_tools->labelmove2D(x, y, dx, dy);
            ecvDisplayTools::UpdateNamePoseRecursive();
            // specific case: move active item(s)
            if (!curActiveItems().empty()) {
                updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
            }
        }
    } else if (event->buttons() & Qt::LeftButton)  // rotation
    {
        m_tools->scheduleFullRedraw(1000);

        if (curInteractionFlags() & ecvDisplayTools::INTERACT_2D_ITEMS) {
            // on the first time, let's check if the mouse is on a (selected) 2D
            // item
            if (!curMouseMoved()) {
                if (curPickingMode() != ecvDisplayTools::NO_PICKING
                    // DGM: in fact we still need to move labels in those modes
                    // below (see the 'Point Picking' tool of CLOUDVIEWER  for
                    // instance)
                    //&&	m_pickingMode != POINT_PICKING
                    //&&	m_pickingMode != TRIANGLE_PICKING
                    //&&	m_pickingMode != POINT_OR_TRIANGLE_PICKING
                    && (QApplication::keyboardModifiers() == Qt::NoModifier ||
                        QApplication::keyboardModifiers() ==
                                Qt::ControlModifier)) {
                    ecvDisplayTools::UpdateActiveItemsList(
                            curLastMousePos().x(),
                            curLastMousePos().y(), true);
                }
            }
        } else {
            // assert(curActiveItems().empty());
            curActiveItems().clear();
        }

        // update label and 3D name if visible
        if (abs(dx) > 0 || abs(dy) > 0) {
            emit m_tools->labelmove2D(x, y, dx, dy);
            ecvDisplayTools::UpdateNamePoseRecursive();
        }

        // specific case: move active item(s)
        if (!curActiveItems().empty()) {
            if (abs(dx) > 0 || abs(dy) > 0) {
                updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
            }
        } else {
            // OPTIMIZATION: Skip 2D label updates during camera rotation to
            // improve performance Only update when actively moving 2D items,
            // not during camera rotation The label will be updated when
            // rotation stops (in mouseReleaseEvent)
            if (abs(dx) > 0 || abs(dy) > 0) {
                // Only emit signal for label movement, but skip expensive
                // Update2DLabel during rotation
                emit m_tools->labelmove2D(x, y, dx, dy);
                ecvDisplayTools::UpdateNamePoseRecursive();
                // specific case: move active item(s)
                if (!curActiveItems().empty()) {
                    updateActivateditems(x, y, dx, dy,
                                         !ecvDisplayTools::USE_2D);
                }
                curActiveItems().clear();
            }

            // specific case: rectangular polyline drawing (for rectangular area
            // selection mode)
            if (curAllowRectangularEntityPicking() &&
                (curPickingMode() == ecvDisplayTools::ENTITY_PICKING ||
                 curPickingMode() ==
                         ecvDisplayTools::ENTITY_RECT_PICKING) &&
                (curRectPickingPoly() ||
                 (QApplication::keyboardModifiers() & Qt::AltModifier))) {
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
                        // QPointF posA =
                        // ecvDisplayTools::ToCenteredGLCoordinates(curLastMousePos().x(),
                        // curLastMousePos().y());
                        CCVector3d pos3D = ecvDisplayTools::ToVtkCoordinates(
                                curLastMousePos().x(),
                                curLastMousePos().y());

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
                        ecvDisplayTools::AddToOwnDB(curRectPickingPoly(),
                                                    false);
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
                    // QPointF posD =
                    // ecvDisplayTools::ToCenteredGLCoordinates(event->x(),
                    // event->y());
                    CCVector3d pos2D = ecvDisplayTools::ToVtkCoordinates(
                            event->x(), event->y());
                    B->x = C->x = static_cast<PointCoordinateType>(pos2D.x);
                    C->y = D->y = static_cast<PointCoordinateType>(pos2D.y);
                }
            }
            // standard rotation around the current pivot
            else if (curInteractionFlags() &
                     ecvDisplayTools::INTERACT_ROTATE) {
                // choose the right rotation mode
                enum RotationMode {
                    StandardMode,
                    BubbleViewMode,
                    LockedAxisMode
                };
                RotationMode rotationMode = StandardMode;
                if ((curInteractionFlags() &
                     ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) !=
                    ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) {
                    if (curBubbleViewModeEnabled())
                        rotationMode = BubbleViewMode;
                    else if (curRotationAxisLocked())
                        rotationMode = LockedAxisMode;
                }

                ccGLMatrixd rotMat;
                switch (rotationMode) {
                    case BubbleViewMode: {
                        QPoint posDelta =
                                curLastMousePos() - event->pos();

                        if (std::abs(posDelta.x()) != 0) {
                            double delta_deg =
                                    (posDelta.x() *
                                     static_cast<double>(
                                             curBubbleViewFov_deg())) /
                                    height();
                            // rotation about the sensor Z axis
                            CCVector3d axis = curViewportParams().viewMat
                                                      .getColumnAsVec3D(2);
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
                            s_lastMouseOrientation = ecvDisplayTools::
                                    ConvertMousePositionToOrientation(
                                            curLastMousePos().x(),
                                            curLastMousePos().y());
                        }

                        CCVector3d currentMouseOrientation = ecvDisplayTools::
                                ConvertMousePositionToOrientation(x, y);
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
                        ecvDisplayTools::GetGLCameraParameters(camera);
                        camera.modelViewMat.applyRotation(axis);

                        // determine whether we are in a side or top view
                        bool topView = (std::abs(axis.z) > 0.5);
                        double angle_rad = 0.0;
                        if (topView) {
                            // rotation origin
                            CCVector3d C2D;
                            if (curViewportParams().objectCenteredView) {
                                // project the current pivot point on screen
                                camera.project(curViewportParams()
                                                       .getPivotPoint(),
                                               C2D);
                                C2D.z = 0.0;
                            } else {
                                C2D = CCVector3d(width() / 2.0, height() / 2.0,
                                                 0.0);
                            }

                            CCVector3d previousMousePos(
                                    static_cast<double>(
                                            curLastMousePos().x()),
                                    static_cast<double>(
                                            height() -
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
                            if (camera.project(curViewportParams()
                                                       .getPivotPoint(),
                                               A2D) &&
                                camera.project(
                                        curViewportParams()
                                                        .getPivotPoint() +
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
                        ecvDisplayTools::RotateWithAxis(
                                CCVector2i(x, y), curLockedRotationAxis(),
                                -cloudViewer::RadiansToDegrees(angle_rad));
                    } break;

                    default:
                        assert(false);
                        break;
                }

                if (curInteractionFlags() &
                    ecvDisplayTools::INTERACT_TRANSFORM_ENTITIES) {
                    rotMat = curViewportParams().viewMat.transposed() *
                             rotMat * curViewportParams().viewMat;
                    // feedback for 'interactive transformation' mode
                    emit m_tools->rotation(rotMat);
                } else {
                    // ecvDisplayTools::RotateBaseViewMat(rotMat);
                    ecvDisplayTools::ShowPivotSymbol(true);
                    QApplication::changeOverrideCursor(
                            QCursor(Qt::ClosedHandCursor));

                    // feedback for 'echo' mode
                    emit m_tools->viewMatRotated(rotMat);
                }
            }
        }
    }

    curMouseMoved() = true;
    curLastMousePos() = event->pos();
    emit m_tools->cameraParamChanged();
    if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
    event->accept();
}

void QVTKWidgetCustom::updateActivateditems(
        int x, int y, int dx, int dy, bool updatePosition) {
    if (updatePosition) {
        // displacement vector (in "3D")
        double pixSize = ecvDisplayTools::ComputeActualPixelSize();
        CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
        curViewportParams().viewMat.transposed().applyRotation(u);

        const int retinaScale = ecvDisplayTools::GetDevicePixelRatio();
        u *= retinaScale;

        for (auto& activeItem : curActiveItems()) {
            if (activeItem->move2D(x * retinaScale, y * retinaScale,
                                   dx * retinaScale, dy * retinaScale,
                                   ecvDisplayTools::GlWidth(),
                                   ecvDisplayTools::GlHeight())) {
                ecvDisplayTools::InvalidateViewport();
            } else if (activeItem->move3D(u)) {
                ecvDisplayTools::InvalidateViewport();
                ecvDisplayTools::Deprecate3DLayer();
            }
        }
    }

    ecvDisplayTools::Redraw2DLabel();
}

void QVTKWidgetCustom::mouseReleaseEvent(QMouseEvent* event) {
    if (curInteractionFlags() & ecvDisplayTools::TRANSFORM_CAMERA()) {
        QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
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
        ecvDisplayTools::INTERACT_SIG_BUTTON_RELEASED) {
        event->accept();
        emit m_tools->buttonReleased();
    }

    if (curPivotSymbolShown()) {
        if (curPivotVisibility() == ecvDisplayTools::PIVOT_SHOW_ON_MOVE) {
            ecvDisplayTools::ToBeRefreshed();
        }
        ecvDisplayTools::ShowPivotSymbol(curPivotVisibility() ==
                                         ecvDisplayTools::PIVOT_ALWAYS_SHOW);
    }

    if ((event->button() == Qt::MiddleButton)) {
        if (mouseHasMoved) {
            event->accept();
            curActiveItems().clear();
            // ecvDisplayTools::ToBeRefreshed();
        } else if (curInteractionFlags() &
                   ecvDisplayTools::INTERACT_2D_ITEMS) {
            // interaction with 2D item(s)
            ecvDisplayTools::UpdateActiveItemsList(event->x(), event->y(),
                                                   false);
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

                ecvDisplayTools::RemoveFromOwnDB(curRectPickingPoly());
                curRectPickingPoly() = nullptr;
                vertices = nullptr;

                ecvDisplayTools::PickingParameters params(
                        ecvDisplayTools::ENTITY_RECT_PICKING,
                        pickX + ecvDisplayTools::Width() / 2,
                        ecvDisplayTools::Height() / 2 - pickY, pickW, pickH);
                ecvDisplayTools::StartPicking(params);
                ecvDisplayTools::ToBeRefreshed();
            }

            event->accept();
        } else {
            // picking?
            // CRITICAL: Don't start deferred picking if a VTK widget was
            // clicked This prevents doPicking() from overriding the widget
            // selection
            if (!curWidgetClicked() &&
                m_tools->m_timer.elapsed() <
                        curLastClickTime() +
                                CC_MAX_PICKING_CLICK_DURATION_MS)  // in msec
            {
                int x = curLastMousePos().x();
                int y = curLastMousePos().y();

                // Save global point size / line width before
                // ProcessClickableItems potentially modifies them.
                float savedGlobalPtSize =
                        ecvDisplayTools::GetViewportParameters()
                                .defaultPointSize;
                float savedGlobalLnWidth =
                        ecvDisplayTools::GetViewportParameters()
                                .defaultLineWidth;

                if (!ecvDisplayTools::ProcessClickableItems(x, y)) {
                    curLastMousePos() =
                            event->pos();
                    m_tools->setPickingTargetView(
                            ecvGenericGLDisplay::FromWidget(this));
                    m_tools->m_deferredPickingTimer.start();
                } else {
                    // Sync per-widget local values from the
                    // (just-updated) global.
                    m_localDefaultPointSize =
                            ecvDisplayTools::GetViewportParameters()
                                    .defaultPointSize;
                    m_localDefaultLineWidth =
                            ecvDisplayTools::GetViewportParameters()
                                    .defaultLineWidth;

                    auto* display =
                            ecvGenericGLDisplay::FromWidget(this);
                    bool isSecondary =
                            display &&
                            display !=
                                    ecvDisplayTools::TheInstance();
                    if (isSecondary) {
                        // Restore global so primary view's state
                        // is not contaminated.
                        ecvDisplayTools::SetViewportDefaultPointSize(
                                savedGlobalPtSize);
                        ecvDisplayTools::SetViewportDefaultLineWidth(
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
    } else if (event->button() == Qt::RightButton) {
        // CRITICAL: Update 2D labels after zoom/scale to ensure they align with
        // their 3D anchor points
        if (mouseHasMoved) {
            m_tools->Update2DLabel(true);
        }
    }

    // CRITICAL: Always update 2D labels after any mouse interaction that moved
    // the camera (rotation, zoom, pan) to ensure labels stay aligned with their
    // 3D anchor points. This fixes the issue where labels become detached after
    // mouse release.
    if (mouseHasMoved) {
        m_tools->Update2DLabel(true);
    }

    ecvDisplayTools::RefreshDisplay(true);
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

        // Gesture start/stop
        case QEvent::TouchBegin:
        case QEvent::TouchEnd: {
            QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
            touchEvent->accept();
            curTouchInProgress() = (evt->type() == QEvent::TouchBegin);
            curTouchBaseDist() = 0.0;
            CVLog::PrintDebug(QString("Touch event %1")
                                      .arg(curTouchInProgress()
                                                   ? "begins"
                                                   : "ends"));
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
            if (curTouchInProgress() &&
                !curViewportParams().perspectiveView) {
                QTouchEvent* touchEvent = static_cast<QTouchEvent*>(evt);
                const QList<QTouchEvent::TouchPoint>& touchPoints =
                        touchEvent->touchPoints();
                if (touchPoints.size() == 2) {
                    QPointF D = (touchPoints[1].pos() - touchPoints[0].pos());
                    qreal dist = std::sqrt(D.x() * D.x() + D.y() * D.y());
                    if (curTouchBaseDist() != 0.0) {
                        float zoomFactor = dist / curTouchBaseDist();
                        ecvDisplayTools::UpdateZoom(zoomFactor);
                    }
                    curTouchBaseDist() = dist;
                    evt->accept();
                    break;
                }
            }
        } break;

        case QEvent::Resize: {
            QSize newSize = static_cast<QResizeEvent*>(evt)->size();
            ecvDisplayTools::ResizeGL(newSize.width(), newSize.height());
            if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
            evt->accept();
        } break;

        case QEvent::KeyPress: {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(evt);

            if (keyEvent->key() == Qt::Key_Escape) {
                CVLog::Print(
                        "[QVTKWidgetCustom] ESC key pressed, forwarding to "
                        "MainWindow");
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
