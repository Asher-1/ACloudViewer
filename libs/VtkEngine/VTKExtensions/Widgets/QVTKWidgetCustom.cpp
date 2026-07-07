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
#include <Tools/SelectionTools/cvInteractorStyleDrawPolygon.h>
#include <VTKExtensions/Views/vtkPVAxesActor.h>
#include <vtkAbstractPicker.h>
#include <vtkAngleRepresentation2D.h>
#include <vtkCamera.h>
#include <vtkClipPolyData.h>
#include <vtkColorTransferFunction.h>
#include <vtkConeSource.h>
#include <vtkDelaunay2D.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkIdFilter.h>
#include <vtkImageData.h>
#include <vtkInteractorStyle.h>
#include <vtkInteractorStyleDrawPolygon.h>
#include <vtkInteractorStyleRubberBand3D.h>
#include <vtkInteractorStyleRubberBandZoom.h>
#include <vtkLogger.h>
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
#include "VTKExtensions/Widgets/VtkShortcutRegistry.h"
#include "Visualization/VtkVis.h"
#include "Visualization/vtkGLView.h"

// CV_DB_LIB
#include <Visualization/vtkGLView.h>
#include <ecvDisplayCoordinates.h>
#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvInteractor.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewRepresentation.h>

// QT
#include <ecv2DLabel.h>
#include <ecvHObjectCaster.h>

#include <QApplication>
#include <QCoreApplication>
#include <QHBoxLayout>
#include <QLayout>
#include <QMainWindow>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QPushButton>
#include <QResizeEvent>
#include <QSettings>
#include <QThread>
#include <QTimer>
#include <QTouchEvent>
#include <QWheelEvent>
#include <QWidget>

#ifdef USE_VLD
#include <vld.h>
#endif

#include <Shortcuts/ecvKeySequences.h>
#include <ecv2DLabel.h>
#include <ecvHObjectCaster.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkTextProperty.h>

#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>

#include "ScaleBarWidget.h"

// macroes
#ifndef VTK_CREATE
#define VTK_CREATE(TYPE, NAME) \
    vtkSmartPointer<TYPE> NAME = vtkSmartPointer<TYPE>::New()
#endif

static bool isVtkToolInteractorStyle(vtkRenderWindowInteractor* interactor) {
    if (!interactor) return false;
    auto* style =
            vtkInteractorStyle::SafeDownCast(interactor->GetInteractorStyle());
    if (!style) return false;

    if (vtkInteractorStyleRubberBand3D::SafeDownCast(style) ||
        vtkInteractorStyleRubberBandZoom::SafeDownCast(style) ||
        vtkInteractorStyleDrawPolygon::SafeDownCast(style) ||
        cvInteractorStyleDrawPolygon::SafeDownCast(style)) {
        return true;
    }

    const char* className = style->GetClassName();
    return className &&
           std::string(className).find("DrawPolygon") != std::string::npos;
}

static bool isSignalOnlyInteraction(
        ecvGenericGLDisplay::INTERACTION_FLAGS flags) {
    return flags == ecvGenericGLDisplay::INTERACT_SEND_ALL_SIGNALS;
}

static bool shouldForwardMouseToVtk(
        vtkRenderWindowInteractor* interactor,
        ecvGenericGLDisplay::INTERACTION_FLAGS flags) {
    if (isSignalOnlyInteraction(flags)) {
        return false;
    }

    const auto cameraFlags = ecvGenericGLDisplay::INTERACT_ROTATE |
                             ecvGenericGLDisplay::INTERACT_PAN |
                             ecvGenericGLDisplay::INTERACT_ZOOM_CAMERA;
    if (!(flags & cameraFlags)) {
        return false;
    }

    return (flags & ecvDisplayTools::TRANSFORM_CAMERA()) ||
           isVtkToolInteractorStyle(interactor);
}

static bool shouldForwardMouseEventToVtk(
        vtkRenderWindowInteractor* interactor,
        ecvGenericGLDisplay::INTERACTION_FLAGS flags,
        bool /*rotationAxisLocked*/,
        const QMouseEvent* /*event*/,
        QEvent::Type /*eventType*/) {
    if (!shouldForwardMouseToVtk(interactor, flags)) {
        return false;
    }

    return true;
}

ecvViewContext& QVTKWidgetCustom::curCtx() {
    if (m_ownerView && m_ownerView->viewContext())
        return *m_ownerView->viewContext();
    return ecvViewManager::instance().resolveViewContext();
}

const ecvViewContext& QVTKWidgetCustom::curCtx() const {
    if (m_ownerView && m_ownerView->viewContext())
        return *m_ownerView->viewContext();
    return ecvViewManager::instance().resolveViewContext();
}

ecvViewContext* QVTKWidgetCustom::ownerCtx() {
    return m_ownerView ? &m_ownerView->context() : nullptr;
}

ccPolyline*& QVTKWidgetCustom::curRectPickingPoly() {
    if (m_ownerView) return m_ownerView->rectPickingPolyRef();
    auto* dt = ecvViewManager::instance().displayTools();
    assert(dt);
    return dt->m_rectPickingPoly;
}

std::list<ccInteractor*>& QVTKWidgetCustom::curActiveItems() {
    if (m_ownerView) return m_ownerView->activeItemsRef();
    return displayTarget()->activeItemsRef();
}

ecvHotZone*& QVTKWidgetCustom::curHotZone() {
    if (m_ownerView) return m_ownerView->hotZoneRef();
    return displayTarget()->hotZonePtrRef();
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
    vtkSmartPointer<vtkOrientationMarkerWidget> orientationMarkerWidget;

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

static QMap<QString, VtkShortcutDef> s_vtkShortcutMap;
static bool s_vtkMapInitialized = false;

void QVTKWidgetCustom::reloadVtkShortcutMap() {
    s_vtkShortcutMap = buildVtkShortcutMap();
    s_vtkMapInitialized = true;
}

static void ensureVtkShortcutMap() {
    if (!s_vtkMapInitialized) {
        s_vtkShortcutMap = buildVtkShortcutMap();
        s_vtkMapInitialized = true;
    }
}
QVTKWidgetCustom::QVTKWidgetCustom(QMainWindow* parentWindow,
                                   ecvDisplayTools* /*tools*/,
                                   bool stereoMode)
    : QVTKOpenGLNativeWidget(parentWindow),
      m_render(nullptr),
      m_win(parentWindow),
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
        if (auto* dt = displayTarget()) dt->update2DLabels(true);
    });

    m_interactionRenderTimer = new QTimer(this);
    m_interactionRenderTimer->setInterval(16);  // ~60Hz target frame rate
    connect(m_interactionRenderTimer, &QTimer::timeout, this, [this]() {
        m_timerTickCount++;
        if (m_hasPendingMousePos) {
            m_hasPendingMousePos = false;
            m_renderFrameCount++;
            if (auto* rw = this->renderWindow()) {
                auto* iren = rw->GetInteractor();
                if (iren) {
                    iren->SetEventInformationFlipY(m_pendingMousePos.x(),
                                                   m_pendingMousePos.y());
                    if (m_skipFirstTimerMove) {
                        m_skipFirstTimerMove = false;
                    } else {
                        iren->MouseMoveEvent();
                    }
                }
                QElapsedTimer rt;
                rt.start();
                rw->Render();
                qint64 renderNs = rt.nsecsElapsed();
                m_renderTimeAccumNs += renderNs;
            }
        }
        if (m_interactionFpsTimer.isValid() &&
            m_interactionFpsTimer.elapsed() >= 2000) {
            double elapsed = m_interactionFpsTimer.elapsed() / 1000.0;
            double fps = m_renderFrameCount / elapsed;
            double avgRenderMs =
                    m_renderFrameCount > 0
                            ? (m_renderTimeAccumNs / 1e6) / m_renderFrameCount
                            : 0;
            if (m_fpsEnabled) {
                CVLog::Print(QString("[INTERACTION-FPS] fps=%1 renders=%2 "
                                     "ticks=%3 avgRender=%4ms elapsed=%5s")
                                     .arg(fps, 0, 'f', 1)
                                     .arg(m_renderFrameCount)
                                     .arg(m_timerTickCount)
                                     .arg(avgRenderMs, 0, 'f', 2)
                                     .arg(elapsed, 0, 'f', 1));
            }
            m_renderFrameCount = 0;
            m_timerTickCount = 0;
            m_renderTimeAccumNs = 0;
            m_interactionFpsTimer.restart();
        }
    });

    QSurfaceFormat fmt = QVTKOpenGLNativeWidget::defaultFormat();
    fmt.setStereo(stereoMode);
    fmt.setSwapInterval(0);
    setFormat(fmt);

    this->setEnableHiDPI(true);

    // drag & drop handling
    setAcceptDrops(true);
    setAttribute(Qt::WA_AcceptTouchEvents, true);
    // setAttribute(Qt::WA_OpaquePaintEvent, true);

    // Prevent the native GL window from being visible before the widget is
    // properly placed in a layout. On Linux/X11, native child windows bypass
    // Qt stacking and can render over the menu bar. The widget becomes visible
    // when the layout system calls show() on it.
    hide();

    vtkObject::GlobalWarningDisplayOff();
    static std::once_flag s_loggerInit;
    std::call_once(s_loggerInit, []() {
        vtkLogger::SetStderrVerbosity(vtkLogger::VERBOSITY_WARNING);
    });
    d_ptr = new VtkWidgetPrivate(this);
}

QVTKWidgetCustom::~QVTKWidgetCustom() {
    m_ownerView = nullptr;
    if (auto* dt = displayTarget()) {
        if (dt->hotZonePtrRef() == m_localHotZone) {
            dt->hotZonePtrRef() = nullptr;
        }
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

ecvGenericGLDisplay* QVTKWidgetCustom::displayTarget() const {
    auto* d = resolveDisplay();
    if (d) return d;
    return ecvViewManager::instance().getEffectiveView();
}

void QVTKWidgetCustom::connectSignalsTo(ecvDisplayTools* target) {
    if (!target) return;
    connect(this, &QVTKWidgetCustom::rightButtonClicked, target,
            &ecvDisplayTools::rightButtonClicked);
    connect(this, &QVTKWidgetCustom::leftButtonClicked, target,
            &ecvDisplayTools::leftButtonClicked);
    connect(this, &QVTKWidgetCustom::doubleButtonClicked, target,
            &ecvDisplayTools::doubleButtonClicked);
    connect(this, &QVTKWidgetCustom::mouseWheelChanged, target,
            &ecvDisplayTools::mouseWheelChanged);
    connect(this, &QVTKWidgetCustom::mouseWheelRotated, target,
            &ecvDisplayTools::mouseWheelRotated);
    connect(this, &QVTKWidgetCustom::mousePosChanged, target,
            &ecvDisplayTools::mousePosChanged);
    connect(this, &QVTKWidgetCustom::mouseMoved, target,
            &ecvDisplayTools::mouseMoved);
    connect(this, &QVTKWidgetCustom::translation, target,
            &ecvDisplayTools::translation);
    connect(this, &QVTKWidgetCustom::rotation, target,
            &ecvDisplayTools::rotation);
    connect(this, &QVTKWidgetCustom::viewMatRotated, target,
            &ecvDisplayTools::viewMatRotated);
    connect(this, &QVTKWidgetCustom::buttonReleased, target,
            &ecvDisplayTools::buttonReleased);
    connect(this, &QVTKWidgetCustom::filesDropped, target,
            &ecvDisplayTools::filesDropped);
    connect(this, &QVTKWidgetCustom::exclusiveFullScreenToggled, target,
            &ecvDisplayTools::exclusiveFullScreenToggled);
    connect(this, &QVTKWidgetCustom::cameraParamChanged, target,
            &ecvDisplayTools::cameraParamChanged);
    connect(this, &QVTKWidgetCustom::labelmove2D, target,
            &ecvDisplayTools::labelmove2D);
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

void QVTKWidgetCustom::startInteractionRenderTimer() {
    if (m_interactionRenderTimer && !m_interactionRenderTimer->isActive()) {
        m_hasPendingMousePos = false;
        m_skipFirstTimerMove = true;
        m_renderFrameCount = 0;
        m_timerTickCount = 0;
        m_renderTimeAccumNs = 0;
        m_interactionFpsTimer.start();
        m_interactionRenderTimer->start();
    }
    if (auto* rw = this->renderWindow()) {
        rw->SetDesiredUpdateRate(5.0);
        if (auto* iren = rw->GetInteractor()) {
            iren->SetEnableRender(false);
        }
    }
}

void QVTKWidgetCustom::stopInteractionRenderTimer() {
    if (m_interactionRenderTimer && m_interactionRenderTimer->isActive()) {
        m_interactionRenderTimer->stop();
        m_hasPendingMousePos = false;
        m_skipFirstTimerMove = false;
    }
    if (auto* rw = this->renderWindow()) {
        if (auto* iren = rw->GetInteractor()) {
            iren->SetEnableRender(true);
        }
        rw->SetDesiredUpdateRate(0.002);
        rw->Render();
    }
    this->update();
}

void QVTKWidgetCustom::updateScaleBarIfNeeded() {
    if (m_scaleBar && m_render) {
        m_scaleBar->update(m_render, m_interactor);
    }
}

void QVTKWidgetCustom::onRenderStart() {
    m_renderStartNs = m_renderTimer.nsecsElapsed();
}

void QVTKWidgetCustom::onRenderForFps() {
    qint64 renderDurationNs = m_renderTimer.nsecsElapsed() - m_renderStartNs;
    m_renderAccumNs += renderDurationNs;
    m_renderTimingCount++;
    m_frameCount++;
    qint64 elapsed = m_fpsTimer.elapsed();
    if (elapsed >= 500) {
        m_currentFps = m_frameCount * 1000.0 / elapsed;
        m_frameCount = 0;
        m_fpsTimer.restart();
        if (m_fpsActor) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f FPS", m_currentFps);
            m_fpsActor->SetInput(buf);
        }
        double interval = m_currentFps > 0 ? 1000.0 / m_currentFps : 0;
        double avgRenderMs =
                m_renderTimingCount > 0
                        ? (m_renderAccumNs / 1e6) / m_renderTimingCount
                        : 0;
        if (m_fpsEnabled) {
            CVLog::Print(
                    QString("[FPS] %1 FPS (interval ~%2 ms, renderTime ~%3 "
                            "ms)")
                            .arg(m_currentFps, 0, 'f', 1)
                            .arg(interval, 0, 'f', 1)
                            .arg(avgRenderMs, 0, 'f', 1));
        }
        m_renderAccumNs = 0;
        m_renderTimingCount = 0;
    }
}

void QVTKWidgetCustom::beginPickCenterOfRotation() {
    m_pickCenterPending = true;
    setCursor(Qt::CrossCursor);
}

void QVTKWidgetCustom::cancelPickCenterOfRotation() {
    if (m_pickCenterPending) {
        m_pickCenterPending = false;
        setCursor(QCursor());
    }
}

void QVTKWidgetCustom::setFpsVisible(bool visible) {
    m_fpsEnabled = visible;
    if (visible) {
        if (!m_fpsActor) {
            m_fpsActor = vtkSmartPointer<vtkTextActor>::New();
            m_fpsActor->SetInput("-- FPS");
            m_fpsActor->GetTextProperty()->SetFontSize(14);
            m_fpsActor->GetTextProperty()->SetColor(0.0, 1.0, 0.0);
            m_fpsActor->GetTextProperty()->SetFontFamilyToCourier();
            m_fpsActor->GetTextProperty()->SetBold(true);
            m_fpsActor->SetDisplayPosition(10, 10);
        }
        if (m_render) {
            m_render->AddActor2D(m_fpsActor);
        }
        if (!m_fpsObserver) {
            m_fpsObserver = vtkSmartPointer<vtkCallbackCommand>::New();
            m_fpsObserver->SetClientData(this);
            m_fpsObserver->SetCallback(
                    [](vtkObject*, unsigned long, void* cd, void*) {
                        auto* self = static_cast<QVTKWidgetCustom*>(cd);
                        if (self) self->onRenderForFps();
                    });
            if (auto* rw = this->GetRenderWindow()) {
                rw->AddObserver(vtkCommand::EndEvent, m_fpsObserver);
            }
        }
        if (!m_startObserver) {
            m_startObserver = vtkSmartPointer<vtkCallbackCommand>::New();
            m_startObserver->SetClientData(this);
            m_startObserver->SetCallback(
                    [](vtkObject*, unsigned long, void* cd, void*) {
                        auto* self = static_cast<QVTKWidgetCustom*>(cd);
                        if (self) self->onRenderStart();
                    });
            if (auto* rw = this->GetRenderWindow()) {
                rw->AddObserver(vtkCommand::StartEvent, m_startObserver);
            }
        }
        m_renderTimer.start();
        m_fpsTimer.start();
        m_frameCount = 0;
    } else {
        if (m_fpsActor && m_render) {
            m_render->RemoveActor2D(m_fpsActor);
        }
        if (m_fpsObserver) {
            if (auto* rw = this->GetRenderWindow()) {
                rw->RemoveObserver(m_fpsObserver);
            }
            m_fpsObserver = nullptr;
        }
    }
}

void QVTKWidgetCustom::initVtk(
        vtkSmartPointer<vtkRenderWindowInteractor> interactor, bool useVBO) {
    this->m_useVBO = useVBO;
    this->m_interactor = interactor;

    this->m_render =
            this->GetRenderWindow()->GetRenderers()->GetFirstRenderer();
    this->m_camera = m_render->GetActiveCamera();
    this->m_renders = this->GetRenderWindow()->GetRenderers();

    // Keep scale bar in sync during VTK-driven zoom/pan/rotate (interactor
    // style calls Render() directly, bypassing QVTKWidgetCustom mouse
    // handlers). InteractionEvent avoids updating on every StartEvent/render
    // frame.
    if (m_interactor && !m_scaleBarUpdateObserver) {
        m_scaleBarUpdateObserver = vtkSmartPointer<vtkCallbackCommand>::New();
        m_scaleBarUpdateObserver->SetClientData(this);
        m_scaleBarUpdateObserver->SetCallback(
                [](vtkObject*, unsigned long, void* cd, void*) {
                    auto* self = static_cast<QVTKWidgetCustom*>(cd);
                    if (self) self->updateScaleBarIfNeeded();
                });
        m_interactor->AddObserver(vtkCommand::InteractionEvent,
                                  m_scaleBarUpdateObserver);
    }

    if (!m_scaleBar) {
        m_scaleBar = new ScaleBarWidget(m_render);
        // CRITICAL: Enable visibility BEFORE notifying layout ready
        // ScaleBarWidget defaults to visible=false in constructor
        m_scaleBar->setVisible(true);

        // Notify layout ready after widget is created and renderer is available
        // so the scale bar can be displayed immediately
        QTimer::singleShot(100, this, [this]() {
            if (m_scaleBar && m_render && m_interactor) {
                if (!m_scaleBar->isLayoutReady()) {
                    m_scaleBar->notifyLayoutReady();
                }
                m_scaleBar->update(m_render, m_interactor);
                // Force a render to ensure the scale bar appears
                if (this->GetRenderWindow()) {
                    this->GetRenderWindow()->Render();
                }
            }
        });
    }

    setFpsVisible(false);
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
    updateScaleBarIfNeeded();
    if (IsCalledFromMainThread()) {
        this->update();
    } else {
        QMetaObject::invokeMethod(
                this, [=]() { this->update(); }, Qt::QueuedConnection);
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

void QVTKWidgetCustom::collectAllLabels(std::vector<ccHObject*>& labels) const {
    unsigned gen = ecvViewManager::instance().labelCacheGeneration();
    if (m_labelCacheGen == gen && !m_cachedLabels.empty()) {
        labels = m_cachedLabels;
        return;
    }
    auto* disp = const_cast<QVTKWidgetCustom*>(this)->displayTarget();
    if (!disp) return;
    ccHObject* sceneDB = disp->getSceneDB();
    ccHObject* ownDB = disp->getOwnDB();
    if (sceneDB)
        sceneDB->filterChildren(labels, true, CV_TYPES::LABEL_2D, false);
    if (ownDB) ownDB->filterChildren(labels, true, CV_TYPES::LABEL_2D, false);
    m_cachedLabels = labels;
    m_labelCacheGen = gen;
}

static Visualization::VtkVis* vtkVisForWidget(QVTKWidgetCustom* widget) {
    if (!widget) return nullptr;
    auto* display = ecvGenericGLDisplay::FromWidget(widget);
    auto* glView = display ? dynamic_cast<vtkGLView*>(display) : nullptr;
    return glView ? dynamic_cast<Visualization::VtkVis*>(
                            glView->getVisualizer3D())
                  : nullptr;
}

static bool applyDirectCameraWheelZoom(QVTKWidgetCustom* widget,
                                       float wheelDelta_deg) {
    if (!widget) return false;
    auto* vis = vtkVisForWidget(widget);
    auto* ren = vis ? vis->getCurrentRenderer() : nullptr;
    auto* cam = ren ? ren->GetActiveCamera() : nullptr;
    if (!ren || !cam) return false;

    static constexpr float c_defaultDeg2Zoom = 20.0f;
    const double zoomFactor =
            std::pow(1.1, static_cast<double>(wheelDelta_deg) /
                                  static_cast<double>(c_defaultDeg2Zoom));
    if (zoomFactor <= 0.0 || zoomFactor == 1.0) return false;

    if (cam->GetParallelProjection()) {
        cam->SetParallelScale(cam->GetParallelScale() / zoomFactor);
    } else {
        cam->Dolly(zoomFactor);
    }
    cam->Modified();
    ren->ResetCameraClippingRange();
    ren->Modified();
    if (auto* rw = widget->renderWindow()) rw->Modified();
    widget->update();
    return true;
}

bool QVTKWidgetCustom::handleCameraOrientationMouse(QMouseEvent* event,
                                                    QEvent::Type eventType) {
    if (isSignalOnlyInteraction(curInteractionFlags())) return false;

    auto* vis = vtkVisForWidget(this);
    if (!vis || !vis->IsCameraOrientationWidgetShown()) return false;

    // Lightweight fix-up: ensure ProcessEvents is on without resizing/resetting
    // the widget (heavy EnsureCameraOrientationWidgetInteractive resets
    // representation state and breaks the hover→click axis-pick flow).
    vis->EnsureCameraOrientationWidgetProcessEvents();

    const bool overWidget =
            vis->IsMouseOverCameraOrientationWidget(event->x(), event->y());
    if (eventType == QEvent::MouseButtonPress && overWidget) {
        m_cameraOrientMouseActive = true;
        curWidgetClicked() = true;
    }

    if (!overWidget && !m_cameraOrientMouseActive) return false;

    // Forward events through the full Qt→VTK pipeline. The widget registers
    // at priority 1.0 and calls AbortFlagOn() which prevents the interactor
    // style from also handling the event (no camera rotation).
    bool handled = true;
    switch (eventType) {
        case QEvent::MouseButtonPress:
            // Fire a move first so the widget's representation enters "Hot"
            // state before receiving the press (required for axis picking).
            QVTKOpenGLNativeWidget::mouseMoveEvent(event);
            QVTKOpenGLNativeWidget::mousePressEvent(event);
            break;
        case QEvent::MouseButtonRelease:
            QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
            break;
        case QEvent::MouseMove:
            QVTKOpenGLNativeWidget::mouseMoveEvent(event);
            break;
        default:
            handled = false;
            break;
    }

    if (!handled) return false;

    if (eventType == QEvent::MouseButtonRelease) {
        m_cameraOrientMouseActive = false;
    }

    if (eventType == QEvent::MouseMove ||
        eventType == QEvent::MouseButtonRelease) {
        if (m_ownerView) emit m_ownerView->cameraParamChanged();
        emit cameraParamChanged();
    }

    return true;
}

// event processing
void QVTKWidgetCustom::mousePressEvent(QMouseEvent* event) {
    if (!isVtkToolInteractorStyle(m_interactor)) {
        startInteractionRenderTimer();
    }

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

    if (handleCameraOrientationMouse(event, QEvent::MouseButtonPress)) {
        event->accept();
        return;
    }

    curLastPointIndex() = -1;
    curLastPickedId() = QString();

    // Signal-only mode (e.g., segment tool active): emit signals but do NOT
    // process camera manipulation or forward to VTK.
    if (isSignalOnlyInteraction(curInteractionFlags())) {
        if (event->buttons() & Qt::LeftButton) {
            if (m_ownerView)
                emit m_ownerView->leftButtonClicked(event->x(), event->y());
            emit leftButtonClicked(event->x(), event->y());
        } else if (event->buttons() & Qt::RightButton) {
            if (m_ownerView)
                emit m_ownerView->rightButtonClicked(event->x(), event->y());
            emit rightButtonClicked(event->x(), event->y());
        }
        event->accept();
        return;
    }

    if ((event->buttons() & Qt::RightButton)) {
        m_rightClickOnLabel = false;
        if (curInteractionFlags() & ecvDisplayTools::INTERACT_2D_ITEMS) {
            ccHObject::Container labels;
            collectAllLabels(labels);
            for (auto* obj : labels) {
                if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isBranchEnabled() ||
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
            if ((curInteractionFlags() & ecvDisplayTools::INTERACT_PAN) ||
                ((QApplication::keyboardModifiers() & Qt::ControlModifier) &&
                 (curInteractionFlags() &
                  ecvDisplayTools::INTERACT_CTRL_PAN))) {
                QApplication::setOverrideCursor(QCursor(Qt::SizeAllCursor));
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
            emit rightButtonClicked(event->x(), event->y());
        }
    } else if (event->buttons() & Qt::LeftButton) {
        auto* dt = displayTarget();
        curLastClickTime() = dt ? dt->elapsedMs() : 0;

        m_labelClickedOnPress = false;
        if (curInteractionFlags() & ecvDisplayTools::INTERACT_2D_ITEMS) {
            ccHObject::Container labels;
            collectAllLabels(labels);
            for (auto* obj : labels) {
                if (!obj->isA(CV_TYPES::LABEL_2D) || !obj->isBranchEnabled() ||
                    !obj->isVisible())
                    continue;
                cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                if (!l) continue;
                QRect roi = l->getLabelROI();
                if (roi.isValid() && roi.contains(event->x(), event->y())) {
                    curActiveItems().clear();
                    curActiveItems().push_back(l);
                    m_labelClickedOnPress = true;
                    break;
                }
            }
        }

        if (curInteractionFlags() & ecvDisplayTools::INTERACT_ROTATE) {
            QApplication::setOverrideCursor(QCursor(Qt::PointingHandCursor));
        }

        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_SIG_LB_CLICKED) {
            if (m_ownerView)
                emit m_ownerView->leftButtonClicked(event->x(), event->y());
            emit leftButtonClicked(event->x(), event->y());
        }

        if (m_pickCenterPending) {
            CCVector3d P;
            bool onSurface = false;

            auto* vis = vtkVisForWidget(this);
            auto* ren = vis ? vis->getCurrentRenderer() : nullptr;
            if (ren) {
                auto rw = vis->getRenderWindow();
                if (rw) {
                    rw->Render();
                }

                const double dpr = devicePixelRatioF();
                int px = static_cast<int>(event->x() * dpr);
                int py = static_cast<int>(ren->GetSize()[1] - 1 -
                                          event->y() * dpr);

                double z = vis->getGLDepth(px, py);

                if (z < 1.0) {
                    ren->SetDisplayPoint(px, py, z);
                    ren->DisplayToWorld();
                    double* wp = ren->GetWorldPoint();
                    if (wp[3] != 0.0) {
                        P = CCVector3d(wp[0] / wp[3], wp[1] / wp[3],
                                       wp[2] / wp[3]);
                        onSurface = true;
                    }
                }

                if (!onSurface) {
                    vtkCamera* cam = ren->GetActiveCamera();
                    if (cam) {
                        double fp[4];
                        cam->GetFocalPoint(fp);
                        fp[3] = 1.0;
                        ren->SetWorldPoint(fp);
                        ren->WorldToDisplay();
                        double* dc = ren->GetDisplayPoint();
                        ren->SetDisplayPoint(px, py, dc[2]);
                        ren->DisplayToWorld();
                        double* wp = ren->GetWorldPoint();
                        if (wp[3] != 0.0) {
                            P = CCVector3d(wp[0] / wp[3], wp[1] / wp[3],
                                           wp[2] / wp[3]);
                            onSurface = true;
                        }
                    }
                }
            }
            if (onSurface) {
                displayTarget()->setPivotPoint(P, true, true);
            }
            m_pickCenterPending = false;
            setCursor(QCursor());
            emit pickCenterOfRotationFinished(onSurface);
            event->accept();
            return;
        }
    } else {
    }

    if (isSignalOnlyInteraction(curInteractionFlags())) {
        event->accept();
        return;
    }

    if (m_labelClickedOnPress || m_rightClickOnLabel) {
        event->accept();
    } else if (shouldForwardMouseEventToVtk(m_interactor, curInteractionFlags(),
                                            curRotationAxisLocked(), event,
                                            QEvent::MouseButtonPress)) {
        QVTKOpenGLNativeWidget::mousePressEvent(event);
        event->accept();
    } else {
        event->accept();
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

    if (isSignalOnlyInteraction(curInteractionFlags())) {
        event->accept();
        return;
    }

    // Check if double-click is within the HotZone area. If so, ignore the
    // pivot-point logic to prevent rapid +/- clicks from resetting the
    // rotation center.
    bool inHotZone = false;
    if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_CLICKABLE_ITEMS) {
        ecvHotZone* hz = curHotZone();
        if (hz && displayTarget()) {
            QRect areaRect = hz->rect(true, curBubbleViewModeEnabled(),
                                      displayTarget()->exclusiveFullScreen());
            const double dpr =
                    static_cast<double>(displayTarget()->getDevicePixelRatio());
            int scaledX = ecvDisplayCoordinates::toPhysical(event->x(), dpr);
            int scaledY = ecvDisplayCoordinates::toPhysical(event->y(), dpr);
            QRect zoneRect = areaRect.translated(hz->topCorner);
            inHotZone = zoneRect.contains(scaledX, scaledY);
        }
    }

    if (inHotZone) {
        event->accept();
        return;
    }

    displayTarget()->stopDeferredPicking();
    curIgnoreMouseReleaseEvent() = true;

    const int x = event->x();
    const int y = event->y();

    CCVector3d P;
    if (displayTarget()->getClick3DPos(x, y, P))
        displayTarget()->setPivotPoint(P, true, true);

    if (m_ownerView)
        emit m_ownerView->doubleButtonClicked(event->x(), event->y());
    emit doubleButtonClicked(event->x(), event->y());

    QVTKOpenGLNativeWidget::mouseDoubleClickEvent(event);
}

// Check whether an entity's ancestor chain is visible in a specific view.
// Returns false if any ancestor is globally disabled.
// HIERARCHY_OBJECT (folders) are pure containers: their per-view reps must
// NOT block children's visibility, so we skip them in the rep check.
static bool isAncestorVisibleInView(ccHObject* entity,
                                    ecvGenericGLDisplay* view) {
    if (!view) return true;
    auto& repMgr = ecvRepresentationManager::instance();
    for (ccHObject* p = entity->getParent(); p; p = p->getParent()) {
        if (!p->isEnabled()) return false;
        if (p->getClassID() == CV_TYPES::HIERARCHY_OBJECT) continue;
        auto* rep = repMgr.getRepresentation(p, view);
        if (rep && rep->hasVisibilityOverride() && !rep->isVisible()) {
            return false;
        }
    }
    return true;
}

void QVTKWidgetCustom::resizeEvent(QResizeEvent* event) {
    QVTKOpenGLNativeWidget::resizeEvent(event);
    // Refresh overlay widgets (camera orientation widget, orientation marker)
    // after resize so their DPI-scaled sizes are computed against the actual
    // window dimensions rather than the potentially-incorrect initial size.
    if (m_ownerView) {
        if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                    m_ownerView->getVisualizer3D())) {
            vis->RefreshOverlayWidgets();
        }
    }
}

void QVTKWidgetCustom::paintGL() {
    if (property("_compViewDiag").toBool() && m_render &&
        m_render->GetActiveCamera()) {
        int cnt = property("_paintCnt").toInt();
        if (cnt < 20) {
            setProperty("_paintCnt", cnt + 1);
            double* p = m_render->GetActiveCamera()->GetPosition();
            auto* rw = this->renderWindow();
            int rwW = 0, rwH = 0;
            if (rw) {
                int* sz = rw->GetSize();
                rwW = sz[0];
                rwH = sz[1];
            }
            int nRen = (rw && rw->GetRenderers())
                               ? rw->GetRenderers()->GetNumberOfItems()
                               : 0;
            CVLog::PrintVerbose(
                    "[paintGL] widget=%p wSz=%dx%d rwSz=%dx%d nRen=%d "
                    "cam=(%.2f,%.2f,%.2f) ps=%.3f",
                    static_cast<void*>(this), width(), height(), rwW, rwH, nRen,
                    p[0], p[1], p[2],
                    m_render->GetActiveCamera()->GetParallelScale());
        }
    }
    QVTKOpenGLNativeWidget::paintGL();

    if (!displayTarget()) return;

    // Ensure the effective view context resolves to THIS widget's view
    // during label rendering. Without this, resolveViewContext() returns
    // the UI-active view which may differ from this widget.
    ecvViewManager::ScopedRenderOverride renderGuard(resolveDisplay());

    ccHObject::Container labels;
    collectAllLabels(labels);
    if (labels.empty()) return;

    CC_DRAW_CONTEXT context;
    if (m_ownerView) {
        m_ownerView->syncVtkCameraToContext();
        m_ownerView->getContext(context);
    } else {
        ecvDisplayTools::GetContext(context);
    }

    ecvGenericGLDisplay* thisDisplay = resolveDisplay();

    auto shouldSkipLabel = [&](cc2DLabel* label) -> bool {
        if (!label->isBranchEnabled() || !label->isVisible()) return true;
        if (thisDisplay) {
            ecvGenericGLDisplay* labelDisp = label->getDisplay();
            if (labelDisp && labelDisp != thisDisplay) return true;
            if (!labelDisp) return true;
        }
        if (!isAncestorVisibleInView(label, thisDisplay)) return true;
        return false;
    };

    int validCount = 0;
    for (auto* obj : labels) {
        if (!obj->isA(CV_TYPES::LABEL_2D)) continue;
        auto* label = static_cast<cc2DLabel*>(obj);
        if (shouldSkipLabel(label)) {
            if (!label->isBranchEnabled() || !label->isVisible() ||
                !isAncestorVisibleInView(label, thisDisplay)) {
                ecvDisplayTools::HideShowEntities(label, false);
                label->clearLabel(true);
            }
            continue;
        }
        label->update2DLabelView(context, false);
        if (!label->overlayValid()) {
            CVLog::Warning(
                    "[paintGL] label '%s' overlay INVALID after "
                    "update2DLabelView (dispIn2D=%d pts=%d)",
                    qPrintable(label->getName()), label->isDisplayedIn2D(),
                    static_cast<int>(label->size()));
            continue;
        }
        ++validCount;
    }
    if (validCount == 0) return;

    if (auto* ctx = QOpenGLContext::currentContext()) {
        auto* f = ctx->functions();
        f->glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
        f->glDisable(GL_DEPTH_TEST);
        f->glDisable(GL_STENCIL_TEST);
        f->glDisable(GL_SCISSOR_TEST);
        f->glDisable(GL_BLEND);
        f->glActiveTexture(GL_TEXTURE0);
        f->glBindTexture(GL_TEXTURE_2D, 0);
        f->glUseProgram(0);
        f->glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    QPainter painter(this);
    if (!painter.isActive()) return;
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setRenderHint(QPainter::TextAntialiasing, true);

    for (auto* obj : labels) {
        if (!obj->isA(CV_TYPES::LABEL_2D)) continue;
        auto* label = static_cast<cc2DLabel*>(obj);
        if (shouldSkipLabel(label)) continue;
        if (!label->overlayValid()) continue;
        label->paintOverlay(painter);
    }

    painter.end();
}

void QVTKWidgetCustom::wheelEvent(QWheelEvent* event) {
    bool doRedraw = false;
    Qt::KeyboardModifiers keyboardModifiers = QApplication::keyboardModifiers();

    if (m_ownerView) emit m_ownerView->mouseWheelChanged(event);
    emit mouseWheelChanged(event);
    double delta = qtCompatWheelEventDelta(event);

    if (isSignalOnlyInteraction(curInteractionFlags())) {
        event->accept();
        return;
    }

    if (!(curInteractionFlags() & ecvGenericGLDisplay::INTERACT_ZOOM_CAMERA)) {
        event->accept();
        return;
    }

    if (keyboardModifiers & Qt::AltModifier) {
        event->accept();

        float sizeModifier = (delta < 0.0 ? -1.0f : 1.0f);
        displayTarget()->setPointSizeOnView(
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
                displayTarget()->setZNearCoef(newCoef);
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
                displayTarget()->setFov(newFOV);
                { ecvRedrawScope redrawScope; }
                doRedraw = true;
            }
        }
    } else if (curInteractionFlags() &
               ecvGenericGLDisplay::INTERACT_ZOOM_CAMERA) {
        float wheelDelta_deg = static_cast<float>(delta) / 8;

        if (m_directCameraWheelZoom) {
            applyDirectCameraWheelZoom(this, wheelDelta_deg);
        } else {
            QVTKOpenGLNativeWidget::wheelEvent(event);
        }

        if (!m_directCameraWheelZoom) {
            if (auto* dt = dynamic_cast<ecvDisplayTools*>(displayTarget()))
                dt->onWheelEvent(wheelDelta_deg);
        }
        if (m_ownerView) {
            emit m_ownerView->mouseWheelRotated(wheelDelta_deg);
            emit m_ownerView->cameraParamChanged();
        }
        emit mouseWheelRotated(wheelDelta_deg);
        emit cameraParamChanged();

        doRedraw = true;
        event->accept();
    }

    if (doRedraw) {
        if (m_ownerView) emit m_ownerView->labelmove2D(0, 0, 0, 0);
        emit labelmove2D(0, 0, 0, 0);

        if (m_wheelZoomUpdateTimer) {
            m_wheelZoomUpdateTimer->stop();
            m_wheelZoomUpdateTimer->start();
        }

        updateScaleBarIfNeeded();
        if (m_ownerView) {
            if (auto* w = m_ownerView->asWidget()) w->update();
            m_ownerView->updateScene();
        } else {
            displayTarget()->updateScene();
        }
    }
}

void QVTKWidgetCustom::mouseMoveEvent(QMouseEvent* event) {
    static QElapsedTimer s_framePerfTimer;
    static int s_perfFrameCount = 0;
    static qint64 s_perfAccum = 0;
    static qint64 s_vtkAccum = 0;
    static int s_pathTrace = 0;
    if (!s_framePerfTimer.isValid()) s_framePerfTimer.start();

    QElapsedTimer localTimer;
    localTimer.start();
    qint64 t_vtkRender = 0;

    if (handleCameraOrientationMouse(event, QEvent::MouseMove)) {
        event->accept();
        return;
    }

    // Do NOT call setActiveView here: ParaView-style UX activates a render view
    // on click (see mousePressEvent), not when the cursor merely moves across a
    // split window. Activating on every mouseMove caused the wrong view to
    // become "current" during passive hover.
    const bool vtkToolStyle =
            isVtkToolInteractorStyle(m_interactor) &&
            shouldForwardMouseEventToVtk(m_interactor, curInteractionFlags(),
                                         curRotationAxisLocked(), event,
                                         QEvent::MouseMove);
    if (vtkToolStyle && !m_labelClickedOnPress) {
        s_pathTrace = 1;
        qint64 t0 = localTimer.nsecsElapsed();
        QVTKOpenGLNativeWidget::mouseMoveEvent(event);
        t_vtkRender = localTimer.nsecsElapsed() - t0;
        if (event->buttons() != Qt::NoButton) {
            updateScaleBarIfNeeded();
            curMouseMoved() = true;
            curLastMousePos() = event->pos();

            // Performance logging for the primary rotation/pan path
            qint64 totalNs = localTimer.nsecsElapsed();
            s_perfAccum += totalNs;
            s_vtkAccum += t_vtkRender;
            s_perfFrameCount++;
            if (s_perfFrameCount >= 30) {
                double avgMs = (s_perfAccum / 1e6) / s_perfFrameCount;
                double avgVtkMs = (s_vtkAccum / 1e6) / s_perfFrameCount;
                double eventFps =
                        s_perfFrameCount * 1000.0 / s_framePerfTimer.elapsed();
                CVLog::PrintDebug(
                        QString("[PERF] vtkToolStyle avg=%1 ms (VTK=%2 ms) "
                                "eventRate=%3/s over %4 frames")
                                .arg(avgMs, 0, 'f', 2)
                                .arg(avgVtkMs, 0, 'f', 2)
                                .arg(eventFps, 0, 'f', 1)
                                .arg(s_perfFrameCount));
                s_perfAccum = 0;
                s_vtkAccum = 0;
                s_perfFrameCount = 0;
                s_framePerfTimer.restart();
            }

            event->accept();
            return;
        }
    }

    bool vtkHandledInteraction = false;

    if (!((curInteractionFlags() & ecvGenericGLDisplay::INTERACT_ROTATE) &&
          (curInteractionFlags() &
           ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES))) {
        if ((curInteractionFlags() &
             ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA)) {
            if (event->buttons() & Qt::MiddleButton) {
                if (m_ownerView) {
                    QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                    vtkHandledInteraction = true;
                } else {
                    event->accept();
                }
            } else {
                if (!m_labelClickedOnPress && curActiveItems().empty() &&
                    shouldForwardMouseEventToVtk(m_interactor,
                                                 curInteractionFlags(),
                                                 curRotationAxisLocked(), event,
                                                 QEvent::MouseMove)) {
                    s_pathTrace = 2;
                    qint64 t0 = localTimer.nsecsElapsed();
                    QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                    t_vtkRender = localTimer.nsecsElapsed() - t0;
                    updateScaleBarIfNeeded();
                    ecvDisplayTools::UpdateDisplayParameters();
                    vtkHandledInteraction = true;
                }
            }
        }
    }

    const int x = event->x();
    const int y = event->y();
    const bool isActiveWidget =
            (ecvGenericGLDisplay::FromWidget(this) == displayTarget());
    if (isActiveWidget) {
        curLastMouseMovePos() = event->pos();
        if (m_ownerView) emit m_ownerView->mousePosChanged(event->pos());
        emit mousePosChanged(event->pos());
    }

    if ((curInteractionFlags() &
         ecvGenericGLDisplay::INTERACT_SIG_MOUSE_MOVED) &&
        isActiveWidget) {
        if (m_ownerView) emit m_ownerView->mouseMoved(x, y, event->buttons());
        emit mouseMoved(x, y, event->buttons());
        event->accept();
    }

    if (isSignalOnlyInteraction(curInteractionFlags())) {
        curMouseMoved() = true;
        curLastMousePos() = event->pos();
        event->accept();
        return;
    }

    // no button pressed
    if (event->buttons() == Qt::NoButton) {
        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_CLICKABLE_ITEMS) {
            ecvHotZone* hz = nullptr;
            if (m_ownerView) {
                hz = curHotZone();
                if (!hz) {
                    hz = new ecvHotZone(this);
                    curHotZone() = hz;
                }
            } else {
                if (!m_localHotZone) {
                    m_localHotZone = new ecvHotZone(this);
                }
                if (curHotZone() && curHotZone() != m_localHotZone) {
                    delete curHotZone();
                    curHotZone() = nullptr;
                }
                curHotZone() = m_localHotZone;
                hz = m_localHotZone;
            }

            QRect areaRect = hz->rect(true, curBubbleViewModeEnabled(),
                                      displayTarget()->exclusiveFullScreen());

            const double dpr =
                    static_cast<double>(displayTarget()->getDevicePixelRatio());
            int scaledX = ecvDisplayCoordinates::toPhysical(x, dpr);
            int scaledY = ecvDisplayCoordinates::toPhysical(y, dpr);
            QRect zoneRect = areaRect.translated(hz->topCorner);
            bool inZone = zoneRect.contains(scaledX, scaledY);

            if (inZone != m_localClickableVisible) {
                m_localClickableVisible = inZone;

                curClickableItemsVisible() = inZone;
                displayTarget()->redraw(true, false);
            }

            event->accept();
        }

        // display the 3D coordinates of the pixel below the mouse cursor (if
        // possible), throttled to avoid excessive VTK ray-casts and redraws
        if (curShowCursorCoordinates()) {
            if (!m_cursorCoordTimerStarted) {
                m_cursorCoordTimer.start();
                m_cursorCoordTimerStarted = true;
            }
            if (m_cursorCoordTimer.elapsed() >= CURSOR_COORD_THROTTLE_MS) {
                m_cursorCoordTimer.restart();
                CCVector3d P;
                QString message = QString("2D (%1 ; %2)").arg(x).arg(y);
                if (displayTarget()->getClick3DPos(x, y, P)) {
                    message += QString(" --> 3D (%1 ; %2 ; %3)")
                                       .arg(P.x)
                                       .arg(P.y)
                                       .arg(P.z);
                }
                displayTarget()->displayNewMessage(
                        message, ecvGenericGLDisplay::LOWER_LEFT_MESSAGE, false,
                        5, ecvGenericGLDisplay::SCREEN_SIZE_MESSAGE);
                displayTarget()->redraw(true);
            }
        }

        // don't need to process any further
        s_pathTrace = 3;
        s_perfFrameCount++;
        if (s_perfFrameCount >= 30) {
            CVLog::PrintDebug(QString("[PERF] noButton path=%1 frames=%2")
                                      .arg(s_pathTrace)
                                      .arg(s_perfFrameCount));
            s_perfFrameCount = 0;
            s_framePerfTimer.restart();
        }
        return;
    }

    int dx = x - curLastMousePos().x();
    int dy = y - curLastMousePos().y();

    if ((event->buttons() & Qt::RightButton)) {
        if (abs(dx) > 0 || abs(dy) > 0) {
            if (m_ownerView) emit m_ownerView->labelmove2D(x, y, 0, 0);
            emit labelmove2D(x, y, 0, 0);
        }
    } else if ((event->buttons() & Qt::MiddleButton)) {
        if (!m_ownerView) {
            if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_PAN) {
                double pixSize = displayTarget()->computeActualPixelSize();
                CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
                if (!curViewportParams().perspectiveView) {
                    u.y *= curViewportParams().cameraAspectRatio;
                }

                const double dpr = static_cast<double>(
                        displayTarget()->getDevicePixelRatio());
                u *= dpr;

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
                        emit translation(u);
                    } else if (curCustomLightEnabled()) {
                        curCustomLightPos()[0] += static_cast<float>(u.x);
                        curCustomLightPos()[1] += static_cast<float>(u.y);
                        curCustomLightPos()[2] += static_cast<float>(u.z);
                        displayTarget()->invalidateViewport();
                        displayTarget()->deprecate3DLayer();
                    }
                } else {
                    if (curViewportParams().objectCenteredView) {
                        u = -u;
                    }
                    displayTarget()->moveCamera(static_cast<float>(u.x),
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
                    displayTarget()->updateActiveItemsList(
                            curLastMousePos().x(), curLastMousePos().y(), true);
                }
            }
        }

        // OPTIMIZATION: Skip 2D label updates during panning to improve
        // performance Only emit signal for label movement, but skip expensive
        // Update2DLabel during panning The label will be updated when panning
        // stops (in mouseReleaseEvent)
        if (abs(dx) > 0 || abs(dy) > 0) {
            if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
            emit labelmove2D(x, y, dx, dy);
            if (!curActiveItems().empty()) {
                updateActivateditems(x, y, dx, dy, true);
            }
        }
    } else if (event->buttons() & Qt::LeftButton)  // rotation
    {
        if (vtkHandledInteraction) {
            s_pathTrace = 5;
        } else if (!m_labelClickedOnPress) {
            s_pathTrace = 4;
        }

        if (curInteractionFlags() & ecvDisplayTools::INTERACT_2D_ITEMS) {
            if (!curMouseMoved() && !m_labelClickedOnPress) {
                if (curPickingMode() != ecvDisplayTools::NO_PICKING &&
                    (QApplication::keyboardModifiers() == Qt::NoModifier ||
                     QApplication::keyboardModifiers() ==
                             Qt::ControlModifier)) {
                    ecvDisplayTools::UpdateActiveItemsList(
                            curLastMousePos().x(), curLastMousePos().y(), true);
                }
            }
        } else if (!m_labelClickedOnPress) {
            curActiveItems().clear();
        }

        if (m_labelClickedOnPress) {
            if (abs(dx) > 0 || abs(dy) > 0) {
                updateActivateditems(x, y, dx, dy, true);
            }
        } else if (!curActiveItems().empty()) {
            if (abs(dx) > 0 || abs(dy) > 0) {
                if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
                emit labelmove2D(x, y, dx, dy);
                updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
            }
        } else {
            if (abs(dx) > 0 || abs(dy) > 0) {
                if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
                emit labelmove2D(x, y, dx, dy);
                if (!curActiveItems().empty()) {
                    updateActivateditems(x, y, dx, dy,
                                         !ecvDisplayTools::USE_2D);
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
                auto* evRect = displayTarget();
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
                        displayTarget()->addToOwnDB(curRectPickingPoly(),
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
                    CCVector3d pos2D =
                            evRect->toVtkCoordinates(event->x(), event->y());
                    B->x = C->x = static_cast<PointCoordinateType>(pos2D.x);
                    C->y = D->y = static_cast<PointCoordinateType>(pos2D.y);
                }
            }
            // standard rotation around the current pivot
            else if (!vtkHandledInteraction &&
                     (curInteractionFlags() &
                      ecvGenericGLDisplay::INTERACT_ROTATE)) {
                // choose the right rotation mode
                enum RotationMode { StandardMode, BubbleViewMode };
                RotationMode rotationMode = StandardMode;
                if ((curInteractionFlags() &
                     ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) !=
                    ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
                    if (curBubbleViewModeEnabled())
                        rotationMode = BubbleViewMode;
                }

                ccGLMatrixd rotMat;
                bool directCameraRotationApplied = false;
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
                        if (!curMouseMoved()) {
                            m_lastMouseOrientation =
                                    displayTarget()
                                            ->convertMousePositionToOrientation(
                                                    curLastMousePos().x(),
                                                    curLastMousePos().y());
                        }

                        CCVector3d currentMouseOrientation =
                                displayTarget()
                                        ->convertMousePositionToOrientation(x,
                                                                            y);
                        rotMat = ccGLMatrixd::FromToRotation(
                                m_lastMouseOrientation,
                                currentMouseOrientation);
                        m_lastMouseOrientation = currentMouseOrientation;
                    } break;

                    default:
                        break;
                }

                if (curInteractionFlags() &
                    ecvGenericGLDisplay::INTERACT_TRANSFORM_ENTITIES) {
                    rotMat = curViewportParams().viewMat.transposed() * rotMat *
                             curViewportParams().viewMat;
                    if (m_ownerView) emit m_ownerView->rotation(rotMat);
                    emit rotation(rotMat);
                } else if (!directCameraRotationApplied) {
                    displayTarget()->showPivotSymbol(true);
                    QApplication::changeOverrideCursor(
                            QCursor(Qt::ClosedHandCursor));

                    if (m_ownerView) emit m_ownerView->viewMatRotated(rotMat);
                    emit viewMatRotated(rotMat);
                } else {
                    displayTarget()->showPivotSymbol(true);
                    QApplication::changeOverrideCursor(
                            QCursor(Qt::ClosedHandCursor));
                }
            }
        }
    }

    curMouseMoved() = true;
    curLastMousePos() = event->pos();
    if (!m_labelClickedOnPress && !vtkHandledInteraction) {
        if (m_ownerView) emit m_ownerView->cameraParamChanged();
        emit cameraParamChanged();
        updateScaleBarIfNeeded();
    }

    // Performance logging: report every 30 frames
    qint64 totalNs = localTimer.nsecsElapsed();
    s_perfAccum += totalNs;
    s_vtkAccum += t_vtkRender;
    s_perfFrameCount++;
    if (s_perfFrameCount >= 30) {
        double avgMs = (s_perfAccum / 1e6) / s_perfFrameCount;
        double avgVtkMs = (s_vtkAccum / 1e6) / s_perfFrameCount;
        double eventFps =
                s_perfFrameCount * 1000.0 / s_framePerfTimer.elapsed();
        CVLog::PrintDebug(QString("[PERF] path=%1 avg=%2 ms (VTK=%3 ms) "
                                  "eventRate=%4/s over %5 frames")
                                  .arg(s_pathTrace)
                                  .arg(avgMs, 0, 'f', 2)
                                  .arg(avgVtkMs, 0, 'f', 2)
                                  .arg(eventFps, 0, 'f', 1)
                                  .arg(s_perfFrameCount));
        s_perfAccum = 0;
        s_vtkAccum = 0;
        s_perfFrameCount = 0;
        s_framePerfTimer.restart();
    }

    event->accept();
}

void QVTKWidgetCustom::updateActivateditems(
        int x, int y, int dx, int dy, bool updatePosition) {
    bool movedAs2D = false;
    if (updatePosition) {
        double pixSize = ecvDisplayTools::ComputeActualPixelSize();
        CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
        curViewportParams().viewMat.transposed().applyRotation(u);

        const double dpr =
                static_cast<double>(displayTarget()->getDevicePixelRatio());
        u *= dpr;

        for (auto& activeItem : curActiveItems()) {
            if (!m_labelClickedOnPress &&
                dynamic_cast<cc2DLabel*>(activeItem)) {
                continue;
            }
            if (activeItem->move2D(ecvDisplayCoordinates::toPhysical(x, dpr),
                                   ecvDisplayCoordinates::toPhysical(y, dpr),
                                   ecvDisplayCoordinates::toPhysical(dx, dpr),
                                   ecvDisplayCoordinates::toPhysical(dy, dpr),
                                   displayTarget()->glWidth(),
                                   displayTarget()->glHeight())) {
                movedAs2D = true;
            } else if (activeItem->move3D(u)) {
                displayTarget()->invalidateViewport();
                displayTarget()->deprecate3DLayer();
            }
        }
    }

    if (m_labelClickedOnPress) {
        for (auto& activeItem : curActiveItems()) {
            cc2DLabel* label = dynamic_cast<cc2DLabel*>(activeItem);
            if (label) {
                CC_DRAW_CONTEXT ctx;
                ecvDisplayTools::GetContext(ctx);
                label->update2DLabelView(ctx, false);
            }
        }
        update();
    } else {
        ecvDisplayTools::Redraw2DLabel();
    }
}

void QVTKWidgetCustom::mouseReleaseEvent(QMouseEvent* event) {
    stopInteractionRenderTimer();

    if (handleCameraOrientationMouse(event, QEvent::MouseButtonRelease)) {
        event->accept();
        return;
    }

    if (m_ownerView) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setPickingTargetView(m_ownerView);
    }

    if (isSignalOnlyInteraction(curInteractionFlags())) {
        curMouseButtonPressed() = false;
        curMouseMoved() = false;
        QApplication::restoreOverrideCursor();

        if (curInteractionFlags() &
            ecvGenericGLDisplay::INTERACT_SIG_BUTTON_RELEASED) {
            if (m_ownerView) emit m_ownerView->buttonReleased();
            emit buttonReleased();
        }

        event->accept();
        return;
    }

    if (shouldForwardMouseEventToVtk(m_interactor, curInteractionFlags(),
                                     curRotationAxisLocked(), event,
                                     QEvent::MouseButtonRelease) &&
        isVtkToolInteractorStyle(m_interactor) && !m_labelClickedOnPress) {
        QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
        curMouseButtonPressed() = false;
        curMouseMoved() = false;
        QApplication::restoreOverrideCursor();
        event->accept();
        return;
    }

    if (shouldForwardMouseEventToVtk(m_interactor, curInteractionFlags(),
                                     curRotationAxisLocked(), event,
                                     QEvent::MouseButtonRelease) &&
        !m_labelClickedOnPress) {
        QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
    } else if (m_labelClickedOnPress) {
        // Block VTK default mouse release while dragging a label.
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

    if (mouseHasMoved) {
        displayTarget()->updateNamePoseRecursive();
    }

    if (curInteractionFlags() &
        ecvGenericGLDisplay::INTERACT_SIG_BUTTON_RELEASED) {
        event->accept();
        if (m_ownerView) emit m_ownerView->buttonReleased();
        emit buttonReleased();
    }

    if (curPivotSymbolShown()) {
        if (curPivotVisibility() == ecvGenericGLDisplay::PIVOT_SHOW_ON_MOVE) {
            displayTarget()->toBeRefreshed();
        }
        displayTarget()->showPivotSymbol(
                curPivotVisibility() == ecvGenericGLDisplay::PIVOT_ALWAYS_SHOW);
    }

    if ((event->button() == Qt::MiddleButton)) {
        if (mouseHasMoved) {
            event->accept();
            curActiveItems().clear();
        } else if (curInteractionFlags() &
                   ecvGenericGLDisplay::INTERACT_2D_ITEMS) {
            // interaction with 2D item(s)
            displayTarget()->updateActiveItemsList(event->x(), event->y(),
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

                displayTarget()->removeFromOwnDB(curRectPickingPoly());
                curRectPickingPoly() = nullptr;
                vertices = nullptr;

                auto* w = displayTarget()->asWidget();
                const double dpr = ecvDisplayCoordinates::dprOf(w);
                const int pickRefW =
                        w ? ecvDisplayCoordinates::toPhysical(w->width(), dpr)
                          : 0;
                const int pickRefH =
                        w ? ecvDisplayCoordinates::toPhysical(w->height(), dpr)
                          : 0;
                displayTarget()->startPicking(
                        ecvGenericGLDisplay::ENTITY_RECT_PICKING,
                        pickX + pickRefW / 2, pickRefH / 2 - pickY, pickW,
                        pickH);
                displayTarget()->toBeRefreshed();
            }

            event->accept();
        } else {
            // picking?
            // CRITICAL: Don't start deferred picking if a VTK widget was
            // clicked This prevents doPicking() from overriding the widget
            // selection
            auto* pickDt = displayTarget();
            qint64 elapsed = pickDt ? pickDt->elapsedMs() : 0;
            qint64 clickTime = curLastClickTime();
            bool timeOk =
                    elapsed < clickTime + CC_MAX_PICKING_CLICK_DURATION_MS;
            if (!curWidgetClicked() && timeOk) {
                int x = curLastMousePos().x();
                int y = curLastMousePos().y();

                // Check if the click landed on a cc2DLabel's QPainter
                // overlay ROI (the VTK CAPTION widget has been replaced
                // by QPainter rendering, so VTK picking can't find it).
                bool labelPicked = false;
                if (curInteractionFlags() &
                    ecvDisplayTools::INTERACT_2D_ITEMS) {
                    ccHObject::Container labels;
                    collectAllLabels(labels);
                    for (auto* obj : labels) {
                        if (!obj->isA(CV_TYPES::LABEL_2D) ||
                            !obj->isBranchEnabled() || !obj->isVisible())
                            continue;
                        cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                        if (!l) continue;
                        QRect roi = l->getLabelROI();
                        if (roi.isValid() && roi.contains(x, y)) {
                            if (!l->isSelected()) {
                                emit ecvViewManager::instance()
                                        .entitySelectionChanged(l);
                                QApplication::processEvents();
                            }
                            labelPicked = true;
                            break;
                        }
                    }
                }

                if (!labelPicked) {
                    // first test if the user has clicked on a particular
                    // item on the screen
                    if (!ecvDisplayTools::ProcessClickableItems(x, y)) {
                        curLastMousePos() = event->pos();
                        if (m_ownerView) {
                            m_ownerView->startDeferredPicking();
                        } else if (auto* dtPick = ecvViewManager::instance()
                                                          .displayTools()) {
                            dtPick->startDeferredPickingFor(resolveDisplay());
                        }
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
            {
                ccHObject::Container labels;
                collectAllLabels(labels);
                for (auto* obj : labels) {
                    if (!obj->isA(CV_TYPES::LABEL_2D) ||
                        !obj->isBranchEnabled() || !obj->isVisible())
                        continue;
                    cc2DLabel* l = ccHObjectCaster::To2DLabel(obj);
                    if (!l) continue;
                    QRect roi = l->getLabelROI();
                    if (roi.isValid() && roi.contains(event->x(), event->y())) {
                        if (!l->isSelected()) {
                            emit ecvViewManager::instance()
                                    .entitySelectionChanged(l);
                            QApplication::processEvents();
                        }
                        if (l->acceptClick(event->x(), event->y(),
                                           Qt::RightButton)) {
                            ecvDisplayTools::Redraw2DLabel();
                            ecvDisplayTools::RedrawDisplay(true, true);
                            event->accept();
                        }
                        break;
                    }
                }
            }
            m_rightClickOnLabel = false;
        } else if (mouseHasMoved) {
            ecvDisplayTools::Update2DLabel(true);
        }
    }

    // CRITICAL: Always update 2D labels after any mouse interaction that moved
    // the camera (rotation, zoom, pan) to ensure labels stay aligned with their
    // 3D anchor points. This fixes the issue where labels become detached after
    // mouse release.
    if (mouseHasMoved) {
        displayTarget()->update2DLabels(true);
    }

    displayTarget()->refresh(true);
}

void QVTKWidgetCustom::dragEnterEvent(QDragEnterEvent* event) {
    const QMimeData* mimeData = event->mimeData();
    if (mimeData->hasFormat("text/uri-list")) event->acceptProposedAction();

    QVTKOpenGLNativeWidget::dragEnterEvent(event);
}

void QVTKWidgetCustom::dropEvent(QDropEvent* event) {
    const QMimeData* mimeData = event->mimeData();

    if (mimeData && mimeData->hasFormat("text/uri-list")) {
        // Activate this view before emitting filesDropped so that the
        // loaded entities are associated with the drop-target view.
        auto* display = ecvGenericGLDisplay::FromWidget(this);
        if (display) {
            auto& vm = ecvViewManager::instance();
            if (vm.getActiveView() != display) {
                vm.setActiveView(display);
            }
        }

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
            emit filesDropped(fileNames, true);
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
            case Qt::Key_M:
            case Qt::Key_Plus:
            case Qt::Key_Minus:
            case Qt::Key_Equal:
                return true;
        }
    } else if (ctrl && shift && !alt) {
        switch (key) {
            case Qt::Key_D:
            case Qt::Key_W:
            case Qt::Key_F:
            case Qt::Key_Plus:
            case Qt::Key_Minus:
                return true;
        }
    } else if (ctrl && !alt && !shift) {
        switch (key) {
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
            int qkey = keyEvent->key();
            if (qkey != Qt::Key_unknown && qkey != Qt::Key_Control &&
                qkey != Qt::Key_Shift && qkey != Qt::Key_Alt &&
                qkey != Qt::Key_Meta) {
                auto mods = keyEvent->modifiers();
                int combo = qkey;
                if (mods & Qt::ControlModifier) combo |= Qt::CTRL;
                if (mods & Qt::AltModifier) combo |= Qt::ALT;
                if (mods & Qt::ShiftModifier) combo |= Qt::SHIFT;
                QKeySequence seq(combo);
                auto* activeMs = ecvKeySequences::instance().active(seq);
                if (activeMs) {
                    evt->ignore();
                    return false;
                }
            }
            if (isVtkViewerShortcut(keyEvent->key(), keyEvent->modifiers())) {
                evt->accept();
                return true;
            }
            evt->ignore();
            return false;
        }

        case QEvent::MouseButtonPress: {
            QMouseEvent* me = static_cast<QMouseEvent*>(evt);
            if (me->button() == Qt::LeftButton ||
                me->button() == Qt::RightButton) {
                bool onLabel = false;
                if (curInteractionFlags() &
                    ecvDisplayTools::INTERACT_2D_ITEMS) {
                    ccHObject::Container labels;
                    collectAllLabels(labels);
                    for (auto* obj : labels) {
                        if (!obj->isA(CV_TYPES::LABEL_2D) ||
                            !obj->isBranchEnabled() || !obj->isVisible())
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
                    return QOpenGLWidget::event(evt);
                }
            }
        } break;

        case QEvent::MouseMove: {
            if (m_labelClickedOnPress || m_rightClickOnLabel) {
                return QOpenGLWidget::event(evt);
            }
            QMouseEvent* me = static_cast<QMouseEvent*>(evt);
            if (me->buttons() != Qt::NoButton && m_interactionRenderTimer &&
                m_interactionRenderTimer->isActive() &&
                !isVtkToolInteractorStyle(m_interactor)) {
                m_hasPendingMousePos = true;
                m_pendingMousePos = me->pos();
                evt->accept();
                return true;
            }
        } break;

        case QEvent::MouseButtonRelease: {
            if (m_labelClickedOnPress || m_rightClickOnLabel) {
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
                        displayTarget()->updateZoom(zoomFactor);
                    }
                    curTouchBaseDist() = dist;
                    evt->accept();
                    break;
                }
            }
        } break;

        case QEvent::Resize: {
            QSize newSize = static_cast<QResizeEvent*>(evt)->size();
            displayTarget()->resizeGL(newSize.width(), newSize.height());
            if (m_scaleBar) {
                if (!m_scaleBar->isLayoutReady() && isVisible() &&
                    newSize.width() > 1 && newSize.height() > 1) {
                    QTimer::singleShot(0, this, [this]() {
                        if (m_scaleBar && isVisible()) {
                            m_scaleBar->notifyLayoutReady();
                            m_scaleBar->update(m_render, m_interactor);
                        }
                    });
                } else if (m_scaleBar->isLayoutReady()) {
                    m_scaleBar->update(m_render, m_interactor);
                }
            }
            evt->accept();
        } break;

        case QEvent::KeyPress: {
            QKeyEvent* keyEvent = static_cast<QKeyEvent*>(evt);

            if (keyEvent->key() == Qt::Key_Escape) {
                if (m_ownerView)
                    emit m_ownerView->exclusiveFullScreenToggled(false);
                emit exclusiveFullScreenToggled(false);
                if (m_win) {
                    QKeyEvent* newEvent =
                            new QKeyEvent(QEvent::KeyPress, Qt::Key_Escape,
                                          keyEvent->modifiers());
                    QCoreApplication::postEvent(m_win, newEvent);
                }
                evt->accept();
                return true;
            }

            {
                int qkey = keyEvent->key();
                auto mods = keyEvent->modifiers();
                if (qkey == Qt::Key_unknown || qkey == Qt::Key_Control ||
                    qkey == Qt::Key_Shift || qkey == Qt::Key_Alt ||
                    qkey == Qt::Key_Meta) {
                    break;
                }

                int combo = qkey;
                if (mods & Qt::ControlModifier) combo |= Qt::CTRL;
                if (mods & Qt::AltModifier) combo |= Qt::ALT;
                if (mods & Qt::ShiftModifier) combo |= Qt::SHIFT;
                QKeySequence seq(combo);

                auto* modalShortcut = ecvKeySequences::instance().active(seq);
                if (modalShortcut) {
                    evt->ignore();
                    return false;
                }

                if (m_customStyle && m_interactor) {
                    ensureVtkShortcutMap();

                    QString seqStr = seq.toString(QKeySequence::PortableText);

                    auto it = s_vtkShortcutMap.find(seqStr);
                    if (it != s_vtkShortcutMap.end()) {
                        if (m_customStyle->handleShortcut(
                                    it->vtkKey, it->vtkCtrl, it->vtkAlt,
                                    it->vtkShift, m_interactor.Get())) {
                            evt->accept();
                            return true;
                        }
                    }

                    bool noMods =
                            !(mods & (Qt::ControlModifier | Qt::AltModifier |
                                      Qt::MetaModifier));
                    if (noMods && qkey >= Qt::Key_A && qkey <= Qt::Key_Z) {
                        evt->ignore();
                        return false;
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
