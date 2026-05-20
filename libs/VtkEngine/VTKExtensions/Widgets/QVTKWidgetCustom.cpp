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
#include "VTKExtensions/Widgets/VtkShortcutRegistry.h"

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvGenericGLDisplay.h>
#include <ecvInteractor.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRedrawScope.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewRepresentation.h>

#include <Visualization/vtkGLView.h>

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

    // Prevent the native GL window from being visible before the widget is
    // properly placed in a layout. On Linux/X11, native child windows bypass
    // Qt stacking and can render over the menu bar. The widget becomes visible
    // when the layout system calls show() on it.
    hide();

    vtkObject::GlobalWarningDisplayOff();
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

    curLastPointIndex() = -1;
    curLastPickedId() = QString();

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

        if (curAutoPickPivotAtCenter()) {
            CCVector3d P;
            if (ecvDisplayTools::GetClick3DPos(event->x(), event->y(), P)) {
                ecvDisplayTools::SetPivotPoint(P, true, false);
            }
        }
    } else {
    }

    if (m_labelClickedOnPress || m_rightClickOnLabel) {
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

void QVTKWidgetCustom::paintGL() {
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
            CVLog::Warning("[paintGL] label '%s' overlay INVALID after "
                           "update2DLabelView (dispIn2D=%d pts=%d)",
                           qPrintable(label->getName()),
                           label->isDisplayedIn2D(),
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
        QVTKOpenGLNativeWidget::wheelEvent(event);

        float wheelDelta_deg = static_cast<float>(delta) / 8;
        if (auto* dt = dynamic_cast<ecvDisplayTools*>(displayTarget()))
            dt->onWheelEvent(wheelDelta_deg);
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
        displayTarget()->updateNamePoseRecursive();

        if (m_wheelZoomUpdateTimer) {
            m_wheelZoomUpdateTimer->stop();
            m_wheelZoomUpdateTimer->start();
        }

        if (m_ownerView) {
            if (auto* w = m_ownerView->asWidget()) w->update();
            m_ownerView->updateCamera();
        } else {
            displayTarget()->updateScene();
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
                if (!m_labelClickedOnPress && curActiveItems().empty()) {
                    QVTKOpenGLNativeWidget::mouseMoveEvent(event);
                    ecvDisplayTools::UpdateDisplayParameters();
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

            const int retinaScale = displayTarget()->getDevicePixelRatio();
            bool inZone = (x * retinaScale * 3 <
                                   hz->topCorner.x() + areaRect.width() * 4 &&
                           y * retinaScale * 2 <
                                   hz->topCorner.y() + areaRect.height() * 4);

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
        return;
    }

    int dx = x - curLastMousePos().x();
    int dy = y - curLastMousePos().y();

    if ((event->buttons() & Qt::RightButton)) {
        if (abs(dx) > 0 || abs(dy) > 0) {
            if (m_ownerView) emit m_ownerView->labelmove2D(x, y, 0, 0);
            emit labelmove2D(x, y, 0, 0);
            displayTarget()->updateNamePoseRecursive();
        }
    } else if ((event->buttons() & Qt::MiddleButton)) {
        if (!m_ownerView) {
            if (curInteractionFlags() & ecvGenericGLDisplay::INTERACT_PAN) {
                double pixSize = displayTarget()->computeActualPixelSize();
                CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
                if (!curViewportParams().perspectiveView) {
                    u.y *= curViewportParams().cameraAspectRatio;
                }

                const int retinaScale = displayTarget()->getDevicePixelRatio();
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
            displayTarget()->updateNamePoseRecursive();
            if (!curActiveItems().empty()) {
                updateActivateditems(x, y, dx, dy, true);
            }
        }
    } else if (event->buttons() & Qt::LeftButton)  // rotation
    {
        if (!m_labelClickedOnPress) {
            if (auto* dt = displayTarget()) dt->scheduleFullRedraw(1000);
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
                ecvDisplayTools::UpdateNamePoseRecursive();
                updateActivateditems(x, y, dx, dy, !ecvDisplayTools::USE_2D);
            }
        } else {
            if (abs(dx) > 0 || abs(dy) > 0) {
                if (m_ownerView) emit m_ownerView->labelmove2D(x, y, dx, dy);
                emit labelmove2D(x, y, dx, dy);
                ecvDisplayTools::UpdateNamePoseRecursive();
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
            else if (curInteractionFlags() &
                     ecvGenericGLDisplay::INTERACT_ROTATE) {
                CVLog::PrintDebug(
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

                    case LockedAxisMode: {
                        // apply rotation about the locked axis
                        CCVector3d axis = curLockedRotationAxis();
                        // curViewportParams().objectCenteredView
                        ccGLCameraParameters camera;
                        displayTarget()->getGLCameraParameters(camera);
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
                        displayTarget()->rotateWithAxis(
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
                    emit rotation(rotMat);
                } else {
                    displayTarget()->showPivotSymbol(true);
                    QApplication::changeOverrideCursor(
                            QCursor(Qt::ClosedHandCursor));

                    if (m_ownerView) emit m_ownerView->viewMatRotated(rotMat);
                    emit viewMatRotated(rotMat);
                }
            }
        }
    }

    curMouseMoved() = true;
    curLastMousePos() = event->pos();
    if (!m_labelClickedOnPress) {
        emit cameraParamChanged();
        if (m_scaleBar) m_scaleBar->update(m_render, m_interactor);
    }
    event->accept();
}

void QVTKWidgetCustom::updateActivateditems(
        int x, int y, int dx, int dy, bool updatePosition) {
    CVLog::PrintDebug(
            "[Label] updateActivateditems: pos(%d,%d) delta(%d,%d) "
            "updatePos=%d activeItems=%zu labelDrag=%d",
            x, y, dx, dy, updatePosition, curActiveItems().size(),
            m_labelClickedOnPress);
    bool movedAs2D = false;
    if (updatePosition) {
        double pixSize = ecvDisplayTools::ComputeActualPixelSize();
        CCVector3d u(dx * pixSize, -dy * pixSize, 0.0);
        curViewportParams().viewMat.transposed().applyRotation(u);

        const int retinaScale = displayTarget()->getDevicePixelRatio();
        u *= retinaScale;

        for (auto& activeItem : curActiveItems()) {
            if (!m_labelClickedOnPress &&
                dynamic_cast<cc2DLabel*>(activeItem)) {
                continue;
            }
            if (activeItem->move2D(x * retinaScale, y * retinaScale,
                                   dx * retinaScale, dy * retinaScale,
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
    if (m_ownerView) {
        if (auto* dt = ecvViewManager::instance().displayTools())
            dt->setPickingTargetView(m_ownerView);
    }

    if ((curInteractionFlags() & ecvDisplayTools::TRANSFORM_CAMERA()) &&
        !m_labelClickedOnPress) {
        QVTKOpenGLNativeWidget::mouseReleaseEvent(event);
    } else if (m_labelClickedOnPress) {
        CVLog::PrintDebug(
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
                const int pickRefW = w ? w->width() : 0;
                const int pickRefH = w ? w->height() : 0;
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
            CVLog::PrintDebug(
                    QString("[mouseRelease] Pick check: "
                            "widgetClicked=%1 elapsed=%2 clickTime=%3 "
                            "threshold=%4 timeOk=%5 mouseHasMoved=%6")
                            .arg(curWidgetClicked())
                            .arg(elapsed)
                            .arg(clickTime)
                            .arg(CC_MAX_PICKING_CLICK_DURATION_MS)
                            .arg(timeOk)
                            .arg(mouseHasMoved));
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
                        CVLog::PrintDebug(QString("[mouseRelease] Starting "
                                                  "deferred pick at (%1,%2) "
                                                  "ownerView=%3 pickMode=%4")
                                                  .arg(event->pos().x())
                                                  .arg(event->pos().y())
                                                  .arg(m_ownerView != nullptr)
                                                  .arg(static_cast<int>(
                                                          curPickingMode())));
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
            if (isVtkViewerShortcut(keyEvent->key(), keyEvent->modifiers())) {
                evt->accept();
                return true;
            }
        } break;

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
                CVLog::Print(
                        "[QVTKWidgetCustom] ESC key pressed, forwarding to "
                        "MainWindow");
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

            if (m_customStyle && m_interactor) {
                ensureVtkShortcutMap();

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
                QString seqStr = QKeySequence(combo).toString(
                        QKeySequence::PortableText);

                auto it = s_vtkShortcutMap.find(seqStr);
                if (it != s_vtkShortcutMap.end()) {
                    if (m_customStyle->handleShortcut(
                                it->vtkKey, it->vtkCtrl, it->vtkAlt,
                                it->vtkShift, m_interactor.Get())) {
                        evt->accept();
                        return true;
                    }
                }

                bool noMods = !(mods & (Qt::ControlModifier |
                                        Qt::AltModifier | Qt::MetaModifier));
                if (noMods && qkey >= Qt::Key_A && qkey <= Qt::Key_Z) {
                    evt->ignore();
                    return false;
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
