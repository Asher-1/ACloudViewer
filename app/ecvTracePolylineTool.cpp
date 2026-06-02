// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTracePolylineTool.h"

// Local
#include "MainWindow.h"

// common
#include <ecvPickingHub.h>

// CV_CORE_LIB
#include <CVLog.h>
#include <ManualSegmentationTools.h>
#include <SquareMatrix.h>

// CV_DB_LIB
#include <ecv2DViewportObject.h>
#include <ecvDisplayTools.h>
#include <ecvDrawContext.h>
#include <ecvGenericGLDisplay.h>
#include <ecvGenericPointCloud.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvViewManager.h>

// Qt
#include <QApplication>
#include <QMenu>
#include <QMessageBox>
#include <QProgressDialog>
#include <QPushButton>

// System
#include <assert.h>

#ifdef USE_VTK_BACKEND
#include <Visualization/vtkGLView.h>
#endif

namespace {

ecvGenericGLDisplay* viewFromWidget(QWidget* win) {
    return win ? ecvGenericGLDisplay::FromWidget(win) : nullptr;
}

ecvGenericGLDisplay* effectiveView() {
    return ecvViewManager::instance().getEffectiveView();
}

ecvDisplayTools* displayTools() {
    return ecvViewManager::instance().displayTools();
}

CCVector3 mouseToVtk2D(ecvGenericGLDisplay* view, int x, int y) {
    if (!view) return CCVector3(0, 0, 0);
    CCVector3d pos2D = view->toVtkCoordinates(x, y);
    return CCVector3(static_cast<PointCoordinateType>(pos2D.x),
                     static_cast<PointCoordinateType>(pos2D.y), 0);
}

CCVector3 point3DToVtk2D(ecvGenericGLDisplay* view, const CCVector3& P3D) {
    if (!view) return CCVector3(0, 0, 0);
    ccGLCameraParameters camera;
    view->getGLCameraParameters(camera);
    CCVector3d A2D;
    camera.project(P3D, A2D);
    return CCVector3(static_cast<PointCoordinateType>(A2D.x),
                     static_cast<PointCoordinateType>(A2D.y), 0);
}

}  // namespace

ccTracePolylineTool::SegmentGLParams::SegmentGLParams(int x, int y) {
    auto* view = ecvViewManager::instance().getEffectiveView();
    if (view) {
        view->getGLCameraParameters(params);
        CCVector3d pos2D = view->toVtkCoordinates(x, y);
        clickPos = CCVector2d(pos2D.x, pos2D.y);
    }
}

ccTracePolylineTool::ccTracePolylineTool(ccPickingHub* pickingHub,
                                         QWidget* parent)
    : ccOverlayDialog(parent),
      Ui::TracePolyLineDlg(),
      m_polyTip(nullptr),
      m_polyTipVertices(nullptr),
      m_poly3D(nullptr),
      m_poly3DVertices(nullptr),
      m_done(false),
      m_pickingHub(pickingHub) {
    assert(pickingHub);

    setupUi(this);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);

    connect(saveToolButton, &QToolButton::clicked, this,
            &ccTracePolylineTool::exportLine);
    connect(resetToolButton, &QToolButton::clicked, this,
            &ccTracePolylineTool::resetLine);
    connect(continueToolButton, &QToolButton::clicked, this,
            &ccTracePolylineTool::continueEdition);
    connect(validButton, &QToolButton::clicked, this,
            &ccTracePolylineTool::apply);
    connect(cancelButton, &QToolButton::clicked, this,
            &ccTracePolylineTool::cancel);
    connect(widthSpinBox,
            static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
            &ccTracePolylineTool::onWidthSizeChanged);

    // add shortcuts
    addOverridenShortcut(Qt::Key_Escape);  // escape key for the "cancel" button
    addOverridenShortcut(Qt::Key_Return);  // return key for the "apply" button
    connect(this, &ccTracePolylineTool::shortcutTriggered, this,
            &ccTracePolylineTool::onShortcutTriggered);

    m_polyTipVertices = new ccPointCloud("Tip vertices");
    m_polyTipVertices->reserve(2);
    m_polyTipVertices->addPoint(CCVector3(0, 0, 0));
    m_polyTipVertices->addPoint(CCVector3(1, 1, 1));
    m_polyTipVertices->setEnabled(false);

    m_polyTip = new ccPolyline(m_polyTipVertices);
    m_polyTip->setForeground(true);
    m_polyTip->setTempColor(ecvColor::green);
    m_polyTip->set2DMode(true);
    m_polyTip->reserve(2);
    m_polyTip->addPointIndex(0, 2);
    m_polyTip->setWidth(
            widthSpinBox->value() < 2
                    ? 0
                    : widthSpinBox->value());  //'1' is equivalent to the
                                               // default line size
    m_polyTip->addChild(m_polyTipVertices);

    validButton->setEnabled(false);
}

ccTracePolylineTool::~ccTracePolylineTool() {
    if (m_polyTip) delete m_polyTip;
    // DGM: already a child of m_polyTip
    // if (m_polyTipVertices)
    //	delete m_polyTipVertices;

    if (m_poly3D) delete m_poly3D;
    // DGM: already a child of m_poly3D
    // if (m_poly3DVertices)
    //	delete m_poly3DVertices;
}

void ccTracePolylineTool::onShortcutTriggered(int key) {
    switch (key) {
        case Qt::Key_Return:
            apply();
            return;

        case Qt::Key_Escape:
            cancel();
            return;

        default:
            // nothing to do
            break;
    }
}

ccPolyline* ccTracePolylineTool::polylineOverSampling(unsigned steps) const {
    if (!m_poly3D || !m_poly3DVertices ||
        m_segmentParams.size() != m_poly3DVertices->size()) {
        assert(false);
        return nullptr;
    }

    if (steps <= 1) {
        // nothing to do
        return nullptr;
    }

    ccHObject* sceneDB = nullptr;
    if (auto* sceneView = ecvViewManager::instance().getEffectiveView()) {
        sceneDB = sceneView->getSceneDB();
    }
    ccHObject::Container clouds;
    if (sceneDB)
        sceneDB->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD, false);
    ccHObject::Container meshes;
    if (sceneDB) sceneDB->filterChildren(meshes, true, CV_TYPES::MESH, false);

    if (clouds.empty() && meshes.empty()) {
        // no entity is currently displayed?!
        assert(false);
        return nullptr;
    }

    unsigned n_verts = m_poly3DVertices->size();
    unsigned n_segments = m_poly3D->size() - (m_poly3D->isClosed() ? 0 : 1);
    unsigned end_size = n_segments * steps + (m_poly3D->isClosed() ? 0 : 1);

    ccPointCloud* newVertices = new ccPointCloud();
    ccPolyline* newPoly = new ccPolyline(newVertices);
    newPoly->addChild(newVertices);

    if (!newVertices->reserve(end_size) || !newPoly->reserve(end_size)) {
        CVLog::Warning(
                "[ccTracePolylineTool::PolylineOverSampling] Not enough "
                "memory");
        delete newPoly;
        return nullptr;
    }
    newVertices->importParametersFrom(m_poly3DVertices);
    newVertices->setName(m_poly3DVertices->getName());
    newVertices->setEnabled(m_poly3DVertices->isEnabled());
    newPoly->importParametersFrom(*m_poly3D);

    QProgressDialog pDlg(QString("Oversampling"), "Cancel", 0,
                         static_cast<int>(end_size),
                         ecvViewManager::instance().activeWidget());
    pDlg.show();
    QCoreApplication::processEvents();

    for (unsigned i = 0; i < n_segments; ++i) {
        const CCVector3* p1 = m_poly3DVertices->getPoint(i);
        newVertices->addPoint(*p1);

        unsigned i2 = (i + 1) % n_verts;
        CCVector2d v =
                m_segmentParams[i2].clickPos - m_segmentParams[i].clickPos;
        v /= steps;

        for (unsigned j = 1; j < steps; j++) {
            CCVector2d vj = m_segmentParams[i].clickPos + v * j;

            CCVector3 nearestPoint;
            double nearestElementSquareDist = -1.0;

            // for each cloud
            for (size_t c = 0; c < clouds.size(); ++c) {
                ccGenericPointCloud* cloud =
                        static_cast<ccGenericPointCloud*>(clouds[c]);

                int nearestPointIndex = -1;
                double nearestSquareDist = 0;
                if (cloud->pointPicking(vj, m_segmentParams[i2].params,
                                        nearestPointIndex, nearestSquareDist,
                                        snapSizeSpinBox->value(),
                                        snapSizeSpinBox->value(), true)) {
                    if (nearestElementSquareDist < 0 ||
                        nearestSquareDist < nearestElementSquareDist) {
                        nearestElementSquareDist = nearestSquareDist;
                        nearestPoint = *cloud->getPoint(nearestPointIndex);
                    }
                }
            }

            // for each mesh
            for (size_t m = 0; m < meshes.size(); ++m) {
                ccGenericMesh* mesh = static_cast<ccGenericMesh*>(meshes[m]);
                int nearestTriIndex = -1;
                double nearestSquareDist = 0;
                CCVector3d _nearestPoint;

                if (mesh->trianglePicking(vj, m_segmentParams[i2].params,
                                          nearestTriIndex, nearestSquareDist,
                                          _nearestPoint)) {
                    if (nearestElementSquareDist < 0 ||
                        nearestSquareDist < nearestElementSquareDist) {
                        nearestElementSquareDist = nearestSquareDist;
                        nearestPoint = CCVector3::fromArray(_nearestPoint.u);
                    }
                }
            }

            if (nearestElementSquareDist >= 0) {
                newVertices->addPoint(nearestPoint);
            }

            if (pDlg.wasCanceled()) {
                steps = 0;  // quick finish ;)
                break;
            }
            pDlg.setValue(pDlg.value() + 1);
        }
    }

    // add last point
    if (!m_poly3D->isClosed()) {
        newVertices->addPoint(*m_poly3DVertices->getPoint(n_verts - 1));
    }

    newVertices->shrinkToFit();
    newPoly->addPointIndex(0, newVertices->size());

    return newPoly;
}

bool ccTracePolylineTool::linkWith(QWidget* win) {
    assert(m_polyTip);
    assert(!m_poly3D);

    ecvGenericGLDisplay* oldView = viewFromWidget(m_associatedWin);

    if (!ccOverlayDialog::linkWith(win)) {
        return false;
    }

    if (oldView) {
        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_polyTip->getViewId()));
#ifdef USE_VTK_BACKEND
        if (auto* oldGlView = dynamic_cast<vtkGLView*>(oldView)) {
            QObject::disconnect(oldGlView, nullptr, this, nullptr);
        }
#endif
        if (m_polyTip) m_polyTip->setDisplay(nullptr);
    }

    ecvGenericGLDisplay* newView = viewFromWidget(win);
    ecvViewManager& vm = ecvViewManager::instance();
    connect(&vm, &ecvViewManager::rightButtonClicked, this,
            [this](int x, int y) { closePolyLine(x, y); },
            Qt::UniqueConnection);
    connect(&vm, &ecvViewManager::mouseMoved, this,
            [this](int x, int y, Qt::MouseButtons buttons) {
                updatePolyLineTip(x, y, buttons);
            },
            Qt::UniqueConnection);

#ifdef USE_VTK_BACKEND
    auto bindGlView = [this](vtkGLView* glView) {
        if (!glView) return;
        QObject::connect(glView, SIGNAL(rightButtonClicked(int, int)), this,
                         SLOT(closePolyLine(int, int)),
                         Qt::UniqueConnection);
        QObject::connect(glView, SIGNAL(mouseMoved(int, int, Qt::MouseButtons)),
                         this, SLOT(updatePolyLineTip(int, int, Qt::MouseButtons)),
                         Qt::UniqueConnection);
    };
    if (auto* glView = dynamic_cast<vtkGLView*>(newView)) {
        bindGlView(glView);
    }
    if (auto* active = dynamic_cast<vtkGLView*>(vm.getActiveView())) {
        if (active != dynamic_cast<vtkGLView*>(newView)) bindGlView(active);
    }
#endif

    return true;
}

static int s_defaultPickingRadius = 1;
static int s_overSamplingCount = 1;
bool ccTracePolylineTool::start() {
    assert(m_polyTip);
    assert(!m_poly3D);

    auto* view = ecvViewManager::instance().getEffectiveView();
    if (!view || !view->asWidget()) {
        CVLog::Warning("[Trace Polyline Tool] No associated window!");
        return false;
    }

#ifdef USE_VTK_BACKEND
    if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
        QObject::connect(glView, SIGNAL(rightButtonClicked(int, int)), this,
                         SLOT(closePolyLine(int, int)),
                         Qt::UniqueConnection);
        QObject::connect(glView, SIGNAL(mouseMoved(int, int, Qt::MouseButtons)),
                         this,
                         SLOT(updatePolyLineTip(int, int, Qt::MouseButtons)),
                         Qt::UniqueConnection);
    }
#endif

    if (m_pickingHub) {
        m_pickingHub->removeListener(this);
    }
    view->setPickingMode(ecvGenericGLDisplay::NO_PICKING);
    view->setInteractionMode(ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA |
                             ecvGenericGLDisplay::INTERACT_SIG_RB_CLICKED |
                             ecvGenericGLDisplay::INTERACT_CTRL_PAN |
                             ecvGenericGLDisplay::INTERACT_SIG_MOUSE_MOVED);
    view->asWidget()->setCursor(Qt::CrossCursor);
    view->asWidget()->setMouseTracking(true);

    m_savedPivot = view->getViewportParameters().getPivotPoint();
    if (const ecvViewContext* ctx = view->viewContext()) {
        m_savedAutoPickPivot = ctx->autoPickPivotAtCenter;
    } else {
        m_savedAutoPickPivot = false;
    }
    m_hasSavedViewState = true;
    view->setAutoPickPivotAtCenter(false);

    m_polyTip->setDisplay(displayTools());

    snapSizeSpinBox->blockSignals(true);
    snapSizeSpinBox->setValue(s_defaultPickingRadius);
    snapSizeSpinBox->blockSignals(false);

    oversampleSpinBox->blockSignals(true);
    oversampleSpinBox->setValue(s_overSamplingCount);
    oversampleSpinBox->blockSignals(false);

    resetLine();  // to reset the GUI

    return ccOverlayDialog::start();
}

void ccTracePolylineTool::stop(bool accepted) {
    assert(m_polyTip);

    if (m_pickingHub) {
        m_pickingHub->removeListener(this);
    }

    ecvViewManager& vm = ecvViewManager::instance();
    QObject::disconnect(&vm, nullptr, this, nullptr);
#ifdef USE_VTK_BACKEND
    if (auto* glView = dynamic_cast<vtkGLView*>(vm.getActiveView())) {
        QObject::disconnect(glView, nullptr, this, nullptr);
    }
    if (auto* effView = dynamic_cast<vtkGLView*>(vm.getEffectiveView())) {
        if (effView != dynamic_cast<vtkGLView*>(vm.getActiveView())) {
            QObject::disconnect(effView, nullptr, this, nullptr);
        }
    }
#endif

    auto* stopView = vm.getEffectiveView();
    if (stopView && stopView->asWidget()) {
        vm.displayMessageOnActiveView(
                "Polyline tracing [OFF]",
                ecvGenericGLDisplay::UPPER_CENTER_MESSAGE, false, 2,
                ecvGenericGLDisplay::MANUAL_SEGMENTATION_MESSAGE);

        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_polyTip->getViewId()));
        if (m_poly3D) stopView->removeFromOwnDB(m_poly3D);

        stopView->setInteractionMode(
                ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);
        stopView->setPickingMode(ecvGenericGLDisplay::DEFAULT_PICKING);
        stopView->asWidget()->setCursor(Qt::ArrowCursor);
        stopView->asWidget()->setMouseTracking(false);

        if (m_hasSavedViewState) {
            stopView->setAutoPickPivotAtCenter(m_savedAutoPickPivot);
            stopView->setPivotPoint(m_savedPivot, true, false);
            m_hasSavedViewState = false;
        }

        stopView->redraw(true, false);
    }

    s_defaultPickingRadius = snapSizeSpinBox->value();
    s_overSamplingCount = oversampleSpinBox->value();

    ccOverlayDialog::stop(accepted);
}

void ccTracePolylineTool::updatePolyLineTip(int x,
                                            int y,
                                            Qt::MouseButtons buttons) {
    auto* view = effectiveView();
    if (!view) {
        return;
    }

    if (buttons != Qt::NoButton) {
        if (m_polyTip->isEnabled()) {
            m_polyTip->setEnabled(false);
            ecvDisplayTools::RemoveWidgets(
                    WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                      m_polyTip->getViewId()));
            ecvDisplayTools::UpdateScreen();
        }
        return;
    }

    if (!m_poly3DVertices || m_poly3DVertices->size() == 0) {
        return;
    }

    if (m_done) {
        return;
    }

    assert(m_polyTip && m_polyTipVertices && m_polyTipVertices->size() == 2);

    CCVector3* lastP = const_cast<CCVector3*>(
            m_polyTipVertices->getPointPersistentPtr(1));
    *lastP = mouseToVtk2D(view, x, y);

    const CCVector3* P3D =
            m_poly3DVertices->getPoint(m_poly3DVertices->size() - 1);
    CCVector3* firstP = const_cast<CCVector3*>(
            m_polyTipVertices->getPointPersistentPtr(0));
    *firstP = point3DToVtk2D(view, *P3D);

    m_polyTip->setEnabled(true);
    ecvDisplayTools::RemoveWidgets(
            WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                              m_polyTip->getViewId()));
    ecvDisplayTools::DrawWidgets(
            WIDGETS_PARAMETER(m_polyTip, WIDGETS_TYPE::WIDGET_POLYLINE_2D),
            false);
    view->redraw(true, false);
}

void ccTracePolylineTool::onItemPicked(const PickedItem& pi) {
    auto* view = pi.pickView;
    if (!view) view = ecvViewManager::instance().getActiveView();
    if (!view) view = ecvViewManager::instance().getEffectiveView();
    if (!view) {
        assert(false);
        return;
    }

    if (!pi.entity) {
        return;
    }

    if (!m_poly3D || !m_poly3DVertices) {
        m_poly3DVertices = new ccPointCloud("Vertices");
        m_poly3DVertices->setEnabled(false);
        m_poly3DVertices->setDisplay(view);

        m_poly3D = new ccPolyline(m_poly3DVertices);
        m_poly3D->setTempColor(ecvColor::green);
        m_poly3D->set2DMode(false);
        m_poly3D->addChild(m_poly3DVertices);
        m_poly3D->setDisplay(view);
        m_poly3D->setWidth(
                widthSpinBox->value() < 2
                        ? 0
                        : widthSpinBox->value());

        ccGenericPointCloud* cloud =
                ccHObjectCaster::ToGenericPointCloud(pi.entity);
        if (cloud) {
            m_poly3D->setGlobalShift(cloud->getGlobalShift());
            m_poly3D->setGlobalScale(cloud->getGlobalScale());
        }

        m_segmentParams.resize(0);
        view->addToOwnDB(m_poly3D);
    }

    // try to add one more point
    if (!m_poly3DVertices->reserve(m_poly3DVertices->size() + 1) ||
        !m_poly3D->reserve(m_poly3DVertices->size() + 1)) {
        CVLog::Error("Not enough memory");
        return;
    }

    try {
        m_segmentParams.reserve(m_segmentParams.size() + 1);
    } catch (const std::bad_alloc&) {
        CVLog::Error("Not enough memory");
        return;
    }

    m_poly3DVertices->addPoint(pi.P3D);
    m_poly3D->addPointIndex(m_poly3DVertices->size() - 1);
    m_segmentParams.emplace_back(pi.clickPoint.x(), pi.clickPoint.y());

    CCVector3* firstTipPoint = const_cast<CCVector3*>(
            m_polyTipVertices->getPointPersistentPtr(0));
    *firstTipPoint =
            mouseToVtk2D(view, pi.clickPoint.x(), pi.clickPoint.y());
    m_polyTip->setEnabled(false);
    ecvDisplayTools::RemoveWidgets(
            WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                              m_polyTip->getViewId()));

    view->redraw(false, true);

    if (m_poly3D && m_poly3D->size() >= 2 && !m_done) {
        saveToolButton->setEnabled(true);
        resetToolButton->setEnabled(true);
    }
}

void ccTracePolylineTool::closePolyLine(int, int) {
    // CTRL + right click = panning
    if (!m_poly3D ||
        (QApplication::keyboardModifiers() & Qt::ControlModifier)) {
        return;
    }

    unsigned vertCount = m_poly3D->size();
    if (vertCount < 2) {
        // discard this polyline
        resetLine();
    } else {
        if (m_polyTip) {
            m_polyTip->setEnabled(false);
        }
        validButton->setEnabled(true);
        saveToolButton->setEnabled(true);
        resetToolButton->setEnabled(true);
        continueToolButton->setEnabled(true);
        if (m_pickingHub) {
            m_pickingHub->removeListener(this);
        }
        auto* v = effectiveView();
        if (v) {
            v->setPickingMode(ecvGenericGLDisplay::NO_PICKING);
        }
        m_done = true;

        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_polyTip->getViewId()));
        if (v) v->redraw(true, false);
    }
}

void ccTracePolylineTool::restart(bool reset) {
    auto* view = effectiveView();
    if (m_poly3D) {
        if (reset) {
            if (view) view->removeFromOwnDB(m_poly3D);

            if (m_polyTip) {
                m_polyTip->setEnabled(false);
                ecvDisplayTools::RemoveWidgets(
                        WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                          m_polyTip->getViewId()));
            }

            delete m_poly3D;
            m_segmentParams.resize(0);
            m_poly3D = nullptr;
            m_poly3DVertices = nullptr;
        } else if (m_polyTip) {
            m_polyTip->setEnabled(true);
        }
    }

    if (m_pickingHub &&
        !m_pickingHub->addListener(
                this, true /*, true, ecvGenericGLDisplay::POINT_PICKING*/)) {
        CVLog::Error(
                "The picking mechanism is already in use. Close the tool using "
                "it first.");
    }

    if (view) view->redraw(false, false);

    validButton->setEnabled(false);
    saveToolButton->setEnabled(false);
    resetToolButton->setEnabled(false);
    continueToolButton->setEnabled(false);
    m_done = false;
}

void ccTracePolylineTool::exportLine() {
    if (!m_poly3D) {
        return;
    }

    auto* view = effectiveView();
    if (view) view->removeFromOwnDB(m_poly3D);

    unsigned overSampling = static_cast<unsigned>(oversampleSpinBox->value());
    if (overSampling > 1) {
        ccPolyline* poly = polylineOverSampling(overSampling);
        if (poly) {
            delete m_poly3D;
            m_segmentParams.resize(0);
            m_poly3DVertices = nullptr;
            m_poly3D = poly;
        }
    }

    m_poly3D->setTempColor(ecvColor::green);
    m_poly3D->setColor(ecvColor::green);
    if (view) m_poly3D->setDisplay(view);
    if (MainWindow::TheInstance()) {
        MainWindow::TheInstance()->addToDB(m_poly3D);
    } else {
        assert(false);
    }

    m_poly3D = nullptr;
    m_segmentParams.resize(0);
    m_poly3DVertices = nullptr;

    resetLine();
}

void ccTracePolylineTool::apply() {
    exportLine();
    stop(true);
}

void ccTracePolylineTool::cancel() {
    resetLine();

    stop(false);
}

void ccTracePolylineTool::onWidthSizeChanged(int width) {
    if (m_poly3D) {
        m_poly3D->setWidth(width);
    }
    if (m_polyTip) {
        m_polyTip->setWidth(width);
    }
    if (auto* v = effectiveView()) {
        v->redraw(m_poly3D == nullptr, false);
    }
}

void ccTracePolylineTool::resetPoly3D() {
    if (auto* v = effectiveView()) {
        if (m_poly3D) v->removeFromOwnDB(m_poly3D);
    }
}

void ccTracePolylineTool::updatePoly3D() {
    if (auto* v = effectiveView()) v->redraw(false, false);
}

void ccTracePolylineTool::resetTip() {
    ecvDisplayTools::RemoveWidgets(
            WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                              m_polyTip->getViewId()));
}

void ccTracePolylineTool::updateTip() {
    if (m_polyTip && m_polyTip->isEnabled()) {
        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_polyTip->getViewId()));
        ecvDisplayTools::DrawWidgets(
                WIDGETS_PARAMETER(m_polyTip, WIDGETS_TYPE::WIDGET_POLYLINE_2D),
                true);
    } else {
        ecvDisplayTools::RemoveWidgets(
                WIDGETS_PARAMETER(WIDGETS_TYPE::WIDGET_POLYLINE_2D,
                                  m_polyTip->getViewId()));
        ecvDisplayTools::UpdateScreen();
    }
}
