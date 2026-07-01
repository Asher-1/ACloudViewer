// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkComparativeViewWidget.h"

#include <CVLog.h>
#include <Tools/SelectionTools/cvSelectionHighlighter.h>
#include <Tools/SelectionTools/cvSelectionToolController.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/VtkCameraLink.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <VtkRendering/Core/VtkLODHelper.h>
#include <ecvHObject.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewTitleRegistry.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkDataSetMapper.h>
#include <vtkImageData.h>
#include <vtkInteractorObserver.h>
#include <vtkMath.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>
#include <vtkWindowToImageFilter.h>

#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QApplication>
#include <QComboBox>
#include <QCoreApplication>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QResizeEvent>
#include <QSettings>
#include <QShowEvent>
#include <QSlider>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

static constexpr int COMPARATIVE_SPACING = 1;
static constexpr double COMPARATIVE_CAMERA_OCCUPANCY = 2.0 / 3.0;
static constexpr double COMPARATIVE_CAMERA_MARGIN =
        1.0 / COMPARATIVE_CAMERA_OCCUPANCY;
static constexpr int COMPARATIVE_ENTITY_ROLE = Qt::UserRole + 1;

static bool validBounds(const double bounds[6]) {
    return bounds && vtkMath::AreBoundsInitialized(bounds) &&
           bounds[0] <= bounds[1] && bounds[2] <= bounds[3] &&
           bounds[4] <= bounds[5];
}

static void applyCameraMargin(vtkCamera* cam,
                              double margin = COMPARATIVE_CAMERA_MARGIN) {
    if (!cam || margin <= 1.0) return;

    if (cam->GetParallelProjection()) {
        cam->SetParallelScale(cam->GetParallelScale() * margin);
        cam->Modified();
        return;
    }

    double pos[3];
    double focal[3];
    cam->GetPosition(pos);
    cam->GetFocalPoint(focal);
    double viewVector[3] = {pos[0] - focal[0], pos[1] - focal[1],
                            pos[2] - focal[2]};
    if (vtkMath::Norm(viewVector) <= 1.0e-12) return;

    cam->SetPosition(focal[0] + viewVector[0] * margin,
                     focal[1] + viewVector[1] * margin,
                     focal[2] + viewVector[2] * margin);
    cam->Modified();
}

static bool resetRendererCameraFramed(vtkRenderer* ren) {
    if (!ren) return false;

    double bounds[6];
    ren->ComputeVisiblePropBounds(bounds);
    const bool hasBounds = validBounds(bounds);
    if (hasBounds) {
        ren->ResetCamera(bounds);
        applyCameraMargin(ren->GetActiveCamera());
        ren->ResetCameraClippingRange(bounds);
    } else {
        ren->ResetCamera();
        applyCameraMargin(ren->GetActiveCamera());
        ren->ResetCameraClippingRange();
    }
    return hasBounds;
}

static bool sameRepresentationProperties(
        const ecvViewRepresentation::Properties& a,
        const ecvViewRepresentation::Properties& b) {
    return a.opacity == b.opacity && a.pointSize == b.pointSize &&
           a.lineWidth == b.lineWidth && a.renderMode == b.renderMode &&
           a.edgeVisibility == b.edgeVisibility &&
           a.scalarFieldIndex == b.scalarFieldIndex &&
           a.showScalarField == b.showScalarField &&
           a.showColors == b.showColors && a.showNormals == b.showNormals &&
           a.normalScale == b.normalScale;
}

static bool hasDirectChild(ccHObject* parent, ccHObject* child) {
    if (!parent || !child) return false;
    for (unsigned i = 0; i < parent->getChildrenNumber(); ++i) {
        if (parent->getChild(i) == child) return true;
    }
    return false;
}

// Return the VtkVis scene renderer for a view.  This is the renderer that
// holds the actual 3D actors.  Using renderWindow->GetFirstRenderer() is
// unreliable because QVTKWidgetCustom::addActor()/defaultRenderer() may add
// extra renderers to the window, making GetFirstRenderer() return a
// different (possibly empty) renderer.
vtkRenderer* vtkComparativeViewWidget::getSceneRenderer(vtkGLView* view) {
    if (!view) return nullptr;
    if (auto* vis =
                dynamic_cast<Visualization::VtkVis*>(view->getVisualizer3D())) {
        return vis->getCurrentRenderer();
    }
    auto* w = view->getVtkWidget();
    if (!w) return nullptr;
    auto* rw = w->renderWindow();
    return (rw && rw->GetRenderers()) ? rw->GetRenderers()->GetFirstRenderer()
                                      : nullptr;
}

static bool isOverlayRenderer(vtkRenderer* ren, vtkRenderer* sceneRen) {
    if (!ren) return true;
    if (ren == sceneRen) return false;

    if (ren->GetLayer() > 0) return true;

    double* vp = ren->GetViewport();
    const bool isFullViewport =
            (vp[0] == 0 && vp[1] == 0 && vp[2] == 1 && vp[3] == 1);
    if (!isFullViewport) return true;

    vtkActorCollection* actors = ren->GetActors();
    const int nActors = actors ? actors->GetNumberOfItems() : 0;
    vtkPropCollection* props2D = ren->GetViewProps();
    const int nProps2D = props2D ? props2D->GetNumberOfItems() : 0;
    if (nActors == 0 && nProps2D > 0 && nProps2D <= 5) return true;

    if (actors && actors->GetNumberOfItems() > 0) {
        actors->InitTraversal();
        while (vtkActor* actor = actors->GetNextItem()) {
            std::string className = actor->GetClassName();
            if (className.find("Axes") != std::string::npos ||
                className.find("Orientation") != std::string::npos ||
                className.find("Axis") != std::string::npos ||
                className.find("Coordinate") != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

static void refreshOverlayWidgetsOnView(vtkGLView* view) {
    if (!view) return;
    if (auto* vis =
                dynamic_cast<Visualization::VtkVis*>(view->getVisualizer3D())) {
        vis->RefreshOverlayWidgets();
    }
}

// Remove every renderer from the render window EXCEPT the VtkVis scene
// renderer and orientation-marker widget.  Extra renderers (defaultRenderer,
// etc.) can overwrite the scene with their own Erase/background, causing
// comparative sub-views to look "zoomed-in" or blank.
static void stripExtraRenderers(vtkGLView* view) {
    auto* sceneRen = vtkComparativeViewWidget::getSceneRenderer(view);
    if (!sceneRen) return;
    auto* w = view->getVtkWidget();
    if (!w) return;
    auto* rw = w->renderWindow();
    if (!rw || !rw->GetRenderers()) return;
    auto* coll = rw->GetRenderers();
    std::vector<vtkRenderer*> toRemove;
    coll->InitTraversal();
    while (auto* ren = coll->GetNextItem()) {
        if (ren == sceneRen) continue;
        if (isOverlayRenderer(ren, sceneRen)) continue;
        toRemove.push_back(ren);
    }

    for (auto* ren : toRemove) rw->RemoveRenderer(ren);
    refreshOverlayWidgetsOnView(view);
}

static void syncCameraToViewRenderers(vtkGLView* view,
                                      vtkCamera* srcCam,
                                      bool updateClipping,
                                      bool markRenderWindow) {
    if (!view || !srcCam) return;
    auto* w = view->getVtkWidget();
    if (!w) return;
    auto* rw = w->renderWindow();
    if (!rw || !rw->GetRenderers()) return;

    auto* sceneRen = vtkComparativeViewWidget::getSceneRenderer(view);
    auto* coll = rw->GetRenderers();
    coll->InitTraversal();
    while (auto* ren = coll->GetNextItem()) {
        if (isOverlayRenderer(ren, sceneRen)) continue;
        if (!ren->GetActiveCamera()) continue;
        ren->GetActiveCamera()->DeepCopy(srcCam);
        if (updateClipping) {
            ren->ResetCameraClippingRange();
        }
        ren->Modified();
    }

    if (markRenderWindow) {
        if (sceneRen) sceneRen->Modified();
        rw->Modified();
    }
}

static void copyCameraToSceneRenderer(vtkGLView* view,
                                      vtkCamera* srcCam,
                                      bool updateClipping,
                                      bool markRenderWindow) {
    syncCameraToViewRenderers(view, srcCam, updateClipping, markRenderWindow);
}

static void deepCopyCameraToSceneRenderer(vtkGLView* view, vtkCamera* srcCam) {
    copyCameraToSceneRenderer(view, srcCam, true, true);
}

static void renderSubViewImmediate(vtkGLView* view, bool includeSource) {
    if (!view) return;
    if (auto* w = view->getVtkWidget()) {
        if (auto* rw = w->renderWindow()) rw->Modified();
        w->repaint();
    } else if (includeSource) {
        view->redraw(false, true);
    }
}

static bool safeRenderWindow(vtkGLView* view, bool immediate = false) {
    if (!view) return false;
    auto* w = view->getVtkWidget();
    if (!w || !w->isVisible() || w->width() < 2 || w->height() < 2)
        return false;
    if (immediate) {
        renderSubViewImmediate(view, false);
    } else {
        w->update();
    }
    return true;
}

static void safeRedraw(vtkGLView* view) {
    if (!view) return;
    auto* w = view->getVtkWidget();
    if (!w || !w->isVisible() || w->width() < 2 || w->height() < 2) return;
    view->redraw(false, true);
}

static void syncSubViewBackgroundFromGlobal(vtkGLView* view) {
    if (!view) return;
    // Follow global Display Settings — do not lock per-window overrides.
    view->clearDisplayParametersOverride();
    view->redraw(false, true);
}

static ecvGenericGLDisplay::INTERACTION_FLAGS stripClickableItems(
        ecvGenericGLDisplay::INTERACTION_FLAGS flags) {
    return static_cast<ecvGenericGLDisplay::INTERACTION_FLAGS>(
            flags & ~ecvGenericGLDisplay::INTERACT_CLICKABLE_ITEMS);
}

// Comparative sub-views use small viewports where bubble-view / hot-zone
// overlay text does not render correctly — disable entirely (ParaView-style).
static void disableBubbleViewForSubView(vtkGLView* view) {
    if (!view) return;
    view->disableOverlayEntities();
    view->context().bubbleViewModeEnabled = false;
    view->setClickableItemsVisible(false);
    view->setInteractionMode(stripClickableItems(view->getInteractionMode()));
    if (QWidget* w = view->asWidget()) {
        w->setMouseTracking(view->getInteractionMode() &
                            ecvGenericGLDisplay::INTERACT_SIG_MOUSE_MOVED);
    }
}

static QString comparativeViewTypeKey(
        vtkComparativeViewWidget::ComparativeType type) {
    switch (type) {
        case vtkComparativeViewWidget::RENDER:
            return QStringLiteral("Render View (Comparative)");
        case vtkComparativeViewWidget::LINE_CHART:
            return QStringLiteral("Line Chart View (Comparative)");
        case vtkComparativeViewWidget::BAR_CHART:
            return QStringLiteral("Bar Chart View (Comparative)");
    }
    return QStringLiteral("Comparative View");
}

vtkComparativeViewWidget::vtkComparativeViewWidget(ComparativeType type,
                                                   QWidget* parent)
    : QWidget(parent), m_type(type) {
    m_viewTypeKey = comparativeViewTypeKey(type);
    m_title = ecvViewTitleRegistry::instance().allocate(m_viewTypeKey);

    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    buildToolbar();
    m_toolbar->setVisible(true);
    m_toolbar->setMinimumHeight(28);
    m_toolbar->setMaximumHeight(32);
    mainLayout->addWidget(m_toolbar);

    auto* gridContainer = new QWidget(this);
    m_gridLayout = new QGridLayout(gridContainer);
    m_gridLayout->setContentsMargins(0, 0, 0, 0);
    m_gridLayout->setSpacing(COMPARATIVE_SPACING);
    mainLayout->addWidget(gridContainer, 1);
    setMinimumSize(100, 100);
    setContextMenuPolicy(Qt::NoContextMenu);
}

vtkComparativeViewWidget::~vtkComparativeViewWidget() {
    shutdown();
    if (!m_viewTypeKey.isEmpty() && !m_title.isEmpty()) {
        ecvViewTitleRegistry::instance().release(m_viewTypeKey, m_title);
    }
}

void vtkComparativeViewWidget::shutdown() {
    if (m_shutdownDone) return;
    m_shutdownDone = true;
    m_closing = true;

    if (m_subViewRefreshTimer) {
        m_subViewRefreshTimer->stop();
        m_subViewRefreshTimer->disconnect();
        delete m_subViewRefreshTimer;
        m_subViewRefreshTimer = nullptr;
    }
    if (m_cameraSyncRenderTimer) {
        m_cameraSyncRenderTimer->stop();
        m_cameraSyncRenderTimer->disconnect();
        delete m_cameraSyncRenderTimer;
        m_cameraSyncRenderTimer = nullptr;
    }

    auto& vm = ecvViewManager::instance();
    for (auto* sv : m_subViews) {
        if (!sv) continue;
        if (vm.getActiveView() == sv) {
            vm.setActiveView(nullptr);
        }
    }

    disconnectExternalHighlighter();
    removeCameraLink();

    disconnect(&ecvRepresentationManager::instance(), nullptr, this, nullptr);
    disconnect(&ecvViewManager::instance(), nullptr, this, nullptr);
    if (m_sourceView) {
        disconnect(m_sourceView, nullptr, this, nullptr);
        m_sourceView = nullptr;
    }
    disconnect(this);

    for (auto* w : m_subWidgets) {
        if (w && m_gridLayout) {
            m_gridLayout->removeWidget(w);
        }
    }

    for (auto* sv : m_subViews) {
        if (!sv) continue;
        auto* vis = dynamic_cast<Visualization::VtkVis*>(sv->getVisualizer3D());
        if (vis) {
            cvSelectionHighlighter::clearAllSelectionOverlays(vis);
        }
    }

    const QList<vtkGLView*> views = m_subViews;
    for (auto* sv : views) {
        if (!sv) continue;
        QWidget* w = sv->asWidget();

        if (m_subViewShutdownHook) {
            m_subViewShutdownHook(sv);
        }
        if (w) {
            w->removeEventFilter(this);
            w->disconnect();
            w->hide();
        }

        sv->blockSignals(true);
        sv->setSceneDB(nullptr);
        if (auto* vis = sv->getVisualizer3D()) {
            Visualization::VtkCameraLink::instance().removeView(vis);
        }
        ecvViewManager::instance().unregisterView(sv);
        sv->disconnect();
    }

    for (auto* sv : views) {
        if (!sv) continue;
        QWidget* w = sv->asWidget();
        if (w) w->setUpdatesEnabled(false);
        if (auto* vtkW = dynamic_cast<QVTKWidgetCustom*>(w)) {
            auto* rw = vtkW->renderWindow();
            if (rw) {
                auto* iren = rw->GetInteractor();
                if (iren) {
                    iren->Disable();
                    iren->TerminateApp();
                }
                rw->Finalize();
            }
        }
        sv->setParent(nullptr);
        delete sv;
    }
    m_subViews.clear();
    m_activeSubView = nullptr;
    m_subWidgets.clear();
    m_pendingFirstResize.clear();
    m_showingEntitiesByView.clear();
    m_ignoredRenderEndWindows.clear();
}

QString vtkComparativeViewWidget::title() const { return m_title; }

void vtkComparativeViewWidget::setRenderViewFactory(RenderViewFactory factory) {
    m_renderFactory = factory;
    if (m_type == RENDER && m_subWidgets.isEmpty()) {
        setupGrid();
    }
}

void vtkComparativeViewWidget::setSubViewInitCallback(SubViewInitCallback cb) {
    m_subViewInitCb = std::move(cb);
}

void vtkComparativeViewWidget::setSubViewShutdownHook(
        SubViewShutdownHook hook) {
    m_subViewShutdownHook = std::move(hook);
}

void vtkComparativeViewWidget::setupGrid() {
    if (m_type == RENDER) {
        createRenderSubViews();
    } else {
        createChartSubViews();
    }

    // 初始化Entity下拉框（ParaView模式：用户手动选择显示的实体）
    QTimer::singleShot(200, this, [this]() {
        if (!m_closing) {
            refreshEntityCombo();
        }
    });
}

void vtkComparativeViewWidget::createRenderSubViews() {
    if (!m_renderFactory) {
        CVLog::Warning("[ComparativeView] No render factory set");
        return;
    }

    vtkGLView* firstView = nullptr;
    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            auto* view = m_renderFactory();
            if (!view) {
                CVLog::Warning(
                        "[ComparativeView] Factory returned null for "
                        "cell (%d,%d)",
                        r, c);
                continue;
            }

            view->setParent(this);
            view->setSceneDB(nullptr);

            view->setInteractionMode(
                    ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);

            QWidget* viewWidget = view->asWidget();
            if (!viewWidget) {
                CVLog::Warning(
                        "[ComparativeView] View widget null for "
                        "cell (%d,%d)",
                        r, c);
                continue;
            }
            viewWidget->setMinimumSize(50, 50);
            viewWidget->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Expanding);
            viewWidget->setContextMenuPolicy(Qt::NoContextMenu);
            viewWidget->setFocusPolicy(Qt::StrongFocus);
            viewWidget->setProperty("_compViewDiag", true);
            if (auto* vtkWidget = view->getVtkWidget()) {
                vtkWidget->setDirectCameraWheelZoom(true);
            }
            m_gridLayout->addWidget(viewWidget, r, c);
            m_gridLayout->setRowStretch(r, 1);
            m_gridLayout->setColumnStretch(c, 1);

            m_subWidgets.append(viewWidget);
            m_subViews.append(view);
            connect(view, &vtkGLView::mouseWheelRotated, this,
                    [this, view](float) {
                        if (m_closing || !m_cameraLinkEnabled) return;
                        const int idx = m_subViews.indexOf(view);
                        if (idx >= 0) onSubViewInteraction(idx, true);
                    });
            connect(view, &vtkGLView::cameraParamChanged, this, [this, view]() {
                if (m_closing || !m_cameraLinkEnabled || m_syncingCameras)
                    return;
                const int idx = m_subViews.indexOf(view);
                if (idx < 0) return;
                onSubViewInteraction(idx, !m_interacting);
                if (m_interacting) {
                    restartInteractionEndTimer();
                }
            });
            if (!firstView) {
                firstView = view;
                m_activeSubView = view;

                // ParaView模式: Comparative视图初始为空
                // 用户通过Showing下拉框手动选择要显示的实体
                if (m_subViewInitCb) {
                    m_subViewInitCb(view);
                }
            } else {
                auto* firstRen =
                        vtkComparativeViewWidget::getSceneRenderer(firstView);
                if (firstRen && firstRen->GetActiveCamera()) {
                    deepCopyCameraToSceneRenderer(view,
                                                  firstRen->GetActiveCamera());
                }
            }

            if (auto* vis = view->getVisualizer3D()) {
                Visualization::VtkCameraLink::instance().removeView(vis);
            }

            disableBubbleViewForSubView(view);
            view->disableContext2DOverlay();
            view->setSceneDB(nullptr);
            syncSubViewBackgroundFromGlobal(view);
            stripExtraRenderers(view);

            if (firstView && firstView != view) {
                syncViewPropertiesFromSource(firstView, view);
            }

            if (!firstView || firstView == view) {
                // already initialized above
            } else if (m_subViewInitCb) {
                m_subViewInitCb(view);
            }
            view->setSceneDB(nullptr);

            m_pendingFirstResize.insert(viewWidget);
            viewWidget->installEventFilter(this);
            emit subViewCreated(viewWidget);
        }
    }

    // ParaView模式: 不自动复制表示，保持4个空视窗
    // 用户通过Showing下拉框选择要显示的实体
    if (m_sourceView) {
        for (auto* view : m_subViews) {
            if (!view) continue;
            view->setEntityBindingSource(m_sourceView);
        }
    }

    auto onRepChanged = [this](ecvViewRepresentation* rep) {
        if (m_closing || !rep || m_subViews.isEmpty()) return;
        if (m_applyingShowingSelection) return;
        ecvGenericGLDisplay* repView =
                const_cast<ecvGenericGLDisplay*>(rep->getView());
        bool isOurView = false;
        for (auto* sv : m_subViews) {
            if (sv == repView) {
                isOurView = true;
                break;
            }
        }
        if (repView == m_sourceView) isOurView = true;
        if (!isOurView) return;
        if (repView == m_sourceView &&
            !m_selectedEntities.contains(rep->getEntity())) {
            return;
        }
        scheduleSubViewRefresh(true);
    };

    connect(&ecvRepresentationManager::instance(),
            &ecvRepresentationManager::representationAdded, this, onRepChanged);
    connect(&ecvRepresentationManager::instance(),
            &ecvRepresentationManager::representationChanged, this,
            onRepChanged);

    if (m_cameraLinkEnabled && m_subViews.size() > 1) {
        syncCamerasFromFirst();
        installCameraLink();
    }

    m_needsCameraReset = false;
    if (!m_closing) {
        // ParaView模式: 初始为空视窗，不执行同步
        // 用户通过Showing下拉框选择实体后会自动触发同步和刷新
        QTimer::singleShot(100, this, [this]() {
            if (m_closing) return;
            // 只初始化相机链接（确保4个视窗的初始状态一致）
            if (m_cameraLinkEnabled && m_subViews.size() > 1) {
                syncCamerasFromFirst();
            }
            scheduleSubViewRefresh(false);
        });
    }

    connect(&ecvViewManager::instance(), &ecvViewManager::pointIndicesSelected,
            this, [this](ccHObject*, const QSet<unsigned>&) {
                scheduleSubViewRefresh(true);
            });

    if (m_sourceView) {
        connect(m_sourceView, &vtkGLView::interactionModeChanged, this,
                &vtkComparativeViewWidget::syncInteractionModeToSubViews);
        connect(m_sourceView, &vtkGLView::pickingModeChanged, this,
                &vtkComparativeViewWidget::syncPickingModeToSubViews);
        connect(
                m_sourceView, &vtkGLView::cameraParamChanged, this,
                [this]() { syncCameraFromSourceView(); }, Qt::UniqueConnection);
        connect(m_sourceView, &vtkGLView::interactionModeChanged, this,
                [this]() {
                    if (!m_closing && m_subViews.size() >= 2) {
                        for (auto* v : m_subViews) {
                            if (v && v != m_sourceView) {
                                v->setInteractionMode(stripClickableItems(
                                        m_sourceView->getInteractionMode()));
                                disableBubbleViewForSubView(v);
                            }
                        }
                    }
                });
        connect(m_sourceView, &vtkGLView::pickingModeChanged, this, [this]() {
            if (!m_closing && m_subViews.size() >= 2) {
                for (auto* v : m_subViews) {
                    if (v && v != m_sourceView) {
                        v->setPickingMode(m_sourceView->getPickingMode());
                    }
                }
            }
        });
        connect(m_sourceView, &vtkGLView::cameraParamChanged, this, [this]() {
            if (!m_closing && m_subViews.size() >= 2) {
                syncCameraFromSourceView();
            }
        });

        // 监听DBTree变化以刷新Entity下拉框（不自动复制表示）
        connect(
                &ecvRepresentationManager::instance(),
                &ecvRepresentationManager::representationAdded, this,
                [this](ecvViewRepresentation* /*rep*/) {
                    if (m_closing || m_subViews.isEmpty()) return;
                    // 延迟刷新entity combo列表
                    QTimer::singleShot(100, this, [this]() {
                        if (m_closing) return;
                        refreshEntityCombo();
                    });
                },
                Qt::UniqueConnection);
    }
}

void vtkComparativeViewWidget::setSourceView(vtkGLView* src) {
    m_sourceView = src;
}

void vtkComparativeViewWidget::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    if (!m_firstShowDone) {
        m_firstShowDone = true;
        if (m_subViews.size() >= 2 && !m_cameraObservers.empty()) {
            syncCamerasFromFirst();
        } else if (m_cameraLinkEnabled) {
            installCameraLink();
        }
        if (m_needsCameraReset) {
            scheduleSubViewRefresh(true);
        }
        // Deferred overlay widget initialization: windows aren't fully
        // realized in showEvent. Wait for the event loop to process geometry
        // changes so that GetSize() returns the correct values for DPI sizing.
        QTimer::singleShot(50, this, [this]() {
            if (m_closing) return;
            // Initialize overlay widgets from global settings so that
            // comparative sub-views match the normal view's widget state.
            QSettings settings;
            const bool cowVisible =
                    settings.value("CameraOrientationWidget/Visible", false)
                            .toBool();
            const bool omVisible =
                    settings.value("OrientationMarker/Visible", true).toBool();
            if (cowVisible) {
                toggleCameraOrientationWidgetOnAllSubViews(true);
            }
            if (omVisible) {
                toggleOrientationMarkerOnAllSubViews(true);
            }
            refreshOverlayWidgetsOnAllSubViews();
        });
    }
}

void vtkComparativeViewWidget::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
}

bool vtkComparativeViewWidget::eventFilter(QObject* obj, QEvent* event) {
    if (m_entityCombo && m_entityCombo->view() &&
        obj == m_entityCombo->view()->viewport() &&
        (event->type() == QEvent::MouseButtonPress ||
         event->type() == QEvent::MouseButtonRelease)) {
        auto* me = static_cast<QMouseEvent*>(event);
        const QModelIndex index = m_entityCombo->view()->indexAt(me->pos());
        if (!index.isValid()) return true;
        if (event->type() == QEvent::MouseButtonRelease) {
            if (auto* model = qobject_cast<QStandardItemModel*>(
                        m_entityCombo->model())) {
                if (auto* item = model->itemFromIndex(index)) {
                    const Qt::CheckState next =
                            item->checkState() == Qt::Checked ? Qt::Unchecked
                                                              : Qt::Checked;
                    item->setCheckState(next);
                }
            }
        }
        return true;
    }

    auto* wheelWidget = qobject_cast<QWidget*>(obj);
    const bool isWheelControl = qobject_cast<QComboBox*>(obj) ||
                                qobject_cast<QSlider*>(obj) ||
                                qobject_cast<QAbstractSpinBox*>(obj);

    if (event->type() == QEvent::Show && isWheelControl && wheelWidget &&
        wheelWidget->focusPolicy() == Qt::WheelFocus) {
        wheelWidget->setFocusPolicy(Qt::StrongFocus);
    }

    if (event->type() == QEvent::Wheel && isWheelControl && wheelWidget &&
        !wheelWidget->hasFocus()) {
        return true;
    }

    // 只阻止ContextMenu事件
    if (event->type() == QEvent::ContextMenu) return true;

    // 明确不拦截 Wheel 事件，让 VTK 交互器正常处理滚轮操作
    if (event->type() == QEvent::Wheel) {
        if (activateSubViewForWidget(qobject_cast<QWidget*>(obj),
                                     Qt::OtherFocusReason)) {
            emit clicked();
            if (m_activeSubView && !m_closing) {
                emit requestToolRebind(m_activeSubView);
            }
        }
        return false;  // 返回 false 表示不拦截，传递给子 widget
    }

    // 处理鼠标按下：更新活跃子视窗并发出信号
    if (event->type() == QEvent::MouseButtonPress) {
        if (activateSubViewForWidget(qobject_cast<QWidget*>(obj),
                                     Qt::MouseFocusReason) &&
            m_activeSubView && m_cameraLinkEnabled && !m_closing) {
            const int idx = m_subViews.indexOf(m_activeSubView);
            if (idx >= 0) {
                m_interacting = true;
                m_interactionSourceIdx = idx;
                m_cameraSyncSourceIdx = idx;
                setInteractiveLOD(true);
            }
        }
        emit clicked();

        QTimer::singleShot(0, this, [this]() {
            if (m_activeSubView && !m_closing) {
                emit requestToolRebind(m_activeSubView);
            }
        });
    }

    if (event->type() == QEvent::MouseButtonRelease) {
        vtkGLView* releaseView = m_activeSubView;
        const int idx = m_subViews.indexOf(releaseView);
        if (idx >= 0 && m_cameraLinkEnabled && !m_closing) {
            QTimer::singleShot(0, this, [this, idx]() {
                if (m_closing || idx < 0 || idx >= m_subViews.size()) return;
                m_interacting = false;
                m_interactionSourceIdx = -1;
                m_cameraSyncSourceIdx = idx;
                stopInteractionTimer();
                setInteractiveLOD(false);
                onSubViewInteraction(idx, true);
            });
        }
    }

    // 处理 Resize 事件
    if (event->type() == QEvent::Resize) {
        auto* w = qobject_cast<QWidget*>(obj);
        if (w && m_pendingFirstResize.contains(w)) {
            auto* re = static_cast<QResizeEvent*>(event);
            if (re->size().width() > 10 && re->size().height() > 10) {
                m_pendingFirstResize.remove(w);
                QTimer::singleShot(50, this, [this, w]() {
                    if (m_closing || !isVisible()) return;
                    for (auto* view : m_subViews) {
                        if (!view || view->asWidget() != w) continue;
                        syncSubViewBackgroundFromGlobal(view);
                        refreshOverlayWidgetsOnView(view);
                        break;
                    }
                    scheduleSubViewRefresh(true);
                });
            }
        }
    }

    // 对于所有其他事件，调用父类默认实现，确保完全透明
    return QWidget::eventFilter(obj, event);
}

bool vtkComparativeViewWidget::activateSubViewForWidget(
        QWidget* widget, Qt::FocusReason reason) {
    if (!widget) return false;

    for (auto* view : m_subViews) {
        if (!view) continue;
        QWidget* viewWidget = view->asWidget();
        if (viewWidget &&
            (viewWidget == widget || viewWidget->isAncestorOf(widget))) {
            m_activeSubView = view;
            viewWidget->setFocus(reason);
            ecvViewManager::instance().setActiveView(view);

            // Re-validate the camera orientation widget for this sub-view
            // so it is ready for interaction (interactor, ProcessEvents, size).
            auto* glView = dynamic_cast<vtkGLView*>(view);
            auto* vis = glView ? dynamic_cast<Visualization::VtkVis*>(
                                         glView->getVisualizer3D())
                               : nullptr;
            if (vis) {
                vis->EnsureCameraOrientationWidgetInteractive();
            }

            return true;
        }
    }
    return false;
}

void vtkComparativeViewWidget::forceRenderAllSubViews() {
    if (m_closing) return;
    for (auto* view : m_subViews) {
        if (!view) continue;
        view->redraw(false, true);
    }
}

void vtkComparativeViewWidget::toggleCameraOrientationWidgetOnAllSubViews(
        bool state) {
    if (m_closing) return;
    // Suppress renderEndCallback during the batch toggle so that
    // refreshOverlayWidgetsOnView doesn't interfere with in-progress toggles.
    m_syncingCameras = true;
    for (auto* sv : m_subViews) {
        if (sv) sv->toggleCameraOrientationWidget(state);
    }
    m_syncingCameras = false;
    forceRenderAllSubViews();
}

void vtkComparativeViewWidget::toggleOrientationMarkerOnAllSubViews(
        bool state) {
    if (m_closing) return;
    m_syncingCameras = true;
    for (auto* sv : m_subViews) {
        if (sv) sv->toggleOrientationMarker(state);
    }
    m_syncingCameras = false;
    forceRenderAllSubViews();
}

void vtkComparativeViewWidget::zoomToData() {
    if (m_closing || m_subViews.isEmpty()) return;

    vtkGLView* sourceView =
            m_activeSubView ? m_activeSubView : m_subViews.first();
    int sourceIdx = m_subViews.indexOf(sourceView);
    if (sourceIdx < 0) {
        sourceIdx = 0;
        sourceView = m_subViews.first();
    }

    const bool restoreCameraLink = m_cameraLinkEnabled;
    removeCameraLink();
    m_cameraLinkEnabled = false;

    auto* sourceRen = vtkComparativeViewWidget::getSceneRenderer(sourceView);
    if (!sourceRen || !sourceRen->GetActiveCamera()) {
        m_cameraLinkEnabled = restoreCameraLink;
        if (restoreCameraLink) installCameraLink();
        return;
    }

    const bool hasBounds = resetRendererCameraFramed(sourceRen);
    double bounds[6];
    sourceRen->ComputeVisiblePropBounds(bounds);
    if (hasBounds && validBounds(bounds)) {
        const CCVector3d pivot((bounds[0] + bounds[1]) * 0.5,
                               (bounds[2] + bounds[3]) * 0.5,
                               (bounds[4] + bounds[5]) * 0.5);
        sourceView->setPivotPoint(pivot, false, false);
    }

    auto* srcCam = sourceRen->GetActiveCamera();
    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* view = m_subViews[i];
        if (!view) continue;
        deepCopyCameraToSceneRenderer(view, srcCam);
        if (auto* w = view->getVtkWidget()) {
            if (auto* rw = w->renderWindow()) rw->Modified();
            w->update();
        }
    }

    syncPivotFromView(sourceIdx);
    m_needsCameraReset = false;
    saveBaselineCamera();
    forceRenderAllSubViews();

    QTimer::singleShot(200, this, [this, restoreCameraLink]() {
        if (m_closing) return;
        m_cameraLinkEnabled = restoreCameraLink;
        if (restoreCameraLink) installCameraLink();
    });
}

void vtkComparativeViewWidget::scheduleSubViewRefresh(bool forceSceneDirty) {
    if (m_closing) return;
    if (forceSceneDirty) m_subViewRefreshForceDirty = true;
    if (!m_subViewRefreshTimer) {
        m_subViewRefreshTimer = new QTimer(this);
        m_subViewRefreshTimer->setSingleShot(true);
        connect(m_subViewRefreshTimer, &QTimer::timeout, this,
                &vtkComparativeViewWidget::performSubViewRefresh);
    }
    if (!m_subViewRefreshTimer->isActive()) m_subViewRefreshTimer->start(32);
}

void vtkComparativeViewWidget::performSubViewRefresh() {
    if (m_closing || m_subViews.isEmpty()) {
        if (m_cameraSyncSourceIdx < 0) m_cameraSyncSourceIdx = 0;
        return;
    }
    if (!isVisible()) {
        if (m_cameraSyncSourceIdx < 0) m_cameraSyncSourceIdx = 0;
        m_ignoredRenderEndWindows.clear();
        return;
    }

    const bool forceSceneDirty = m_subViewRefreshForceDirty;
    m_subViewRefreshForceDirty = false;

    for (auto* view : m_subViews) {
        if (!view) continue;
        QWidget* w = view->asWidget();
        if (w && isVisible()) w->show();
        view->setAutoPickPivotAtCenter(false);
    }

    const bool needsCameraReset = m_needsCameraReset;
    if (forceSceneDirty) {
        syncRepresentationsFromFirst();
    }

    // Suppress EndEvent camera sync during scene refresh.  Showing changes
    // create/update representations in each pane, and each redraw emits its
    // own EndEvent.  If those events are allowed to sync cameras, the last
    // pane that redraws can overwrite the intended linked camera state.
    const int prevSyncSource =
            m_cameraSyncSourceIdx >= 0 ? m_cameraSyncSourceIdx : 0;
    if (forceSceneDirty) m_cameraSyncSourceIdx = -1;

    for (auto* view : m_subViews) {
        if (!view) continue;
        if (forceSceneDirty) {
            ccHObject* root = view->getSceneDB();
            if (root) root->setRedrawFlagRecursive(true);
        }
        stripExtraRenderers(view);
        safeRedraw(view);
    }

    // Now that actors exist, reset the camera on the first view so
    // ResetCamera computes correct bounds, then sync to all others.
    bool cameraResetDone = false;
    if (forceSceneDirty && needsCameraReset && !m_subViews.isEmpty()) {
        auto* first = m_subViews.first();
        auto* ren = vtkComparativeViewWidget::getSceneRenderer(first);
        auto* vtkW = first ? first->getVtkWidget() : nullptr;
        if (ren && vtkW && vtkW->isVisible() && vtkW->width() >= 2 &&
            vtkW->height() >= 2) {
            int nRens = vtkW->renderWindow() ? vtkW->renderWindow()
                                                       ->GetRenderers()
                                                       ->GetNumberOfItems()
                                             : 0;
            int nActors =
                    ren->GetActors() ? ren->GetActors()->GetNumberOfItems() : 0;
            resetRendererCameraFramed(ren);
            double bounds[6];
            ren->ComputeVisiblePropBounds(bounds);
            if (validBounds(bounds)) {
                double cx = (bounds[0] + bounds[1]) * 0.5;
                double cy = (bounds[2] + bounds[3]) * 0.5;
                double cz = (bounds[4] + bounds[5]) * 0.5;
                CCVector3d pivot(cx, cy, cz);
                first->setPivotPoint(pivot, false, false);
                syncPivotFromFirst();
            }
            cameraResetDone = true;
            // Propagate the reset camera to ALL renderers in this view's
            // render window so that stray renderers (defaultRenderer,
            // orientation widget renderer, etc.) don't paint with stale
            // cameras.
            deepCopyCameraToSceneRenderer(first, ren->GetActiveCamera());
        }
    }

    if (cameraResetDone) {
        m_needsCameraReset = false;
    } else if (needsCameraReset && forceSceneDirty) {
        m_needsCameraReset = true;
        QTimer::singleShot(50, this, [this]() {
            if (!m_closing && m_needsCameraReset) scheduleSubViewRefresh(true);
        });
    }

    if (forceSceneDirty && m_cameraLinkEnabled && m_subViews.size() > 1) {
        syncCamerasFromFirst();
    }

    if (cameraResetDone) {
        saveBaselineCamera();
    }

    // Re-enable EndEvent camera sync now that cameras are correct.  A scene
    // rebuild/reset always uses the first pane as the authoritative source,
    // mirroring ParaView's root comparative view.
    m_cameraSyncSourceIdx =
            (forceSceneDirty && m_cameraLinkEnabled && m_subViews.size() > 1)
                    ? 0
                    : prevSyncSource;

    if (forceSceneDirty && (cameraResetDone || needsCameraReset)) {
        for (auto* view : m_subViews) {
            if (!view || !view->getVtkWidget()) continue;
            stripExtraRenderers(view);
            auto* rw = view->getVtkWidget()->renderWindow();
            if (rw) rw->Modified();
            view->getVtkWidget()->update();
        }
    }

    refreshOverlayWidgetsOnAllSubViews();
}

void vtkComparativeViewWidget::syncInteractionModeToSubViews() {
    if (!m_sourceView || m_subViews.isEmpty()) return;
    auto flags = stripClickableItems(m_sourceView->getInteractionMode());
    for (auto* view : m_subViews) {
        if (view && view != m_sourceView) {
            view->setInteractionMode(flags);
            disableBubbleViewForSubView(view);
        }
    }
}

void vtkComparativeViewWidget::syncPickingModeToSubViews() {
    if (!m_sourceView || m_subViews.isEmpty()) return;
    auto mode = m_sourceView->getPickingMode();
    for (auto* view : m_subViews) {
        if (view && view != m_sourceView) {
            view->setPickingMode(mode);
        }
    }
}

void vtkComparativeViewWidget::syncCameraFromSourceView() {
    if (!m_sourceView || m_closing || !isVisible() || m_syncingCameras ||
        !m_cameraLinkEnabled)
        return;
    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(m_sourceView);
    if (!srcRen || !srcRen->GetActiveCamera()) return;
    auto* srcCam = srcRen->GetActiveCamera();

    const int sourceIdx = m_subViews.indexOf(m_sourceView);
    m_cameraSyncSourceIdx = sourceIdx >= 0 ? sourceIdx : m_subViews.size();

    m_syncingCameras = true;
    for (auto* view : m_subViews) {
        deepCopyCameraToSceneRenderer(view, srcCam);
    }
    syncPivotFromFirst();
    m_syncingCameras = false;
    scheduleCameraSyncRender();
}

void vtkComparativeViewWidget::copyActorsAcrossSubViews() {
    scheduleSubViewRefresh(false);
}

void vtkComparativeViewWidget::syncRepresentationsFromFirst() {
    if (m_type == RENDER) {
        applySelectedEntitiesToRenderViews(false);
    }
}

void vtkComparativeViewWidget::syncCamerasFromFirst() {
    if (m_closing || m_subViews.size() < 2) return;
    vtkGLView* first = m_subViews.first();
    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(first);
    if (!srcRen) return;
    auto* srcCam = srcRen->GetActiveCamera();
    if (!srcCam) return;

    const bool wasSyncing = m_syncingCameras;
    m_syncingCameras = true;
    for (int i = 1; i < m_subViews.size(); ++i) {
        deepCopyCameraToSceneRenderer(m_subViews[i], srcCam);
    }
    m_cameraSyncSourceIdx = 0;
    syncPivotFromFirst();
    m_syncingCameras = wasSyncing;
}

void vtkComparativeViewWidget::syncPivotFromView(int srcIdx) {
    if (m_closing || m_subViews.size() < 2) return;
    if (srcIdx < 0 || srcIdx >= m_subViews.size()) return;

    vtkGLView* srcView = m_subViews[srcIdx];
    if (!srcView || !srcView->viewContext()) return;

    const CCVector3d& pivot =
            srcView->viewContext()->viewportParams.getPivotPoint();
    auto* srcVis =
            dynamic_cast<Visualization::VtkVis*>(srcView->getVisualizer3D());
    const int vtkInteractionMode = srcVis ? srcVis->getInteractionMode() : -1;
    const double rotationFactor = srcVis ? srcVis->getRotationFactor() : 1.0;
    const auto interactionFlags =
            stripClickableItems(srcView->getInteractionMode());

    for (int i = 0; i < m_subViews.size(); ++i) {
        if (i == srcIdx) continue;
        auto* dstView = m_subViews[i];
        if (!dstView || !dstView->viewContext()) continue;
        dstView->viewContext()->viewportParams.setPivotPoint(pivot, true);
        dstView->viewContext()->autoPivotCandidate = pivot;
        if (dstView->getInteractionMode() != interactionFlags) {
            dstView->setInteractionMode(interactionFlags);
        }
        if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                    dstView->getVisualizer3D())) {
            vis->setCenterOfRotation(pivot.x, pivot.y, pivot.z);
            vis->setRotationFactor(rotationFactor);
            if (vtkInteractionMode >= 0 &&
                vis->getInteractionMode() != vtkInteractionMode) {
                vis->setInteractionMode(vtkInteractionMode);
            }
        }
    }
}

void vtkComparativeViewWidget::syncPivotFromFirst() { syncPivotFromView(0); }

void vtkComparativeViewWidget::installCameraLink() {
    removeCameraLink();
    if (!m_cameraLinkEnabled || m_subViews.size() < 2) return;

    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* view = m_subViews[i];
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw) continue;

        // --- Interaction events (Start/Interaction/End/Wheel) ---
        auto* interactor = rw->GetInteractor();
        if (interactor) {
            auto* style = interactor->GetInteractorStyle();
            if (style) {
                auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
                cb->SetClientData(this);
                cb->SetCallback(&vtkComparativeViewWidget::interactionCallback);

                unsigned long t;
                t = style->AddObserver(vtkCommand::StartInteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(vtkCommand::InteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(vtkCommand::EndInteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(vtkCommand::MouseWheelForwardEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(vtkCommand::MouseWheelBackwardEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
            }
        }

        // --- EndEvent on RenderWindow (ParaView vtkSMCameraLink pattern) ---
        // After each render on the source view, copy camera to all others.
        // This guarantees synchronization regardless of what changed the camera
        // (interaction, ResetCamera, programmatic change, etc.).
        {
            auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
            cb->SetClientData(this);
            cb->SetCallback(&vtkComparativeViewWidget::renderEndCallback);
            unsigned long t = rw->AddObserver(vtkCommand::EndEvent, cb);
            m_cameraObservers.push_back({rw, t, cb});
        }

        // NOTE: Removed Camera ModifiedEvent observer. VTK fires Modified()
        // on camera extremely frequently (every property set, clipping range
        // reset, etc.) which caused re-entrant rendering and deadlocks.
        // The Qt-level cameraParamChanged signal connection already provides
        // reliable camera sync without the re-entrancy risk.
    }
}

void vtkComparativeViewWidget::removeCameraLink() {
    for (auto& obs : m_cameraObservers) {
        if (obs.callback) {
            obs.callback->SetClientData(nullptr);
        }
        if (obs.observed && obs.tag) {
            obs.observed->RemoveObserver(obs.tag);
        }
        obs.observed = nullptr;
        obs.tag = 0;
        obs.callback = nullptr;
    }
    m_cameraObservers.clear();
    if (m_interactionEndTimer) m_interactionEndTimer->stop();
    m_interacting = false;
    m_interactionSourceIdx = -1;
}

void vtkComparativeViewWidget::cameraModifiedCallback(vtkObject* caller,
                                                      unsigned long /*eid*/,
                                                      void* clientData,
                                                      void*) {
    auto* self = static_cast<vtkComparativeViewWidget*>(clientData);
    if (!self || self->m_shutdownDone || self->m_closing ||
        self->m_syncingCameras || !self->m_cameraLinkEnabled)
        return;

    auto* cam = vtkCamera::SafeDownCast(caller);
    if (!cam) return;

    int srcIdx = -1;
    for (int i = 0; i < self->m_subViews.size(); ++i) {
        auto* ren = getSceneRenderer(self->m_subViews[i]);
        if (ren && ren->GetActiveCamera() == cam) {
            srcIdx = i;
            break;
        }
    }
    if (srcIdx < 0) return;
    if (self->m_cameraSyncSourceIdx < 0) return;

    self->onSubViewInteraction(srcIdx, true);
}

void vtkComparativeViewWidget::renderEndCallback(vtkObject* caller,
                                                 unsigned long /*eid*/,
                                                 void* clientData,
                                                 void*) {
    auto* self = static_cast<vtkComparativeViewWidget*>(clientData);
    if (!self || self->m_shutdownDone || self->m_closing ||
        self->m_syncingCameras || !self->m_cameraLinkEnabled)
        return;

    // ParaView disables synchronized interactive renders for comparative
    // views.  Keep drag interaction responsive and sync the other panes at
    // EndInteraction; discrete wheel/programmatic changes still sync below.
    if (self->m_interacting) return;

    auto* rw = vtkRenderWindow::SafeDownCast(caller);
    if (!rw) return;

    int srcIdx = -1;
    for (int i = 0; i < self->m_subViews.size(); ++i) {
        auto* v = self->m_subViews[i];
        if (!v || !v->getVtkWidget()) continue;
        if (v->getVtkWidget()->renderWindow() == rw) {
            srcIdx = i;
            break;
        }
    }
    if (srcIdx < 0) return;

    if (self->m_ignoredRenderEndWindows.remove(rw) > 0) return;

    // Negative means a controlled refresh/reset is in progress; suppress
    // EndEvent sync until actors and cameras are both coherent.
    if (self->m_cameraSyncSourceIdx < 0) return;
    // When the source is an external view (index >= sub-view count),
    // sub-view EndEvents should not trigger cross-pane sync.
    if (self->m_cameraSyncSourceIdx >= self->m_subViews.size()) return;

    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(
            self->m_subViews[srcIdx]);
    if (!srcRen || !srcRen->GetActiveCamera()) return;
    auto* srcCam = srcRen->GetActiveCamera();

    self->m_syncingCameras = true;
    for (int i = 0; i < self->m_subViews.size(); ++i) {
        if (i == srcIdx) continue;
        deepCopyCameraToSceneRenderer(self->m_subViews[i], srcCam);
        refreshOverlayWidgetsOnView(self->m_subViews[i]);
        auto* dv = self->m_subViews[i];
        if (dv) {
            if (auto* w = dv->getVtkWidget()) {
                if (auto* dstRw = w->renderWindow()) {
                    self->m_ignoredRenderEndWindows.insert(dstRw);
                    dstRw->Render();
                }
            }
        }
    }
    self->syncPivotFromView(srcIdx);
    self->m_syncingCameras = false;
}

void vtkComparativeViewWidget::interactionCallback(vtkObject* caller,
                                                   unsigned long eid,
                                                   void* clientData,
                                                   void*) {
    if (!clientData) return;
    auto* self = static_cast<vtkComparativeViewWidget*>(clientData);
    if (!self || self->m_shutdownDone || self->m_closing ||
        self->m_syncingCameras || !self->m_cameraLinkEnabled)
        return;

    int srcIdx = -1;
    for (int i = 0; i < self->m_subViews.size(); ++i) {
        auto* view = self->m_subViews[i];
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw || !rw->GetInteractor()) continue;
        if (rw->GetInteractor()->GetInteractorStyle() == caller) {
            srcIdx = i;
            break;
        }
    }
    if (srcIdx < 0) return;

    if (auto* srcWidget = self->m_subViews[srcIdx]->getVtkWidget()) {
        if (auto* srcRw = srcWidget->renderWindow()) {
            self->m_ignoredRenderEndWindows.remove(srcRw);
        }
    }

    switch (eid) {
        case vtkCommand::StartInteractionEvent:
            self->m_interacting = true;
            self->m_interactionSourceIdx = srcIdx;
            self->m_cameraSyncSourceIdx = srcIdx;
            if (self->m_interactionEndTimer)
                self->m_interactionEndTimer->stop();
            self->setInteractiveLOD(true);
            return;

        case vtkCommand::EndInteractionEvent:
            self->m_interacting = false;
            self->m_interactionSourceIdx = -1;
            self->m_cameraSyncSourceIdx = srcIdx;
            self->stopInteractionTimer();
            if (self->m_interactionEndTimer)
                self->m_interactionEndTimer->stop();
            self->setInteractiveLOD(false);
            self->onSubViewInteraction(srcIdx, true);
            // Refresh overlay widgets on ALL sub-views after interaction ends
            // to fix disappearing camera orientation widget/marker.
            self->refreshOverlayWidgetsOnAllSubViews();
            return;

        case vtkCommand::InteractionEvent:
            self->onSubViewInteraction(srcIdx, true);
            return;

        case vtkCommand::MouseWheelForwardEvent:
        case vtkCommand::MouseWheelBackwardEvent:
            QTimer::singleShot(0, self, [self, srcIdx]() {
                if (self->m_closing || !self->m_cameraLinkEnabled) return;
                self->onSubViewInteraction(srcIdx, true);
            });
            return;

        default:
            return;
    }
}

void vtkComparativeViewWidget::onSubViewInteraction(int viewIdx,
                                                    bool renderOthers) {
    if (m_syncingCameras || m_closing || !isVisible()) return;
    if (viewIdx < 0 || viewIdx >= m_subViews.size()) return;

    auto* srcView = m_subViews[viewIdx];
    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(srcView);
    if (!srcRen) return;
    auto* srcCam = srcRen->GetActiveCamera();
    if (!srcCam) return;

    m_cameraSyncSourceIdx = viewIdx;

    m_syncingCameras = true;
    // Refresh the source view's overlay widgets unless the camera orientation
    // widget is actively being dragged — On() inside RefreshOverlayWidgets
    // would reset the widget's interaction state.
    bool srcWidgetInteracting = false;
    if (auto* vtkW = srcView->getVtkWidget()) {
        srcWidgetInteracting = vtkW->isCameraOrientationWidgetInteracting();
    }
    if (!srcWidgetInteracting) {
        refreshOverlayWidgetsOnView(srcView);
    }
    for (int i = 0; i < m_subViews.size(); ++i) {
        if (i == viewIdx) continue;
        deepCopyCameraToSceneRenderer(m_subViews[i], srcCam);
        refreshOverlayWidgetsOnView(m_subViews[i]);
        if (auto* dv = m_subViews[i]) {
            if (auto* w = dv->getVtkWidget()) {
                if (auto* dstRw = w->renderWindow()) {
                    m_ignoredRenderEndWindows.insert(dstRw);
                    dstRw->Render();
                }
            }
        }
    }
    syncPivotFromView(viewIdx);
    m_syncingCameras = false;
}

void vtkComparativeViewWidget::refreshOverlayWidgetsOnAllSubViews() {
    if (m_closing) return;
    for (auto* view : m_subViews) {
        if (!view) continue;
        refreshOverlayWidgetsOnView(view);
        if (auto* w = view->getVtkWidget()) {
            w->update();
        }
    }
}

void vtkComparativeViewWidget::stopInteractionTimer() {
    if (m_cameraSyncRenderTimer && m_cameraSyncRenderTimer->isActive()) {
        m_cameraSyncRenderTimer->stop();
    }
}

void vtkComparativeViewWidget::restartInteractionEndTimer() {
    if (!m_interactionEndTimer) {
        m_interactionEndTimer = new QTimer(this);
        m_interactionEndTimer->setSingleShot(true);
        connect(m_interactionEndTimer, &QTimer::timeout, this,
                &vtkComparativeViewWidget::onInteractionEndTimer);
    }
    m_interactionEndTimer->start(200);
}

void vtkComparativeViewWidget::onInteractionEndTimer() {
    if (!m_interacting) return;
    m_interacting = false;
    m_interactionSourceIdx = -1;
    setInteractiveLOD(false);
    if (m_cameraSyncSourceIdx >= 0 &&
        m_cameraSyncSourceIdx < m_subViews.size()) {
        onSubViewInteraction(m_cameraSyncSourceIdx, true);
    }
}

void vtkComparativeViewWidget::scheduleCameraSyncRender() {
    if (m_closing) return;
    if (!m_cameraSyncRenderTimer) {
        m_cameraSyncRenderTimer = new QTimer(this);
        connect(m_cameraSyncRenderTimer, &QTimer::timeout, this,
                &vtkComparativeViewWidget::performCameraSyncRender);
    }
    m_cameraSyncRenderTimer->setSingleShot(true);
    if (!m_cameraSyncRenderTimer->isActive()) m_cameraSyncRenderTimer->start(0);
}

void vtkComparativeViewWidget::performCameraSyncRender() {
    if (m_closing || !isVisible() || m_subViews.isEmpty()) return;

    const int sourceIdx = (m_cameraSyncSourceIdx >= 0 &&
                           m_cameraSyncSourceIdx < m_subViews.size())
                                  ? m_cameraSyncSourceIdx
                                  : -1;

    for (auto* view : m_subViews) {
        const int idx = m_subViews.indexOf(view);
        if (idx == sourceIdx) continue;
        if (!view || !view->getVtkWidget()) continue;
        refreshOverlayWidgetsOnView(view);
        if (auto* w = view->getVtkWidget()) {
            if (auto* rw = w->renderWindow()) {
                rw->Modified();
            }
            w->update();
        }
    }
}

void vtkComparativeViewWidget::setInteractiveLOD(bool enable) {
    // ParaView: vtkPVRenderView::InteractiveRender / StillRender +
    // ShouldUseLODRendering(geometry_size).
    for (auto* view : m_subViews) {
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw) continue;

        auto* renderers = rw->GetRenderers();
        if (!renderers) continue;
        renderers->InitTraversal();
        while (auto* ren = renderers->GetNextItem()) {
            if (enable) {
                VtkRendering::BeginInteractiveLOD(ren);
            } else {
                VtkRendering::EndInteractiveLOD(ren);
            }
        }
    }
}

void vtkComparativeViewWidget::clearHighlightClones() {
    for (auto it = m_highlightClonesBySource.begin();
         it != m_highlightClonesBySource.end(); ++it) {
        int idx = 0;
        for (auto* view : m_subViews) {
            if (!view || idx >= it.value().size()) break;
            vtkSmartPointer<vtkActor> clone = it.value()[idx++];
            if (!clone) continue;
            auto* vis = view->getVisualizer3D();
            auto* ren = vis ? vis->getCurrentRenderer() : nullptr;
            if (ren) ren->RemoveActor(clone);
        }
    }
    m_highlightClonesBySource.clear();
}

void vtkComparativeViewWidget::disconnectExternalHighlighter() {
    QObject::disconnect(m_hlActorAddedConn);
    QObject::disconnect(m_hlActorRemovedConn);
    QObject::disconnect(m_hlClearedConn);
    QObject::disconnect(m_hlOverlayConn);
    QObject::disconnect(m_hlSelectionFinishedConn);
    m_hlActorAddedConn = {};
    m_hlActorRemovedConn = {};
    m_hlClearedConn = {};
    m_hlOverlayConn = {};
    m_hlSelectionFinishedConn = {};
    clearHighlightClones();
}

void vtkComparativeViewWidget::connectExternalHighlighter(
        QObject* highlighter) {
    disconnectExternalHighlighter();

    auto* hl = qobject_cast<cvSelectionHighlighter*>(highlighter);
    if (!hl) return;

    auto cloneHighlightActor = [](vtkActor* src) -> vtkSmartPointer<vtkActor> {
        if (!src) return {};
        auto clone = vtkSmartPointer<vtkActor>::New();
        clone->ShallowCopy(src);
        if (auto* srcMapper =
                    vtkDataSetMapper::SafeDownCast(src->GetMapper())) {
            auto mapper = vtkSmartPointer<vtkDataSetMapper>::New();
            mapper->ShallowCopy(srcMapper);
            clone->SetMapper(mapper);
        }
        clone->SetPickable(0);
        return clone;
    };

    auto safeUpdate = [this]() { scheduleSubViewRefresh(true); };

    auto removeClones = [this](vtkActor* source) {
        auto it = m_highlightClonesBySource.find(source);
        if (it == m_highlightClonesBySource.end()) return;
        int idx = 0;
        for (auto* view : m_subViews) {
            if (!view || idx >= it.value().size()) break;
            vtkSmartPointer<vtkActor> clone = it.value()[idx++];
            if (!clone) continue;
            auto* vis = view->getVisualizer3D();
            auto* ren = vis ? vis->getCurrentRenderer() : nullptr;
            if (ren) ren->RemoveActor(clone);
        }
        m_highlightClonesBySource.erase(it);
    };

    // SELECTED uses selectionOverlayUpdated; highlightActorAdded is
    // hover/preselect.
    m_hlActorAddedConn = connect(
            hl, &cvSelectionHighlighter::highlightActorAdded, this,
            [this, cloneHighlightActor, safeUpdate,
             removeClones](vtkActor* actor) {
                if (m_closing || !actor) return;
                removeClones(actor);
                QList<vtkSmartPointer<vtkActor>> clones;
                for (auto* view : m_subViews) {
                    if (!view) continue;
                    auto* vis = view->getVisualizer3D();
                    auto* ren = vis ? vis->getCurrentRenderer() : nullptr;
                    if (!ren) continue;
                    auto clone = cloneHighlightActor(actor);
                    if (clone) {
                        ren->AddActor(clone);
                        clones.append(clone);
                    }
                }
                if (!clones.isEmpty())
                    m_highlightClonesBySource.insert(actor, clones);
                safeUpdate();
            });
    m_hlActorRemovedConn =
            connect(hl, &cvSelectionHighlighter::highlightActorRemoved, this,
                    [this, removeClones, safeUpdate](vtkActor* actor) {
                        if (m_closing) return;
                        removeClones(actor);
                        safeUpdate();
                    });
    m_hlClearedConn = connect(
            hl, &cvSelectionHighlighter::highlightsCleared, this, [this]() {
                if (m_closing) return;
                clearHighlightClones();
                for (auto* view : m_subViews) {
                    if (!view) continue;
                    auto* vis = dynamic_cast<Visualization::VtkVis*>(
                            view->getVisualizer3D());
                    if (vis) {
                        cvSelectionHighlighter::clearSelectionOverlay(vis);
                    }
                }
                forceRenderAllSubViews();
            });

    auto applyOverlayToSubViews = [this](vtkPolyData* poly, int kind) {
        if (m_closing) return;
        for (auto* view : m_subViews) {
            if (!view) continue;
            auto* vis = dynamic_cast<Visualization::VtkVis*>(
                    view->getVisualizer3D());
            if (!vis) continue;
            if (kind == cvSelectionHighlighter::SelectionOverlayNone) {
                cvSelectionHighlighter::clearSelectionOverlay(vis);
            } else {
                cvSelectionHighlighter::applySelectionOverlay(
                        vis, poly,
                        static_cast<
                                cvSelectionHighlighter::SelectionOverlayKind>(
                                kind));
            }
        }
    };

    m_hlOverlayConn = connect(
            hl, &cvSelectionHighlighter::selectionOverlayUpdated, this,
            [this, applyOverlayToSubViews](vtkPolyData* poly, int kind) {
                applyOverlayToSubViews(poly, kind);
                if (!m_closing) forceRenderAllSubViews();
            });

    if (auto* ctrl = cvSelectionToolController::instance()) {
        m_hlSelectionFinishedConn =
                connect(ctrl, &cvSelectionToolController::selectionFinished,
                        this, [this](const cvSelectionData&) {
                            if (!m_closing) forceRenderAllSubViews();
                        });
    }
}

void vtkComparativeViewWidget::refreshSubViews() {
    for (auto* view : m_subViews) {
        if (view) syncSubViewBackgroundFromGlobal(view);
    }
    scheduleSubViewRefresh(true);
}

void vtkComparativeViewWidget::setEntityListProvider(
        EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshEntityCombo();
}

void vtkComparativeViewWidget::refreshEntityCombo() {
    if (!m_entityCombo || !m_entityListProvider) return;

    auto* model = qobject_cast<QStandardItemModel*>(m_entityCombo->model());
    if (!model) {
        model = new QStandardItemModel(m_entityCombo);
        m_entityCombo->setModel(model);
    }

    m_updatingShowingCombo = true;
    model->clear();
    auto entities = m_entityListProvider();

    QSet<ccHObject*> preserved;
    for (auto* entity : entities) {
        if (entity && m_selectedEntities.contains(entity)) {
            preserved.insert(entity);
        }
    }
    m_selectedEntities = preserved;

    auto* noneItem = new QStandardItem(tr("None"));
    noneItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
    noneItem->setData(QVariant::fromValue<quintptr>(0),
                      COMPARATIVE_ENTITY_ROLE);
    noneItem->setCheckState(m_selectedEntities.isEmpty() ? Qt::Checked
                                                         : Qt::Unchecked);
    model->appendRow(noneItem);

    for (auto* entity : entities) {
        if (!entity) continue;
        auto* item = new QStandardItem(entity->getName());
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
        item->setData(QVariant::fromValue<quintptr>(
                              reinterpret_cast<quintptr>(entity)),
                      COMPARATIVE_ENTITY_ROLE);
        item->setCheckState(m_selectedEntities.contains(entity)
                                    ? Qt::Checked
                                    : Qt::Unchecked);
        model->appendRow(item);
    }

    m_entityCombo->setCurrentIndex(0);
    m_updatingShowingCombo = false;
    updateShowingComboText();
}

QSet<ccHObject*> vtkComparativeViewWidget::selectedEntitiesFromCombo() const {
    QSet<ccHObject*> result;
    if (!m_entityCombo) return result;
    auto* model = qobject_cast<QStandardItemModel*>(m_entityCombo->model());
    if (!model) return result;

    for (int row = 1; row < model->rowCount(); ++row) {
        auto* item = model->item(row);
        if (!item || item->checkState() != Qt::Checked) continue;
        auto* entity = reinterpret_cast<ccHObject*>(
                item->data(COMPARATIVE_ENTITY_ROLE).value<quintptr>());
        if (entity) result.insert(entity);
    }
    return result;
}

void vtkComparativeViewWidget::updateShowingComboText() {
    if (!m_entityCombo || !m_entityCombo->lineEdit()) return;

    QString text;
    if (m_selectedEntities.isEmpty()) {
        text = tr("None");
    } else if (m_selectedEntities.size() == 1) {
        ccHObject* entity = *m_selectedEntities.constBegin();
        text = entity ? entity->getName() : tr("None");
    } else {
        text = tr("%1 selected").arg(m_selectedEntities.size());
    }
    m_entityCombo->lineEdit()->setText(text);
}

void vtkComparativeViewWidget::applyShowingSelection(bool resetCamera) {
    if (m_updatingShowingCombo || m_closing) return;

    m_selectedEntities = selectedEntitiesFromCombo();
    updateShowingComboText();

    if (m_type == RENDER) {
        // Representation changes emit per-view redraws.  Keep camera sync
        // suppressed until performSubViewRefresh has rebuilt all panes and
        // explicitly synced from the first view.
        m_cameraSyncSourceIdx = -1;
        applySelectedEntitiesToRenderViews(resetCamera);
        m_needsCameraReset = resetCamera && !m_selectedEntities.isEmpty();
        scheduleSubViewRefresh(true);
    } else {
        applySelectedEntitiesToChartViews();
        forceRenderAllSubViews();
    }

    if (m_statusLabel) {
        m_statusLabel->setText(
                m_selectedEntities.isEmpty()
                        ? tr("Showing: None")
                        : tr("Showing: %1 selected")
                                  .arg(m_selectedEntities.size()));
    }
}

void vtkComparativeViewWidget::applySelectedEntitiesToRenderViews(
        bool /*resetCamera*/) {
    if (m_type != RENDER || m_subViews.isEmpty()) return;

    auto& repMgr = ecvRepresentationManager::instance();
    m_applyingShowingSelection = true;
    for (auto* view : m_subViews) {
        if (!view) continue;
        view->setSceneDB(nullptr);

        const QSet<ccHObject*> previous = m_showingEntitiesByView.value(view);
        for (auto* entity : previous) {
            if (!entity || m_selectedEntities.contains(entity)) continue;
            view->removeFromOwnDB(entity);
            repMgr.removeRepresentation(entity, view);
        }

        const auto reps = repMgr.getRepresentationsForView(view);
        for (auto* rep : reps) {
            if (!rep || !rep->getEntity()) continue;
            auto* entity = rep->getEntity();
            if (entity->isKindOf(CV_TYPES::LABEL_2D)) continue;
            if (!m_selectedEntities.contains(entity)) {
                view->removeFromOwnDB(entity);
                repMgr.removeRepresentation(entity, view);
            }
        }

        for (auto* entity : m_selectedEntities) {
            if (!entity) continue;
            if (auto* ownDB = view->getOwnDB()) {
                if (!hasDirectChild(ownDB, entity)) {
                    view->addToOwnDB(entity, true);
                }
            }
            auto* rep = repMgr.ensureRepresentation(entity, view);
            if (!rep) continue;

            if (m_sourceView) {
                if (auto* sourceRep =
                            repMgr.getRepresentation(entity, m_sourceView)) {
                    if (sourceRep != rep &&
                        !sameRepresentationProperties(
                                rep->properties(), sourceRep->properties())) {
                        rep->setProperties(sourceRep->properties());
                    }
                }
            }
            if (!rep->isVisible()) {
                rep->setVisible(true);
            }
        }

        m_showingEntitiesByView[view] = m_selectedEntities;

        if (auto* ren = vtkComparativeViewWidget::getSceneRenderer(view)) {
            ren->Modified();
        }
        if (auto* widget = view->getVtkWidget()) {
            if (auto* rw = widget->renderWindow()) rw->Modified();
        }
    }
    m_applyingShowingSelection = false;
}

void vtkComparativeViewWidget::applySelectedEntitiesToChartViews() {
    ccHObject* firstSelected = nullptr;
    if (m_entityCombo) {
        if (auto* model =
                    qobject_cast<QStandardItemModel*>(m_entityCombo->model())) {
            for (int row = 1; row < model->rowCount(); ++row) {
                auto* item = model->item(row);
                if (!item || item->checkState() != Qt::Checked) continue;
                firstSelected = reinterpret_cast<ccHObject*>(
                        item->data(COMPARATIVE_ENTITY_ROLE).value<quintptr>());
                if (firstSelected) break;
            }
        }
    }

    for (auto* widget : m_subWidgets) {
        auto* chart = qobject_cast<vtkChartView*>(widget);
        if (chart) chart->setEntity(firstSelected);
    }
}

void vtkComparativeViewWidget::setInitialEntity(ccHObject* entity) {
    m_initialEntity = entity;
}

void vtkComparativeViewWidget::createChartSubViews() {
    vtkChartView::ChartType chartType = (m_type == BAR_CHART)
                                                ? vtkChartView::BAR_CHART
                                                : vtkChartView::LINE_CHART;

    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            auto* chart = new vtkChartView(chartType, this);
            chart->setCompactMode(true);
            chart->setSizePolicy(QSizePolicy::Expanding,
                                 QSizePolicy::Expanding);
            if (m_entityListProvider) {
                chart->setEntityListProvider(m_entityListProvider);
            }
            chart->setCompactMode(true);
            m_gridLayout->addWidget(chart, r, c);
            m_gridLayout->setRowStretch(r, 1);
            m_gridLayout->setColumnStretch(c, 1);
            m_subWidgets.append(chart);
            emit subViewCreated(chart);
        }
    }

    refreshEntityCombo();
}

void vtkComparativeViewWidget::buildToolbar() {
    m_toolbar = new QWidget(this);
    auto* lay = new QHBoxLayout(m_toolbar);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(2);

    auto* resetCamBtn = new QPushButton(tr("Reset"), m_toolbar);
    resetCamBtn->setToolTip(tr("Reset camera for all sub-views"));
    lay->addWidget(resetCamBtn);
    connect(resetCamBtn, &QPushButton::clicked, this, [this]() {
        m_baselineCamera.valid = false;
        zoomToData();
    });

    auto* refreshBtn = new QPushButton(tr("Refresh"), m_toolbar);
    refreshBtn->setToolTip(
            tr("Re-load geometry from source view into all sub-views"));
    lay->addWidget(refreshBtn);
    connect(refreshBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::refreshSubViews);

    // 对所有类型都启用 entity 选择（不仅是图表类型）
    if (true) {  // 原来是 if (m_type != RENDER)
        lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));
        auto* showLabel = new QLabel(tr("<b>Showing:</b>"), m_toolbar);
        lay->addWidget(showLabel);
        m_entityCombo = new QComboBox(m_toolbar);
        m_entityCombo->setMinimumWidth(120);
        m_entityCombo->setToolTip(tr("Entities displayed in all sub-views"));
        m_entityCombo->setEditable(true);
        m_entityCombo->lineEdit()->setReadOnly(true);
        m_entityCombo->lineEdit()->setFrame(false);
        m_entityCombo->setInsertPolicy(QComboBox::NoInsert);
        auto* showingModel = new QStandardItemModel(m_entityCombo);
        m_entityCombo->setModel(showingModel);
        if (m_entityCombo->view() && m_entityCombo->view()->viewport()) {
            m_entityCombo->view()->viewport()->installEventFilter(this);
        }
        lay->addWidget(m_entityCombo);

        connect(showingModel, &QStandardItemModel::itemChanged, this,
                [this, showingModel](QStandardItem* item) {
                    if (!item || m_updatingShowingCombo) return;

                    m_updatingShowingCombo = true;
                    if (item->row() == 0 && item->checkState() == Qt::Checked) {
                        for (int row = 1; row < showingModel->rowCount();
                             ++row) {
                            if (auto* other = showingModel->item(row)) {
                                other->setCheckState(Qt::Unchecked);
                            }
                        }
                    } else {
                        bool anyChecked = false;
                        for (int row = 1; row < showingModel->rowCount();
                             ++row) {
                            auto* other = showingModel->item(row);
                            if (other && other->checkState() == Qt::Checked) {
                                anyChecked = true;
                                break;
                            }
                        }
                        if (auto* noneItem = showingModel->item(0)) {
                            noneItem->setCheckState(anyChecked ? Qt::Unchecked
                                                               : Qt::Checked);
                        }
                    }
                    m_updatingShowingCombo = false;

                    applyShowingSelection(true);
                });
    }

    m_statusLabel = new QLabel(m_toolbar);
    m_statusLabel->setContentsMargins(4, 0, 4, 0);
    lay->addWidget(m_statusLabel);
    lay->addStretch(1);

    const auto wheelControls = m_toolbar->findChildren<QWidget*>();
    for (auto* control : wheelControls) {
        if (qobject_cast<QComboBox*>(control) ||
            qobject_cast<QSlider*>(control) ||
            qobject_cast<QAbstractSpinBox*>(control)) {
            control->setFocusPolicy(Qt::StrongFocus);
            control->installEventFilter(this);
        }
    }
}

void vtkComparativeViewWidget::saveBaselineCamera() {
    if (m_subViews.isEmpty()) return;
    auto* ren = vtkComparativeViewWidget::getSceneRenderer(m_subViews.first());
    if (!ren) return;
    auto* cam = ren->GetActiveCamera();
    if (!cam) return;

    cam->GetPosition(m_baselineCamera.position);
    cam->GetFocalPoint(m_baselineCamera.focalPoint);
    cam->GetViewUp(m_baselineCamera.viewUp);
    m_baselineCamera.viewAngle = cam->GetViewAngle();
    m_baselineCamera.parallelScale = cam->GetParallelScale();
    cam->GetClippingRange(m_baselineCamera.clippingRange);
    m_baselineCamera.valid = true;
}

void vtkComparativeViewWidget::restoreBaselineCamera(vtkGLView* view) {
    if (!m_baselineCamera.valid) return;
    auto* ren = vtkComparativeViewWidget::getSceneRenderer(view);
    if (!ren) return;
    auto* cam = ren->GetActiveCamera();
    if (!cam) return;

    cam->SetPosition(m_baselineCamera.position);
    cam->SetFocalPoint(m_baselineCamera.focalPoint);
    cam->SetViewUp(m_baselineCamera.viewUp);
    cam->SetViewAngle(m_baselineCamera.viewAngle);
    cam->SetParallelScale(m_baselineCamera.parallelScale);
    cam->SetClippingRange(m_baselineCamera.clippingRange);
}

void vtkComparativeViewWidget::syncViewPropertiesFromSource(vtkGLView* source,
                                                            vtkGLView* target) {
    if (!source || !target || source == target) return;

    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(source);
    auto* dstRen = vtkComparativeViewWidget::getSceneRenderer(target);
    if (srcRen && dstRen) {
        double bg[3];
        srcRen->GetBackground(bg);
        dstRen->SetBackground(bg);

        double bg2[3];
        srcRen->GetBackground2(bg2);
        dstRen->SetBackground2(bg2);

        dstRen->SetGradientBackground(srcRen->GetGradientBackground());
    }

    target->setInteractionMode(
            stripClickableItems(source->getInteractionMode()));
    target->setPickingMode(source->getPickingMode());

    disableBubbleViewForSubView(target);
}

void vtkComparativeViewWidget::syncAllPropertiesFromFirst() {
    if (m_subViews.size() < 2) return;
    vtkGLView* first = m_subViews.first();
    if (!first) return;

    for (int i = 1; i < m_subViews.size(); ++i) {
        auto* view = m_subViews[i];
        if (!view) continue;

        auto* firstRen = vtkComparativeViewWidget::getSceneRenderer(first);
        if (firstRen && firstRen->GetActiveCamera()) {
            deepCopyCameraToSceneRenderer(view, firstRen->GetActiveCamera());
        }

        syncSubViewBackgroundFromGlobal(view);

        view->setInteractionMode(
                stripClickableItems(first->getInteractionMode()));
        view->setPickingMode(first->getPickingMode());

        disableBubbleViewForSubView(view);
    }

    syncRepresentationsFromFirst();
}
