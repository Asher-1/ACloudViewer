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
#include <Visualization/vtkGLView.h>
#include <Visualization/VtkCameraLink.h>
#include <Visualization/VtkVis.h>
#include <ecvHObject.h>

#include <vtkImageData.h>
#include <vtkRenderWindow.h>
#include <vtkWindowToImageFilter.h>
#include <vtkActorCollection.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewTitleRegistry.h>

#include <QCheckBox>
#include <QComboBox>
#include <QCoreApplication>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QResizeEvent>
#include <QShowEvent>
#include <QMenu>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <VtkRendering/Core/VtkLODHelper.h>

#include <vtkCallbackCommand.h>
#include <vtkInteractorObserver.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

#include <QApplication>

#include <cmath>

#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>

static constexpr int COMPARATIVE_SPACING = 1;

// Return the VtkVis scene renderer for a view.  This is the renderer that
// holds the actual 3D actors.  Using renderWindow->GetFirstRenderer() is
// unreliable because QVTKWidgetCustom::addActor()/defaultRenderer() may add
// extra renderers to the window, making GetFirstRenderer() return a
// different (possibly empty) renderer.
vtkRenderer* vtkComparativeViewWidget::vtkComparativeViewWidget::getSceneRenderer(vtkGLView* view) {
    if (!view) return nullptr;
    if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                view->getVisualizer3D())) {
        return vis->getCurrentRenderer();
    }
    auto* w = view->getVtkWidget();
    if (!w) return nullptr;
    auto* rw = w->renderWindow();
    return (rw && rw->GetRenderers())
                   ? rw->GetRenderers()->GetFirstRenderer()
                   : nullptr;
}

// Remove every renderer from the render window EXCEPT the VtkVis scene
// renderer and orientation-marker widget.  Extra renderers (defaultRenderer,
// etc.) can overwrite the scene with their own Erase/background, causing
// comparative sub-views to look "zoomed-in" or blank.
static void stripExtraRenderers(vtkGLView* view) {
    auto* sceneRen = vtkComparativeViewWidget::vtkComparativeViewWidget::getSceneRenderer(view);
    if (!sceneRen) return;
    auto* w = view->getVtkWidget();
    if (!w) return;
    auto* rw = w->renderWindow();
    if (!rw || !rw->GetRenderers()) return;
    auto* coll = rw->GetRenderers();
    std::vector<vtkRenderer*> toRemove;
    coll->InitTraversal();
    while (auto* ren = coll->GetNextItem()) {
        // ✅ 保留场景renderer
        if (ren == sceneRen) continue;

        // ✅ 方法1: 检查viewport（非全屏的通常是orientation widget等overlay）
        double* vp = ren->GetViewport();
        bool isFullViewport = (vp[0] == 0 && vp[1] == 0 &&
                               vp[2] == 1 && vp[3] == 1);
        if (!isFullViewport) continue;  // 保留非全屏renderer

        // ✅ 方法2: 检查layer（foreground layer通常是overlay widget）
        if (ren->GetLayer() > 0) continue;  // 保留前景层renderer

        // ✅ 方法3: 检查actor数量（orientation widget通常只有少量2D prop）
        vtkActorCollection* actors = ren->GetActors();
        int nActors = actors ? actors->GetNumberOfItems() : 0;
        vtkPropCollection* props2D = ren->GetViewProps();
        int nProps2D = props2D ? props2D->GetNumberOfItems() : 0;
        if (nActors == 0 && nProps2D > 0 && nProps2D <= 5) continue;

        // ✅ 方法4: 通过类名检查（原有逻辑作为补充）
        bool isOrientationWidget = false;
        if (actors && actors->GetNumberOfItems() > 0) {
            actors->InitTraversal();
            while (vtkActor* actor = actors->GetNextItem()) {
                std::string className = actor->GetClassName();
                if (className.find("Axes") != std::string::npos ||
                    className.find("Orientation") != std::string::npos ||
                    className.find("Axis") != std::string::npos ||
                    className.find("Coordinate") != std::string::npos) {
                    isOrientationWidget = true;
                    break;
                }
            }
        }

        // 如果不是任何类型的overlay/widget，则移除
        if (!isOrientationWidget) {
            toRemove.push_back(ren);
        }
    }

    for (auto* ren : toRemove) rw->RemoveRenderer(ren);
}

static void deepCopyCameraToAllRenderers(vtkGLView* view,
                                         vtkCamera* srcCam) {
    if (!view || !srcCam) return;
    auto* w = view->getVtkWidget();
    if (!w) return;
    auto* rw = w->renderWindow();
    if (!rw || !rw->GetRenderers()) return;
    auto* coll = rw->GetRenderers();
    coll->InitTraversal();
    while (auto* ren = coll->GetNextItem()) {
        if (ren->GetActiveCamera()) {
            ren->GetActiveCamera()->DeepCopy(srcCam);
            ren->ResetCameraClippingRange();
        }
    }
}

static bool safeRenderWindow(vtkGLView* view, bool immediate = false) {
    if (!view) return false;
    auto* w = view->getVtkWidget();
    if (!w || !w->isVisible() || w->width() < 2 || w->height() < 2)
        return false;
    if (immediate) {
        // Force the render window to acknowledge the camera change by
        // bumping its MTime.  Without this, VTK may short-circuit
        // the Render() because it thinks nothing changed.
        auto* rw = w->renderWindow();
        if (rw) rw->Modified();
        w->repaint();
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

static QString comparativeViewTypeKey(vtkComparativeViewWidget::ComparativeType type) {
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

static void copyRepresentationsBetweenViews(ecvGenericGLDisplay* sourceView,
                                            ecvGenericGLDisplay* destView) {
    if (!sourceView || !destView || sourceView == destView) return;
    auto& repMgr = ecvRepresentationManager::instance();
    auto reps = repMgr.getRepresentationsForView(sourceView);
    for (auto* srcRep : reps) {
        if (!srcRep || !srcRep->getEntity()) continue;
        if (srcRep->getEntity()->isKindOf(CV_TYPES::LABEL_2D)) continue;
        auto* dst = repMgr.ensureRepresentation(srcRep->getEntity(), destView);
        if (dst) {
            dst->setProperties(srcRep->properties());
            if (srcRep->hasVisibilityOverride()) {
                dst->setVisible(srcRep->isVisible());
            }
        }
    }
    if (auto* glDest = dynamic_cast<vtkGLView*>(destView)) {
        if (ccHObject* root = sourceView->getSceneDB()) {
            if (!glDest->getSceneDB()) glDest->setSceneDB(root);
        }
    }
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

    // Undo overlay actors merged into the first renderer (avoid dangling refs).
    if (m_overlayMode && m_type == RENDER && m_subViews.size() > 1) {
        auto* dstRen = vtkComparativeViewWidget::getSceneRenderer(m_subViews.first());
        if (dstRen) {
            for (int i = 1; i < m_subViews.size(); ++i) {
                auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(m_subViews[i]);
                if (!srcRen) continue;
                auto* actors = srcRen->GetActors();
                if (!actors) continue;
                actors->InitTraversal();
                vtkActor* a = nullptr;
                while ((a = actors->GetNextActor())) {
                    dstRen->RemoveActor(a);
                }
            }
        }
        m_overlayMode = false;
    }

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
}

QString vtkComparativeViewWidget::title() const { return m_title; }

void vtkComparativeViewWidget::setSpacing(int spacing) {
    m_spacing = spacing;
    if (m_gridLayout) {
        m_gridLayout->setSpacing(spacing);
    }
}

void vtkComparativeViewWidget::setDimensions(int rows, int cols) {
    if (rows < 1 || cols < 1 || (rows == m_rows && cols == m_cols)) return;

    shutdown();
    m_shutdownDone = false;
    m_closing = false;

    m_rows = rows;
    m_cols = cols;
    setupGrid();
}

void vtkComparativeViewWidget::setRenderViewFactory(
        RenderViewFactory factory) {
    m_renderFactory = factory;
    if (m_type == RENDER && m_subWidgets.isEmpty()) {
        setupGrid();
    }
}

void vtkComparativeViewWidget::setSubViewInitCallback(SubViewInitCallback cb) {
    m_subViewInitCb = std::move(cb);
}

void vtkComparativeViewWidget::setSubViewShutdownHook(SubViewShutdownHook hook) {
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
                CVLog::Warning("[ComparativeView] Factory returned null for "
                               "cell (%d,%d)", r, c);
                continue;
            }

            view->setParent(this);

            view->setInteractionMode(
                    ecvGenericGLDisplay::MODE_TRANSFORM_CAMERA);

            QWidget* viewWidget = view->asWidget();
            if (!viewWidget) {
                CVLog::Warning("[ComparativeView] View widget null for "
                               "cell (%d,%d)", r, c);
                continue;
            }
            viewWidget->setMinimumSize(50, 50);
            viewWidget->setSizePolicy(QSizePolicy::Expanding,
                                      QSizePolicy::Expanding);
            viewWidget->setContextMenuPolicy(Qt::NoContextMenu);
            viewWidget->setProperty("_compViewDiag", true);
            m_gridLayout->addWidget(viewWidget, r, c);
            m_gridLayout->setRowStretch(r, 1);
            m_gridLayout->setColumnStretch(c, 1);

            m_subWidgets.append(viewWidget);
            m_subViews.append(view);
            if (!firstView) {
                firstView = view;
                m_activeSubView = view;

                // ParaView模式: Comparative视图初始为空
                // 用户通过Showing下拉框手动选择要显示的实体
                if (m_subViewInitCb) {
                    m_subViewInitCb(view);
                }
            } else {
                auto* firstRen = vtkComparativeViewWidget::getSceneRenderer(firstView);
                if (firstRen && firstRen->GetActiveCamera()) {
                    deepCopyCameraToAllRenderers(
                            view, firstRen->GetActiveCamera());
                }
            }

            if (auto* vis = view->getVisualizer3D()) {
                Visualization::VtkCameraLink::instance().removeView(vis);
            }

            disableBubbleViewForSubView(view);
            view->disableContext2DOverlay();
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
        ecvGenericGLDisplay* repView =
                const_cast<ecvGenericGLDisplay*>(rep->getView());
        bool isOurView = false;
        for (auto* sv : m_subViews) {
            if (sv == repView) { isOurView = true; break; }
        }
        if (repView == m_sourceView) isOurView = true;
        if (!isOurView) return;
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

    m_needsCameraReset = true;
    if (!m_closing) {
        // ParaView模式: 初始为空视窗，不执行同步
        // 用户通过Showing下拉框选择实体后会自动触发同步和刷新
        QTimer::singleShot(100, this, [this]() {
            if (m_closing) return;
            // 只初始化相机链接（确保4个视窗的初始状态一致）
            if (m_cameraLinkEnabled && m_subViews.size() > 1) {
                syncCamerasFromFirst();
            }
            scheduleSubViewRefresh(true);
        });
    }

    connect(&ecvViewManager::instance(),
            &ecvViewManager::pointIndicesSelected, this,
            [this](ccHObject*, const QSet<unsigned>&) {
                scheduleSubViewRefresh(true);
            });

    if (m_sourceView) {
        connect(m_sourceView, &vtkGLView::interactionModeChanged, this,
                &vtkComparativeViewWidget::syncInteractionModeToSubViews);
        connect(m_sourceView, &vtkGLView::pickingModeChanged, this,
                &vtkComparativeViewWidget::syncPickingModeToSubViews);
        connect(m_sourceView, &vtkGLView::cameraParamChanged, this,
                [this]() { syncCameraFromSourceView(); },
                Qt::UniqueConnection);
        connect(m_sourceView, &vtkGLView::interactionModeChanged, this,
                [this]() {
                    if (!m_closing && m_subViews.size() >= 2) {
                        for (auto* v : m_subViews) {
                            if (v && v != m_sourceView) {
                                v->setInteractionMode(
                                        stripClickableItems(
                                                m_sourceView->getInteractionMode()));
                                disableBubbleViewForSubView(v);
                            }
                        }
                    }
                });
        connect(m_sourceView, &vtkGLView::pickingModeChanged, this,
                [this]() {
                    if (!m_closing && m_subViews.size() >= 2) {
                        for (auto* v : m_subViews) {
                            if (v && v != m_sourceView) {
                                v->setPickingMode(m_sourceView->getPickingMode());
                            }
                        }
                    }
                });
        connect(m_sourceView, &vtkGLView::cameraParamChanged, this,
                [this]() {
                    if (!m_closing && m_subViews.size() >= 2) {
                        syncCameraFromSourceView();
                    }
                });

        // 监听DBTree变化以刷新Entity下拉框（不自动复制表示）
        connect(&ecvRepresentationManager::instance(),
                &ecvRepresentationManager::representationAdded, this,
                [this](ecvViewRepresentation* /*rep*/) {
                    if (m_closing || m_subViews.isEmpty()) return;
                    // 延迟刷新entity combo列表
                    QTimer::singleShot(100, this, [this]() {
                        if (m_closing) return;
                        refreshEntityCombo();
                    });
                }, Qt::UniqueConnection);
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
    }
}

void vtkComparativeViewWidget::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
}

bool vtkComparativeViewWidget::eventFilter(QObject* obj, QEvent* event) {
    // 只阻止ContextMenu事件
    if (event->type() == QEvent::ContextMenu) return true;

    // 明确不拦截 Wheel 事件，让 VTK 交互器正常处理滚轮操作
    if (event->type() == QEvent::Wheel) {
        return false;  // 返回 false 表示不拦截，传递给子 widget
    }

    // 处理鼠标按下：更新活跃子视窗并发出信号
    if (event->type() == QEvent::MouseButtonPress) {
        auto* w = qobject_cast<QWidget*>(obj);
        for (auto* view : m_subViews) {
            if (!view) continue;
            QWidget* vw = view->asWidget();
            if (vw && (vw == w || vw->isAncestorOf(w))) {
                m_activeSubView = view;
                ecvViewManager::instance().setActiveView(view);
                break;
            }
        }
        emit clicked();

        QTimer::singleShot(0, this, [this]() {
            if (m_activeSubView && !m_closing) {
                emit requestToolRebind(m_activeSubView);
            }
        });
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

void vtkComparativeViewWidget::forceRenderAllSubViews() {
    if (m_closing) return;
    for (auto* view : m_subViews) {
        if (!view) continue;
        view->redraw(false, true);
    }
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
    if (!m_subViewRefreshTimer->isActive())
        m_subViewRefreshTimer->start(32);
}

void vtkComparativeViewWidget::performSubViewRefresh() {
    if (m_closing || !isVisible() || m_subViews.isEmpty()) return;

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

    // Suppress the EndEvent camera-sync during the draw loop.  Actors
    // are being lazily created here; the camera is not meaningful yet.
    // We will enable sync after ResetCamera + explicit syncCamerasFromFirst.
    const int prevSyncSource = m_cameraSyncSourceIdx;
    if (needsCameraReset) m_cameraSyncSourceIdx = -1;

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
            int nRens = vtkW->renderWindow()
                                ? vtkW->renderWindow()
                                          ->GetRenderers()
                                          ->GetNumberOfItems()
                                : 0;
            int nActors = ren->GetActors()
                                  ? ren->GetActors()->GetNumberOfItems()
                                  : 0;
            CVLog::PrintDebug("[CompView] Before ResetCamera: nRenderers=%d "
                         "nActors=%d",
                         nRens, nActors);

            ren->ResetCamera();
            ren->ResetCameraClippingRange();
            double bounds[6];
            ren->ComputeVisiblePropBounds(bounds);
            if (bounds[0] <= bounds[1]) {
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
            deepCopyCameraToAllRenderers(first, ren->GetActiveCamera());
            double* rpos = ren->GetActiveCamera()->GetPosition();
            CVLog::PrintDebug("[CompView] ResetCamera done: "
                         "pos=(%.2f,%.2f,%.2f) ps=%.3f bounds=(%.2f..%.2f)",
                         rpos[0], rpos[1], rpos[2],
                         ren->GetActiveCamera()->GetParallelScale(), bounds[0],
                         bounds[1]);
        }
    }

    if (cameraResetDone) {
        m_needsCameraReset = false;
    } else if (needsCameraReset && forceSceneDirty) {
        CVLog::PrintDebug("[CompView] ResetCamera DEFERRED (widget not ready)");
        m_needsCameraReset = true;
        QTimer::singleShot(50, this, [this]() {
            if (!m_closing && m_needsCameraReset)
                scheduleSubViewRefresh(true);
        });
    }

    if (forceSceneDirty && m_cameraLinkEnabled && m_subViews.size() > 1) {
        syncCamerasFromFirst();
    }

    if (cameraResetDone) {
        saveBaselineCamera();
    }

    // Re-enable EndEvent camera sync now that cameras are correct.
    m_cameraSyncSourceIdx = cameraResetDone ? 0 : prevSyncSource;

    if (forceSceneDirty && (cameraResetDone || needsCameraReset)) {
        for (auto* view : m_subViews) {
            if (!view || !view->getVtkWidget()) continue;
            stripExtraRenderers(view);
            auto* rw = view->getVtkWidget()->renderWindow();
            if (rw) rw->Modified();
            view->getVtkWidget()->update();
        }
    }

    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* v = m_subViews[i];
        if (!v) continue;
        auto* w = v->getVtkWidget();
        auto* ren = vtkComparativeViewWidget::getSceneRenderer(v);
        auto* rw = w ? w->renderWindow() : nullptr;
        int rwW = 0, rwH = 0;
        if (rw) { int* sz = rw->GetSize(); rwW = sz[0]; rwH = sz[1]; }
        int nRen = (rw && rw->GetRenderers())
                           ? rw->GetRenderers()->GetNumberOfItems()
                           : 0;
        int nAct = (ren && ren->GetActors())
                           ? ren->GetActors()->GetNumberOfItems()
                           : 0;
        double ps = 0;
        double pos[3] = {0, 0, 0};
        if (ren && ren->GetActiveCamera()) {
            ps = ren->GetActiveCamera()->GetParallelScale();
            double* p = ren->GetActiveCamera()->GetPosition();
            pos[0] = p[0]; pos[1] = p[1]; pos[2] = p[2];
        }
        CVLog::PrintDebug("[CompView] EndRefresh v%d: widget=%dx%d rw=%dx%d "
                     "nRen=%d nAct=%d cam=(%.2f,%.2f,%.2f) ps=%.3f vis=%d",
                     i, w ? w->width() : 0, w ? w->height() : 0, rwW, rwH,
                     nRen, nAct, pos[0], pos[1], pos[2], ps,
                     w ? w->isVisible() : 0);
    }
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

    m_syncingCameras = true;
    for (auto* view : m_subViews) {
        deepCopyCameraToAllRenderers(view, srcCam);
    }
    syncPivotFromFirst();
    m_syncingCameras = false;
    scheduleCameraSyncRender();
}

void vtkComparativeViewWidget::copyActorsAcrossSubViews() {
    scheduleSubViewRefresh(false);
}

void vtkComparativeViewWidget::syncRepresentationsFromFirst() {
    if (m_subViews.size() < 2) return;
    vtkGLView* first = m_subViews.first();
    if (!first) return;

    ecvGenericGLDisplay* srcDisplay = first;
    if (m_sourceView && m_sourceView != first) {
        auto srcReps =
                ecvRepresentationManager::instance().getRepresentationsForView(
                        m_sourceView);
        if (!srcReps.isEmpty()) srcDisplay = m_sourceView;
    }

    ccHObject* sceneDB = srcDisplay->getSceneDB();
    if (!sceneDB) sceneDB = first->getSceneDB();

    if (!sceneDB) {
        return;
    }

    auto srcReps = ecvRepresentationManager::instance().getRepresentationsForView(srcDisplay);
    if (srcReps.isEmpty()) {
        return;
    }

    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* dstView = m_subViews[i];
        if (!dstView || dstView == srcDisplay) continue;

        if (!dstView->getSceneDB()) dstView->setSceneDB(sceneDB);

        copyRepresentationsBetweenViews(srcDisplay, dstView);
    }
}

void vtkComparativeViewWidget::syncCamerasFromFirst() {
    if (m_closing || m_subViews.size() < 2) return;
    vtkGLView* first = m_subViews.first();
    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(first);
    if (!srcRen) return;
    auto* srcCam = srcRen->GetActiveCamera();
    if (!srcCam) return;

    double* pos = srcCam->GetPosition();
    CVLog::PrintDebug("[CompView] syncCamerasFromFirst: src pos=(%.2f,%.2f,%.2f) "
                 "ps=%.3f nRens=%d",
                 pos[0], pos[1], pos[2], srcCam->GetParallelScale(),
                 first->getVtkWidget() && first->getVtkWidget()->renderWindow()
                         ? first->getVtkWidget()
                                   ->renderWindow()
                                   ->GetRenderers()
                                   ->GetNumberOfItems()
                         : -1);

    for (int i = 1; i < m_subViews.size(); ++i) {
        deepCopyCameraToAllRenderers(m_subViews[i], srcCam);
    }

    syncPivotFromFirst();
}

void vtkComparativeViewWidget::syncPivotFromView(int srcIdx) {
    if (m_closing || m_subViews.size() < 2) return;
    if (srcIdx < 0 || srcIdx >= m_subViews.size()) return;

    vtkGLView* srcView = m_subViews[srcIdx];
    if (!srcView || !srcView->viewContext()) return;

    const CCVector3d& pivot =
            srcView->viewContext()->viewportParams.getPivotPoint();

    for (int i = 0; i < m_subViews.size(); ++i) {
        if (i == srcIdx) continue;
        auto* dstView = m_subViews[i];
        if (!dstView || !dstView->viewContext()) continue;
        dstView->viewContext()->viewportParams.setPivotPoint(pivot, true);
        dstView->viewContext()->autoPivotCandidate = pivot;
        if (auto* vis = dynamic_cast<Visualization::VtkVis*>(
                    dstView->getVisualizer3D())) {
            vis->setCenterOfRotation(pivot.x, pivot.y, pivot.z);
        }
    }
}

void vtkComparativeViewWidget::syncPivotFromFirst() {
    syncPivotFromView(0);
}

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
                cb->SetCallback(
                        &vtkComparativeViewWidget::interactionCallback);

                unsigned long t;
                t = style->AddObserver(
                        vtkCommand::StartInteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(vtkCommand::InteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(
                        vtkCommand::EndInteractionEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(
                        vtkCommand::MouseWheelForwardEvent, cb);
                m_cameraObservers.push_back({style, t, cb});
                t = style->AddObserver(
                        vtkCommand::MouseWheelBackwardEvent, cb);
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
            cb->SetCallback(
                    &vtkComparativeViewWidget::renderEndCallback);
            unsigned long t =
                    rw->AddObserver(vtkCommand::EndEvent, cb);
            m_cameraObservers.push_back({rw, t, cb});
        }
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
}

void vtkComparativeViewWidget::renderEndCallback(
        vtkObject* caller, unsigned long /*eid*/, void* clientData, void*) {
    auto* self = static_cast<vtkComparativeViewWidget*>(clientData);
    if (!self || self->m_shutdownDone || self->m_closing ||
        self->m_syncingCameras || !self->m_cameraLinkEnabled)
        return;

    // 只在真正的连续鼠标拖拽时跳过（有活跃的 repeating interaction timer）
    // 方向键、滚轮、程序化相机变更等离散事件都应该正常同步
    bool hasActiveDragTimer = self->m_cameraSyncRenderTimer &&
                              !self->m_cameraSyncRenderTimer->isSingleShot() &&
                              self->m_cameraSyncRenderTimer->isActive();
    if (hasActiveDragTimer) return;

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

    // 移除 m_cameraSyncSourceIdx 过滤，允许任何视窗的渲染结束都触发同步
    // 这确保鼠标滚轮等操作能正确传播到其他视窗

    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(self->m_subViews[srcIdx]);
    if (!srcRen || !srcRen->GetActiveCamera()) return;
    auto* srcCam = srcRen->GetActiveCamera();

    self->m_syncingCameras = true;
    for (int i = 0; i < self->m_subViews.size(); ++i) {
        if (i == srcIdx) continue;
        deepCopyCameraToAllRenderers(self->m_subViews[i], srcCam);
        auto* dv = self->m_subViews[i];
        if (dv && dv->getVtkWidget()) {
            auto* dstRw = dv->getVtkWidget()->renderWindow();
            if (dstRw) dstRw->Modified();
            dv->getVtkWidget()->update();
        }
    }
    self->syncPivotFromView(srcIdx);
    self->m_syncingCameras = false;
}

void vtkComparativeViewWidget::interactionCallback(
        vtkObject* caller, unsigned long eid, void* clientData, void*) {
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

    switch (eid) {
        case vtkCommand::StartInteractionEvent:
            self->m_interacting = true;
            self->m_interactionSourceIdx = srcIdx;
            self->m_cameraSyncSourceIdx = srcIdx;
            self->setInteractiveLOD(true);
            self->startInteractionTimer();
            return;

        case vtkCommand::EndInteractionEvent:
            self->m_interacting = false;
            self->m_interactionSourceIdx = -1;
            self->m_cameraSyncSourceIdx = 0;
            self->stopInteractionTimer();
            self->setInteractiveLOD(false);
            self->onSubViewInteraction(srcIdx, true);
            return;

        case vtkCommand::InteractionEvent:
            self->onSubViewInteraction(srcIdx, false);
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

    m_syncingCameras = true;
    for (int i = 0; i < m_subViews.size(); ++i) {
        if (i == viewIdx) continue;
        deepCopyCameraToAllRenderers(m_subViews[i], srcCam);
    }
    syncPivotFromView(viewIdx);
    m_syncingCameras = false;

    if (!renderOthers) return;

    scheduleCameraSyncRender();
}

void vtkComparativeViewWidget::startInteractionTimer() {
    if (m_closing) return;
    if (!m_cameraSyncRenderTimer) {
        m_cameraSyncRenderTimer = new QTimer(this);
        connect(m_cameraSyncRenderTimer, &QTimer::timeout, this,
                &vtkComparativeViewWidget::performCameraSyncRender);
    }
    // Repeating 16ms (~60 Hz) timer drives rendering while the user drags.
    // A Qt 0ms single-shot timer would be starved by the continuous stream
    // of mouse events — it only fires when the event queue is idle, which
    // may never happen during a fast drag.  A repeating timer fires
    // regardless of pending events, giving reliable visual feedback.
    m_cameraSyncRenderTimer->setSingleShot(false);
    if (!m_cameraSyncRenderTimer->isActive())
        m_cameraSyncRenderTimer->start(16);
}

void vtkComparativeViewWidget::stopInteractionTimer() {
    if (m_cameraSyncRenderTimer && m_cameraSyncRenderTimer->isActive()) {
        m_cameraSyncRenderTimer->stop();
    }
}

void vtkComparativeViewWidget::scheduleCameraSyncRender() {
    if (m_closing) return;
    // During continuous drag the repeating interaction timer handles
    // rendering, so skip redundant scheduling.
    if (m_interacting && m_cameraSyncRenderTimer &&
        m_cameraSyncRenderTimer->isActive())
        return;
    if (!m_cameraSyncRenderTimer) {
        m_cameraSyncRenderTimer = new QTimer(this);
        connect(m_cameraSyncRenderTimer, &QTimer::timeout, this,
                &vtkComparativeViewWidget::performCameraSyncRender);
    }
    m_cameraSyncRenderTimer->setSingleShot(true);
    if (!m_cameraSyncRenderTimer->isActive())
        m_cameraSyncRenderTimer->start(0);
}

void vtkComparativeViewWidget::performCameraSyncRender() {
    if (m_closing || !isVisible() || m_subViews.isEmpty()) return;

    // During interaction, re-sync camera from the active source view so
    // the latest mouse position is captured even if InteractionEvent
    // callbacks were coalesced by Qt.
    if (m_interacting && m_interactionSourceIdx >= 0 &&
        m_interactionSourceIdx < m_subViews.size()) {
        auto* srcRen =
                vtkComparativeViewWidget::getSceneRenderer(m_subViews[m_interactionSourceIdx]);
        if (srcRen && srcRen->GetActiveCamera()) {
            auto* srcCam = srcRen->GetActiveCamera();
            for (int i = 0; i < m_subViews.size(); ++i) {
                if (i == m_interactionSourceIdx) continue;
                deepCopyCameraToAllRenderers(m_subViews[i], srcCam);
            }
        }
    }

    for (auto* view : m_subViews) {
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (rw) rw->Modified();
        view->getVtkWidget()->update();
    }
}

void vtkComparativeViewWidget::setInteractiveLOD(bool enable) {
    // ParaView pattern: enable LOD actors during interaction for faster
    // frame rates, disable on EndInteraction for full-quality still render.
    // Also adjusts DesiredUpdateRate (ParaView: 5.0 interactive / 0.002 still).
    const double rate = enable ? 5.0 : 0.002;
    for (auto* view : m_subViews) {
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw) continue;

        auto* iren = rw->GetInteractor();
        if (iren) {
            iren->SetDesiredUpdateRate(rate);
        }

        auto* renderers = rw->GetRenderers();
        if (!renderers) continue;
        renderers->InitTraversal();
        while (auto* ren = renderers->GetNextItem()) {
            VtkRendering::SetLODEnabledForRenderer(ren, enable);
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
        if (!src) return nullptr;
        auto clone = vtkSmartPointer<vtkActor>::New();
        clone->ShallowCopy(src);
        if (auto* srcMapper = vtkDataSetMapper::SafeDownCast(src->GetMapper())) {
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

    // SELECTED uses selectionOverlayUpdated; highlightActorAdded is hover/preselect.
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
    m_hlActorRemovedConn = connect(
            hl, &cvSelectionHighlighter::highlightActorRemoved, this,
            [this, removeClones, safeUpdate](vtkActor* actor) {
                if (m_closing) return;
                removeClones(actor);
                safeUpdate();
            });
    m_hlClearedConn = connect(hl, &cvSelectionHighlighter::highlightsCleared,
                              this, [this]() {
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

    auto applyOverlayToSubViews =
            [this](vtkPolyData* poly, int kind) {
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
                                static_cast<cvSelectionHighlighter::
                                                    SelectionOverlayKind>(
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
        m_hlSelectionFinishedConn = connect(
                ctrl, &cvSelectionToolController::selectionFinished, this,
                [this](const cvSelectionData&) {
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

    // 对所有类型都刷新 entity 列表
    m_entityCombo->blockSignals(true);
    m_entityCombo->clear();

    auto entities = m_entityListProvider();
    ccHObject* currentEntity = nullptr;
    int currentIndex = m_entityCombo->currentIndex();
    if (currentIndex >= 0) {
        currentEntity = reinterpret_cast<ccHObject*>(
                m_entityCombo->itemData(currentIndex).value<quintptr>());
    }

    int matchIdx = -1;
    int idx = 0;
    for (auto* entity : entities) {
        if (!entity) continue;
        m_entityCombo->addItem(entity->getName(),
                              QVariant::fromValue<quintptr>(reinterpret_cast<quintptr>(entity)));
        if (entity == currentEntity) matchIdx = idx;
        ++idx;
    }

    m_entityCombo->setCurrentIndex(matchIdx >= 0 ? matchIdx : 0);
    m_entityCombo->blockSignals(false);
}

void vtkComparativeViewWidget::setInitialEntity(ccHObject* entity) {
    m_initialEntity = entity;
}

void vtkComparativeViewWidget::createChartSubViews() {
    vtkChartView::ChartType chartType =
            (m_type == BAR_CHART) ? vtkChartView::BAR_CHART
                                  : vtkChartView::LINE_CHART;

    ccHObject* initEntity = m_initialEntity;
    if (!initEntity && m_entityListProvider) {
        auto entities = m_entityListProvider();
        if (!entities.empty()) initEntity = entities.first();
    }

    QList<vtkChartView*> newCharts;
    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            auto* chart = new vtkChartView(chartType, this);
            chart->setCompactMode(true);
            chart->setSizePolicy(QSizePolicy::Expanding,
                                 QSizePolicy::Expanding);
            if (m_entityListProvider) {
                chart->setEntityListProvider(m_entityListProvider);
            }
            if (initEntity) {
                chart->setEntity(initEntity);
            }
            chart->setCompactMode(true);
            m_gridLayout->addWidget(chart, r, c);
            m_gridLayout->setRowStretch(r, 1);
            m_gridLayout->setColumnStretch(c, 1);
            m_subWidgets.append(chart);
            newCharts.append(chart);
            emit subViewCreated(chart);
        }
    }

    refreshEntityCombo();

    QTimer::singleShot(200, this, [this, newCharts, initEntity]() {
        ccHObject* entity = initEntity;
        if (!entity && m_entityListProvider) {
            auto entities = m_entityListProvider();
            if (!entities.isEmpty()) entity = entities.first();
        }
        if (!entity) return;
        for (auto* chart : newCharts) {
            if (!chart) continue;
            chart->setEntity(entity);
            chart->setCompactMode(true);
        }
    });
}

void vtkComparativeViewWidget::buildToolbar() {
    m_toolbar = new QWidget(this);
    auto* lay = new QHBoxLayout(m_toolbar);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(2);

    auto* dimLabel = new QLabel(tr("<b>Grid:</b>"), m_toolbar);
    lay->addWidget(dimLabel);

    m_rowSpin = new QSpinBox(m_toolbar);
    m_rowSpin->setRange(1, 8);
    m_rowSpin->setValue(m_rows);
    m_rowSpin->setPrefix(tr("R:"));
    lay->addWidget(m_rowSpin);

    auto* xLabel = new QLabel(tr("x"), m_toolbar);
    lay->addWidget(xLabel);

    m_colSpin = new QSpinBox(m_toolbar);
    m_colSpin->setRange(1, 8);
    m_colSpin->setValue(m_cols);
    m_colSpin->setPrefix(tr("C:"));
    lay->addWidget(m_colSpin);

    auto* spLabel = new QLabel(tr("Sp:"), m_toolbar);
    lay->addWidget(spLabel);

    auto* spacingSpin = new QSpinBox(m_toolbar);
    spacingSpin->setRange(0, 20);
    spacingSpin->setValue(m_spacing);
    spacingSpin->setToolTip(tr("Grid spacing (ParaView Spacing property)"));
    lay->addWidget(spacingSpin);

    connect(spacingSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::setSpacing);

    lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));

    auto* cueLabel = new QLabel(tr("<b>Cue:</b>"), m_toolbar);
    lay->addWidget(cueLabel);

    m_cueParamCombo = new QComboBox(m_toolbar);
    m_cueParamCombo->addItem(tr("None"), 0);
    m_cueParamCombo->addItem(tr("Azimuth"), 1);
    m_cueParamCombo->addItem(tr("Elevation"), 2);
    m_cueParamCombo->addItem(tr("Opacity"), 3);
    m_cueParamCombo->addItem(tr("Zoom"), 4);
    m_cueParamCombo->addItem(tr("Point Size"), 5);
    m_cueParamCombo->addItem(tr("Representation"), 6);
    m_cueParamCombo->addItem(tr("Line Width"), 7);
    m_cueParamCombo->setToolTip(
            tr("Parameter to sweep across sub-views "
               "(ParaView vtkPVComparativeAnimationCue)"));
    lay->addWidget(m_cueParamCombo);

    m_cueModeCombo = new QComboBox(m_toolbar);
    m_cueModeCombo->addItem(tr("X-Range"), 0);
    m_cueModeCombo->addItem(tr("Y-Range"), 1);
    m_cueModeCombo->addItem(tr("T-Range"), 2);
    m_cueModeCombo->setToolTip(
            tr("Sweep mode: X=vary along columns, Y=vary along rows, "
               "T=vary across all (ParaView XRANGE/YRANGE/TRANGE)"));
    lay->addWidget(m_cueModeCombo);

    auto* minLabel = new QLabel(tr("Min:"), m_toolbar);
    lay->addWidget(minLabel);

    m_cueMinSpin = new QDoubleSpinBox(m_toolbar);
    m_cueMinSpin->setRange(-360, 360);
    m_cueMinSpin->setDecimals(1);
    m_cueMinSpin->setValue(0.0);
    m_cueMinSpin->setMaximumWidth(70);
    lay->addWidget(m_cueMinSpin);

    auto* maxLabel = new QLabel(tr("Max:"), m_toolbar);
    lay->addWidget(maxLabel);

    m_cueMaxSpin = new QDoubleSpinBox(m_toolbar);
    m_cueMaxSpin->setRange(-360, 360);
    m_cueMaxSpin->setDecimals(1);
    m_cueMaxSpin->setValue(90.0);
    m_cueMaxSpin->setMaximumWidth(70);
    lay->addWidget(m_cueMaxSpin);

    auto* playBtn = new QPushButton(tr("Apply"), m_toolbar);
    playBtn->setToolTip(tr("Apply parameter sweep to sub-views"));
    lay->addWidget(playBtn);

    lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));

    m_overlayCheck = new QCheckBox(tr("Overlay"), m_toolbar);
    m_overlayCheck->setToolTip(
            tr("Overlay all comparisons into first view "
               "(ParaView OverlayAllComparisons)"));
    lay->addWidget(m_overlayCheck);

    auto* resetCamBtn = new QPushButton(tr("Reset"), m_toolbar);
    resetCamBtn->setToolTip(tr("Reset camera for all sub-views"));
    lay->addWidget(resetCamBtn);
    connect(resetCamBtn, &QPushButton::clicked, this, [this]() {
        m_baselineCamera.valid = false;
        for (auto* v : m_subViews) {
            if (v) v->resetCamera();
        }
        forceRenderAllSubViews();
    });

    auto* refreshBtn = new QPushButton(tr("Refresh"), m_toolbar);
    refreshBtn->setToolTip(
            tr("Re-load geometry from source view into all sub-views"));
    lay->addWidget(refreshBtn);
    connect(refreshBtn, &QPushButton::clicked,
            this, &vtkComparativeViewWidget::refreshSubViews);

    m_syncCamCheck = new QCheckBox(tr("Sync"), m_toolbar);
    m_syncCamCheck->setChecked(m_cameraLinkEnabled);
    m_syncCamCheck->setToolTip(tr("Synchronize cameras across all sub-views"));
    lay->addWidget(m_syncCamCheck);
    connect(m_syncCamCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_cameraLinkEnabled = on;
        if (on) {
            if (m_cueParamCombo &&
                m_cueParamCombo->currentData().toInt() != 0) {
                m_cueParamCombo->setCurrentIndex(0);
            }
            syncRepresentationsFromFirst();
            syncCamerasFromFirst();
            forceRenderAllSubViews();
        } else {
            removeCameraLink();
        }
    });

    // 对所有类型都启用 entity 选择（不仅是图表类型）
    if (true) {  // 原来是 if (m_type != RENDER)
        lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));
        auto* showLabel = new QLabel(tr("<b>Showing:</b>"), m_toolbar);
        lay->addWidget(showLabel);
        m_entityCombo = new QComboBox(m_toolbar);
        m_entityCombo->setMinimumWidth(120);
        m_entityCombo->setToolTip(tr("Entity displayed in all sub-views"));
        lay->addWidget(m_entityCombo);

        connect(m_entityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int idx) {
                if (idx < 0 || !m_entityCombo) return;
                auto* entity = reinterpret_cast<ccHObject*>(
                        m_entityCombo->itemData(idx).value<quintptr>());
                if (!entity) return;

                // 更新所有子视窗显示的实体
                if (m_type == RENDER) {
                    bool anySuccess = false;
                    for (auto* view : m_subViews) {
                        if (auto* glView = dynamic_cast<vtkGLView*>(view)) {
                            // 清除旧表示
                            ecvRepresentationManager::instance()
                                    .removeRepresentationsForView(glView);

                            // 添加新表示（确保 renderer 存在）
                            auto* ren = vtkComparativeViewWidget::getSceneRenderer(glView);
                            if (!ren) continue;

                            if (auto* rep = ecvRepresentationManager::instance()
                                    .ensureRepresentation(entity, glView)) {
                                rep->setVisible(true);
                                ren->Modified();  // 标记需要重新渲染
                                anySuccess = true;
                            }
                        }
                    }

                    // 如果至少有一个视窗成功添加了表示，则同步并刷新
                    if (anySuccess) {
                        syncRepresentationsFromFirst();
                        m_needsCameraReset = true;  // 标记需要 ResetCamera
                        scheduleSubViewRefresh(true);  // 强制刷新
                    }
                } else {
                    // 图表类型：更新 chart 的 entity
                    for (auto* w : m_subWidgets) {
                        auto* chart = qobject_cast<vtkChartView*>(w);
                        if (chart) chart->setEntity(entity);
                    }

                    forceRenderAllSubViews();
                }

                if (m_statusLabel)
                    m_statusLabel->setText(tr("Synced: %1").arg(entity->getName()));
            });
    }

    // Screenshot 功能在 Comparative View 中冗余（主窗口已有），已移除按钮
    // auto* screenshotBtn = new QPushButton(tr("Screenshot"), m_toolbar);
    // screenshotBtn->setToolTip(
    //         tr("Export stitched screenshot of all sub-views"));
    // lay->addWidget(screenshotBtn);

    m_statusLabel = new QLabel(m_toolbar);
    m_statusLabel->setContentsMargins(4, 0, 4, 0);
    lay->addWidget(m_statusLabel);
    lay->addStretch(1);

    connect(m_rowSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::onDimensionChanged);
    connect(m_colSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &vtkComparativeViewWidget::onDimensionChanged);
    connect(m_cueParamCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            &vtkComparativeViewWidget::onCueParameterChanged);
    connect(playBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::onPlayCue);
    connect(m_overlayCheck, &QCheckBox::toggled, this,
            &vtkComparativeViewWidget::onToggleOverlay);
    // connect(screenshotBtn, &QPushButton::clicked, this,
    //         &vtkComparativeViewWidget::onExportScreenshot);  // 已移除按钮
}

void vtkComparativeViewWidget::onDimensionChanged() {
    int r = m_rowSpin ? m_rowSpin->value() : m_rows;
    int c = m_colSpin ? m_colSpin->value() : m_cols;
    setDimensions(r, c);
    m_statusLabel->setText(
            tr("%1x%2 = %3 views").arg(r).arg(c).arg(r * c));
}

void vtkComparativeViewWidget::onCueParameterChanged(int index) {
    int cueParam = m_cueParamCombo ? m_cueParamCombo->currentData().toInt() : 0;

    if (cueParam == 0) {
        m_cameraLinkEnabled = true;
        if (m_syncCamCheck) {
            QSignalBlocker blk(m_syncCamCheck);
            m_syncCamCheck->setChecked(true);
        }
        if (m_baselineCamera.valid) {
            for (auto* v : m_subViews) restoreBaselineCamera(v);
        }
        syncCamerasFromFirst();
    }

    if (m_cueMinSpin && m_cueMaxSpin) {
        switch (cueParam) {
            case 1:
            case 2:
                m_cueMinSpin->setRange(-360, 360);
                m_cueMaxSpin->setRange(-360, 360);
                m_cueMinSpin->setValue(0.0);
                m_cueMaxSpin->setValue(90.0);
                break;
            case 3:
                m_cueMinSpin->setRange(0, 100);
                m_cueMaxSpin->setRange(0, 100);
                m_cueMinSpin->setValue(20.0);
                m_cueMaxSpin->setValue(100.0);
                break;
            case 4:
                m_cueMinSpin->setRange(0.1, 10.0);
                m_cueMaxSpin->setRange(0.1, 10.0);
                m_cueMinSpin->setValue(0.5);
                m_cueMaxSpin->setValue(2.0);
                break;
            case 5:
                m_cueMinSpin->setRange(1, 20);
                m_cueMaxSpin->setRange(1, 20);
                m_cueMinSpin->setValue(1.0);
                m_cueMaxSpin->setValue(10.0);
                break;
            case 6:
                m_cueMinSpin->setRange(0, 2);
                m_cueMaxSpin->setRange(0, 2);
                m_cueMinSpin->setValue(0.0);
                m_cueMaxSpin->setValue(2.0);
                break;
            case 7:
                m_cueMinSpin->setRange(0.5, 10.0);
                m_cueMaxSpin->setRange(0.5, 10.0);
                m_cueMinSpin->setValue(1.0);
                m_cueMaxSpin->setValue(5.0);
                break;
            default:
                m_cueMinSpin->setRange(-360, 360);
                m_cueMaxSpin->setRange(-360, 360);
                break;
        }
    }
    if (m_statusLabel) {
        m_statusLabel->setText(
                tr("Cue: %1").arg(m_cueParamCombo->itemText(index)));
    }

    // Cue sweep runs only when user clicks Apply (onPlayCue), not on combo edits.
}

void vtkComparativeViewWidget::onToggleOverlay(bool checked) {
    m_overlayMode = checked;
    if (m_subWidgets.size() <= 1) return;

    if (m_type == RENDER && m_subViews.size() > 1) {
        vtkGLView* firstView = m_subViews.first();

        if (checked) {
            auto* dstRen = vtkComparativeViewWidget::getSceneRenderer(firstView);
            for (int i = 1; i < m_subViews.size(); ++i) {
                auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(m_subViews[i]);
                if (srcRen && dstRen) {
                    auto* actors = srcRen->GetActors();
                    if (actors) {
                        actors->InitTraversal();
                        vtkActor* a = nullptr;
                        while ((a = actors->GetNextActor()))
                            dstRen->AddActor(a);
                    }
                }
                if (i < m_subWidgets.size())
                    m_subWidgets[i]->setVisible(false);
            }
            m_gridLayout->setColumnStretch(0, 1);
            m_gridLayout->setRowStretch(0, 1);
        } else {
            auto* dstRen = vtkComparativeViewWidget::getSceneRenderer(firstView);
            if (dstRen) {
                for (int i = 1; i < m_subViews.size(); ++i) {
                    auto* srcRen = vtkComparativeViewWidget::getSceneRenderer(m_subViews[i]);
                    if (!srcRen) continue;
                    auto* actors = srcRen->GetActors();
                    if (!actors) continue;
                    actors->InitTraversal();
                    vtkActor* a = nullptr;
                    while ((a = actors->GetNextActor()))
                        dstRen->RemoveActor(a);
                }
            }
            for (int i = 0; i < m_subWidgets.size(); ++i) {
                m_subWidgets[i]->setVisible(true);
            }
        }
    } else {
        if (checked) {
            for (int i = 1; i < m_subWidgets.size(); ++i)
                m_subWidgets[i]->setVisible(false);
            m_gridLayout->setColumnStretch(0, 1);
            m_gridLayout->setRowStretch(0, 1);
        } else {
            for (int i = 0; i < m_subWidgets.size(); ++i)
                m_subWidgets[i]->setVisible(true);
        }
    }

    forceRenderAllSubViews();

    if (m_statusLabel) {
        m_statusLabel->setText(checked ? tr("Overlay mode ON")
                                       : tr("Overlay mode OFF"));
    }
}

void vtkComparativeViewWidget::onExportScreenshot() {
    if (m_closing || m_subViews.isEmpty()) return;

    QString path = QFileDialog::getSaveFileName(
            this, tr("Save Comparative Screenshot"), QString(),
            tr("PNG (*.png);;JPEG (*.jpg)"));
    if (path.isEmpty()) return;

    QApplication::setOverrideCursor(Qt::WaitCursor);

    for (auto* view : m_subViews) {
        if (view && view->getVtkWidget()) {
            auto* w = view->getVtkWidget();
            if (auto* rw = w->renderWindow()) {
                rw->Modified();
            }
            w->update();
        }
    }

    QTimer::singleShot(150, this, [this, path]() {
        for (auto* view : m_subViews) {
            if (view && view->getVtkWidget()) {
                view->redraw(false, true);
            }
        }
        QTimer::singleShot(50, this, [this, path]() {
            captureScreenshotsAsync(path);
        });
    });
}

void vtkComparativeViewWidget::captureScreenshotsAsync(const QString& path) {
    const int spacing = m_spacing;
    const int totalWidth = width();
    const int totalHeight = height();
    const int cellW = totalWidth / m_cols;
    const int cellH = totalHeight / m_rows;

    QImage composite(totalWidth, totalHeight, QImage::Format_RGB32);
    composite.fill(Qt::black);

    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* view = m_subViews[i];
        QWidget* subWidget = (i < m_subWidgets.size()) ? m_subWidgets[i] : nullptr;
        if (!view || !subWidget || !subWidget->isVisible()) continue;

        int row = i / m_cols;
        int col = i % m_cols;
        int x = col * cellW + spacing;
        int y = row * cellH + spacing;
        int w = cellW - 2 * spacing;
        int h = cellH - 2 * spacing;

        if (w <= 0 || h <= 0) continue;

        // 使用 QWidget::grab() 作为主要方式（Qt原生，无 GL 冲突）
        QPixmap subPixmap = subWidget->grab(subWidget->rect());
        if (!subPixmap.isNull()) {
            QPainter painter(&composite);
            painter.drawPixmap(x, y, w, h,
                              subPixmap.scaled(w, h,
                                              Qt::IgnoreAspectRatio,
                                              Qt::FastTransformation));
            painter.end();
        }
    }

    // 保存文件
    bool saved = composite.save(path);
    QApplication::restoreOverrideCursor();

    if (saved) {
        if (m_statusLabel) {
            QFileInfo fi(path);
            m_statusLabel->setText(
                    tr("Saved: %1 (%2x%3)")
                            .arg(fi.fileName())
                            .arg(composite.width())
                            .arg(composite.height()));
        }
    } else {
        CVLog::Warning("[CompView] Failed to save screenshot: %1", path);
    }
}

void vtkComparativeViewWidget::onPlayCue() {
    applyCueToSubViews();
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

void vtkComparativeViewWidget::applyCueToSubViews() {
    if (!m_cueParamCombo || m_subWidgets.isEmpty()) return;

    int cueParam = m_cueParamCombo->currentData().toInt();
    if (cueParam <= 0) return;

    bool isCameraCue = (cueParam == 1 || cueParam == 2 || cueParam == 4);
    if (isCameraCue) {
        m_cameraLinkEnabled = false;
        if (m_syncCamCheck) {
            QSignalBlocker blk(m_syncCamCheck);
            m_syncCamCheck->setChecked(false);
        }
        removeCameraLink();
    }

    if (!m_baselineCamera.valid) saveBaselineCamera();

    int cueMode = m_cueModeCombo ? m_cueModeCombo->currentData().toInt() : 2;
    double minVal = m_cueMinSpin ? m_cueMinSpin->value() : 0.0;
    double maxVal = m_cueMaxSpin ? m_cueMaxSpin->value() : 90.0;

    int dx = m_cols;
    int dy = m_rows;
    int applied = 0;

    for (int y = 0; y < dy; ++y) {
        for (int x = 0; x < dx; ++x) {
            int index = y * dx + x;
            if (index >= m_subWidgets.size()) break;

            double value = minVal;
            switch (cueMode) {
                case 0:  // XRANGE
                    value = (dx > 1)
                            ? minVal + x * (maxVal - minVal) / (dx - 1)
                            : minVal;
                    break;
                case 1:  // YRANGE
                    value = (dy > 1)
                            ? minVal + y * (maxVal - minVal) / (dy - 1)
                            : minVal;
                    break;
                case 2:  // TRANGE
                default:
                    value = (dx * dy > 1)
                            ? minVal + (y * dx + x) * (maxVal - minVal) /
                                               (dx * dy - 1)
                            : minVal;
                    break;
            }

            if (m_type == RENDER) {
                vtkGLView* view = nullptr;
                if (index < m_subViews.size()) view = m_subViews[index];
                if (!view) {
                    view = qobject_cast<vtkGLView*>(
                            m_subWidgets[index]->findChild<vtkGLView*>());
                }
                if (!view) {
                    view = dynamic_cast<vtkGLView*>(
                            static_cast<QObject*>(m_subWidgets[index]));
                }
                if (view) {
                    restoreBaselineCamera(view);
                }
                auto* ren = vtkComparativeViewWidget::getSceneRenderer(view);
                if (ren) {
                    auto* cam = ren->GetActiveCamera();
                    if (cam) {
                        switch (cueParam) {
                            case 1:  // Azimuth
                                cam->Azimuth(value);
                                break;
                            case 2:  // Elevation
                                cam->Elevation(value);
                                break;
                            case 4: {  // Zoom
                                double zf = std::pow(2.0, value / 45.0);
                                cam->Zoom(zf);
                                break;
                            }
                            default:
                                break;
                        }
                        ren->ResetCameraClippingRange();
                    }

                    if (cueParam == 3 || cueParam == 5 ||
                        cueParam == 6 || cueParam == 7) {
                        auto* actors = ren->GetActors();
                        if (actors) {
                            actors->InitTraversal();
                            vtkActor* actor = nullptr;
                            while ((actor = actors->GetNextActor())) {
                                if (cueParam == 3) {
                                    double opacity =
                                            qBound(0.0, value / 100.0, 1.0);
                                    actor->GetProperty()->SetOpacity(opacity);
                                }
                                if (cueParam == 5) {
                                    int ptSz = qBound(1, static_cast<int>(value), 20);
                                    actor->GetProperty()->SetPointSize(ptSz);
                                }
                                if (cueParam == 6) {
                                    int repr = qBound(0, static_cast<int>(std::round(value)), 2);
                                    actor->GetProperty()->SetRepresentation(repr);
                                    actor->GetProperty()->SetEdgeVisibility(repr == 2 ? 1 : 0);
                                }
                                if (cueParam == 7) {
                                    float lw = qBound(0.5f, static_cast<float>(value), 10.0f);
                                    actor->GetProperty()->SetLineWidth(lw);
                                }
                            }
                        }
                        ren->Modified();
                    }
                    ++applied;
                }
            }
        }
    }

    if (m_type != RENDER) {
        for (int idx = 0; idx < m_subWidgets.size(); ++idx) {
            auto* chartView = qobject_cast<vtkChartView*>(m_subWidgets[idx]);
            if (!chartView) continue;

            int y = idx / dx;
            int x = idx % dx;
            double value = minVal;
            switch (cueMode) {
                case 0:
                    value = (dx > 1) ? minVal + x * (maxVal - minVal) / (dx - 1) : minVal;
                    break;
                case 1:
                    value = (dy > 1) ? minVal + y * (maxVal - minVal) / (dy - 1) : minVal;
                    break;
                case 2:
                default:
                    value = (dx * dy > 1)
                            ? minVal + idx * (maxVal - minVal) / (dx * dy - 1)
                            : minVal;
                    break;
            }
            switch (cueParam) {
                case 1:
                case 2: {
                    double scale = qBound(0.1, value / 45.0, 10.0);
                    chartView->setYAxisScale(scale);
                    break;
                }
                case 3: {
                    double opacity = qBound(0.01, value / 100.0, 1.0);
                    chartView->setPlotOpacity(opacity);
                    break;
                }
                case 4: {
                    double zoom = qBound(0.1, value, 10.0);
                    chartView->setXAxisScale(zoom);
                    chartView->setYAxisScale(zoom);
                    break;
                }
                case 7: {
                    double lw = qBound(0.5, value, 10.0);
                    chartView->setPlotOpacity(1.0);
                    (void)lw;
                    break;
                }
                default:
                    break;
            }
            ++applied;
        }
    }

    for (auto* v : m_subViews) {
        safeRenderWindow(v);
    }

    Q_UNUSED(isCameraCue);

    if (m_statusLabel) {
        m_statusLabel->setText(
                tr("Applied %1 [%2..%3] to %4/%5 views")
                        .arg(m_cueParamCombo->currentText())
                        .arg(minVal, 0, 'f', 1)
                        .arg(maxVal, 0, 'f', 1)
                        .arg(applied)
                        .arg(m_subWidgets.size()));
    }
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
            deepCopyCameraToAllRenderers(view, firstRen->GetActiveCamera());
        }

        syncSubViewBackgroundFromGlobal(view);

        view->setInteractionMode(
                stripClickableItems(first->getInteractionMode()));
        view->setPickingMode(first->getPickingMode());

        disableBubbleViewForSubView(view);
    }

    syncRepresentationsFromFirst();
}
