// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "vtkComparativeViewWidget.h"

#include <CVLog.h>
#include <Tools/SelectionTools/cvSelectionHighlighter.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/vtkGLView.h>
#include <Visualization/VtkVis.h>
#include <ecvHObject.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QResizeEvent>
#include <QShowEvent>
#include <QMenu>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QSpinBox>
#include <QTimer>
#include <QVBoxLayout>

#include <vtkCallbackCommand.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>
#include <vtkInteractorObserver.h>

#include <cmath>

#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkCamera.h>
#include <vtkMapper.h>
#include <vtkPolyDataMapper.h>
#include <vtkPropCollection.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkTexture.h>

static constexpr int COMPARATIVE_SPACING = 1;

static bool safeRenderWindow(vtkGLView* view, bool immediate = false) {
    if (!view) return false;
    auto* w = view->getVtkWidget();
    if (!w || !w->isVisible() || w->width() < 2 || w->height() < 2)
        return false;
    if (immediate) {
        auto* rw = w->renderWindow();
        if (rw) {
            rw->Render();
            return true;
        }
    }
    w->update();
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

vtkComparativeViewWidget::vtkComparativeViewWidget(ComparativeType type,
                                                   QWidget* parent)
    : QWidget(parent), m_type(type) {
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
    m_closing = true;
    removeCameraLink();
    disconnect(&ecvRepresentationManager::instance(), nullptr, this, nullptr);
    disconnect(&ecvViewManager::instance(), nullptr, this, nullptr);
    for (auto* sv : m_subViews) {
        if (!sv) continue;
        QWidget* w = sv->asWidget();
        if (w) w->removeEventFilter(this);
        sv->setSceneDB(nullptr);
        sv->disconnect();
    }
    m_subViews.clear();
    for (auto* w : m_subWidgets) {
        if (w) {
            w->removeEventFilter(this);
            w->hide();
        }
    }
    m_subWidgets.clear();
}

QString vtkComparativeViewWidget::title() const {
    switch (m_type) {
        case RENDER:
            return tr("Render View (Comparative)");
        case LINE_CHART:
            return tr("Line Chart View (Comparative)");
        case BAR_CHART:
            return tr("Bar Chart View (Comparative)");
    }
    return tr("Comparative View");
}

void vtkComparativeViewWidget::setSpacing(int spacing) {
    m_spacing = spacing;
    if (m_gridLayout) {
        m_gridLayout->setSpacing(spacing);
    }
}

void vtkComparativeViewWidget::setDimensions(int rows, int cols) {
    if (rows < 1 || cols < 1 || (rows == m_rows && cols == m_cols)) return;

    for (auto* w : m_subWidgets) {
        m_gridLayout->removeWidget(w);
        w->setParent(nullptr);
        w->deleteLater();
    }
    m_subWidgets.clear();
    m_subViews.clear();

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

void vtkComparativeViewWidget::setupGrid() {
    if (m_type == RENDER) {
        createRenderSubViews();
    } else {
        createChartSubViews();
    }
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
            m_gridLayout->addWidget(viewWidget, r, c);
            m_gridLayout->setRowStretch(r, 1);
            m_gridLayout->setColumnStretch(c, 1);

            m_subWidgets.append(viewWidget);
            m_subViews.append(view);
            if (!firstView) firstView = view;

            syncSubViewBackgroundFromGlobal(view);
            view->disableContext2DOverlay();
            disableBubbleViewForSubView(view);

            if (m_subViewInitCb) {
                m_subViewInitCb(view);
            }

            m_pendingFirstResize.insert(viewWidget);
            viewWidget->installEventFilter(this);
            emit subViewCreated(viewWidget);
        }
    }

    if (firstView && m_subViews.size() > 1) {
        if (m_sourceView) {
            firstView->setEntityBindingSource(m_sourceView);
        }
        ccHObject* sceneDB = firstView->getSceneDB();
        for (int i = 1; i < m_subViews.size(); ++i) {
            m_subViews[i]->setEntityBindingSource(firstView);
            if (sceneDB && !m_subViews[i]->getSceneDB()) {
                m_subViews[i]->setSceneDB(sceneDB);
            }
        }
        syncCamerasFromFirst();
    }

    if (m_sourceView && firstView) {
        ccHObject* srcDB = m_sourceView->getSceneDB();
        if (!firstView->getSceneDB() && srcDB) {
            firstView->setSceneDB(srcDB);
        }
        ccHObject* db = firstView->getSceneDB();
        if (db) {
            for (int i = 1; i < m_subViews.size(); ++i) {
                if (!m_subViews[i]->getSceneDB())
                    m_subViews[i]->setSceneDB(db);
            }
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
        QTimer::singleShot(50, this, [this]() {
            if (m_closing || !isVisible()) return;
            copyActorsAcrossSubViews();
            forceRenderAllSubViews();
        });
    };

    connect(&ecvRepresentationManager::instance(),
            &ecvRepresentationManager::representationAdded, this, onRepChanged);
    connect(&ecvRepresentationManager::instance(),
            &ecvRepresentationManager::representationChanged, this,
            onRepChanged);

    forceRenderAllSubViews();

    static const int kRetryDelays[] = {200, 500, 1000, 2000, 4000};
    for (int d : kRetryDelays) {
        QTimer::singleShot(d, this, [this]() {
            if (m_closing || !isVisible()) return;
            copyActorsAcrossSubViews();
            if (m_subViews.size() > 1) syncCamerasFromFirst();
            forceRenderAllSubViews();
        });
    }

    connect(&ecvViewManager::instance(),
            &ecvViewManager::pointIndicesSelected, this,
            [this](ccHObject*, const QSet<unsigned>&) {
                QTimer::singleShot(100, this, [this]() {
                    if (m_closing || !isVisible()) return;
                    copyActorsAcrossSubViews();
                    forceRenderAllSubViews();
                });
            });

    if (m_sourceView) {
        connect(m_sourceView, &vtkGLView::interactionModeChanged, this,
                &vtkComparativeViewWidget::syncInteractionModeToSubViews);
        connect(m_sourceView, &vtkGLView::pickingModeChanged, this,
                &vtkComparativeViewWidget::syncPickingModeToSubViews);
    }
}

void vtkComparativeViewWidget::setSourceView(vtkGLView* src) {
    m_sourceView = src;
}

void vtkComparativeViewWidget::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    if (!m_firstShowDone) {
        m_firstShowDone = true;
        static const int kDelays[] = {100, 300, 600, 1500, 3000};
        for (int d : kDelays) {
            QTimer::singleShot(d, this, [this]() {
                if (m_closing || !isVisible()) return;
                for (auto* view : m_subViews) {
                    if (!view) continue;
                    syncSubViewBackgroundFromGlobal(view);
                    ccHObject* root = view->getSceneDB();
                    if (root) root->setRedrawFlagRecursive(true);
                    safeRedraw(view);
                }
                copyActorsAcrossSubViews();
                if (m_subViews.size() > 1) syncCamerasFromFirst();
                for (auto* view : m_subViews) {
                    if (!view) continue;
                    auto* vtkW = view->getVtkWidget();
                    if (!vtkW || !vtkW->isVisible() ||
                        vtkW->width() < 2 || vtkW->height() < 2)
                        continue;
                    auto* rw = vtkW->renderWindow();
                    if (!rw) continue;
                    auto* ren = rw->GetRenderers()
                            ? rw->GetRenderers()->GetFirstRenderer()
                            : nullptr;
                    if (ren) {
                        ren->ResetCamera();
                        ren->ResetCameraClippingRange();
                    }
                    vtkW->update();
                }
            });
        }
        installCameraLink();
    }
}

void vtkComparativeViewWidget::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
}

bool vtkComparativeViewWidget::eventFilter(QObject* obj, QEvent* event) {
    if (event->type() == QEvent::ContextMenu) return true;
    if (event->type() == QEvent::MouseButtonPress) {
        emit clicked();
    }
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
                        ccHObject* root = view->getSceneDB();
                        if (root) root->setRedrawFlagRecursive(true);
                        safeRedraw(view);
                        auto* vtkW = view->getVtkWidget();
                        if (!vtkW || !vtkW->isVisible() ||
                            vtkW->width() < 2 || vtkW->height() < 2)
                            break;
                        auto* rw = vtkW->renderWindow();
                        if (!rw) break;
                        auto* ren = rw->GetRenderers()
                                ? rw->GetRenderers()->GetFirstRenderer()
                                : nullptr;
                        if (ren) {
                            ren->ResetCamera();
                            ren->ResetCameraClippingRange();
                        }
                        vtkW->update();
                        break;
                    }
                });
            }
        }
    }
    return QWidget::eventFilter(obj, event);
}

void vtkComparativeViewWidget::forceRenderAllSubViews() {
    if (m_closing || !isVisible()) return;
    for (auto* view : m_subViews) {
        if (!view) continue;
        QWidget* w = view->asWidget();
        if (w && isVisible()) {
            w->show();
            w->update();
        }
        view->setAutoPickPivotAtCenter(false);
        syncSubViewBackgroundFromGlobal(view);
        ccHObject* root = view->getSceneDB();
        if (root) root->setRedrawFlagRecursive(true);
    }

    QTimer::singleShot(100, this, [this]() {
        if (m_closing || !isVisible()) return;
        for (auto* view : m_subViews) safeRedraw(view);

        copyActorsAcrossSubViews();

        if (!m_subViews.isEmpty()) {
            auto* first = m_subViews.first();
            if (first && first->getVtkWidget()) {
                auto* vtkW = first->getVtkWidget();
                if (vtkW->isVisible() && vtkW->width() >= 2 &&
                    vtkW->height() >= 2) {
                    auto* rw = vtkW->renderWindow();
                    if (rw && rw->GetRenderers()) {
                        auto* ren = rw->GetRenderers()->GetFirstRenderer();
                        if (ren) {
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
                            }
                            vtkW->update();
                        }
                    }
                }
            }
        }

        if (m_subViews.size() > 1) syncCamerasFromFirst();
        for (int i = 1; i < m_subViews.size(); ++i)
            safeRenderWindow(m_subViews[i]);
    });
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

void vtkComparativeViewWidget::copyActorsAcrossSubViews() {
    if (m_closing || m_subViews.isEmpty()) return;

    // Each sub-view must render entities through its own VtkVis / OpenGL context.
    // GL textures cannot be shared by cloning actors from another viewport.
    for (auto* view : m_subViews) {
        if (!view) continue;
        ccHObject* root = view->getSceneDB();
        if (root) root->setRedrawFlagRecursive(true);
        safeRedraw(view);
    }

    // Each sub-view owns a separate OpenGL context. Overlays (selection
    // highlights, etc.) must be created per view via safeRedraw — never
    // share vtkMapper / vtkTexture pointers across contexts (GPU crash).
}

void vtkComparativeViewWidget::syncCamerasFromFirst() {
    if (m_closing || m_subViews.size() < 2) return;
    vtkGLView* first = m_subViews.first();
    if (!first || !first->getVtkWidget()) return;
    auto* srcRw = first->getVtkWidget()->renderWindow();
    if (!srcRw || !srcRw->GetRenderers()) return;
    auto* srcRen = srcRw->GetRenderers()->GetFirstRenderer();
    if (!srcRen) return;
    auto* srcCam = srcRen->GetActiveCamera();
    if (!srcCam) return;

    for (int i = 1; i < m_subViews.size(); ++i) {
        auto* dstView = m_subViews[i];
        if (!dstView || !dstView->getVtkWidget()) continue;
        auto* dstRw = dstView->getVtkWidget()->renderWindow();
        if (!dstRw || !dstRw->GetRenderers()) continue;
        auto* dstRen = dstRw->GetRenderers()->GetFirstRenderer();
        if (dstRen && dstRen->GetActiveCamera()) {
            dstRen->GetActiveCamera()->DeepCopy(srcCam);
        }
    }

    installCameraLink();
}

void vtkComparativeViewWidget::installCameraLink() {
    if (!m_cameraObservers.empty() || m_subViews.size() < 2) return;

    for (int i = 0; i < m_subViews.size(); ++i) {
        auto* view = m_subViews[i];
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw) continue;
        auto* interactor = rw->GetInteractor();
        if (!interactor) continue;
        auto* style = interactor->GetInteractorStyle();
        if (!style) continue;

        auto cb = vtkSmartPointer<vtkCallbackCommand>::New();
        cb->SetClientData(this);
        cb->SetCallback(&vtkComparativeViewWidget::interactionCallback);

        unsigned long tag =
                style->AddObserver(vtkCommand::EndInteractionEvent, cb);
        m_cameraObservers.push_back({style, tag});
    }
}

void vtkComparativeViewWidget::removeCameraLink() {
    for (auto& obs : m_cameraObservers) {
        if (obs.observed) obs.observed->RemoveObserver(obs.tag);
    }
    m_cameraObservers.clear();
}

void vtkComparativeViewWidget::interactionCallback(
        vtkObject* caller, unsigned long eid, void* clientData, void*) {
    if (eid != vtkCommand::EndInteractionEvent) return;
    auto* self = static_cast<vtkComparativeViewWidget*>(clientData);
    if (!self || self->m_closing || self->m_syncingCameras ||
        !self->m_cameraLinkEnabled)
        return;

    for (int i = 0; i < self->m_subViews.size(); ++i) {
        auto* view = self->m_subViews[i];
        if (!view || !view->getVtkWidget()) continue;
        auto* rw = view->getVtkWidget()->renderWindow();
        if (!rw || !rw->GetInteractor()) continue;
        if (rw->GetInteractor()->GetInteractorStyle() == caller) {
            self->onSubViewInteraction(i);
            return;
        }
    }
}

void vtkComparativeViewWidget::onSubViewInteraction(int viewIdx) {
    if (m_syncingCameras || m_closing || !isVisible()) return;
    if (viewIdx < 0 || viewIdx >= m_subViews.size()) return;

    auto* srcView = m_subViews[viewIdx];
    if (!srcView || !srcView->getVtkWidget()) return;
    auto* srcRw = srcView->getVtkWidget()->renderWindow();
    if (!srcRw || !srcRw->GetRenderers()) return;
    auto* srcRen = srcRw->GetRenderers()->GetFirstRenderer();
    if (!srcRen) return;
    auto* srcCam = srcRen->GetActiveCamera();
    if (!srcCam) return;

    m_syncingCameras = true;
    for (int i = 0; i < m_subViews.size(); ++i) {
        if (i == viewIdx) continue;
        auto* dstView = m_subViews[i];
        if (!dstView || !dstView->getVtkWidget()) continue;
        auto* rw = dstView->getVtkWidget()->renderWindow();
        if (!rw || !rw->GetRenderers()) continue;
        auto* dstRen = rw->GetRenderers()->GetFirstRenderer();
        if (!dstRen) continue;
        auto* dstCam = dstRen->GetActiveCamera();
        if (!dstCam) continue;
        dstCam->DeepCopy(srcCam);
        dstRen->ResetCameraClippingRange();
        safeRenderWindow(dstView, true);
    }
    m_syncingCameras = false;
}

void vtkComparativeViewWidget::connectExternalHighlighter(
        QObject* highlighter) {
    auto* hl = qobject_cast<cvSelectionHighlighter*>(highlighter);
    if (!hl) return;

    auto syncLater = [this]() {
        QTimer::singleShot(50, this, [this]() {
            if (m_closing || !isVisible()) return;
            copyActorsAcrossSubViews();
            forceRenderAllSubViews();
        });
    };

    connect(hl, &cvSelectionHighlighter::highlightActorAdded, this, syncLater);
    connect(hl, &cvSelectionHighlighter::highlightActorRemoved, this, syncLater);
    connect(hl, &cvSelectionHighlighter::highlightsCleared, this, syncLater);
}

void vtkComparativeViewWidget::refreshSubViews() {
    if (m_sourceView && !m_subViews.isEmpty()) {
        auto* srcWidget = m_sourceView->getVtkWidget();
        vtkRenderer* srcRen = nullptr;
        if (srcWidget && srcWidget->renderWindow() &&
            srcWidget->renderWindow()->GetRenderers()) {
            srcRen = srcWidget->renderWindow()->GetRenderers()->GetFirstRenderer();
        }
        if (srcRen) {
        }
    }
    for (auto* view : m_subViews) {
        if (!view) continue;
        syncSubViewBackgroundFromGlobal(view);
        ccHObject* root = view->getSceneDB();
        if (root) root->setRedrawFlagRecursive(true);
        safeRedraw(view);
    }
    copyActorsAcrossSubViews();

    if (!m_subViews.isEmpty()) {
        auto* firstView = m_subViews.first();
        if (firstView && firstView->getVtkWidget()) {
            auto* vtkW = firstView->getVtkWidget();
            if (vtkW->isVisible() && vtkW->width() >= 2 && vtkW->height() >= 2) {
                auto* rw = vtkW->renderWindow();
                if (rw && rw->GetRenderers()) {
                    auto* ren = rw->GetRenderers()->GetFirstRenderer();
                    if (ren) {
                        ren->ResetCamera();
                        ren->ResetCameraClippingRange();
                    }
                    vtkW->update();
                }
            }
        }
    }

    if (m_subViews.size() > 1) syncCamerasFromFirst();

    for (int i = 1; i < m_subViews.size(); ++i) {
        safeRenderWindow(m_subViews[i]);
    }
}

void vtkComparativeViewWidget::setEntityListProvider(
        EntityListProvider provider) {
    m_entityListProvider = std::move(provider);
    refreshEntityCombo();
}

void vtkComparativeViewWidget::refreshEntityCombo() {
    if (!m_entityCombo || !m_entityListProvider) return;
    m_entityCombo->blockSignals(true);
    m_entityCombo->clear();
    auto entities = m_entityListProvider();
    for (auto* e : entities) {
        if (!e) continue;
        m_entityCombo->addItem(
                e->getName(),
                QVariant::fromValue(reinterpret_cast<quintptr>(e)));
    }
    if (m_initialEntity) {
        for (int i = 0; i < m_entityCombo->count(); ++i) {
            auto* stored = reinterpret_cast<ccHObject*>(
                    m_entityCombo->itemData(i).value<quintptr>());
            if (stored == m_initialEntity) {
                m_entityCombo->setCurrentIndex(i);
                break;
            }
        }
    }
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

    auto* syncCamCheck = new QCheckBox(tr("Sync"), m_toolbar);
    syncCamCheck->setChecked(m_cameraLinkEnabled);
    syncCamCheck->setToolTip(tr("Synchronize cameras across all sub-views"));
    lay->addWidget(syncCamCheck);
    connect(syncCamCheck, &QCheckBox::toggled, this, [this](bool on) {
        m_cameraLinkEnabled = on;
        if (on) syncCamerasFromFirst();
    });

    if (m_type != RENDER) {
        lay->addWidget(new QLabel(QStringLiteral("|"), m_toolbar));
        auto* showLabel = new QLabel(tr("<b>Showing:</b>"), m_toolbar);
        lay->addWidget(showLabel);
        m_entityCombo = new QComboBox(m_toolbar);
        m_entityCombo->setMinimumWidth(120);
        m_entityCombo->setToolTip(tr("Entity displayed in all sub-charts"));
        lay->addWidget(m_entityCombo);
        connect(m_entityCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this](int idx) {
                    if (idx < 0 || !m_entityCombo) return;
                    auto* entity = reinterpret_cast<ccHObject*>(
                            m_entityCombo->itemData(idx).value<quintptr>());
                    if (!entity) return;
                    for (auto* w : m_subWidgets) {
                        auto* chart = qobject_cast<vtkChartView*>(w);
                        if (chart) chart->setEntity(entity);
                    }
                    if (m_statusLabel)
                        m_statusLabel->setText(tr("Synced: %1").arg(entity->getName()));
                });
    }

    auto* screenshotBtn = new QPushButton(tr("Screenshot"), m_toolbar);
    screenshotBtn->setToolTip(
            tr("Export stitched screenshot of all sub-views"));
    lay->addWidget(screenshotBtn);

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
    connect(m_cueModeCombo,
            QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int) { applyCueToSubViews(); });
    connect(m_cueMinSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { applyCueToSubViews(); });
    connect(m_cueMaxSpin,
            QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            [this](double) { applyCueToSubViews(); });
    connect(playBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::onPlayCue);
    connect(m_overlayCheck, &QCheckBox::toggled, this,
            &vtkComparativeViewWidget::onToggleOverlay);
    connect(screenshotBtn, &QPushButton::clicked, this,
            &vtkComparativeViewWidget::onExportScreenshot);
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

    if (cueParam > 0) applyCueToSubViews();
}

void vtkComparativeViewWidget::onToggleOverlay(bool checked) {
    m_overlayMode = checked;
    if (m_subWidgets.size() <= 1) return;

    if (m_type == RENDER && m_subViews.size() > 1) {
        vtkGLView* firstView = m_subViews.first();

        auto getFirstRenderer = [](vtkGLView* v) -> vtkRenderer* {
            if (!v || !v->getVtkWidget()) return nullptr;
            auto* rw = v->getVtkWidget()->renderWindow();
            if (!rw || !rw->GetRenderers()) return nullptr;
            return rw->GetRenderers()->GetFirstRenderer();
        };

        if (checked) {
            auto* dstRen = getFirstRenderer(firstView);
            for (int i = 1; i < m_subViews.size(); ++i) {
                auto* srcRen = getFirstRenderer(m_subViews[i]);
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
            auto* dstRen = getFirstRenderer(firstView);
            if (dstRen) {
                for (int i = 1; i < m_subViews.size(); ++i) {
                    auto* srcRen = getFirstRenderer(m_subViews[i]);
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
    QString path = QFileDialog::getSaveFileName(
            this, tr("Save Comparative Screenshot"), QString(),
            tr("PNG (*.png);;JPEG (*.jpg)"));
    if (path.isEmpty()) return;

    QPixmap composite(size());
    render(&composite);
    composite.save(path);

    if (m_statusLabel) {
        m_statusLabel->setText(tr("Saved: %1").arg(path));
    }
}

void vtkComparativeViewWidget::onPlayCue() {
    applyCueToSubViews();
}

void vtkComparativeViewWidget::saveBaselineCamera() {
    if (m_subViews.isEmpty()) return;
    auto* first = m_subViews.first();
    if (!first || !first->getVtkWidget()) return;
    auto* rw = first->getVtkWidget()->renderWindow();
    if (!rw || !rw->GetRenderers()) return;
    auto* ren = rw->GetRenderers()->GetFirstRenderer();
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
    if (!m_baselineCamera.valid || !view || !view->getVtkWidget()) return;
    auto* rw = view->getVtkWidget()->renderWindow();
    if (!rw || !rw->GetRenderers()) return;
    auto* ren = rw->GetRenderers()->GetFirstRenderer();
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
                if (view && view->getVtkWidget()) {
                    auto* rw = view->getVtkWidget()->renderWindow();
                    if (rw && rw->GetRenderers()) {
                        auto* ren = rw->GetRenderers()->GetFirstRenderer();
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
                        }
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
