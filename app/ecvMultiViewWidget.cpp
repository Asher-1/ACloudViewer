// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMultiViewWidget.h"

#include "MainWindow.h"
#include "ecvPythonView.h"
#include "ecvSpreadSheetView.h"

#ifdef USE_VTK_BACKEND
#include <Tools/SelectionTools/cvSelectionHighlighter.h>
#include <Tools/SelectionTools/cvSelectionToolController.h>
#include <VTKExtensions/Views/vtkChartView.h>
#include <VTKExtensions/Views/vtkComparativeViewWidget.h>
#include <VTKExtensions/Views/vtkOrthoSliceViewWidget.h>
#include <VTKExtensions/Views/vtkSliceViewWidget.h>
#include <VTKExtensions/Widgets/QVTKWidgetCustom.h>
#include <Visualization/VtkVis.h>
#include <Visualization/vtkGLView.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkProp.h>
#include <vtkPropCollection.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#endif

#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvRepresentationManager.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>

#include <QApplication>
#include <QCheckBox>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QScrollArea>
#include <QSet>
#include <QSpinBox>
#include <QSplitter>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

namespace {
constexpr int SPLITTER_GAP = 4;  // ParaView PARAVIEW_DEFAULT_LAYOUT_SPACING

QList<ccHObject*> collectDisplayableEntities() {
    QList<ccHObject*> result;
    QSet<ccHObject*> seen;
    auto* mw = MainWindow::TheInstance();
    if (!mw) return result;
    ccHObject* root = mw->dbRootObject();
    if (!root) return result;

    auto addUnique = [&](ccHObject* obj) {
        if (obj && obj->isEnabled() && !seen.contains(obj)) {
            seen.insert(obj);
            result.append(obj);
        }
    };

    ccHObject::Container clouds;
    root->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD, false);
    for (auto* c : clouds) addUnique(c);

    ccHObject::Container meshes;
    root->filterChildren(meshes, true, CV_TYPES::MESH, false);
    for (auto* m : meshes) addUnique(m);

    ccHObject::Container subMeshes;
    root->filterChildren(subMeshes, true, CV_TYPES::SUB_MESH, false);
    for (auto* sm : subMeshes) addUnique(sm);

    ccHObject::Container facets;
    root->filterChildren(facets, true, CV_TYPES::FACET, false);
    for (auto* f : facets) addUnique(f);

    return result;
}
}  // namespace

ecvMultiViewWidget::ecvMultiViewWidget(QWidget* parent) : QWidget(parent) {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    m_contentContainer = new QWidget(this);
    m_contentContainer->setObjectName("MVWContentContainer");
    auto* containerLayout = new QVBoxLayout(m_contentContainer);
    containerLayout->setContentsMargins(0, 0, 0, 0);
    containerLayout->setSpacing(0);
    rootLayout->addWidget(m_contentContainer);

    auto* placeholderWidget = new QWidget();
    auto* phLayout = new QVBoxLayout(placeholderWidget);
    phLayout->setContentsMargins(20, 20, 20, 20);
    auto* phLabel = new QLabel(tr("Layout shown in separate window"));
    phLabel->setAlignment(Qt::AlignCenter);
    phLayout->addStretch(1);
    phLayout->addWidget(phLabel);
    auto* restoreBtn = new QPushButton(tr("Click to restore"));
    restoreBtn->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    auto* btnLayout = new QHBoxLayout();
    btnLayout->addStretch(1);
    btnLayout->addWidget(restoreBtn);
    btnLayout->addStretch(1);
    phLayout->addWidget(restoreBtn, 0, Qt::AlignCenter);
    phLayout->addStretch(1);
    connect(restoreBtn, &QPushButton::clicked, this,
            [this]() { togglePopout(); });
    m_popoutPlaceholder.reset(placeholderWidget);

    qApp->installEventFilter(this);

    connect(&ecvViewManager::instance(), &ecvViewManager::activeViewChanged,
            this, [this](ecvGenericGLDisplay* newActive, ecvGenericGLDisplay*) {
                markActive(newActive);
            });
}

ecvMultiViewWidget::~ecvMultiViewWidget() { qApp->removeEventFilter(this); }

void ecvMultiViewWidget::setLayoutManager(ecvViewLayoutProxy* layout) {
    if (m_layout == layout) return;

    if (m_layout) {
        disconnect(m_layout, nullptr, this, nullptr);
    }

    m_layout = layout;

    if (m_layout) {
        connect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
                &ecvMultiViewWidget::reload);
    }

    reload();
}

QList<ecvGenericGLDisplay*> ecvMultiViewWidget::viewProxies() const {
    QList<ecvGenericGLDisplay*> result;
    if (!m_layout) return result;
    auto views = m_layout->getViews();
    for (auto* v : views) result.append(v);
    return result;
}

bool ecvMultiViewWidget::isViewAssigned(ecvGenericGLDisplay* view) const {
    return m_layout && m_layout->containsView(view);
}

int ecvMultiViewWidget::activeFrameLocation() const {
    if (!m_activeFrame) return -1;
    return findLocationForFrame(m_activeFrame);
}

// ============================================================================
// Reload — rebuild the entire widget tree from the layout proxy
// ============================================================================

void ecvMultiViewWidget::reload() {
    // Suppress painting while tearing down and rebuilding the widget tree.
    // Without this, the brief interval after hide() + reparent(nullptr) and
    // before the new layout is composited shows a gray/white background
    // (the naked container).  ParaView avoids this by rearranging in-place;
    // setUpdatesEnabled achieves the same visual atomicity here.
    if (m_contentContainer) {
        m_contentContainer->setUpdatesEnabled(false);
    }

    // Detach view widgets from their frame wrappers BEFORE deleting frames.
    // buildCell() reparents view->asWidget() into a frame; when that frame is
    // destroyed the view widget would be destroyed too.  Reparenting to nullptr
    // first keeps the view widget alive so it can be re-wrapped in the new
    // tree.
    for (auto it = m_viewFrames.constBegin(); it != m_viewFrames.constEnd();
         ++it) {
        if (auto* display = it.key()) {
            QWidget* vw = display->asWidget();
            if (vw) {
                vw->setParent(nullptr);
                vw->hide();
            }
        }
    }

    // Preserve non-GL cell frames (charts, spreadsheets, comparatives) so
    // they survive layout rebuilds triggered by split-fraction changes etc.
    // These views are not registered in the layout proxy (no assignView),
    // so buildCell() would otherwise replace them with empty "Create View".
    QHash<int, QWidget*> preservedNonGLFrames;
    for (auto it = m_cellFrames.constBegin(); it != m_cellFrames.constEnd();
         ++it) {
        int loc = it.key();
        QWidget* frame = it.value();
        if (!frame) continue;
        bool isGLFrame = m_viewFrames.key(frame, nullptr) != nullptr;
        bool isEmptyCell = frame->property("IS_EMPTY_CELL").toBool();
        if (!isGLFrame && !isEmptyCell && m_layout &&
            !m_layout->isSplitCell(loc) && !m_layout->getView(loc)) {
            frame->setParent(nullptr);
            frame->hide();
            preservedNonGLFrames[loc] = frame;
        }
    }

    m_cellFrames.clear();
    m_viewFrames.clear();
    m_activeFrame = nullptr;

    auto* containerLayout =
            m_contentContainer ? m_contentContainer->layout() : nullptr;
    if (!containerLayout) {
        if (m_contentContainer) m_contentContainer->setUpdatesEnabled(true);
        return;
    }

    QLayoutItem* child;
    while ((child = containerLayout->takeAt(0)) != nullptr) {
        if (child->widget()) {
            bool preserved = false;
            for (auto& pf : preservedNonGLFrames) {
                if (pf == child->widget()) {
                    preserved = true;
                    break;
                }
            }
            if (!preserved) {
                child->widget()->setParent(nullptr);
                child->widget()->deleteLater();
            }
        }
        delete child;
    }

    if (!m_layout) {
        for (auto* w : preservedNonGLFrames) {
            if (w) w->deleteLater();
        }
        if (m_contentContainer) m_contentContainer->setUpdatesEnabled(true);
        return;
    }

    m_preservedNonGLFrames = std::move(preservedNonGLFrames);
    QWidget* rootWidget = buildCell(0);
    m_preservedNonGLFrames.clear();
    if (rootWidget) {
        containerLayout->addWidget(rootWidget);
    }

    auto* activeView = ecvViewManager::instance().getActiveView();
    if (activeView && m_layout->containsView(activeView)) {
        markActive(activeView);
    } else if (isVisible()) {
        makeFrameActive();
    }

    // Re-enable painting now that the new tree is assembled.
    if (m_contentContainer) {
        m_contentContainer->setUpdatesEnabled(true);
    }

    // After reparenting, every view needs a VTK render to refill its
    // framebuffer.  rebindToolsToActiveView() only redraws the *active*
    // view, leaving other visible panes blank/gray.  Schedule a
    // deferred redraw for every view in this layout.
    QTimer::singleShot(0, this, [this]() {
        for (auto it = m_viewFrames.constBegin(); it != m_viewFrames.constEnd();
             ++it) {
            if (auto* display = it.key()) {
                QWidget* vw = display->asWidget();
                if (vw && vw->isVisible()) {
#ifdef USE_VTK_BACKEND
                    auto* glView = dynamic_cast<vtkGLView*>(display);
                    if (glView) {
                        glView->redraw(false, true);
                    }
#endif
                }
            }
        }
    });
}

QWidget* ecvMultiViewWidget::buildCell(int location) {
    if (!m_layout) return nullptr;

    if (m_layout->isSplitCell(location)) {
        auto dir = m_layout->splitDirection(location);
        auto fraction = m_layout->splitFraction(location);

        Qt::Orientation qtDir = dir == ecvViewLayoutProxy::HORIZONTAL
                                        ? Qt::Horizontal
                                        : Qt::Vertical;
        auto* splitter = new QSplitter(qtDir, this);
        splitter->setChildrenCollapsible(false);
        splitter->setHandleWidth(SPLITTER_GAP);
        splitter->setOpaqueResize(true);
        splitter->setStyleSheet(QStringLiteral(
                "QSplitter::handle { background: palette(window); }"
                "QSplitter::handle:hover { background: palette(mid); }"
                "QSplitter::handle:pressed { background: palette(highlight); "
                "}"));
        splitter->setProperty("CELL_INDEX", location);

        QWidget* left = buildCell(ecvViewLayoutProxy::firstChild(location));
        QWidget* right = buildCell(ecvViewLayoutProxy::secondChild(location));

        if (left) splitter->addWidget(left);
        if (right) splitter->addWidget(right);

        int leftSize = static_cast<int>(1000 * fraction);
        splitter->setSizes({leftSize, 1000 - leftSize});

        QTimer::singleShot(0, splitter, [splitter, fraction]() {
            if (!splitter->isVisible()) return;
            int total = (splitter->orientation() == Qt::Horizontal)
                                ? splitter->width()
                                : splitter->height();
            if (total > 0) {
                int sz = static_cast<int>(total * fraction);
                splitter->setSizes({sz, total - sz});
            }
        });

        auto* undoTimer = new QTimer(splitter);
        undoTimer->setSingleShot(true);
        undoTimer->setInterval(500);
        undoTimer->setProperty("_undoActive", false);
        connect(undoTimer, &QTimer::timeout, this, [this, undoTimer]() {
            if (m_layout && undoTimer->property("_undoActive").toBool()) {
                m_layout->endUndoSet();
                undoTimer->setProperty("_undoActive", false);
            }
        });

        connect(splitter, &QSplitter::splitterMoved, this,
                [this, location, splitter, undoTimer](int, int) {
                    if (!m_layout) return;
                    if (!undoTimer->property("_undoActive").toBool()) {
                        m_layout->beginUndoSet(QStringLiteral("Resize Split"));
                        undoTimer->setProperty("_undoActive", true);
                    }
                    undoTimer->start();

                    QList<int> sizes = splitter->sizes();
                    int total = sizes[0] + sizes[1];
                    if (total > 0) {
                        double frac = static_cast<double>(sizes[0]) / total;
                        QSignalBlocker blocker(m_layout);
                        m_layout->setSplitFraction(location, frac);
                    }
                });

        if (m_layout->maximizedCell() >= 0) {
            bool leftHasMax = false, rightHasMax = false;
            int maxCell = m_layout->maximizedCell();
            auto checkContains = [&](int root, auto& self) -> bool {
                if (root == maxCell) return true;
                if (!m_layout->isSplitCell(root)) return false;
                return self(ecvViewLayoutProxy::firstChild(root), self) ||
                       self(ecvViewLayoutProxy::secondChild(root), self);
            };
            leftHasMax = checkContains(ecvViewLayoutProxy::firstChild(location),
                                       checkContains);
            rightHasMax = checkContains(
                    ecvViewLayoutProxy::secondChild(location), checkContains);

            if (left && !leftHasMax && (leftHasMax || rightHasMax))
                left->hide();
            if (right && !rightHasMax && (leftHasMax || rightHasMax))
                right->hide();
        }

        return splitter;
    }

    ecvGenericGLDisplay* view = m_layout->getView(location);
    QWidget* frame = nullptr;

    if (view) {
        QWidget* viewWidget = view->asWidget();
        if (viewWidget && m_frameFactory) {
            frame = m_frameFactory(viewWidget, view->getTitle());
        } else if (viewWidget) {
            frame = viewWidget;
        }
        if (viewWidget && !viewWidget->isVisible()) {
            viewWidget->show();
        }
    } else if (m_preservedNonGLFrames.contains(location)) {
        frame = m_preservedNonGLFrames.take(location);
        if (frame) frame->show();
    } else {
        frame = createEmptyCellWidget(location);
    }

    if (frame) {
        frame->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = frame;
        if (view) {
            m_viewFrames[view] = frame;
            if (m_frameWiredCallback) {
                m_frameWiredCallback(frame, view);
            }
        }
    }

    return frame;
}

// P6: Copy all per-view representations from sourceView to destView so the
// newly split pane starts with the same visual state as the original.
// Exception: LABEL_2D entities are skipped because labels belong to their
// original view — cc2DLabel::drawMeOnly() uses getDisplay() binding.
// Cloning label reps would cause cross-window pollution.
static void copyRepresentationsOnSplit(ecvGenericGLDisplay* sourceView,
                                       ecvGenericGLDisplay* destView) {
    if (!sourceView || !destView) return;
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
}

// ============================================================================
// Empty cell placeholder (ParaView pqEmptyView / "Create View" pattern)
// ============================================================================

QWidget* ecvMultiViewWidget::createEmptyCellWidget(int location) {
    auto* frame = new QFrame(this);
    frame->setFrameStyle(QFrame::NoFrame);
    frame->setMinimumSize(50, 50);
    frame->setProperty("IS_EMPTY_CELL", true);

    auto* outerLayout = new QVBoxLayout(frame);
    outerLayout->setContentsMargins(0, 0, 0, 0);
    outerLayout->setSpacing(0);

    // ParaView pqViewFrame-style header for empty cells: split H/V, maximize,
    // close
    auto* headerBar = new QWidget(frame);
    headerBar->setObjectName("ViewTitleBar");
    auto* headerLayout = new QHBoxLayout(headerBar);
    headerLayout->setContentsMargins(0, 0, 0, 0);
    headerLayout->setSpacing(1);
    headerLayout->addStretch(1);

    static constexpr int kBtnSize = 20;
    auto makeBtn = [headerBar](const QIcon& icon, const QString& tip) {
        auto* btn = new QToolButton(headerBar);
        btn->setIcon(icon);
        btn->setToolTip(tip);
        btn->setAutoRaise(true);
        btn->setIconSize(QSize(kBtnSize - 4, kBtnSize - 4));
        btn->setFixedSize(kBtnSize, kBtnSize);
        return btn;
    };

    auto* splitHBtn =
            makeBtn(QIcon(":/Resources/images/svg/pqSplitHorizontal.svg"),
                    tr("Split Left|Right"));
    connect(splitHBtn, &QToolButton::clicked, this,
            [this, frame]() { onSplitHorizontal(frame); });
    headerLayout->addWidget(splitHBtn);

    auto* splitVBtn =
            makeBtn(QIcon(":/Resources/images/svg/pqSplitVertical.svg"),
                    tr("Split Top|Bottom"));
    connect(splitVBtn, &QToolButton::clicked, this,
            [this, frame]() { onSplitVertical(frame); });
    headerLayout->addWidget(splitVBtn);

    auto* maxBtn = makeBtn(
            headerBar->style()->standardIcon(QStyle::SP_TitleBarMaxButton),
            tr("Maximize"));
    connect(maxBtn, &QToolButton::clicked, this,
            [this, frame]() { onMaximize(frame); });
    headerLayout->addWidget(maxBtn);

    auto* closeBtn = makeBtn(QIcon(":/Resources/images/svg/pqCloseView.svg"),
                             tr("Close View"));
    connect(closeBtn, &QToolButton::clicked, this,
            [this, frame]() { onCloseView(frame); });
    closeBtn->setEnabled(location != 0);
    headerLayout->addWidget(closeBtn);

    outerLayout->addWidget(headerBar);

    auto* scrollArea = new QScrollArea(frame);
    scrollArea->setWidgetResizable(true);
    scrollArea->setFrameShape(QFrame::NoFrame);

    auto* scrollContent = new QWidget(scrollArea);
    auto* gridLayout = new QGridLayout(scrollContent);

    // Centered content with spacers (ParaView pqEmptyView.ui pattern)
    gridLayout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum,
                                        QSizePolicy::Expanding),
                        0, 1);
    gridLayout->addItem(new QSpacerItem(40, 20, QSizePolicy::Expanding,
                                        QSizePolicy::Minimum),
                        1, 0);
    gridLayout->addItem(new QSpacerItem(40, 20, QSizePolicy::Expanding,
                                        QSizePolicy::Minimum),
                        1, 2);
    gridLayout->addItem(new QSpacerItem(20, 40, QSizePolicy::Minimum,
                                        QSizePolicy::Expanding),
                        2, 1);

    auto* actionsFrame = new QFrame(scrollContent);
    actionsFrame->setFrameShape(QFrame::NoFrame);
    auto* btnLayout = new QVBoxLayout(actionsFrame);
    btnLayout->setSpacing(2);

    auto* title =
            new QLabel(QStringLiteral("<b>Create View</b>"), actionsFrame);
    title->setAlignment(Qt::AlignCenter);
    btnLayout->addWidget(title);

    // ParaView-style view type list (sorted alphabetically, Render View first)
    struct ViewType {
        QString canonicalName;
        QString label;
        bool available;
    };
#ifdef USE_VTK_BACKEND
    constexpr bool vtkAvail = true;
#else
    constexpr bool vtkAvail = false;
#endif
    QList<ViewType> viewTypes = {
            {QStringLiteral("Render View"), tr("Render View"), true},
            {QStringLiteral("Render View (Comparative)"),
             tr("Render View (Comparative)"), vtkAvail},
            {QStringLiteral("Bar Chart View"), tr("Bar Chart View"), vtkAvail},
            {QStringLiteral("Bar Chart View (Comparative)"),
             tr("Bar Chart View (Comparative)"), vtkAvail},
            {QStringLiteral("Box Chart View"), tr("Box Chart View"), vtkAvail},
            {QStringLiteral("Eye Dome Lighting"), tr("Eye Dome Lighting"),
             vtkAvail},
            {QStringLiteral("Histogram View"), tr("Histogram View"), vtkAvail},
            {QStringLiteral("Image Chart View"), tr("Image Chart View"),
             vtkAvail},
            {QStringLiteral("Line Chart View"), tr("Line Chart View"),
             vtkAvail},
            {QStringLiteral("Line Chart View (Comparative)"),
             tr("Line Chart View (Comparative)"), vtkAvail},
            {QStringLiteral("Orthographic Slice View"),
             tr("Orthographic Slice View"), vtkAvail},
            {QStringLiteral("Parallel Coordinates View"),
             tr("Parallel Coordinates View"), vtkAvail},
            {QStringLiteral("Plot Matrix View"), tr("Plot Matrix View"),
             vtkAvail},
            {QStringLiteral("Point Chart View"), tr("Point Chart View"),
             vtkAvail},
            {QStringLiteral("Python View"), tr("Python View"), true},
            {QStringLiteral("Quartile Chart View"), tr("Quartile Chart View"),
             vtkAvail},
            {QStringLiteral("Slice View"), tr("Slice View"), vtkAvail},
            {QStringLiteral("SpreadSheet View"), tr("SpreadSheet View"), true},
    };

    auto createRenderViewForCell = [this, location]() {
        if (!m_viewFactory || !m_layout) return;
        auto* newView = m_viewFactory();
        if (!newView) return;

        ecvGenericGLDisplay* siblingView = nullptr;
        int parentIdx = ecvViewLayoutProxy::parent(location);
        if (parentIdx >= 0) {
            int sibling =
                    (location == ecvViewLayoutProxy::firstChild(parentIdx))
                            ? ecvViewLayoutProxy::secondChild(parentIdx)
                            : ecvViewLayoutProxy::firstChild(parentIdx);
            siblingView = m_layout->getView(sibling);
        }
        if (!siblingView) {
            auto views = m_layout->getViews();
            for (auto* v : views) {
                if (v) {
                    siblingView = v;
                    break;
                }
            }
        }

        m_layout->assignView(location, newView);
        if (siblingView) {
            copyRepresentationsOnSplit(siblingView, newView);
        }
        ecvViewManager::instance().setActiveView(newView);
    };

    auto createSpreadSheetForCell = [this, location, frame]() {
        QWidget* oldWidget = m_cellFrames.value(location);
        QWidget* parentWidget = oldWidget ? oldWidget->parentWidget() : nullptr;
        auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
        int splitterIdx =
                parentSplitter ? parentSplitter->indexOf(oldWidget) : -1;

        auto* spreadSheet = new ecvSpreadSheetView(this);
        spreadSheet->setEntityListProvider(collectDisplayableEntities);
        auto& sel = MainWindow::TheInstance()->getSelectedEntities();
        if (!sel.empty()) {
            spreadSheet->setEntity(sel.front());
        }

        connect(spreadSheet, &ecvSpreadSheetView::tableSelectionChanged, this,
                [](ccHObject* entity, const QVector<unsigned>& indices) {
                    if (!entity || indices.isEmpty()) return;
                    QSet<unsigned> idxSet(indices.begin(), indices.end());
                    emit ecvViewManager::instance().pointIndicesSelected(
                            entity, idxSet);
                });

        QWidget* wrapped = spreadSheet;
        if (m_frameFactory) {
            wrapped = m_frameFactory(spreadSheet, spreadSheet->title());
        }
        if (!wrapped) return;

        wrapped->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = wrapped;

        if (parentSplitter && splitterIdx >= 0) {
            parentSplitter->insertWidget(splitterIdx, wrapped);
        } else if (parentWidget && parentWidget->layout()) {
            auto* lay = parentWidget->layout();
            if (oldWidget)
                lay->replaceWidget(oldWidget, wrapped);
            else
                lay->addWidget(wrapped);
        }

        if (oldWidget && oldWidget != wrapped) {
            oldWidget->setParent(nullptr);
            oldWidget->deleteLater();
        }
    };

    auto createPythonViewForCell = [this, location]() {
        QWidget* oldWidget = m_cellFrames.value(location);
        QWidget* parentWidget = oldWidget ? oldWidget->parentWidget() : nullptr;
        auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
        int splitterIdx =
                parentSplitter ? parentSplitter->indexOf(oldWidget) : -1;

        auto* pyView = new ecvPythonView(this);
        pyView->setEntityListProvider([]() -> QList<ccHObject*> {
            QList<ccHObject*> result;
            auto* mw = MainWindow::TheInstance();
            if (!mw) return result;
            ccHObject* dbRoot = mw->dbRootObject();
            if (!dbRoot) return result;
            ccHObject::Container clouds;
            dbRoot->filterChildren(clouds, true, CV_TYPES::POINT_CLOUD, true);
            for (auto* c : clouds) result.append(c);
            ccHObject::Container meshes;
            dbRoot->filterChildren(meshes, true, CV_TYPES::MESH, true);
            for (auto* m : meshes) result.append(m);
            return result;
        });
        auto& sel = MainWindow::TheInstance()->getSelectedEntities();
        if (!sel.empty()) {
            pyView->setEntity(sel.front());
        }

        QWidget* wrapped = pyView;
        if (m_frameFactory) {
            wrapped = m_frameFactory(pyView, pyView->title());
        }
        if (!wrapped) return;

        wrapped->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = wrapped;

        if (parentSplitter && splitterIdx >= 0) {
            parentSplitter->insertWidget(splitterIdx, wrapped);
        } else if (parentWidget && parentWidget->layout()) {
            auto* lay = parentWidget->layout();
            if (oldWidget)
                lay->replaceWidget(oldWidget, wrapped);
            else
                lay->addWidget(wrapped);
        }

        if (oldWidget && oldWidget != wrapped) {
            oldWidget->setParent(nullptr);
            oldWidget->deleteLater();
        }
    };

#ifdef USE_VTK_BACKEND
    auto createChartForCell = [this, location](vtkChartView::ChartType type) {
        QWidget* oldWidget = m_cellFrames.value(location);
        QWidget* parentWidget = oldWidget ? oldWidget->parentWidget() : nullptr;
        auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
        int splitterIdx =
                parentSplitter ? parentSplitter->indexOf(oldWidget) : -1;

        auto* chartView = new vtkChartView(type, this);
        chartView->setEntityListProvider(collectDisplayableEntities);

        connect(chartView, &vtkChartView::pointsHighlighted, this,
                [](ccHObject* entity, const QVector<unsigned>& indices) {
                    if (!entity || indices.isEmpty()) return;
                    QSet<unsigned> idxSet(indices.begin(), indices.end());
                    emit ecvViewManager::instance().pointIndicesSelected(
                            entity, idxSet);
                });

        ccHObject* initEntity = nullptr;
        auto& sel = MainWindow::TheInstance()->getSelectedEntities();
        if (!sel.empty()) {
            initEntity = sel.front();
        } else {
            auto entities = collectDisplayableEntities();
            if (!entities.empty()) initEntity = entities.front();
        }
        if (initEntity) {
            QTimer::singleShot(0, chartView, [chartView, initEntity]() {
                chartView->setEntity(initEntity);
            });
        }

        QWidget* wrapped = chartView;
        if (m_frameFactory) {
            wrapped = m_frameFactory(chartView, chartView->title());
        }
        if (!wrapped) return;

        wrapped->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = wrapped;

        if (parentSplitter && splitterIdx >= 0) {
            parentSplitter->insertWidget(splitterIdx, wrapped);
        } else if (parentWidget && parentWidget->layout()) {
            auto* lay = parentWidget->layout();
            if (oldWidget)
                lay->replaceWidget(oldWidget, wrapped);
            else
                lay->addWidget(wrapped);
        }

        if (oldWidget && oldWidget != wrapped) {
            oldWidget->setParent(nullptr);
            oldWidget->deleteLater();
        }
    };

    auto createEDLViewForCell = [this, location]() {
        if (!m_viewFactory || !m_layout) return;
        auto* view = m_viewFactory();
        if (!view) return;

        view->setViewXmlLabel(QStringLiteral("Eye Dome Lighting"));

        ecvGenericGLDisplay* siblingView = nullptr;
        int parentIdx = ecvViewLayoutProxy::parent(location);
        if (parentIdx >= 0) {
            int sibling =
                    (location == ecvViewLayoutProxy::firstChild(parentIdx))
                            ? ecvViewLayoutProxy::secondChild(parentIdx)
                            : ecvViewLayoutProxy::firstChild(parentIdx);
            siblingView = m_layout->getView(sibling);
        }
        if (!siblingView) {
            auto views = m_layout->getViews();
            for (auto* v : views) {
                if (v) {
                    siblingView = v;
                    break;
                }
            }
        }

        m_layout->assignView(location, view);
        if (siblingView) {
            copyRepresentationsOnSplit(siblingView, view);
        }
        ecvViewManager::instance().setActiveView(view);
        view->enableEDL(true);

        {
            auto* wrapped = m_cellFrames.value(location);
            if (wrapped) {
                auto* edlBar = new QWidget(wrapped);
                edlBar->setObjectName("EDLPropsBar");
                auto* edlLayout = new QHBoxLayout(edlBar);
                edlLayout->setContentsMargins(4, 2, 4, 2);
                edlLayout->setSpacing(6);

                edlLayout->addWidget(
                        new QLabel(QStringLiteral("<b>EDL</b>"), edlBar));

                auto* radiusLabel = new QLabel(tr("Radius:"), edlBar);
                edlLayout->addWidget(radiusLabel);
                auto* radiusSpin = new QSpinBox(edlBar);
                radiusSpin->setRange(1, 16);
                radiusSpin->setValue(view->edlLowResFactor());
                radiusSpin->setToolTip(
                        tr("EDL low-resolution sampling factor (higher = wider "
                           "radius)"));
                radiusSpin->setFixedWidth(50);
                edlLayout->addWidget(radiusSpin);
                connect(radiusSpin, QOverload<int>::of(&QSpinBox::valueChanged),
                        view, &vtkGLView::setEDLLowResFactor);

                auto* fxaaCheck = new QCheckBox(tr("FXAA"), edlBar);
                fxaaCheck->setChecked(view->isEDLFXAAEnabled());
                fxaaCheck->setToolTip(
                        tr("Enable FXAA anti-aliasing on top of EDL"));
                edlLayout->addWidget(fxaaCheck);
                connect(fxaaCheck, &QCheckBox::toggled, view,
                        &vtkGLView::setEDLFXAAEnabled);

                edlLayout->addStretch(1);

                auto* frameLayout =
                        qobject_cast<QVBoxLayout*>(wrapped->layout());
                if (frameLayout) {
                    int insertIdx = 1;
                    for (int i = 0; i < frameLayout->count(); ++i) {
                        auto* item = frameLayout->itemAt(i);
                        if (item && item->widget() &&
                            item->widget()->objectName() ==
                                    QLatin1String("ViewTitleBar")) {
                            insertIdx = i + 1;
                            break;
                        }
                    }
                    frameLayout->insertWidget(insertIdx, edlBar);
                } else {
                    edlBar->raise();
                    edlBar->move(4, 26);
                    edlBar->adjustSize();
                    edlBar->show();
                }
            }
            auto* titleLabel = wrapped->findChild<QLabel*>("ViewTitleLabel");
            if (titleLabel) {
                const QString t = view->getTitle();
                titleLabel->setProperty("plainTitle", t);
                titleLabel->setText(t);
            }
        }
    };

    auto createSliceViewForCell = [this, location]() {
        if (!m_viewFactory || !m_layout) return;
        auto* view = m_viewFactory();
        if (!view) return;

        view->setViewXmlLabel(QStringLiteral("Slice View"));

        ecvGenericGLDisplay* siblingView =
                ecvViewManager::instance().getActiveView();
        if (!siblingView) {
            auto views = m_layout->getViews();
            for (auto* v : views) {
                if (v) {
                    siblingView = v;
                    break;
                }
            }
        }

        m_layout->assignView(location, view);
        view->enableSliceMode(true);

        if (siblingView && siblingView != view) {
            copyRepresentationsOnSplit(siblingView, view);
        }

        QWidget* oldWidget = m_cellFrames.value(location);
        QWidget* parentWidget = oldWidget ? oldWidget->parentWidget() : nullptr;
        auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
        int splitterIdx =
                parentSplitter ? parentSplitter->indexOf(oldWidget) : -1;

        auto* sliceWrapper = new vtkSliceViewWidget(view, nullptr);

        if (view->getVisualizer3D()) {
            auto rens = view->getVisualizer3D()->getRendererCollection();
            if (rens) {
                rens->InitTraversal();
                auto* ren = rens->GetNextItem();
                if (ren) {
                    double bounds[6];
                    ren->ComputeVisiblePropBounds(bounds);
                    if (bounds[0] <= bounds[1]) {
                        sliceWrapper->setDataBounds(bounds);
                    }
                }
            }
        }

        QWidget* wrapped = sliceWrapper;
        if (m_frameFactory) {
            wrapped = m_frameFactory(sliceWrapper, view->getTitle());
        }
        if (!wrapped) return;

        wrapped->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = wrapped;
        m_viewFrames[view] = wrapped;
        if (m_frameWiredCallback) {
            m_frameWiredCallback(wrapped, view);
        }

        if (parentSplitter && splitterIdx >= 0) {
            parentSplitter->insertWidget(splitterIdx, wrapped);
        } else if (parentWidget && parentWidget->layout()) {
            auto* lay = parentWidget->layout();
            if (oldWidget)
                lay->replaceWidget(oldWidget, wrapped);
            else
                lay->addWidget(wrapped);
        }

        if (oldWidget && oldWidget != wrapped) {
            oldWidget->setParent(nullptr);
            oldWidget->deleteLater();
        }

        ecvViewManager::instance().setActiveView(view);
    };

    auto createComparativeForCell =
            [this, location](vtkComparativeViewWidget::ComparativeType type) {
                QWidget* oldWidget = m_cellFrames.value(location);
                QWidget* parentWidget =
                        oldWidget ? oldWidget->parentWidget() : nullptr;
                auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
                int splitterIdx = parentSplitter
                                          ? parentSplitter->indexOf(oldWidget)
                                          : -1;

                auto* sourceViewGL = dynamic_cast<vtkGLView*>(
                        ecvViewManager::instance().getActiveView());

                auto* compView = new vtkComparativeViewWidget(type, this);
                if (sourceViewGL) compView->setSourceView(sourceViewGL);

                auto* selCtrl2 = cvSelectionToolController::instance();
                if (selCtrl2 && selCtrl2->highlighter()) {
                    compView->connectExternalHighlighter(
                            selCtrl2->highlighter());
                }
                if (type == vtkComparativeViewWidget::RENDER && m_viewFactory) {
                    compView->setSubViewShutdownHook([](vtkGLView* subView) {
                        if (!subView) return;
                        QWidget* w = subView->asWidget();
                        if (!w) return;
                        if (auto* mw = MainWindow::TheInstance()) {
                            mw->removeEventFilter(w);
                        }
                    });
                    // ParaView模式: 不自动复制表示，保持空视窗
                    compView->setSubViewInitCallback(
                            [](vtkGLView* /*subView*/) {
                                // 不再自动复制，用户通过Showing下拉框选择实体
                            });
                    compView->setRenderViewFactory([this]() -> vtkGLView* {
                        return m_viewFactory ? m_viewFactory() : nullptr;
                    });
                    // ✅ 为RENDER类型也设置EntityListProvider
                    compView->setEntityListProvider(collectDisplayableEntities);
                } else {
                    compView->setEntityListProvider(collectDisplayableEntities);
                    ccHObject* initEntity = nullptr;
                    auto& sel2 =
                            MainWindow::TheInstance()->getSelectedEntities();
                    if (!sel2.empty()) {
                        initEntity = sel2.front();
                    } else {
                        auto entities = collectDisplayableEntities();
                        if (!entities.empty()) initEntity = entities.front();
                    }
                    if (initEntity) {
                        compView->setInitialEntity(initEntity);
                    }
                    compView->setupGrid();
                }

                QWidget* wrapped = compView;
                if (m_frameFactory) {
                    wrapped = m_frameFactory(compView, compView->title());
                }
                if (!wrapped) return;

                wrapped->setProperty("CELL_INDEX", location);
                m_cellFrames[location] = wrapped;

                connect(compView, &vtkComparativeViewWidget::clicked, this,
                        [this, wrapped]() { makeActive(wrapped); });

                connect(compView, &vtkComparativeViewWidget::requestToolRebind,
                        this, [this](vtkGLView* view) {
                            if (view) {
                                ecvViewManager::instance().setActiveView(view);
                                if (auto* mw = MainWindow::TheInstance()) {
                                    mw->rebindToolsToActiveView(view);
                                }
                            }
                        });

                if (parentSplitter && splitterIdx >= 0) {
                    parentSplitter->insertWidget(splitterIdx, wrapped);
                } else if (parentWidget && parentWidget->layout()) {
                    auto* lay = parentWidget->layout();
                    if (oldWidget)
                        lay->replaceWidget(oldWidget, wrapped);
                    else
                        lay->addWidget(wrapped);
                }

                if (oldWidget && oldWidget != wrapped) {
                    oldWidget->setParent(nullptr);
                    oldWidget->deleteLater();
                }
            };

    auto createOrthoSliceForCell = [this, location]() {
        QWidget* oldWidget = m_cellFrames.value(location);
        QWidget* parentWidget = oldWidget ? oldWidget->parentWidget() : nullptr;
        auto* parentSplitter = qobject_cast<QSplitter*>(parentWidget);
        int splitterIdx =
                parentSplitter ? parentSplitter->indexOf(oldWidget) : -1;

        auto* orthoView = new vtkOrthoSliceViewWidget(this);
        orthoView->setEntityListProvider(collectDisplayableEntities);

        auto* selCtrl = cvSelectionToolController::instance();
        if (selCtrl && selCtrl->highlighter()) {
            orthoView->connectExternalHighlighter(selCtrl->highlighter());
        }

        auto* activeView = dynamic_cast<vtkGLView*>(
                ecvViewManager::instance().getActiveView());
        if (activeView && activeView->getVtkWidget()) {
            auto* rw = activeView->getVtkWidget()->renderWindow();
            if (rw && rw->GetRenderers()) {
                auto* srcRen = rw->GetRenderers()->GetFirstRenderer();
                if (srcRen) {
                    orthoView->populateFromRenderer(srcRen);
                }
            }
        }

        QWidget* wrapped = orthoView;
        if (m_frameFactory) {
            wrapped = m_frameFactory(orthoView, orthoView->title());
        }
        if (!wrapped) return;

        wrapped->setProperty("CELL_INDEX", location);
        m_cellFrames[location] = wrapped;

        connect(orthoView, &vtkOrthoSliceViewWidget::clicked, this,
                [this, wrapped]() { makeActive(wrapped); });

        if (parentSplitter && splitterIdx >= 0) {
            parentSplitter->insertWidget(splitterIdx, wrapped);
        } else if (parentWidget && parentWidget->layout()) {
            auto* lay = parentWidget->layout();
            if (oldWidget)
                lay->replaceWidget(oldWidget, wrapped);
            else
                lay->addWidget(wrapped);
        }

        if (oldWidget && oldWidget != wrapped) {
            oldWidget->setParent(nullptr);
            oldWidget->deleteLater();
        }
    };
#endif  // USE_VTK_BACKEND

    for (const auto& vt : viewTypes) {
        auto* btn = new QPushButton(vt.label, actionsFrame);
        btn->setProperty("VIEW_TYPE_ID", vt.canonicalName);
        btn->setCursor(Qt::PointingHandCursor);
        btn->setEnabled(vt.available);
        if (vt.available) {
            if (vt.canonicalName == QLatin1String("Render View")) {
                connect(btn, &QPushButton::clicked, this,
                        createRenderViewForCell);
            } else if (vt.canonicalName == QLatin1String("SpreadSheet View")) {
                connect(btn, &QPushButton::clicked, this,
                        createSpreadSheetForCell);
            } else if (vt.canonicalName == QLatin1String("Python View")) {
                connect(btn, &QPushButton::clicked, this,
                        createPythonViewForCell);
#ifdef USE_VTK_BACKEND
            } else if (vt.canonicalName ==
                       QLatin1String("Render View (Comparative)")) {
                connect(btn, &QPushButton::clicked, this,
                        [createComparativeForCell]() {
                            createComparativeForCell(
                                    vtkComparativeViewWidget::RENDER);
                        });
            } else if (vt.canonicalName == QLatin1String("Eye Dome Lighting")) {
                connect(btn, &QPushButton::clicked, this, createEDLViewForCell);
            } else if (vt.canonicalName == QLatin1String("Line Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::LINE_CHART);
                        });
            } else if (vt.canonicalName ==
                       QLatin1String("Line Chart View (Comparative)")) {
                connect(btn, &QPushButton::clicked, this,
                        [createComparativeForCell]() {
                            createComparativeForCell(
                                    vtkComparativeViewWidget::LINE_CHART);
                        });
            } else if (vt.canonicalName == QLatin1String("Bar Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::BAR_CHART);
                        });
            } else if (vt.canonicalName ==
                       QLatin1String("Bar Chart View (Comparative)")) {
                connect(btn, &QPushButton::clicked, this,
                        [createComparativeForCell]() {
                            createComparativeForCell(
                                    vtkComparativeViewWidget::BAR_CHART);
                        });
            } else if (vt.canonicalName == QLatin1String("Histogram View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::HISTOGRAM);
                        });
            } else if (vt.canonicalName == QLatin1String("Box Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::BOX_CHART);
                        });
            } else if (vt.canonicalName == QLatin1String("Image Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::IMAGE_CHART);
                        });
            } else if (vt.canonicalName == QLatin1String("Point Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::POINT_CHART);
                        });
            } else if (vt.canonicalName ==
                       QLatin1String("Quartile Chart View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::QUARTILE_CHART);
                        });
            } else if (vt.canonicalName ==
                       QLatin1String("Parallel Coordinates View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(
                                    vtkChartView::PARALLEL_COORDINATES);
                        });
            } else if (vt.canonicalName == QLatin1String("Plot Matrix View")) {
                connect(btn, &QPushButton::clicked, this,
                        [createChartForCell]() {
                            createChartForCell(vtkChartView::PLOT_MATRIX);
                        });
            } else if (vt.canonicalName == QLatin1String("Slice View")) {
                connect(btn, &QPushButton::clicked, this,
                        createSliceViewForCell);
            } else if (vt.canonicalName ==
                       QLatin1String("Orthographic Slice View")) {
                connect(btn, &QPushButton::clicked, this,
                        createOrthoSliceForCell);
#endif  // USE_VTK_BACKEND
            } else {
                connect(btn, &QPushButton::clicked, this,
                        createRenderViewForCell);
            }
        }
        btnLayout->addWidget(btn);
    }

    gridLayout->addWidget(actionsFrame, 1, 1);

    scrollArea->setWidget(scrollContent);
    outerLayout->addWidget(scrollArea, 1);

    return frame;
}

// ============================================================================
// Activation (ParaView pqMultiViewWidget::makeActive pattern)
// ============================================================================

bool ecvMultiViewWidget::eventFilter(QObject* caller, QEvent* evt) {
    if (evt->type() == QEvent::MouseButtonPress) {
        auto* wdg = qobject_cast<QWidget*>(caller);
        bool isInScope = wdg && (isAncestorOf(wdg) || wdg == this);
        if (!isInScope && m_poppedOut && m_popoutWindow &&
            (m_popoutWindow->isAncestorOf(wdg) ||
             wdg == m_popoutWindow.data())) {
            isInScope = true;
        }
        if (isInScope) {
            for (auto it = m_cellFrames.begin(); it != m_cellFrames.end();
                 ++it) {
                QWidget* frame = it.value();
                if (frame && (frame == wdg || frame->isAncestorOf(wdg))) {
                    makeActive(frame);
                    break;
                }
            }
        }
    } else if (evt->type() == QEvent::Close &&
               caller == m_popoutWindow.data()) {
        togglePopout();
    }
    return QWidget::eventFilter(caller, evt);
}

void ecvMultiViewWidget::makeActive(QWidget* frame) {
    if (m_activeFrame == frame) return;
    m_activeFrame = frame;

    ecvGenericGLDisplay* view = nullptr;
    if (m_layout && frame) {
        int location = findLocationForFrame(frame);
        if (location >= 0) {
            view = m_layout->getView(location);
        }
    }

    if (view) {
        ecvViewManager::instance().setActiveView(view);
    } else {
#ifdef USE_VTK_BACKEND
        auto* compView =
                frame ? frame->findChild<vtkComparativeViewWidget*>() : nullptr;
        if (!compView && frame)
            compView = qobject_cast<vtkComparativeViewWidget*>(frame);
        if (compView) {
            if (vtkGLView* sub = compView->activeSubView()) {
                ecvViewManager::instance().setActiveView(sub);
            }
        } else
#endif
        {
            updateFrameHighlighting();
        }
    }

    emit frameActivated();
}

void ecvMultiViewWidget::makeFrameActive() {
    if (m_activeFrame) {
        // Tab switch: frame already selected, but ecvViewManager may still
        // point at a different tab's view.  Force-sync the active view.
        ecvGenericGLDisplay* view = nullptr;
        if (m_layout) {
            int location = findLocationForFrame(m_activeFrame);
            if (location >= 0) {
                view = m_layout->getView(location);
            }
        }
        if (view) {
            ecvViewManager::instance().setActiveView(view);
        }
        return;
    }

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        if (it.value()) {
            makeActive(it.value());
            return;
        }
    }
}

void ecvMultiViewWidget::redrawAllViews() {
    QTimer::singleShot(0, this, [this]() {
        for (auto it = m_viewFrames.constBegin(); it != m_viewFrames.constEnd();
             ++it) {
            if (auto* display = it.key()) {
#ifdef USE_VTK_BACKEND
                auto* glView = dynamic_cast<vtkGLView*>(display);
                if (glView) {
                    glView->redraw(false, true);
                }
#endif
            }
        }
    });
}

void ecvMultiViewWidget::markActive(ecvGenericGLDisplay* view) {
    QWidget* frame = findFrameForView(view);
    if (frame) {
        m_activeFrame = frame;
    }
    updateFrameHighlighting();
}

void ecvMultiViewWidget::updateFrameHighlighting() {
    QColor activeColor = palette().link().color();
    QString activeSS = QString("QFrame#CentralWidgetFrame "
                               "{ border: 2px solid rgb(%1, %2, %3); }")
                               .arg(activeColor.red())
                               .arg(activeColor.green())
                               .arg(activeColor.blue());

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        QWidget* f = it.value();
        if (!f) continue;
        auto* contentFrame = f->findChild<QFrame*>("CentralWidgetFrame");
        if (!contentFrame) continue;

        bool isActive = (f == m_activeFrame);
        contentFrame->setStyleSheet(isActive ? activeSS : QString());

        auto* titleLabel = f->findChild<QLabel*>("ViewTitleLabel");
        if (titleLabel) {
            QString plain = titleLabel->property("plainTitle").toString();
            if (plain.isEmpty()) plain = titleLabel->text();
            titleLabel->setText(
                    isActive ? QString("<b><u>%1</u></b>").arg(plain) : plain);
        }
    }
}

// ============================================================================
// Popout (ParaView pqMultiViewWidget::togglePopout pattern)
// ============================================================================

bool ecvMultiViewWidget::togglePopout() {
    m_poppedOut = !m_poppedOut;

    if (m_poppedOut) {
        if (!m_popoutWindow) {
            auto* win = new QWidget(this, Qt::Window | Qt::CustomizeWindowHint |
                                                  Qt::WindowTitleHint |
                                                  Qt::WindowMaximizeButtonHint |
                                                  Qt::WindowCloseButtonHint);
            win->setObjectName("PopoutWindow");
            auto* wl = new QVBoxLayout(win);
            wl->setContentsMargins(0, 0, 0, 0);
            win->resize(this->size());
            m_popoutWindow.reset(win);
        }

        QString title = m_layout ? m_layout->name() : tr("Layout");
        m_popoutWindow->setWindowTitle(title);

        layout()->removeWidget(m_contentContainer);
        layout()->addWidget(m_popoutPlaceholder.data());
        m_popoutPlaceholder->show();

        m_popoutWindow->layout()->addWidget(m_contentContainer);
        m_popoutWindow->show();
    } else {
        Q_ASSERT(m_popoutWindow);
        m_popoutWindow->hide();
        m_popoutWindow->layout()->removeWidget(m_contentContainer);

        m_popoutPlaceholder->hide();
        layout()->removeWidget(m_popoutPlaceholder.data());
        layout()->addWidget(m_contentContainer);
    }
    return m_poppedOut;
}

// ============================================================================
// Split / Close / Maximize
// ============================================================================

void ecvMultiViewWidget::splitImpl(QWidget* frame, int dir) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    QWidget* cellFrame = m_cellFrames.value(location);
    bool isEmptyCell =
            cellFrame && cellFrame->property("IS_EMPTY_CELL").toBool();
    bool hasNonGLFrame = cellFrame && !isEmptyCell &&
                         (m_viewFrames.key(cellFrame, nullptr) == nullptr);
    ecvGenericGLDisplay* origView = m_layout->getView(location);

    disconnect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
               &ecvMultiViewWidget::reload);

    int child1 = m_layout->split(
            location, static_cast<ecvViewLayoutProxy::Direction>(dir));
    if (child1 < 0) {
        connect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
                &ecvMultiViewWidget::reload);
        return;
    }

    if (hasNonGLFrame) {
        QWidget* nonGLFrame = m_cellFrames.take(location);
        if (nonGLFrame) {
            m_cellFrames[child1] = nonGLFrame;
            nonGLFrame->setProperty("CELL_INDEX", child1);
        }
    }

    connect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
            &ecvMultiViewWidget::reload);
    reload();
}

void ecvMultiViewWidget::onSplitHorizontal(QWidget* frame) {
    splitImpl(frame, static_cast<int>(ecvViewLayoutProxy::HORIZONTAL));
}

void ecvMultiViewWidget::onSplitVertical(QWidget* frame) {
    splitImpl(frame, static_cast<int>(ecvViewLayoutProxy::VERTICAL));
}

void ecvMultiViewWidget::onCloseView(QWidget* frame) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* view = m_layout->getView(location);

    if (view) {
        emit viewClosing(view);
    }

    disconnect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
               &ecvMultiViewWidget::reload);

    m_layout->removeViewAt(location);

    int parentLocation = -1;
    if (location != 0) {
        parentLocation = ecvViewLayoutProxy::parent(location);
        int sibling =
                (location == ecvViewLayoutProxy::firstChild(parentLocation))
                        ? ecvViewLayoutProxy::secondChild(parentLocation)
                        : ecvViewLayoutProxy::firstChild(parentLocation);

        QWidget* sibFrame = m_cellFrames.value(sibling);
        bool sibIsEmptyCell =
                sibFrame && sibFrame->property("IS_EMPTY_CELL").toBool();
        if (sibFrame && !sibIsEmptyCell &&
            !m_cellFrames.contains(parentLocation)) {
            m_cellFrames.take(sibling);
            m_cellFrames[parentLocation] = sibFrame;
            sibFrame->setProperty("CELL_INDEX", parentLocation);
        }

        m_layout->collapse(location);
    }

    connect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
            &ecvMultiViewWidget::reload);

    if (view) {
#ifdef USE_VTK_BACKEND
        auto* glView = dynamic_cast<vtkGLView*>(view);
        if (glView) {
            glView->setSceneDB(nullptr);
            emit glView->aboutToClose(glView);
            glView->disconnect();
            QWidget* vw = glView->asWidget();
            if (vw) {
                vw->setParent(nullptr);
                vw->hide();
            }
            m_cellFrames.remove(location);
            m_viewFrames.remove(view);
            QTimer::singleShot(0, this, [glView, vw]() {
                glView->deleteLater();
                if (vw) vw->deleteLater();
            });
        }
#endif
    } else {
#ifdef USE_VTK_BACKEND
        auto* compView =
                frame ? frame->findChild<vtkComparativeViewWidget*>() : nullptr;
        if (!compView && frame)
            compView = qobject_cast<vtkComparativeViewWidget*>(frame);
        if (compView) {
#ifdef USE_VTK_BACKEND
            if (auto* selCtrl = cvSelectionToolController::instance()) {
                compView->disconnectExternalHighlighter();
                if (auto* hl = selCtrl->highlighter()) {
                    auto* activeVis = dynamic_cast<Visualization::VtkVis*>(
                            hl->getVisualizer());
                    for (auto* sv : compView->subViews()) {
                        if (sv && sv->getVisualizer3D() == activeVis) {
                            selCtrl->setVisualizer(nullptr);
                            break;
                        }
                    }
                }
            }
#endif
            compView->shutdown();
            compView->disconnect();
        }

        auto* orthoView =
                frame ? frame->findChild<vtkOrthoSliceViewWidget*>() : nullptr;
        if (!orthoView && frame)
            orthoView = qobject_cast<vtkOrthoSliceViewWidget*>(frame);
        if (orthoView) orthoView->disconnect();
#endif
        m_cellFrames.remove(location);
        if (frame) {
            frame->setParent(nullptr);
            frame->hide();
            frame->deleteLater();
        }
    }

    reload();
    makeFrameActive();
}

void ecvMultiViewWidget::onMaximize(QWidget* frame) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    if (m_layout->maximizedCell() == location) {
        m_layout->restoreMaximizedState();
    } else {
        m_layout->maximizeCell(location);
    }
}

// ============================================================================
// Decoration visibility
// ============================================================================

void ecvMultiViewWidget::setDecorationsVisibility(bool visible) {
    if (m_decorationsVisible == visible) return;
    m_decorationsVisible = visible;

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        QWidget* f = it.value();
        if (!f) continue;
        auto* titleBar = f->findChild<QWidget*>("ViewTitleBar");
        if (titleBar) titleBar->setVisible(visible);
    }

    emit decorationsVisibilityChanged(visible);
}

void ecvMultiViewWidget::lockViewSize(const QSize& size) {
    QSize maxSz =
            size.isEmpty() ? QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX) : size;

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        QWidget* f = it.value();
        if (!f) continue;
        auto* contentFrame = f->findChild<QFrame*>("CentralWidgetFrame");
        if (contentFrame) contentFrame->setMaximumSize(maxSz);
    }
}

void ecvMultiViewWidget::reset() {
    if (m_layout) {
        destroyAllViews();
        m_layout->reset();
    }
}

QList<vtkGLView*> ecvMultiViewWidget::destroyAllViews() {
    QList<vtkGLView*> orphaned;
    if (!m_layout) return orphaned;

    if (m_layout) {
        disconnect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
                   &ecvMultiViewWidget::reload);
    }

    auto views = m_layout->getViews();
    for (auto* v : views) {
        emit viewClosing(v);
        m_layout->removeView(v);
#ifdef USE_VTK_BACKEND
        auto* glView = dynamic_cast<vtkGLView*>(v);
        if (glView) {
            emit glView->aboutToClose(glView);
            orphaned.append(glView);
        }
#endif
    }

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        QWidget* frame = it.value();
        if (frame) {
#ifdef USE_VTK_BACKEND
            auto* compView = frame->findChild<vtkComparativeViewWidget*>();
            if (!compView)
                compView = qobject_cast<vtkComparativeViewWidget*>(frame);
            if (compView) compView->shutdown();
#endif
            frame->setParent(nullptr);
            frame->hide();
            frame->deleteLater();
        }
    }
    m_cellFrames.clear();

    return orphaned;
}

// ============================================================================
// Convert Cell (ParaView "Convert To..." pattern)
// ============================================================================

void ecvMultiViewWidget::convertCell(int location,
                                     const QString& canonicalName) {
    if (!m_layout) return;

    auto findBtn = [this, location, &canonicalName]() -> QPushButton* {
        QWidget* emptyWidget = m_cellFrames.value(location);
        if (!emptyWidget) return nullptr;
        for (auto* btn : emptyWidget->findChildren<QPushButton*>()) {
            if (btn->property("VIEW_TYPE_ID").toString() == canonicalName)
                return btn;
        }
        return nullptr;
    };

    QTimer::singleShot(0, this, [this, location, canonicalName, findBtn]() {
        reload();
        QTimer::singleShot(50, this, [findBtn]() {
            auto* btn = findBtn();
            if (btn && btn->isEnabled()) {
                btn->click();
            }
        });
    });
}

// ============================================================================
// Helpers
// ============================================================================

QWidget* ecvMultiViewWidget::findFrameForView(ecvGenericGLDisplay* view) const {
    auto it = m_viewFrames.find(view);
    return (it != m_viewFrames.end()) ? it.value() : nullptr;
}

int ecvMultiViewWidget::findLocationForFrame(QWidget* frame) const {
    if (!frame) return -1;
    QVariant v = frame->property("CELL_INDEX");
    if (v.isValid()) return v.toInt();

    for (auto it = m_cellFrames.begin(); it != m_cellFrames.end(); ++it) {
        if (it.value() == frame) return it.key();
    }
    return -1;
}
