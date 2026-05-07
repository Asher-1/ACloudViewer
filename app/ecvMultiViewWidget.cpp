// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMultiViewWidget.h"

#include <ecvGLView.h>
#include <ecvRepresentationManager.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>

#include <QApplication>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QSplitter>
#include <QStyle>
#include <QTimer>
#include <QVBoxLayout>

namespace {
constexpr int SPLITTER_GAP = 4;  // ParaView PARAVIEW_DEFAULT_LAYOUT_SPACING
}

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

    m_cellFrames.clear();
    m_viewFrames.clear();
    m_activeFrame = nullptr;

    auto* containerLayout =
            m_contentContainer ? m_contentContainer->layout() : nullptr;
    if (!containerLayout) return;

    QLayoutItem* child;
    while ((child = containerLayout->takeAt(0)) != nullptr) {
        if (child->widget()) {
            child->widget()->setParent(nullptr);
            child->widget()->deleteLater();
        }
        delete child;
    }

    if (!m_layout) return;

    QWidget* rootWidget = buildCell(0);
    if (rootWidget) {
        containerLayout->addWidget(rootWidget);
    }

    auto* activeView = ecvViewManager::instance().getActiveView();
    if (activeView && m_layout->containsView(activeView)) {
        markActive(activeView);
    } else if (isVisible()) {
        makeFrameActive();
    }

    // After reparenting, every view needs a VTK render to refill its
    // framebuffer.  rebindToolsToActiveView() only redraws the *active*
    // view, leaving other visible panes blank/gray.  Schedule a
    // deferred redraw for every view in this layout.
    QTimer::singleShot(0, this, [this]() {
        for (auto it = m_viewFrames.constBegin();
             it != m_viewFrames.constEnd(); ++it) {
            if (auto* display = it.key()) {
                QWidget* vw = display->asWidget();
                if (vw && vw->isVisible()) {
                    auto* glView = dynamic_cast<ecvGLView*>(display);
                    if (glView) {
                        glView->redraw(false, true);
                    }
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

// ============================================================================
// Empty cell placeholder (ParaView pqEmptyView / "Create View" pattern)
// ============================================================================

QWidget* ecvMultiViewWidget::createEmptyCellWidget(int location) {
    auto* frame = new QFrame(this);
    frame->setFrameStyle(QFrame::StyledPanel | QFrame::Sunken);
    frame->setMinimumSize(50, 50);

    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(20, 20, 20, 20);

    layout->addStretch(1);

    auto* icon = new QLabel(frame);
    icon->setAlignment(Qt::AlignCenter);
    icon->setText(QStringLiteral("\xF0\x9F\x96\xBC"));
    icon->setStyleSheet(QStringLiteral("font-size: 32px;"));
    layout->addWidget(icon);

    auto* label = new QLabel(tr("No view created for this cell"), frame);
    label->setAlignment(Qt::AlignCenter);
    label->setWordWrap(true);
    label->setStyleSheet(
            QStringLiteral("color: palette(mid); font-size: 11pt;"));
    layout->addWidget(label);

    layout->addSpacing(10);

    auto* btnRow = new QHBoxLayout();
    btnRow->addStretch(1);

    auto* createBtn = new QPushButton(tr("Create Render View"), frame);
    createBtn->setIcon(frame->style()->standardIcon(QStyle::SP_DesktopIcon));
    createBtn->setIconSize(QSize(16, 16));
    createBtn->setCursor(Qt::PointingHandCursor);
    createBtn->setStyleSheet(QStringLiteral(
            "QPushButton { padding: 6px 16px; font-weight: bold; }"));

    connect(createBtn, &QPushButton::clicked, this, [this, location]() {
        if (!m_viewFactory || !m_layout) return;
        auto* newView = m_viewFactory();
        if (!newView) return;
        m_layout->assignView(location, newView);
    });

    btnRow->addWidget(createBtn);
    btnRow->addStretch(1);
    layout->addLayout(btnRow);

    layout->addStretch(1);

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

void ecvMultiViewWidget::markActive(ecvGenericGLDisplay* view) {
    QWidget* frame = findFrameForView(view);
    if (frame) {
        m_activeFrame = frame;
    }

    QColor activeColor = palette().link().color();
    QString activeSS = QString("QFrame#CentralWidgetFrame "
                               "{ color: rgb(%1, %2, %3); }")
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

// P6: Copy all per-view representations from sourceView to destView so the
// newly split pane starts with the same visual state as the original.
static void copyRepresentationsOnSplit(ecvGenericGLDisplay* sourceView,
                                       ecvGenericGLDisplay* destView) {
    if (!sourceView || !destView) return;
    auto& repMgr = ecvRepresentationManager::instance();
    auto reps = repMgr.getRepresentationsForView(sourceView);
    for (auto* srcRep : reps) {
        if (!srcRep || !srcRep->getEntity()) continue;
        auto* dst = repMgr.ensureRepresentation(srcRep->getEntity(), destView);
        if (dst) {
            dst->setProperties(srcRep->properties());
            if (srcRep->hasVisibilityOverride()) {
                dst->setVisible(srcRep->isVisible());
            }
        }
    }
}

void ecvMultiViewWidget::onSplitHorizontal(QWidget* frame) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* sourceView = m_layout->getView(location);

    int child = m_layout->split(location, ecvViewLayoutProxy::HORIZONTAL);
    if (child >= 0 && m_viewFactory) {
        auto* newView = m_viewFactory();
        if (newView) {
            int newCell = ecvViewLayoutProxy::secondChild(
                    ecvViewLayoutProxy::parent(child));
            m_layout->assignView(newCell, newView);
            copyRepresentationsOnSplit(sourceView, newView);
            ecvViewManager::instance().setActiveView(newView);
        }
    }
}

void ecvMultiViewWidget::onSplitVertical(QWidget* frame) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* sourceView = m_layout->getView(location);

    int child = m_layout->split(location, ecvViewLayoutProxy::VERTICAL);
    if (child >= 0 && m_viewFactory) {
        auto* newView = m_viewFactory();
        if (newView) {
            int newCell = ecvViewLayoutProxy::secondChild(
                    ecvViewLayoutProxy::parent(child));
            m_layout->assignView(newCell, newView);
            copyRepresentationsOnSplit(sourceView, newView);
            ecvViewManager::instance().setActiveView(newView);
        }
    }
}

void ecvMultiViewWidget::onCloseView(QWidget* frame) {
    if (!m_layout) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* view = m_layout->getView(location);

    if (view) {
        emit viewClosing(view);
    }

    m_layout->removeViewAt(location);

    if (location != 0) {
        m_layout->collapse(location);
    }

    if (view) {
        auto* glView = dynamic_cast<ecvGLView*>(view);
        if (glView) {
            emit glView->aboutToClose(glView);
            QWidget* vw = glView->asWidget();
            if (vw) {
                vw->setParent(nullptr);
                vw->hide();
            }
            QTimer::singleShot(0, this, [glView, vw]() {
                glView->deleteLater();
                if (vw) vw->deleteLater();
            });
        }
    }

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

QList<ecvGLView*> ecvMultiViewWidget::destroyAllViews() {
    QList<ecvGLView*> orphaned;
    if (!m_layout) return orphaned;

    if (m_layout) {
        disconnect(m_layout, &ecvViewLayoutProxy::layoutChanged, this,
                   &ecvMultiViewWidget::reload);
    }

    auto views = m_layout->getViews();
    for (auto* v : views) {
        emit viewClosing(v);
        m_layout->removeView(v);
        auto* glView = dynamic_cast<ecvGLView*>(v);
        if (glView) {
            emit glView->aboutToClose(glView);
            orphaned.append(glView);
        }
    }

    return orphaned;
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
