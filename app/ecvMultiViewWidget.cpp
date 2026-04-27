// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvMultiViewWidget.h"

#include <ecvGLView.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>

#include <QApplication>
#include <QEvent>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QSplitter>
#include <QVBoxLayout>

ecvMultiViewWidget::ecvMultiViewWidget(QWidget* parent) : QWidget(parent) {
    auto* rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(0, 0, 0, 0);
    rootLayout->setSpacing(0);

    qApp->installEventFilter(this);

    connect(&ecvViewManager::instance(),
            &ecvViewManager::activeViewChanged, this,
            [this](ecvGenericGLDisplay* newActive, ecvGenericGLDisplay*) {
                markActive(newActive);
            });
}

ecvMultiViewWidget::~ecvMultiViewWidget() {
    qApp->removeEventFilter(this);
}

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
    m_cellFrames.clear();
    m_viewFrames.clear();

    auto* rootLayout = layout();
    if (!rootLayout) return;

    QLayoutItem* child;
    while ((child = rootLayout->takeAt(0)) != nullptr) {
        if (child->widget()) {
            child->widget()->setParent(nullptr);
            child->widget()->deleteLater();
        }
        delete child;
    }

    if (!m_layout) return;

    QWidget* rootWidget = buildCell(0);
    if (rootWidget) {
        rootLayout->addWidget(rootWidget);
    }

    auto* activeView = ecvViewManager::instance().getActiveView();
    if (activeView && m_layout->containsView(activeView)) {
        markActive(activeView);
    } else {
        makeFrameActive();
    }
}

QWidget* ecvMultiViewWidget::buildCell(int location) {
    if (!m_layout) return nullptr;

    if (m_layout->isSplitCell(location)) {
        auto dir = m_layout->splitDirection(location);
        auto fraction = m_layout->splitFraction(location);

        auto* splitter = new QSplitter(
                dir == ecvViewLayoutProxy::HORIZONTAL ? Qt::Horizontal
                                                     : Qt::Vertical,
                this);
        splitter->setChildrenCollapsible(false);
        splitter->setProperty("CELL_INDEX", location);

        QWidget* left = buildCell(ecvViewLayoutProxy::firstChild(location));
        QWidget* right = buildCell(ecvViewLayoutProxy::secondChild(location));

        if (left) splitter->addWidget(left);
        if (right) splitter->addWidget(right);

        int total = (dir == ecvViewLayoutProxy::HORIZONTAL) ? 1000 : 1000;
        int leftSize = static_cast<int>(total * fraction);
        splitter->setSizes({leftSize, total - leftSize});

        connect(splitter, &QSplitter::splitterMoved, this,
                [this, location, splitter](int, int) {
                    if (!m_layout) return;
                    QList<int> sizes = splitter->sizes();
                    int total = sizes[0] + sizes[1];
                    if (total > 0) {
                        double frac =
                                static_cast<double>(sizes[0]) / total;
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
            leftHasMax = checkContains(
                    ecvViewLayoutProxy::firstChild(location), checkContains);
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
    } else {
        frame = new QWidget(this);
        frame->setMinimumSize(50, 50);
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
// Activation (ParaView pqMultiViewWidget::makeActive pattern)
// ============================================================================

bool ecvMultiViewWidget::eventFilter(QObject* caller, QEvent* evt) {
    if (evt->type() == QEvent::MouseButtonPress) {
        auto* wdg = qobject_cast<QWidget*>(caller);
        if (wdg && (isAncestorOf(wdg) || wdg == this)) {
            for (auto it = m_cellFrames.begin(); it != m_cellFrames.end();
                 ++it) {
                QWidget* frame = it.value();
                if (frame && (frame == wdg || frame->isAncestorOf(wdg))) {
                    makeActive(frame);
                    break;
                }
            }
        }
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
    if (m_activeFrame) return;

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
// Split / Close / Maximize
// ============================================================================

void ecvMultiViewWidget::onSplitHorizontal(QWidget* frame) {
    if (!m_layout || !m_viewFactory) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* newView = m_viewFactory();
    if (!newView) return;

    int child = m_layout->split(location, ecvViewLayoutProxy::HORIZONTAL);
    if (child >= 0) {
        int newCell = ecvViewLayoutProxy::secondChild(
                ecvViewLayoutProxy::parent(child));
        m_layout->assignView(newCell, newView);
    }
}

void ecvMultiViewWidget::onSplitVertical(QWidget* frame) {
    if (!m_layout || !m_viewFactory) return;
    int location = findLocationForFrame(frame);
    if (location < 0) return;

    auto* newView = m_viewFactory();
    if (!newView) return;

    int child = m_layout->split(location, ecvViewLayoutProxy::VERTICAL);
    if (child >= 0) {
        int newCell = ecvViewLayoutProxy::secondChild(
                ecvViewLayoutProxy::parent(child));
        m_layout->assignView(newCell, newView);
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

void ecvMultiViewWidget::destroyAllViews() {
    if (!m_layout) return;
    auto views = m_layout->getViews();
    for (auto* v : views) {
        emit viewClosing(v);
        m_layout->removeView(v);
        auto* glView = dynamic_cast<ecvGLView*>(v);
        if (glView) {
            emit glView->aboutToClose(glView);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

QWidget* ecvMultiViewWidget::findFrameForView(
        ecvGenericGLDisplay* view) const {
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
