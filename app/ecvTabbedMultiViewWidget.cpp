// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvTabbedMultiViewWidget.h"

#include "ecvMultiViewWidget.h"

#include <ecvGLView.h>
#include <ecvViewLayoutProxy.h>
#include <ecvViewManager.h>

#include <QHBoxLayout>
#include <QInputDialog>
#include <QMenu>
#include <QTabBar>
#include <QToolButton>
#include <QVBoxLayout>

ecvTabbedMultiViewWidget::ecvTabbedMultiViewWidget(QWidget* parent)
    : QWidget(parent) {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    m_tabWidget = new QTabWidget(this);
    m_tabWidget->setTabsClosable(true);
    m_tabWidget->setMovable(true);
    m_tabWidget->setDocumentMode(true);
    mainLayout->addWidget(m_tabWidget);

    connect(m_tabWidget, &QTabWidget::currentChanged, this,
            &ecvTabbedMultiViewWidget::onCurrentTabChanged);
    connect(m_tabWidget, &QTabWidget::tabCloseRequested, this,
            &ecvTabbedMultiViewWidget::onTabCloseRequested);

    setupPlusButton();

    m_tabWidget->tabBar()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_tabWidget->tabBar(), &QWidget::customContextMenuRequested, this,
            [this](const QPoint& pos) {
                QTabBar* tb = m_tabWidget->tabBar();
                int tabIdx = tb->tabAt(pos);
                if (tabIdx < 0) return;

                QMenu menu;

                menu.addAction(tr("Rename Tab"), [this, tabIdx]() {
                    bool ok = false;
                    QString current = m_tabWidget->tabText(tabIdx);
                    QString newName = QInputDialog::getText(
                            this, tr("Rename Tab"), tr("New name:"),
                            QLineEdit::Normal, current, &ok);
                    if (ok && !newName.isEmpty()) {
                        m_tabWidget->setTabText(tabIdx, newName);
                        auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                                m_tabWidget->widget(tabIdx));
                        if (mvw && mvw->layoutManager()) {
                            mvw->layoutManager()->setName(newName);
                        }
                    }
                });

                auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                        m_tabWidget->widget(tabIdx));
                if (mvw && mvw->layoutManager()) {
                    auto* eqMenu = menu.addMenu(tr("Equalize"));
                    eqMenu->addAction(tr("Horizontal"), [mvw]() {
                        mvw->layoutManager()->equalize(
                                ecvViewLayoutProxy::HORIZONTAL);
                    });
                    eqMenu->addAction(tr("Vertical"), [mvw]() {
                        mvw->layoutManager()->equalize(
                                ecvViewLayoutProxy::VERTICAL);
                    });
                    eqMenu->addAction(tr("Both"), [mvw]() {
                        mvw->layoutManager()->equalize();
                    });
                }

                if (m_tabWidget->count() > 1) {
                    menu.addSeparator();
                    menu.addAction(tr("Close Tab"), [this, tabIdx]() {
                        closeTab(tabIdx);
                    });
                }

                menu.exec(tb->mapToGlobal(pos));
            });
}

ecvTabbedMultiViewWidget::~ecvTabbedMultiViewWidget() = default;

// ============================================================================
// Tab creation
// ============================================================================

int ecvTabbedMultiViewWidget::createTab() {
    auto* layout = new ecvViewLayoutProxy(this);
    ++m_layoutCounter;
    layout->setName(tr("Layout #%1").arg(m_layoutCounter));

    auto* mvw = createMultiViewWidget(layout);
    int idx = m_tabWidget->insertTab(m_tabWidget->count(), mvw,
                                     layout->name());
    m_tabWidget->setCurrentIndex(idx);
    return idx;
}

int ecvTabbedMultiViewWidget::createTabWithView(ecvGLView* view) {
    int idx = createTab();
    auto* mvw = qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(idx));
    if (mvw && mvw->layoutManager() && view) {
        mvw->layoutManager()->assignView(0, view);
    }
    return idx;
}

ecvMultiViewWidget* ecvTabbedMultiViewWidget::createMultiViewWidget(
        ecvViewLayoutProxy* layout) {
    auto* mvw = new ecvMultiViewWidget(m_tabWidget);
    mvw->setViewFactory(m_viewFactory);
    mvw->setFrameFactory(m_frameFactory);
    mvw->setFrameWiredCallback(m_frameWiredCallback);
    mvw->setLayoutManager(layout);

    connect(mvw, &ecvMultiViewWidget::frameActivated, this, [this, mvw]() {
        int idx = m_tabWidget->indexOf(mvw);
        if (idx >= 0 && m_tabWidget->currentIndex() != idx) {
            m_tabWidget->setCurrentIndex(idx);
        }
    });

    connect(layout, &ecvViewLayoutProxy::nameChanged, this,
            [this, mvw](const QString& name) {
                int idx = m_tabWidget->indexOf(mvw);
                if (idx >= 0) m_tabWidget->setTabText(idx, name);
            });

    connect(mvw, &ecvMultiViewWidget::viewClosing, this,
            &ecvTabbedMultiViewWidget::viewClosing);

    return mvw;
}

// ============================================================================
// Tab operations
// ============================================================================

void ecvTabbedMultiViewWidget::closeTab(int index) {
    if (index < 0 || index >= m_tabWidget->count()) return;
    if (m_tabWidget->count() <= 1) return;

    auto* mvw = qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(index));
    if (mvw) {
        mvw->destroyAllViews();
        auto* layout = mvw->layoutManager();
        m_tabWidget->removeTab(index);
        mvw->deleteLater();
        if (layout) layout->deleteLater();
    }
}

void ecvTabbedMultiViewWidget::setCurrentTab(int index) {
    m_tabWidget->setCurrentIndex(index);
}

void ecvTabbedMultiViewWidget::onCurrentTabChanged(int index) {
    auto* mvw =
            qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(index));
    if (mvw) {
        mvw->makeFrameActive();
    }
}

void ecvTabbedMultiViewWidget::onTabCloseRequested(int index) {
    closeTab(index);
}

// ============================================================================
// Query
// ============================================================================

ecvViewLayoutProxy* ecvTabbedMultiViewWidget::layoutProxy() const {
    auto* mvw = currentMultiView();
    return mvw ? mvw->layoutManager() : nullptr;
}

ecvMultiViewWidget* ecvTabbedMultiViewWidget::currentMultiView() const {
    return qobject_cast<ecvMultiViewWidget*>(m_tabWidget->currentWidget());
}

ecvMultiViewWidget* ecvTabbedMultiViewWidget::findTab(
        ecvViewLayoutProxy* layout) const {
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw && mvw->layoutManager() == layout) return mvw;
    }
    return nullptr;
}

QList<ecvViewLayoutProxy*> ecvTabbedMultiViewWidget::allLayouts() const {
    QList<ecvViewLayoutProxy*> result;
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw && mvw->layoutManager())
            result.append(mvw->layoutManager());
    }
    return result;
}

QList<ecvGenericGLDisplay*> ecvTabbedMultiViewWidget::allViews() const {
    QList<ecvGenericGLDisplay*> result;
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw) result.append(mvw->viewProxies());
    }
    return result;
}

int ecvTabbedMultiViewWidget::tabCount() const {
    return m_tabWidget->count();
}

// ============================================================================
// Decorations
// ============================================================================

void ecvTabbedMultiViewWidget::setDecorationsVisibility(bool visible) {
    m_decorationsVisible = visible;
    m_tabWidget->tabBar()->setVisible(visible);

    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw) mvw->setDecorationsVisibility(visible);
    }
}

void ecvTabbedMultiViewWidget::lockViewSize(const QSize& size) {
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw) mvw->lockViewSize(size);
    }
    emit viewSizeLocked(!size.isEmpty());
}

void ecvTabbedMultiViewWidget::toggleFullScreen() {
    QWidget* topLevel = window();
    if (topLevel->isFullScreen()) {
        topLevel->showNormal();
    } else {
        topLevel->showFullScreen();
    }
    emit fullScreenEnabled(topLevel->isFullScreen());
}

void ecvTabbedMultiViewWidget::reset() {
    while (m_tabWidget->count() > 1) {
        closeTab(m_tabWidget->count() - 1);
    }
    auto* mvw = currentMultiView();
    if (mvw) mvw->reset();
}

QSize ecvTabbedMultiViewWidget::preview(const QSize& previewSize) {
    if (previewSize.isEmpty()) {
        for (int i = 0; i < m_tabWidget->count(); ++i) {
            auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(i));
            if (mvw) mvw->lockViewSize(QSize());
        }
        return QSize();
    }
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw) mvw->lockViewSize(previewSize);
    }
    return previewSize;
}

// ============================================================================
// "+" button on tab bar
// ============================================================================

void ecvTabbedMultiViewWidget::setupPlusButton() {
    m_plusButton = new QToolButton(m_tabWidget);
    m_plusButton->setObjectName("TabBarPlusButton");
    m_plusButton->setText("+");
    m_plusButton->setAutoRaise(true);
    m_plusButton->setToolTip(tr("New Layout Tab"));

    connect(m_plusButton, &QToolButton::clicked, this, [this]() {
        int idx = createTab();
        if (m_viewFactory) {
            auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(idx));
            if (mvw && mvw->layoutManager()) {
                auto* view = m_viewFactory();
                if (view) {
                    mvw->layoutManager()->assignView(0, view);
                }
            }
        }
    });

    m_tabWidget->setCornerWidget(m_plusButton, Qt::TopRightCorner);
}
