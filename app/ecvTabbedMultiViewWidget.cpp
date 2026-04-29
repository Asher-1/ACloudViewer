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

#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QShortcut>
#include <QTabBar>
#include <QTimer>
#include <QVBoxLayout>

ecvTabbedMultiViewWidget::ecvTabbedMultiViewWidget(QWidget* parent)
    : QWidget(parent) {
    auto* mainLayout = new QGridLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    m_tabWidget = new QTabWidget(this);
    m_tabWidget->setTabsClosable(false);
    m_tabWidget->setMovable(true);
    m_tabWidget->setDocumentMode(true);
    mainLayout->addWidget(m_tabWidget, 0, 0);

    m_tabWidget->tabBar()->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(m_tabWidget->tabBar(), &QWidget::customContextMenuRequested, this,
            [this](const QPoint& pos) {
                QTabBar* tb = m_tabWidget->tabBar();
                int tabIdx = tb->tabAt(pos);
                if (tabIdx < 0) return;
                if (m_tabWidget->widget(tabIdx) == m_newTabWidget) return;

                auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                        m_tabWidget->widget(tabIdx));
                if (!mvw || !mvw->layoutManager()) return;

                QMenu menu;

                menu.addAction(tr("Rename Tab"), [this, tabIdx]() {
                    bool ok = false;
                    QString current = m_tabWidget->tabText(tabIdx);
                    QString newName = QInputDialog::getText(
                            this, tr("Rename Tab"), tr("New name:"),
                            QLineEdit::Normal, current, &ok);
                    if (ok && !newName.isEmpty()) {
                        m_tabWidget->setTabText(tabIdx, newName);
                        auto* w = qobject_cast<ecvMultiViewWidget*>(
                                m_tabWidget->widget(tabIdx));
                        if (w && w->layoutManager()) {
                            w->layoutManager()->setName(newName);
                        }
                    }
                });

                if (!m_readOnly) {
                    auto* closeAct =
                            menu.addAction(tr("Close Tab"), [this, tabIdx]() {
                                closeTab(tabIdx);
                            });
                    Q_UNUSED(closeAct);
                }

                auto* eqMenu = menu.addMenu(tr("Equalize Views"));
                eqMenu->addAction(tr("Horizontally"), [mvw]() {
                    mvw->layoutManager()->equalize(
                            ecvViewLayoutProxy::HORIZONTAL);
                });
                eqMenu->addAction(tr("Vertically"), [mvw]() {
                    mvw->layoutManager()->equalize(
                            ecvViewLayoutProxy::VERTICAL);
                });
                eqMenu->addAction(tr("Both"),
                                  [mvw]() { mvw->layoutManager()->equalize(); });

                menu.exec(tb->mapToGlobal(pos));
            });

    addNewTabWidget();

    connect(m_tabWidget, &QTabWidget::currentChanged, this,
            &ecvTabbedMultiViewWidget::onCurrentTabChanged);

    connect(m_tabWidget, &QTabWidget::tabBarClicked, this,
            [this](int index) {
                if (m_tabWidget->currentIndex() == 0 &&
                    m_tabWidget->count() == 1) {
                    createTab();
                }
                Q_UNUSED(index);
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

    int insertPos = m_newTabWidget
                            ? m_tabWidget->indexOf(m_newTabWidget)
                            : m_tabWidget->count();
    if (insertPos < 0) insertPos = m_tabWidget->count();

    int idx = m_tabWidget->insertTab(insertPos, mvw, layout->name());
    setupTabButtons(idx);
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
    mvw->setDecorationsVisibility(m_decorationsVisible);

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
    if (m_tabWidget->widget(index) == m_newTabWidget) return;

    auto* mvw = qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(index));
    if (!mvw) return;

    QList<ecvGLView*> orphanedViews = mvw->destroyAllViews();

    for (auto* glView : orphanedViews) {
        if (!glView) continue;
        ecvViewManager::instance().unregisterView(glView);
        QWidget* vw = glView->asWidget();
        if (vw) {
            vw->setParent(nullptr);
            vw->hide();
        }
    }

    auto* layout = mvw->layoutManager();
    m_tabWidget->removeTab(index);
    mvw->deleteLater();
    if (layout) layout->deleteLater();

    int realCount = tabCount();
    if (!m_readOnly && realCount == 0) {
        m_layoutCounter = 0;
        int newIdx = createTab();
        if (m_viewFactory) {
            auto* mvw2 = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(newIdx));
            if (mvw2 && mvw2->layoutManager()) {
                auto* view = m_viewFactory();
                if (view) mvw2->layoutManager()->assignView(0, view);
            }
        }
    }

    if (!orphanedViews.isEmpty()) {
        QTimer::singleShot(0, this, [orphanedViews]() {
            for (auto* glView : orphanedViews) {
                if (glView) glView->deleteLater();
            }
        });
    }
}

void ecvTabbedMultiViewWidget::setCurrentTab(int index) {
    if (index >= 0 && index < m_tabWidget->count()) {
        m_tabWidget->setCurrentIndex(index);
    }
}

void ecvTabbedMultiViewWidget::onCurrentTabChanged(int index) {
    if (index < 0) return;

    QWidget* currentWidget = m_tabWidget->widget(index);
    if (auto* mvw = qobject_cast<ecvMultiViewWidget*>(currentWidget)) {
        mvw->makeFrameActive();
    } else if (currentWidget == m_newTabWidget &&
               m_tabWidget->count() > 1) {
        int newIdx = createTab();
        if (m_viewFactory) {
            auto* mvw2 = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(newIdx));
            if (mvw2 && mvw2->layoutManager()) {
                auto* view = m_viewFactory();
                if (view) mvw2->layoutManager()->assignView(0, view);
            }
        }
        setCurrentTab(newIdx);
        return;
    }

    emit currentTabChanged(index);
}

void ecvTabbedMultiViewWidget::onTabCloseRequested(int index) {
    closeTab(index);
}

// ============================================================================
// "+" tab management (ParaView pattern)
// ============================================================================

void ecvTabbedMultiViewWidget::addNewTabWidget() {
    if (m_newTabWidget) return;

    m_newTabWidget = new QWidget(m_tabWidget);
    m_newTabWidget->setObjectName("NewTabPlaceholder");
    m_tabWidget->addTab(m_newTabWidget, QStringLiteral("+"));

    int idx = m_tabWidget->indexOf(m_newTabWidget);
    m_tabWidget->tabBar()->setTabButton(idx, QTabBar::LeftSide, nullptr);
    m_tabWidget->tabBar()->setTabButton(idx, QTabBar::RightSide, nullptr);
    m_tabWidget->tabBar()->setTabToolTip(idx, tr("New Layout Tab"));
}

void ecvTabbedMultiViewWidget::removeNewTabWidget() {
    if (!m_newTabWidget) return;
    m_tabWidget->removeTab(m_tabWidget->indexOf(m_newTabWidget));
    delete m_newTabWidget;
    m_newTabWidget = nullptr;
}

// ============================================================================
// Tab buttons (popout + close per tab, ParaView pqTabWidget pattern)
// ============================================================================

void ecvTabbedMultiViewWidget::setupTabButtons(int tabIndex) {
    auto* popoutLabel = new QLabel();
    popoutLabel->setObjectName("popout");
    popoutLabel->setToolTip(popoutTooltip(false));
    popoutLabel->setPixmap(
            popoutLabel->style()
                    ->standardIcon(popoutPixmap(false))
                    .pixmap(TAB_BUTTON_PIXMAP_SIZE, TAB_BUTTON_PIXMAP_SIZE));
    m_tabWidget->tabBar()->setTabButton(tabIndex, QTabBar::LeftSide,
                                        popoutLabel);
    popoutLabel->installEventFilter(this);

    auto* closeLabel = new QLabel();
    closeLabel->setObjectName("close");
    closeLabel->setToolTip(tr("Close Tab"));
    closeLabel->setPixmap(
            closeLabel->style()
                    ->standardIcon(QStyle::SP_TitleBarCloseButton)
                    .pixmap(TAB_BUTTON_PIXMAP_SIZE, TAB_BUTTON_PIXMAP_SIZE));
    m_tabWidget->tabBar()->setTabButton(tabIndex, QTabBar::RightSide,
                                        closeLabel);
    closeLabel->installEventFilter(this);
    closeLabel->setVisible(!m_readOnly);
}

int ecvTabbedMultiViewWidget::tabButtonIndex(
        QWidget* button, QTabBar::ButtonPosition pos) const {
    auto* bar = m_tabWidget->tabBar();
    for (int i = 0; i < bar->count(); ++i) {
        if (bar->tabButton(i, pos) == button) return i;
    }
    return -1;
}

QString ecvTabbedMultiViewWidget::popoutTooltip(bool poppedOut) {
    return poppedOut ? tr("Bring popped out window back to the frame")
                     : tr("Pop out layout in separate window");
}

QStyle::StandardPixmap ecvTabbedMultiViewWidget::popoutPixmap(bool poppedOut) {
    return poppedOut ? QStyle::SP_TitleBarNormalButton
                     : QStyle::SP_TitleBarMaxButton;
}

bool ecvTabbedMultiViewWidget::eventFilter(QObject* obj, QEvent* evt) {
    if (evt->type() == QEvent::MouseButtonRelease &&
        qobject_cast<QLabel*>(obj)) {
        auto* me = dynamic_cast<QMouseEvent*>(evt);
        if (me && me->button() == Qt::LeftButton) {
            int closeIdx = tabButtonIndex(qobject_cast<QWidget*>(obj),
                                          QTabBar::RightSide);
            if (closeIdx != -1) {
                closeTab(closeIdx);
                return true;
            }

            int popoutIdx = tabButtonIndex(qobject_cast<QWidget*>(obj),
                                           QTabBar::LeftSide);
            if (popoutIdx != -1) {
                auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                        m_tabWidget->widget(popoutIdx));
                if (mvw) {
                    auto* label = qobject_cast<QLabel*>(obj);
                    bool poppedOut = mvw->togglePopout();
                    label->setPixmap(
                            label->style()
                                    ->standardIcon(popoutPixmap(poppedOut))
                                    .pixmap(TAB_BUTTON_PIXMAP_SIZE,
                                            TAB_BUTTON_PIXMAP_SIZE));
                    label->setToolTip(popoutTooltip(poppedOut));
                }
                return true;
            }
        }
    }
    return QWidget::eventFilter(obj, evt);
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
    return m_tabWidget->count() - (m_newTabWidget ? 1 : 0);
}

QSize ecvTabbedMultiViewWidget::clientSize() const {
    if (m_tabWidget->currentWidget()) {
        return m_tabWidget->currentWidget()->size();
    }
    return size();
}

// ============================================================================
// Read-only mode
// ============================================================================

void ecvTabbedMultiViewWidget::setReadOnly(bool val) {
    if (m_readOnly == val) return;
    m_readOnly = val;

    QList<QLabel*> closeLabels = m_tabWidget->findChildren<QLabel*>("close");
    for (QLabel* label : closeLabels) {
        label->setVisible(!val);
    }

    if (val) {
        removeNewTabWidget();
    } else {
        addNewTabWidget();
    }
}

// ============================================================================
// Decorations
// ============================================================================

void ecvTabbedMultiViewWidget::setTabVisibility(bool visible) {
    m_tabBarVisible = visible;
    m_tabWidget->tabBar()->setVisible(visible);
}

void ecvTabbedMultiViewWidget::setDecorationsVisibility(bool visible) {
    m_decorationsVisible = visible;
    m_tabWidget->tabBar()->setVisible(visible);

    for (int i = 0; i < m_tabWidget->count(); ++i) {
        auto* mvw =
                qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        if (mvw) mvw->setDecorationsVisibility(visible);
    }
}

void ecvTabbedMultiViewWidget::toggleWidgetDecoration() {
    setDecorationsVisibility(!m_decorationsVisible);
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
    if (m_fullScreenWindow) {
        m_fullScreenWindow->layout()->removeWidget(m_tabWidget);
        layout()->addWidget(m_tabWidget);
        delete m_fullScreenWindow;
        emit fullScreenEnabled(false);
    } else {
        auto* fsw = new QWidget(this, Qt::Window);
        m_fullScreenWindow = fsw;
        fsw->setObjectName("FullScreenWindow");
        layout()->removeWidget(m_tabWidget);

        auto* gl = new QGridLayout(fsw);
        gl->setSpacing(0);
        gl->setContentsMargins(0, 0, 0, 0);
        gl->addWidget(m_tabWidget, 0, 0);
        fsw->showFullScreen();
        fsw->show();

        auto* esc = new QShortcut(Qt::Key_Escape, fsw);
        connect(esc, &QShortcut::activated, this,
                &ecvTabbedMultiViewWidget::toggleFullScreen);
        auto* f11 = new QShortcut(QKeySequence(Qt::Key_F11), fsw);
        connect(f11, &QShortcut::activated, this,
                &ecvTabbedMultiViewWidget::toggleFullScreen);
        emit fullScreenEnabled(true);
    }

    setDecorationsVisibility(m_fullScreenWindow == nullptr);
}

void ecvTabbedMultiViewWidget::toggleFullScreenActiveView() {
    if (m_fullScreenWindow) {
        auto widgets = m_fullScreenWindow->findChildren<QWidget*>(
                QString(), Qt::FindDirectChildrenOnly);
        if (m_fullScreenActiveFrame) {
            for (auto* wid : widgets) {
                if (wid->objectName() == "QVTKWidgetCustom" ||
                    wid->inherits("QVTKOpenGLNativeWidget")) {
                    m_fullScreenActiveFrame->layout()->addWidget(wid);
                }
            }
        }
        layout()->addWidget(m_tabWidget);
        delete m_fullScreenWindow;
        m_fullScreenActiveFrame = nullptr;
        emit fullScreenActiveViewEnabled(false);
        return;
    }

    layout()->removeWidget(m_tabWidget);

    auto* fsw = new QWidget(this, Qt::Window);
    m_fullScreenWindow = fsw;
    fsw->setObjectName("FullScreenWindow");

    auto* gl = new QGridLayout(fsw);
    gl->setSpacing(0);
    gl->setContentsMargins(0, 0, 0, 0);

    auto* mvw = currentMultiView();
    QWidget* activeFrame = mvw ? mvw->activeFrame() : nullptr;
    if (!activeFrame) {
        delete m_fullScreenWindow;
        layout()->addWidget(m_tabWidget);
        return;
    }

    auto vtkWidgets = activeFrame->findChildren<QWidget*>();
    for (auto* wid : vtkWidgets) {
        if (wid->objectName() == "QVTKWidgetCustom" ||
            wid->inherits("QVTKOpenGLNativeWidget")) {
            m_fullScreenActiveFrame =
                    qobject_cast<QFrame*>(wid->parentWidget());
            gl->addWidget(wid, 0, 0);
            break;
        }
    }

    if (!m_fullScreenActiveFrame) {
        delete m_fullScreenWindow;
        layout()->addWidget(m_tabWidget);
        return;
    }

    fsw->showFullScreen();
    fsw->show();

    auto* esc = new QShortcut(Qt::Key_Escape, fsw);
    connect(esc, &QShortcut::activated, this,
            &ecvTabbedMultiViewWidget::toggleFullScreenActiveView);
    auto* ctrlF11 =
            new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_F11), fsw);
    connect(ctrlF11, &QShortcut::activated, this,
            &ecvTabbedMultiViewWidget::toggleFullScreenActiveView);
    emit fullScreenActiveViewEnabled(true);
}

void ecvTabbedMultiViewWidget::reset() {
    for (int cc = m_tabWidget->count() - 1; cc > 1; --cc) {
        if (m_tabWidget->widget(cc) != m_newTabWidget) {
            closeTab(cc);
        }
    }

    auto* mvw = currentMultiView();
    if (mvw) mvw->reset();
}

QSize ecvTabbedMultiViewWidget::preview(const QSize& previewSize) {
    if (auto* mvw = currentMultiView()) {
        mvw->lockViewSize(previewSize);
        return previewSize.isEmpty() ? QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)
                                     : previewSize;
    }
    return QSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
}

// ============================================================================
// Session persistence
// ============================================================================

QJsonObject ecvTabbedMultiViewWidget::saveLayoutState() const {
    QJsonObject state;
    state["tab_count"] = tabCount();
    state["current_tab"] = m_tabWidget->currentIndex();

    QJsonArray tabs;
    for (int i = 0; i < m_tabWidget->count(); ++i) {
        if (m_tabWidget->widget(i) == m_newTabWidget) continue;
        auto* mvw = qobject_cast<ecvMultiViewWidget*>(m_tabWidget->widget(i));
        QJsonObject tabState;
        if (mvw && mvw->layoutManager()) {
            tabState["layout"] = mvw->layoutManager()->saveState();
        }
        tabs.append(tabState);
    }
    state["tabs"] = tabs;
    return state;
}

bool ecvTabbedMultiViewWidget::restoreLayoutState(const QJsonObject& state) {
    int tc = state["tab_count"].toInt(1);
    int currentTab = state["current_tab"].toInt(0);
    QJsonArray tabs = state["tabs"].toArray();

    if (tc <= 1 && tabs.size() <= 1) {
        auto layoutObj = tabs.isEmpty()
                                 ? QJsonObject()
                                 : tabs[0].toObject()["layout"].toObject();
        auto cells = layoutObj["cells"].toArray();
        if (cells.size() <= 1) return false;
    }

    for (int i = tabCount() - 1; i >= 1; --i) {
        int widgetIdx = -1;
        int real = 0;
        for (int j = 0; j < m_tabWidget->count(); ++j) {
            if (m_tabWidget->widget(j) == m_newTabWidget) continue;
            if (real == i) {
                widgetIdx = j;
                break;
            }
            ++real;
        }
        if (widgetIdx >= 0) {
            auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(widgetIdx));
            if (mvw) {
                mvw->destroyAllViews();
                auto* layout = mvw->layoutManager();
                m_tabWidget->removeTab(widgetIdx);
                mvw->deleteLater();
                if (layout) layout->deleteLater();
            }
        }
    }

    // ParaView restores the layout STRUCTURE (splits, fractions, tab names)
    // but does NOT create views for every saved cell. Views are reconnected
    // by the caller (MainWindow assigns the primary view, user creates new
    // ones via "Create Render View" placeholders or the view factory).
    for (int i = 0; i < tc; ++i) {
        int tabIdx;
        if (i == 0) {
            tabIdx = 0;
        } else {
            tabIdx = createTab();
        }

        if (i < tabs.size()) {
            QJsonObject tabState = tabs[i].toObject();
            QJsonObject layoutState = tabState["layout"].toObject();

            auto* mvw = qobject_cast<ecvMultiViewWidget*>(
                    m_tabWidget->widget(tabIdx));
            if (mvw && mvw->layoutManager() && !layoutState.isEmpty()) {
                mvw->layoutManager()->loadState(layoutState);
            }
        }
    }

    if (currentTab >= 0 && currentTab < m_tabWidget->count())
        m_tabWidget->setCurrentIndex(currentTab);

    return true;
}
