// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QTabWidget>
#include <QList>
#include <functional>

class ecvMultiViewWidget;
class ecvViewLayoutProxy;
class ecvGLView;
class ecvGenericGLDisplay;
class QToolButton;

/// Tabbed container for multiple layout widgets.
///
/// Directly mirrors ParaView's pqTabbedMultiViewWidget:
///   - Each tab is one ecvMultiViewWidget backed by one ecvViewLayoutProxy
///   - "+" button to create new tabs
///   - Tab close / rename
///   - Tab switch activates the appropriate view frame
///   - Manages layout proxy lifecycle
class ecvTabbedMultiViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit ecvTabbedMultiViewWidget(QWidget* parent = nullptr);
    ~ecvTabbedMultiViewWidget() override;

    /// Get the current layout proxy (for the active tab).
    ecvViewLayoutProxy* layoutProxy() const;

    /// Get the multi-view widget for the current tab.
    ecvMultiViewWidget* currentMultiView() const;

    /// Find the multi-view widget for a specific layout proxy.
    ecvMultiViewWidget* findTab(ecvViewLayoutProxy* layout) const;

    /// Get all layout proxies.
    QList<ecvViewLayoutProxy*> allLayouts() const;

    /// Get all views across all tabs.
    QList<ecvGenericGLDisplay*> allViews() const;

    /// Tab count.
    int tabCount() const;

    /// Whether frame decorations are visible.
    bool decorationsVisibility() const { return m_decorationsVisible; }

    /// Set the factory for creating new views.
    using ViewFactory = std::function<ecvGLView*()>;
    void setViewFactory(ViewFactory factory) { m_viewFactory = factory; }

    /// Set the factory for creating view frames.
    using FrameFactory =
            std::function<QWidget*(QWidget* viewWidget, const QString& title)>;
    void setFrameFactory(FrameFactory factory) { m_frameFactory = factory; }

    /// Set the callback invoked when a frame is wired to a view.
    using FrameWiredCallback =
            std::function<void(QWidget* frame, ecvGenericGLDisplay* view)>;
    void setFrameWiredCallback(FrameWiredCallback cb) {
        m_frameWiredCallback = cb;
    }

    /// The underlying QTabWidget (for MainWindow to access tab bar, etc.).
    QTabWidget* tabWidget() const { return m_tabWidget; }

signals:
    /// Forwarded from each ecvMultiViewWidget: a view is about to be
    /// removed/destroyed. Receivers should perform cleanup before the view
    /// is actually detached (unregister, camera-link removal, etc.).
    void viewClosing(ecvGenericGLDisplay* view);

    void viewSizeLocked(bool locked);
    void fullScreenEnabled(bool enabled);

public slots:
    /// Create a new tab with an empty layout. Returns the tab index.
    int createTab();

    /// Create a new tab with a view already in it. Returns the tab index.
    int createTabWithView(ecvGLView* view);

    /// Close a tab by index.
    void closeTab(int index);

    /// Set the current tab.
    void setCurrentTab(int index);

    /// Show/hide all decorations (title bars + tab bar).
    void setDecorationsVisibility(bool visible);

    /// Lock view sizes.
    void lockViewSize(const QSize& size);

    /// Toggle fullscreen mode.
    void toggleFullScreen();

    /// Reset all tabs.
    void reset();

    /// Preview layout at specific size.
    QSize preview(const QSize& previewSize = QSize());

protected slots:
    void onCurrentTabChanged(int index);
    void onTabCloseRequested(int index);

private:
    ecvMultiViewWidget* createMultiViewWidget(ecvViewLayoutProxy* layout);
    void setupPlusButton();

    QTabWidget* m_tabWidget;
    QToolButton* m_plusButton;
    int m_layoutCounter = 0;
    bool m_decorationsVisible = true;

    ViewFactory m_viewFactory;
    FrameFactory m_frameFactory;
    FrameWiredCallback m_frameWiredCallback;
};
