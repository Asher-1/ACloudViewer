// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QJsonObject>
#include <QList>
#include <QPointer>
#include <QStyle>
#include <QTabBar>
#include <QTabWidget>
#include <functional>

class ecvMultiViewWidget;
class ecvViewLayoutProxy;
class vtkGLView;
class ecvGenericGLDisplay;
/// Tabbed container for multiple layout widgets.
///
/// Directly mirrors ParaView's pqTabbedMultiViewWidget:
///   - Each tab is one ecvMultiViewWidget backed by one ecvViewLayoutProxy
///   - "+" tab to create new layouts (positioned after last real tab)
///   - Per-tab popout and close buttons
///   - Tab right-click context menu (rename, close, equalize)
///   - Read-only mode (hides "+" tab and close buttons)
///   - Tab switch activates the appropriate view frame
///   - Full-screen mode for all views or active view only
///   - Manages layout proxy lifecycle
class ecvTabbedMultiViewWidget : public QWidget {
    Q_OBJECT
    Q_PROPERTY(bool readOnly READ readOnly WRITE setReadOnly)

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

    /// Number of real layout tabs (excludes the "+" tab).
    int tabCount() const;

    /// Returns the size of the current tab's content area.
    QSize clientSize() const;

    /// Whether frame decorations are visible.
    bool decorationsVisibility() const { return m_decorationsVisible; }

    /// When true, user cannot add/remove tabs (hides "+" tab and close
    /// buttons). Mirrors ParaView's readOnly property.
    void setReadOnly(bool val);
    bool readOnly() const { return m_readOnly; }

    /// Tab bar visibility, independent of decoration visibility.
    void setTabVisibility(bool visible);
    bool tabVisibility() const { return m_tabBarVisible; }

    /// Set the factory for creating new views.
    using ViewFactory = std::function<vtkGLView*()>;
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

    /// Save the full layout state (all tabs + structure) for session
    /// persistence.
    QJsonObject saveLayoutState() const;

    /// Restore layout state from a previously saved JSON object.
    bool restoreLayoutState(const QJsonObject& state);

signals:
    /// Forwarded from each ecvMultiViewWidget: a view is about to be
    /// removed/destroyed. Receivers should perform cleanup before the view
    /// is actually detached (unregister, camera-link removal, etc.).
    void viewClosing(ecvGenericGLDisplay* view);

    /// Emitted when the active tab changes.
    void currentTabChanged(int index);

    void viewSizeLocked(bool locked);
    void fullScreenEnabled(bool enabled);
    void fullScreenActiveViewEnabled(bool enabled);

public slots:
    /// Create a new tab with an empty layout. Returns the tab index.
    int createTab();

    /// Create a new tab with a view already in it. Returns the tab index.
    int createTabWithView(vtkGLView* view);

    /// Close a tab by index. If all real tabs are closed, a new one is
    /// automatically created (ParaView behavior).
    void closeTab(int index);

    /// Set the current tab.
    void setCurrentTab(int index);

    /// Show/hide all decorations (title bars + tab bar).
    void setDecorationsVisibility(bool visible);

    /// Toggle decoration visibility.
    void toggleWidgetDecoration();

    /// Lock view sizes.
    void lockViewSize(const QSize& size);

    /// Toggle fullscreen mode (reparents TabWidget into a fullscreen window).
    void toggleFullScreen();

    /// Toggle fullscreen for the active render view only (Ctrl+F11).
    void toggleFullScreenActiveView();

    /// Reset all tabs.
    void reset();

    /// Preview layout at specific size.
    QSize preview(const QSize& previewSize = QSize());

protected:
    bool eventFilter(QObject* obj, QEvent* evt) override;

protected slots:
    void onCurrentTabChanged(int index);
    void onTabCloseRequested(int index);

private:
    ecvMultiViewWidget* createMultiViewWidget(ecvViewLayoutProxy* layout);
    void addNewTabWidget();
    void removeNewTabWidget();
    void setupTabButtons(int tabIndex);
    int tabButtonIndex(QWidget* button, QTabBar::ButtonPosition pos) const;

    static constexpr int TAB_BUTTON_PIXMAP_SIZE = 16;
    static QString popoutTooltip(bool poppedOut);
    static QStyle::StandardPixmap popoutPixmap(bool poppedOut);

    QTabWidget* m_tabWidget;
    QWidget* m_newTabWidget = nullptr;
    QPointer<QWidget> m_fullScreenWindow;
    QPointer<QFrame> m_fullScreenActiveFrame;
    int m_layoutCounter = 0;
    bool m_decorationsVisible = true;
    bool m_readOnly = false;
    bool m_tabBarVisible = true;
    bool m_closingTab = false;  // guard: suppress auto-create during closeTab

    ViewFactory m_viewFactory;
    FrameFactory m_frameFactory;
    FrameWiredCallback m_frameWiredCallback;
};
