// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QHash>
#include <QMap>
#include <QPointer>
#include <QScopedPointer>
#include <QVector>
#include <QWidget>
#include <functional>

class ecvViewLayoutProxy;
class vtkGLView;
class ecvGenericGLDisplay;
class QSplitter;
class QFrame;
class QToolButton;

/// UI widget that mirrors a single ecvViewLayoutProxy (KD-tree layout).
///
/// Directly inspired by ParaView's pqMultiViewWidget:
///   - Subscribes to ecvViewLayoutProxy::layoutChanged()
///   - Rebuilds the QSplitter tree to match the KD-tree on every change
///   - Creates view frames for each leaf cell
///   - Handles click-to-activate via application-wide event filter
///   - Provides split/maximize/close/swap operations that go through the proxy
///
/// One ecvMultiViewWidget per Tab in ecvTabbedMultiViewWidget.
class ecvMultiViewWidget : public QWidget {
    Q_OBJECT

public:
    explicit ecvMultiViewWidget(QWidget* parent = nullptr);
    ~ecvMultiViewWidget() override;

    /// Set the layout proxy this widget mirrors. Triggers reload().
    void setLayoutManager(ecvViewLayoutProxy* layout);
    ecvViewLayoutProxy* layoutManager() const { return m_layout; }

    /// Get the list of all views currently in this layout.
    QList<ecvGenericGLDisplay*> viewProxies() const;

    /// Check if a view is assigned to this layout.
    bool isViewAssigned(ecvGenericGLDisplay* view) const;

    /// Returns the currently active view frame (may be null).
    QWidget* activeFrame() const { return m_activeFrame; }

    /// Returns the cell index of the active frame, or -1.
    int activeFrameLocation() const;

    /// Set the factory used to create new views for split operations.
    using ViewFactory = std::function<vtkGLView*()>;
    void setViewFactory(ViewFactory factory) { m_viewFactory = factory; }

    /// Set the callback for creating the title-bar/frame wrapper.
    using FrameFactory =
            std::function<QWidget*(QWidget* viewWidget, const QString& title)>;
    void setFrameFactory(FrameFactory factory) { m_frameFactory = factory; }

    /// Set a callback invoked after each frame is wired to a view.
    using FrameWiredCallback =
            std::function<void(QWidget* frame, ecvGenericGLDisplay* view)>;
    void setFrameWiredCallback(FrameWiredCallback cb) {
        m_frameWiredCallback = cb;
    }

    /// Find the cell index for a given frame widget.
    int findLocationForFrame(QWidget* frame) const;

    /// Whether view frame decorations (title bar) are visible.
    bool decorationsVisibility() const { return m_decorationsVisible; }

    /// Whether the layout is currently popped out into a separate window.
    bool isPoppedOut() const { return m_poppedOut; }

    /// Toggle popout: moves the layout content into a floating window
    /// (or restores it back). Returns true if now popped out.
    /// Mirrors ParaView's pqMultiViewWidget::togglePopout().
    bool togglePopout();

    /// Destroy all views in this layout.  Returns the list of vtkGLView
    /// pointers that were detached (caller may schedule deletion).
    QList<vtkGLView*> destroyAllViews();

signals:
    /// Emitted when the active frame changes (click on a view).
    void frameActivated();

    /// Emitted before a view is removed/destroyed from this layout.
    /// Receivers should perform cleanup (unregister, camera link removal,
    /// primary-view adoption, etc.) before the view is actually detached.
    void viewClosing(ecvGenericGLDisplay* view);

    void decorationsVisibilityChanged(bool visible);

public slots:
    /// Rebuild the entire widget tree from the layout proxy.
    void reload();

    /// Ensure some frame is active (used when tab is shown).
    void makeFrameActive();

    /// Show/hide title bars and decorations.
    void setDecorationsVisibility(bool visible);
    void normalizeViewFrameLayout();

    /// Schedule a deferred VTK redraw for every view in this layout.
    /// Used after tab switches to repaint VTK framebuffers that were
    /// invalidated while the tab was hidden.
    void redrawAllViews();

    /// Lock all view sizes to the given dimension (empty = unlock).
    void lockViewSize(const QSize& size);

    /// Reset the layout (single empty cell).
    void reset();

    /// Enable or disable all "Close View" buttons.
    /// Used by freezeUI to prevent view closure during plugin operations.
    void setViewCloseButtonsEnabled(bool enabled);

public slots:
    /// Make a specific frame the active one.
    void makeActive(QWidget* frame);

    /// Update the visual "active" highlight.
    void markActive(ecvGenericGLDisplay* view);

    /// Update all frame border highlighting based on m_activeFrame.
    void updateFrameHighlighting();

    /// Handle standard buttons (split/close/maximize).
    void onSplitHorizontal(QWidget* frame);
    void onSplitVertical(QWidget* frame);
    void onCloseView(QWidget* frame);
    void onMaximize(QWidget* frame);

    /// Convert an empty cell to a specific view type (ParaView "Convert
    /// To...").
    void convertCell(int location, const QString& viewTypeLabel);

protected:
    bool eventFilter(QObject* caller, QEvent* evt) override;

private:
    /// Common logic for horizontal/vertical split (auto-creates second view).
    void splitImpl(QWidget* frame, int dir);

    /// Recursively build the splitter tree for a KD-tree cell.
    QWidget* buildCell(int location);

    /// Create the placeholder widget shown for empty (no-view) cells.
    /// Provides a "Create View" button that fills the cell with a new view.
    QWidget* createEmptyCellWidget(int location);

    /// Find the frame for a given view.
    QWidget* findFrameForView(ecvGenericGLDisplay* view) const;

    ecvViewLayoutProxy* m_layout = nullptr;
    QWidget* m_activeFrame = nullptr;
    bool m_decorationsVisible = true;
    bool m_poppedOut = false;

    QScopedPointer<QWidget> m_popoutWindow;
    QScopedPointer<QWidget> m_popoutPlaceholder;
    QWidget* m_contentContainer = nullptr;

    ViewFactory m_viewFactory;
    FrameFactory m_frameFactory;
    FrameWiredCallback m_frameWiredCallback;

    QMap<int, QWidget*> m_cellFrames;
    QMap<ecvGenericGLDisplay*, QWidget*> m_viewFrames;

    QHash<int, QWidget*> m_preservedNonGLFrames;
};
