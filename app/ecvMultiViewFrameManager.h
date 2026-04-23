// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QObject>
#include <QSplitter>
#include <QTabBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <functional>

class ecvGLView;
class ecvGenericGLDisplay;

/// Multi-view frame manager — owns QMdiArea and all view-frame layout
/// operations.  Inspired by ParaView's pqMultiViewWidget +
/// pqTabbedMultiViewWidget, extracted from MainWindow to keep view-frame
/// management in a dedicated, testable class.
class ecvMultiViewFrameManager : public QObject {
    Q_OBJECT

public:
    explicit ecvMultiViewFrameManager(QWidget* parent);
    ~ecvMultiViewFrameManager() override;

    QMdiArea* mdiArea() const { return m_mdiArea; }

    /// Adopt an externally-owned QMdiArea (e.g. the one from MainWindow.ui).
    void setMdiArea(QMdiArea* area) { m_mdiArea = area; }

    // -- Frame creation --

    /// Build a view frame (title bar + content frame) for the given widget.
    /// If onPerViewToolbarReady is set, it is called with the toolbar widget
    /// and the inner view widget so the caller can populate per-view actions.
    QWidget* createViewFrame(
            QWidget* innerWidget,
            const QString& title,
            std::function<void(QWidget* toolbar, QWidget* viewWidget)>
                    perViewToolbarCallback = nullptr);

    /// Create a new MDI sub-window wrapping the given frame widget.
    QMdiSubWindow* addFrameToMdi(QWidget* frame, const QString& tabTitle);

    // -- Layout operations --

    void splitViewFrame(QWidget* frameToSplit,
                        Qt::Orientation orientation,
                        std::function<ecvGLView*()> viewFactory);

    void toggleMaximizeViewFrame(QWidget* frame, QToolButton* btn);

    void swapViewFrames(QWidget* frameA, QWidget* frameB);

    void equalizeSplitter(QSplitter* splitter, bool horizontal, bool vertical);

    void lockViewSize(const QSize& size);

    // -- Active frame tracking --

    void markActiveViewFrame(QWidget* activeViewWidget);

    // -- Tab bar --

    void updateTabBarVisibility();

    // -- Query --

    int layoutCounter() const { return m_layoutCounter; }
    int nextLayoutCounter() { return ++m_layoutCounter; }
    QSize lockedViewSize() const { return m_lockedViewSize; }

signals:
    /// Emitted after a view frame is fully constructed (before show).
    void viewFrameCreated(QWidget* frame, QWidget* innerWidget);

    /// Emitted when a view frame close button is clicked. The receiver
    /// should handle cleanup (unregister view, camera link, etc.).
    void viewFrameCloseRequested(QWidget* frame);

    /// Emitted when a split creates a new view (the newFrame/view are ready).
    void splitViewCreated(QWidget* newFrame, ecvGLView* newView);

    /// Emitted when the active frame highlight changes.
    void activeFrameChanged(QWidget* activeViewWidget);

    /// Emitted when the user right-click-renames a view.
    void viewRenamed(QWidget* frame, const QString& newName);

private:
    static constexpr int PV_ICON_SIZE = 24;

    QWidget* m_parentWidget = nullptr;
    QMdiArea* m_mdiArea = nullptr;
    int m_layoutCounter = 0;
    QSize m_lockedViewSize;
};
