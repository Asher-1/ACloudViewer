// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QObject>
#include <QSignalBlocker>
#include <QToolButton>
#include <functional>

#include "cvSelectionToolController.h"
#include "qVTK.h"

class ecvGenericGLDisplay;

/// Per-view selection toolbar builder and cross-view isolation manager.
///
/// Analogous to ParaView's pqStandardViewFrameActionsImplementation:
///   - Creates per-view QAction mirrors of the global selection actions
///   - Manages radio-button isolation (one tool active at a time per view)
///   - Unchecks mirrored actions in OTHER views when one view activates a tool
///
/// Cross-view uncheck walks a configurable root widget tree to find sibling
/// ViewSelectionToolBar instances.
class QVTK_ENGINE_LIB_API cvPerViewSelectionManager : public QObject {
    Q_OBJECT

public:
    static constexpr int PV_ICON_SIZE = 24;

    explicit cvPerViewSelectionManager(QObject* parent = nullptr);

    /// Set the root widget whose subtree contains all per-view
    /// ViewSelectionToolBar widgets. Both uncheckOtherViews() and
    /// uncheckAllMirrors() search from this root.
    void setViewRoot(QWidget* root) { m_viewRoot = root; }

    /// Callback type invoked when a per-view action activates — the caller
    /// (MainWindow) should rebind tools to the specified display.
    using ActivateViewFn = std::function<void(ecvGenericGLDisplay*)>;

    void setActivateViewCallback(ActivateViewFn fn) {
        m_activateViewFn = std::move(fn);
    }

    /// Populate the given toolbar widget with per-view selection buttons.
    /// @param toolbar      Target QWidget (must have a QHBoxLayout).
    /// @param viewWidget   The inner VTK render widget for this view.
    /// @param actions      The global selection actions to mirror.
    void populateToolbar(
            QWidget* toolbar,
            QWidget* viewWidget,
            const cvSelectionToolController::SelectionActions& actions);

    /// Uncheck all per-view mirror actions in every view toolbar.
    /// Called when ESC or disableAllTools clears the global actions with
    /// signals blocked — the per-view mirrors would otherwise stay checked.
    void uncheckAllMirrors();

private:
    QAction* mirrorSimple(QWidget* parent, QAction* global);
    QAction* mirrorIsolated(QWidget* parent,
                            QAction* global,
                            QWidget* viewWidget);
    void uncheckOtherViews(QWidget* viewWidget, const QString& globalName);

    QWidget* m_viewRoot = nullptr;
    ActivateViewFn m_activateViewFn;
};
