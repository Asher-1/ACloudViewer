// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QHBoxLayout>
#include <QMdiArea>
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
/// This class is stateless per-instance — each call to populateToolbar() wires
/// up a self-contained set of actions on the target toolbar widget.  Cross-view
/// uncheck operates by walking the QMdiArea to find sibling toolbars.
class QVTK_ENGINE_LIB_API cvPerViewSelectionManager : public QObject {
    Q_OBJECT

public:
    static constexpr int PV_ICON_SIZE = 24;

    explicit cvPerViewSelectionManager(QObject* parent = nullptr);

    /// The mdiArea pointer is used by uncheckOtherViews() to walk all open
    /// views.  Must be set before calling populateToolbar().
    void setMdiArea(QMdiArea* mdi) { m_mdiArea = mdi; }

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

    QMdiArea* m_mdiArea = nullptr;
    ActivateViewFn m_activateViewFn;
};
