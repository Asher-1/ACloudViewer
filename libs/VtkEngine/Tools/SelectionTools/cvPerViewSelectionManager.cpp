// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cvPerViewSelectionManager.h"

#include <ecvGenericGLDisplay.h>

#include <QMdiSubWindow>

cvPerViewSelectionManager::cvPerViewSelectionManager(QObject* parent)
    : QObject(parent) {}

QAction* cvPerViewSelectionManager::mirrorSimple(QWidget* parent,
                                                 QAction* global) {
    if (!global) return nullptr;
    auto* local = new QAction(global->icon(), global->toolTip(), parent);
    local->setObjectName(global->objectName());
    local->setCheckable(global->isCheckable());
    local->setChecked(global->isChecked());
    local->setEnabled(global->isEnabled());
    connect(global, &QAction::toggled, local, [local](bool c) {
        QSignalBlocker blk(local);
        local->setChecked(c);
    });
    connect(global, &QAction::changed, local,
            [global, local]() { local->setEnabled(global->isEnabled()); });
    return local;
}

QAction* cvPerViewSelectionManager::mirrorIsolated(QWidget* parent,
                                                   QAction* global,
                                                   QWidget* viewWidget) {
    if (!global) return nullptr;
    auto* local = new QAction(global->icon(), global->toolTip(), parent);
    local->setCheckable(true);
    local->setChecked(false);
    local->setEnabled(global->isEnabled());
    local->setObjectName(global->objectName());
    local->setProperty("viewWidget",
                       QVariant::fromValue(static_cast<void*>(viewWidget)));
    connect(global, &QAction::changed, local,
            [global, local]() { local->setEnabled(global->isEnabled()); });
    connect(global, &QAction::toggled, local, [local](bool checked) {
        if (!checked && local->isChecked()) {
            QSignalBlocker blk(local);
            local->setChecked(false);
        }
    });
    return local;
}

void cvPerViewSelectionManager::uncheckOtherViews(QWidget* viewWidget,
                                                  const QString& globalName) {
    if (!m_mdiArea) return;
    for (auto* sub : m_mdiArea->subWindowList()) {
        QWidget* frame = sub->widget();
        if (!frame) continue;
        for (auto* tb : frame->findChildren<QWidget*>("ViewSelectionToolBar")) {
            for (auto* btn : tb->findChildren<QToolButton*>()) {
                auto* act = btn->defaultAction();
                if (!act || !act->isCheckable()) continue;
                auto stored = act->property("viewWidget");
                if (!stored.isValid()) continue;
                auto* w = static_cast<QWidget*>(stored.value<void*>());
                if (w == viewWidget) continue;
                if (act->toolTip() == globalName && act->isChecked()) {
                    QSignalBlocker blk(act);
                    act->setChecked(false);
                }
            }
        }
    }
}

void cvPerViewSelectionManager::uncheckAllMirrors() {
    if (!m_mdiArea) return;
    for (auto* sub : m_mdiArea->subWindowList()) {
        QWidget* frame = sub->widget();
        if (!frame) continue;
        for (auto* tb :
             frame->findChildren<QWidget*>("ViewSelectionToolBar")) {
            for (auto* btn : tb->findChildren<QToolButton*>()) {
                auto* act = btn->defaultAction();
                if (!act || !act->isCheckable()) continue;
                if (!act->property("viewWidget").isValid()) continue;
                if (act->isChecked()) {
                    QSignalBlocker blk(act);
                    act->setChecked(false);
                }
            }
        }
    }
}

void cvPerViewSelectionManager::populateToolbar(
        QWidget* toolbar,
        QWidget* viewWidget,
        const cvSelectionToolController::SelectionActions& actions) {
    if (!toolbar) return;
    auto* tbLayout = toolbar->layout();
    if (!tbLayout) return;

    const int iconSz = PV_ICON_SIZE;

    auto addActionBtn = [toolbar, tbLayout,
                         iconSz](QAction* action) -> QToolButton* {
        if (!action) return nullptr;
        auto* btn = new QToolButton(toolbar);
        btn->setDefaultAction(action);
        btn->setAutoRaise(true);
        btn->setIconSize(QSize(iconSz, iconSz));
        btn->setFixedSize(iconSz + 6, iconSz + 6);
        tbLayout->addWidget(btn);
        return btn;
    };

    auto addSeparator = [toolbar, tbLayout]() {
        auto* sep = new QFrame(toolbar);
        sep->setFrameShape(QFrame::VLine);
        sep->setFrameShadow(QFrame::Sunken);
        sep->setFixedWidth(2);
        tbLayout->addWidget(sep);
    };

    ecvGenericGLDisplay* display = ecvGenericGLDisplay::FromWidget(viewWidget);

    auto activateView = [this, display]() {
        if (m_activateViewFn && display) m_activateViewFn(display);
    };

    auto syncToGlobal = [](QAction*, QAction* global, bool checked) {
        if (global->isChecked() != checked) {
            QSignalBlocker blk(global);
            global->setChecked(checked);
        }
        emit global->triggered(checked);
    };

    // --- Modifier actions (shared state) ---
    QAction* modifiers[] = {
            mirrorSimple(toolbar, actions.addSelection),
            mirrorSimple(toolbar, actions.subtractSelection),
            mirrorSimple(toolbar, actions.toggleSelection),
    };
    QAction* globals[] = {
            actions.addSelection,
            actions.subtractSelection,
            actions.toggleSelection,
    };
    for (int i = 0; i < 3; ++i) {
        if (!modifiers[i]) continue;
        addActionBtn(modifiers[i]);
        connect(modifiers[i], &QAction::toggled, this,
                [activateView, local = modifiers[i], g = globals[i],
                 syncToGlobal](bool c) {
                    activateView();
                    syncToGlobal(local, g, c);
                });
    }

    addSeparator();

    // --- Checkable selection-mode tools (isolated per view) ---
    QAction* selectionGlobals[] = {
            actions.selectSurfaceCells,
            actions.selectSurfacePoints,
            actions.selectFrustumCells,
            actions.selectFrustumPoints,
            actions.selectPolygonCells,
            actions.selectPolygonPoints,
            actions.selectBlocks,
            actions.selectFrustumBlocks,
            actions.interactiveSelectCells,
            actions.interactiveSelectPoints,
            actions.hoverCells,
            actions.hoverPoints,
    };

    for (auto* global : selectionGlobals) {
        if (!global) continue;
        QAction* local = mirrorIsolated(toolbar, global, viewWidget);
        if (!local) continue;
        addActionBtn(local);
        connect(local, &QAction::toggled, this,
                [this, activateView, local, toolbar, global, viewWidget,
                 syncToGlobal](bool checked) {
                    activateView();
                    if (checked) {
                        uncheckOtherViews(viewWidget, global->toolTip());
                        for (auto* btn :
                             toolbar->findChildren<QToolButton*>()) {
                            auto* act = btn->defaultAction();
                            if (!act || act == local || !act->isCheckable())
                                continue;
                            if (!act->property("viewWidget").isValid())
                                continue;
                            if (act->isChecked()) {
                                QSignalBlocker blk(act);
                                act->setChecked(false);
                            }
                        }
                    }
                    syncToGlobal(local, global, checked);
                });
    }

    addSeparator();

    // --- Non-checkable manipulation actions ---
    QAction* manipGlobals[] = {
            actions.growSelection,
            actions.shrinkSelection,
            actions.clearSelection,
    };
    for (auto* global : manipGlobals) {
        if (!global) continue;
        QAction* local = mirrorSimple(toolbar, global);
        if (!local) continue;
        addActionBtn(local);
        connect(local, &QAction::triggered, this, [activateView, global]() {
            activateView();
            global->trigger();
        });
    }
}
