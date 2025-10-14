// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QPixmap>
#include <QPointer>
#include <QToolBar>

#include "CVAppCommon.h"

/**
 * ecvCustomViewpointsToolbar is the toolbar that has buttons for using and
 * configuring custom views (aka camera positions)
 */
class QAction;

class CVAPPCOMMON_LIB_API ecvCustomViewpointsToolbar : public QToolBar {
    Q_OBJECT
    typedef QToolBar Superclass;

public:
    ecvCustomViewpointsToolbar(const QString& title, QWidget* parentObject = 0)
        : Superclass(title, parentObject), BasePixmap(64, 64) {
        this->constructor();
    }
    ecvCustomViewpointsToolbar(QWidget* parentObject = 0)
        : Superclass(parentObject), BasePixmap(64, 64) {
        this->constructor();
    }
    ~ecvCustomViewpointsToolbar() override = default;

protected slots:

    /**
     * Clear and recreate all custom viewpoint actions
     * based on current settings
     */
    void updateCustomViewpointActions();

    /**
     * Update the state of the toolbuttons
     * depending of the type of the current active view
     */
    void updateEnabledState();

    /**
     * Open the Custom Viewpoint
     * button dialog to configure the viewpoints
     * manually
     */
    void ConfigureCustomViewpoints();

    /**
     * Slot to apply a custom view point
     */
    void ApplyCustomViewpoint();

    /**
     * Slot to add current viewpoint
     * to a new custom viewpoint
     */
    void addCurrentViewpointToCustomViewpoints();

    /**
     * Slot to set a custom viewpoint
     * to a current viewpoint
     */
    void SetToCurrentViewpoint();

    /**
     * Slot to delete a custom view point
     */
    void DeleteCustomViewpoint();

private:
    Q_DISABLE_COPY(ecvCustomViewpointsToolbar)
    void constructor();

    QPointer<QAction> PlusAction;
    QPointer<QAction> ConfigAction;
    QPixmap BasePixmap;
    QPixmap PlusPixmap;
    QPixmap ConfigPixmap;
    QVector<QPointer<QAction>> ViewpointActions;
};
