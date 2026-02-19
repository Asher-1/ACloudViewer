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
 * @class ecvCustomViewpointsToolbar
 * @brief Toolbar for managing custom camera viewpoints
 * 
 * Provides UI controls for saving, loading, and managing custom camera
 * positions/orientations. Users can:
 * - Save current camera position as a named viewpoint
 * - Quickly restore saved viewpoints via toolbar buttons
 * - Configure viewpoint properties (name, icon, camera parameters)
 * - Delete unwanted viewpoints
 * 
 * Viewpoints include complete camera state:
 * - Position and look-at direction
 * - Up vector
 * - Projection type (orthographic/perspective)
 * - Zoom/scale factors
 * 
 * This toolbar integrates with CloudViewer's settings system to persist
 * custom viewpoints across application sessions.
 * 
 * @see ecvCustomViewpointButtonDlg
 */
class QAction;

class CVAPPCOMMON_LIB_API ecvCustomViewpointsToolbar : public QToolBar {
    Q_OBJECT
    typedef QToolBar Superclass;

public:
    /**
     * @brief Constructor with title
     * @param title Toolbar title
     * @param parentObject Parent widget (optional)
     */
    ecvCustomViewpointsToolbar(const QString& title, QWidget* parentObject = 0)
        : Superclass(title, parentObject), BasePixmap(64, 64) {
        this->constructor();
    }
    
    /**
     * @brief Constructor with default title
     * @param parentObject Parent widget (optional)
     */
    ecvCustomViewpointsToolbar(QWidget* parentObject = 0)
        : Superclass(parentObject), BasePixmap(64, 64) {
        this->constructor();
    }
    
    /**
     * @brief Destructor
     */
    ~ecvCustomViewpointsToolbar() override = default;

protected slots:

    /**
     * @brief Update custom viewpoint actions
     * 
     * Clears and recreates all custom viewpoint buttons based on
     * current settings. Called when viewpoint list changes.
     */
    void updateCustomViewpointActions();

    /**
     * @brief Update button enabled states
     * 
     * Updates toolbar button states based on current active view type.
     * Some operations may be disabled for certain view types.
     */
    void updateEnabledState();

    /**
     * @brief Open viewpoint configuration dialog
     * 
     * Opens dialog for manually configuring custom viewpoints
     * (add, edit, delete, reorder, etc.).
     */
    void ConfigureCustomViewpoints();

    /**
     * @brief Apply selected custom viewpoint
     * 
     * Restores camera to the state saved in the selected viewpoint.
     * Triggered when user clicks a viewpoint button.
     */
    void ApplyCustomViewpoint();

    /**
     * @brief Add current viewpoint to custom list
     * 
     * Saves current camera position as a new custom viewpoint.
     * Opens dialog to set viewpoint name and properties.
     */
    void addCurrentViewpointToCustomViewpoints();

    /**
     * @brief Update viewpoint to current camera state
     * 
     * Updates an existing custom viewpoint with current camera position.
     */
    void SetToCurrentViewpoint();

    /**
     * @brief Delete custom viewpoint
     * 
     * Removes selected custom viewpoint from the list.
     */
    void DeleteCustomViewpoint();

private:
    Q_DISABLE_COPY(ecvCustomViewpointsToolbar)
    
    /**
     * @brief Common constructor logic
     * 
     * Shared initialization code called by all constructors.
     */
    void constructor();

    QPointer<QAction> PlusAction;                    ///< "Add viewpoint" action
    QPointer<QAction> ConfigAction;                  ///< "Configure" action
    QPixmap BasePixmap;                              ///< Base icon pixmap
    QPixmap PlusPixmap;                              ///< "Plus" icon pixmap
    QPixmap ConfigPixmap;                            ///< "Config" icon pixmap
    QVector<QPointer<QAction>> ViewpointActions;    ///< Custom viewpoint actions
};
