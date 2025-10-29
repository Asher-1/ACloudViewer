// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QLineEdit>
#include <QList>
#include <QPointer>
#include <QPushButton>
#include <QString>
#include <QStringList>

#include "CVAppCommon.h"

class pqCustomViewpointButtonDialogUI;
class vtkSMCameraConfigurationReader;

/*
 * @class pqCustomViewpointDialog
 * @brief Dialog for configuring custom view buttons.
 *
 * Provides the machinery for associating the current camera configuration
 * to a custom view button, and importing or exporting all of the custom view
 * button configurations.
 *
 * @section thanks Thanks
 * This class was contributed by SciberQuest Inc.
 *
 * @sa pqCameraDialog
 */
class CVAPPCOMMON_LIB_API ecvCustomViewpointButtonDlg : public QDialog {
    Q_OBJECT

public:
    /**
     * Create and initialize the dialog.
     */
    ecvCustomViewpointButtonDlg(QWidget* parent,
                                Qt::WindowFlags f,
                                QStringList& toolTips,
                                QStringList& configurations,
                                QString& currentConfig);

    ~ecvCustomViewpointButtonDlg() override;

    /**
     * Constant variable that contains the default name for the tool tips.
     */
    const static QString DEFAULT_TOOLTIP;

    /**
     * Constant variable that defines the minimum number of items.
     */
    const static int MINIMUM_NUMBER_OF_ITEMS;

    /**
     * Constant variable that defines the maximum number of items.
     */
    const static int MAXIMUM_NUMBER_OF_ITEMS;

    /**
     * Set the list of tool tips and configurations. This is the preferred way
     * of settings these as it supports changing the number of items.
     */
    void setToolTipsAndConfigurations(const QStringList& toolTips,
                                      const QStringList& configs);

    //@{
    /**
     * Set/get a list of tool tips, one for each button. The number of items in
     * the `toolTips` list must match the current number of tooltips being
     * shown. Use `setToolTipsAndConfigurations` to change the number of items.
     */
    void setToolTips(const QStringList& toolTips);
    QStringList getToolTips();
    //@}

    //@{
    /**
     * Set/get a list of camera configurations, one for each button. The number
     * of items in `configs` must match the current number of configs. Use
     * `setToolTipsAndConfigurations` to change the number of items.
     */
    void setConfigurations(const QStringList& configs);
    QStringList getConfigurations();
    //@}

    //@{
    /**
     * Set/get the current camera configuration.
     */
    void setCurrentConfiguration(const QString& config);
    QString getCurrentConfiguration();
    //@}

private slots:
    void appendRow();
    void importConfigurations();
    void exportConfigurations();
    void clearAll();

    void assignCurrentViewpoint();
    void deleteRow();

private:
    ecvCustomViewpointButtonDlg() {}
    QStringList Configurations;
    QString CurrentConfiguration;
    pqCustomViewpointButtonDialogUI* ui;

    friend class pqCustomViewpointButtonDialogUI;
};
