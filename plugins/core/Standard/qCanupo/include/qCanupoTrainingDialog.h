// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_qCanupoTrainingDialog.h>

class ecvMainAppInterface;
class ccPointCloud;

//! CANUPO plugin's training dialog
class qCanupoTrainingDialog : public QDialog, public Ui::CanupoTrainingDialog {
    Q_OBJECT

public:
    //! Default constructor
    qCanupoTrainingDialog(ecvMainAppInterface* app);

    //! Get origin point cloud
    ccPointCloud* getOriginPointCloud();
    //! Get class #1 point cloud
    ccPointCloud* getClass1Cloud();
    //! Get class #2 point cloud
    ccPointCloud* getClass2Cloud();
    //! Get evaluation point cloud
    ccPointCloud* getEvaluationCloud();

    //! Loads parameters from persistent settings
    void loadParamsFromPersistentSettings();
    //! Saves parameters to persistent settings
    void saveParamsToPersistentSettings();

    //! Returns input scales
    bool getScales(std::vector<float>& scales) const;
    //! Returns the max number of threads to use
    int getMaxThreadCount() const;

    //! Returns the selected descriptor ID
    unsigned getDescriptorID() const;

protected slots:

    void onClassChanged(int);
    void onCloudChanged(int);

protected:
    //! Gives access to the application (data-base, UI, etc.)
    ecvMainAppInterface* m_app;

    // Returns whether the current parameters are valid or not
    bool validParameters() const;
};
