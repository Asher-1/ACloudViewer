// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_CorrespondenceMatchingDialog.h>

// Qt
#include <QDialog>

class ecvMainAppInterface;
class ccPointCloud;
class ccHObject;

//! CANUPO plugin's training dialog
class CorrespondenceMatchingDialog : public QDialog,
                                     public Ui::CorrespondenceMatchingDialog {
    Q_OBJECT

public:
    //! Default constructor
    CorrespondenceMatchingDialog(ecvMainAppInterface* app);

    //! Get model #1 point cloud
    ccPointCloud* getModel1Cloud();
    //! Get model #2 point cloud
    ccPointCloud* getModel2Cloud();

    ccPointCloud* getModelCloudByIndex(int index);

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

    float getVoxelGridLeafSize() const;

    //! Returns the Model Search Radius
    float getModelSearchRadius() const;
    //! Returns the Scene Search Radius
    float getSceneSearchRadius() const;
    //! Returns the Shot Descriptor Radius
    float getShotDescriptorRadius() const;
    //! Returns the normal KSearch
    float getNormalKSearch() const;

    bool getVerificationFlag() const;

    bool isGCActivated() const;

    float getGcConsensusSetResolution() const;
    float getGcMinClusterSize() const;

    float getHoughLRFRadius() const;
    float getHoughBinSize() const;
    float getHoughThreshold() const;

    void refreshCloudComboBox();

protected slots:
    void onCloudChanged(int);

protected:
    //! Gives access to the application (data-base, UI, etc.)
    ecvMainAppInterface* m_app;

    // Returns whether the current parameters are valid or not
    bool validParameters() const;

    QString getEntityName(ccHObject* obj);

    ccPointCloud* getCloudFromCombo(QComboBox* comboBox, ccHObject* dbRoot);
};
