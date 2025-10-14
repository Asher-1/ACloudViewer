// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER
#define Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER

#include <ui_TemplateAlignmentDialog.h>

// Qt
#include <QDialog>

class ecvMainAppInterface;
class ccPointCloud;
class ccHObject;

//! CANUPO plugin's training dialog
class TemplateAlignmentDialog : public QDialog,
                                public Ui::TemplateAlignmentDialog {
    Q_OBJECT

public:
    //! Default constructor
    TemplateAlignmentDialog(ecvMainAppInterface* app);

    //! Get template #1 point cloud
    ccPointCloud* getTemplate1Cloud();
    //! Get template #2 point cloud
    ccPointCloud* getTemplate2Cloud();
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

    //! Returns the Maximum Iterations
    int getMaxIterations() const;

    float getVoxelGridLeafSize() const;

    //! Returns the Normal Radius
    float getNormalRadius() const;
    //! Returns the Feature Radius
    float getFeatureRadius() const;
    //! Returns the Minimum Sample Distance
    float getMinSampleDistance() const;
    //! Returns the Maximum Correspondence Distance
    float getMaxCorrespondenceDistance() const;

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

#endif  // Q_PCL_TEMPLATEALIGNMENT_DIALOG_HEADER
