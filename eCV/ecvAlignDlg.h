// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ui_alignDlg.h>

#include <QDialog>

namespace cloudViewer {
class ReferenceCloud;
}

namespace Ui {
class AlignDialog;
}

class ccGenericPointCloud;

//! Rough registration dialog
class ccAlignDlg : public QDialog {
    Q_OBJECT

public:
    enum CC_SAMPLING_METHOD { NONE = 0, RANDOM, SPACE, OCTREE };

    ccAlignDlg(ccGenericPointCloud *data,
               ccGenericPointCloud *model,
               QWidget *parent = nullptr);
    virtual ~ccAlignDlg();

    unsigned getNbTries();
    double getOverlap();
    double getDelta();
    ccGenericPointCloud *getModelObject();
    ccGenericPointCloud *getDataObject();
    CC_SAMPLING_METHOD getSamplingMethod();
    bool isNumberOfCandidatesLimited();
    unsigned getMaxNumberOfCandidates();
    cloudViewer::ReferenceCloud *getSampledModel();
    cloudViewer::ReferenceCloud *getSampledData();

protected slots:
    void swapModelAndData();
    void modelSliderReleased();
    void dataSliderReleased();
    void modelSamplingRateChanged(double value);
    void dataSamplingRateChanged(double value);
    void estimateDelta();
    void changeSamplingMethod(int index);
    void toggleNbMaxCandidates(bool activ);

protected:
    //! 'Model' cloud (static)
    ccGenericPointCloud *modelObject;

    //! 'Data' cloud (static)
    ccGenericPointCloud *dataObject;

    void setColorsAndLabels();

    Ui::AlignDialog *m_ui;
};
