// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "BasePclModule.h"

// QT
#include <QString>

class EuclideanClusterDlg;

//! SIFT keypoints extraction
class EuclideanClusterSegmentation : public BasePclModule {
public:
    EuclideanClusterSegmentation();
    virtual ~EuclideanClusterSegmentation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    EuclideanClusterDlg* m_dialog;

    int m_minClusterSize;
    int m_maxClusterSize;
    float m_clusterTolerance;
    bool m_randomClusterColor;
};
