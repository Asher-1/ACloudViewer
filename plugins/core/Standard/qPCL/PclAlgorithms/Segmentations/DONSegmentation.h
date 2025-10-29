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

class DONSegmentationDlg;

//! SIFT keypoints extraction
class DONSegmentation : public BasePclModule {
public:
    DONSegmentation();
    virtual ~DONSegmentation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    DONSegmentationDlg* m_dialog;

    QString m_comparisonField;
    QStringList m_comparisonTypes;
    float m_smallScale;
    float m_largeScale;
    float m_minDonMagnitude;
    float m_maxDonMagnitude;

    int m_minClusterSize;
    int m_maxClusterSize;
    float m_clusterTolerance;
    bool m_randomClusterColor;
};
