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

class SACSegmentationDlg;

//! SIFT keypoints extraction
class SACSegmentation : public BasePclModule {
public:
    SACSegmentation();
    virtual ~SACSegmentation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    // int extractRecursive(
    //	PointCloudT::Ptr xyzCloud,
    //	PointCloudT::Ptr cloudRemained,
    //	std::vector<PointCloudT::Ptr> &cloudExtractions,
    //	bool recursive = false);

    SACSegmentationDlg* m_dialog;

    QString m_selectedModel;
    QString m_selectedMethod;

    int m_maxIterations;
    float m_probability;
    float m_minRadiusLimits;
    float m_maxRadiusLimits;
    float m_distanceThreshold;
    int m_methodType;
    int m_modelType;

    bool m_exportExtraction;
    bool m_exportRemaining;

    bool m_useVoxelGrid;
    float m_leafSize;
    bool m_recursiveMode;
    float m_normalDisWeight;
    float m_maxRemainingRatio;
};
