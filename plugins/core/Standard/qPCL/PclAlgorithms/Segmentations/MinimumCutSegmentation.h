// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_MINIMUMCUT_HEADER
#define Q_PCL_PLUGIN_MINIMUMCUT_HEADER

#include "BasePclModule.h"

// Qt
#include <QString>

class MinimumCutSegmentationDlg;

//! Region Growing Segmentation
class MinimumCutSegmentation : public BasePclModule {
public:
    MinimumCutSegmentation();
    virtual ~MinimumCutSegmentation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    MinimumCutSegmentationDlg* m_dialog;

    bool m_colored;

    int m_neighboursNumber;
    float m_smoothSigma;
    float m_backWeightRadius;
    float m_foregroundWeight;

    float m_cx;
    float m_cy;
    float m_cz;
};

#endif  // Q_PCL_PLUGIN_MINIMUMCUT_HEADER
