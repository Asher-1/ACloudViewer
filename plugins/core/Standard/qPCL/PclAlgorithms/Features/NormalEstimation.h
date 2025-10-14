// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_NORMALESTIMATION_HEADER
#define Q_PCL_PLUGIN_NORMALESTIMATION_HEADER

#include "BasePclModule.h"

class NormalEstimationDialog;

class NormalEstimation : public BasePclModule {
public:
    NormalEstimation();
    virtual ~NormalEstimation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int openInputDialog();
    virtual void getParametersFromDialog();

    NormalEstimationDialog* m_dialog;
    int m_knn_radius;
    float m_radius;
    bool m_useKnn;
    bool m_overwrite_curvature;
};

#endif  // Q_PCL_PLUGIN_NORMALESTIMATION_HEADER
