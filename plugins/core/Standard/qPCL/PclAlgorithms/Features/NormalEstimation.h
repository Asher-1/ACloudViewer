// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

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
