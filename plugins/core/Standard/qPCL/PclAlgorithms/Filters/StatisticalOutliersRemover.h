// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "BasePclModule.h"

class SORDialog;

class StatisticalOutliersRemover : public BasePclModule {
public:
    StatisticalOutliersRemover();
    virtual ~StatisticalOutliersRemover();

protected:
    int compute();
    int openInputDialog();
    void getParametersFromDialog();

    SORDialog* m_dialog;
    int m_k;
    float m_std;
};
