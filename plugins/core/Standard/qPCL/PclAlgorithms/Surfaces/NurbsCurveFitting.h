// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "BasePclModule.h"

// Qt
#include <QString>

class NurbsCurveFittingDlg;

//! Greedy Triangulation
class NurbsCurveFitting : public BasePclModule {
public:
    NurbsCurveFitting();
    virtual ~NurbsCurveFitting();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    NurbsCurveFittingDlg* m_dialog;
    bool m_exportProjectedCloud;
    bool m_useVoxelGrid;
    int m_minimizationType;

    bool m_curveFitting3D;
    bool m_closed;

    int m_order;
    int m_curveResolution;
    int m_controlPoints;
    float m_curveSmoothness;
    float m_curveRscale;
};
