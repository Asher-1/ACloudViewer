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

class NurbsSurfaceDlg;

//! Greedy Triangulation
class NurbsSurfaceReconstruction : public BasePclModule {
public:
    NurbsSurfaceReconstruction();
    virtual ~NurbsSurfaceReconstruction();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    NurbsSurfaceDlg* m_dialog;
    bool m_useVoxelGrid;

    int m_order;
    int m_meshResolution;
    int m_curveResolution;
    int m_refinements;
    int m_iterations;
    bool m_twoDim;
    bool m_fitBSplineCurve;

    float m_interiorSmoothness;
    float m_interiorWeight;
    float m_boundarySmoothness;
    float m_boundaryWeight;
};
