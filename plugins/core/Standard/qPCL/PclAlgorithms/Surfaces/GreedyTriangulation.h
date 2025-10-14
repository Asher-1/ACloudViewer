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

class GreedyTriangulationDlg;

//! Greedy Triangulation
class GreedyTriangulation : public BasePclModule {
public:
    GreedyTriangulation();
    virtual ~GreedyTriangulation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    GreedyTriangulationDlg* m_dialog;
    int m_trigulationSearchRadius;
    float m_weightingFactor;
    int m_maxNearestNeighbors;
    int m_maxSurfaceAngle;
    int m_minAngle;
    int m_maxAngle;
    bool m_normalConsistency;

    int m_knn_radius;
    float m_normalSearchRadius;
    bool m_useKnn;
};
