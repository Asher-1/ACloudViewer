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

class PoissonReconstructionDlg;

//! Poisson Reconstruction
class PoissonReconstruction : public BasePclModule {
public:
    PoissonReconstruction();
    virtual ~PoissonReconstruction();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    PoissonReconstructionDlg* m_dialog;
    int m_degree;
    int m_treeDepth;
    int m_isoDivideDepth;
    int m_solverDivideDepth;
    float m_scale;
    float m_samplesPerNode;

    bool m_useConfidence;
    bool m_useManifold;
    bool m_outputPolygons;

    int m_knn_radius;
    float m_normalSearchRadius;
    bool m_useKnn;
};
