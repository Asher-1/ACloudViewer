// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER
#define Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER

#include "BasePclModule.h"

class ProjectionFilterDlg;

//! Projection Filter
class ProjectionFilter : public BasePclModule {
public:
    ProjectionFilter();
    virtual ~ProjectionFilter();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int openInputDialog();
    virtual int checkParameters();
    virtual void getParametersFromDialog();
    virtual QString getErrorMessage(int errorCode);

    ProjectionFilterDlg* m_dialog;

    bool m_projectionMode;

    // projection parameters
    float m_coefficientA;
    float m_coefficientB;
    float m_coefficientC;
    float m_coefficientD;

    // boundary parameters
    bool m_useVoxelGrid;
    float m_leafSize;
    bool m_useKnn;
    int m_knn_radius;
    float m_normalSearchRadius;
    float m_boundarySearchRadius;
    float m_boundaryAngleThreshold;
};

#endif  // Q_PCL_PLUGIN_PROJECTIONFILTER_HEADER
