// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_MARCHINGCUBE_HEADER
#define Q_PCL_PLUGIN_MARCHINGCUBE_HEADER

#include "BasePclModule.h"

// Qt
#include <QString>

class MarchingCubeDlg;

//! Pcl Grid Projection
class MarchingCubeReconstruction : public BasePclModule {
public:
    MarchingCubeReconstruction();
    virtual ~MarchingCubeReconstruction();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    MarchingCubeDlg* m_dialog;
    int m_marchingMethod;
    float m_epsilon;
    float m_isoLevel;
    float m_gridResolution;
    float m_percentageExtendGrid;

    int m_knn_radius;
    float m_normalSearchRadius;
    bool m_useKnn;
};

#endif  // Q_PCL_PLUGIN_MARCHINGCUBE_HEADER
