// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef Q_PCL_PLUGIN_REGIONGROWING_HEADER
#define Q_PCL_PLUGIN_REGIONGROWING_HEADER

#include "BasePclModule.h"

// Qt
#include <QString>

class RegionGrowingSegmentationDlg;

//! Region Growing Segmentation
class RegionGrowingSegmentation : public BasePclModule {
public:
    RegionGrowingSegmentation();
    virtual ~RegionGrowingSegmentation();

    // inherited from BasePclModule
    virtual int compute();

protected:
    // inherited from BasePclModule
    virtual int checkSelected();
    virtual int openInputDialog();
    virtual void getParametersFromDialog();
    virtual int checkParameters();
    virtual QString getErrorMessage(int errorCode);

    RegionGrowingSegmentationDlg* m_dialog;

    bool m_basedRgb;

    // Basic Region Growing Segmentation Parameters
    int m_k_search;
    int m_min_cluster_size;
    int m_max_cluster_size;
    unsigned int m_neighbour_number;
    float m_smoothness_theta;
    float m_curvature;

    // Region Growing Segmentation Based on RGB Parameters
    int m_min_cluster_size_rgb;
    float m_neighbors_distance;
    float m_point_color_diff;
    float m_region_color_diff;
};

#endif  // Q_PCL_PLUGIN_REGIONGROWING_HEADER
