// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "cloudViewer/core/EigenConverter.h"
#include "cloudViewer/geometry/LineSet.h"
#include "cloudViewer/io/PointCloudIO.h"
#include "cloudViewer/pipelines/registration/PoseGraph.h"
#include "cloudViewer/t/pipelines/registration/Registration.h"
#include "cloudViewer/t/pipelines/slac/ControlGrid.h"
#include <FileSystem.h>

namespace cloudViewer {
namespace t {
namespace pipelines {
namespace slac {

/// \brief Visualize pairs with correspondences.
///
/// \param tpcd_i, source point cloud.
/// \param tpcd_j, target point cloud.
/// \param correspondences Putative correspondence between tcpd_i and tpcd_j.
/// \param T_ij Transformation from tpcd_i to tpcd_j. Use T_j.Inverse() @ T_i
/// (node transformation in a pose graph) to check global correspondences , and
/// T_ij (edge transformation) to check pairwise correspondences.
void VisualizePointCloudCorrespondences(const t::geometry::PointCloud& tpcd_i,
                                        const t::geometry::PointCloud& tpcd_j,
                                        const core::Tensor correspondences,
                                        const core::Tensor& T_ij);

void VisualizePointCloudEmbedding(t::geometry::PointCloud& tpcd_param,
                                  ControlGrid& ctr_grid,
                                  bool show_lines = true);

void VisualizePointCloudDeformation(const geometry::PointCloud& tpcd_param,
                                    ControlGrid& ctr_grid);

void VisualizeGridDeformation(ControlGrid& cgrid);

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer
