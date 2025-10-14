// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "mvs/fusion.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"

namespace cloudViewer {

int MeshDelaunay(
        const std::string& input_path,
        const std::string& output_path,
        const std::string& input_type = "dense",
        const colmap::mvs::DelaunayMeshingOptions& delaunay_meshing_options =
                colmap::mvs::DelaunayMeshingOptions());

int StereoPatchMatch(const std::string& workspace_path,
                     const std::string& config_path = "",
                     const std::string& workspace_format = "COLMAP",
                     const std::string& pmvs_option_name = "option-all",
                     const colmap::mvs::PatchMatchOptions& patch_match_options =
                             colmap::mvs::PatchMatchOptions());

int MeshPoisson(
        const std::string& input_path,
        const std::string& output_path,
        const colmap::mvs::PoissonMeshingOptions& poisson_meshing_options =
                colmap::mvs::PoissonMeshingOptions());

int StereoFuse(const std::string& workspace_path,
               const std::string& output_path,
               const std::string& bbox_path = "",
               const std::string& stereo_input_type = "geometric",
               const std::string& output_type = "PLY",
               const std::string& workspace_format = "COLMAP",
               const std::string& pmvs_option_name = "option-all",
               const colmap::mvs::StereoFusionOptions& stereo_fusion_options =
                       colmap::mvs::StereoFusionOptions());

}  // namespace cloudViewer
