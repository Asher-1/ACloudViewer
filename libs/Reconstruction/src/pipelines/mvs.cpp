// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "exe/mvs.h"

#include "pipelines/mvs.h"
#include "pipelines/option_utils.h"

namespace cloudViewer {

int MeshDelaunay(
        const std::string& input_path,
        const std::string& output_path,
        const std::string& input_type /*= "dense"*/,
        const colmap::mvs::DelaunayMeshingOptions& delaunay_meshing_options) {
    OptionsParser parser;
    // Path to either the dense workspace folder or the sparse reconstruction
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    // supported {dense, sparse}
    parser.registerOption("input_type", &input_type);
    parser.addDelaunayMeshingOptions(delaunay_meshing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunDelaunayMesher(parser.getArgc(), parser.getArgv());
}

int StereoPatchMatch(
        const std::string& workspace_path,
        const std::string& config_path /*= ""*/,
        const std::string& workspace_format /*= "COLMAP"*/,
        const std::string& pmvs_option_name /*= "option-all"*/,
        const colmap::mvs::PatchMatchOptions& patch_match_options) {
    OptionsParser parser;
    // Path to the folder containing the undistorted images
    parser.registerOption("workspace_path", &workspace_path);
    parser.registerOption("config_path", &config_path);
    // supported {COLMAP, PMVS}
    parser.registerOption("workspace_format", &workspace_format);
    parser.registerOption("pmvs_option_name", &pmvs_option_name);
    parser.addPatchMatchStereoOptions(patch_match_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunPatchMatchStereo(parser.getArgc(), parser.getArgv());
}

int MeshPoisson(
        const std::string& input_path,
        const std::string& output_path,
        const colmap::mvs::PoissonMeshingOptions& poisson_meshing_options) {
    OptionsParser parser;
    parser.registerOption("input_path", &input_path);
    parser.registerOption("output_path", &output_path);
    parser.addPoissonMeshingOptions(poisson_meshing_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunPoissonMesher(parser.getArgc(), parser.getArgv());
}

int StereoFuse(const std::string& workspace_path,
               const std::string& output_path,
               const std::string& bbox_path /*= ""*/,
               const std::string& stereo_input_type /*= "geometric"*/,
               const std::string& output_type /*= "PLY"*/,
               const std::string& workspace_format /*= "COLMAP"*/,
               const std::string& pmvs_option_name /*= "option-all"*/,
               const colmap::mvs::StereoFusionOptions& stereo_fusion_options) {
    OptionsParser parser;
    // Path to the folder containing the undistorted images
    parser.registerOption("workspace_path", &workspace_path);
    parser.registerOption("bbox_path", &bbox_path);
    parser.registerOption("output_path", &output_path);
    // supported {photometric, geometric}
    parser.registerOption("input_type", &stereo_input_type);
    // supported {BIN, TXT, PLY}
    parser.registerOption("output_type", &output_type);
    // supported {COLMAP, PMVS}
    parser.registerOption("workspace_format", &workspace_format);
    parser.registerOption("pmvs_option_name", &pmvs_option_name);
    parser.addStereoFusionOptions(stereo_fusion_options);
    if (!parser.parseOptions()) return EXIT_FAILURE;

    return colmap::RunStereoFuser(parser.getArgc(), parser.getArgv());
}

}  // namespace cloudViewer
