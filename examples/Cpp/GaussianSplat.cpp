// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// To run this example, download some sample Gaussian Splats. You can use these
// samples to get started:
// clang-format off
// curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.ply
// curl -O https://github.com/isl-org/open3d_downloads/releases/download/3dgs-1/mipnerf360_garden_crop_table.splat
// clang-format on

#include <cstdlib>

#include "CloudViewer.h"

using namespace cloudViewer;

void PrintUsage() {
    utility::LogInfo("Visualize Gaussian Splat from PLY or SPLAT file.");
    utility::LogInfo("Usage: GaussianSplat <filename.[ply|splat]>");
}

int main(int argc, char **argv) {
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintUsage();
        return 1;
    }
    std::shared_ptr<t::geometry::PointCloud> gsplat =
            std::make_shared<t::geometry::PointCloud>();
    if (!t::io::ReadPointCloud(argv[1], *gsplat)) {
        utility::LogWarning("Failed to read file {}", argv[1]);
        return 1;
    }
    visualization::Draw({visualization::DrawObject(argv[1], gsplat)},
                        "Gaussian Splat");
}