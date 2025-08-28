// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;
    PrintCloudViewerVersion();

    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ManuallyCropGeometry [--pointcloud/mesh] geometry_file [options]");
    utility::LogInfo("      Manually crop geometry in speficied file.");
    utility::LogInfo("");
    utility::LogInfo("Options:");
    utility::LogInfo("    --pointcloud,             : Read geometry as point cloud.");
    utility::LogInfo("    --mesh,                   : Read geometry as mesh.");
    utility::LogInfo("    --help, -h                : Print help information.");
    utility::LogInfo("    --verbose n               : Set verbose level (0-4).");
    utility::LogInfo("    --voxel_size d            : Set downsample voxel size.");
    utility::LogInfo("    --without_dialog          : Disable dialogs. Default files will be used.");
    // clang-format on
}

int main(int argc, char **argv) {
    using namespace cloudViewer;

    if (argc < 2 || utility::ProgramOptionExists(argc, argv, "--help") ||
        utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 0;
    }

    int verbose = utility::GetProgramOptionAsInt(argc, argv, "--verbose", 2);
    utility::SetVerbosityLevel(static_cast<utility::VerbosityLevel>(verbose));
    double voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", -1.0);
    bool with_dialog =
            !utility::ProgramOptionExists(argc, argv, "--without_dialog");

    cloudViewer::visualization::VisualizerWithEditing vis(
            voxel_size, with_dialog,
            utility::filesystem::GetFileParentDirectory(argv[1]));
    vis.CreateVisualizerWindow("Crop Point Cloud", 1920, 1080, 100, 100);
    if (utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd_ptr = io::CreatePointCloudFromFile(argv[2]);
        if (pcd_ptr == nullptr || !pcd_ptr->isEmpty()) {
            utility::LogWarning("Failed to read the point cloud.");
            return 1;
        }
        vis.AddGeometry(pcd_ptr);
        if (pcd_ptr->size() > 5000000) {
            vis.GetRenderOption().point_size_ = 1.0;
        }
    } else if (utility::ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh_ptr = io::CreateMeshFromFile(argv[2]);
        if (mesh_ptr == nullptr || mesh_ptr->isEmpty()) {
            utility::LogWarning("Failed to read the mesh.");
            return 1;
        }
        vis.AddGeometry(mesh_ptr);
    }
    vis.Run();
    vis.DestroyVisualizerWindow();
    return 0;
}
