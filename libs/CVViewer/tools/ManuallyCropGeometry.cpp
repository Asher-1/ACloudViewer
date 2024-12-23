// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
