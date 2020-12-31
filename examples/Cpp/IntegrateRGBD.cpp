// ----------------------------------------------------------------------------
// -                        CloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include <iostream>
#include <memory>

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;
    // clang-format off
    CVLib::utility::LogInfo("Usage:");
    CVLib::utility::LogInfo("    > IntegrateRGBD [options]");
    CVLib::utility::LogInfo("      Integrate RGBD stream and extract geometry.");
    CVLib::utility::LogInfo("");
    CVLib::utility::LogInfo("Basic options:");
    CVLib::utility::LogInfo("    --help, -h                : Print help information.");
    CVLib::utility::LogInfo("    --match file              : The match file of an RGBD stream. Must have.");
    CVLib::utility::LogInfo("    --log file                : The log trajectory file. Must have.");
    CVLib::utility::LogInfo("    --save_pointcloud         : Save a point cloud created by marching cubes.");
    CVLib::utility::LogInfo("    --save_mesh               : Save a mesh created by marching cubes.");
    CVLib::utility::LogInfo("    --save_voxel              : Save a point cloud of the TSDF voxel.");
    CVLib::utility::LogInfo("    --every_k_frames k        : Save/reset every k frames. Default: 0 (none).");
    CVLib::utility::LogInfo("    --length l                : Length of the volume, in meters. Default: 4.0.");
    CVLib::utility::LogInfo("    --resolution r            : Resolution of the voxel grid. Default: 512.");
    CVLib::utility::LogInfo("    --sdf_trunc_percentage t  : TSDF truncation percentage, of the volume length. Default: 0.01.");
    CVLib::utility::LogInfo("    --verbose n               : Set verbose level (0-4). Default: 2.");
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    if (argc <= 1 || CVLib::utility::ProgramOptionExists(argc, argv, "--help") ||
        CVLib::utility::ProgramOptionExists(argc, argv, "-h")) {
        PrintHelp();
        return 1;
    }

    std::string match_filename =
            CVLib::utility::GetProgramOptionAsString(argc, argv, "--match");
    std::string log_filename =
            CVLib::utility::GetProgramOptionAsString(argc, argv, "--log");
    bool save_pointcloud =
            CVLib::utility::ProgramOptionExists(argc, argv, "--save_pointcloud");
    bool save_mesh = CVLib::utility::ProgramOptionExists(argc, argv, "--save_mesh");
    bool save_voxel = CVLib::utility::ProgramOptionExists(argc, argv, "--save_voxel");
    int every_k_frames =
            CVLib::utility::GetProgramOptionAsInt(argc, argv, "--every_k_frames", 0);
    double length =
            CVLib::utility::GetProgramOptionAsDouble(argc, argv, "--length", 4.0);
    int resolution =
            CVLib::utility::GetProgramOptionAsInt(argc, argv, "--resolution", 512);
    double sdf_trunc_percentage = CVLib::utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc_percentage", 0.01);
    int verbose = CVLib::utility::GetProgramOptionAsInt(argc, argv, "--verbose", 5);
    CVLib::utility::SetVerbosityLevel((CVLib::utility::VerbosityLevel)verbose);

    auto camera_trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(log_filename);
    std::string dir_name =
            CVLib::utility::filesystem::GetFileParentDirectory(match_filename).c_str();
    FILE *file = CVLib::utility::filesystem::FOpen(match_filename, "r");
    if (!file) {
        CVLib::utility::LogWarning("Unable to open file {}", match_filename);
        fclose(file);
        return 0;
    }
    char buffer[DEFAULT_IO_BUFFER_SIZE];
    int index = 0;
    int save_index = 0;
    pipelines::integration::ScalableTSDFVolume volume(
            length / (double)resolution, length * sdf_trunc_percentage,
        pipelines::integration::TSDFVolumeColorType::RGB8);
    CVLib::utility::FPSTimer timer("Process RGBD stream",
                            (int)camera_trajectory->parameters_.size());
    geometry::Image depth, color;
    while (fgets(buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::vector<std::string> st;
        CVLib::utility::SplitString(st, buffer, "\t\r\n ");
        if (st.size() >= 2) {
            CVLib::utility::LogInfo("Processing frame {:d} ...", index);
            io::ReadImage(dir_name + st[0], depth);
            io::ReadImage(dir_name + st[1], color);
            auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
                    color, depth, 1000.0, 4.0, false);
            if (index == 0 ||
                (every_k_frames > 0 && index % every_k_frames == 0)) {
                volume.Reset();
            }
            volume.Integrate(*rgbd,
                             camera_trajectory->parameters_[index].intrinsic_,
                             camera_trajectory->parameters_[index].extrinsic_);
            index++;
            if (index == (int)camera_trajectory->parameters_.size() ||
                (every_k_frames > 0 && index % every_k_frames == 0)) {
                CVLib::utility::LogInfo("Saving fragment {:d} ...", save_index);
                std::string save_index_str = std::to_string(save_index);
                if (save_pointcloud) {
                    CVLib::utility::LogInfo("Saving pointcloud {:d} ...", save_index);
                    auto pcd = volume.ExtractPointCloud();
                    io::WritePointCloud("pointcloud_" + save_index_str + ".ply",
                                        *pcd);
                }
                if (save_mesh) {
                    CVLib::utility::LogInfo("Saving mesh {:d} ...", save_index);
                    auto mesh = volume.ExtractTriangleMesh();
                    io::WriteTriangleMesh("mesh_" + save_index_str + ".ply",
                                          *mesh);
                }
                if (save_voxel) {
                    CVLib::utility::LogInfo("Saving voxel {:d} ...", save_index);
                    auto voxel = volume.ExtractVoxelPointCloud();
                    io::WritePointCloud("voxel_" + save_index_str + ".ply",
                                        *voxel);
                }
                save_index++;
            }
            timer.Signal();
        }
    }
    fclose(file);
    return 0;
}
