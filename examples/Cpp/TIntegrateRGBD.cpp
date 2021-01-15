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

#include "CloudViewer.h"

using namespace cloudViewer;
using namespace cloudViewer::core;

void PrintHelp() {
    cloudViewer::PrintCloudViewerVersion();
    // clang-format off
    CVLib::utility::LogInfo("Usage:");
    CVLib::utility::LogInfo(">    TIntegrateRGBD [color_folder] [depth_folder] [trajectory] [options]");
    CVLib::utility::LogInfo("     Given RGBD images, reconstruct mesh or point cloud from color and depth images");
    CVLib::utility::LogInfo("     [options]");
    CVLib::utility::LogInfo("     --voxel_size [=0.0058 (m)]");
    CVLib::utility::LogInfo("     --intrinsic_path [camera_intrinsic]");
    CVLib::utility::LogInfo("     --depth_scale [=1000.0]");
    CVLib::utility::LogInfo("     --max_depth [=3.0]");
    CVLib::utility::LogInfo("     --sdf_trunc [=0.04]");
    CVLib::utility::LogInfo("     --camera_intrinsic [intrinsic_path]");
    CVLib::utility::LogInfo("     --device [CPU:0]");
    CVLib::utility::LogInfo("     --mesh");
    CVLib::utility::LogInfo("     --pointcloud");
    // clang-format on
    CVLib::utility::LogInfo("");
}

int main(int argc, char** argv) {
    if (argc == 1 || CVLib::utility::ProgramOptionExists(argc, argv, "--help") ||
        argc < 4) {
        PrintHelp();
        return 1;
    }

    // Color and depth
    std::string color_folder = std::string(argv[1]);
    std::string depth_folder = std::string(argv[2]);

    std::vector<std::string> color_filenames;
    CVLib::utility::filesystem::ListFilesInDirectory(color_folder, color_filenames);
    std::sort(color_filenames.begin(), color_filenames.end());

    std::vector<std::string> depth_filenames;
    CVLib::utility::filesystem::ListFilesInDirectory(depth_folder, depth_filenames);
    std::sort(depth_filenames.begin(), depth_filenames.end());

    if (color_filenames.size() != depth_filenames.size()) {
        CVLib::utility::LogError(
                "[TIntegrateRGBD] numbers of color and depth files mismatch. "
                "Please provide folders with same number of images.");
    }

    // Trajectory
    std::string trajectory_path = std::string(argv[3]);
    auto trajectory =
            io::CreatePinholeCameraTrajectoryFromFile(trajectory_path);

    // Intrinsics
    std::string intrinsic_path = CVLib::utility::GetProgramOptionAsString(
            argc, argv, "--intrinsic_path", "");
    camera::PinholeCameraIntrinsic intrinsic = camera::PinholeCameraIntrinsic(
            camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
    if (intrinsic_path.empty()) {
        CVLib::utility::LogWarning("Using default Primesense intrinsics");
    } else if (!io::ReadIJsonConvertible(intrinsic_path, intrinsic)) {
        CVLib::utility::LogError("Unable to convert json to intrinsics.");
    }

    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();
    Tensor intrinsic_t = Tensor(
            std::vector<float>({static_cast<float>(focal_length.first), 0,
                                static_cast<float>(principal_point.first), 0,
                                static_cast<float>(focal_length.second),
                                static_cast<float>(principal_point.second), 0,
                                0, 1}),
            {3, 3}, Dtype::Float32);

    int block_count =
            CVLib::utility::GetProgramOptionAsInt(argc, argv, "--block_count", 1000);

    float voxel_size = static_cast<float>(CVLib::utility::GetProgramOptionAsDouble(
            argc, argv, "--voxel_size", 3.f / 512.f));
    float depth_scale = static_cast<float>(CVLib::utility::GetProgramOptionAsDouble(
            argc, argv, "--depth_scale", 1000.f));
    float max_depth = static_cast<float>(
            CVLib::utility::GetProgramOptionAsDouble(argc, argv, "--max_depth", 3.f));
    float sdf_trunc = static_cast<float>(CVLib::utility::GetProgramOptionAsDouble(
            argc, argv, "--sdf_trunc", 0.04f));

    // Device
    std::string device_code = "CPU:0";
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--device")) {
        device_code = CVLib::utility::GetProgramOptionAsString(argc, argv, "--device");
    }
    core::Device device(device_code);
    CVLib::utility::LogInfo("Using device: {}", device.ToString());

    t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", core::Dtype::Float32},
                                           {"weight", core::Dtype::UInt16},
                                           {"color", core::Dtype::UInt16}},
                                           voxel_size, sdf_trunc, 16,
                                           block_count, device);

    for (size_t i = 0; i < trajectory->parameters_.size(); ++i) {
        // Load image
        std::shared_ptr<geometry::Image> depth_legacy =
                io::CreateImageFromFile(depth_filenames[i]);
        std::shared_ptr<geometry::Image> color_legacy =
                io::CreateImageFromFile(color_filenames[i]);

        t::geometry::Image depth =
                t::geometry::Image::FromLegacyImage(*depth_legacy, device);
        t::geometry::Image color =
                t::geometry::Image::FromLegacyImage(*color_legacy, device);

        Eigen::Matrix4f extrinsic =
                trajectory->parameters_[i].extrinsic_.cast<float>();
        Tensor extrinsic_t =
                core::eigen_converter::EigenMatrixToTensor(extrinsic).To(
                        device);

        CVLib::utility::Timer timer;
        timer.Start();
        voxel_grid.Integrate(depth, color, intrinsic_t, extrinsic_t,
                             depth_scale, max_depth);
        timer.Stop();
        CVLib::utility::LogInfo("{}: Integration takes {}", i, timer.GetDuration());
    }

    if (CVLib::utility::ProgramOptionExists(argc, argv, "--mesh")) {
        auto mesh = voxel_grid.ExtractSurfaceMesh();
        auto mesh_legacy = std::make_shared<ccMesh>(mesh.ToLegacyTriangleMesh());
        cloudViewer::io::WriteTriangleMesh("mesh_" + device.ToString() + ".ply", *mesh_legacy);
    }

    if (CVLib::utility::ProgramOptionExists(argc, argv, "--pointcloud")) {
        auto pcd = voxel_grid.ExtractSurfacePoints();
        auto pcd_legacy = std::make_shared<ccPointCloud>(pcd.ToLegacyPointCloud());
        cloudViewer::io::WritePointCloud("pcd_" + device.ToString() + ".ply", *pcd_legacy);
    }
}
