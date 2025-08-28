// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo(">    CameraPoseTrajectory [trajectory_file] [pcds_dir]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc == 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"}) ||
        argc != 2) {
        PrintHelp();
        return 1;
    }
    const int NUM_OF_COLOR_PALETTE = 5;
    Eigen::Vector3d color_palette[NUM_OF_COLOR_PALETTE] = {
            Eigen::Vector3d(255, 180, 0) / 255.0,
            Eigen::Vector3d(0, 166, 237) / 255.0,
            Eigen::Vector3d(246, 81, 29) / 255.0,
            Eigen::Vector3d(127, 184, 0) / 255.0,
            Eigen::Vector3d(13, 44, 84) / 255.0,
    };

    camera::PinholeCameraTrajectory trajectory;
    io::ReadPinholeCameraTrajectory(argv[1], trajectory);

    data::DemoICPPointClouds sample_icp_data;
    std::vector<std::shared_ptr<const ccHObject>> pcds;
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        if (utility::filesystem::FileExists(sample_icp_data.GetPaths()[i])) {
            auto pcd =
                    io::CreatePointCloudFromFile(sample_icp_data.GetPaths()[i]);
            pcd->transform(trajectory.parameters_[i].extrinsic_);
            if ((int)i < NUM_OF_COLOR_PALETTE) {
                pcd->setRGBColor(ecvColor::Rgb::FromEigen(color_palette[i]));
            } else {
                Eigen::Vector3d col = (Eigen::Vector3d::Random() +
                                       Eigen::Vector3d::Constant(1.0)) *
                                      0.5;
                pcd->setRGBColor(ecvColor::Rgb::FromEigen(col));
            }
            pcds.push_back(pcd);
        }
    }
    visualization::DrawGeometriesWithCustomAnimation(pcds);

    return 0;
}
