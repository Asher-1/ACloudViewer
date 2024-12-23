// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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
        argc != 3) {
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
    std::vector<std::shared_ptr<const ccHObject>> pcds;
    for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
        std::string buffer =
                fmt::format("{}cloud_bin_{:d}.pcd", argv[2], (int)i);
        if (utility::filesystem::FileExists(buffer.c_str())) {
            auto pcd = io::CreatePointCloudFromFile(buffer.c_str());
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
