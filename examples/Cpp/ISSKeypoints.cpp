// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <cstdlib>
#include <memory>
#include <string>

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > ISSKeypoints [mesh|pointcloud] [filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    const std::string option(argv[1]);
    const std::string filename(argv[2]);
    auto cloud = std::make_shared<ccPointCloud>();
    auto mesh = std::make_shared<ccMesh>();
    if (!mesh->createInternalCloud()) {
        utility::LogError("creating internal cloud failed!");
        return -1;
    }
    if (option == "mesh") {
        if (!io::ReadTriangleMesh(filename, *mesh)) {
            utility::LogWarning("Failed to read {}", filename);
            return 1;
        }
        cloud->setEigenPoints(mesh->getEigenVertices());
    } else if (option == "pointcloud") {
        if (!io::ReadPointCloud(filename, *cloud)) {
            utility::LogWarning("Failed to read {}\n\n", filename);
            return 1;
        }
    } else {
        utility::LogError("option {} not supported\n", option);
    }

    // Compute the ISS Keypoints
    auto iss_keypoints = std::make_shared<ccPointCloud>();
    {
        utility::ScopeTimer timer("ISS Keypoints estimation");
        iss_keypoints = geometry::keypoint::ComputeISSKeypoints(*cloud);
        utility::LogInfo("Detected {} keypoints", iss_keypoints->size());
    }

    // Visualize the results
    cloud->paintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
    iss_keypoints->paintUniformColor(Eigen::Vector3d(1.0, 0.75, 0.0));
    if (option == "mesh") {
        mesh->paintUniformColor(Eigen::Vector3d(0.5, 0.5, 0.5));
        mesh->computeVertexNormals();
        mesh->computeTriangleNormals();
        visualization::DrawGeometries({mesh, iss_keypoints}, "ISS", 1600, 900);
    } else {
        visualization::DrawGeometries({iss_keypoints}, "ISS", 1600, 900);
    }

    return 0;
}
