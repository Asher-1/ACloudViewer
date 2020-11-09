// ----------------------------------------------------------------------------
// -                        Open3D: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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
// @author Ignacio Vizzo     [ivizzo@uni-bonn.de]
//
// Copyright (c) 2020 Ignacio Vizzo, Cyrill Stachniss, University of Bonn.
// ----------------------------------------------------------------------------

#include <Eigen/Core>
#include <cstdlib>
#include <memory>
#include <string>

#include "CloudViewer.h"

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);
    if (argc < 3) {
        CVLib::utility::LogInfo("CloudViewer {}", CLOUDVIEWER_VERSION);
        CVLib::utility::LogInfo("Usage:");
        CVLib::utility::LogInfo("\t> {} [mesh|pointcloud] [filename] ...\n", argv[0]);
        return 1;
    }

    const std::string option(argv[1]);
    const std::string filename(argv[2]);
    auto cloud = std::make_shared<ccPointCloud>();
    auto mesh = std::make_shared<ccMesh>();
    if (option == "mesh") {
        if (!io::ReadTriangleMesh(filename, *mesh)) {
            CVLib::utility::LogWarning("Failed to read {}", filename);
            return 1;
        }
        cloud->setEigenPoints(mesh->getEigenVertices());
    } else if (option == "pointcloud") {
        if (!io::ReadPointCloud(filename, *cloud)) {
            CVLib::utility::LogWarning("Failed to read {}\n\n", filename);
            return 1;
        }
    } else {
        CVLib::utility::LogError("option {} not supported\n", option);
    }

    // Compute the ISS Keypoints
    auto iss_keypoints = std::make_shared<ccPointCloud>();
    {
        CVLib::utility::ScopeTimer timer("ISS Keypoints estimation");
        iss_keypoints = geometry::keypoint::ComputeISSKeypoints(*cloud);
        CVLib::utility::LogInfo("Detected {} keypoints",
                         iss_keypoints->size());
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
