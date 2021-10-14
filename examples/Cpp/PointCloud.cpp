// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
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

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "CloudViewer.h"

void PrintPointCloud(const ccPointCloud &pointcloud) {
    using namespace cloudViewer;

    bool pointcloud_has_normal = pointcloud.hasNormals();
    utility::LogInfo("Pointcloud has %d points.", (int)pointcloud.size());

    Eigen::Vector3d min_bound = pointcloud.getMinBound();
    Eigen::Vector3d max_bound = pointcloud.getMaxBound();
    utility::LogInfo(
            "Bounding box is: ({:.4f}, {:.4f}, {:.4f}) - ({:.4f}, {:.4f}, "
            "{:.4f})",
            min_bound(0), min_bound(1), min_bound(2), max_bound(0),
            max_bound(1), max_bound(2));

    for (unsigned int i = 0; i < pointcloud.size(); i++) {
        if (pointcloud_has_normal) {
            const CCVector3 &point = *pointcloud.getPoint(i);
            const CCVector3 &normal = pointcloud.getPointNormal(i);
            utility::LogInfo("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                             point.x, point.y, point.z, normal.x, normal.y,
                             normal.z);
        } else {
            const CCVector3 &point = *pointcloud.getPoint(i);
            utility::LogInfo("{:.6f} {:.6f} {:.6f}", point.x, point.y, point.z);
        }
    }
    utility::LogInfo("End of the list.");
}

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > PointCloud [pointcloud_filename]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc != 2 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    auto pcd = cloudViewer::io::CreatePointCloudFromFile(argv[1]);
    {
        utility::ScopeTimer timer("FPFH estimation with Radius 0.25");
        // for (int i = 0; i < 20; i++) {
        utility::ComputeFPFHFeature(*pcd,
                                    geometry::KDTreeSearchParamRadius(0.25));
        //}
    }

    {
        utility::ScopeTimer timer("Normal estimation with KNN20");
        for (int i = 0; i < 20; i++) {
            pcd->estimateNormals(geometry::KDTreeSearchParamKNN(20));
        }
    }
    std::cout << pcd->getPointNormal(0) << std::endl;
    std::cout << pcd->getPointNormal(10) << std::endl;

    {
        utility::ScopeTimer timer("Normal estimation with Radius 0.01666");
        for (int i = 0; i < 20; i++) {
            pcd->estimateNormals(geometry::KDTreeSearchParamRadius(0.01666));
        }
    }
    std::cout << pcd->getPointNormal(0) << std::endl;
    std::cout << pcd->getPointNormal(10) << std::endl;

    {
        utility::ScopeTimer timer("Normal estimation with Hybrid 0.01666, 60");
        for (int i = 0; i < 20; i++) {
            pcd->estimateNormals(
                    geometry::KDTreeSearchParamHybrid(0.01666, 60));
        }
    }
    std::cout << pcd->getPointNormal(0) << std::endl;
    std::cout << pcd->getPointNormal(10) << std::endl;

    auto downpcd = pcd->voxelDownSample(0.05);

    // 1. test basic pointcloud functions.

    ccPointCloud pointcloud;
    PrintPointCloud(pointcloud);
    if (!pointcloud.reserveThePointsTable(4)) {
        return -1;
    }
    pointcloud.addPoint(CCVector3(0.0f, 0.0f, 0.0f));
    pointcloud.addPoint(CCVector3(1.0f, 0.0f, 0.0f));
    pointcloud.addPoint(CCVector3(0.0f, 1.0f, 0.0f));
    pointcloud.addPoint(CCVector3(0.0f, 0.0f, 1.0f));
    PrintPointCloud(pointcloud);

    // 2. test pointcloud IO.

    const std::string filename_xyz("test.xyz");
    const std::string filename_ply("test.ply");

    if (io::ReadPointCloud(argv[1], pointcloud)) {
        utility::LogInfo("Successfully read {}", argv[1]);

        /*
        geometry::PointCloud pointcloud_copy;
        pointcloud_copy.CloneFrom(pointcloud);

        if (io::WritePointCloud(filename_xyz, pointcloud)) {
            utility::LogInfo("Successfully wrote {}",
        filename_xyz.c_str()); } else { utility::LogError("Failed to write
        {}", filename_xyz);
        }

        if (io::WritePointCloud(filename_ply, pointcloud_copy)) {
            utility::LogInfo("Successfully wrote {}",
        filename_ply); } else { utility::LogError("Failed to write
        {}", filename_ply);
        }
         */
    } else {
        utility::LogWarning("Failed to read {}", argv[1]);
    }

    // 3. test pointcloud visualization

    cloudViewer::visualization::Visualizer visualizer;
    std::shared_ptr<ccPointCloud> pointcloud_ptr(new ccPointCloud);
    *pointcloud_ptr = pointcloud;
    pointcloud_ptr->normalizeNormals();
    auto bounding_box = pointcloud_ptr->getAxisAlignedBoundingBox();

    std::shared_ptr<ccPointCloud> pointcloud_transformed_ptr(new ccPointCloud);
    *pointcloud_transformed_ptr = *pointcloud_ptr;
    Eigen::Matrix4d trans_to_origin = Eigen::Matrix4d::Identity();
    trans_to_origin.block<3, 1>(0, 3) = bounding_box.getGeometryCenter() * -1.0;
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    transformation.block<3, 3>(0, 0) = static_cast<Eigen::Matrix3d>(
            Eigen::AngleAxisd(M_PI / 4.0, Eigen::Vector3d::UnitX()));
    pointcloud_transformed_ptr->transform(trans_to_origin.inverse() *
                                          transformation * trans_to_origin);

    visualizer.CreateVisualizerWindow("CloudViewer", 1600, 900);
    visualizer.AddGeometry(pointcloud_ptr);
    visualizer.AddGeometry(pointcloud_transformed_ptr);
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();

    // 4. test operations
    *pointcloud_transformed_ptr += *pointcloud_ptr;
    visualization::DrawGeometries({pointcloud_transformed_ptr},
                                  "Combined Pointcloud");

    // 5. test downsample
    auto downsampled = pointcloud_ptr->voxelDownSample(0.05);
    visualization::DrawGeometries({downsampled}, "Down Sampled Pointcloud");

    // 6. test normal estimation
    visualization::DrawGeometriesWithKeyCallbacks(
            {pointcloud_ptr},
            {{GLFW_KEY_SPACE,
              [&](cloudViewer::visualization::Visualizer *vis) {
                  // estimateNormals(*pointcloud_ptr,
                  //        cloudViewer::KDTreeSearchParamKNN(20));
                  pointcloud_ptr->estimateNormals(
                          geometry::KDTreeSearchParamRadius(0.05));
                  utility::LogInfo("Done.");
                  return true;
              }}},
            "Press Space to Estimate Normal", 1600, 900);

    // n. test end

    utility::LogInfo("End of the test.");
    return 0;
}
