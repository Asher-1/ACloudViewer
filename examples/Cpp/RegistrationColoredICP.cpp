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

using namespace cloudViewer;

void VisualizeRegistration(const ccPointCloud &source,
                           const ccPointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<ccPointCloud> source_transformed_ptr(new ccPointCloud);
    std::shared_ptr<ccPointCloud> target_ptr(new ccPointCloud);
    *source_transformed_ptr = source;
    *target_ptr = target;
    source_transformed_ptr->transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RegistrationColoredICP source_pcd target_pcd [--visualize]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    bool visualize = false;
    if (utility::ProgramOptionExists(argc, argv, "--visualize")) {
        visualize = true;
    }

    // Prepare input
    std::shared_ptr<ccPointCloud> source =
            cloudViewer::io::CreatePointCloudFromFile(argv[1]);
    std::shared_ptr<ccPointCloud> target =
            cloudViewer::io::CreatePointCloudFromFile(argv[2]);
    if (source == nullptr || target == nullptr) {
        utility::LogWarning("Unable to load source or target file.");
        return -1;
    }

    std::vector<double> voxel_sizes = {0.05, 0.05 / 2, 0.05 / 4};
    std::vector<int> iterations = {50, 30, 14};
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; ++i) {
        double voxel_size = voxel_sizes[i];

        auto source_down = source->voxelDownSample(voxel_size);
        source_down->estimateNormals(
                cloudViewer::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0,
                                                               30));

        auto target_down = target->voxelDownSample(voxel_size);
        target_down->estimateNormals(
                cloudViewer::geometry::KDTreeSearchParamHybrid(voxel_size * 2.0,
                                                               30));

        auto result = pipelines::registration::RegistrationColoredICP(
                *source_down, *target_down, 0.07, trans,
                pipelines::registration::
                        TransformationEstimationForColoredICP(),
                pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6,
                                                                iterations[i]));
        trans = result.transformation_;

        if (visualize) {
            VisualizeRegistration(*source, *target, trans);
        }
    }

    std::stringstream ss;
    ss << trans;
    utility::LogInfo("Final transformation = \n{}", ss.str());

    return 0;
}
