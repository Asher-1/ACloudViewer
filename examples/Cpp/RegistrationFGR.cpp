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

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<utility::Feature>>
PreprocessPointCloud(const char *file_name, const float voxel_size) {
    auto pcd = cloudViewer::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->voxelDownSample(voxel_size);
    pcd_down->estimateNormals(
            cloudViewer::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = utility::ComputeFPFHFeature(
            *pcd_down,
            cloudViewer::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd, pcd_down, pcd_fpfh);
}

void VisualizeRegistration(const cloudViewer::geometry::PointCloud &source,
                           const cloudViewer::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
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
    utility::LogInfo("    > RegistrationFGR source_pcd target_pcd"
                     "[--voxel_size=0.05] [--distance_multiplier=1.5]"
                     "[--max_iterations=64] [--max_tuples=1000]"
                     );
    // clang-format on
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc < 3 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    float voxel_size =
            utility::GetProgramOptionAsDouble(argc, argv, "--voxel_size", 0.05);
    float distance_multiplier = utility::GetProgramOptionAsDouble(
            argc, argv, "--distance_multiplier", 1.5);
    float distance_threshold = voxel_size * distance_multiplier;
    int max_iterations =
            utility::GetProgramOptionAsInt(argc, argv, "--max_iterations", 64);
    int max_tuples =
            utility::GetProgramOptionAsInt(argc, argv, "--max_tuples", 1000);

    // Prepare input
    std::shared_ptr<geometry::PointCloud> source, source_down, target,
            target_down;
    std::shared_ptr<utility::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_down, source_fpfh) =
            PreprocessPointCloud(argv[1], voxel_size);
    std::tie(target, target_down, target_fpfh) =
            PreprocessPointCloud(argv[2], voxel_size);

    pipelines::registration::RegistrationResult registration_result =
            pipelines::registration::
                    FastGlobalRegistrationBasedOnFeatureMatching(
                            *source_down, *target_down, *source_fpfh,
                            *target_fpfh,
                            pipelines::registration::
                                    FastGlobalRegistrationOption(
                                            /* decrease_mu =  */ 1.4, true,
                                            true, distance_threshold,
                                            max_iterations,
                                            /* tuple_scale =  */ 0.95,
                                            max_tuples));

    VisualizeRegistration(*source, *target,
                          registration_result.transformation_);

    return 0;
}
