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

std::tuple<std::shared_ptr<ccPointCloud>, std::shared_ptr<utility::Feature>>
PreprocessPointCloud(const char *file_name) {
    auto pcd = cloudViewer::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(0.05);
    pcd_down->EstimateNormals(
            cloudViewer::geometry::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = utility::ComputeFPFHFeature(
            *pcd_down,
            cloudViewer::geometry::KDTreeSearchParamHybrid(0.25, 100));
    return std::make_tuple(pcd_down, pcd_fpfh);
}

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
    utility::LogInfo("    > RegistrationRANSAC source_pcd target_pcd [--method=feature_matching] [--mutual_filter] [--visualize]");
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

    std::string method = "";
    const std::string kMethodFeature = "feature_matching";
    const std::string kMethodCorres = "correspondence";
    if (utility::ProgramOptionExists(argc, argv, "--method")) {
        method = utility::GetProgramOptionAsString(argc, argv, "--method");
    } else {
        method = "feature_matching";
    }
    if (method != kMethodFeature && method != kMethodCorres) {
        utility::LogInfo(
                "--method must be \'feature_matching\' or "
                "\'correspondence\'");
        return 1;
    }

    bool mutual_filter = false;
    if (utility::ProgramOptionExists(argc, argv, "--mutual_filter")) {
        mutual_filter = true;
    }

    // Prepare input
    std::shared_ptr<ccPointCloud> source, target;
    std::shared_ptr<utility::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_fpfh) = PreprocessPointCloud(argv[1]);
    std::tie(target, target_fpfh) = PreprocessPointCloud(argv[2]);

    pipelines::registration::RegistrationResult registration_result;

    // Prepare checkers
    std::vector<std::reference_wrapper<
            const pipelines::registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
                    0.9);
    auto correspondence_checker_distance =
            pipelines::registration::CorrespondenceCheckerBasedOnDistance(
                    0.075);
    auto correspondence_checker_normal =
            pipelines::registration::CorrespondenceCheckerBasedOnNormal(
                    0.52359878);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    correspondence_checker.push_back(correspondence_checker_normal);

    if (method == kMethodFeature) {
        registration_result = pipelines::registration::
                RegistrationRANSACBasedOnFeatureMatching(
                        *source, *target, *source_fpfh, *target_fpfh,
                        mutual_filter, 0.075,
                        pipelines::registration::
                                TransformationEstimationPointToPoint(false),
                        3, correspondence_checker,
                        pipelines::registration::RANSACConvergenceCriteria(
                                100000, 0.999));
    } else if (method == kMethodCorres) {
        // Use mutual filter by default
        int nPti = int(source->size());
        int nPtj = int(target->size());

        geometry::KDTreeFlann feature_tree_i(*source_fpfh);
        geometry::KDTreeFlann feature_tree_j(*target_fpfh);

        pipelines::registration::CorrespondenceSet corres_ji;
        std::vector<int> i_to_j(nPti, -1);

        // Buffer all correspondences
        for (int j = 0; j < nPtj; j++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)),
                                     1, corres_tmp, dist_tmp);
            int i = corres_tmp[0];
            corres_ji.push_back(Eigen::Vector2i(i, j));
        }

        if (mutual_filter) {
            pipelines::registration::CorrespondenceSet mutual;
            for (auto &corres : corres_ji) {
                int j = corres(1);
                int j2i = corres(0);

                std::vector<int> corres_tmp(1);
                std::vector<double> dist_tmp(1);
                feature_tree_j.SearchKNN(
                        Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1,
                        corres_tmp, dist_tmp);
                int i2j = corres_tmp[0];
                if (i2j == j) {
                    mutual.push_back(corres);
                }
            }

            utility::LogDebug("{:d} points remain", mutual.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source, *target, mutual, 0.075,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    100000, 0.999));
        } else {
            utility::LogDebug("{:d} points remain", corres_ji.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source, *target, corres_ji, 0.075,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    100000, 0.999));
        }
    }

    if (visualize) {
        VisualizeRegistration(*source, *target,
                              registration_result.transformation_);
    }

    return 0;
}
