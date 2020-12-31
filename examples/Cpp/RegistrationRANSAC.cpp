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

#include <Eigen/Dense>
#include <iostream>
#include <memory>

#include "CloudViewer.h"

std::tuple<std::shared_ptr<ccPointCloud>,
           std::shared_ptr<cloudViewer::utility::Feature>>
PreprocessPointCloud(const char *file_name) {
    auto pcd = cloudViewer::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->voxelDownSample(0.05);
    pcd_down->estimateNormals(
            cloudViewer::geometry::KDTreeSearchParamHybrid(0.1, 30));
    auto pcd_fpfh = cloudViewer::utility::ComputeFPFHFeature(
            *pcd_down, cloudViewer::geometry::KDTreeSearchParamHybrid(0.25, 100));
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
    cloudViewer::visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;

	CVLib::utility::SetVerbosityLevel(CVLib::utility::VerbosityLevel::Debug);

    if (argc < 3 || argc > 7) {
        CVLib::utility::LogInfo(
                "Usage : RegistrationRANSAC path_to_first_point_cloud "
                "path_to_second_point_cloud [--method=feature_matching] "
                "[--mutual_filter] [--visualize]");
        return 1;
    }

    bool visualize = false;
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--visualize")) {
        visualize = true;
    }

	{ // ICP registration
		auto source_pcd = cloudViewer::io::CreatePointCloudFromFile(argv[1]);
		auto target_pcd = cloudViewer::io::CreatePointCloudFromFile(argv[2]);
		double threshold = 0.02;
		Eigen::Matrix4d transInit;
		transInit << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7,
			0.487, 0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;
		VisualizeRegistration(*source_pcd, *target_pcd, transInit);

		pipelines::registration::RegistrationResult regP2P = 
            pipelines::registration::RegistrationICP(*source_pcd, *target_pcd, threshold, transInit,
                pipelines::registration::TransformationEstimationPointToPoint());
		std::cout << regP2P.transformation_ << std::endl;
		VisualizeRegistration(*source_pcd, *target_pcd, regP2P.transformation_);

        pipelines::registration::RegistrationResult regP2L =
            pipelines::registration::RegistrationICP(*source_pcd, *target_pcd, threshold, transInit,
                pipelines::registration::TransformationEstimationPointToPlane());
		std::cout << regP2L.transformation_ << std::endl;
		VisualizeRegistration(*source_pcd, *target_pcd, regP2L.transformation_);
	}

    std::string method = "";
    const std::string kMethodFeature = "feature_matching";
    const std::string kMethodCorres = "correspondence";
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--method")) {
        method = CVLib::utility::GetProgramOptionAsString(argc, argv, "--method");
    } else {
        method = "feature_matching";
    }
    if (method != kMethodFeature && method != kMethodCorres) {
        CVLib::utility::LogInfo(
                "--method must be \'feature_matching\' or \'correspondence\'");
        return 1;
    }

    bool mutual_filter = false;
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--mutual_filter")) {
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

            CVLib::utility::LogDebug("{:d} points remain", mutual.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source, *target, mutual, 0.075,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    100000, 0.999));
        } else {
            CVLib::utility::LogDebug("{:d} points remain", corres_ji.size());
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
