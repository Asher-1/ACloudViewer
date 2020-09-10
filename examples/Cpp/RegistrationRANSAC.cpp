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

    if (argc != 3 && argc != 4) {
		CVLib::utility::LogInfo(
                "Usage : RegistrationRANSAC [path_to_first_point_cloud] "
                "[path_to_second_point_cloud] --visualize");
        return 1;
    }

    bool visualize = false;
    if (CVLib::utility::ProgramOptionExists(argc, argv, "--visualize"))
        visualize = true;

	{ // ICP registration
		auto source_pcd = cloudViewer::io::CreatePointCloudFromFile(argv[1]);
		auto target_pcd = cloudViewer::io::CreatePointCloudFromFile(argv[2]);
		double threshold = 0.02;
		Eigen::Matrix4d transInit;
		transInit << 0.862, 0.011, -0.507, 0.5, -0.139, 0.967, -0.215, 0.7,
			0.487, 0.255, 0.835, -1.4, 0.0, 0.0, 0.0, 1.0;
		VisualizeRegistration(*source_pcd, *target_pcd, transInit);

		registration::RegistrationResult regP2P = 
			registration::RegistrationICP(*source_pcd, *target_pcd, threshold, transInit, 
				registration::TransformationEstimationPointToPoint());
		std::cout << regP2P.transformation_ << std::endl;
		VisualizeRegistration(*source_pcd, *target_pcd, regP2P.transformation_);

		registration::RegistrationResult regP2L =
			registration::RegistrationICP(*source_pcd, *target_pcd, threshold, transInit,
				registration::TransformationEstimationPointToPlane());
		std::cout << regP2L.transformation_ << std::endl;
		VisualizeRegistration(*source_pcd, *target_pcd, regP2L.transformation_);
	}

    std::shared_ptr<ccPointCloud> source, target;
    std::shared_ptr<utility::Feature> source_fpfh, target_fpfh;
    std::tie(source, source_fpfh) = PreprocessPointCloud(argv[1]);
    std::tie(target, target_fpfh) = PreprocessPointCloud(argv[2]);

    std::vector<
            std::reference_wrapper<const registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            registration::CorrespondenceCheckerBasedOnEdgeLength(0.9);
    auto correspondence_checker_distance =
            registration::CorrespondenceCheckerBasedOnDistance(0.075);
    auto correspondence_checker_normal =
            registration::CorrespondenceCheckerBasedOnNormal(0.52359878);

    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);
    correspondence_checker.push_back(correspondence_checker_normal);
    auto registration_result =
            registration::RegistrationRANSACBasedOnFeatureMatching(
                    *source, *target, *source_fpfh, *target_fpfh, 0.075,
                    registration::TransformationEstimationPointToPoint(false),
                    4, correspondence_checker,
                    registration::RANSACConvergenceCriteria(4000000, 1000));

    if (visualize)
        VisualizeRegistration(*source, *target,
                              registration_result.transformation_);

    return 0;
}
