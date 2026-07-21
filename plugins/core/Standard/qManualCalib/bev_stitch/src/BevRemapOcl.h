// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace mcalib {
namespace bev_ocl {

bool probePlatform();
bool isAvailable();
bool remap(const cv::Mat& src,
           cv::Mat& dst,
           const cv::Mat& mapx,
           const cv::Mat& mapy);

bool projectPoints(const float* points_xyz,
                   int num_points,
                   const float rotation[9],
                   const float translation[3],
                   float fx,
                   float fy,
                   float cx,
                   float cy,
                   std::vector<cv::Point2f>& image_points,
                   std::vector<float>& depths);

bool projectPointsKb(const float* points_xyz,
                     int num_points,
                     const float rotation[9],
                     const float translation[3],
                     float fx,
                     float fy,
                     float cx,
                     float cy,
                     const float kb[4],
                     std::vector<cv::Point2f>& image_points,
                     std::vector<float>& depths);

bool initAlphaFusion(cv::Size bev_size,
                     const std::vector<std::string>& camera_order,
                     const std::map<std::string, cv::Mat>& weights);
void updateAlphaFusionWeights(const std::map<std::string, cv::Mat>& weights);
void releaseAlphaFusion();
bool alphaFusion(const std::vector<std::string>& camera_order,
                 const std::map<std::string, cv::Mat>& warped_images,
                 cv::Mat& output);

}  // namespace bev_ocl
}  // namespace mcalib
