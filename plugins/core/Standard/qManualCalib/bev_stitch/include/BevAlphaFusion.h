// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "BevRemapBackend.h"

namespace mcalib {

class BevAlphaFusion {
public:
    BevAlphaFusion(cv::Size bev_size,
                   const std::vector<std::string>& camera_order,
                   const std::map<std::string, cv::Mat>& weights,
                   BevRemapMode mode);
    ~BevAlphaFusion();

    BevAlphaFusion(const BevAlphaFusion&) = delete;
    BevAlphaFusion& operator=(const BevAlphaFusion&) = delete;

    bool valid() const { return valid_; }
    BevRemapMode activeMode() const { return active_mode_; }

    bool fuse(const std::map<std::string, cv::Mat>& warped_images,
              cv::Mat& output);

    void updateWeights(const std::map<std::string, cv::Mat>& weights);

private:
    cv::Size bev_size_;
    std::vector<std::string> camera_order_;
    std::map<std::string, cv::Mat> weights_;
    bool valid_ = false;
    BevRemapMode active_mode_ = BevRemapMode::CPU;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace mcalib
