// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <opencv2/core.hpp>

namespace mcalib {

enum class BevRemapMode { Auto = 0, CPU, OpenCL, CUDA };

class BevRemapper {
public:
    BevRemapper(cv::Size src_size,
                const cv::Mat& mapx,
                const cv::Mat& mapy,
                BevRemapMode mode = BevRemapMode::Auto);
    ~BevRemapper();

    BevRemapper(const BevRemapper&) = delete;
    BevRemapper& operator=(const BevRemapper&) = delete;

    cv::Mat remap(const cv::Mat& src) const;
    bool valid() const { return valid_; }
    BevRemapMode activeMode() const { return active_mode_; }

    void updateMaps(const cv::Mat& mapx, const cv::Mat& mapy);

    static bool cudaAvailable();
    static bool openclAvailable();
    static BevRemapMode resolveMode(BevRemapMode requested);
    static const char* modeName(BevRemapMode mode);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool valid_ = false;
    BevRemapMode active_mode_ = BevRemapMode::CPU;
};

}  // namespace mcalib
