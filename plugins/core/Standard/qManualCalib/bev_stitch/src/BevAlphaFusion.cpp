// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BevAlphaFusion.h"

#include <CVLog.h>

#include <opencv2/imgproc.hpp>

#include "BevRemapOcl.h"

#ifdef MCALIB_BEV_CUDA_ENABLED
#include "BevRemapCuda.cuh"
#endif

namespace mcalib {

struct BevAlphaFusion::Impl {
    BevRemapMode mode = BevRemapMode::CPU;
};

static bool cpuAlphaFusion(const std::vector<std::string>& camera_order,
                           const std::map<std::string, cv::Mat>& weights,
                           const std::map<std::string, cv::Mat>& warped_images,
                           cv::Mat& output) {
    if (output.empty()) {
        output = cv::Mat::zeros(warped_images.begin()->second.size(), CV_8UC3);
    } else {
        output.setTo(0);
    }

    cv::Mat acc;
    output.convertTo(acc, CV_32FC3);

    for (const auto& name : camera_order) {
        auto img_it = warped_images.find(name);
        auto w_it = weights.find(name);
        if (img_it == warped_images.end() || img_it->second.empty() ||
            w_it == weights.end() || w_it->second.empty()) {
            continue;
        }

        const cv::Mat& img = img_it->second;
        const cv::Mat& weight = w_it->second;
        for (int y = 0; y < acc.rows; ++y) {
            const float* w_row = weight.ptr<float>(y);
            const cv::Vec3b* src_row = img.ptr<cv::Vec3b>(y);
            cv::Vec3f* dst_row = acc.ptr<cv::Vec3f>(y);
            for (int x = 0; x < acc.cols; ++x) {
                const float w = w_row[x];
                if (w <= 0.f) continue;
                dst_row[x][0] += w * static_cast<float>(src_row[x][0]);
                dst_row[x][1] += w * static_cast<float>(src_row[x][1]);
                dst_row[x][2] += w * static_cast<float>(src_row[x][2]);
            }
        }
    }

    acc.convertTo(output, CV_8UC3);
    return true;
}

BevAlphaFusion::BevAlphaFusion(cv::Size bev_size,
                               const std::vector<std::string>& camera_order,
                               const std::map<std::string, cv::Mat>& weights,
                               BevRemapMode mode)
    : bev_size_(bev_size), camera_order_(camera_order), weights_(weights) {
    impl_ = std::make_unique<Impl>();
    active_mode_ = BevRemapper::resolveMode(mode);
    impl_->mode = active_mode_;

    if (bev_size_.width <= 0 || bev_size_.height <= 0 ||
        camera_order_.empty()) {
        return;
    }

    for (const auto& name : camera_order_) {
        auto it = weights_.find(name);
        if (it == weights_.end() || it->second.empty()) {
            CVLog::Warning("[BevFusion] missing weight for camera %s",
                           name.c_str());
            return;
        }
    }

#ifdef MCALIB_BEV_CUDA_ENABLED
    if (active_mode_ == BevRemapMode::CUDA) {
        if (bev_cuda::initAlphaFusion(bev_size_, camera_order_, weights_)) {
            valid_ = true;
            CVLog::Print("[BevFusion] CUDA alpha fusion ready (%zu cameras)",
                         camera_order_.size());
            return;
        }
        CVLog::Warning("[BevFusion] CUDA alpha fusion init failed, try OpenCL");
        active_mode_ = BevRemapMode::OpenCL;
        impl_->mode = active_mode_;
    }
#endif

#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (active_mode_ == BevRemapMode::OpenCL) {
        if (bev_ocl::initAlphaFusion(bev_size_, camera_order_, weights_)) {
            valid_ = true;
            CVLog::Print("[BevFusion] OpenCL alpha fusion ready (%zu cameras)",
                         camera_order_.size());
            return;
        }
        CVLog::Warning(
                "[BevFusion] OpenCL alpha fusion init failed, fallback CPU");
        active_mode_ = BevRemapMode::CPU;
        impl_->mode = active_mode_;
    }
#endif

    valid_ = true;
    CVLog::Print("[BevFusion] CPU alpha fusion ready (%zu cameras)",
                 camera_order_.size());
}

BevAlphaFusion::~BevAlphaFusion() {
#ifdef MCALIB_BEV_CUDA_ENABLED
    if (active_mode_ == BevRemapMode::CUDA) {
        bev_cuda::releaseAlphaFusion();
    }
#endif
#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (active_mode_ == BevRemapMode::OpenCL) {
        bev_ocl::releaseAlphaFusion();
    }
#endif
}

void BevAlphaFusion::updateWeights(
        const std::map<std::string, cv::Mat>& weights) {
    weights_ = weights;
    if (!valid_) return;

#ifdef MCALIB_BEV_CUDA_ENABLED
    if (active_mode_ == BevRemapMode::CUDA) {
        bev_cuda::updateAlphaFusionWeights(weights_);
        return;
    }
#endif
#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (active_mode_ == BevRemapMode::OpenCL) {
        bev_ocl::updateAlphaFusionWeights(weights_);
        return;
    }
#endif
}

bool BevAlphaFusion::fuse(const std::map<std::string, cv::Mat>& warped_images,
                          cv::Mat& output) {
    if (!valid_) return false;

#ifdef MCALIB_BEV_CUDA_ENABLED
    if (active_mode_ == BevRemapMode::CUDA) {
        if (bev_cuda::alphaFusion(camera_order_, warped_images, output)) {
            return true;
        }
        CVLog::Warning("[BevFusion] CUDA fuse failed, try OpenCL");
#ifdef MCALIB_BEV_OPENCL_ENABLED
        if (bev_ocl::initAlphaFusion(bev_size_, camera_order_, weights_) &&
            bev_ocl::alphaFusion(camera_order_, warped_images, output)) {
            return true;
        }
#endif
        CVLog::Warning("[BevFusion] OpenCL fuse failed, fallback CPU");
    }
#endif

#ifdef MCALIB_BEV_OPENCL_ENABLED
    if (active_mode_ == BevRemapMode::OpenCL) {
        if (bev_ocl::alphaFusion(camera_order_, warped_images, output)) {
            return true;
        }
        CVLog::Warning("[BevFusion] OpenCL fuse failed, fallback CPU");
    }
#endif

    return cpuAlphaFusion(camera_order_, weights_, warped_images, output);
}

}  // namespace mcalib
