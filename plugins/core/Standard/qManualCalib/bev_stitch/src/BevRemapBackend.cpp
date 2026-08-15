// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "BevRemapBackend.h"

#include <CVLog.h>

#include <opencv2/imgproc.hpp>

#include "BevRemapOcl.h"

#ifdef MCALIB_BEV_CUDA_ENABLED
#include "BevRemapCuda.cuh"
#endif

namespace mcalib {

struct BevRemapper::Impl {
    cv::Size src_size;
    cv::Mat mapx;
    cv::Mat mapy;
    BevRemapMode mode = BevRemapMode::CPU;
};

bool BevRemapper::cudaAvailable() {
#ifdef MCALIB_BEV_CUDA_ENABLED
    return bev_cuda::isAvailable();
#else
    return false;
#endif
}

bool BevRemapper::openclAvailable() {
#ifdef MCALIB_BEV_OPENCL_ENABLED
    return bev_ocl::probePlatform();
#else
    return false;
#endif
}

BevRemapMode BevRemapper::resolveMode(BevRemapMode requested) {
    // Priority: CUDA (when compiled) -> OpenCL (CPU builds / macOS / no GPU) ->
    // CPU.
    switch (requested) {
        case BevRemapMode::CUDA:
            if (cudaAvailable()) return BevRemapMode::CUDA;
            CVLog::Warning("[BevRemap] CUDA unavailable, try OpenCL");
            if (openclAvailable()) return BevRemapMode::OpenCL;
            CVLog::Warning("[BevRemap] OpenCL unavailable, fallback CPU");
            return BevRemapMode::CPU;

        case BevRemapMode::OpenCL:
            if (openclAvailable()) return BevRemapMode::OpenCL;
            CVLog::Warning("[BevRemap] OpenCL unavailable, fallback CPU");
            return BevRemapMode::CPU;

        case BevRemapMode::Auto:
            if (cudaAvailable()) return BevRemapMode::CUDA;
            if (openclAvailable()) return BevRemapMode::OpenCL;
            return BevRemapMode::CPU;

        case BevRemapMode::CPU:
        default:
            return BevRemapMode::CPU;
    }
}

const char* BevRemapper::modeName(BevRemapMode mode) {
    switch (mode) {
        case BevRemapMode::Auto:
            return "Auto";
        case BevRemapMode::CPU:
            return "CPU";
        case BevRemapMode::OpenCL:
            return "OpenCL";
        case BevRemapMode::CUDA:
            return "CUDA";
    }
    return "Unknown";
}

BevRemapper::BevRemapper(cv::Size src_size,
                         const cv::Mat& mapx,
                         const cv::Mat& mapy,
                         BevRemapMode mode) {
    impl_ = std::make_unique<Impl>();
    impl_->src_size = src_size;
    impl_->mapx = mapx.isContinuous() ? mapx : mapx.clone();
    impl_->mapy = mapy.isContinuous() ? mapy : mapy.clone();
    active_mode_ = resolveMode(mode);
    impl_->mode = active_mode_;
    valid_ = !impl_->mapx.empty() && !impl_->mapy.empty();
}

BevRemapper::~BevRemapper() = default;

void BevRemapper::updateMaps(const cv::Mat& mapx, const cv::Mat& mapy) {
    if (!impl_ || mapx.empty() || mapy.empty()) return;
    if (impl_->mapx.size() == mapx.size() &&
        impl_->mapx.type() == mapx.type()) {
        mapx.copyTo(impl_->mapx);
        mapy.copyTo(impl_->mapy);
    } else {
        impl_->mapx = mapx.isContinuous() ? mapx : mapx.clone();
        impl_->mapy = mapy.isContinuous() ? mapy : mapy.clone();
    }
    valid_ = true;
}

cv::Mat BevRemapper::remap(const cv::Mat& src) const {
    if (!valid_ || !impl_) return cv::Mat();

    cv::Mat dst;

    auto tryOpenClRemap = [&]() -> bool {
#ifdef MCALIB_BEV_OPENCL_ENABLED
        if (bev_ocl::remap(src, dst, impl_->mapx, impl_->mapy)) {
            return true;
        }
#endif
        return false;
    };

    switch (impl_->mode) {
#ifdef MCALIB_BEV_CUDA_ENABLED
        case BevRemapMode::CUDA:
            if (bev_cuda::remap(src, dst, impl_->mapx, impl_->mapy)) {
                return dst;
            }
            CVLog::Warning("[BevRemap] CUDA remap failed, try OpenCL");
            if (tryOpenClRemap()) return dst;
            CVLog::Warning("[BevRemap] OpenCL remap failed, fallback CPU");
            break;
#endif
#ifdef MCALIB_BEV_OPENCL_ENABLED
        case BevRemapMode::OpenCL:
            if (tryOpenClRemap()) return dst;
            CVLog::Warning("[BevRemap] OpenCL remap failed, fallback CPU");
            break;
#endif
        default:
            break;
    }

    cv::remap(src, dst, impl_->mapx, impl_->mapy, cv::INTER_LINEAR,
              cv::BORDER_CONSTANT);
    return dst;
}

}  // namespace mcalib
