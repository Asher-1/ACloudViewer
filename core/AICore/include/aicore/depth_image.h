// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QString>
#include <vector>

#include "aicore/export.h"

namespace aicore {
namespace depth {

//! Monocular depth (+ optional pose) result for a single image.
struct AICORE_CXX_API ImageDepthResult {
    std::vector<float> depth;
    std::vector<float> confidence;
    int width = 0;
    int height = 0;
    bool has_pose = false;
    float extrinsics[12] = {};
    float intrinsics[9] = {};
};

//! Qt image helpers for the depth module. Inference lives in libAICore.so.
class AICORE_CXX_API ImageDepth {
public:
    static bool isAvailable();

    static bool estimateDepth(const QImage& image,
                              const QString& model_path,
                              int n_threads,
                              ImageDepthResult& out,
                              const QString& metric_model_path = QString());

    static bool estimateDepthAndPose(
            const QImage& image,
            const QString& model_path,
            int n_threads,
            ImageDepthResult& out,
            const QString& metric_model_path = QString());
};

}  // namespace depth
}  // namespace aicore
