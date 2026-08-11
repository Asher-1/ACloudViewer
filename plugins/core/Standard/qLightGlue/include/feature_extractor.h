// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QString>
#include <string>
#include <vector>

#include "aicore/lightglue_capi.h"

namespace lightglue_plugin {

/** Load image with EXIF orientation applied (matches ccImage / dialog preview).
 */
QImage load_oriented_qimage(const QString& path);

struct OwnedFeatures {
    std::vector<aicore_lightglue_keypoint> keypoints;
    std::vector<float> descriptors;
    aicore_lightglue_features view{};
};

/** RootSIFT via OpenCV — native C++ path (COLMAP-style classical extraction).
 */
bool extract_sift_opencv(const QString& image_path,
                         int max_keypoints,
                         int max_resize,
                         OwnedFeatures* out,
                         std::string* error);

/** ALIKED feature extraction via AICore GGML (128-D descriptors). */
bool extract_aliked_ggml(const QString& image_path,
                         const QString& extractor_gguf,
                         const QString& device,
                         int max_keypoints,
                         int max_resize,
                         int threads,
                         OwnedFeatures* out,
                         std::string* error);
bool load_fixture_pair(const QString& fixture_path,
                       OwnedFeatures* out0,
                       OwnedFeatures* out1,
                       std::string* error);

void release_owned(aicore_lightglue_features* features);

}  // namespace lightglue_plugin
