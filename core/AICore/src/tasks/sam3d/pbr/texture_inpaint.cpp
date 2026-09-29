// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "texture_inpaint.hpp"

#include <cmath>
#include <cstddef>
#include <opencv2/photo.hpp>
#include <vector>

namespace sam3d {

bool inpaint_texture_telea(RgbaImage& texture,
                           const std::vector<uint8_t>& hole_mask,
                           float radius,
                           std::string& error) {
    const size_t pixel_count =
            static_cast<size_t>(texture.width) * texture.height;
    if (texture.width <= 0 || texture.height <= 0 ||
        texture.rgba.size() != pixel_count * 4 ||
        hole_mask.size() != pixel_count) {
        error = "Telea inpaint requires a non-empty RGBA texture and one mask "
                "byte per texel";
        return false;
    }
    if (!std::isfinite(radius) || radius <= 0.0f) {
        error = "Telea inpaint radius must be finite and positive";
        return false;
    }

    // OpenCV's Telea implementation processes channels independently. Retain
    // RGB order here: it is immaterial to the solver and avoids a needless
    // BGR conversion at the native PBR boundary.
    cv::Mat source(texture.height, texture.width, CV_8UC3);
    cv::Mat mask(texture.height, texture.width, CV_8UC1);
    for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
        source.at<cv::Vec3b>(static_cast<int>(pixel / texture.width),
                             static_cast<int>(pixel % texture.width)) =
                cv::Vec3b(texture.rgba[pixel * 4], texture.rgba[pixel * 4 + 1],
                          texture.rgba[pixel * 4 + 2]);
        mask.at<uint8_t>(static_cast<int>(pixel / texture.width),
                         static_cast<int>(pixel % texture.width)) =
                hole_mask[pixel] == 0 ? 0 : 255;
    }
    cv::Mat repaired;
    try {
        cv::inpaint(source, mask, repaired, radius, cv::INPAINT_TELEA);
    } catch (const cv::Exception& exception) {
        error = std::string("cv::inpaint(INPAINT_TELEA) failed: ") +
                exception.what();
        return false;
    }
    if (repaired.type() != CV_8UC3 || repaired.rows != texture.height ||
        repaired.cols != texture.width) {
        error = "cv::inpaint(INPAINT_TELEA) returned an invalid image";
        return false;
    }
    for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const cv::Vec3b value =
                repaired.at<cv::Vec3b>(static_cast<int>(pixel / texture.width),
                                       static_cast<int>(pixel % texture.width));
        texture.rgba[pixel * 4] = value[0];
        texture.rgba[pixel * 4 + 1] = value[1];
        texture.rgba[pixel * 4 + 2] = value[2];
        texture.rgba[pixel * 4 + 3] = 255;
    }
    return true;
}

}  // namespace sam3d
