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

#pragma once

#include "camera/PinholeCameraIntrinsic.h"
#include "core/Dtype.h"
#include <IJsonConvertible.h>

namespace cloudViewer {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace t {
namespace io {

enum class SensorType { AZURE_KINECT = 0, REAL_SENSE = 1 };

/// RGBD video metadata.
class RGBDVideoMetadata : public CVLib::utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;

    bool ConvertFromJsonValue(const Json::Value &value) override;

    /// Text description
    using CVLib::utility::IJsonConvertible::ToString;

public:
    /// \brief Shared intrinsics between RGB & depth.
    ///
    /// We assume depth image is always warped to the color image system.
    camera::PinholeCameraIntrinsic intrinsics_;

    /// Capture device name.
    std::string device_name_ = "";

    /// Capture device serial number.
    std::string serial_number_ = "";

    /// Length of the video (usec). 0 for live capture.
    uint64_t stream_length_usec_ = 0;

    /// Width of the video frame.
    int width_;

    /// Height of the video frame.
    int height_;

    /// Frame rate.
    //
    /// We assume both color and depth streams have the same frame rate.
    double fps_;

    /// Pixel format for color data.
    std::string color_format_;

    /// Pixel format for depth data.
    std::string depth_format_;

    /// Pixel Dtype for color data.
    core::Dtype color_dt_;

    /// Pixel Dtype for depth data.
    core::Dtype depth_dt_;

    /// Number of color channels.
    uint8_t color_channels_;

    /// Number of depth units per meter (depth in m =
    /// depth_pixel_value/depth_scale).
    double depth_scale_;
};

}  // namespace io
}  // namespace t
}  // namespace cloudViewer
