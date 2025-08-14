// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <IJsonConvertible.h>
#include "camera/PinholeCameraIntrinsic.h"

enum class SensorType { AZURE_KINECT = 0, REAL_SENSE = 1 };

namespace cloudViewer {

namespace camera {
class PinholeCameraIntrinsic;
}

namespace io {

/// class MKVMetadata
///
/// AzureKinect mkv metadata.
class MKVMetadata : public utility::IJsonConvertible {
public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    /// \brief Shared intrinsics betwee RGB & depth.
    ///
    /// We assume depth image is always warped to the color image system.
    camera::PinholeCameraIntrinsic intrinsics_;

    std::string serial_number_ = "";
    /// Length of the video (usec).
    uint64_t stream_length_usec_ = 0;
    /// Width of the video.
    int width_;
    /// Height of the video.
    int height_;
    std::string color_mode_;
    std::string depth_mode_;
};

}  // namespace io
}  // namespace cloudViewer
