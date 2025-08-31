// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>
#include <string>

#include "pipelines/color_map/ImageWarpingField.h"

namespace cloudViewer {

namespace io {

/// Factory function to create a ImageWarpingField from a file
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<pipelines::color_map::ImageWarpingField> CreateImageWarpingFieldFromFile(
        const std::string &filename);

/// The general entrance for reading a ImageWarpingField from a file
/// \return If the read function is successful.
bool ReadImageWarpingField(const std::string &filename,
                           pipelines::color_map::ImageWarpingField &warping_field);

/// The general entrance for writing a ImageWarpingField to a file
/// \return If the write function is successful.
bool WriteImageWarpingField(const std::string &filename,
                            const pipelines::color_map::ImageWarpingField &warping_field);

}  // namespace io
}  // namespace cloudViewer
