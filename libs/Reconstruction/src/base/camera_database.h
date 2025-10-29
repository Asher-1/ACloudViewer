// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "util/camera_specs.h"

namespace colmap {

// Database that contains sensor widths for many cameras, which is useful
// to automatically extract the focal length if EXIF information is incomplete.
class CameraDatabase {
public:
    CameraDatabase();

    size_t NumEntries() const { return specs_.size(); }

    bool QuerySensorWidth(const std::string& make,
                          const std::string& model,
                          double* sensor_width);

private:
    static const camera_specs_t specs_;
};

}  // namespace colmap
