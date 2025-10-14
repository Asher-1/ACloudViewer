// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UTIL_CAMERA_SPECS_H_
#define COLMAP_SRC_UTIL_CAMERA_SPECS_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace colmap {

// { make1 : ({ model1 : sensor-width in mm }, ...), ... }
typedef std::vector<std::pair<std::string, float>> camera_make_specs_t;
typedef std::unordered_map<std::string, camera_make_specs_t> camera_specs_t;

camera_specs_t InitializeCameraSpecs();

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_CAMERA_SPECS_H_
