// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "util/hash_containers.h"

namespace colmap {

// { make1 : ({ model1 : sensor-width in mm }, ...), ... }
using camera_make_specs_t = std::vector<std::pair<std::string, float>>;
using camera_specs_t = NodeHashMap<std::string, camera_make_specs_t>;

camera_specs_t InitializeCameraSpecs();

}  // namespace colmap
