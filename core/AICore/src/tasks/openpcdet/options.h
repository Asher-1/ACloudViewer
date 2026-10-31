// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Configuration for openpcdet_load, behind an ABI-stable builder so new knobs
// are new setters, never struct-layout changes across the C boundary.
#pragma once

#include <string>

namespace openpcdet {

struct options {
    std::string device = "cpu";  // "" | cpu | gpu | cuda | vulkan [:N]
    int n_threads = 0;           // <= 0 => auto (CPU)
    std::string dump_taps_dir;   // empty => tap dumping disabled
    float score_thresh = -1.0f;  // < 0 => use model metadata
    float nms_thresh = -1.0f;    // < 0 => use model metadata
    int max_boxes = -1;          // < 0 => use model metadata
};

}  // namespace openpcdet
