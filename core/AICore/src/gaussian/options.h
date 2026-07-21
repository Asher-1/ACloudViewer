// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Configuration for free_splatter_load, behind an ABI-stable builder so new
// knobs are new setters, never struct-layout changes across the C boundary.
#pragma once

#include <string>

namespace aicore {
namespace gaussian {

struct options {
    std::string device = "vulkan";  // "" | cpu | gpu | cuda | vulkan [:N];
                                    // default GPU, fail-closed
    int n_threads = 0;              // <= 0 => auto (CPU)
    std::string dump_taps_dir;      // empty => tap dumping disabled
};

}  // namespace gaussian
}  // namespace aicore
