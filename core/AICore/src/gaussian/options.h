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
    // Platform auto order is owned by the private backend registry. On macOS
    // this preserves CPU fallback when Metal is unavailable.
    std::string device = "auto";
    int n_threads = 0;          // <= 0 => auto (CPU)
    std::string dump_taps_dir;  // empty => tap dumping disabled
};

}  // namespace gaussian
}  // namespace aicore
