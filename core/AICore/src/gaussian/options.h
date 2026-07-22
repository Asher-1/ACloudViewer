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
#if defined(__APPLE__)
    // macOS: Metal is the native ggml GPU backend (OpenCL is not built).
    std::string device = "metal";  // metal | auto | cpu | cuda [:N]
#else
    // Auto order: CUDA -> OpenCL -> CPU (see
    // ggml_common::find_auto_gpu_backend).
    std::string device = "auto";  // auto | cpu | gpu | cuda | opencl [:N]
#endif
    int n_threads = 0;          // <= 0 => auto (CPU)
    std::string dump_taps_dir;  // empty => tap dumping disabled
};

}  // namespace gaussian
}  // namespace aicore
