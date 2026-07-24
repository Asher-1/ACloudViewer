// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "image.h"

#include <cmath>
#include <cstdint>

namespace aicore {
namespace gaussian {

bool ingest_images(const hparams& hp,
                   const float* images,
                   int32_t n_views,
                   int32_t height,
                   int32_t width,
                   std::vector<float>& out,
                   std::string& err) {
    out.clear();
    if (!images) {
        err = "images pointer is NULL";
        return false;
    }
    if (n_views <= 0) {
        err = "n_views must be > 0";
        return false;
    }
    if (height != hp.image_size || width != hp.image_size) {
        err = "input must be " + std::to_string(hp.image_size) + "x" +
              std::to_string(hp.image_size) + " (got " + std::to_string(width) +
              "x" + std::to_string(height) + ")";
        return false;
    }

    // 64-bit element count, guarded against overflow before any allocation.
    const uint64_t per_view =
            (uint64_t)hp.in_channels * (uint64_t)height * (uint64_t)width;
    const uint64_t total = per_view * (uint64_t)n_views;
    // Cap at a sane ceiling (e.g. 64 views) so a malformed n_views can't
    // request an absurd allocation; the real model uses ~7 views.
    if (n_views > 64 || total > (uint64_t)1 << 31) {
        err = "input too large (n_views or resolution out of range)";
        return false;
    }

    out.resize((size_t)total);
    for (uint64_t i = 0; i < total; i++) {
        const float v = images[i];
        if (!std::isfinite(v)) {
            err = "input contains a non-finite value at index " +
                  std::to_string(i);
            out.clear();
            return false;
        }
        out[(size_t)i] = v;
    }
    return true;
}

}  // namespace gaussian
}  // namespace aicore
