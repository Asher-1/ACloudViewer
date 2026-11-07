// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Official MapAnything preprocessing for the in-memory (borrowed views)
// integration path. NO third-party image codecs: the resize runs through
// the in-tree separable Catmull-Rom resampler below and every file decode
// belongs to the caller (aicore_map_anything_load_path_rgb uses Qt::Gui).
//
// PARITY NOTE (upstream contract, kept verbatim): the bit-exact parity feed
// is the raw f32 --bin path (aicore_map_anything_run_frames / the official
// PIL-bicubic loader); upstream itself documents its stb-images smoke path
// as not identical to PIL. The in-tree resampler implements the same
// Catmull-Rom (Keys a=-0.5) kernel family with stbir's half-pixel-centered
// sample mapping and EDGE_CLAMP, so the smoke path stays semantically on
// the official "balanced"/"max_size" contract.
#pragma once

#include "aicore/map_anything_capi.h"

#include <string>
#include <utility>
#include <vector>

namespace map_anything {

// Official load_and_preprocess_images equivalent (mode = "balanced" or
// "max_size") over decoded views:
//   balanced: extreme aspect ratios center-cropped into [0.5, 2.0],
//     patch-multiple target shape around (image_resolution/patch_size)^2
//     tokens; mixed sizes white-padded to a common size.
//   max_size: longest side = image_resolution, other side rounded to a
//     patch multiple; mixed sizes white-padded to a common size.
// Bicubic resize; returns (S, 3, H, W) floats in [0, 1] plus the common
// H, W. When \p valid_sizes is non-null it receives, per view, the
// RESIZED content size (w, h) BEFORE the centered white pad to the common
// grid — consumers use it to skip padded-border pixels.
bool load_images_official_views(const aicore_image_view* views, int n,
                                int image_resolution, int patch_size,
                                std::vector<float>& out, int& height,
                                int& width, const std::string& mode,
                                std::vector<std::pair<int, int>>* valid_sizes = nullptr);

}  // namespace map_anything
