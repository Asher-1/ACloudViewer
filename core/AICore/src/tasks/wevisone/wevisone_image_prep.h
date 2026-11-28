// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aicore/image_view.h"

namespace wwv {

// Decoded tightly-packed RGB image (HWC, 3 bytes per pixel).
struct RgbImage {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<uint8_t> pixels;  // row-major RGB8, stride = width * 3
};

// Loads JPEG/PNG from disk through libjpeg when available (byte-identical
// to PIL), falling back to the shared AICore Qt decoder otherwise (no stb).
bool load_path_rgb(const std::string& path, RgbImage* out, std::string* err);

// Converts a borrowed image view (RGB8 native; RGBA8/BGRA8/BGR8/GRAY8
// converted) to a tightly-packed RgbImage. Returns false on invalid views.
bool view_to_rgb(const aicore_image_view& img, RgbImage* out, std::string* err);

// PIL-BICUBIC (= torch AA bicubic) resize to an exact square canvas.
// Output is CHW f32 in [0, 1] (row-major, 3 * size * size floats).
std::vector<float> bicubic_square_chw(const RgbImage& src, int32_t size);

// Qwen smart-resize: rounds both dimensions to multiples of
// factor = patch_size * merge_size while keeping the pixel count within
// [min_pixels, max_pixels]. Returns the resized CHW f32 tensor, normalized
// with the released Qwen2-VL mean/std, plus the post-resize dimensions.
struct VitInput {
    int32_t width = 0;  // after smart resize
    int32_t height = 0;
    std::vector<float> pixels;  // CHW f32, normalized
};

VitInput smart_resize_normalized(const RgbImage& src,
                                 int32_t patch_size,
                                 int32_t merge_size,
                                 int64_t min_pixels,
                                 int64_t max_pixels);

// The Qwen factor arithmetic alone (used by the runtime to compute grids).
void smart_resize_dims(int32_t width,
                       int32_t height,
                       int32_t factor,
                       int64_t min_pixels,
                       int64_t max_pixels,
                       int32_t* out_width,
                       int32_t* out_height);

// Released Qwen2-VL image normalization constants.
extern const float kQwenImageMean[3];
extern const float kQwenImageStd[3];

}  // namespace wwv
