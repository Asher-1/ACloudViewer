// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Task-agnostic image toolkit for AICore tasks (Qt-free header).
//
// Every task crosses the C ABI with a borrowed, stride-aware
// aicore_image_view; this layer wraps that view into a uniform RGB-like
// working image and provides the OpenCV-aligned bilinear resize shared by
// the task preprocess paths. The Qt-dependent file decode lives in the
// .cpp (Qt is already a libAICore dependency via common/capi_utils).
//
// Previously this toolkit lived inside tasks/facedetect/ (image_io.hpp
// plus a duplicated cv_resize_linear_u8 in scrfd_graph.cpp and
// yunet_graph.cpp); the ocr task borrowed the symbol through an extern
// declaration, which left an unresolved reference in libAICore (the two
// definitions were internal). Owned here now: one image view type, one
// decode entry, one resize kernel.

#include <cstdint>
#include <string>
#include <vector>

#include "aicore/image_view.h"

namespace aicore {
namespace img {

// An RGB-like input view or an owned tightly-packed RGB working image.
struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
    const uint8_t* borrowed_data = nullptr;
    size_t row_stride_bytes = 0;
    int channels = 3;
    bool bgr = false;

    const uint8_t* data() const {
        return borrowed_data != nullptr
                       ? borrowed_data
                       : (rgb.empty() ? nullptr : rgb.data());
    }
    size_t stride() const {
        return row_stride_bytes != 0
                       ? row_stride_bytes
                       : static_cast<size_t>(width) * 3;
    }
    uint8_t channel(int x, int y, int c) const {
        const uint8_t* pixel =
                data() + static_cast<size_t>(y) * stride() +
                static_cast<size_t>(x) * channels;
        if (channels == 1) return pixel[0];
        return pixel[bgr ? 2 - c : c];
    }
    // Hoistable pieces of channel() for hot loops: `row(y)[x * channels +
    // channel_index(c)]` is the same byte channel(x, y, c) returns, but the
    // data()/stride()/channels/bgr branches are evaluated once per row (or
    // once per call) instead of per sample. Writes through a raw `uint8_t*` /
    // `float*` destination may alias this struct, so the compiler cannot do
    // that hoisting on its own; the resize/pack/crop kernels ran 3-5x slower
    // than the arithmetic warranted before the callers hoisted explicitly.
    const uint8_t* row(int y) const {
        return data() + static_cast<size_t>(y) * stride();
    }
    int channel_index(int c) const {
        return channels == 1 ? 0 : (bgr ? 2 - c : c);
    }
    bool empty() const {
        return width <= 0 || height <= 0 || data() == nullptr;
    }
};

// Decode an image file through Qt's image plugins into tightly packed RGB.
// Returns true on success and fills `out`; false on failure (sets nothing).
bool load_image_rgb(const std::string& path, Image& out);

// Wrap caller-owned raw RGB bytes (copied) into an Image. `rgb` must hold at
// least width*height*3 bytes. Returns false on invalid dimensions.
bool image_from_rgb(const uint8_t* rgb, int width, int height, Image& out);

// Borrow a row-stride-aware public image view. The caller retains ownership and
// keeps it alive for the full inference call.
bool image_from_view(const aicore_image_view& view, Image& out);

// Encode top-first pixels to an in-memory PNG, preserving RGBA/BGRA alpha.
// Honors the caller's row stride. Invalid input leaves `out` unchanged.
// Qt image plugins and encoded storage remain inside the private implementation.
bool encode_png(const aicore_image_view& view, std::vector<uint8_t>& out);

// OpenCV-aligned (cv2.resize INTER_LINEAR, 11-bit fixed point) bilinear
// resample of a strided source view into a tightly packed `dw`x`dh`x`cn`
// destination. Bit-exact with the OpenCV 8U path (coefficients, border
// handling and the uchar vertical rounding); see the .cpp for the rules.
void cv_resize_linear_u8(
        const Image& src, uint8_t* dst, int dw, int dh, int cn);

}  // namespace img
}  // namespace aicore
