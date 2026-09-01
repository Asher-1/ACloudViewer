#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "aicore/image_view.h"

namespace fd {

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

}  // namespace fd
