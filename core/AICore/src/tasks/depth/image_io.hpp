#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace aicore {
namespace depth {
struct Image {
    int w = 0;
    int h = 0;
    int channels = 3;
    size_t row_stride_bytes = 0;
    const uint8_t* borrowed = nullptr;
    bool bgr = false;
    std::vector<uint8_t> rgb;

    const uint8_t* data() const {
        return borrowed != nullptr ? borrowed : rgb.data();
    }
    size_t stride() const {
        return row_stride_bytes != 0 ? row_stride_bytes
                                     : static_cast<size_t>(w) * channels;
    }
    uint8_t channel(int x, int y, int c) const {
        const uint8_t* pixel = data() + static_cast<size_t>(y) * stride() +
                               static_cast<size_t>(x) * channels;
        if (channels == 1) return pixel[0];
        return pixel[bgr ? 2 - c : c];
    }
    void reset_owned_layout() {
        borrowed = nullptr;
        channels = 3;
        bgr = false;
        row_stride_bytes = static_cast<size_t>(w) * 3;
    }
};

// Borrows a caller-owned RGB/RGBA/GRAY view for the duration of inference.
bool borrow_image_view(const uint8_t* data, int w, int h, int channels,
                       bool bgr,
                       size_t row_stride_bytes, Image& out);
bool load_image_rgb(const std::string& path, Image& out);
bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out);
} // namespace depth
} // namespace aicore
