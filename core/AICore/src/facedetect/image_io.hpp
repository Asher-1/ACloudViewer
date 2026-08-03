#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace fd {

// A decoded 8-bit RGB image: tightly-packed, row-major, `width*height*3` bytes
// (no row padding), channel order R,G,B.
struct Image {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> rgb;   // size == width*height*3
    bool empty() const { return width <= 0 || height <= 0 || rgb.empty(); }
};

// Decode an image FILE (JPEG/PNG/BMP/... via the vendored stb_image) to RGB.
// Returns true on success and fills `out`; false on failure (sets nothing).
bool load_image_rgb(const std::string& path, Image& out);

// Wrap caller-owned raw RGB bytes (copied) into an Image. `rgb` must hold at
// least width*height*3 bytes. Returns false on invalid dimensions.
bool image_from_rgb(const uint8_t* rgb, int width, int height, Image& out);

} // namespace fd
