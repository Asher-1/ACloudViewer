// Qt-based image I/O for FreeSplatter — replaces stb_image dependency.
// Loads an image file, center-crops to a square, resizes to model resolution,
// and converts to NCHW float32 [0,1] layout.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace aicore {
namespace gaussian {

// Load an image file, center-crop to square, resize to size x size,
// scale to [0,1], lay out as CHW (channels, height, width).
// Appends 3*size*size floats to `out` (RGB, channel-major).
// Returns true on success, false on error (sets err).
bool load_image_chw(const std::string & path, int size,
                    std::vector<float> & out, std::string & err);

// Load multiple images and concatenate into a single NCHW buffer.
// Each image contributes 3*size*size floats. Returns true on success.
bool load_images_chw(const std::vector<std::string> & paths, int size,
                     std::vector<float> & out, std::string & err);

} // namespace gaussian
} // namespace aicore
