// Qt-based image I/O for FreeSplatter.
// Loads an image file, center-crops to a square, resizes to model resolution,
// and converts to NCHW float32 [0,1] layout.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "aicore/image_view.h"

class QImage;

namespace aicore {
namespace gaussian {

// Load an image file, center-crop to square, resize to size x size,
// scale to [0,1], lay out as CHW (channels, height, width).
// Appends 3*size*size floats to `out` (RGB, channel-major).
// Returns true on success, false on error (sets err).
bool load_image_chw(const std::string& path,
                    int size,
                    std::vector<float>& out,
                    std::string& err);

// Load multiple images and concatenate into a single NCHW buffer.
// Each image contributes 3*size*size floats. Returns true on success.
bool load_images_chw(const std::vector<std::string>& paths,
                     int size,
                     std::vector<float>& out,
                     std::string& err);

// Center-crop a decoded image to a square, resize to size x size, scale to
// [0,1], and append 3*size*size CHW floats to `out`. Shared tail of the
// file-loading and image_view entry points (F-01 batch C). Returns true on
// success, false on error (sets err).
bool append_qimage_chw(const QImage& image,
                       int size,
                       std::vector<float>& out,
                       std::string& err);

// Structured-memory twin of load_image_chw (F-01 batch C): the view must be
// AICORE_IMAGE_RGB8 and is borrowed for the duration of the call.
bool append_image_view_chw(const aicore_image_view& view,
                           int size,
                           std::vector<float>& out,
                           std::string& err);

// Append multiple image views as a single NCHW buffer.
// Each view contributes 3*size*size floats. Returns true on success.
bool append_image_views_chw(const aicore_image_view* views,
                            int32_t n_views,
                            int size,
                            std::vector<float>& out,
                            std::string& err);

}  // namespace gaussian
}  // namespace aicore
