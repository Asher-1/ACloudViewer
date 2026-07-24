#pragma once
#include <string>
#include <vector>
namespace aicore {
namespace depth {
// Write a PFM (Portable Float Map) of the depth, H rows x W cols, single channel. Returns false on IO error.
bool write_pfm(const std::string& path, const std::vector<float>& depth, int H, int W);
// Write an 8-bit grayscale PNG (min-max normalized) via stb_image_write. invert=true for near=bright.
bool write_depth_png(const std::string& path, const std::vector<float>& depth, int H, int W, bool invert=true);
} // namespace depth
} // namespace aicore
