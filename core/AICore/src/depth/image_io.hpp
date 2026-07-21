#pragma once
#include <string>
#include <vector>
#include <cstddef>
namespace aicore {
namespace depth {
struct Image { int w=0, h=0; std::vector<unsigned char> rgb; };   // HWC uint8
bool load_image_rgb(const std::string& path, Image& out);
bool load_image_rgb_buffer(const unsigned char* bytes, size_t len, Image& out);
} // namespace depth
} // namespace aicore
