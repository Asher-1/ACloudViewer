#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rmbg {

bool decode_preprocess(const void * bytes, int length, int size,
                       const float mean[3], const float std[3],
                       std::vector<uint8_t> & original_rgba, int & width, int & height,
                       std::vector<float> & input_nchw, std::string & err);

/* Same contract as decode_preprocess but takes a tightly-packed 8-bit RGB
 * buffer (HWC row-major, width*height*3 bytes) directly — used by the C API
 * so plugins can skip the encode/decode round-trip. */
bool decode_preprocess_rgb(const uint8_t * rgb, int rgb_w, int rgb_h, int size,
                           const float mean[3], const float std[3],
                           std::vector<uint8_t> & original_rgba, int & width, int & height,
                           std::vector<float> & input_nchw, std::string & err);

bool encode_result_png(const std::vector<uint8_t> & original_rgba, int width, int height,
                       const std::vector<float> & alpha, int alpha_width, int alpha_height,
                       std::vector<uint8_t> & png, std::string & err);

/* Bicubic-upsample a float alpha matte (values in [0,1]) to (width, height)
 * and write it as an 8-bit row-major array (0 = background, 255 = foreground).
 * This is the primitive future plugins use to build their own composites. */
bool upsample_alpha(const std::vector<float> & alpha, int alpha_width, int alpha_height,
                    int width, int height, std::vector<uint8_t> & out_alpha8,
                    std::string & err);

} // namespace rmbg
