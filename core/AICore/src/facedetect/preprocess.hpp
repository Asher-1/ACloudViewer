#pragma once
#include <vector>
#include "image_io.hpp"

namespace fd {

// Build the network input blob from an already-resized RGB image.
//
// Produces NCHW f32 of shape [1,3,size,size] applying per-pixel
// `(p - mean) / std` normalization. When `swap_rb` is true the output plane
// order is B,G,R (matching insightface/OpenCV `swapRB=True`, whose nominal
// input is BGR); when false it is R,G,B. The input `Image` must already be
// `size`x`size` (the caller resizes/letterboxes/aligns first).
//
// Per-head parameters used by the pipeline:
//   SCRFD detector  : mean 127.5, std 128.0, swap_rb true
//   ArcFace         : mean 127.5, std 127.5, swap_rb true
//   genderage (ga)  : mean 0.0,   std 1.0,   swap_rb true
//   MiniFASNet (as) : mean 0.0,   std 1.0,   swap_rb true   (reference feeds raw
//                     BGR planes; swap_rb=true on the RGB Image lands on B,G,R)
std::vector<float> to_blob(const Image& rgb, int size, float mean, float std, bool swap_rb);

} // namespace fd
