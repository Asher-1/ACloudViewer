#pragma once

// Compatibility forwarding header. The blocked-layout (nChw16c) island
// kernels lived here when facedetect/ArcFace was their only consumer; the
// OCR (PP-OCR) graphs became the second consumer, so the kernels moved to
// the shared AICore common layer (common/blocked_conv.{hpp,cpp}) per the
// two-real-consumers promotion rule. This header keeps the historical
// `fd::` spellings working for the facedetect task sources.

#include "common/blocked_conv.hpp"

namespace fd {

using aicore::blk::blocked_add;
using aicore::blk::blocked_add_relu;
using aicore::blk::blocked_bias;
using aicore::blk::blocked_conv1x1;
using aicore::blk::blocked_conv3x3;
using aicore::blk::blocked_prelu;
using aicore::blk::blocked_relu;
using aicore::blk::blocked_reorder_in;
using aicore::blk::blocked_reorder_out;
using aicore::blk::blocked_scale_shift;
using aicore::blk::directconv_conv3x3;

inline bool directconv_blocked_available() {
    return aicore::blk::blocked_available();
}

}  // namespace fd
