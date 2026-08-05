#pragma once
#include <vector>
#include "image_io.hpp"

namespace fd {

class ModelLoader;
class Backend;

// Raw per-stride SCRFD head outputs for ONE stride, flattened to insightface's
// row order `(H*W*num_anchors, C)` (height-major, then width, then anchor):
//   * score : post-sigmoid classification, C = 1  (so size H*W*num_anchors)
//   * bbox  : raw distance regression (after the per-stride learned scale),
//             C = 4  (size H*W*num_anchors*4)
//   * kps   : raw 5-point landmark regression, C = 10 (size H*W*num_anchors*10)
// These are the inputs Task 3.2's anchor decode consumes (distance->box/landmark
// against the stride anchor grid). Gated component-by-component against the raw
// ONNX head tensors (scrfd_out_0..8) BEFORE any decode.
struct ScrfdRawOut {
    std::vector<float> score;
    std::vector<float> bbox;
    std::vector<float> kps;
};

// Run the SCRFD detector forward pass (ResNet-style backbone + PAFPN neck +
// three shared stride heads) as a ggml compute graph over an already
// letterboxed `size`x`size` RGB image, returning the RAW per-stride head outputs
// for strides [8, 16, 32] in that order. `blob_img640` must be the square
// detector input (see scrfd_letterbox); to_blob normalization (mean 127.5, std
// 128, R,G,B plane order) is applied internally so the blob matches the
// reference cv2.dnn.blobFromImage(..., swapRB=True).
//
// The conv weights are read by their verbatim `det.*` GGUF names via ModelLoader
// (referenced directly as graph leaves). BN is already folded into the exported
// conv weights+bias, so each block is conv(+bias)(+relu); no separate BN fold.
std::vector<ScrfdRawOut> scrfd_forward(const ModelLoader& ml,
                                       const Image& blob_img640, Backend& be);

// Reproduce insightface SCRFD.detect's letterbox: scale `src` to fit a
// `size`x`size` box preserving aspect ratio (cv2.resize INTER_LINEAR, 11-bit
// fixed-point; matches OpenCV to within 1 LSB - OpenCV's SIMD path narrows the
// horizontal pass to int16 first, so a scalar reimplementation differs by at
// most 1 on ~8% of pixels), then top-left zero-pad into a `size`x`size` RGB
// canvas. Sets
// `det_scale` = new_h/orig_h (== new_w/orig_w), used by the decode to map boxes
// back to source pixels. `new_h = int(size/im_ratio)` when `im_ratio>1`, else
// `new_w = int(size*im_ratio)`, with `im_ratio = orig_h/orig_w`.
void scrfd_letterbox(const Image& src, int size, Image& out, float& det_scale);

} // namespace fd
