#pragma once
#include <vector>
#include "image_io.hpp"
#include "detect.hpp"

namespace fd {

class ModelLoader;
class Backend;

// Raw per-stride YuNet head outputs for ONE stride, each flattened to row order
// (H*W, C) (height-major then width; YuNet has num_anchors == 1) by the head's
// in-graph Transpose([0,2,3,1]) + Reshape:
//   * cls  : post-sigmoid classification, C = 1
//   * obj  : post-sigmoid objectness,     C = 1
//   * bbox : raw distance regression,     C = 4  (dx,dy,dw,dh)
//   * kps  : raw 5-point landmark offsets, C = 10
// These are the inputs the YuNet decode consumes. Gated component-by-component
// against the raw ONNX head tensors (yunet_out_0..11) BEFORE any decode.
struct YunetRawOut {
    std::vector<float> cls;
    std::vector<float> obj;
    std::vector<float> bbox;
    std::vector<float> kps;
};

// Run the YuNet detector forward pass (small CNN backbone + FPN neck + per-stride
// cls/obj/bbox/kps heads) as a ggml compute graph over an already-resized
// `det_input_size` square RGB image, returning the RAW per-stride head outputs for
// strides [8,16,32] in that order. The whole forward is replayed metadata-driven
// through the shared run_onnx_graph_multi interpreter (the same path as the SCRFD
// det_2.5g detector) from the `det.`-prefixed weights + the facedetect.detector.graph
// KV; YuNet adds NO new ops. The blob is raw BGR planes (mean 0, std 1, swapRB),
// matching cv2.FaceDetectorYN's blobFromImage(img, 1.0, size).
std::vector<YunetRawOut> yunet_forward(const ModelLoader& ml,
                                       const Image& blob_img, Backend& be);

// Decode the raw YuNet heads into faces, reproducing cv2.FaceDetectorYN exactly:
// per stride s and grid cell (c,r), score = sqrt(cls*obj); cx=(c+dx)*s, cy=(r+dy)*s,
// w=exp(dw)*s, h=exp(dh)*s, x1=cx-w/2; landmark k = (kps+c/r)*s. Candidates above
// det_score_thresh are NMS'd (greedy, OpenCV cv2.dnn.NMSBoxes IoU - standard area,
// NO Pascal-VOC +1) in the det_input_size space, then survivors are scaled to source
// by the per-axis factors (sx,sy) = (orig_w/size, orig_h/size). Returns descending
// score order.
std::vector<Detection> yunet_decode(const ModelLoader& ml,
                                    const std::vector<YunetRawOut>& heads,
                                    float sx, float sy);

// Plain resize `src` to `size`x`size` (cv2.resize INTER_LINEAR, 11-bit fixed point;
// same kernel as the SCRFD letterbox, but NO aspect preservation and NO padding -
// FaceDetectorYN distorts the image to the square net input). Sets the per-axis
// back-scales (sx,sy) = (orig_w/size, orig_h/size) used by the decode to map boxes
// back to source pixels.
void yunet_resize(const Image& src, int size, Image& out, float& sx, float& sy);

// Full production YuNet detection path from a decoded RGB source image:
// yunet_resize -> to_blob(raw BGR) -> yunet_forward -> yunet_decode (+ scale-back).
std::vector<Detection> yunet_detect(const ModelLoader& ml, const Image& img);

} // namespace fd
