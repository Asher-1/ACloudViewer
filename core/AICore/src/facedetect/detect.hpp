#pragma once
#include <vector>
#include "align.hpp"      // fd::Landmarks5
#include "image_io.hpp"

namespace fd {

class ModelLoader;

// One detected face: a pixel-space box (x1,y1,x2,y2), a detection score, and the
// 5 landmarks SCRFD regresses alongside each box.
struct Detection {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    Landmarks5 landmarks{};
};

// Run the SCRFD detector over `img` and return all faces above the score
// threshold, in NMS-keep (descending-score) order, with pixel-space boxes + 5
// landmarks. Full production path: scrfd_letterbox (stb-decoded source) ->
// to_blob -> scrfd_forward (the ggml conv graph) -> host-side anchor decode ->
// NMS. Reproduces insightface SCRFD.forward/detect/nms numerically: per-stride
// (8/16/32) anchor centers (col*stride, row*stride, duplicated num_anchors
// times), distance2bbox / distance2kps with preds scaled by stride, results
// divided by det_scale to map back to source pixels, score filter
// (`score >= det_score_thresh`), then greedy Pascal-VOC IoU NMS at
// det_nms_thresh. Parity is gated by a golden detection set (boxes + landmarks)
// in docs/parity.md (tests/test_detect.cpp), <=1px end to end from a real JPEG.
std::vector<Detection> scrfd_detect(const ModelLoader& ml, const Image& img);

// Greedy IoU non-maximum suppression over `dets` (descending score), returning
// the kept indices. Pure host-side helper, exposed for testing the decode path
// independently of the network graph.
std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh);

} // namespace fd
