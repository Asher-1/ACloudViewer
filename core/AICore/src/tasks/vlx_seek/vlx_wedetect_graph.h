// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// WeDetect-Uni proposal detector (detect_tools/WeDetect/generate_proposal.py
// SimpleYOLOWorldDetector, backbone "base"): ConvNeXt-B -> CSPRepBiFPANNeck
// -> YOLO-World BN-contrastive head, as ONE ggml graph, plus the exact host
// post-process of head_predict + forward (sigmoid / top-30000 / per-class
// NMS 0.7 / top-100 / letterbox undo / clamp / validity filter).
//
// Weights come from convert_wedetect_to_gguf.py (BN / layer-scale / logit
// scale folded; GEMM matrices f32, f16 or q8_0). Layout contract identical to
// simplefpn_graph: every activation is token-major ggml ne=(C, W, H).
#pragma once

#include <string>
#include <vector>

#include "tasks/vlx_seek/vlx_image_prep.h"
#include "tasks/vlx_seek/vlx_types.h"

struct ggml_backend;

namespace vlx {

class GgufFile;

struct WeDetectResult {
    std::vector<Box> boxes;  // original-image xyxy, score-descending
    std::vector<float> scores;
    // optional taps (dump mode): token-major (H*W, C) f32
    std::vector<std::vector<float>> taps;  // c1..c4, neck0..2
    std::vector<float> cls_logits;         // (8400, 256) pre-sigmoid
    std::vector<float> bbox_preds;         // (8400, 4) distance * stride
};

class WeDetectGraph {
public:
    WeDetectGraph(ggml_backend* backend, const GgufFile* gguf);
    ~WeDetectGraph();

    // img: decoded RGB (PIL Image.open(...).convert("RGB") equivalent).
    bool run(const ImageU8& img,
             WeDetectResult* out,
             bool want_taps = false,
             std::string* err = nullptr);

    double last_graph_ms() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace vlx
