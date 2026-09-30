// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/facedetect/yunet_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "antispoof_graph.hpp"  // run_onnx_graph_multi (shared ONNX-graph interpreter)
#include "tasks/facedetect/backend.hpp"
#include "tasks/facedetect/common.hpp"
#include "tasks/facedetect/detect.hpp"
#include "tasks/facedetect/model_loader.hpp"
#include "tasks/facedetect/preprocess.hpp"

namespace fd {

namespace {

// Greedy NMS with OpenCV cv2.dnn.NMSBoxes IoU: standard box area (w*h), NO
// Pascal-VOC +1 (that +1 is the insightface SCRFD convention; YuNet's reference
// is OpenCV).
float iou_std(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms_std(const std::vector<Detection>& dets, float iou_thresh) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int i, int j) { return dets[i].score > dets[j].score; });
    std::vector<char> suppressed(dets.size(), 0);
    std::vector<int> keep;
    for (size_t a = 0; a < order.size(); ++a) {
        const int i = order[a];
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (size_t b = a + 1; b < order.size(); ++b) {
            const int j = order[b];
            if (suppressed[j]) continue;
            if (iou_std(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
        }
    }
    return keep;
}

}  // namespace

void yunet_resize(
        const Image& src, int size, Image& out, float& sx, float& sy) {
    out.width = size;
    out.height = size;
    out.borrowed_data = nullptr;
    out.row_stride_bytes = static_cast<size_t>(size) * 3;
    out.channels = 3;
    out.rgb.assign((size_t)size * size * 3, 0);
    cv_resize_linear_u8(src, out.rgb.data(), size, size, 3);
    sx = (float)src.width / (float)size;
    sy = (float)src.height / (float)size;
}

std::vector<YunetRawOut> yunet_forward(const ModelLoader& ml,
                                       const Image& blob_img,
                                       Backend& be) {
    const FaceConfig& cfg = ml.config();
    const int S = (int)cfg.det_input_size;
    const size_t nstride = cfg.det_strides.size();
    std::vector<YunetRawOut> outs(nstride);
    // YuNet's 12 graph outputs: cls/obj/bbox/kps x strides 8/16/32 (4 per
    // stride).
    if (cfg.det_graph_outputs.size() != nstride * 4) {
        FD_LOG("yunet_forward: expected %zu outputs, graph has %zu",
               nstride * 4, cfg.det_graph_outputs.size());
        return outs;
    }
    // FaceDetectorYN blobFromImage(img, 1.0, size): raw BGR planes, no
    // mean/std. Our Image is RGB, so swap_rb=true lands on B,G,R (== a BGR
    // image, no swap).
    std::vector<float> blob =
            to_blob(blob_img, S, 0.0f, 1.0f, /*swap_rb=*/true);
    std::vector<std::vector<float>> heads;
    try {
        heads = run_onnx_graph_multi(ml, "det.", cfg.det_graph,
                                     cfg.det_graph_input, cfg.det_graph_outputs,
                                     blob, S, be, 1e-5f);
    } catch (const std::exception& e) {
        FD_LOG("yunet_forward: %s", e.what());
        return outs;
    }
    for (size_t i = 0; i < nstride; ++i) {
        outs[i].cls = std::move(heads[i]);
        outs[i].obj = std::move(heads[nstride + i]);
        outs[i].bbox = std::move(heads[2 * nstride + i]);
        outs[i].kps = std::move(heads[3 * nstride + i]);
    }
    return outs;
}

std::vector<Detection> yunet_decode(const ModelLoader& ml,
                                    const std::vector<YunetRawOut>& heads,
                                    float sx,
                                    float sy) {
    const FaceConfig& cfg = ml.config();
    const int S = (int)cfg.det_input_size;
    const float thresh = cfg.det_score_thresh;

    // Build candidates in the NET (S x S) space; NMS is done there so its IoU
    // matches cv2.FaceDetectorYN (anisotropic source scaling is NOT
    // IoU-preserving). Survivors are scaled to source afterward.
    std::vector<Detection> cand;
    for (size_t si = 0; si < heads.size() && si < cfg.det_strides.size();
         ++si) {
        const int stride = cfg.det_strides[si];
        const int cols = S / stride, rows = S / stride;
        const YunetRawOut& o = heads[si];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const int idx =
                        r * cols + c;  // row-major (NHWC, num_anchors == 1)
                // cls/obj are post-sigmoid in-graph (in [0,1]); score =
                // geometric mean.
                const float score = std::sqrt(o.cls[idx] * o.obj[idx]);
                if (score < thresh) continue;
                // Anchor-free bbox: center offset + log-scaled size, in stride
                // units.
                const float cx = (c + o.bbox[idx * 4 + 0]) * stride;
                const float cy = (r + o.bbox[idx * 4 + 1]) * stride;
                const float w = std::exp(o.bbox[idx * 4 + 2]) * stride;
                const float h = std::exp(o.bbox[idx * 4 + 3]) * stride;
                Detection d;
                d.score = score;
                d.x1 = cx - w * 0.5f;
                d.y1 = cy - h * 0.5f;
                d.x2 = d.x1 + w;
                d.y2 = d.y1 + h;
                for (int k = 0; k < 5; ++k) {
                    d.landmarks[k][0] =
                            (o.kps[idx * 10 + k * 2 + 0] + c) * stride;
                    d.landmarks[k][1] =
                            (o.kps[idx * 10 + k * 2 + 1] + r) * stride;
                }
                cand.push_back(d);
            }
    }

    const std::vector<int> keep = nms_std(cand, cfg.det_nms_thresh);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int i : keep) {
        Detection d = cand[i];
        d.x1 *= sx;
        d.x2 *= sx;
        d.y1 *= sy;
        d.y2 *= sy;
        for (int k = 0; k < 5; ++k) {
            d.landmarks[k][0] *= sx;
            d.landmarks[k][1] *= sy;
        }
        out.push_back(d);
    }
    return out;
}

std::vector<Detection> yunet_detect(const ModelLoader& ml, const Image& img) {
    const int S = (int)ml.config().det_input_size;
    Image rz;
    float sx = 0.f, sy = 0.f;
    yunet_resize(img, S, rz, sx, sy);
    FD_ASSERT(sx > 0.0f && sy > 0.0f);
    auto heads = yunet_forward(ml, rz, global_backend());
    return yunet_decode(ml, heads, sx, sy);
}

}  // namespace fd
