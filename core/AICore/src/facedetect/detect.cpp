#include "detect.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "scrfd_graph.hpp"
#include "yunet_graph.hpp"
#include "backend.hpp"

#include <algorithm>
#include <numeric>

namespace fd {

// IoU with the Pascal-VOC `+1` box convention insightface SCRFD.nms uses:
// areas = (x2-x1+1)*(y2-y1+1), intersection sides clamped at 0 then `+1`. The
// `+1` matters: it is what the golden detection set was produced with, and a
// no-`+1` IoU drifts the suppress decisions on tightly-packed faces.
static float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1 + 1.0f);
    const float ih = std::max(0.0f, iy2 - iy1 + 1.0f);
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1 + 1.0f) * (a.y2 - a.y1 + 1.0f);
    const float area_b = (b.x2 - b.x1 + 1.0f) * (b.y2 - b.y1 + 1.0f);
    const float uni = area_a + area_b - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh) {
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
            if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
        }
    }
    return keep;
}

std::vector<Detection> scrfd_detect(const ModelLoader& ml, const Image& img) {
    const FaceConfig& c = ml.config();
    // YuNet (OpenCV-Zoo, Apache) is a different detector architecture/decode than
    // SCRFD; dispatch to its dedicated path so all callers (Model::detect/embed/
    // analyze) transparently use the configured detector.
    if (c.detector == "yunet") return yunet_detect(ml, img);

    const int S = (int)c.det_input_size;            // 640
    const int na = (int)c.det_num_anchors;          // 2

    // Production preprocess: stb-decoded source -> aspect-preserving letterbox
    // into the square detector input. det_scale maps decoded boxes back to
    // source pixels (== reference's float(new_h)/orig_h).
    Image lb;
    float det_scale = 0.f;
    scrfd_letterbox(img, S, lb, det_scale);
    // det_scale is geometry-only (derived from source dims, not pixels): it must
    // be strictly positive or every decoded box collapses. Hard-assert it, per
    // the Task 3.1 review (silent zero would map all boxes to +inf).
    FD_ASSERT(det_scale > 0.0f);

    auto raw = scrfd_forward(ml, lb, global_backend());
    FD_ASSERT(raw.size() == c.det_strides.size());

    std::vector<Detection> cand;
    for (size_t si = 0; si < c.det_strides.size(); ++si) {
        const int stride = c.det_strides[si];
        const int H = S / stride, W = S / stride;
        const ScrfdRawOut& o = raw[si];

        for (int r = 0; r < H; ++r)
            for (int col = 0; col < W; ++col)
                for (int a = 0; a < na; ++a) {
                    const int idx = ((r * W + col) * na) + a;
                    const float sc = o.score[idx];
                    // insightface filters with `scores >= threshold`.
                    if (sc < c.det_score_thresh) continue;

                    // Anchor center: insightface mgrid grid is (x=col, y=row),
                    // multiplied by stride; the same center serves all na anchors.
                    const float cx = (float)col * stride;
                    const float cy = (float)r * stride;

                    // bbox/kps preds are scaled by `stride` here (scrfd_forward
                    // captures the RAW per-stride heads, pre-stride-multiply, so
                    // the Task 3.1 graph gate stays a clean compare vs raw ONNX).
                    const float l  = o.bbox[idx * 4 + 0] * stride;
                    const float t  = o.bbox[idx * 4 + 1] * stride;
                    const float rr = o.bbox[idx * 4 + 2] * stride;
                    const float bb = o.bbox[idx * 4 + 3] * stride;

                    Detection d;
                    d.score = sc;
                    d.x1 = (cx - l)  / det_scale;
                    d.y1 = (cy - t)  / det_scale;
                    d.x2 = (cx + rr) / det_scale;
                    d.y2 = (cy + bb) / det_scale;
                    for (int k = 0; k < 5; ++k) {
                        const float px = cx + o.kps[idx * 10 + k * 2 + 0] * stride;
                        const float py = cy + o.kps[idx * 10 + k * 2 + 1] * stride;
                        d.landmarks[k][0] = px / det_scale;
                        d.landmarks[k][1] = py / det_scale;
                    }
                    cand.push_back(d);
                }
    }

    const std::vector<int> keep = nms(cand, c.det_nms_thresh);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int i : keep) out.push_back(cand[i]);
    return out;
}

} // namespace fd
