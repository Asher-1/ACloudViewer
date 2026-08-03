#include "yunet_graph.hpp"
#include "antispoof_graph.hpp"   // run_onnx_graph_multi (shared ONNX-graph interpreter)
#include "detect.hpp"
#include "backend.hpp"
#include "model_loader.hpp"
#include "preprocess.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace fd {

namespace {

// cv2.resize INTER_LINEAR for uint8, 11-bit fixed point (OpenCV HResize/VResize 8U
// path), full-frame (no padding) - the YuNet equivalent of scrfd_graph's letterbox
// resize. cv2.FaceDetectorYN distorts the source to the square net input, so this
// resizes the whole image to dw x dh with independent x/y scales. Matches OpenCV to
// within 1 LSB on a minority of pixels (the SIMD path narrows the horizontal pass to
// int16 first); the raw heads are robust to that drift, and the decode parity gate
// is additionally exercised from the reference's EXACT resized pixels.
void cv_resize_linear_u8(const uint8_t* src, int sw, int sh,
                         uint8_t* dst, int dw, int dh, int cn) {
    const int BITS = 11;
    const float SCALE = (float)(1 << BITS);
    auto sat_short = [](float v) -> int {
        int i = (int)std::lrint(v);
        return i < -32768 ? -32768 : (i > 32767 ? 32767 : i);
    };
    std::vector<int> xofs(dw), ialpha(dw * 2), yofs(dh), ibeta(dh * 2);
    const double sxs = (double)sw / dw, sys = (double)sh / dh;
    for (int dx = 0; dx < dw; ++dx) {
        float fx = (float)((dx + 0.5) * sxs - 0.5);
        int ix = (int)std::floor(fx);
        fx -= ix;
        if (ix < 0) { ix = 0; fx = 0.f; }
        if (ix >= sw - 1) { ix = sw - 1; fx = 0.f; }
        xofs[dx] = ix;
        ialpha[dx * 2] = sat_short((1.f - fx) * SCALE);
        ialpha[dx * 2 + 1] = sat_short(fx * SCALE);
    }
    for (int dy = 0; dy < dh; ++dy) {
        float fy = (float)((dy + 0.5) * sys - 0.5);
        int iy = (int)std::floor(fy);
        fy -= iy;
        if (iy < 0) { iy = 0; fy = 0.f; }
        if (iy >= sh - 1) { iy = sh - 1; fy = 0.f; }
        yofs[dy] = iy;
        ibeta[dy * 2] = sat_short((1.f - fy) * SCALE);
        ibeta[dy * 2 + 1] = sat_short(fy * SCALE);
    }
    for (int dy = 0; dy < dh; ++dy) {
        const int sy0 = yofs[dy], sy1 = std::min(sy0 + 1, sh - 1);
        const int b0 = ibeta[dy * 2], b1 = ibeta[dy * 2 + 1];
        const uint8_t* r0 = src + (size_t)sy0 * sw * cn;
        const uint8_t* r1 = src + (size_t)sy1 * sw * cn;
        uint8_t* drow = dst + (size_t)dy * dw * cn;
        for (int dx = 0; dx < dw; ++dx) {
            const int ix = xofs[dx], ix1 = std::min(ix + 1, sw - 1);
            const int a0 = ialpha[dx * 2], a1 = ialpha[dx * 2 + 1];
            for (int c = 0; c < cn; ++c) {
                const int p0 = r0[ix * cn + c] * a0 + r0[ix1 * cn + c] * a1;
                const int p1 = r1[ix * cn + c] * a0 + r1[ix1 * cn + c] * a1;
                int v = ((int64_t)p0 * b0 + (int64_t)p1 * b1 + (1 << 21)) >> 22;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                drow[dx * cn + c] = (uint8_t)v;
            }
        }
    }
}

// Greedy NMS with OpenCV cv2.dnn.NMSBoxes IoU: standard box area (w*h), NO Pascal-VOC
// +1 (that +1 is the insightface SCRFD convention; YuNet's reference is OpenCV).
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

} // namespace

void yunet_resize(const Image& src, int size, Image& out, float& sx, float& sy) {
    out.width = size;
    out.height = size;
    out.rgb.assign((size_t)size * size * 3, 0);
    cv_resize_linear_u8(src.rgb.data(), src.width, src.height,
                        out.rgb.data(), size, size, 3);
    sx = (float)src.width / (float)size;
    sy = (float)src.height / (float)size;
}

std::vector<YunetRawOut> yunet_forward(const ModelLoader& ml,
                                       const Image& blob_img, Backend& be) {
    const FaceConfig& cfg = ml.config();
    const int S = (int)cfg.det_input_size;
    const size_t nstride = cfg.det_strides.size();
    std::vector<YunetRawOut> outs(nstride);
    // YuNet's 12 graph outputs: cls/obj/bbox/kps x strides 8/16/32 (4 per stride).
    if (cfg.det_graph_outputs.size() != nstride * 4) {
        FD_LOG("yunet_forward: expected %zu outputs, graph has %zu",
               nstride * 4, cfg.det_graph_outputs.size());
        return outs;
    }
    // FaceDetectorYN blobFromImage(img, 1.0, size): raw BGR planes, no mean/std.
    // Our Image is RGB, so swap_rb=true lands on B,G,R (== a BGR image, no swap).
    std::vector<float> blob = to_blob(blob_img, S, 0.0f, 1.0f, /*swap_rb=*/true);
    std::vector<std::vector<float>> heads;
    try {
        heads = run_onnx_graph_multi(ml, "det.", cfg.det_graph, cfg.det_graph_input,
                                     cfg.det_graph_outputs, blob, S, be, 1e-5f);
    } catch (const std::exception& e) {
        FD_LOG("yunet_forward: %s", e.what());
        return outs;
    }
    for (size_t i = 0; i < nstride; ++i) {
        outs[i].cls  = std::move(heads[i]);
        outs[i].obj  = std::move(heads[nstride + i]);
        outs[i].bbox = std::move(heads[2 * nstride + i]);
        outs[i].kps  = std::move(heads[3 * nstride + i]);
    }
    return outs;
}

std::vector<Detection> yunet_decode(const ModelLoader& ml,
                                    const std::vector<YunetRawOut>& heads,
                                    float sx, float sy) {
    const FaceConfig& cfg = ml.config();
    const int S = (int)cfg.det_input_size;
    const float thresh = cfg.det_score_thresh;

    // Build candidates in the NET (S x S) space; NMS is done there so its IoU matches
    // cv2.FaceDetectorYN (anisotropic source scaling is NOT IoU-preserving). Survivors
    // are scaled to source afterward.
    std::vector<Detection> cand;
    for (size_t si = 0; si < heads.size() && si < cfg.det_strides.size(); ++si) {
        const int stride = cfg.det_strides[si];
        const int cols = S / stride, rows = S / stride;
        const YunetRawOut& o = heads[si];
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                const int idx = r * cols + c;  // row-major (NHWC, num_anchors == 1)
                // cls/obj are post-sigmoid in-graph (in [0,1]); score = geometric mean.
                const float score = std::sqrt(o.cls[idx] * o.obj[idx]);
                if (score < thresh) continue;
                // Anchor-free bbox: center offset + log-scaled size, in stride units.
                const float cx = (c + o.bbox[idx * 4 + 0]) * stride;
                const float cy = (r + o.bbox[idx * 4 + 1]) * stride;
                const float w  = std::exp(o.bbox[idx * 4 + 2]) * stride;
                const float h  = std::exp(o.bbox[idx * 4 + 3]) * stride;
                Detection d;
                d.score = score;
                d.x1 = cx - w * 0.5f;
                d.y1 = cy - h * 0.5f;
                d.x2 = d.x1 + w;
                d.y2 = d.y1 + h;
                for (int k = 0; k < 5; ++k) {
                    d.landmarks[k][0] = (o.kps[idx * 10 + k * 2 + 0] + c) * stride;
                    d.landmarks[k][1] = (o.kps[idx * 10 + k * 2 + 1] + r) * stride;
                }
                cand.push_back(d);
            }
    }

    const std::vector<int> keep = nms_std(cand, cfg.det_nms_thresh);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int i : keep) {
        Detection d = cand[i];
        d.x1 *= sx; d.x2 *= sx;
        d.y1 *= sy; d.y2 *= sy;
        for (int k = 0; k < 5; ++k) { d.landmarks[k][0] *= sx; d.landmarks[k][1] *= sy; }
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

} // namespace fd
