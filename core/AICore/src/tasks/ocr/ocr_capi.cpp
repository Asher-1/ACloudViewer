// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C API implementation of the `ocr` task inference contract
// (core/AICore/include/aicore/ocr_capi.h). The pipeline is:
//
//   image_view -> bilinear resize (limit side, multiple of 32) -> NCHW
//   normalize -> det interpreter -> DB probability map -> threshold +
//   connected components (BFS) -> boxes + unclip offset -> per-box bilinear
//   crop -> 48xH rec batch -> rec interpreter -> CTC greedy decode ->
//   typed aicore_ocr_box{quad, text, score}.
//
// The DB post-process is hand-written (no OpenCV in the AICore tree). The
// first release uses per-component bounding boxes with the DB offset
// formula; polygon-accurate unclip is deferred to the det-eval round.

#include "aicore/ocr_capi.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/blocked_conv.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/image_utils.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ocr_runtime.hpp"

namespace {

thread_local std::string g_lastError;

void setLastError(const std::string& message) { g_lastError = message; }

constexpr float kDetDefaultMaxSide = 960.0f;
// Official PP-OCR det defaults (the inference.yml of PP-OCRv4_mobile_det
// plus the paddlex DetResizeForTest constructor defaults).
constexpr float kDetOfficialLimitSide = 736.0f;
constexpr float kDetOfficialMaxSideLimit = 4000.0f;
constexpr float kDetDefaultBoxThresh = 0.3f;
constexpr float kDetDefaultBoxScoreThresh = 0.6f;
constexpr float kDetDefaultUnclipRatio = 1.5f;
constexpr int kRecHeight = 48;
constexpr int kRecMaxWidth = 320;
constexpr int kDetAlign = 32;

// PP-OCRv4 preprocessing constants (official DetPreProcess/RecPreProcess).
constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kRecMean = 0.5f;
constexpr float kRecStd = 0.5f;
// Fixed rec batch chunk (run- and recognize-phase): see the run-phase note
// for the measured scaling curve and the A/B rationale. The linear segment
// reaches N=72 (VRAM ~3 GiB); the cliff is between 72 and 96, where the
// depthwise im2col workspace pushes the allocator against VRAM (7.6+ GiB
// resident, 20 GiB peak at 96) and every compute re-warms the CUDA graphs.
constexpr int64_t kRecChunkBatch = 48;
// Vulkan sessions run the direct-conv route (no im2col materialization), so
// the workspace scales LINEARLY with N and there are no CUDA graphs to
// re-warm. The per-chunk cost there is dominated by fixed per-op dispatch
// overhead (~58.6 ms per 40-row chunk, of which only ~14 ms is kernel time),
// so the cap is raised to cover a dense image in a single chunk: 1 compute
// instead of 3 cuts the fixed overhead proportionally (measured this
// session: 3x40 rows -> 176 ms rec, 1x104 rows -> 81 ms rec).
constexpr int64_t kRecChunkBatchVulkan = 112;
// Chunk cap per execution backend. CUDA sessions keep 48 (the im2col VRAM
// cliff); Vulkan single-backend sessions keep 112. CPU sessions never batch
// (the blocked island wins there), so the cap value is irrelevant.
inline int64_t recChunkBatchCap(ggml_backend_t gpuBackend) {
    if (gpuBackend && !ggml_common::is_cpu_backend(gpuBackend)) {
        const char* name = ggml_backend_name(gpuBackend);
        if (name && std::string(name).find("Vulkan") != std::string::npos)
            return kRecChunkBatchVulkan;
    }
    return kRecChunkBatch;
}
// Official rec pad (paddlex text_recognition/processors.py resize_norm_img
// L61-62): padding_im = np.zeros((imgC, imgH, imgW)) in the NORMALIZED
// domain, i.e. the fill is 0.0 = pixel 127.5 = (0.5 - mean) / std. A black
// (-1) pad is out-of-distribution: the model was trained on ratio-bucketed
// batches whose right side is exactly this gray fill, and the SVTR
// self-attention mixes pad keys into every output position.
constexpr float kRecPadFill = (0.5f - kRecMean) / kRecStd;

struct OcrGraph {
    ggml_context* ctx = nullptr;  // graph tensors (no_alloc)
    ggml_cgraph* graph = nullptr;
    ggml_tensor* input = nullptr;
    ggml_tensor* output = nullptr;
    ggml_gallocr_t alloc = nullptr;        // CPU gallocr fast path
    ggml_backend_sched_t sched = nullptr;  // GPU scheduler (device-resident)
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    int64_t inputN = 0;

    bool matches(int64_t c, int64_t h, int64_t w) const {
        return graph && inputC == c && inputH == h && inputW == w;
    }

    void clear() {
        if (alloc) ggml_gallocr_free(alloc);
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
        ctx = nullptr;
        graph = nullptr;
        input = nullptr;
        output = nullptr;
        alloc = nullptr;
        sched = nullptr;
        inputC = inputH = inputW = inputN = 0;
    }
};

// --- Stage helpers (all pure functions over plain buffers) -----------------

// CHW f32 packing with the official PP-OCR normalization. dstW is the
// packed row width: the rec path passes the fixed kRecMaxWidth so every row
// is laid out as [content | right-side pad] exactly like the official
// resize_norm_img (padding_im[:, :, 0:resized_w] = normalized). Appending
// the pad after a w-wide pack would scramble channels/rows the moment
// dstW > w (the packed (3,h,w) re-read as (3,h,dstW) interleaves planes) —
// the root cause of the historical empty/garbled rec text on short crops.
// CHW f32 packing with the official PP-OCR normalization. dstW is the
// packed row width: the rec path passes the fixed kRecMaxWidth so every row
// is laid out as [content | right-side pad] exactly like the official
// resize_norm_img (padding_im[:, :, ..., w:] = 0 in normalized space). The
// pack must emit the final stride IN one pass: appending the pad after a
// w-wide pack would scramble channels/rows the moment dstW > w (the packed
// (3,h,w) re-read as (3,h,dstW) interleaves planes) — the root cause of the
// historical empty/garbled rec text on short crops.
void packNchwImageTo(const aicore::img::Image& img,
                     float mean,
                     float stdDev,
                     int dstW,
                     float padValue,
                     float* nchw) {
    const int w = img.width;
    const int h = img.height;
    const float scale = 1.0f / 255.0f;
    // Pre-fill the pad columns once per plane, then overwrite the content
    // region (the pad lives at the row END: [content | pad] per row).
    const size_t plane = static_cast<size_t>(h) * dstW;
    for (int ch = 0; ch < 3; ++ch) {
        float* p = nchw + static_cast<size_t>(ch) * plane;
        for (int y = 0; y < h; ++y) {
            float* row = p + static_cast<size_t>(y) * dstW;
            for (int x = w; x < dstW; ++x) row[x] = padValue;
        }
    }
    for (int ch = 0; ch < 3; ++ch) {
        float* planePtr = nchw + static_cast<size_t>(ch) * plane;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                planePtr[static_cast<size_t>(y) * dstW + x] =
                        (img.channel(x, y, ch) * scale - mean) / stdDev;
            }
        }
    }
}

void packNchwImage(const aicore::img::Image& img,
                   float mean,
                   float stdDev,
                   int dstW,
                   float padValue,
                   std::vector<float>& nchw) {
    nchw.assign(static_cast<size_t>(3) * img.height * dstW, padValue);
    packNchwImageTo(img, mean, stdDev, dstW, padValue, nchw.data());
}

struct DetBox {
    // Corner polygon in det-map pixel coordinates, ordered clockwise from
    // the top-left corner (Paddle order_points_clockwise). Rotated text
    // yields a genuinely non-axis-aligned quad.
    float quad[4][2];
    float score;  // component mean probability
};

// Official DB component gate (det_db_box_thresh in the upstream det
// pipeline): components whose mean probability sits below it are dropped
// before the unclip expands them.
constexpr float kComponentScoreThresh = 0.6f;

// Andrew monotone chain convex hull over the component boundary points.
// O(n log n); the hull feeds the minimum-area rectangle below.
std::vector<std::pair<float, float>> convexHull(
        std::vector<std::pair<float, float>> pts) {
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::vector<std::pair<float, float>> h(2 * n);
    int k = 0;
    auto cross = [](const std::pair<float, float>& o,
                    const std::pair<float, float>& a,
                    const std::pair<float, float>& b) {
        return (a.first - o.first) * (b.second - o.second) -
               (a.second - o.second) * (b.first - o.first);
    };
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i) {
        while (k >= t && cross(h[k - 2], h[k - 1], pts[i]) <= 0) --k;
        h[k++] = pts[i];
    }
    h.resize(k - 1);
    return h;
}

// Corner-center form of the minimum-area rectangle.
struct RotRect {
    float cx = 0, cy = 0;  // center
    float ux = 1, uy = 0;  // width direction (unit)
    float vx = 0, vy = 1;  // height direction (unit)
    float halfW = 0, halfH = 0;
};

// Rectangle corners -> clockwise-from-top-left quad (Paddle
// order_points_clockwise: min(x+y), max(x-y), max(x+y), min(x-y)).
void rectToQuad(const RotRect& r, float quad[4][2]) {
    const float corners[4][2] = {
            {r.cx + r.halfW * r.ux + r.halfH * r.vx,
             r.cy + r.halfW * r.uy + r.halfH * r.vy},
            {r.cx + r.halfW * r.ux - r.halfH * r.vx,
             r.cy + r.halfW * r.uy - r.halfH * r.vy},
            {r.cx - r.halfW * r.ux - r.halfH * r.vx,
             r.cy - r.halfW * r.uy - r.halfH * r.vy},
            {r.cx - r.halfW * r.ux + r.halfH * r.vx,
             r.cy - r.halfW * r.uy + r.halfH * r.vy},
    };
    int order[4] = {0, 1, 2, 3};
    float sums[4], diffs[4];
    int iMinSum = 0, iMaxDiff = 0, iMaxSum = 0, iMinDiff = 0;
    for (int i = 0; i < 4; ++i) {
        sums[i] = corners[i][0] + corners[i][1];
        diffs[i] = corners[i][0] - corners[i][1];
    }
    for (int i = 1; i < 4; ++i) {
        if (sums[i] < sums[iMinSum]) iMinSum = i;
        if (sums[i] > sums[iMaxSum]) iMaxSum = i;
        if (diffs[i] > diffs[iMaxDiff]) iMaxDiff = i;
        if (diffs[i] < diffs[iMinDiff]) iMinDiff = i;
    }
    order[0] = iMinSum;
    order[1] = iMaxDiff;
    order[2] = iMaxSum;
    order[3] = iMinDiff;
    for (int i = 0; i < 4; ++i) {
        quad[i][0] = corners[order[i]][0];
        quad[i][1] = corners[order[i]][1];
    }
}
// Official DBPostProcess boxes_from_bitmap semantics (paddlex
// text_detection/processors.py): 8-connected components over the binarized
// map (cv2.findContours RETR_LIST is 8-connected), a minimum-area rectangle
// over each component's convex hull (cv2.minAreaRect), the box_thresh gate
// measured as the mean probability over the raw rect region
// (score_mode="fast" box_score_fast), and the min_size edge filter. The
// unclip itself stays in unclipBox (Clipper-equivalent rect offset).
// O(pixels) fill + O(hull^2) rect search (hulls are tens of points).
std::vector<DetBox> connectedComponentsBoxes(const float* prob,
                                             int w,
                                             int h,
                                             float thresh) {
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 0);
    for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i)
        mask[i] = prob[i] > thresh ? 1 : 0;
    std::vector<int32_t> visited(static_cast<size_t>(w) * h, 0);
    std::vector<DetBox> boxes;
    std::vector<int32_t> stack;
    std::vector<std::pair<float, float>> pts;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (!mask[idx] || visited[idx]) continue;
            // Flood fill the 8-connected component, collecting every pixel
            // coordinate (the cv2 contour's convex hull equals the hull of
            // the full component for solid blobs, without depending on any
            // boundary-extraction detail).
            stack.clear();
            pts.clear();
            stack.push_back(static_cast<int32_t>(idx));
            visited[idx] = 1;
            while (!stack.empty()) {
                const int32_t cur = stack.back();
                stack.pop_back();
                const int cy = cur / w;
                const int cx = cur % w;
                pts.emplace_back(static_cast<float>(cx),
                                 static_cast<float>(cy));
                constexpr int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
                constexpr int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
                for (int d = 0; d < 8; ++d) {
                    const int nx = cx + kDx[d];
                    const int ny = cy + kDy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                    const size_t nIdx = static_cast<size_t>(ny) * w + nx;
                    if (mask[nIdx] && !visited[nIdx]) {
                        visited[nIdx] = 1;
                        stack.push_back(static_cast<int32_t>(nIdx));
                    }
                }
            }
            if (pts.size() < 3) continue;  // noise specks
            const auto hull = convexHull(pts);
            if (hull.size() < 2) continue;
            // Minimum-area rectangle by brute force over hull edge
            // directions: exact cv2.minAreaRect semantics without relying
            // on rotating-calipers invariants. The hull of a text blob is
            // small, so the quadratic scan is trivially cheap.
            RotRect best;
            float bestArea = std::numeric_limits<float>::max();
            const int hn = static_cast<int>(hull.size());
            for (int e = 0; e < hn; ++e) {
                const auto& a = hull[e];
                const auto& b = hull[(e + 1) % hn];
                float dx = b.first - a.first, dy = b.second - a.second;
                const float len = std::sqrt(dx * dx + dy * dy);
                if (len < 1e-6f) continue;
                const float ux = dx / len, uy = dy / len;
                const float vx = -uy, vy = ux;
                float minU = 0, maxU = 0, minV = 0, maxV = 0;
                bool first = true;
                for (const auto& p : hull) {
                    const float pu = p.first * ux + p.second * uy;
                    const float pv = p.first * vx + p.second * vy;
                    if (first) {
                        minU = maxU = pu;
                        minV = maxV = pv;
                        first = false;
                    } else {
                        minU = std::min(minU, pu);
                        maxU = std::max(maxU, pu);
                        minV = std::min(minV, pv);
                        maxV = std::max(maxV, pv);
                    }
                }
                const float area = (maxU - minU) * (maxV - minV);
                if (area < bestArea) {
                    bestArea = area;
                    best.ux = ux;
                    best.uy = uy;
                    best.vx = vx;
                    best.vy = vy;
                    best.halfW = (maxU - minU) / 2;
                    best.halfH = (maxV - minV) / 2;
                    const float cu = (minU + maxU) / 2;
                    const float cv = (minV + maxV) / 2;
                    best.cx = cu * ux + cv * vx;
                    best.cy = cu * uy + cv * vy;
                }
            }
            // Official min_size filter: both rect edges must reach 3 px.
            if (best.halfW * 2 < 3 || best.halfH * 2 < 3) continue;
            DetBox box;
            rectToQuad(best, box.quad);
            // box_score_fast: mean probability over the raw rect region
            // (pixel centers inside the rotated rect), applied BEFORE the
            // unclip expands the box.
            float probSum = 0.f;
            int64_t probCount = 0;
            const int bx0 = std::max(
                    0, static_cast<int>(std::floor(
                               best.cx - (std::abs(best.ux) * best.halfW +
                                          std::abs(best.vx) * best.halfH))));
            const int bx1 = std::min(
                    w - 1,
                    static_cast<int>(std::ceil(
                            best.cx + (std::abs(best.ux) * best.halfW +
                                       std::abs(best.vx) * best.halfH))));
            const int by0 = std::max(
                    0, static_cast<int>(std::floor(
                               best.cy - (std::abs(best.uy) * best.halfW +
                                          std::abs(best.vy) * best.halfH))));
            const int by1 = std::min(
                    h - 1,
                    static_cast<int>(std::ceil(
                            best.cy + (std::abs(best.uy) * best.halfW +
                                       std::abs(best.vy) * best.halfH))));
            for (int py = by0; py <= by1; ++py) {
                for (int px = bx0; px <= bx1; ++px) {
                    const float rx =
                            (px - best.cx) * best.ux + (py - best.cy) * best.uy;
                    const float ry =
                            (px - best.cx) * best.vx + (py - best.cy) * best.vy;
                    if (std::abs(rx) <= best.halfW &&
                        std::abs(ry) <= best.halfH) {
                        probSum += prob[static_cast<size_t>(py) * w + px];
                        ++probCount;
                    }
                }
            }
            if (probCount == 0) continue;
            box.score = probSum / static_cast<float>(probCount);
            // Official DB component gate (det_db_box_thresh).
            if (box.score < kComponentScoreThresh) continue;
            boxes.push_back(std::move(box));
        }
    }
    return boxes;
}

// DB unclip on the minimum-area rectangle: the Vatti offset from the
// paper reduces to a uniform outward expansion of area * unclip_ratio /
// perimeter for rectangles, applied along both rect axes.
DetBox unclipBox(const DetBox& box, float ratio) {
    const float width = std::sqrt((box.quad[1][0] - box.quad[0][0]) *
                                          (box.quad[1][0] - box.quad[0][0]) +
                                  (box.quad[1][1] - box.quad[0][1]) *
                                          (box.quad[1][1] - box.quad[0][1]));
    const float height = std::sqrt((box.quad[3][0] - box.quad[0][0]) *
                                           (box.quad[3][0] - box.quad[0][0]) +
                                   (box.quad[3][1] - box.quad[0][1]) *
                                           (box.quad[3][1] - box.quad[0][1]));
    const float area = width * height;
    const float perimeter = 2.0f * (width + height);
    DetBox out = box;
    if (perimeter <= 0) return out;
    const float offset = area * ratio / perimeter;
    const float cx = (box.quad[0][0] + box.quad[2][0]) / 2;
    const float cy = (box.quad[0][1] + box.quad[2][1]) / 2;
    // Rebuild the rect axes from the ordered corners, grow both halves by
    // the offset, and re-emit the corner order.
    RotRect r;
    r.cx = cx;
    r.cy = cy;
    const float wux = box.quad[1][0] - box.quad[0][0];
    const float wuy = box.quad[1][1] - box.quad[0][1];
    const float wLen = std::sqrt(wux * wux + wuy * wuy);
    if (wLen < 1e-6f) return out;
    r.ux = wux / wLen;
    r.uy = wuy / wLen;
    r.vx = -r.uy;
    r.vy = r.ux;
    r.halfW = width / 2 + offset;
    r.halfH = height / 2 + offset;
    rectToQuad(r, out.quad);
    return out;
}

// Perspective crop of an ordered quad out of an RGB buffer (the official
// get_rotate_crop_image): the quad is warped onto an upright rectangle
// whose width is the longer text edge. Bilinear sampling keeps the crop
// dependency-free; the rec head is robust to the cubic-vs-linear
// interpolation delta.
void cropQuadRgb(const std::vector<uint8_t>& rgb,
                 int imgW,
                 int imgH,
                 const float quad[4][2],
                 int cropW,
                 int cropH,
                 std::vector<uint8_t>& dst) {
    // Solve the 3x3 homography mapping (0,0)(w,0)(w,h)(0,h) -> quad by
    // the standard 8-unknown linear system through adjugate inversion.
    const float src[4][2] = {
            {0, 0},
            {static_cast<float>(cropW), 0},
            {static_cast<float>(cropW), static_cast<float>(cropH)},
            {0, static_cast<float>(cropH)}};
    double A[8][9] = {};
    for (int i = 0; i < 4; ++i) {
        const double x = src[i][0], y = src[i][1];
        const double X = quad[i][0], Y = quad[i][1];
        double* r1 = A[i * 2];
        double* r2 = A[i * 2 + 1];
        r1[0] = x;
        r1[1] = y;
        r1[2] = 1;
        r1[6] = -X * x;
        r1[7] = -X * y;
        r1[8] = X;
        r2[3] = x;
        r2[4] = y;
        r2[5] = 1;
        r2[6] = -Y * x;
        r2[7] = -Y * y;
        r2[8] = Y;
    }
    // Gauss elimination with partial pivoting.
    // Size the destination up front: a singular quad (degenerate
    // component) must yield a zero-filled crop, never an empty buffer that
    // the downstream resize would dereference as null.
    dst.assign(static_cast<size_t>(cropW) * cropH * 3, 0);
    double hInv[9] = {};
    for (int col = 0; col < 8; ++col) {
        int piv = col;
        for (int r = col + 1; r < 8; ++r)
            if (std::abs(A[r][col]) > std::abs(A[piv][col])) piv = r;
        if (std::abs(A[piv][col]) < 1e-12) return;
        if (piv != col)
            for (int c = col; c < 9; ++c) std::swap(A[col][c], A[piv][c]);
        for (int r = 0; r < 8; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (int c = col; c < 9; ++c) A[r][c] -= f * A[col][c];
        }
    }
    for (int i = 0; i < 8; ++i) hInv[i] = A[i][8] / A[i][i];
    hInv[8] = 1.0;
    for (int y = 0; y < cropH; ++y) {
        for (int x = 0; x < cropW; ++x) {
            const double denom = hInv[6] * x + hInv[7] * y + hInv[8];
            if (std::abs(denom) < 1e-12) continue;
            const double sx = (hInv[0] * x + hInv[1] * y + hInv[2]) / denom;
            const double sy = (hInv[3] * x + hInv[4] * y + hInv[5]) / denom;
            const int x0 = static_cast<int>(std::floor(sx));
            const int y0 = static_cast<int>(std::floor(sy));
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            if (x0 < 0 || y0 < 0 || x1 >= imgW || y1 >= imgH) continue;
            const float fx = static_cast<float>(sx - x0);
            const float fy = static_cast<float>(sy - y0);
            auto sample = [&](int px, int py, int ch) {
                return static_cast<float>(
                        rgb[(static_cast<size_t>(py) * imgW + px) * 3 + ch]);
            };
            auto* out = &dst[(static_cast<size_t>(y) * cropW + x) * 3];
            for (int ch = 0; ch < 3; ++ch) {
                const float v00 = sample(x0, y0, ch);
                const float v10 = sample(x1, y0, ch);
                const float v01 = sample(x0, y1, ch);
                const float v11 = sample(x1, y1, ch);
                out[ch] = static_cast<uint8_t>(std::clamp(
                        v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                                v01 * (1 - fx) * fy + v11 * fx * fy,
                        0.0f, 255.0f));
            }
        }
    }
}

// Crop an axis-aligned rectangle out of an RGB buffer with bilinear
// sampling into (cropW, cropH).
void cropResizeRgb(const std::vector<uint8_t>& rgb,
                   int imgW,
                   int imgH,
                   float x0,
                   float y0,
                   float x1,
                   float y1,
                   int cropW,
                   int cropH,
                   std::vector<uint8_t>& dst) {
    dst.assign(static_cast<size_t>(cropW) * cropH * 3, 0);
    for (int y = 0; y < cropH; ++y) {
        const float fy = y0 + (y + 0.5f) * (y1 - y0) / cropH - 0.5f;
        const int sy0 =
                std::clamp(static_cast<int>(std::floor(fy)), 0, imgH - 1);
        const int sy1 = std::min(imgH - 1, sy0 + 1);
        const float ly = std::clamp(fy - sy0, 0.0f, 1.0f);
        for (int x = 0; x < cropW; ++x) {
            const float fx = x0 + (x + 0.5f) * (x1 - x0) / cropW - 0.5f;
            const int sx0 =
                    std::clamp(static_cast<int>(std::floor(fx)), 0, imgW - 1);
            const int sx1 = std::min(imgW - 1, sx0 + 1);
            const float lx = std::clamp(fx - sx0, 0.0f, 1.0f);
            for (int ch = 0; ch < 3; ++ch) {
                const float v00 =
                        rgb[(static_cast<size_t>(sy0) * imgW + sx0) * 3 + ch];
                const float v01 =
                        rgb[(static_cast<size_t>(sy0) * imgW + sx1) * 3 + ch];
                const float v10 =
                        rgb[(static_cast<size_t>(sy1) * imgW + sx0) * 3 + ch];
                const float v11 =
                        rgb[(static_cast<size_t>(sy1) * imgW + sx1) * 3 + ch];
                const float top = v00 + (v01 - v00) * lx;
                const float bottom = v10 + (v11 - v10) * lx;
                dst[(static_cast<size_t>(y) * cropW + x) * 3 + ch] =
                        static_cast<uint8_t>(std::clamp(
                                top + (bottom - top) * ly, 0.0f, 255.0f));
            }
        }
    }
}

// Batched-rec enablement: automatic per execution side. Sessions whose
// graphs execute on a GPU run the chunked batch — CUDA (2.6x dense-image
// e2e, fixture-exact at every measured N with per-slab bitwise
// self-consistency) and Vulkan alike: the single-backend VK route
// (AICORE_OCR_VK_SINGLE) executes the rec graph on the device, and the
// per-crop fallback there is dispatch-bound (~98 x 2.2-3.9 ms of per-op
// vkCmdSubmit overhead; the probe measured 14.1 ms for a 40-row direct-conv
// chunk, i.e. 0.35 ms/row amortized). CPU-executing sessions keep per-crop
// N=1 forwards (the blocked island wins there, and CPU batch measured
// slower).
bool ocrBatchedRecEnabled(ggml_backend_t gpuBackend, bool cpuExecution) {
    // A session whose graphs execute on the CPU (plain CPU session, or a
    // Vulkan session with the sched_gpu_tagging gate closed and the
    // single-backend route off) keeps the per-crop path: the batched
    // graph's strided mul_mat is slower on the CPU, and a gated-off VK
    // batch would execute on the CPU secondary anyway.
    if (cpuExecution) return false;
    if (!gpuBackend || ggml_common::is_cpu_backend(gpuBackend)) return false;
    return true;
}

// Execute one graph over a freshly packed input, reusing the cached build
// for a stable input shape. The build (300+ op walk, folded constants,
// allocation) costs far more than a steady-state compute and dominated real
// images with dozens of rec crops, so the det graph is cached per geometry
// and the rec graph (fixed 48x320) exactly once per context.
// Backend routing: CPU keeps the plain gallocr path (verified bit-exact);
// GPU runs through ggml_backend_sched with the weights realized after the
// build (the measured-parity order; see the context_create note for why the
// realize-before-build alternative was rejected).
// CUDA and Vulkan OCR graphs are standard-op graphs with no CPU-only island.
// Executing them through one device gallocr avoids scheduler split/capture
// overhead. Vulkan additionally needs this path because a mixed CPU/Vulkan
// graph can hand a host-buffered view to a Vulkan subbuffer. Keep the
// Vulkan-only build flag separate: its direct convolution route excludes the
// detector geometry that faults in the NVIDIA Vulkan driver.
bool isVulkanBackend(ggml_backend_t backend) {
    if (!backend || ggml_common::is_cpu_backend(backend)) return false;
    const char* name = ggml_backend_name(backend);
    return name && std::string(name).find("Vulkan") != std::string::npos;
}

bool ocrSingleBackendEnabled(ggml_backend_t backend) {
    if (!backend || ggml_common::is_cpu_backend(backend)) return false;
    const char* name = ggml_backend_name(backend);
    if (!name) return false;
    const std::string backendName(name);
    return backendName.find("CUDA") != std::string::npos ||
           backendName.find("Vulkan") != std::string::npos;
}

bool initializeSingleBackendViews(ggml_cgraph* graph, std::string& error) {
    std::vector<ggml_tensor*> tensors;
    std::unordered_set<ggml_tensor*> seen;
    std::vector<ggml_tensor*> pending;
    const auto push = [&](ggml_tensor* tensor) {
        if (tensor && seen.insert(tensor).second) pending.push_back(tensor);
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        push(ggml_graph_node(graph, i));
    while (!pending.empty()) {
        ggml_tensor* tensor = pending.back();
        pending.pop_back();
        tensors.push_back(tensor);
        push(tensor->view_src);
        for (int i = 0; i < GGML_MAX_SRC; ++i) push(tensor->src[i]);
    }
    // gallocr allocates regular graph tensors but can encounter chained views
    // before their base view has been initialized. Rebind after allocation in
    // dependency order so every Vulkan descriptor derives from device storage.
    for (size_t pass = 0; pass < tensors.size(); ++pass) {
        bool advanced = false;
        for (ggml_tensor* tensor : tensors) {
            if (!tensor->view_src || tensor->buffer ||
                !tensor->view_src->buffer)
                continue;
            if (ggml_backend_view_init(tensor) != GGML_STATUS_SUCCESS) {
                error = "ocr: backend view initialization failed";
                return false;
            }
            advanced = true;
        }
        if (!advanced) break;
    }
    for (ggml_tensor* tensor : tensors) {
        if (tensor->view_src && !tensor->buffer) {
            error = "ocr: graph contains an unbound device view";
            return false;
        }
    }
    return true;
}

bool runGraphCached(aicore_ocr_ctx* session,
                    aicore::ocr::OcrWeights& weights,
                    OcrGraph& cached,
                    ggml_backend_t gpuBackend,
                    ggml_backend_t cpuBackend,
                    int64_t c,
                    int64_t h,
                    int64_t w,
                    int64_t n,
                    const float* nchwData,
                    size_t nchwFloats,
                    std::vector<float>& output,
                    int64_t& outC,
                    int64_t& outH,
                    int64_t& outW,
                    int64_t& outN,
                    double& inferenceMs,
                    double& inputUploadMs,
                    double& outputReadbackMs,
                    std::string& error) {
    ggml_backend_t backend = gpuBackend ? gpuBackend : cpuBackend;
    if (!backend) {
        error = "ocr: no ggml backend available";
        return false;
    }
    const bool isCpu = ggml_common::is_cpu_backend(backend);
    // CUDA/Vulkan execute their standard-op OCR graphs on one device. The
    // Vulkan build option remains narrower because only its conv route needs
    // the detector direct-convolution exclusion.
    const bool singleBackend = ocrSingleBackendEnabled(backend);
    const bool vkSingle = isVulkanBackend(backend);
    const bool cpuExecution =
            isCpu || (!weights.sched_gpu_tagging && !singleBackend);
    if (!cached.matches(c, h, w) || cached.inputN != n) {
        cached.clear();
        // Fresh fold pool per build: every build emits new constants and
        // the pool would exhaust; the previous graph is discarded above.
        weights.resetFoldPool();
        const size_t tensorSize = ggml_tensor_overhead() * 500000;
        cached.ctx = ggml_init({tensorSize * 2, nullptr,
                                /*no_alloc=*/true});
        if (!cached.ctx) {
            error = "ocr: ggml_init failed";
            return false;
        }
        cached.input =
                ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32, w, h, c, n);
        // Marked as the user input: the scheduler assigns it to the CPU
        // side and copies it before the first split; the gallocr path
        // simply allocates it like any other leaf.
        ggml_set_input(cached.input);
        aicore::ocr::OcrBuildOptions buildOptions;
        // Judge by where the graph EXECUTES, not the session backend: a
        // Vulkan session with the sched_gpu_tagging gate closed runs every
        // node on the CPU secondary, and the CPU conv paths (winograd /
        // island / batched-GEMM) are ~2x faster there than the direct-op
        // CPU kernel.
        buildOptions.cpu_conv = cpuExecution;
        buildOptions.gpu_rec_direct_conv =
                !cpuExecution && weights.role == "rec";
        buildOptions.vk_single = vkSingle;
        // Blocked-island routing: CPU-executing graphs only, and only when
        // the host advertises the AVX-512 fast path (the island has no AVX2
        // kernel, mirroring the facedetect island gate). Fixture-proven for
        // all three graphs (rec 3e-4 / det 3.6e-3 / layout 6.3e-3 max|diff|
        // with the per-input-var routing; stock: 6.5e-3 / 3.6e-3 / 1.2e-2).
        buildOptions.blocked_island =
                cpuExecution && aicore::blk::blocked_available();
        cached.output = aicore::ocr::buildGraph(cached.ctx, weights,
                                                cached.input, &error, nullptr,
                                                nullptr, &buildOptions);
        if (!cached.output) {
            cached.clear();
            return false;
        }
        if (cpuExecution) {
            // Plain CPU execution: host buffers only (see the runGraph
            // counterpart for why a gated-off Vulkan session must NOT use
            // the mixed sched allocation).
            cached.alloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(cpuBackend));
        } else {
            // Realize the host pools (zero-copy views) and the device
            // weights once per weights object, then cover the folded
            // constants emitted by this build; the scheduler routes the
            // small per-compute host->device copies (weights + constants).
            if (!weights.realizeHostBuffers()) {
                error = "ocr: host buffer realization failed";
                cached.clear();
                return false;
            }
            weights.tagHostLeafs();
            if (!weights.realizeToDevice(gpuBackend, &error)) {
                error = "ocr: device weight realization failed: " + error;
                cached.clear();
                return false;
            }
            // Mirror the fresh fold pool to the device once per build —
            // without this the sched re-uploads every fold constant on
            // EVERY compute (measured ~12 ms per rec chunk, more than all
            // rec kernels combined). Must run BEFORE sched_alloc_graph so
            // the plan sees the device-resident buffers.
            if (!weights.realizeFoldPoolToDevice(gpuBackend, &error)) {
                error = "ocr: fold pool device realization failed: " + error;
                cached.clear();
                return false;
            }
            if (singleBackend) {
                // CUDA/Vulkan standard-op graphs keep weights, folded
                // constants and activations in one device allocation.
                cached.alloc = ggml_gallocr_new(
                        ggml_backend_get_default_buffer_type(gpuBackend));
            } else {
                cached.sched = ggml_common::new_gpu_sched({gpuBackend},
                                                          cpuBackend, 1u << 20);
                if (!cached.sched) {
                    error = "ocr: backend scheduler init failed";
                    cached.clear();
                    return false;
                }
            }
        }
        cached.graph = ggml_new_graph_custom(cached.ctx, 1u << 20, false);
        ggml_build_forward_expand(cached.graph, cached.output);
        if (singleBackend) {
            weights.retargetGraphToDevice(cached.graph);
        }
        const bool allocated =
                (cpuExecution || singleBackend)
                        ? ggml_gallocr_alloc_graph(cached.alloc, cached.graph)
                        : ggml_backend_sched_alloc_graph(cached.sched,
                                                         cached.graph);
        if (!allocated) {
            error = "ocr: graph allocation failed";
            cached.clear();
            return false;
        }
        if (singleBackend && !weights.materializeUnalignedStaticViews(
                                     cached.graph, gpuBackend, &error)) {
            cached.clear();
            return false;
        }
        if (singleBackend &&
            !initializeSingleBackendViews(cached.graph, error)) {
            cached.clear();
            return false;
        }
        cached.inputC = c;
        cached.inputH = h;
        cached.inputW = w;
        cached.inputN = n;
    }
    const auto inputUploadStart = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(cached.input, nchwData, 0,
                            sizeof(float) * nchwFloats);
    inputUploadMs = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - inputUploadStart)
                            .count();
    const auto start = std::chrono::steady_clock::now();
    // cpuExecution (plain CPU session, or a gated-off Vulkan session whose
    // nodes all land on the CPU secondary) runs the plain graph_compute:
    // the sched would add split/copy overhead to a single-backend graph for
    // no routing benefit.
    if (cpuExecution)
        ggml_backend_graph_compute(cpuBackend, cached.graph);
    else if (singleBackend) {
        ggml_backend_graph_compute(gpuBackend, cached.graph);
    } else
        ggml_backend_sched_graph_compute(cached.sched, cached.graph);

    const auto stop = std::chrono::steady_clock::now();
    inferenceMs =
            std::chrono::duration<double, std::milli>(stop - start).count();
    outC = cached.output->ne[2];
    outH = cached.output->ne[1];
    outW = cached.output->ne[0];
    outN = cached.output->ne[3];
    output.resize(static_cast<size_t>(ggml_nelements(cached.output)));
    const auto readbackStart = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(cached.output, output.data(), 0,
                            sizeof(float) * output.size());
    outputReadbackMs = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - readbackStart)
                               .count();
    return true;
}

// PicoDet layout graph: same scheduling as runGraph, but every fetch
// (4 cls + 4 reg heads per FPN stride) is expanded into the graph and
// read back as a flat vector.
bool runLayoutHeads(aicore::ocr::OcrWeights& weights,
                    ggml_backend_t gpuBackend,
                    ggml_backend_t cpuBackend,
                    int64_t c,
                    int64_t h,
                    int64_t w,
                    const std::vector<float>& nchw,
                    std::vector<std::vector<float>>& heads,
                    double& inferenceMs,
                    std::string& error) {
    const size_t tensorSize = ggml_tensor_overhead() * 500000;
    ggml_context* ctx = ggml_init({tensorSize * 2, nullptr,
                                   /*no_alloc=*/true});
    if (!ctx) {
        error = "ocr: ggml_init failed";
        return false;
    }
    ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w, h, c, 1);
    ggml_set_input(input);
    // Fresh fold pool per frame (see runGraph).
    weights.resetFoldPool();
    ggml_backend_t backend = gpuBackend ? gpuBackend : cpuBackend;
    const bool isCpu = ggml_common::is_cpu_backend(backend);
    // CUDA/Vulkan standard-op layout graphs follow the same one-device path
    // as det/rec; only Vulkan needs the special direct-conv routing flag.
    const bool singleBackend = ocrSingleBackendEnabled(backend);
    const bool vkSingle = isVulkanBackend(backend);
    const bool cpuExecution =
            isCpu || ((backend != cpuBackend) || !weights.sched_gpu_tagging) &&
                             !singleBackend;
    aicore::ocr::OcrBuildOptions buildOptions;
    buildOptions.cpu_conv = cpuExecution;
    buildOptions.gpu_rec_direct_conv = !cpuExecution && weights.role == "rec";
    buildOptions.vk_single = vkSingle;
    buildOptions.blocked_island =
            cpuExecution && aicore::blk::blocked_available();
    std::vector<ggml_tensor*> fetchList;
    ggml_tensor* outputTensor = aicore::ocr::buildGraph(
            ctx, weights, input, &error, nullptr, &fetchList, &buildOptions);
    if (!outputTensor || fetchList.size() != 8) {
        if (error.empty()) error = "ocr: layout graph fetch mismatch";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool ok = false;
    do {
        if (cpuExecution) {
            // Plain CPU execution (CPU session, or a gated-off Vulkan
            // session): host buffers only, no device realization and no
            // sched — the mixed sched allocation would leave activations in
            // GPU memory that the direct graph_compute below cannot read.
            alloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(cpuBackend));
        } else {
            if (!weights.realizeHostBuffers()) {
                error = "ocr: host buffer realization failed";
                break;
            }
            weights.tagHostLeafs();
            if (!weights.realizeToDevice(backend, &error)) {
                error = "ocr: device weight realization failed: " + error;
                break;
            }
            // Fold-pool device mirror: same per-compute-copy elimination as
            // runGraphCached (must precede sched_alloc_graph).
            if (!weights.realizeFoldPoolToDevice(backend, &error)) {
                error = "ocr: fold pool device realization failed: " + error;
                break;
            }
            if (singleBackend) {
                alloc = ggml_gallocr_new(
                        ggml_backend_get_default_buffer_type(backend));
            } else {
                sched = ggml_common::new_gpu_sched({backend}, cpuBackend,
                                                   1u << 20);
                if (!sched) {
                    error = "ocr: backend scheduler init failed";
                    break;
                }
            }
        }
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1u << 20, false);
        for (ggml_tensor* t : fetchList) ggml_build_forward_expand(graph, t);
        if (singleBackend) {
            weights.retargetGraphToDevice(graph);
        }
        const bool allocated =
                (cpuExecution || singleBackend)
                        ? ggml_gallocr_alloc_graph(alloc, graph)
                        : ggml_backend_sched_alloc_graph(sched, graph);
        if (!allocated) {
            error = "ocr: graph allocation failed";
            break;
        }
        if (singleBackend &&
            !weights.materializeUnalignedStaticViews(graph, backend, &error))
            break;
        if (singleBackend && !initializeSingleBackendViews(graph, error)) break;
        ggml_backend_tensor_set(input, nchw.data(), 0,
                                sizeof(float) * nchw.size());
        const auto start = std::chrono::steady_clock::now();
        if (isCpu)
            ggml_backend_graph_compute(backend, graph);
        else if (singleBackend)
            ggml_backend_graph_compute(backend, graph);
        else if (cpuExecution)
            // Gated-off Vulkan session: single-backend (CPU) graph, same
            // rationale as the runGraphCached compute branch above.
            ggml_backend_graph_compute(cpuBackend, graph);
        else
            ggml_backend_sched_graph_compute(sched, graph);
        const auto stop = std::chrono::steady_clock::now();
        inferenceMs =
                std::chrono::duration<double, std::milli>(stop - start).count();
        heads.clear();
        heads.reserve(fetchList.size());
        for (ggml_tensor* t : fetchList) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0,
                                    sizeof(float) * data.size());
            heads.push_back(std::move(data));
        }
        ok = true;
    } while (false);
    if (sched) ggml_backend_sched_free(sched);
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return ok;
}

// ---- PicoDet GFL decode + MultiClassNMS (PaddleDetection semantics) ----

constexpr int kLayoutSizeH = 800;  // official PicoDet layout input
constexpr int kLayoutSizeW = 608;
constexpr int kGflRegMax = 8;
constexpr float kLayoutScoreThresh = 0.3f;
constexpr float kLayoutNmsThresh = 0.5f;
constexpr int kLayoutKeepTopK = 100;
constexpr int kLayoutStrides[4] = {8, 16, 32, 64};

struct LayoutRegion {
    float score;
    int classId;
    float x1, y1, x2, y2;
};

// DFL integral over the (4, reg_max) logits per anchor, then the anchor
// center decode and the final clip.
void gflDecode(const std::vector<float>& clsHead,
               const std::vector<float>& regHead,
               int stride,
               std::vector<LayoutRegion>& out) {
    const int fh = (kLayoutSizeH + stride - 1) / stride;
    const int fw = (kLayoutSizeW + stride - 1) / stride;
    const int anchors = fh * fw;
    const int classes = static_cast<int>(clsHead.size()) / anchors;
    for (int a = 0; a < anchors; ++a) {
        const float* clsRow = &clsHead[static_cast<size_t>(a) * classes];
        // Sigmoid class scores.
        for (int cat = 0; cat < classes; ++cat) {
            const float score = 1.0f / (1.0f + std::exp(-clsRow[cat]));
            if (score <= kLayoutScoreThresh) continue;
            const float* regRow =
                    &regHead[static_cast<size_t>(a) * (4 * kGflRegMax)];
            float dist[4] = {0.f, 0.f, 0.f, 0.f};
            for (int edge = 0; edge < 4; ++edge) {
                float maxv = -3e38f, sum = 0.f;
                for (int bin = 0; bin < kGflRegMax; ++bin) {
                    maxv = std::max(maxv, regRow[edge * kGflRegMax + bin]);
                }
                for (int bin = 0; bin < kGflRegMax; ++bin) {
                    sum += std::exp(regRow[edge * kGflRegMax + bin] - maxv);
                }
                for (int bin = 0; bin < kGflRegMax; ++bin) {
                    const float p =
                            std::exp(regRow[edge * kGflRegMax + bin] - maxv) /
                            sum;
                    dist[edge] += p * bin;
                }
                dist[edge] *= stride;
            }
            const int ah = a / fw, aw = a % fw;
            const float cx = (aw + 0.5f) * stride;
            const float cy = (ah + 0.5f) * stride;
            LayoutRegion r;
            r.score = score;
            r.classId = cat;
            r.x1 = std::clamp(cx - dist[0], 0.0f,
                              static_cast<float>(kLayoutSizeW));
            r.y1 = std::clamp(cy - dist[1], 0.0f,
                              static_cast<float>(kLayoutSizeH));
            r.x2 = std::clamp(cx + dist[2], 0.0f,
                              static_cast<float>(kLayoutSizeW));
            r.y2 = std::clamp(cy + dist[3], 0.0f,
                              static_cast<float>(kLayoutSizeH));
            out.push_back(r);
        }
    }
}

// MultiClassNMS across all classes (one global list, per-class filtering
// happens through the score threshold in gflDecode).
std::vector<LayoutRegion> layoutNms(std::vector<LayoutRegion> regions) {
    std::sort(regions.begin(), regions.end(),
              [](const LayoutRegion& a, const LayoutRegion& b) {
                  return a.score > b.score;
              });
    std::vector<LayoutRegion> keep;
    std::vector<bool> removed(regions.size(), false);
    for (size_t i = 0; i < regions.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(regions[i]);
        if (static_cast<int>(keep.size()) >= kLayoutKeepTopK) break;
        for (size_t j = i + 1; j < regions.size(); ++j) {
            if (removed[j]) continue;
            const float ix1 = std::max(regions[i].x1, regions[j].x1);
            const float iy1 = std::max(regions[i].y1, regions[j].y1);
            const float ix2 = std::min(regions[i].x2, regions[j].x2);
            const float iy2 = std::min(regions[i].y2, regions[j].y2);
            const float inter =
                    std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
            const float areaI = (regions[i].x2 - regions[i].x1) *
                                (regions[i].y2 - regions[i].y1);
            const float areaJ = (regions[j].x2 - regions[j].x1) *
                                (regions[j].y2 - regions[j].y1);
            const float iou = inter / (areaI + areaJ - inter + 1e-9f);
            if (iou > kLayoutNmsThresh) removed[j] = true;
        }
    }
    return keep;
}

}  // namespace

struct aicore_ocr_ctx {
    aicore::ocr::OcrWeights detWeights;
    aicore::ocr::OcrWeights recWeights;
    bool recLoaded = false;
    // Optional PP-Structure layout head (PicoDet CDLA).
    aicore::ocr::OcrWeights layoutWeights;
    bool layoutLoaded = false;
    std::string language = "ch";
    float detMaxSide = kDetOfficialLimitSide;
    int detLimitType = 0;  // 0 = official "min" resize, 1 = legacy "max" cap
    float boxThresh = kDetDefaultBoxThresh;
    float boxScoreThresh = kDetDefaultBoxScoreThresh;
    float unclipRatio = kDetDefaultUnclipRatio;
    int recMaxBatch = 8;
    // Session-owned physical backends, acquired once from the process-wide
    // runtime lease registry (GPU/auto primary + CPU secondary for the
    // scheduler, or a single CPU lease) and reused by every runGraph call;
    // the weights realize against the GPU backend on first use, so its
    // lifetime must span the whole context. The lease keeps the handle
    // shared with other task sessions without handing out raw ownership.
    aicore::runtime::BackendLease primaryLease;
    aicore::runtime::BackendLease cpuLease;
    std::string resolvedDevice;
    // Det graph parameters are shape-stable only per resolution, so the
    // graph is rebuilt whenever the resized input geometry changes.
    OcrGraph det;
    int64_t detC = 0;
    int64_t detH = 0;
    int64_t detW = 0;
    OcrGraph rec;
    bool ready = false;
};

struct aicore_ocr_result {
    std::vector<aicore_ocr_box> boxes;
    std::vector<aicore_ocr_region> regions;
    std::vector<aicore_ocr_relation> relations;
    aicore_pipeline_timings timings{};
    aicore_ocr_post_timing_breakdown postTimings{};
    // Text storage backing aicore_ocr_box::text pointers. MUST be a deque:
    // the fill sites take c_str() pointers into these strings as they push,
    // and a vector's push_back reallocation would move every SSO string
    // body, dangling ALL previously handed-out box.text pointers. That
    // use-after-free read stale-but-intact heap in the common case (which
    // is why it looked benign) and flipped single-box transcriptions
    // between empty/garbled whenever a later malloc reused the freed
    // buffer — the root cause behind the "threaded CTC race" and the
    // 98<->97 single-box drift. deque never invalidates element pointers
    // on push_back.
    std::deque<std::string> textStorage;
};

extern "C" {

aicore_ocr_ctx* aicore_ocr_context_create(const aicore_ocr_options* options) {
    if (!options || options->struct_size < sizeof(aicore_ocr_options)) {
        setLastError("ocr: invalid options (struct_size)");
        return nullptr;
    }
    if (options->abi_version != AICORE_OCR_ABI_VERSION) {
        setLastError("ocr: abi version mismatch");
        return nullptr;
    }
    if (!options->det_model_path || !options->rec_model_path) {
        setLastError("ocr: det/rec model paths are required");
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_ocr_ctx();
    if (!ctx) {
        setLastError("ocr: context allocation failed");
        return nullptr;
    }
    std::string error;
    if (!ctx->detWeights.load(options->det_model_path, &error)) {
        setLastError("ocr: det model load failed: " + error);
        delete ctx;
        return nullptr;
    }
    if (!ctx->recWeights.load(options->rec_model_path, &error)) {
        setLastError("ocr: rec model load failed: " + error);
        delete ctx;
        return nullptr;
    }
    ctx->recLoaded = true;
    // Optional layout model: the PicoDet CDLA pack with the embedded
    // class table; regions stay empty when absent.
    if (options->layout_model_path) {
        if (!ctx->layoutWeights.load(options->layout_model_path, &error)) {
            setLastError("ocr: layout model load failed: " + error);
            delete ctx;
            return nullptr;
        }
        if (ctx->layoutWeights.labels.empty()) {
            setLastError("ocr: layout model has no embedded labels");
            delete ctx;
            return nullptr;
        }
        ctx->layoutLoaded = true;
    }
    // Script tag from the rec catalog filename (ppocr-rec-<lang>-*).
    const std::string recPath = options->rec_model_path;
    const auto langPos = recPath.find("ppocr-rec-");
    if (langPos != std::string::npos) {
        std::string tail = recPath.substr(langPos + std::strlen("ppocr-rec-"));
        const auto dash = tail.find('-');
        if (dash != std::string::npos) tail = tail.substr(0, dash);
        if (!tail.empty()) ctx->language = tail;
    }
    // det_limit_type 0 (default) aligns with the official PP-OCR
    // pipeline: "min"-type resize with limit_side_len 736, so small
    // images are UPSCALED to keep small text legible (this is what the
    // official recall depends on). det_max_side overrides the 736 limit
    // side; 1 restores the legacy never-upscale long-side cap.
    ctx->detLimitType = options->det_limit_type;
    ctx->detMaxSide =
            ctx->detLimitType == 0
                    ? (options->det_max_side > 0
                               ? static_cast<float>(options->det_max_side)
                               : kDetOfficialLimitSide)
                    : (options->det_max_side > 0
                               ? static_cast<float>(options->det_max_side)
                               : kDetDefaultMaxSide);
    ctx->boxThresh = options->det_box_thresh > 0 ? options->det_box_thresh
                                                 : kDetDefaultBoxThresh;
    ctx->boxScoreThresh = options->det_box_score_thresh > 0
                                  ? options->det_box_score_thresh
                                  : kDetDefaultBoxScoreThresh;
    ctx->unclipRatio = options->det_unclip_ratio > 0 ? options->det_unclip_ratio
                                                     : kDetDefaultUnclipRatio;
    ctx->recMaxBatch = options->rec_max_batch > 0 ? options->rec_max_batch : 8;
    // One backend pair for the whole context, resolved from the caller's
    // device request. load_backends_once() is mandatory in GGML_BACKEND_DL
    // builds: without it no device is registered and every init silently
    // returns nullptr. "auto" follows the platform order (CUDA -> Vulkan ->
    // CPU on Linux/Windows, Metal -> CPU on macOS) and already includes the
    // CPU fallback; an explicit unavailable request is an honest failure.
    ggml_common::load_backends_once();
    const std::string deviceRequest =
            (options->device && options->device[0]) ? options->device : "auto";
    // The OCR graphs are hundreds of small nodes; beyond a handful of CPU
    // threads the per-node pool barrier dominates (measured: rec graph
    // 94 ms at 32 threads vs 61 ms at 8). Cap the default like the
    // facedetect runtime does; an explicit options->n_threads still wins.
    int nThreads = options->n_threads;
    if (nThreads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        nThreads = static_cast<int>(std::min(hw, 8u));
    }
    std::string leaseError;
    ctx->primaryLease = aicore::runtime::acquire_backend_lease(
            deviceRequest, nThreads, &leaseError);
    if (!ctx->primaryLease) {
        setLastError("ocr: requested device unavailable (" + deviceRequest +
                     "): " + leaseError);
        delete ctx;
        return nullptr;
    }
    ctx->resolvedDevice = ctx->primaryLease.device();
    if (!ctx->primaryLease.is_cpu()) {
        // WEIGHTS tagging gate: CUDA keeps the sched GPU offload (5-8x per
        // forward); Vulkan keeps the legacy all-CPU routing for now because
        // offloaded det graphs trip an independent ggml-vulkan subbuffer
        // fault on some geometries (352x544).
        const bool isVulkan =
                !ctx->primaryLease.is_cpu() &&
                std::string(ggml_backend_name(ctx->primaryLease.handle()))
                                .find("Vulkan") != std::string::npos;
        ctx->detWeights.sched_gpu_tagging = !isVulkan;
        // Measured reversal (this session): routing rec through the sched
        // offload on Vulkan is SLOWER than the single-backend route
        // (73-77 ms vs 58.6 ms per 40-row chunk). The sched shards the
        // ~600-node rec graph into ~172 splits (small nodes below the
        // offload threshold alternate CPU/VK) and every split pays a
        // queue submit + weight-check + barrier; the plain gallocr pays
        // one dispatch per op instead. The probe numbers that suggested
        // sched was 5x faster were mislabeled: the probe's
        // init_by_name("Vulkan") falls back to init_by_type and returns
        // CUDA0, so "backend=Vulkan device=CUDA0" probes measured CUDA.
        // rec keeps the gate closed like det/layout and runs through
        // AICORE_OCR_VK_SINGLE.
        ctx->recWeights.sched_gpu_tagging = !isVulkan;
        if (ctx->layoutLoaded) {
            ctx->layoutWeights.sched_gpu_tagging = !isVulkan;
        }
        // GPU runs schedule against a CPU secondary (see runGraphCached); the
        // parallel lease keeps independent CPU graphs concurrent.
        ctx->cpuLease = aicore::runtime::acquire_parallel_backend_lease(
                "cpu", nThreads, &leaseError);
        if (!ctx->cpuLease) {
            setLastError("ocr: CPU secondary backend unavailable: " +
                         leaseError);
            delete ctx;
            return nullptr;
        }
        // NOTE: the weights are realized after each graph build (see
        // runGraphCached). Realizing before the build was measured as a net
        // regression: the scheduler assigns more nodes to the GPU, which
        // both slowed the small-input case ~5x (90 vs 19 ms det fixture)
        // and drifted the f32 det parity beyond its 0.01 gate at high
        // resolutions (0.0139 vs 0.0012 max diff) through the changed
        // op->backend mapping.
    }
    // Only the rec graph carries the embedded dictionary (the det graph is
    // a pure DBNet CNN with no vocabulary).
    if (ctx->recWeights.dictionary.empty()) {
        setLastError("ocr: rec model has no embedded dictionary");
        delete ctx;
        return nullptr;
    }
    ctx->ready = true;
    return ctx;
}

void aicore_ocr_context_destroy(aicore_ocr_ctx* ctx) {
    if (!ctx) return;
    ctx->det.clear();
    ctx->rec.clear();
    // Release the leases after the graphs; the weights' device buffer is
    // released by the OcrWeights destructor and must outlive neither.
    ctx->cpuLease.reset();
    ctx->primaryLease.reset();
    delete ctx;
}

int aicore_ocr_context_ready(const aicore_ocr_ctx* ctx) {
    return ctx && ctx->ready ? 1 : 0;
}

const char* aicore_ocr_context_device(const aicore_ocr_ctx* ctx) {
    return ctx ? ctx->resolvedDevice.c_str() : "";
}

aicore_ocr_result* aicore_ocr_run(aicore_ocr_ctx* ctx,
                                  const aicore_image_view* image) {
    if (!ctx || !ctx->ready || !image || !image->data) {
        setLastError("ocr: run called with an invalid context or image");
        return nullptr;
    }
    const auto e2eStart = std::chrono::steady_clock::now();
    const auto preprocessStart = std::chrono::steady_clock::now();

    int channels = 3;
    bool swapRB = false;
    switch (image->format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            swapRB = false;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            swapRB = false;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            swapRB = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            swapRB = true;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            swapRB = false;
            break;
        default:
            setLastError("ocr: unsupported image format");
            return nullptr;
    }
    if (image->width <= 0 || image->height <= 0) {
        setLastError("ocr: empty image");
        return nullptr;
    }

    // --- det preprocessing: limit the long side, align to 32, normalize.
    // The resize goes through the shared OpenCV-aligned bilinear
    // (common/image_utils); channel mapping and the borrowed stride are
    // handled by the Image view wrapper.
    aicore::img::Image srcImage;
    if (!aicore::img::image_from_view(*image, srcImage)) {
        setLastError("ocr: unsupported image view");
        return nullptr;
    }
    srcImage.bgr = swapRB;  // BGR/BGRA views are channel-swapped on read
    // Upstream DetResizeForTest. det_limit_type 0 (default) replicates
    // the official PP-OCR default (limit_type="min", limit_side_len=736,
    // max_side_limit=4000): the SHORT side is scaled UP to the limit when
    // it is below it, which is what the official det recall on small-text
    // images depends on. det_limit_type 1 restores the historical
    // never-upscale long-side cap. Either way each side TRUNCATES to int,
    // then rounds to the nearest multiple of 32 with a floor at 32 (the
    // network needs 32-stride alignment); the historical ceil() overshoot
    // by up to 31 px skewed the det map ratio and the box mapping.
    float scale;
    if (ctx->detLimitType == 0) {
        const int minSide = std::min(image->width, image->height);
        scale = minSide < static_cast<int>(ctx->detMaxSide)
                        ? ctx->detMaxSide / static_cast<float>(minSide)
                        : 1.0f;
    } else {
        scale = std::min(
                1.0f, ctx->detMaxSide / static_cast<float>(std::max(
                                                image->width, image->height)));
    }
    const int resizedW = static_cast<int>(image->width * scale);
    const int resizedH = static_cast<int>(image->height * scale);
    // Python round() is round-half-to-even; replicate it so pinned inputs
    // land on the exact upstream geometry.
    const auto roundHalfEven = [](double v) {
        const double f = std::floor(v);
        const double d = v - f;
        if (d > 0.5) return f + 1.0;
        if (d < 0.5) return f;
        return std::fmod(f, 2.0) != 0.0 ? f + 1.0 : f;
    };
    int detW = std::max(static_cast<int>(roundHalfEven(resizedW / 32.0)) * 32,
                        kDetAlign);
    int detH = std::max(static_cast<int>(roundHalfEven(resizedH / 32.0)) * 32,
                        kDetAlign);
    if (ctx->detLimitType == 0) {
        // Official max_side_limit (4000): if either rounded side exceeds
        // it, rescale both sides to fit before the 32 alignment stands.
        const int longSide = std::max(detW, detH);
        if (longSide > static_cast<int>(kDetOfficialMaxSideLimit)) {
            const float down =
                    kDetOfficialMaxSideLimit / static_cast<float>(longSide);
            detW = std::max(static_cast<int>(roundHalfEven(
                                    static_cast<double>(detW) * down / 32.0)) *
                                    32,
                            kDetAlign);
            detH = std::max(static_cast<int>(roundHalfEven(
                                    static_cast<double>(detH) * down / 32.0)) *
                                    32,
                            kDetAlign);
        }
    }
    aicore::img::Image detImage;
    detImage.width = detW;
    detImage.height = detH;
    detImage.channels = 3;
    detImage.rgb.resize(static_cast<size_t>(detW) * detH * 3);
    aicore::img::cv_resize_linear_u8(srcImage, detImage.rgb.data(), detW, detH,
                                     3);
    // The official pipeline feeds the networks BGR (cv2 order, never
    // converted); align the channel order here. Color-sensitive rec
    // features and the det map both shift without this.
    for (size_t i = 0; i + 2 < detImage.rgb.size(); i += 3)
        std::swap(detImage.rgb[i], detImage.rgb[i + 2]);
    std::vector<float> detNchw;
    packNchwImage(detImage, kImageMean[0], kImageStd[0], detW, 0.f, detNchw);
    const auto preprocessStop = std::chrono::steady_clock::now();

    // --- det inference: DB probability map (1,1,H,W). The graph time is
    // captured by runGraphCached (detInferenceMs), not a wall clock, so
    // the inference stage can stay truthful when rec/layout also run.
    // Session backend handles: GPU primary + CPU secondary, or a single
    // CPU backend when the lease resolved to CPU.
    ggml_backend_t gpuBackend =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                        ? ctx->primaryLease.handle()
                                        : ctx->cpuLease.handle();

    std::vector<float> detOutput;
    int64_t detOutC = 0, detOutH = 0, detOutW = 0, detOutN = 0;
    double detInferenceMs = 0.0;
    double detInputUploadMs = 0.0;
    double detOutputReadbackMs = 0.0;
    // rec/layout graph time is accumulated separately so the stage timing
    // stays truthful: inference_ms = det + rec + layout graph compute,
    // postprocess_ms excludes it.
    double recLayoutInferenceMs = 0.0;
    std::string error;
    if (!runGraphCached(ctx, ctx->detWeights, ctx->det, gpuBackend, cpuBackend,
                        3, detH, detW, 1, detNchw.data(),
                        static_cast<size_t>(3) * detH * detW, detOutput,
                        detOutC, detOutH, detOutW, detOutN, detInferenceMs,
                        detInputUploadMs, detOutputReadbackMs, error)) {
        setLastError(error);
        return nullptr;
    }

    // --- det post-process: threshold + components + unclip.
    const auto postStart = std::chrono::steady_clock::now();
    const auto dbBoxesStart = postStart;
    if (getenv("AICORE_OCR_DUMP_DETMAP")) {
        FILE* f = fopen("/tmp/our_detmap.f32", "wb");
        if (f) {
            fwrite(detOutput.data(), sizeof(float), detOutput.size(), f);
            fclose(f);
        }
        printf("[detmap] %zdx%zdx%zdx%zd\n", (size_t)detOutN, (size_t)detOutC,
               (size_t)detOutH, (size_t)detOutW);
        fflush(stdout);
    }
    const auto boxes = connectedComponentsBoxes(
            detOutput.data(), static_cast<int>(detOutW),
            static_cast<int>(detOutH), ctx->boxThresh);
    const float mapToImageX =
            static_cast<float>(image->width) / static_cast<float>(detOutW);
    const float mapToImageY =
            static_cast<float>(image->height) / static_cast<float>(detOutH);
    const auto dbBoxesStop = std::chrono::steady_clock::now();

    auto* result = new (std::nothrow) aicore_ocr_result();
    if (!result) {
        setLastError("ocr: result allocation failed");
        return nullptr;
    }
    result->boxes.reserve(boxes.size());

    // Packed full-resolution RGB for the perspective crops (the det map
    // is a downscaled working image; the official pipeline warps from the
    // original image).
    const auto cropResizePackStart = std::chrono::steady_clock::now();
    std::vector<uint8_t> fullRgb(static_cast<size_t>(image->width) *
                                 image->height * 3);
    for (int y = 0; y < image->height; ++y) {
        for (int x = 0; x < image->width; ++x) {
            uint8_t* dst =
                    &fullRgb[(static_cast<size_t>(y) * image->width + x) * 3];
            dst[0] = srcImage.channel(x, y, 0);
            dst[1] = srcImage.channel(x, y, 1);
            dst[2] = srcImage.channel(x, y, 2);
        }
    }
    // Two-phase batched recognition (same structure as the recognize
    // phase): A) per-box unclip/map/crop/pack in parallel, B) ONE batched
    // rec forward + per-row CTC decode.
    // Batched rec: automatic per execution side (see ocrBatchedRecEnabled).
    // The chunk geometry must be known BEFORE Phase A: batched frames pack
    // each crop straight into its chunk slot (slot index = box index).
    const bool recCpuExecution =
            !gpuBackend || (!ctx->recWeights.sched_gpu_tagging &&
                            !ocrSingleBackendEnabled(gpuBackend));
    const bool batchedRec = ocrBatchedRecEnabled(gpuBackend, recCpuExecution);
    // Batch chunking: measured single-forward scaling on the idle RTX 4090
    // is linear to N=72 (10/30/84/91/100/120 ms at N=4/16/48/56/64/72,
    // fixture-exact at every N) and collapses by N=96 (15.5 s) because the
    // depthwise im2col workspace pushes the gallocr buffer against VRAM
    // (20.2/24.5 GiB) and every compute re-warms the CUDA graphs. Chunks
    // are FIXED (tail padded) so the cache key never varies and one build
    // serves every frame; pad rows are computed at full cost, so the cap
    // trades linear-scaling gains against pad waste: for the 74-box bench
    // frame a same-load A/B measured 48 rows at 215 ms e2e vs 64 rows at
    // 257 ms (the 64-cap tail chunk runs 54/64 pad rows), so 48 stands.
    const size_t planeSz = static_cast<size_t>(3) * kRecHeight * kRecMaxWidth;
    std::vector<float> batchNchw;
    // Batched chunk: ONE quantized chunk width per frame, computed once
    // from the total box count — ceil(total/cap) chunks each carrying
    // round8(ceil(total/chunks)) rows (tail pad only in the last chunk).
    // This never computes more rows than the historical fixed-cap rule
    // (same chunk count, each chunk ≤ cap) while eliminating the full-cap
    // tail compute (a 2-row tail in a 48 build cost ~17 ms of pure pad
    // compute on the 98-box bench frame). The single build per frame keeps
    // the OcrGraph single-entry cache hit across frames with a stable box
    // count, and N stays in the bounded {8,16,...,48} set. Determinism
    // note: the historical "multi-N builds break decodes" correlation was
    // the dangling-vector UAF in the result text storage (fixed: deque
    // textStorage); mixed-N cached builds were re-verified bit-stable ×3
    // after that fix. The pad fill happens ONCE per frame: the chunk loop
    // overwrites every REAL row and pad rows are never decoded (skipped by
    // the valid flag), so re-memsetting per chunk is pure waste.
    size_t chunkN = 0;
    if (batchedRec && !boxes.empty()) {
        const int64_t chunkCap = recChunkBatchCap(gpuBackend);
        const size_t totalBoxes = boxes.size();
        const size_t chunksWanted =
                (totalBoxes + (size_t)chunkCap - 1) / (size_t)chunkCap;
        const size_t rowsPerChunk =
                (totalBoxes + chunksWanted - 1) / chunksWanted;
        chunkN = std::min<int64_t>(chunkCap,
                                   (int64_t)((rowsPerChunk + 7) / 8) * 8);
        // One UNIQUE slot per box (slot = box index) with the pad tail
        // carried through the last chunk: chunk c then uploads the
        // contiguous row range [c*chunkN, (c+1)*chunkN) straight from this
        // buffer — no per-chunk staging copy at all.
        batchNchw.assign(planeSz * chunksWanted * chunkN, kRecPadFill);
    }
    struct RecCrop {
        std::vector<float> nchw;
        float quad[4][2];
        float detScore = 0.f;
        bool valid = false;
    };
    std::vector<RecCrop> crops(boxes.size());
    const auto cropOne = [&](size_t bi) {
        const DetBox& raw = boxes[bi];
        RecCrop& out = crops[bi];
        DetBox region = unclipBox(raw, ctx->unclipRatio);
        // Map det-map coordinates back to full-resolution image pixels.
        float quad[4][2];
        for (int i = 0; i < 4; ++i) {
            quad[i][0] = std::clamp(region.quad[i][0] * mapToImageX, 0.0f,
                                    static_cast<float>(image->width));
            quad[i][1] = std::clamp(region.quad[i][1] * mapToImageY, 0.0f,
                                    static_cast<float>(image->height));
        }
        const auto edgeLen = [](const float a[2], const float b[2]) {
            const float dx = b[0] - a[0], dy = b[1] - a[1];
            return std::sqrt(dx * dx + dy * dy);
        };
        const float cropWidth =
                std::max(edgeLen(quad[0], quad[1]), edgeLen(quad[3], quad[2]));
        const float cropHeight =
                std::max(edgeLen(quad[0], quad[3]), edgeLen(quad[1], quad[2]));
        const int cropW = std::max(1, static_cast<int>(std::lround(cropWidth)));
        const int cropH =
                std::max(1, static_cast<int>(std::lround(cropHeight)));
        if (cropW < 3 || cropH < 3) return;
        const int recW = std::clamp(
                static_cast<int>(
                        std::lround(48.0f * cropW / static_cast<float>(cropH))),
                1, kRecMaxWidth);
        std::vector<uint8_t> quadRgb;
        // Sample with quad in IMAGE pixel coordinates (the unclipped region
        // mapped out of the det map). The det-map coordinate space is 2x the
        // image under the official "min"/736 resize; sampling the full-res
        // image with det-map coordinates produces edge-clamped garbage.
        cropQuadRgb(fullRgb, image->width, image->height, quad, cropW, cropH,
                    quadRgb);
        aicore::img::Image warped;
        warped.width = cropW;
        warped.height = cropH;
        warped.channels = 3;
        warped.rgb = std::move(quadRgb);
        std::vector<uint8_t> cropRgb(static_cast<size_t>(recW) * kRecHeight *
                                     3);
        aicore::img::cv_resize_linear_u8(warped, cropRgb.data(), recW,
                                         kRecHeight, 3);
        // Official BGR channel order (see the det note above).
        for (size_t i = 0; i + 2 < cropRgb.size(); i += 3)
            std::swap(cropRgb[i], cropRgb[i + 2]);
        aicore::img::Image cropImage;
        cropImage.width = recW;
        cropImage.height = kRecHeight;
        cropImage.channels = 3;
        cropImage.rgb = std::move(cropRgb);
        if (batchedRec) {
            // Batched frames pack straight into this crop's UNIQUE slot
            // (slot = box index; the buffer spans all chunks with the pad
            // tail). No per-crop vector and no second copy in the chunk
            // loop; distinct slots per worker keep parallel Phase A
            // race-free (shared per-chunk slots would make the write order
            // between same-slot boxes of different chunks depend on thread
            // scheduling).
            packNchwImageTo(cropImage, kRecMean, kRecStd, kRecMaxWidth,
                            kRecPadFill, batchNchw.data() + bi * planeSz);
        } else {
            std::vector<float> recNchw;
            packNchwImage(cropImage, kRecMean, kRecStd, kRecMaxWidth,
                          kRecPadFill, recNchw);
            out.nchw = std::move(recNchw);
        }
        std::memcpy(out.quad, quad, sizeof(out.quad));
        out.detScore = raw.score;
        out.valid = true;
    };
    {
        const int cropCount = static_cast<int>(boxes.size());
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (hw > static_cast<unsigned>(cropCount))
            hw = static_cast<unsigned>(cropCount);
        std::atomic<int> next{0};
        auto worker = [&]() {
            for (;;) {
                const int bi = next.fetch_add(1);
                if (bi >= cropCount) return;
                cropOne(static_cast<size_t>(bi));
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(hw - 1);
        for (unsigned i = 1; i < hw; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    const auto cropResizePackStop = std::chrono::steady_clock::now();
    double ctcDecodeMs = 0.0;
    double resultAssemblyMs = 0.0;
    double inputUploadMs = detInputUploadMs;
    double outputReadbackMs = detOutputReadbackMs;
    std::vector<float> recOutput;
    int64_t recOutC = 0, recOutT = 0, recOutW2 = 0, recOutN = 0;
    double recInferenceMs = 0.0;
    double recInputUploadMs = 0.0;
    double recReadbackMs = 0.0;
    for (size_t bi = 0; bi < boxes.size();) {
        const RecCrop& c = crops[bi];
        if (!batchedRec) {
            // Legacy per-crop forward: run the graph once for this crop and
            // decode from its single-row output.
            std::vector<float> oneOutput;
            int64_t oC = 0, oT = 0, oW = 0, oN = 0;
            double oneMs = 0.0;
            double oneInputUploadMs = 0.0;
            double oneReadbackMs = 0.0;
            if (!runGraphCached(
                        ctx, ctx->recWeights, ctx->rec, gpuBackend, cpuBackend,
                        3, kRecHeight, kRecMaxWidth, 1, c.nchw.data(),
                        static_cast<size_t>(3) * kRecHeight * kRecMaxWidth,
                        oneOutput, oC, oT, oW, oN, oneMs, oneInputUploadMs,
                        oneReadbackMs, error)) {
                setLastError(error);
                delete result;
                return nullptr;
            }
            recLayoutInferenceMs += oneMs;
            inputUploadMs += oneInputUploadMs;
            outputReadbackMs += oneReadbackMs;
            float meanScore0 = 0.0f;
            const auto ctcStart = std::chrono::steady_clock::now();
            const std::string text0 = aicore::ocr::ctcGreedyDecode(
                    oneOutput.data(), static_cast<int>(oT),
                    static_cast<int>(oW), ctx->recWeights.dictionary,
                    &meanScore0);
            ctcDecodeMs += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - ctcStart)
                                   .count();
            const auto assemblyStart = std::chrono::steady_clock::now();
            aicore_ocr_box box{};
            std::memcpy(box.quad, c.quad, sizeof(box.quad));
            result->textStorage.push_back(text0);
            box.text = result->textStorage.back().c_str();
            box.score = meanScore0 > 0 ? meanScore0 : c.detScore;
            std::strncpy(box.language, ctx->language.c_str(),
                         sizeof(box.language) - 1);
            result->boxes.push_back(box);
            resultAssemblyMs +=
                    std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - assemblyStart)
                            .count();
            ++bi;
            continue;
        }
        // Rows were packed straight into their chunk slots by Phase A (see
        // cropOne — slot index = box index); pad rows were pre-filled once
        // per frame and are never decoded, so there is nothing to copy.
        const size_t chunkRows = std::min(boxes.size() - bi, chunkN);
        if (!runGraphCached(
                    ctx, ctx->recWeights, ctx->rec, gpuBackend, cpuBackend, 3,
                    kRecHeight, kRecMaxWidth, static_cast<int64_t>(chunkN),
                    batchNchw.data() + bi * planeSz, chunkN * planeSz,
                    recOutput, recOutC, recOutT, recOutW2, recOutN,
                    recInferenceMs, recInputUploadMs, recReadbackMs, error)) {
            setLastError(error);
            delete result;
            return nullptr;
        }
        recLayoutInferenceMs += recInferenceMs;
        inputUploadMs += recInputUploadMs;
        outputReadbackMs += recReadbackMs;
        // Per-row CTC decode in parallel: rows are independent pure
        // functions over the chunk buffer and the argmax sweep (T*V per row)
        // dominates the post stage. Each worker takes a row via the atomic
        // counter and writes ONLY that slot of the pre-sized result vectors;
        // the box fill below then runs serially so textStorage pushes stay
        // ordered (the deque-backed box.text pointers never dangle).
        std::vector<std::string> chunkTexts(chunkRows);
        std::vector<float> chunkScores(chunkRows);
        const auto ctcStart = std::chrono::steady_clock::now();
        {
            std::atomic<size_t> nextRow{0};
            // Cap the worker count: spawning hardware_concurrency threads
            // 2-3x per frame costs more in pthread_create than the extra
            // parallelism returns at 18-48 rows per chunk (~3 ms/frame at
            // hw=32; ~0.6 ms capped at 8, with the decode still >10x
            // narrower than the serial sweep).
            const unsigned workers = std::min<unsigned>(
                    {8u, std::max(1u, std::thread::hardware_concurrency()),
                     (unsigned)chunkRows});
            auto decoder = [&]() {
                for (;;) {
                    const size_t r = nextRow.fetch_add(1);
                    if (r >= chunkRows) return;
                    const RecCrop& cc = crops[bi + r];
                    if (!cc.valid) continue;
                    const float* rowBase = recOutput.data() +
                                           r * static_cast<size_t>(recOutW2) *
                                                   static_cast<size_t>(recOutT);
                    float meanScore = 0.0f;
                    chunkTexts[r] = aicore::ocr::ctcGreedyDecode(
                            rowBase, static_cast<int>(recOutT),
                            static_cast<int>(recOutW2),
                            ctx->recWeights.dictionary, &meanScore);
                    chunkScores[r] = meanScore;
                }
            };
            if (workers > 1) {
                std::vector<std::thread> pool;
                pool.reserve(workers - 1);
                for (unsigned i = 1; i < workers; ++i)
                    pool.emplace_back(decoder);
                decoder();
                for (auto& t : pool) t.join();
            } else {
                decoder();
            }
        }
        ctcDecodeMs += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - ctcStart)
                               .count();
        const auto assemblyStart = std::chrono::steady_clock::now();
        for (size_t r = 0; r < chunkRows; ++r, ++bi) {
            const RecCrop& cc = crops[bi];
            if (!cc.valid) continue;
            aicore_ocr_box box{};
            std::memcpy(box.quad, cc.quad, sizeof(box.quad));
            result->textStorage.push_back(chunkTexts[r]);
            box.text = result->textStorage.back().c_str();
            box.score = chunkScores[r] > 0 ? chunkScores[r] : cc.detScore;
            std::strncpy(box.language, ctx->language.c_str(),
                         sizeof(box.language) - 1);
            result->boxes.push_back(box);
        }
        resultAssemblyMs +=
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - assemblyStart)
                        .count();
    }

    // --- optional layout stage: PicoDet CDLA heads -> regions.
    if (ctx->layoutLoaded) {
        const auto layoutAssemblyStart = std::chrono::steady_clock::now();
        aicore::img::Image layoutImage;
        layoutImage.width = kLayoutSizeW;
        layoutImage.height = kLayoutSizeH;
        layoutImage.channels = 3;
        layoutImage.rgb.resize(static_cast<size_t>(kLayoutSizeW) *
                               kLayoutSizeH * 3);
        aicore::img::cv_resize_linear_u8(srcImage, layoutImage.rgb.data(),
                                         kLayoutSizeW, kLayoutSizeH, 3);
        std::vector<float> layoutNchw;
        packNchwImage(layoutImage, kImageMean[0], kImageStd[0], kLayoutSizeW,
                      0.f, layoutNchw);
        std::vector<std::vector<float>> heads;
        double layoutMs = 0.0;
        std::string layoutError;
        if (!runLayoutHeads(
                    ctx->layoutWeights,
                    ctx->primaryLease.is_cpu() ? nullptr
                                               : ctx->primaryLease.handle(),
                    ctx->primaryLease.is_cpu() ? ctx->primaryLease.handle()
                                               : ctx->cpuLease.handle(),
                    3, kLayoutSizeH, kLayoutSizeW, layoutNchw, heads, layoutMs,
                    layoutError)) {
            // The layout stage is an enhancement: a failure downgrades the
            // result to boxes-only (with the reason on last_error for the
            // caller's log) instead of discarding the recognition pass.
            setLastError("ocr: layout stage failed: " + layoutError);
        } else {
            recLayoutInferenceMs += layoutMs;
            std::vector<LayoutRegion> regions;
            for (int si = 0; si < 4; ++si) {
                // Fetch order: transpose_0..7 = cls(8,16,32,64) then
                // reg(8,16,32,64).
                gflDecode(heads[si], heads[4 + si], kLayoutStrides[si],
                          regions);
            }
            for (LayoutRegion& r : layoutNms(std::move(regions))) {
                aicore_ocr_region region{};
                region.class_id = r.classId;
                region.reading_order = -1;
                region.quad[0][0] = r.x1;
                region.quad[0][1] = r.y1;
                region.quad[1][0] = r.x2;
                region.quad[1][1] = r.y1;
                region.quad[2][0] = r.x2;
                region.quad[2][1] = r.y2;
                region.quad[3][0] = r.x1;
                region.quad[3][1] = r.y2;
                region.score = r.score;
                result->regions.push_back(region);
            }
            // Reading order: top-to-bottom band, left-to-right within a band
            // (the upstream PP-Structure sort, adequate for single-column
            // documents and refined per-class in a later round). The band
            // test and the comparison run in the layout input space before
            // the coordinate mapping below.
            std::vector<int> order(result->regions.size());
            for (int i = 0; i < static_cast<int>(order.size()); ++i)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                const auto& ra = result->regions[a];
                const auto& rb = result->regions[b];
                const float ay = (ra.quad[0][1] + ra.quad[3][1]) * 0.5f;
                const float by = (rb.quad[0][1] + rb.quad[3][1]) * 0.5f;
                if (std::fabs(ay - by) > 0.02f * kLayoutSizeH) return ay < by;
                return ra.quad[0][0] < rb.quad[0][0];
            });
            for (int rank = 0; rank < static_cast<int>(order.size()); ++rank)
                result->regions[order[rank]].reading_order = rank;
            // Map the region quads from the 608x800 layout input space back
            // to source image pixels (the same ratio walk the upstream
            // PP-Structure visualizer applies). Regions share the box
            // contract: image pixel coordinates.
            const float regionToImageX = static_cast<float>(image->width) /
                                         static_cast<float>(kLayoutSizeW);
            const float regionToImageY = static_cast<float>(image->height) /
                                         static_cast<float>(kLayoutSizeH);
            for (auto& region : result->regions) {
                for (int c = 0; c < 4; ++c) {
                    region.quad[c][0] =
                            std::clamp(region.quad[c][0] * regionToImageX, 0.0f,
                                       static_cast<float>(image->width));
                    region.quad[c][1] =
                            std::clamp(region.quad[c][1] * regionToImageY, 0.0f,
                                       static_cast<float>(image->height));
                }
            }
        }
        resultAssemblyMs +=
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - layoutAssemblyStart)
                        .count() -
                layoutMs;
    }
    const auto postStop = std::chrono::steady_clock::now();
    const auto e2eStop = postStop;
    // The det graph ran inside the inference window above; the rec and
    // layout graphs ran inside the post window, so their graph time moves
    // from postprocess into inference and the remainder is the true
    // post-process work (boxes, crops, decode, NMS).
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields =
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
            AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    result->timings.preprocess_ms = std::chrono::duration<double, std::milli>(
                                            preprocessStop - preprocessStart)
                                            .count();
    result->timings.inference_ms = detInferenceMs + recLayoutInferenceMs;
    double postMs =
            std::chrono::duration<double, std::milli>(postStop - postStart)
                    .count() -
            recLayoutInferenceMs;
    result->timings.postprocess_ms = postMs > 0.0 ? postMs : 0.0;
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(e2eStop - e2eStart)
                    .count();
    result->postTimings.abi_version = AICORE_OCR_POST_TIMINGS_ABI_VERSION;
    result->postTimings.valid_fields =
            AICORE_OCR_POST_TIMING_DB_BOXES |
            AICORE_OCR_POST_TIMING_CROP_RESIZE_PACK |
            AICORE_OCR_POST_TIMING_CTC_DECODE |
            AICORE_OCR_POST_TIMING_RESULT_ASSEMBLY |
            AICORE_OCR_POST_TIMING_OUTPUT_READBACK |
            AICORE_OCR_POST_TIMING_INPUT_UPLOAD |
            AICORE_OCR_POST_TIMING_DET_GRAPH_COMPUTE |
            AICORE_OCR_POST_TIMING_REC_GRAPH_COMPUTE;
    result->postTimings.db_boxes_ms = std::chrono::duration<double, std::milli>(
                                              dbBoxesStop - dbBoxesStart)
                                              .count();
    result->postTimings.crop_resize_pack_ms =
            std::chrono::duration<double, std::milli>(cropResizePackStop -
                                                      cropResizePackStart)
                    .count();
    result->postTimings.ctc_decode_ms = ctcDecodeMs;
    result->postTimings.result_assembly_ms = std::max(0.0, resultAssemblyMs);
    result->postTimings.output_readback_ms = outputReadbackMs;
    result->postTimings.input_upload_ms = inputUploadMs;
    // 将总 inference 拆回两个图，避免仅凭总数猜测应优化 det 还是 rec。
    result->postTimings.det_graph_compute_ms = detInferenceMs;
    result->postTimings.rec_graph_compute_ms = recLayoutInferenceMs;
    return result;
}

// Skip-detection path: recognize text inside caller-provided quads.
// Mirrors the rec stage of run() (perspective warp -> height-48 resize ->
// CTC) without the det pass; quads are full-image pixel coordinates.
aicore_ocr_result* aicore_ocr_recognize(aicore_ocr_ctx* ctx,
                                        const aicore_image_view* image,
                                        const float* quads,
                                        int32_t count) {
    if (!ctx || !image || !quads || count <= 0) {
        setLastError("ocr: recognize requires ctx, image and quads");
        return nullptr;
    }
    if (!ctx->ready) {
        setLastError("ocr: context not ready");
        return nullptr;
    }
    aicore::img::Image srcImage;
    int channels = 0;
    bool swapRB = false;
    switch (image->format) {
        case AICORE_IMAGE_RGB8:
            channels = 3;
            swapRB = false;
            break;
        case AICORE_IMAGE_RGBA8:
            channels = 4;
            swapRB = false;
            break;
        case AICORE_IMAGE_BGR8:
            channels = 3;
            swapRB = true;
            break;
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            swapRB = true;
            break;
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            swapRB = false;
            break;
        default:
            setLastError("ocr: unsupported image format");
            return nullptr;
    }
    if (image->width <= 0 || image->height <= 0) {
        setLastError("ocr: empty image");
        return nullptr;
    }
    if (!aicore::img::image_from_view(*image, srcImage)) {
        setLastError("ocr: unsupported image view");
        return nullptr;
    }
    srcImage.bgr = swapRB;
    // cropQuadRgb reads a packed 3-channel buffer; flatten the (possibly
    // strided, 4- or 1-channel, BGR) view into a packed RGB copy once.
    const auto cropResizePackStart = std::chrono::steady_clock::now();
    std::vector<uint8_t> flat(static_cast<size_t>(srcImage.width) *
                              srcImage.height * 3);
    for (int y = 0; y < srcImage.height; ++y) {
        for (int x = 0; x < srcImage.width; ++x) {
            uint8_t* dst =
                    &flat[(static_cast<size_t>(y) * srcImage.width + x) * 3];
            if (channels == 1) {
                const uint8_t g = srcImage.channel(x, y, 0);
                dst[0] = dst[1] = dst[2] = g;
            } else {
                dst[0] = srcImage.channel(x, y, 0);
                dst[1] = srcImage.channel(x, y, 1);
                dst[2] = srcImage.channel(x, y, 2);
            }
        }
    }
    const int imgW = srcImage.width;
    const int imgH = srcImage.height;
    auto* result = new (std::nothrow) aicore_ocr_result();
    if (!result) {
        setLastError("ocr: result allocation failed");
        return nullptr;
    }
    result->boxes.reserve(static_cast<size_t>(count));
    const auto start = std::chrono::steady_clock::now();
    double recInferenceMs = 0.0;
    std::string error;
    // Two-phase batched recognition:
    //   A) per-quad perspective crop + resize + pack — embarrassingly
    //      parallel pure functions over the packed `flat` buffer;
    //   B) ONE batched rec forward over all crops + per-row CTC decode
    //      (the buildGraph batched-attention bypass makes N>1 legal).
    struct RecCrop {
        std::vector<float> nchw;
        std::vector<float> quad;  // 8 floats, kept for the output box
        bool valid = false;
    };
    std::vector<RecCrop> crops(static_cast<size_t>(count));
    const auto cropOne = [&](int32_t q) {
        RecCrop& out = crops[static_cast<size_t>(q)];
        const float* quad = quads + static_cast<size_t>(q) * 8;
        const auto edgeLen = [](const float* a, const float* b) {
            const float dx = b[0] - a[0], dy = b[1] - a[1];
            return std::sqrt(dx * dx + dy * dy);
        };
        const float cropWidth =
                std::max(edgeLen(quad, quad + 2), edgeLen(quad + 6, quad + 4));
        const float cropHeight =
                std::max(edgeLen(quad, quad + 6), edgeLen(quad + 2, quad + 4));
        const int cropW = std::max(1, static_cast<int>(std::lround(cropWidth)));
        const int cropH =
                std::max(1, static_cast<int>(std::lround(cropHeight)));
        if (cropW < 3 || cropH < 3) return;
        const int recW = std::clamp(
                static_cast<int>(
                        std::lround(48.0f * cropW / static_cast<float>(cropH))),
                1, kRecMaxWidth);
        std::vector<uint8_t> quadRgb;
        const float(*quadPtr)[2] = reinterpret_cast<const float(*)[2]>(quad);
        cropQuadRgb(flat, imgW, imgH, quadPtr, cropW, cropH, quadRgb);
        aicore::img::Image warped;
        warped.width = cropW;
        warped.height = cropH;
        warped.channels = 3;
        warped.rgb = std::move(quadRgb);
        std::vector<uint8_t> cropRgb(static_cast<size_t>(recW) * kRecHeight *
                                     3);
        aicore::img::cv_resize_linear_u8(warped, cropRgb.data(), recW,
                                         kRecHeight, 3);
        // Official BGR channel order (see the det note above).
        for (size_t i = 0; i + 2 < cropRgb.size(); i += 3)
            std::swap(cropRgb[i], cropRgb[i + 2]);
        std::vector<float> recNchw;
        aicore::img::Image cropImage;
        cropImage.width = recW;
        cropImage.height = kRecHeight;
        cropImage.channels = 3;
        cropImage.rgb = std::move(cropRgb);
        packNchwImage(cropImage, kRecMean, kRecStd, kRecMaxWidth, kRecPadFill,
                      recNchw);
        out.nchw = std::move(recNchw);
        out.quad.assign(quad, quad + 8);
        out.valid = true;
    };
    {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 1;
        if (hw > static_cast<unsigned>(count))
            hw = static_cast<unsigned>(count);
        std::atomic<int> next{0};
        auto worker = [&]() {
            for (;;) {
                const int q = next.fetch_add(1);
                if (q >= count) return;
                cropOne(q);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(hw - 1);
        for (unsigned i = 1; i < hw; ++i) pool.emplace_back(worker);
        worker();
        for (auto& t : pool) t.join();
    }
    const auto cropResizePackStop = std::chrono::steady_clock::now();
    double ctcDecodeMs = 0.0;
    double resultAssemblyMs = 0.0;
    double inputUploadMs = 0.0;
    double outputReadbackMs = 0.0;

    // Phase B: pack crops into fixed 48-row chunk inputs (tail padded) and
    // run the rec graph once per chunk. The fixed chunk keeps the runGraph
    // cache key constant; invalid (degenerate) crops occupy their slot with
    // the pad fill and their boxes are simply not emitted below.
    ggml_backend_t gpuBackend =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    const bool recCpuExecution =
            !gpuBackend || (!ctx->recWeights.sched_gpu_tagging &&
                            !ocrSingleBackendEnabled(gpuBackend));
    const bool batchedRec = ocrBatchedRecEnabled(gpuBackend, recCpuExecution);
    const size_t planeSz = static_cast<size_t>(3) * kRecHeight * kRecMaxWidth;
    std::vector<float> batchNchw;
    std::vector<float> recOutput;
    int64_t recOutC = 0, recOutT = 0, recOutW2 = 0, recOutN = 0;
    double recInputUploadMs = 0.0;
    double recReadbackMs = 0.0;
    // ONE quantized chunk width per frame — same rule and rationale as the
    // run phase (bounded build space, no full-cap pad compute, single build
    // per frame for the cache; pad fill once per frame since pad rows are
    // never decoded).
    int64_t chunkN = 0;
    if (batchedRec && count > 0) {
        const int64_t chunkCap = recChunkBatchCap(gpuBackend);
        const int32_t totalCrops = count;
        const int32_t chunksWanted =
                (int32_t)((totalCrops + chunkCap - 1) / chunkCap);
        const int32_t rowsPerChunk =
                (totalCrops + chunksWanted - 1) / chunksWanted;
        chunkN = std::min<int64_t>(chunkCap,
                                   (int64_t)((rowsPerChunk + 7) / 8) * 8);
        batchNchw.assign(planeSz * static_cast<size_t>(chunkN), kRecPadFill);
    }
    ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                        ? ctx->primaryLease.handle()
                                        : ctx->cpuLease.handle();
    // The rec head is (vocab, T) per row: ne={vocab, T, 1, N}, so row n
    // starts at n * vocab * T and successive t steps are vocab floats.
    for (int32_t q = 0; q < count;) {
        const RecCrop& c = crops[static_cast<size_t>(q)];
        if (!batchedRec) {
            std::vector<float> oneOutput;
            int64_t oC = 0, oT = 0, oW = 0, oN = 0;
            double oneMs = 0.0;
            double oneInputUploadMs = 0.0;
            double oneReadbackMs = 0.0;
            if (!runGraphCached(
                        ctx, ctx->recWeights, ctx->rec, gpuBackend, cpuBackend,
                        3, kRecHeight, kRecMaxWidth, 1, c.nchw.data(),
                        static_cast<size_t>(3) * kRecHeight * kRecMaxWidth,
                        oneOutput, oC, oT, oW, oN, oneMs, oneInputUploadMs,
                        oneReadbackMs, error)) {
                setLastError(error);
                delete result;
                return nullptr;
            }
            recInferenceMs += oneMs;
            inputUploadMs += oneInputUploadMs;
            outputReadbackMs += oneReadbackMs;
            float meanScore0 = 0.0f;
            const auto ctcStart = std::chrono::steady_clock::now();
            const std::string text0 = aicore::ocr::ctcGreedyDecode(
                    oneOutput.data(), static_cast<int>(oT),
                    static_cast<int>(oW), ctx->recWeights.dictionary,
                    &meanScore0);
            ctcDecodeMs += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - ctcStart)
                                   .count();
            const auto assemblyStart = std::chrono::steady_clock::now();
            aicore_ocr_box box{};
            std::memcpy(box.quad, c.quad.data(), sizeof(box.quad));
            result->textStorage.push_back(text0);
            box.text = result->textStorage.back().c_str();
            box.score = meanScore0;
            std::strncpy(box.language, ctx->language.c_str(),
                         sizeof(box.language) - 1);
            result->boxes.push_back(box);
            resultAssemblyMs +=
                    std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - assemblyStart)
                            .count();
            ++q;
            continue;
        }
        // chunkN and the pad fill were computed once per frame above the
        // loop (see the note there); pad rows are never decoded.
        int32_t chunkRows = 0;
        const int32_t remaining = count - q;
        while (chunkRows < remaining && (int64_t)chunkRows < chunkN) {
            const RecCrop& cc = crops[static_cast<size_t>(q + chunkRows)];
            if (cc.valid)
                std::copy(cc.nchw.begin(), cc.nchw.end(),
                          batchNchw.begin() +
                                  static_cast<size_t>(chunkRows) * planeSz);
            ++chunkRows;
        }
        // Upload from the buffer head: this phase stages each chunk's rows
        // into the single-chunk buffer from row 0 (the copy loop above), so
        // the pointer is the buffer start — the historical global-crop
        // offset (q * planeSz) read past the allocation from the second
        // chunk on (the run phase was fixed by the unique-slot rewrite;
        // this caller was missed).
        if (!runGraphCached(ctx, ctx->recWeights, ctx->rec, gpuBackend,
                            cpuBackend, 3, kRecHeight, kRecMaxWidth, chunkN,
                            batchNchw.data(), chunkN * planeSz, recOutput,
                            recOutC, recOutT, recOutW2, recOutN, recInferenceMs,
                            recInputUploadMs, recReadbackMs, error)) {
            setLastError(error);
            delete result;
            return nullptr;
        }
        inputUploadMs += recInputUploadMs;
        outputReadbackMs += recReadbackMs;
        // Per-row CTC decode in parallel: same structure as the run phase
        // (atomic row take, exclusive slot writes, serial box fill). See the
        // run-phase note for the ownership argument.
        std::vector<std::string> chunkTexts(chunkRows);
        std::vector<float> chunkScores(chunkRows);
        const auto ctcStart = std::chrono::steady_clock::now();
        {
            std::atomic<int32_t> nextRow{0};
            // Cap the worker count: spawning hardware_concurrency threads
            // 2-3x per frame costs more in pthread_create than the extra
            // parallelism returns at 18-48 rows per chunk (~3 ms/frame at
            // hw=32; ~0.6 ms capped at 8, with the decode still >10x
            // narrower than the serial sweep).
            const unsigned workers = std::min<unsigned>(
                    {8u, std::max(1u, std::thread::hardware_concurrency()),
                     (unsigned)chunkRows});
            auto decoder = [&]() {
                for (;;) {
                    const int32_t r = nextRow.fetch_add(1);
                    if (r >= chunkRows) return;
                    const RecCrop& cc = crops[static_cast<size_t>(q + r)];
                    if (!cc.valid) continue;
                    const float* rowBase =
                            recOutput.data() +
                            static_cast<size_t>(r) *
                                    static_cast<size_t>(recOutW2) *
                                    static_cast<size_t>(recOutT);
                    float meanScore = 0.0f;
                    chunkTexts[r] = aicore::ocr::ctcGreedyDecode(
                            rowBase, static_cast<int>(recOutT),
                            static_cast<int>(recOutW2),
                            ctx->recWeights.dictionary, &meanScore);
                    chunkScores[r] = meanScore;
                }
            };
            if (workers > 1) {
                std::vector<std::thread> pool;
                pool.reserve(workers - 1);
                for (unsigned i = 1; i < workers; ++i)
                    pool.emplace_back(decoder);
                decoder();
                for (auto& t : pool) t.join();
            } else {
                decoder();
            }
        }
        ctcDecodeMs += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - ctcStart)
                               .count();
        const auto assemblyStart = std::chrono::steady_clock::now();
        for (int32_t r = 0; r < chunkRows; ++r, ++q) {
            const RecCrop& cc = crops[static_cast<size_t>(q)];
            if (!cc.valid) continue;
            aicore_ocr_box box{};
            std::memcpy(box.quad, cc.quad.data(), sizeof(box.quad));
            result->textStorage.push_back(chunkTexts[r]);
            box.text = result->textStorage.back().c_str();
            box.score = chunkScores[r];
            std::strncpy(box.language, ctx->language.c_str(),
                         sizeof(box.language) - 1);
            result->boxes.push_back(box);
        }
        resultAssemblyMs +=
                std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - assemblyStart)
                        .count();
    }

    const auto stop = std::chrono::steady_clock::now();
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields = AICORE_TIMING_INFERENCE | AICORE_TIMING_E2E;
    result->timings.inference_ms = recInferenceMs;
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
    result->postTimings.abi_version = AICORE_OCR_POST_TIMINGS_ABI_VERSION;
    result->postTimings.valid_fields = AICORE_OCR_POST_TIMING_CROP_RESIZE_PACK |
                                       AICORE_OCR_POST_TIMING_CTC_DECODE |
                                       AICORE_OCR_POST_TIMING_RESULT_ASSEMBLY |
                                       AICORE_OCR_POST_TIMING_OUTPUT_READBACK |
                                       AICORE_OCR_POST_TIMING_INPUT_UPLOAD;
    result->postTimings.crop_resize_pack_ms =
            std::chrono::duration<double, std::milli>(cropResizePackStop -
                                                      cropResizePackStart)
                    .count();
    result->postTimings.ctc_decode_ms = ctcDecodeMs;
    result->postTimings.result_assembly_ms = resultAssemblyMs;
    result->postTimings.output_readback_ms = outputReadbackMs;
    result->postTimings.input_upload_ms = inputUploadMs;
    return result;
}

int aicore_ocr_box_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->boxes.size()) : 0;
}

const aicore_ocr_box* aicore_ocr_box_at(const aicore_ocr_result* result,
                                        int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->boxes.size()))
        return nullptr;
    return &result->boxes[index];
}

int aicore_ocr_label_count(const aicore_ocr_ctx* ctx) {
    return ctx ? static_cast<int>(ctx->layoutWeights.labels.size()) : 0;
}

const char* aicore_ocr_label_at(const aicore_ocr_ctx* ctx, int index) {
    if (!ctx || index < 0 ||
        index >= static_cast<int>(ctx->layoutWeights.labels.size()))
        return nullptr;
    return ctx->layoutWeights.labels[index].c_str();
}

int aicore_ocr_region_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->regions.size()) : 0;
}

const aicore_ocr_region* aicore_ocr_region_at(const aicore_ocr_result* result,
                                              int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->regions.size()))
        return nullptr;
    return &result->regions[index];
}

int aicore_ocr_relation_count(const aicore_ocr_result* result) {
    return result ? static_cast<int>(result->relations.size()) : 0;
}

const aicore_ocr_relation* aicore_ocr_relation_at(
        const aicore_ocr_result* result, int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->relations.size()))
        return nullptr;
    return &result->relations[index];
}

void aicore_ocr_result_release(aicore_ocr_result* result) { delete result; }

aicore_pipeline_timings aicore_ocr_timings(const aicore_ocr_result* result) {
    aicore_pipeline_timings timings{};
    if (!result) return timings;
    timings = result->timings;
    return timings;
}

aicore_ocr_post_timing_breakdown aicore_ocr_post_timings(
        const aicore_ocr_result* result) {
    aicore_ocr_post_timing_breakdown timings{};
    if (!result) return timings;
    timings = result->postTimings;
    return timings;
}

const char* aicore_ocr_last_error(void) { return g_lastError.c_str(); }

void aicore_ocr_runtime_shutdown(void) {
    // Delegates to the process-wide shutdown; live contexts are untouched.
    aicore_runtime_shutdown();
}

}  // extern "C"
