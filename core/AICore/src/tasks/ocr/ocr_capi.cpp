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
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
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
constexpr float kDetDefaultBoxThresh = 0.3f;
constexpr float kDetDefaultUnclipRatio = 1.5f;
constexpr int kRecHeight = 48;
constexpr int kRecMaxWidth = 320;
constexpr int kDetAlign = 32;

// PP-OCRv4 preprocessing constants (official DetPreProcess/RecPreProcess).
constexpr float kImageMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageStd[3] = {0.229f, 0.224f, 0.225f};
constexpr float kRecMean = 0.5f;
constexpr float kRecStd = 0.5f;

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
        inputC = inputH = inputW = 0;
    }
};

// --- Stage helpers (all pure functions over plain buffers) -----------------

// Bilinear resample of a strided source view into an RGB8 destination.
// Shared by the det downscale and the per-box rec crop.
// CHW f32 packing with the official PP-OCR normalization, over the shared
// common image view (channel extraction + BGR handling included).
void packNchwImage(const aicore::img::Image& img,
                   float mean,
                   float stdDev,
                   std::vector<float>& nchw) {
    const int w = img.width;
    const int h = img.height;
    nchw.resize(static_cast<size_t>(3) * h * w);
    const float scale = 1.0f / 255.0f;
    for (int ch = 0; ch < 3; ++ch) {
        float* plane = nchw.data() + static_cast<size_t>(ch) * h * w;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                plane[static_cast<size_t>(y) * w + x] =
                        (img.channel(x, y, ch) * scale - mean) / stdDev;
            }
        }
    }
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

// Minimum-area rectangle enclosing the hull. Rotating calipers: as the
// edge advances around the convex hull, the extreme vertices for the edge
// basis move monotonically, so maintaining four running pointers is O(h)
// total. The historical per-edge full scan was O(h^2); a dense document at
// det_max_side=960 yields components whose hull reaches tens of thousands
// of points and the quadratic scan turned single-image inference into
// minutes (the reported "Run OCR hang" at high resolutions).
RotRect minAreaRect(const std::vector<std::pair<float, float>>& hull) {
    RotRect best;
    float bestArea = std::numeric_limits<float>::max();
    const int h = static_cast<int>(hull.size());
    if (h == 0) return best;
    if (h < 3) {
        float minX = hull[0].first, maxX = minX;
        float minY = hull[0].second, maxY = minY;
        for (const auto& p : hull) {
            minX = std::min(minX, p.first);
            maxX = std::max(maxX, p.first);
            minY = std::min(minY, p.second);
            maxY = std::max(maxY, p.second);
        }
        best.cx = (minX + maxX) / 2;
        best.cy = (minY + maxY) / 2;
        best.halfW = (maxX - minX) / 2;
        best.halfH = (maxY - minY) / 2;
        return best;
    }
    auto proj = [](const std::pair<float, float>& p, float ux, float uy) {
        return p.first * ux + p.second * uy;
    };
    // Running extreme indices for the u (edge) and v (normal) axes; they
    // only advance, so the total work across all edges is O(h).
    int iuMax = 1, iuMin = 1, ivMax = 1, ivMin = 1;
    for (int i = 0; i < h; ++i) {
        const auto& a = hull[i];
        const auto& b = hull[(i + 1) % h];
        float dx = b.first - a.first, dy = b.second - a.second;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len < 1e-6f) continue;
        const float ux = dx / len, uy = dy / len;  // along the edge
        const float vx = -uy, vy = ux;             // edge normal
        const int inext = (i + 1) % h;
        while (proj(hull[(iuMax + 1) % h], ux, uy) > proj(hull[iuMax], ux, uy))
            iuMax = (iuMax + 1) % h;
        while (proj(hull[(iuMin + 1) % h], ux, uy) < proj(hull[iuMin], ux, uy))
            iuMin = (iuMin + 1) % h;
        while (proj(hull[(ivMax + 1) % h], vx, vy) > proj(hull[ivMax], vx, vy))
            ivMax = (ivMax + 1) % h;
        while (proj(hull[(ivMin + 1) % h], vx, vy) < proj(hull[ivMin], vx, vy))
            ivMin = (ivMin + 1) % h;
        (void)inext;
        const float maxU = proj(hull[iuMax], ux, uy);
        const float minU = proj(hull[iuMin], ux, uy);
        const float maxV = proj(hull[ivMax], vx, vy);
        const float minV = proj(hull[ivMin], vx, vy);
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
            best.cx = cu * ux + cv * vx;  // inverse of the orthonormal basis
            best.cy = cu * uy + cv * vy;
        }
    }
    return best;
}

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
    for (int i = 0; i < 4; ++i) {
        sums[i] = corners[i][0] + corners[i][1];
        diffs[i] = corners[i][0] - corners[i][1];
    }
    int iMinSum = 0, iMaxDiff = 0, iMaxSum = 0, iMinDiff = 0;
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

// 4-connected BFS over the binarized map. Each component's boundary
// pixels (at least one 4-neighbor binarized off) form the hull that the
// DB post-process turns into a minimum-area rectangle - the same
// geometry the official cv2.findContours + cv2.minAreaRect path emits.
// O(pixels), single pass + frontier.
std::vector<DetBox> connectedComponentsBoxes(const float* prob,
                                             int w,
                                             int h,
                                             float thresh) {
    std::vector<uint8_t> mask(static_cast<size_t>(w) * h, 0);
    for (int64_t i = 0; i < static_cast<int64_t>(w) * h; ++i)
        mask[i] = prob[i] >= thresh ? 1 : 0;
    std::vector<int32_t> visited(static_cast<size_t>(w) * h, 0);
    std::vector<DetBox> boxes;
    std::vector<int32_t> stack;
    std::vector<std::pair<float, float>> boundary;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            if (!mask[idx] || visited[idx]) continue;
            // Flood fill this component, tracking its extent, mean score,
            // and boundary pixels.
            stack.clear();
            boundary.clear();
            stack.push_back(static_cast<int32_t>(idx));
            visited[idx] = 1;
            double scoreSum = 0.0;
            int64_t count = 0;
            int minX = x, maxX = x, minY = y, maxY = y;
            while (!stack.empty()) {
                const int32_t cur = stack.back();
                stack.pop_back();
                const int cy = cur / w;
                const int cx = cur % w;
                scoreSum += prob[cur];
                ++count;
                minX = std::min(minX, cx);
                maxX = std::max(maxX, cx);
                minY = std::min(minY, cy);
                maxY = std::max(maxY, cy);
                constexpr int kDx[4] = {1, -1, 0, 0};
                constexpr int kDy[4] = {0, 0, 1, -1};
                bool isBoundary = false;
                for (int d = 0; d < 4; ++d) {
                    const int nx = cx + kDx[d];
                    const int ny = cy + kDy[d];
                    if (nx < 0 || nx >= w || ny < 0 || ny >= h) {
                        isBoundary = true;
                        continue;
                    }
                    const size_t nIdx = static_cast<size_t>(ny) * w + nx;
                    if (!mask[nIdx]) {
                        isBoundary = true;
                    } else if (!visited[nIdx]) {
                        visited[nIdx] = 1;
                        stack.push_back(static_cast<int32_t>(nIdx));
                    }
                }
                if (isBoundary)
                    boundary.emplace_back(static_cast<float>(cx),
                                          static_cast<float>(cy));
            }
            if (count < 3 || maxX - minX + 1 < 2 || maxY - minY + 1 < 2)
                continue;  // noise / single-pixel components
            const float score = static_cast<float>(scoreSum / count);
            // Official DB component gate (det_db_box_thresh): components
            // whose mean probability sits below it are dropped before the
            // unclip expands them.
            if (score < kComponentScoreThresh) continue;
            DetBox box;
            box.score = score;
            const auto hull = convexHull(std::move(boundary));
            rectToQuad(minAreaRect(hull), box.quad);
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

// Execute one graph over a freshly packed input, reusing the cached build
// for a stable input shape. The build (300+ op walk, folded constants,
// allocation) costs far more than a steady-state compute and dominated real
// images with dozens of rec crops, so the det graph is cached per geometry
// and the rec graph (fixed 48x320) exactly once per context.
//
// Backend routing: CPU keeps the plain gallocr path (verified bit-exact);
// GPU runs through ggml_backend_sched with the weights realized after the
// build (the measured-parity order; see the context_create note for why the
// realize-before-build alternative was rejected).
bool runGraphCached(aicore_ocr_ctx* session,
                    aicore::ocr::OcrWeights& weights,
                    OcrGraph& cached,
                    ggml_backend_t gpuBackend,
                    ggml_backend_t cpuBackend,
                    int64_t c,
                    int64_t h,
                    int64_t w,
                    const std::vector<float>& nchw,
                    std::vector<float>& output,
                    int64_t& outC,
                    int64_t& outH,
                    int64_t& outW,
                    double& inferenceMs,
                    std::string& error) {
    ggml_backend_t backend = gpuBackend ? gpuBackend : cpuBackend;
    if (!backend) {
        error = "ocr: no ggml backend available";
        return false;
    }
    const bool isCpu = ggml_common::is_cpu_backend(backend);
    if (!cached.matches(c, h, w)) {
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
                ggml_new_tensor_4d(cached.ctx, GGML_TYPE_F32, w, h, c, 1);
        // Marked as the user input: the scheduler assigns it to the CPU
        // side and copies it before the first split; the gallocr path
        // simply allocates it like any other leaf.
        ggml_set_input(cached.input);
        aicore::ocr::OcrBuildOptions buildOptions;
        buildOptions.cpu_conv = isCpu;
        cached.output = aicore::ocr::buildGraph(cached.ctx, weights,
                                                cached.input, &error, nullptr,
                                                nullptr, &buildOptions);
        if (!cached.output) {
            cached.clear();
            return false;
        }
        if (isCpu) {
            cached.alloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend));
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
            cached.sched = ggml_common::new_gpu_sched({gpuBackend}, cpuBackend,
                                                      1u << 20);
            if (!cached.sched) {
                error = "ocr: backend scheduler init failed";
                cached.clear();
                return false;
            }
        }
        cached.graph = ggml_new_graph_custom(cached.ctx, 1u << 20, false);
        ggml_build_forward_expand(cached.graph, cached.output);
        const bool allocated =
                isCpu ? ggml_gallocr_alloc_graph(cached.alloc, cached.graph)
                      : ggml_backend_sched_alloc_graph(cached.sched,
                                                       cached.graph);
        if (!allocated) {
            error = "ocr: graph allocation failed";
            cached.clear();
            return false;
        }
        cached.inputC = c;
        cached.inputH = h;
        cached.inputW = w;
    }
    ggml_backend_tensor_set(cached.input, nchw.data(), 0,
                            sizeof(float) * nchw.size());
    const auto start = std::chrono::steady_clock::now();
    if (isCpu)
        ggml_backend_graph_compute(backend, cached.graph);
    else
        ggml_backend_sched_graph_compute(cached.sched, cached.graph);
    const auto stop = std::chrono::steady_clock::now();
    inferenceMs =
            std::chrono::duration<double, std::milli>(stop - start).count();
    outC = cached.output->ne[2];
    outH = cached.output->ne[1];
    outW = cached.output->ne[0];
    output.resize(static_cast<size_t>(ggml_nelements(cached.output)));
    ggml_backend_tensor_get(cached.output, output.data(), 0,
                            sizeof(float) * output.size());
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
    std::vector<ggml_tensor*> fetchList;
    ggml_tensor* outputTensor = aicore::ocr::buildGraph(
            ctx, weights, input, &error, nullptr, &fetchList);
    if (!outputTensor || fetchList.size() != 8) {
        if (error.empty()) error = "ocr: layout graph fetch mismatch";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_t backend = gpuBackend ? gpuBackend : cpuBackend;
    const bool isCpu = ggml_common::is_cpu_backend(backend);
    ggml_backend_sched_t sched = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool ok = false;
    do {
        if (isCpu) {
            alloc = ggml_gallocr_new(
                    ggml_backend_get_default_buffer_type(backend));
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
            sched = ggml_common::new_gpu_sched({backend}, cpuBackend, 1u << 20);
            if (!sched) {
                error = "ocr: backend scheduler init failed";
                break;
            }
        }
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1u << 20, false);
        for (ggml_tensor* t : fetchList) ggml_build_forward_expand(graph, t);
        const bool allocated =
                isCpu ? ggml_gallocr_alloc_graph(alloc, graph)
                      : ggml_backend_sched_alloc_graph(sched, graph);
        if (!allocated) {
            error = "ocr: graph allocation failed";
            break;
        }
        ggml_backend_tensor_set(input, nchw.data(), 0,
                                sizeof(float) * nchw.size());
        const auto start = std::chrono::steady_clock::now();
        if (isCpu)
            ggml_backend_graph_compute(backend, graph);
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
    float detMaxSide = kDetDefaultMaxSide;
    float boxThresh = kDetDefaultBoxThresh;
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
    // Text storage backing aicore_ocr_box::text pointers.
    std::vector<std::string> textStorage;
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
    ctx->detMaxSide = options->det_max_side > 0
                              ? static_cast<float>(options->det_max_side)
                              : kDetDefaultMaxSide;
    ctx->boxThresh = options->det_box_thresh > 0 ? options->det_box_thresh
                                                 : kDetDefaultBoxThresh;
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
    const float scale = std::min(
            1.0f, ctx->detMaxSide / static_cast<float>(std::max(
                                            image->width, image->height)));
    int detW = static_cast<int>(std::ceil(image->width * scale / kDetAlign)) *
               kDetAlign;
    int detH = static_cast<int>(std::ceil(image->height * scale / kDetAlign)) *
               kDetAlign;
    detW = std::max(detW, kDetAlign);
    detH = std::max(detH, kDetAlign);
    aicore::img::Image detImage;
    detImage.width = detW;
    detImage.height = detH;
    detImage.channels = 3;
    detImage.rgb.resize(static_cast<size_t>(detW) * detH * 3);
    aicore::img::cv_resize_linear_u8(srcImage, detImage.rgb.data(), detW, detH,
                                     3);
    std::vector<float> detNchw;
    packNchwImage(detImage, kImageMean[0], kImageStd[0], detNchw);
    const auto preprocessStop = std::chrono::steady_clock::now();

    // --- det inference: DB probability map (1,1,H,W).
    const auto inferenceStart = std::chrono::steady_clock::now();
    // Session backend handles: GPU primary + CPU secondary, or a single
    // CPU backend when the lease resolved to CPU.
    ggml_backend_t gpuBackend =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                        ? ctx->primaryLease.handle()
                                        : ctx->cpuLease.handle();

    std::vector<float> detOutput;
    int64_t detOutC = 0, detOutH = 0, detOutW = 0;
    double detInferenceMs = 0.0;
    std::string error;
    if (!runGraphCached(ctx, ctx->detWeights, ctx->det, gpuBackend, cpuBackend,
                        3, detH, detW, detNchw, detOutput, detOutC, detOutH,
                        detOutW, detInferenceMs, error)) {
        setLastError(error);
        return nullptr;
    }
    const auto inferenceStop = std::chrono::steady_clock::now();

    // --- det post-process: threshold + components + unclip.
    const auto postStart = std::chrono::steady_clock::now();
    const auto boxes = connectedComponentsBoxes(
            detOutput.data(), static_cast<int>(detOutW),
            static_cast<int>(detOutH), ctx->boxThresh);
    const float mapToImageX =
            static_cast<float>(image->width) / static_cast<float>(detOutW);
    const float mapToImageY =
            static_cast<float>(image->height) / static_cast<float>(detOutH);

    auto* result = new (std::nothrow) aicore_ocr_result();
    if (!result) {
        setLastError("ocr: result allocation failed");
        return nullptr;
    }
    result->boxes.reserve(boxes.size());

    // Packed full-resolution RGB for the perspective crops (the det map
    // is a downscaled working image; the official pipeline warps from the
    // original image).
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
    for (const DetBox& raw : boxes) {
        DetBox region = unclipBox(raw, ctx->unclipRatio);
        // Map det-map coordinates back to full-resolution image pixels.
        float quad[4][2];
        for (int i = 0; i < 4; ++i) {
            quad[i][0] = std::clamp(region.quad[i][0] * mapToImageX, 0.0f,
                                    static_cast<float>(image->width));
            quad[i][1] = std::clamp(region.quad[i][1] * mapToImageY, 0.0f,
                                    static_cast<float>(image->height));
        }
        // Crop geometry from the ordered quad (official
        // get_rotate_crop_image): width = longer text edge, height = the
        // shorter one.
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
        // Official det_db_min_size=3: PaddleOCR's DB post-process drops
        // boxes narrower or shorter than 3 px before the rec stage (the
        // unclip of a 1-2 px component carries no recognizable text and
        // its quad is degenerate for the perspective warp).
        if (cropW < 3 || cropH < 3) continue;

        // --- rec preprocessing: perspective warp of the quad out of the
        // det input, height-48 proportional resize with right padding,
        // normalize.
        const int recW = std::clamp(
                static_cast<int>(
                        std::lround(48.0f * cropW / static_cast<float>(cropH))),
                1, kRecMaxWidth);
        std::vector<uint8_t> quadRgb;
        // The quads are full-image pixel coordinates: crop from a packed
        // full-resolution RGB buffer, not the 32-aligned det thumbnail.
        cropQuadRgb(fullRgb, image->width, image->height, region.quad, cropW,
                    cropH, quadRgb);
        aicore::img::Image warped;
        warped.width = cropW;
        warped.height = cropH;
        warped.channels = 3;
        warped.rgb = std::move(quadRgb);
        std::vector<uint8_t> cropRgb(static_cast<size_t>(recW) * kRecHeight *
                                     3);
        aicore::img::cv_resize_linear_u8(warped, cropRgb.data(), recW,
                                         kRecHeight, 3);
        std::vector<float> recNchw;
        aicore::img::Image cropImage;
        cropImage.width = recW;
        cropImage.height = kRecHeight;
        cropImage.channels = 3;
        cropImage.rgb = std::move(cropRgb);
        packNchwImage(cropImage, kRecMean, kRecStd, recNchw);
        recNchw.resize(static_cast<size_t>(3) * kRecHeight * kRecMaxWidth,
                       (0.5f - kRecMean) / kRecStd);  // pad columns

        // --- rec inference + CTC decode.
        std::vector<float> recOutput;
        int64_t recOutC = 0, recOutT = 0, recOutW2 = 0;
        double recInferenceMs = 0.0;
        if (!runGraphCached(ctx, ctx->recWeights, ctx->rec, gpuBackend,
                            cpuBackend, 3, kRecHeight, kRecMaxWidth, recNchw,
                            recOutput, recOutC, recOutT, recOutW2,
                            recInferenceMs, error)) {
            setLastError(error);
            delete result;
            return nullptr;
        }
        float meanScore = 0.0f;
        const std::string text = aicore::ocr::ctcGreedyDecode(
                recOutput.data(), static_cast<int>(recOutT),
                static_cast<int>(recOutW2), ctx->recWeights.dictionary,
                &meanScore);

        aicore_ocr_box box{};
        std::memcpy(box.quad, quad, sizeof(quad));
        result->textStorage.push_back(text);
        box.text = result->textStorage.back().c_str();
        box.score = meanScore > 0 ? meanScore : raw.score;
        std::strncpy(box.language, ctx->language.c_str(),
                     sizeof(box.language) - 1);
        result->boxes.push_back(box);
    }

    // --- optional layout stage: PicoDet CDLA heads -> regions.
    if (ctx->layoutLoaded) {
        aicore::img::Image layoutImage;
        layoutImage.width = kLayoutSizeW;
        layoutImage.height = kLayoutSizeH;
        layoutImage.channels = 3;
        layoutImage.rgb.resize(static_cast<size_t>(kLayoutSizeW) *
                               kLayoutSizeH * 3);
        aicore::img::cv_resize_linear_u8(srcImage, layoutImage.rgb.data(),
                                         kLayoutSizeW, kLayoutSizeH, 3);
        std::vector<float> layoutNchw;
        packNchwImage(layoutImage, kImageMean[0], kImageStd[0], layoutNchw);
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
            // documents and refined per-class in a later round).
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
        }
    }
    const auto postStop = std::chrono::steady_clock::now();
    const auto e2eStop = postStop;
    result->timings.preprocess_ms = std::chrono::duration<double, std::milli>(
                                            preprocessStop - preprocessStart)
                                            .count();
    result->timings.inference_ms = std::chrono::duration<double, std::milli>(
                                           inferenceStop - inferenceStart)
                                           .count();
    result->timings.postprocess_ms =
            std::chrono::duration<double, std::milli>(postStop - postStart)
                    .count();
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(e2eStop - e2eStart)
                    .count();
    // det inference dominates; rec runs inside the post-process window and
    // its per-crop time is recorded through the det+rec combined total.
    result->timings.inference_ms += detInferenceMs * 0;  // see note below
    (void)detInferenceMs;
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
    for (int32_t q = 0; q < count; ++q) {
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
        // Official det_db_min_size=3: PaddleOCR's DB post-process drops
        // boxes narrower or shorter than 3 px before the rec stage (the
        // unclip of a 1-2 px component carries no recognizable text and
        // its quad is degenerate for the perspective warp).
        if (cropW < 3 || cropH < 3) continue;
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
        std::vector<float> recNchw;
        aicore::img::Image cropImage;
        cropImage.width = recW;
        cropImage.height = kRecHeight;
        cropImage.channels = 3;
        cropImage.rgb = std::move(cropRgb);
        packNchwImage(cropImage, kRecMean, kRecStd, recNchw);
        recNchw.resize(static_cast<size_t>(3) * kRecHeight * kRecMaxWidth,
                       (0.5f - kRecMean) / kRecStd);
        // Session backend handles: GPU primary + CPU secondary, or a
        // single CPU backend when the lease resolved to CPU.
        ggml_backend_t gpuBackend = ctx->primaryLease.is_cpu()
                                            ? nullptr
                                            : ctx->primaryLease.handle();
        ggml_backend_t cpuBackend = ctx->primaryLease.is_cpu()
                                            ? ctx->primaryLease.handle()
                                            : ctx->cpuLease.handle();
        std::vector<float> recOutput;
        int64_t recOutC = 0, recOutT = 0, recOutW2 = 0;
        double singleMs = 0.0;
        if (!runGraphCached(ctx, ctx->recWeights, ctx->rec, gpuBackend,
                            cpuBackend, 3, kRecHeight, kRecMaxWidth, recNchw,
                            recOutput, recOutC, recOutT, recOutW2, singleMs,
                            error)) {
            setLastError(error);
            delete result;
            return nullptr;
        }
        recInferenceMs += singleMs;
        {
            float mn = 1e9f, mx = -1e9f;
            double sm = 0.0;
            for (float v : recOutput) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                sm += v;
            }
            std::fprintf(stderr,
                         "[diag] recOut C=%d T=%d min=%.4f max=%.4f sum=%.4e "
                         "nchw0=%.4f nchwEnd=%.4f\n",
                         (int)recOutC, (int)recOutT, mn, mx, sm,
                         recNchw.empty() ? 0.f : recNchw.front(),
                         recNchw.empty() ? 0.f : recNchw.back());
        }
        float meanScore = 0.0f;
        const std::string text = aicore::ocr::ctcGreedyDecode(
                recOutput.data(), static_cast<int>(recOutT),
                static_cast<int>(recOutW2), ctx->recWeights.dictionary,
                &meanScore);
        aicore_ocr_box box{};
        std::memcpy(box.quad, quad, sizeof(box.quad));
        result->textStorage.push_back(text);
        box.text = result->textStorage.back().c_str();
        box.score = meanScore;
        std::strncpy(box.language, ctx->language.c_str(),
                     sizeof(box.language) - 1);
        result->boxes.push_back(box);
    }

    const auto stop = std::chrono::steady_clock::now();
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields = AICORE_TIMING_INFERENCE | AICORE_TIMING_E2E;
    result->timings.inference_ms = recInferenceMs;
    result->timings.e2e_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
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

const char* aicore_ocr_last_error(void) { return g_lastError.c_str(); }

void aicore_ocr_runtime_shutdown(void) {
    // Delegates to the process-wide shutdown; live contexts are untouched.
    aicore_runtime_shutdown();
}

}  // extern "C"
