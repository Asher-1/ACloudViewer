// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Whitebox test for yolo::compose_masks() mask upsampling modes. No GGUF
// assets needed: synthetic proto / mask-coefficient tensors.
//   (a) default (nearest blocks) is bit-identical to the original algorithm
//       (a verbatim copy of the old loop is kept below as the reference);
//   (b) bilinear logit interpolation follows an analytic disc boundary far
//       more closely than nearest blocks and never writes outside the box;
// The C API option plumbing lives in test_yolo_mask_upsample_capi.cpp.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "tasks/yolo/yolo_postprocess.hpp"

namespace {

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            return 1;                                                    \
        }                                                                \
    } while (0)

inline float sigmoidRef(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Verbatim copy of the pre-option compose_masks() (nearest proto-cell blocks).
std::vector<yolo::SegMask> composeMasksReference(
        const std::vector<yolo::Detection>& dets,
        const std::vector<float>& raw,
        int na,
        const yolo::ModelMeta& meta,
        const std::vector<float>& proto,
        int proto_w,
        int proto_h,
        int canvas_w,
        int canvas_h) {
    std::vector<yolo::SegMask> masks(dets.size());
    const int nm = meta.nm;
    if (nm <= 0 || (int)proto.size() != nm * proto_w * proto_h ||
        canvas_w <= 0 || canvas_h <= 0)
        return masks;

    const int mc_base = 4 * meta.reg_max + meta.nc;
    for (size_t k = 0; k < dets.size(); k++) {
        const yolo::Detection& d = dets[k];
        if (d.anchor < 0 || d.anchor >= na) continue;

        const int x1 = std::clamp((int)std::floor(d.x1), 0, canvas_w - 1);
        const int y1 = std::clamp((int)std::floor(d.y1), 0, canvas_h - 1);
        const int x2 = std::clamp((int)std::ceil(d.x2), x1 + 1, canvas_w);
        const int y2 = std::clamp((int)std::ceil(d.y2), y1 + 1, canvas_h);
        const int px1 = std::clamp(
                (int)std::ceil(d.x1 * proto_w / (float)canvas_w), 0, proto_w);
        const int px2 = std::clamp(
                (int)std::ceil(d.x2 * proto_w / (float)canvas_w), 0, proto_w);
        const int py1 = std::clamp(
                (int)std::ceil(d.y1 * proto_h / (float)canvas_h), 0, proto_h);
        const int py2 = std::clamp(
                (int)std::ceil(d.y2 * proto_h / (float)canvas_h), 0, proto_h);

        yolo::SegMask& m = masks[k];
        m.x = x1;
        m.y = y1;
        m.w = x2 - x1;
        m.h = y2 - y1;
        m.bits.assign((size_t)m.w * m.h, 0);
        if (px2 <= px1 || py2 <= py1) continue;

        float mc[64];
        if (nm > 64) {
            m.w = m.h = 0;
            continue;
        }
        for (int c = 0; c < nm; c++)
            mc[c] = raw[(size_t)(mc_base + c) * na + d.anchor];

        for (int py = py1; py < py2; py++) {
            const int by0 = std::max(
                    (int)std::ceil((float)py * canvas_h / proto_h), y1);
            const int by1 = std::min(
                    (int)std::ceil((float)(py + 1) * canvas_h / proto_h), y2);
            for (int px = px1; px < px2; px++) {
                const float* pc = proto.data() + (size_t)py * proto_w + px;
                float v = 0.0f;
                for (int c = 0; c < nm; c++)
                    v += mc[c] * pc[(size_t)c * proto_w * proto_h];
                if (sigmoidRef(v) <= 0.5f) continue;
                const int bx0 = std::max(
                        (int)std::ceil((float)px * canvas_w / proto_w), x1);
                const int bx1 = std::min(
                        (int)std::ceil((float)(px + 1) * canvas_w / proto_w),
                        x2);
                for (int y = by0; y < by1; y++) {
                    std::memset(&m.bits[(size_t)(y - y1) * m.w + (bx0 - x1)], 1,
                                (size_t)(bx1 - bx0));
                }
            }
        }
    }
    return masks;
}

yolo::ModelMeta makeMeta(int nm) {
    yolo::ModelMeta meta;
    meta.task = "segment";
    meta.nc = 2;
    meta.nm = nm;
    meta.reg_max = 16;
    return meta;
}

// raw is [no, na], mask coefficients at rows mc_base .. mc_base+nm-1.
std::vector<float> makeRaw(const yolo::ModelMeta& meta,
                           int na,
                           const std::vector<std::vector<float>>& mcPerAnchor) {
    const int mcBase = 4 * meta.reg_max + meta.nc;
    const int no = mcBase + meta.nm;
    std::vector<float> raw((size_t)no * na, 0.0f);
    for (int a = 0; a < na; a++)
        for (int c = 0; c < meta.nm; c++)
            raw[(size_t)(mcBase + c) * na + a] = mcPerAnchor[a][c];
    return raw;
}

bool sameMask(const yolo::SegMask& a, const yolo::SegMask& b) {
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h &&
           a.bits == b.bits;
}

// (a) default path == reference on random data, several geometries.
int runDefaultIdentical() {
    struct Geo {
        int pw, ph, cw, ch;
    };
    const Geo geos[] = {{160, 160, 640, 640},
                        {160, 120, 640, 480},
                        {80, 80, 640, 640},
                        {100, 96, 650, 477}};  // non-integer ratios
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const int nm = 8, na = 12;
    yolo::ModelMeta meta = makeMeta(nm);

    for (const Geo& g : geos) {
        std::vector<float> proto((size_t)nm * g.pw * g.ph);
        for (float& v : proto) v = nd(rng);
        std::vector<std::vector<float>> mcs(na, std::vector<float>(nm));
        for (auto& row : mcs)
            for (float& v : row) v = nd(rng);
        std::vector<float> raw = makeRaw(meta, na, mcs);

        std::vector<yolo::Detection> dets;
        for (int a = 0; a < na; a++) {
            yolo::Detection d{};
            const float bx = ud(rng) * g.cw, by = ud(rng) * g.ch;
            const float bw = ud(rng) * g.cw * 0.6f + 1.0f;
            const float bh = ud(rng) * g.ch * 0.6f + 1.0f;
            d.x1 = bx - bw * 0.5f;  // may exceed the canvas (edge cases)
            d.y1 = by - bh * 0.5f;
            d.x2 = bx + bw * 0.5f;
            d.y2 = by + bh * 0.5f;
            d.score = 0.9f;
            d.anchor = a;
            dets.push_back(d);
        }
        // Sub-cell box and an invalid anchor.
        yolo::Detection tiny{};
        tiny.x1 = 100.3f;
        tiny.y1 = 50.2f;
        tiny.x2 = 101.1f;
        tiny.y2 = 51.0f;
        tiny.anchor = 0;
        dets.push_back(tiny);
        yolo::Detection bad{};
        bad.x2 = bad.y2 = 10.0f;
        bad.anchor = -1;
        dets.push_back(bad);

        auto ref = composeMasksReference(dets, raw, na, meta, proto, g.pw, g.ph,
                                         g.cw, g.ch);
        auto defMasks = yolo::compose_masks(dets, raw, na, meta, proto, g.pw,
                                            g.ph, g.cw, g.ch);
        auto offMasks = yolo::compose_masks(dets, raw, na, meta, proto, g.pw,
                                            g.ph, g.cw, g.ch, false);
        CHECK(ref.size() == defMasks.size() && ref.size() == offMasks.size());
        size_t set = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            CHECK(sameMask(ref[i], defMasks[i]));
            CHECK(sameMask(ref[i], offMasks[i]));
            for (uint8_t b : ref[i].bits) set += b;
        }
        CHECK(set > 0);  // the comparison is not vacuous
    }
    std::printf("(a) default path bit-identical to reference: OK\n");
    return 0;
}

// (b) analytic disc. Logit v(X,Y) = R^2 - |(X,Y)-C|^2 is linear in the proto
// channels [1, X, Y, X^2+Y^2] evaluated at proto cell centres (canvas px).
struct DiscResult {
    double maxErrNearest = 0, maxErrBilinear = 0;
    long mismatchNearest = 0, mismatchBilinear = 0;
};

int runDisc(int pw,
            int ph,
            int cw,
            int ch,
            float cx,
            float cy,
            float R,
            float bx1,
            float by1,
            float bx2,
            float by2,
            DiscResult* out) {
    const int nm = 4;
    yolo::ModelMeta meta = makeMeta(nm);
    std::vector<float> proto((size_t)nm * pw * ph);
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            const float X = ((float)px + 0.5f) * cw / pw;
            const float Y = ((float)py + 0.5f) * ch / ph;
            const size_t i = (size_t)py * pw + px;
            proto[0 * (size_t)pw * ph + i] = 1.0f;
            proto[1 * (size_t)pw * ph + i] = X;
            proto[2 * (size_t)pw * ph + i] = Y;
            proto[3 * (size_t)pw * ph + i] = X * X + Y * Y;
        }
    std::vector<std::vector<float>> mcs = {
            {R * R - cx * cx - cy * cy, 2 * cx, 2 * cy, -1.0f}};
    std::vector<float> raw = makeRaw(meta, 1, mcs);
    yolo::Detection d{};
    d.x1 = bx1;
    d.y1 = by1;
    d.x2 = bx2;
    d.y2 = by2;
    d.score = 0.9f;
    d.anchor = 0;
    std::vector<yolo::Detection> dets{d};

    auto nearest = yolo::compose_masks(dets, raw, 1, meta, proto, pw, ph, cw,
                                       ch, false);
    auto bilin = yolo::compose_masks(dets, raw, 1, meta, proto, pw, ph, cw, ch,
                                     true);
    CHECK(nearest.size() == 1 && bilin.size() == 1);

    // Window contract: identical window, inside the canvas, bits sized w*h.
    CHECK(sameMask(nearest[0], nearest[0]));
    CHECK(bilin[0].x == nearest[0].x && bilin[0].y == nearest[0].y);
    CHECK(bilin[0].w == nearest[0].w && bilin[0].h == nearest[0].h);
    CHECK(bilin[0].x >= 0 && bilin[0].y >= 0);
    CHECK(bilin[0].x + bilin[0].w <= cw && bilin[0].y + bilin[0].h <= ch);
    CHECK(bilin[0].bits.size() == (size_t)bilin[0].w * bilin[0].h);

    auto evaluate = [&](const yolo::SegMask& m, double& maxErr, long& mism) {
        for (int j = 0; j < m.h; j++)
            for (int i = 0; i < m.w; i++) {
                const float X = (float)(m.x + i) + 0.5f;
                const float Y = (float)(m.y + j) + 0.5f;
                const bool inBox =
                        (float)(m.x + i) >= d.x1 && (float)(m.x + i) < d.x2 &&
                        (float)(m.y + j) >= d.y1 && (float)(m.y + j) < d.y2;
                const double dist = std::hypot(X - cx, Y - cy);
                const bool truth = inBox && dist < R;
                const bool got = m.bits[(size_t)j * m.w + i] != 0;
                // Never set outside the (float) box window.
                if (got && !inBox) return false;
                if (got != truth) {
                    mism++;
                    maxErr = std::max(maxErr, std::fabs(dist - R));
                }
            }
        return true;
    };
    CHECK(evaluate(nearest[0], out->maxErrNearest, out->mismatchNearest));
    CHECK(evaluate(bilin[0], out->maxErrBilinear, out->mismatchBilinear));
    return 0;
}

int runBilinearDisc() {
    // Full disc inside a generous box: 640/160 (4 px cells), R = 50 px.
    DiscResult r1;
    if (runDisc(160, 160, 640, 640, 321.3f, 287.9f, 50.0f, 240.0f, 200.0f,
                400.0f, 380.0f, &r1))
        return 1;
    std::printf(
            "(b) disc 640/160 R=50: max boundary err nearest=%.3f px "
            "(mismatch %ld), bilinear=%.3f px (mismatch %ld)\n",
            r1.maxErrNearest, r1.mismatchNearest, r1.maxErrBilinear,
            r1.mismatchBilinear);
    CHECK(r1.maxErrBilinear < 0.5);
    CHECK(r1.maxErrBilinear * 2.0 < r1.maxErrNearest);
    CHECK(r1.maxErrNearest > 1.0);

    // Non-square, coarser grid (8 px cells horizontally 640/80, 6 px
    // vertically).
    DiscResult r2;
    if (runDisc(80, 80, 640, 480, 301.7f, 233.2f, 70.0f, 200.0f, 140.0f, 420.0f,
                330.0f, &r2))
        return 1;
    std::printf(
            "(b) disc 640x480/80x80 R=70: nearest=%.3f px, bilinear=%.3f "
            "px\n",
            r2.maxErrNearest, r2.maxErrBilinear);
    CHECK(r2.maxErrBilinear < 0.5);
    CHECK(r2.maxErrBilinear < r2.maxErrNearest);

    // Box cuts the disc (tight crop) and touches the canvas border: window
    // must stay inside the canvas, nothing set outside the box.
    DiscResult r3;
    if (runDisc(160, 160, 640, 640, 20.4f, 600.6f, 60.0f, -15.0f, 520.5f,
                70.25f, 700.0f, &r3))
        return 1;
    std::printf(
            "(b) disc clipped at canvas border + box crop: nearest=%.3f "
            "px, bilinear=%.3f px\n",
            r3.maxErrNearest, r3.maxErrBilinear);
    CHECK(r3.maxErrBilinear < 0.5);

    // Degenerate sub-cell box must not crash and stays within bounds.
    DiscResult r4;
    if (runDisc(160, 160, 640, 640, 100.5f, 50.5f, 30.0f, 100.3f, 50.2f, 101.1f,
                51.0f, &r4))
        return 1;
    return 0;
}

}  // namespace

int main() {
    if (runDefaultIdentical()) return 1;
    if (runBilinearDisc()) return 1;
    std::printf("PASS test_yolo_mask_upsample\n");
    return 0;
}
