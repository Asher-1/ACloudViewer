// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "nms_3d.h"

#include <algorithm>
#include <cmath>

namespace openpcdet {

namespace {

constexpr float EPS = 1e-8f;

struct Point {
    float x, y;
    Point() : x(0), y(0) {}
    Point(float x_, float y_) : x(x_), y(y_) {}
};

inline float cross(const Point& a, const Point& b) {
    return a.x * b.y - a.y * b.x;
}
inline float cross(const Point& p1, const Point& p2, const Point& p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}
inline float minf(float a, float b) { return a < b ? a : b; }
inline float maxf(float a, float b) { return a > b ? a : b; }

inline int check_rect_cross(const Point& p1,
                            const Point& p2,
                            const Point& q1,
                            const Point& q2) {
    return minf(p1.x, p2.x) <= maxf(q1.x, q2.x) &&
           minf(q1.x, q2.x) <= maxf(p1.x, p2.x) &&
           minf(p1.y, p2.y) <= maxf(q1.y, q2.y) &&
           minf(q1.y, q2.y) <= maxf(p1.y, p2.y);
}

inline int check_in_box2d(const float* box, const Point& p) {
    constexpr float MARGIN = 1e-2f;
    const float cx = box[0], cy = box[1];
    const float ac = std::cos(-box[6]), as = std::sin(-box[6]);
    const float rx = (p.x - cx) * ac + (p.y - cy) * (-as);
    const float ry = (p.x - cx) * as + (p.y - cy) * ac;
    return std::fabs(rx) < box[3] / 2 + MARGIN &&
           std::fabs(ry) < box[4] / 2 + MARGIN;
}

inline int intersection(const Point& p1,
                        const Point& p0,
                        const Point& q1,
                        const Point& q0,
                        Point& ans) {
    if (!check_rect_cross(p0, p1, q0, q1)) return 0;
    const float s1 = cross(q0, p1, p0);
    const float s2 = cross(p1, q1, p0);
    const float s3 = cross(p0, q1, q0);
    const float s4 = cross(q1, p1, q0);
    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;
    const float s5 = cross(q1, p1, p0);
    if (std::fabs(s5 - s1) > EPS) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        const float a0 = p0.y - p1.y, b0 = p1.x - p0.x,
                    c0 = p0.x * p1.y - p1.x * p0.y;
        const float a1 = q0.y - q1.y, b1 = q1.x - q0.x,
                    c1 = q0.x * q1.y - q1.x * q0.y;
        const float D = a0 * b1 - a1 * b0;
        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }
    return 1;
}

inline void rotate_around_center(const Point& center,
                                 float ac,
                                 float as,
                                 Point& p) {
    const float nx =
            (p.x - center.x) * ac + (p.y - center.y) * (-as) + center.x;
    const float ny = (p.x - center.x) * as + (p.y - center.y) * ac + center.y;
    p.x = nx;
    p.y = ny;
}

inline int point_cmp(const Point& a, const Point& b, const Point& center) {
    return std::atan2(a.y - center.y, a.x - center.x) >
           std::atan2(b.y - center.y, b.x - center.x);
}

inline float box_overlap(const float* ba, const float* bb) {
    const float a_angle = ba[6], b_angle = bb[6];
    const float a_dx_h = ba[3] / 2, b_dx_h = bb[3] / 2;
    const float a_dy_h = ba[4] / 2, b_dy_h = bb[4] / 2;
    const float a_x1 = ba[0] - a_dx_h, a_y1 = ba[1] - a_dy_h;
    const float a_x2 = ba[0] + a_dx_h, a_y2 = ba[1] + a_dy_h;
    const float b_x1 = bb[0] - b_dx_h, b_y1 = bb[1] - b_dy_h;
    const float b_x2 = bb[0] + b_dx_h, b_y2 = bb[1] + b_dy_h;

    const Point ca(ba[0], ba[1]);
    const Point cb(bb[0], bb[1]);

    Point pa[5] = {
            {a_x1, a_y1}, {a_x2, a_y1}, {a_x2, a_y2}, {a_x1, a_y2}, {0, 0}};
    Point pb[5] = {
            {b_x1, b_y1}, {b_x2, b_y1}, {b_x2, b_y2}, {b_x1, b_y2}, {0, 0}};

    const float a_ac = std::cos(a_angle), a_as = std::sin(a_angle);
    const float b_ac = std::cos(b_angle), b_as = std::sin(b_angle);
    for (int k = 0; k < 4; k++) {
        rotate_around_center(ca, a_ac, a_as, pa[k]);
        rotate_around_center(cb, b_ac, b_as, pb[k]);
    }
    pa[4] = pa[0];
    pb[4] = pb[0];

    Point cross_pts[16];
    Point poly_center{0, 0};
    int cnt = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            Point ip;
            if (intersection(pa[i + 1], pa[i], pb[j + 1], pb[j], ip)) {
                poly_center.x += ip.x;
                poly_center.y += ip.y;
                cross_pts[cnt++] = ip;
            }
        }
    }
    for (int k = 0; k < 4; k++) {
        if (check_in_box2d(ba, pb[k])) {
            poly_center.x += pb[k].x;
            poly_center.y += pb[k].y;
            cross_pts[cnt++] = pb[k];
        }
        if (check_in_box2d(bb, pa[k])) {
            poly_center.x += pa[k].x;
            poly_center.y += pa[k].y;
            cross_pts[cnt++] = pa[k];
        }
    }
    if (cnt == 0) return 0.0f;
    poly_center.x /= cnt;
    poly_center.y /= cnt;
    // bubble sort (cnt is tiny, <= 16)
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (point_cmp(cross_pts[i], cross_pts[i + 1], poly_center)) {
                std::swap(cross_pts[i], cross_pts[i + 1]);
            }
        }
    }
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        Point a(cross_pts[k].x - cross_pts[0].x,
                cross_pts[k].y - cross_pts[0].y);
        Point b(cross_pts[k + 1].x - cross_pts[0].x,
                cross_pts[k + 1].y - cross_pts[0].y);
        area += cross(a, b);
    }
    return std::fabs(area) / 2.0f;
}

}  // namespace

float iou_bev(const float* ba, const float* bb) {
    const float sa = ba[3] * ba[4];
    const float sb = bb[3] * bb[4];
    const float so = box_overlap(ba, bb);
    return so / std::fmaxf(sa + sb - so, EPS);
}

std::vector<int32_t> nms_3d(const std::vector<float>& boxes,
                            const std::vector<float>& scores,
                            float nms_thresh,
                            int max_boxes) {
    const int32_t N = (int32_t)scores.size();
    if (N == 0) return {};
    if (boxes.size() != (size_t)N * 7) return {};

    std::vector<int32_t> order(N);
    for (int32_t i = 0; i < N; i++) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return scores[a] > scores[b]; });

    // Precompute the exact axis-aligned bounding box of each rotated box.
    // Two rotated boxes can only overlap if their AABBs overlap, so this is a
    // conservative (and exact) quick-reject test that skips the expensive
    // polygon clipping for the vast majority of pairs.
    //   half-width  = dx/2*|cos| + dy/2*|sin|
    //   half-height = dx/2*|sin| + dy/2*|cos|
    std::vector<float> aabb(N * 4);
    for (int32_t i = 0; i < N; i++) {
        const float* b = boxes.data() + (size_t)i * 7;
        const float ac = std::fabs(std::cos(b[6])),
                    as = std::fabs(std::sin(b[6]));
        const float hx = b[3] / 2 * ac + b[4] / 2 * as;
        const float hy = b[3] / 2 * as + b[4] / 2 * ac;
        aabb[(size_t)i * 4 + 0] = b[0] - hx;
        aabb[(size_t)i * 4 + 1] = b[0] + hx;
        aabb[(size_t)i * 4 + 2] = b[1] - hy;
        aabb[(size_t)i * 4 + 3] = b[1] + hy;
    }

    std::vector<bool> suppressed(N, false);
    std::vector<int32_t> kept;
    kept.reserve(std::min(N, std::max(1, max_boxes)));
    for (int32_t i = 0; i < N; i++) {
        const int32_t idx = order[i];
        if (suppressed[idx]) continue;
        kept.push_back(idx);
        if ((int)kept.size() >= max_boxes) break;
        const float* ba = boxes.data() + (size_t)idx * 7;
        const float* aa = aabb.data() + (size_t)idx * 4;
        for (int32_t j = i + 1; j < N; j++) {
            const int32_t jdx = order[j];
            if (suppressed[jdx]) continue;
            const float* ab = aabb.data() + (size_t)jdx * 4;
            if (ab[1] < aa[0] || ab[0] > aa[1] || ab[3] < aa[2] ||
                ab[2] > aa[3])
                continue;
            const float* bb = boxes.data() + (size_t)jdx * 7;
            if (iou_bev(ba, bb) > nms_thresh) suppressed[jdx] = true;
        }
    }
    return kept;
}

}  // namespace openpcdet
