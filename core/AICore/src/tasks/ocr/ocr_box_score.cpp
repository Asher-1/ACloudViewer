// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ocr_box_score.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Line-for-line port of the OpenCV 4.10 drawing.cpp paths cv2.fillPoly takes
// for a single integer contour with LINE_8 and shift 0 (clipLine,
// LineIterator, CollectPolyEdges, FillEdgeCollection). The fixed-point
// arithmetic, edge ordering and span rounding are kept verbatim: the set of
// masked pixels, not an approximation of the polygon area, is the upstream
// contract. 4.10 (the PaddleOCR 3.0.1 reference environment) differs from
// 4.7 in exactly these paths: edges built from clipped endpoints with a
// half-pixel x offset when unclipped, and LINE_8 spans floor both ends.

namespace aicore {
namespace ocr {
namespace {

constexpr int kXYShift = 16;
constexpr int64_t kXYOne = int64_t(1) << kXYShift;

struct PolyEdge {
    int y0 = 0, y1 = 0;
    int64_t x = 0, dx = 0;
    PolyEdge* next = nullptr;
};

bool clipLine(int64_t width,
              int64_t height,
              int64_t& x1,
              int64_t& y1,
              int64_t& x2,
              int64_t& y2) {
    if (width <= 0 || height <= 0) return false;
    const int64_t right = width - 1, bottom = height - 1;
    int c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    int c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
        int64_t a;
        if (c1 & 12) {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int64_t)((double)(a - y1) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if (c2 & 12) {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int64_t)((double)(a - y2) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if ((c1 & c2) == 0 && (c1 | c2) != 0) {
            if (c1) {
                a = c1 == 1 ? 0 : right;
                y1 += (int64_t)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if (c2) {
                a = c2 == 1 ? 0 : right;
                y2 += (int64_t)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }
    }
    return (c1 | c2) == 0;
}

// cv::Line -> LineIterator(img, pt1, pt2, 8, leftToRight=true).
void line8(std::uint8_t* mask, int mw, int mh, int ax, int ay, int bx, int by) {
    if ((unsigned)ax >= (unsigned)mw || (unsigned)bx >= (unsigned)mw ||
        (unsigned)ay >= (unsigned)mh || (unsigned)by >= (unsigned)mh) {
        int64_t x1 = ax, y1 = ay, x2 = bx, y2 = by;
        if (!clipLine(mw, mh, x1, y1, x2, y2)) return;
        ax = (int)x1;
        ay = (int)y1;
        bx = (int)x2;
        by = (int)y2;
    }
    int deltaX = 1, deltaY = 1;
    int dx = bx - ax;
    int dy = by - ay;
    if (dx < 0) {
        dx = -dx;
        dy = -dy;
        ax = bx;
        ay = by;
    }
    if (dy < 0) {
        dy = -dy;
        deltaY = -1;
    }
    const bool vert = dy > dx;
    if (vert) {
        std::swap(dx, dy);
        std::swap(deltaX, deltaY);
    }
    int err = dx - (dy + dy);
    const int plusDelta = dx + dx;
    const int minusDelta = -(dy + dy);
    int minusShift = deltaX, plusShift = 0, minusStep = 0, plusStep = deltaY;
    const int count = dx + 1;
    if (vert) {
        std::swap(plusStep, plusShift);
        std::swap(minusStep, minusShift);
    }
    int x = ax, y = ay;
    for (int i = 0; i < count; ++i) {
        mask[static_cast<size_t>(y) * mw + x] = 1;
        const int m = err < 0 ? -1 : 0;
        err += minusDelta + (plusDelta & m);
        x += minusShift + (plusShift & m);
        y += minusStep + (plusStep & m);
    }
}

void fillEdgeCollection(std::vector<PolyEdge>& edges,
                        std::uint8_t* mask,
                        int width,
                        int height) {
    PolyEdge tmp;
    const int total = static_cast<int>(edges.size());
    int yMax = INT_MIN, yMin = INT_MAX;
    int64_t xMax = -1, xMin = INT64_MAX;  // OpenCV's 0xFF..FF / 0x7F..FF
    if (total < 2) return;
    for (const PolyEdge& e1 : edges) {
        const int64_t x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
        yMin = std::min(yMin, e1.y0);
        yMax = std::max(yMax, e1.y1);
        xMin = std::min({xMin, e1.x, x1});
        xMax = std::max({xMax, e1.x, x1});
    }
    if (yMax < 0 || yMin >= height || xMax < 0 ||
        xMin >= (static_cast<int64_t>(width) << kXYShift))
        return;
    std::sort(edges.begin(), edges.end(),
              [](const PolyEdge& e1, const PolyEdge& e2) {
                  return e1.y0 - e2.y0 ? e1.y0 < e2.y0
                         : e1.x - e2.x ? e1.x < e2.x
                                       : e1.dx < e2.dx;
              });
    tmp.y0 = INT_MAX;
    edges.push_back(tmp);  // end marker; no reallocation below
    int i = 0;
    tmp.next = nullptr;
    PolyEdge* e = &edges[i];
    yMax = std::min(yMax, height);
    for (int y = e->y0; y < yMax; ++y) {
        PolyEdge *last, *prelast, *keepPrelast;
        int draw = 0;
        const bool clipline = y < 0;
        prelast = &tmp;
        last = tmp.next;
        while (last || e->y0 == y) {
            if (last && last->y1 == y) {
                // exclude edge if y reaches its lower point
                prelast->next = last->next;
                last = last->next;
                continue;
            }
            keepPrelast = prelast;
            if (last && (e->y0 > y || last->x < e->x)) {
                prelast = last;
                last = last->next;
            } else if (i < total) {
                prelast->next = e;
                e->next = last;
                prelast = e;
                e = &edges[++i];
            } else {
                break;
            }
            if (draw) {
                if (!clipline) {
                    // LINE_8: delta = 0, so both span ends floor.
                    int x1, x2;
                    if (keepPrelast->x > prelast->x) {
                        x1 = (int)(prelast->x >> kXYShift);
                        x2 = (int)(keepPrelast->x >> kXYShift);
                    } else {
                        x1 = (int)(keepPrelast->x >> kXYShift);
                        x2 = (int)(prelast->x >> kXYShift);
                    }
                    if (x1 < width && x2 >= 0) {
                        x1 = std::max(x1, 0);
                        x2 = std::min(x2, width - 1);
                        std::uint8_t* row =
                                mask + static_cast<size_t>(y) * width;
                        for (int x = x1; x <= x2; ++x) row[x] = 1;
                    }
                }
                keepPrelast->x += keepPrelast->dx;
                prelast->x += prelast->dx;
            }
            draw ^= 1;
        }
        // Bubble sort the active list by x (OpenCV keeps it nearly sorted).
        keepPrelast = nullptr;
        do {
            prelast = &tmp;
            last = tmp.next;
            PolyEdge* lastExchange = nullptr;
            while (last != keepPrelast && last->next != nullptr) {
                PolyEdge* te = last->next;
                if (last->x > te->x) {
                    prelast->next = te;
                    last->next = te->next;
                    te->next = last;
                    prelast = te;
                    lastExchange = prelast;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            if (lastExchange == nullptr) break;
            keepPrelast = lastExchange;
        } while (keepPrelast != tmp.next && keepPrelast != &tmp);
    }
}

}  // namespace

void fillPolyLine8(
        const int (*pts)[2], int count, std::uint8_t* mask, int mw, int mh) {
    if (count <= 0) return;
    std::vector<PolyEdge> edges;
    edges.reserve(static_cast<size_t>(count) + 1);
    // CollectPolyEdges with shift 0 / offset 0: x in 16.16 fixed point, y
    // integral, every edge drawn as a LINE_8 segment before the fill.
    int64_t px0 = static_cast<int64_t>(pts[count - 1][0]) << kXYShift;
    int64_t py0 = pts[count - 1][1];
    for (int i = 0; i < count; ++i) {
        const int64_t px1 = static_cast<int64_t>(pts[i][0]) << kXYShift;
        const int64_t py1 = pts[i][1];
        int64_t tx0 = (px0 + (kXYOne >> 1)) >> kXYShift, ty0 = py0;
        int64_t tx1 = (px1 + (kXYOne >> 1)) >> kXYShift, ty1 = py1;
        line8(mask, mw, mh, (int)tx0, (int)ty0, (int)tx1, (int)ty1);
        // Edges are built from the clipped segment (start x re-projected to
        // the unclipped upper row); unclipped edges get a half-pixel offset.
        int64_t cx0 = px0, cy0 = py0, cx1 = px1, cy1 = py1;
        if ((uint64_t)tx0 >= (uint64_t)mw || (uint64_t)tx1 >= (uint64_t)mw ||
            (uint64_t)ty0 >= (uint64_t)mh || (uint64_t)ty1 >= (uint64_t)mh) {
            clipLine(mw, mh, tx0, ty0, tx1, ty1);  // result unused upstream
            if (ty0 != ty1) {
                cy0 = ty0;
                cy1 = ty1;
                cx0 = tx0 << kXYShift;
                cx1 = tx1 << kXYShift;
            }
        } else {
            cx0 += kXYOne >> 1;
            cx1 += kXYOne >> 1;
        }
        if (py0 != py1) {
            PolyEdge edge;
            edge.dx = (cx1 - cx0) / (cy1 - cy0);
            if (py0 < py1) {
                edge.y0 = (int)py0;
                edge.y1 = (int)py1;
                edge.x = cx0 + (py0 - cy0) * edge.dx;
            } else {
                edge.y0 = (int)py1;
                edge.y1 = (int)py0;
                edge.x = cx1 + (py1 - cy1) * edge.dx;
            }
            edges.push_back(edge);
        }
        px0 = px1;
        py0 = py1;
    }
    fillEdgeCollection(edges, mask, mw, mh);
}

double dbBoxScoreFast(const float* prob, int w, int h, const float quad[4][2]) {
    float minX = quad[0][0], maxX = quad[0][0];
    float minY = quad[0][1], maxY = quad[0][1];
    for (int i = 1; i < 4; ++i) {
        minX = std::min(minX, quad[i][0]);
        maxX = std::max(maxX, quad[i][0]);
        minY = std::min(minY, quad[i][1]);
        maxY = std::max(maxY, quad[i][1]);
    }
    const int xmin = std::max(0, std::min((int)std::floor(minX), w - 1));
    const int xmax = std::max(0, std::min((int)std::ceil(maxX), w - 1));
    const int ymin = std::max(0, std::min((int)std::floor(minY), h - 1));
    const int ymax = std::max(0, std::min((int)std::ceil(maxY), h - 1));
    const int mw = xmax - xmin + 1;
    const int mh = ymax - ymin + 1;
    // numpy: float32 corners minus a Python int stay float32, then
    // astype(int32) truncates toward zero.
    int pts[4][2];
    for (int i = 0; i < 4; ++i) {
        pts[i][0] = (int)(quad[i][0] - (float)xmin);
        pts[i][1] = (int)(quad[i][1] - (float)ymin);
    }
    // Memory: one mw*mh byte mask per candidate box (box-sized, transient).
    std::vector<std::uint8_t> mask(static_cast<size_t>(mw) * mh, 0);
    fillPolyLine8(pts, 4, mask.data(), mw, mh);
    double sum = 0.0;
    int64_t n = 0;
    for (int y = 0; y < mh; ++y) {
        const float* row = prob + static_cast<size_t>(y + ymin) * w + xmin;
        const std::uint8_t* m = mask.data() + static_cast<size_t>(y) * mw;
        for (int x = 0; x < mw; ++x) {
            if (m[x]) {
                sum += row[x];
                ++n;
            }
        }
    }
    return n ? sum / static_cast<double>(n) : 0.0;
}

}  // namespace ocr
}  // namespace aicore
