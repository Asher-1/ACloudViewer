// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ocr_db_postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "ocr_box_score.hpp"
#include "ocr_rec_crop.hpp"

// Ports keep upstream's operand types; the TU is compiled with
// -ffp-contract=off (core/AICore/CMakeLists.txt) like ocr_rec_crop.cpp.

namespace aicore {
namespace ocr {
namespace {

// ---- OpenCV 4.10 contours_new.cpp, CV_8UC1 RETR_LIST path ----------------

constexpr std::int8_t kMaskRight = static_cast<std::int8_t>(0x80);
constexpr std::int8_t kMaskNew = 0x02;
constexpr std::int8_t kMaskBlack = 0x01;
constexpr int kChainDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kChainDy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

inline long delta(int s, long step) {
    return kChainDx[s % 8] + kChainDy[s % 8] * step;
}

// icvFetchContourEx<schar>(nbd = MASK8_NEW, CHAIN_APPROX_SIMPLE). `pt` is the
// contour origin in output coordinates (the scanner offset is -1, -1).
void fetchContour(std::int8_t* i0,
                  long step,
                  int ptX,
                  int ptY,
                  bool isHole,
                  std::vector<std::array<int, 2>>& pts) {
    const std::int8_t nbd = kMaskNew;
    std::int8_t* i1 = nullptr;
    std::int8_t* i3 = nullptr;
    std::int8_t* i4 = nullptr;
    int sEnd = isHole ? 0 : 4;
    int s = sEnd;
    do {
        s = (s - 1) & 7;
        i1 = i0 + delta(s, step);
    } while (*i1 == 0 && s != sEnd);

    if (s == sEnd) {  // single pixel domain
        *i0 = static_cast<std::int8_t>(nbd | kMaskRight);
        pts.push_back({ptX, ptY});
        return;
    }
    i3 = i0;
    int prevS = s ^ 4;
    for (;;) {
        sEnd = s;
        s = std::min(s, 15);
        while (s < 15) {
            ++s;
            i4 = i3 + delta(s, step);
            if (*i4 != 0) break;
        }
        s &= 7;
        if (static_cast<unsigned>(s - 1) < static_cast<unsigned>(sEnd))
            *i3 = static_cast<std::int8_t>(nbd | kMaskRight);
        else if (*i3 == kMaskBlack)
            *i3 = nbd;
        if (s != prevS) pts.push_back({ptX, ptY});
        prevS = s;
        ptX += kChainDx[s];
        ptY += kChainDy[s];
        if (i4 == i0 && i3 == i1) break;
        i3 = i4;
        s = (s + 4) & 7;
    }
}

// ---- Clipper 6.4.2 (pyclipper 1.4.0) ClipperOffset, one closed path ------

struct IntPt {
    long long x, y;
    bool operator==(const IntPt& o) const { return x == o.x && y == o.y; }
    bool operator!=(const IntPt& o) const { return !(*this == o); }
};
struct DblPt {
    double x, y;
};

inline long long clipperRound(double v) {
    return v < 0 ? static_cast<long long>(v - 0.5)
                 : static_cast<long long>(v + 0.5);
}

double clipperArea(const std::vector<IntPt>& poly) {
    const int size = static_cast<int>(poly.size());
    if (size < 3) return 0;
    double a = 0;
    for (int i = 0, j = size - 1; i < size; ++i) {
        a += ((double)poly[j].x + poly[i].x) * ((double)poly[j].y - poly[i].y);
        j = i;
    }
    return -a * 0.5;
}

DblPt unitNormal(const IntPt& pt1, const IntPt& pt2) {
    if (pt2.x == pt1.x && pt2.y == pt1.y) return {0, 0};
    double dx = (double)(pt2.x - pt1.x);
    double dy = (double)(pt2.y - pt1.y);
    const double f = 1 * 1.0 / std::sqrt(dx * dx + dy * dy);
    dx *= f;
    dy *= f;
    return {dy, -dx};
}

// AddPath(jtRound, etClosedPolygon) + FixOrientations + DoOffset for one
// path; empty when AddPath rejects it (fewer than 3 distinct points).
std::vector<IntPt> doOffsetRound(const std::vector<IntPt>& path, double delta) {
    int highI = static_cast<int>(path.size()) - 1;
    if (highI < 0) return {};
    while (highI > 0 && path[0] == path[highI]) highI--;
    std::vector<IntPt> src;
    src.push_back(path[0]);
    for (int i = 1; i <= highI; i++)
        if (src.back() != path[i]) src.push_back(path[i]);
    if (src.size() < 3) return {};
    if (!(clipperArea(src) >= 0)) std::reverse(src.begin(), src.end());

    constexpr double kPi = 3.141592653589793238;
    constexpr double kTwoPi = kPi * 2;
    constexpr double kDefArcTolerance = 0.25;
    const double arcTolerance = 0.25;  // PyclipperOffset default
    double y;
    if (arcTolerance > std::fabs(delta) * kDefArcTolerance)
        y = std::fabs(delta) * kDefArcTolerance;
    else
        y = arcTolerance;
    double steps = kPi / std::acos(1 - y / std::fabs(delta));
    if (steps > std::fabs(delta) * kPi) steps = std::fabs(delta) * kPi;
    double sinStep = std::sin(kTwoPi / steps);
    const double cosStep = std::cos(kTwoPi / steps);
    const double stepsPerRad = steps / kTwoPi;
    if (delta < 0.0) sinStep = -sinStep;

    const int len = static_cast<int>(src.size());
    std::vector<DblPt> normals;
    normals.reserve(len);
    for (int j = 0; j < len - 1; ++j)
        normals.push_back(unitNormal(src[j], src[j + 1]));
    normals.push_back(unitNormal(src[len - 1], src[0]));

    std::vector<IntPt> dest;
    auto at = [&](int j, const DblPt& n) {
        return IntPt{clipperRound(src[j].x + n.x * delta),
                     clipperRound(src[j].y + n.y * delta)};
    };
    int k = len - 1;
    for (int j = 0; j < len; ++j) {  // OffsetPoint(j, k, jtRound)
        double sinA = normals[k].x * normals[j].y - normals[j].x * normals[k].y;
        bool joined = false;
        if (std::fabs(sinA * delta) < 1.0) {
            const double cosA =
                    normals[k].x * normals[j].x + normals[j].y * normals[k].y;
            if (cosA > 0) {
                dest.push_back(at(j, normals[k]));
                continue;  // upstream returns before k = j
            }
        } else if (sinA > 1.0) {
            sinA = 1.0;
        } else if (sinA < -1.0) {
            sinA = -1.0;
        }
        if (sinA * delta < 0) {
            dest.push_back(at(j, normals[k]));
            dest.push_back(src[j]);
            dest.push_back(at(j, normals[j]));
            joined = true;
        }
        if (!joined) {  // DoRound(j, k)
            const double a =
                    std::atan2(sinA, normals[k].x * normals[j].x +
                                             normals[k].y * normals[j].y);
            const int n = std::max(
                    static_cast<int>(clipperRound(stepsPerRad * std::fabs(a))),
                    1);
            double X = normals[k].x, Y = normals[k].y, X2;
            for (int i = 0; i < n; ++i) {
                dest.push_back({clipperRound(src[j].x + X * delta),
                                clipperRound(src[j].y + Y * delta)});
                X2 = X;
                X = X * cosStep - sinStep * Y;
                Y = X2 * sinStep + Y * cosStep;
            }
            dest.push_back(at(j, normals[j]));
        }
        k = j;
    }
    return dest;
}

// The ctUnion/pftPositive pass Execute runs over the offset polygon. For the
// convex boxes DB emits it removes duplicate and collinear vertices and
// starts the path right after the min-Y (then max-X) vertex; derived from and
// checked against pyclipper on every ICDAR2015 candidate and the golden set
// (tests/ocr/db_postprocess_golden.txt).
std::vector<IntPt> unionCleanup(std::vector<IntPt> q) {
    bool changed = true;
    while (changed && q.size() > 2) {
        changed = false;
        const size_t n = q.size();
        for (size_t i = 0; i < n; ++i) {
            const IntPt& a = q[(i + n - 1) % n];
            const IntPt& b = q[i];
            const IntPt& c = q[(i + 1) % n];
            // Exact for |coords| < 2^31 (det maps are a few thousand px).
            if (b == a ||
                (b.x - a.x) * (c.y - b.y) == (b.y - a.y) * (c.x - b.x)) {
                q.erase(q.begin() + static_cast<long>(i));
                changed = true;
                break;
            }
        }
    }
    if (q.empty()) return q;
    size_t last = 0;
    for (size_t i = 1; i < q.size(); ++i)
        if (q[i].y < q[last].y || (q[i].y == q[last].y && q[i].x > q[last].x))
            last = i;
    std::rotate(q.begin(), q.begin() + static_cast<long>(last + 1), q.end());
    return q;
}

// cv::contourArea / cv::arcLength of a float32 closed contour.
double contourArea4(const float box[4][2]) {
    double a00 = 0;
    float px = box[3][0], py = box[3][1];
    for (int i = 0; i < 4; ++i) {
        a00 += (double)px * box[i][1] - (double)py * box[i][0];
        px = box[i][0];
        py = box[i][1];
    }
    a00 *= 0.5;
    return std::fabs(a00);
}
double arcLength4(const float box[4][2]) {
    double perimeter = 0;
    float px = box[3][0], py = box[3][1];
    for (int i = 0; i < 4; ++i) {
        const float dx = box[i][0] - px, dy = box[i][1] - py;
        perimeter += std::sqrt(dx * dx + dy * dy);
        px = box[i][0];
        py = box[i][1];
    }
    return perimeter;
}

}  // namespace

std::vector<std::vector<std::array<int, 2>>> findContoursList(
        const std::uint8_t* bitmap,
        int w,
        int h,
        std::vector<std::int8_t>& scratch) {
    // copyMakeBorder(1, BORDER_CONSTANT 0) + threshold(0, 1, BINARY).
    const int W = w + 2, H = h + 2;
    const long step = W;
    scratch.assign(static_cast<size_t>(W) * H, 0);
    for (int y = 0; y < h; ++y) {
        const std::uint8_t* src = bitmap + static_cast<size_t>(y) * w;
        std::int8_t* dst = scratch.data() + static_cast<size_t>(y + 1) * W + 1;
        for (int x = 0; x < w; ++x) dst[x] = src[x] ? 1 : 0;
    }
    std::int8_t* img = scratch.data();
    std::vector<std::vector<std::array<int, 2>>> found;
    // ContourScanner_::findNext over the whole image; RETR_LIST never needs
    // the parent search, so lnbd bookkeeping is dropped.
    const int width = W - 1, height = H - 1;
    int x = 1, y = 1;
    int prev = img[static_cast<size_t>(y) * W + x - 1];
    for (; y < height; y++) {
        int p = 0;
        for (; x < width; x++) {
            const std::int8_t* row = img + static_cast<size_t>(y) * W;
            for (; x < width && (p = row[x]) == prev; x++) {
            }
            if (x >= width) break;
            bool isHole = false;
            bool start = prev == 0 && p == 1;
            if (!start && !(p != 0 || prev < 1)) {
                isHole = true;
                start = true;
            }
            if (start) {
                const int sx = x - (isHole ? 1 : 0);
                std::vector<std::array<int, 2>> pts;
                pts.reserve(64);
                fetchContour(img + static_cast<size_t>(y) * W + sx, step,
                             sx - 1, y - 1, isHole, pts);
                found.push_back(std::move(pts));
                // Resume at x + 1 with prev re-read from the traced image.
                prev = img[static_cast<size_t>(y) * W + x];
                continue;
            }
            prev = p;
        }
        x = 1;
        prev = 0;
    }
    // Tree::addChild prepends and the result iterator walks first_child
    // first: OpenCV returns RETR_LIST contours in reverse discovery order.
    std::reverse(found.begin(), found.end());
    return found;
}

std::vector<std::array<long long, 2>> clipperOffsetRound(const float box[4][2],
                                                         double distance) {
    std::vector<IntPt> path(4);
    for (int i = 0; i < 4; ++i)  // pyclipper IntPoint(float): truncation
        path[i] = {static_cast<long long>(box[i][0]),
                   static_cast<long long>(box[i][1])};
    std::vector<IntPt> out = doOffsetRound(path, distance);
    if (out.empty()) return {};
    out = unionCleanup(std::move(out));
    std::vector<std::array<long long, 2>> res(out.size());
    for (size_t i = 0; i < out.size(); ++i) res[i] = {out[i].x, out[i].y};
    return res;
}

std::vector<DbBox> dbBoxesFromBitmap(const float* pred,
                                     int w,
                                     int h,
                                     int destW,
                                     int destH,
                                     const DbParams& params,
                                     std::vector<std::int8_t>& scratch) {
    std::vector<std::uint8_t> bitmap(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < bitmap.size(); ++i)
        bitmap[i] = pred[i] > params.thresh ? 1 : 0;
    const auto contours = findContoursList(bitmap.data(), w, h, scratch);
    const double widthScale = static_cast<double>(destW) / w;
    const double heightScale = static_cast<double>(destH) / h;
    const size_t numContours = std::min(
            contours.size(), static_cast<size_t>(params.maxCandidates));
    std::vector<DbBox> boxes;
    std::vector<int> pts;
    for (size_t ci = 0; ci < numContours; ++ci) {
        const auto& contour = contours[ci];
        float box[4][2];
        float sside = 0;
        minAreaRectBox(reinterpret_cast<const int(*)[2]>(contour.data()),
                       static_cast<int>(contour.size()), box, &sside);
        if (sside < params.minSize) continue;
        const double score = dbBoxScoreFast(pred, w, h, box);
        if (params.boxThresh > score) continue;
        // unclip: distance = area * unclip_ratio / length (Python doubles).
        const double distance =
                contourArea4(box) * params.unclipRatio / arcLength4(box);
        const auto expanded = clipperOffsetRound(box, distance);
        if (expanded.empty()) continue;
        pts.resize(expanded.size() * 2);
        for (size_t i = 0; i < expanded.size(); ++i) {
            // cv2 takes the int64 array as CV_32S.
            pts[i * 2] = static_cast<int>(expanded[i][0]);
            pts[i * 2 + 1] = static_cast<int>(expanded[i][1]);
        }
        minAreaRectBox(reinterpret_cast<const int(*)[2]>(pts.data()),
                       static_cast<int>(expanded.size()), box, &sside);
        if (sside < params.minSize + 2) continue;
        DbBox out;
        for (int i = 0; i < 4; ++i) {
            // round(np.float32 * float) is half-to-even on the double
            // product; the clamped value is stored back as float32, then
            // astype(np.int16).
            const double sx = std::nearbyint((double)box[i][0] * widthScale);
            const double sy = std::nearbyint((double)box[i][1] * heightScale);
            out.quad[i][0] = static_cast<int>(
                    std::max(0.0, std::min(sx, static_cast<double>(destW))));
            out.quad[i][1] = static_cast<int>(
                    std::max(0.0, std::min(sy, static_cast<double>(destH))));
        }
        out.score = score;
        boxes.push_back(out);
    }
    return boxes;
}

void sortQuadBoxes(std::vector<DbBox>& boxes) {
    std::stable_sort(boxes.begin(), boxes.end(),
                     [](const DbBox& a, const DbBox& b) {
                         if (a.quad[0][1] != b.quad[0][1])
                             return a.quad[0][1] < b.quad[0][1];
                         return a.quad[0][0] < b.quad[0][0];
                     });
    const int n = static_cast<int>(boxes.size());
    for (int i = 0; i < n - 1; ++i)
        for (int j = i; j >= 0; --j) {
            if (std::abs(boxes[j + 1].quad[0][1] - boxes[j].quad[0][1]) < 10 &&
                boxes[j + 1].quad[0][0] < boxes[j].quad[0][0])
                std::swap(boxes[j], boxes[j + 1]);
            else
                break;
        }
}

}  // namespace ocr
}  // namespace aicore
