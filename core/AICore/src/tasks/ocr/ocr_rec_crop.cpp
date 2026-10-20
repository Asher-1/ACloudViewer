// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ocr_rec_crop.hpp"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <vector>

// Line-for-line ports of the OpenCV 4.10 paths cv2 takes for this crop:
// convexHull (Sklansky, int points), rotatingCalipers + minAreaRect,
// RotatedRect::points, getPerspectiveTransform (8x8 LUImpl), invert (3x3
// closed form), WarpPerspectiveInvoker (32x32 blocks, parallel_for_ stripes)
// and remapBicubic with the fixed-point INTER_CUBIC table. Mixed float /
// double expressions keep upstream's operand types; the TU is compiled
// with -ffp-contract=off (core/AICore/CMakeLists.txt) because upstream's
// baseline x86-64 build never fuses a*b+c.

namespace aicore {
namespace ocr {
namespace {

struct Pt2i {
    int x, y;
};
struct Pt2f {
    float x, y;
};

inline int cvSign(long long v) { return (v > 0) - (v < 0); }

// convhull.cpp Sklansky_<int, int64>; `array` holds indices into `pts`.
int sklansky(const Pt2i* pts,
             const int* array,
             int start,
             int end,
             int* stack,
             int nsign,
             int sign2) {
    const int incr = end > start ? 1 : -1;
    int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
    int stacksize = 3;
    if (start == end || (pts[array[start]].x == pts[array[end]].x &&
                         pts[array[start]].y == pts[array[end]].y)) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    end += incr;
    while (pnext != end) {
        const int cury = pts[array[pcur]].y;
        const int nexty = pts[array[pnext]].y;
        const int by = nexty - cury;
        if (cvSign(by) != nsign) {
            const int ax = pts[array[pcur]].x - pts[array[pprev]].x;
            const int bx = pts[array[pnext]].x - pts[array[pcur]].x;
            const int ay = cury - pts[array[pprev]].y;
            const long long convexity = (long long)ay * bx - (long long)ax * by;
            if (cvSign(convexity) == sign2 && (ax != 0 || ay != 0)) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize++;
            } else if (pprev == start) {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize--;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    return --stacksize;
}

// convexHull(points, hull, clockwise=false, returnPoints=true): indices of
// the hull points in output order.
std::vector<int> convexHullIndices(const Pt2i* data0, int total) {
    std::vector<int> pointer(total), stackBuf(total + 2), hullbuf(total);
    int* stack = stackBuf.data();
    int nout = 0, miny_ind = 0, maxy_ind = 0;
    for (int i = 0; i < total; i++) pointer[i] = i;
    // CHullCmpPoints: x, then y, then the element address (= index).
    std::sort(pointer.begin(), pointer.end(), [&](int a, int b) {
        if (data0[a].x != data0[b].x) return data0[a].x < data0[b].x;
        if (data0[a].y != data0[b].y) return data0[a].y < data0[b].y;
        return a < b;
    });
    for (int i = 1; i < total; i++) {
        const int y = data0[pointer[i]].y;
        if (data0[pointer[miny_ind]].y > y) miny_ind = i;
        if (data0[pointer[maxy_ind]].y < y) maxy_ind = i;
    }
    const int* ptr = pointer.data();
    if (data0[ptr[0]].x == data0[ptr[total - 1]].x &&
        data0[ptr[0]].y == data0[ptr[total - 1]].y) {
        hullbuf[nout++] = 0;
    } else {
        int* tl_stack = stack;
        int tl_count = sklansky(data0, ptr, 0, maxy_ind, tl_stack, -1, 1);
        int* tr_stack = stack + tl_count;
        int tr_count =
                sklansky(data0, ptr, total - 1, maxy_ind, tr_stack, -1, -1);
        // clockwise == false
        std::swap(tl_stack, tr_stack);
        std::swap(tl_count, tr_count);
        for (int i = 0; i < tl_count - 1; i++)
            hullbuf[nout++] = ptr[tl_stack[i]];
        for (int i = tr_count - 1; i > 0; i--)
            hullbuf[nout++] = ptr[tr_stack[i]];
        const int stop_idx = tr_count > 2   ? tr_stack[1]
                             : tl_count > 2 ? tl_stack[tl_count - 2]
                                            : -1;
        int* bl_stack = stack;
        int bl_count = sklansky(data0, ptr, 0, miny_ind, bl_stack, 1, -1);
        int* br_stack = stack + bl_count;
        int br_count =
                sklansky(data0, ptr, total - 1, miny_ind, br_stack, 1, 1);
        if (stop_idx >= 0) {
            const int check_idx = bl_count > 2 ? bl_stack[1]
                                  : bl_count + br_count > 2
                                          ? br_stack[2 - bl_count]
                                          : -1;
            if (check_idx == stop_idx ||
                (check_idx >= 0 &&
                 data0[ptr[check_idx]].x == data0[ptr[stop_idx]].x &&
                 data0[ptr[check_idx]].y == data0[ptr[stop_idx]].y)) {
                bl_count = std::min(bl_count, 2);
                br_count = std::min(br_count, 2);
            }
        }
        for (int i = 0; i < bl_count - 1; i++)
            hullbuf[nout++] = ptr[bl_stack[i]];
        for (int i = br_count - 1; i > 0; i--)
            hullbuf[nout++] = ptr[br_stack[i]];
        // Cyclic shift towards an ascending/descending index sequence.
        if (nout >= 3) {
            int min_idx = 0, max_idx = 0, lt = 0;
            for (int i = 1; i < nout; i++) {
                const int idx = hullbuf[i];
                lt += hullbuf[i - 1] < idx;
                if (lt > 1 && lt <= i - 2) break;
                if (idx < hullbuf[min_idx]) min_idx = i;
                if (idx > hullbuf[max_idx]) max_idx = i;
            }
            const int mmdist = std::abs(max_idx - min_idx);
            if ((mmdist == 1 || mmdist == nout - 1) &&
                (lt <= 1 || lt >= nout - 2)) {
                const int ascending = (max_idx + 1) % nout == min_idx;
                const int i0 = ascending ? min_idx : max_idx;
                int j = i0, i = 0;
                if (i0 > 0) {
                    for (i = 0; i < nout; i++) {
                        const int curr_idx = stack[i] = hullbuf[j];
                        const int next_j = j + 1 < nout ? j + 1 : 0;
                        const int next_idx = hullbuf[next_j];
                        if (i < nout - 1 &&
                            (ascending != (curr_idx < next_idx)))
                            break;
                        j = next_j;
                    }
                    if (i == nout)
                        std::memcpy(hullbuf.data(), stack,
                                    nout * sizeof(hullbuf[0]));
                }
            }
        }
    }
    hullbuf.resize(nout);
    return hullbuf;
}

void rotate90CCW(const Pt2f& in, Pt2f& out) { out = {-in.y, in.x}; }
void rotate90CW(const Pt2f& in, Pt2f& out) { out = {in.y, -in.x}; }
void rotate180(const Pt2f& in, Pt2f& out) { out = {-in.x, -in.y}; }
bool firstVecIsRight(const Pt2f& vec1, const Pt2f& vec2) {
    Pt2f tmp;
    rotate90CW(vec1, tmp);
    return tmp.x * vec2.x + tmp.y * vec2.y < 0;
}

// rotcalipers.cpp rotatingCalipers(CALIPERS_MINAREARECT): corner + two
// edge vectors.
void rotatingCalipers(const Pt2f* points, int n, Pt2f out[3]) {
    float minarea = FLT_MAX;
    std::vector<float> inv_vect_length(n);
    std::vector<Pt2f> vect(n);
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = {-1, -1, -1, -1};
    Pt2f rot_vect[4];
    float orientation = 0;
    float base_a;
    float base_b = 0;
    float left_x, right_x, top_y, bottom_y;
    Pt2f pt0 = points[0];
    left_x = right_x = pt0.x;
    top_y = bottom_y = pt0.y;
    for (int i = 0; i < n; i++) {
        if (pt0.x < left_x) left_x = pt0.x, left = i;
        if (pt0.x > right_x) right_x = pt0.x, right = i;
        if (pt0.y > top_y) top_y = pt0.y, top = i;
        if (pt0.y < bottom_y) bottom_y = pt0.y, bottom = i;
        const Pt2f pt = points[i + 1 < n ? i + 1 : 0];
        // float - float, then widened (upstream's double dx = pt.x - pt0.x)
        const double dx = pt.x - pt0.x;
        const double dy = pt.y - pt0.y;
        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1. / std::sqrt(dx * dx + dy * dy));
        pt0 = pt;
    }
    {
        double ax = vect[n - 1].x;
        double ay = vect[n - 1].y;
        for (int i = 0; i < n; i++) {
            const double bx = vect[i].x;
            const double by = vect[i].y;
            const double convexity = ax * by - ay * bx;
            if (convexity != 0) {
                orientation = (convexity > 0) ? 1.f : (-1.f);
                break;
            }
            ax = bx;
            ay = by;
        }
    }
    base_a = orientation;
    seq[0] = bottom;
    seq[1] = right;
    seq[2] = top;
    seq[3] = left;
    int bestLeft = 0, bestBottom = 0;
    float bestA = 0, bestWidth = 0, bestB = 0, bestHeight = 0;
    for (int k = 0; k < n; k++) {
        int main_element = 0;
        rot_vect[0] = vect[seq[0]];
        rotate90CW(vect[seq[1]], rot_vect[1]);
        rotate180(vect[seq[2]], rot_vect[2]);
        rotate90CCW(vect[seq[3]], rot_vect[3]);
        for (int i = 1; i < 4; i++)
            if (firstVecIsRight(rot_vect[i], rot_vect[main_element]))
                main_element = i;
        {
            const int pindex = seq[main_element];
            const float lead_x = vect[pindex].x * inv_vect_length[pindex];
            const float lead_y = vect[pindex].y * inv_vect_length[pindex];
            switch (main_element) {
                case 0:
                    base_a = lead_x;
                    base_b = lead_y;
                    break;
                case 1:
                    base_a = lead_y;
                    base_b = -lead_x;
                    break;
                case 2:
                    base_a = -lead_x;
                    base_b = -lead_y;
                    break;
                default:
                    base_a = -lead_y;
                    base_b = lead_x;
                    break;
            }
        }
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];
        float dx = points[seq[1]].x - points[seq[3]].x;
        float dy = points[seq[1]].y - points[seq[3]].y;
        const float width = dx * base_a + dy * base_b;
        dx = points[seq[2]].x - points[seq[0]].x;
        dy = points[seq[2]].y - points[seq[0]].y;
        const float height = -dx * base_b + dy * base_a;
        const float area = width * height;
        if (area <= minarea) {
            minarea = area;
            bestLeft = seq[3];
            bestA = base_a;
            bestWidth = width;
            bestB = base_b;
            bestHeight = height;
            bestBottom = seq[0];
        }
    }
    const float A1 = bestA;
    const float B1 = bestB;
    const float A2 = -bestB;
    const float B2 = bestA;
    const float C1 = A1 * points[bestLeft].x + points[bestLeft].y * B1;
    const float C2 = A2 * points[bestBottom].x + points[bestBottom].y * B2;
    const float idet = 1.f / (A1 * B2 - A2 * B1);
    const float px = (C1 * B2 - C2 * B1) * idet;
    const float py = (A1 * C2 - A2 * C1) * idet;
    out[0] = {px, py};
    out[1] = {A1 * bestWidth, B1 * bestWidth};
    out[2] = {A2 * bestHeight, B2 * bestHeight};
}

constexpr double kPi = 3.1415926535897932384626433832795;  // CV_PI

// OpenCV 4.10 imgwarp.cpp INTER_CUBIC fixed-point table (initInterTab2D).
constexpr int kInterBits = 5;
constexpr int kInterTabSize = 1 << kInterBits;
constexpr int kRemapCoefBits = 15;
constexpr int kRemapCoefScale = 1 << kRemapCoefBits;

struct BicubicTab {
    short w[kInterTabSize * kInterTabSize][16];
};

void interpolateCubic(float x, float* coeffs) {
    const float A = -0.75f;
    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

// saturate_cast<short>(float): cvRound (nearest-even) then clamp.
short saturateShort(float v) {
    const long r = std::lrint(v);
    return (short)std::clamp<long>(r, SHRT_MIN, SHRT_MAX);
}

BicubicTab makeBicubicTab() {
    BicubicTab t{};
    float tab1d[kInterTabSize * 4];
    const float scale = 1.f / kInterTabSize;
    for (int i = 0; i < kInterTabSize; i++)
        interpolateCubic(i * scale, tab1d + i * 4);
    const int ksize = 4;
    for (int i = 0; i < kInterTabSize; i++)
        for (int j = 0; j < kInterTabSize; j++) {
            short* itab = t.w[i * kInterTabSize + j];
            int isum = 0;
            for (int k1 = 0; k1 < ksize; k1++) {
                const float vy = tab1d[i * ksize + k1];
                for (int k2 = 0; k2 < ksize; k2++) {
                    const float v = vy * tab1d[j * ksize + k2];
                    isum += itab[k1 * ksize + k2] =
                            saturateShort(v * kRemapCoefScale);
                }
            }
            if (isum != kRemapCoefScale) {
                const int diff = isum - kRemapCoefScale;
                const int ksize2 = ksize / 2;
                int Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                for (int k1 = ksize2; k1 < ksize2 + 2; k1++)
                    for (int k2 = ksize2; k2 < ksize2 + 2; k2++) {
                        if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                            mk1 = k1, mk2 = k2;
                        else if (itab[k1 * ksize + k2] >
                                 itab[Mk1 * ksize + Mk2])
                            Mk1 = k1, Mk2 = k2;
                    }
                if (diff < 0)
                    itab[Mk1 * ksize + Mk2] =
                            (short)(itab[Mk1 * ksize + Mk2] - diff);
                else
                    itab[mk1 * ksize + mk2] =
                            (short)(itab[mk1 * ksize + mk2] - diff);
            }
        }
    return t;
}

const BicubicTab& bicubicTab() {
    static const BicubicTab tab = makeBicubicTab();
    return tab;
}

// matrix_decomp.cpp LUImpl<double> (eps = DBL_EPSILON * 100), one rhs.
bool luSolve(double A[8][8], double b[8]) {
    const int m = 8;
    for (int i = 0; i < m; i++) {
        int k = i;
        for (int j = i + 1; j < m; j++)
            if (std::abs(A[j][i]) > std::abs(A[k][i])) k = j;
        if (std::abs(A[k][i]) < DBL_EPSILON * 100) return false;
        if (k != i) {
            for (int j = i; j < m; j++) std::swap(A[i][j], A[k][j]);
            std::swap(b[i], b[k]);
        }
        const double d = -1 / A[i][i];
        for (int j = i + 1; j < m; j++) {
            const double alpha = A[j][i] * d;
            for (int c = i + 1; c < m; c++) A[j][c] += alpha * A[i][c];
            b[j] += alpha * b[i];
        }
    }
    for (int i = m - 1; i >= 0; i--) {
        double s = b[i];
        for (int c = i + 1; c < m; c++) s -= A[i][c] * b[c];
        b[i] = s / A[i][i];
    }
    return true;
}

// getPerspectiveTransform(src, dst, DECOMP_LU) then invert(DECOMP_LU) of
// the 3x3 (closed form): the destination -> source map warpPerspective
// samples with.
bool inverseWarpMatrix(const Pt2f src[4], const Pt2f dst[4], double M[9]) {
    double a[8][8], b[8];
    for (int i = 0; i < 4; ++i) {
        a[i][0] = a[i + 4][3] = src[i].x;
        a[i][1] = a[i + 4][4] = src[i].y;
        a[i][2] = a[i + 4][5] = 1;
        a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] =
                0;
        // float products, as upstream (Point2f operands)
        a[i][6] = -src[i].x * dst[i].x;
        a[i][7] = -src[i].y * dst[i].x;
        a[i + 4][6] = -src[i].x * dst[i].y;
        a[i + 4][7] = -src[i].y * dst[i].y;
        b[i] = dst[i].x;
        b[i + 4] = dst[i].y;
    }
    // A singular transform has a zero inverse in cv2.warpPerspective; the
    // caller preserves its top-left BORDER_REPLICATE sampling below.
    if (!luSolve(a, b)) return false;
    double S[9];
    for (int i = 0; i < 8; ++i) S[i] = b[i];
    S[8] = 1.;
    auto s = [&](int y, int x) { return S[y * 3 + x]; };
    double d =
            s(0, 0) * ((double)s(1, 1) * s(2, 2) - (double)s(1, 2) * s(2, 1)) -
            s(0, 1) * ((double)s(1, 0) * s(2, 2) - (double)s(1, 2) * s(2, 0)) +
            s(0, 2) * ((double)s(1, 0) * s(2, 1) - (double)s(1, 1) * s(2, 0));
    if (d == 0.) return false;
    d = 1. / d;
    M[0] = (s(1, 1) * s(2, 2) - s(1, 2) * s(2, 1)) * d;
    M[1] = (s(0, 2) * s(2, 1) - s(0, 1) * s(2, 2)) * d;
    M[2] = (s(0, 1) * s(1, 2) - s(0, 2) * s(1, 1)) * d;
    M[3] = (s(1, 2) * s(2, 0) - s(1, 0) * s(2, 2)) * d;
    M[4] = (s(0, 0) * s(2, 2) - s(0, 2) * s(2, 0)) * d;
    M[5] = (s(0, 2) * s(1, 0) - s(0, 0) * s(1, 2)) * d;
    M[6] = (s(1, 0) * s(2, 1) - s(1, 1) * s(2, 0)) * d;
    M[7] = (s(0, 1) * s(2, 0) - s(0, 0) * s(2, 1)) * d;
    M[8] = (s(0, 0) * s(1, 1) - s(0, 1) * s(1, 0)) * d;
    return true;
}

// saturate_cast<int>(double) after upstream's [INT_MIN, INT_MAX] clamp.
int roundToInt(double v) { return (int)std::lrint(v); }
short saturateShortInt(int v) {
    return (short)std::clamp(v, (int)SHRT_MIN, (int)SHRT_MAX);
}

}  // namespace

void minAreaRectBox(const int (*pts)[2],
                    int count,
                    float box[4][2],
                    float* minSide) {
    if (count <= 0) {  // cv::convexHull of no points: an empty RotatedRect
        std::memset(box, 0, sizeof(float) * 8);
        if (minSide) *minSide = 0;
        return;
    }
    std::vector<Pt2i> data(count);
    for (int i = 0; i < count; ++i) data[i] = {pts[i][0], pts[i][1]};
    const std::vector<int> hullIdx = convexHullIndices(data.data(), count);
    std::vector<Pt2f> hull(hullIdx.size());
    for (size_t i = 0; i < hullIdx.size(); ++i)
        hull[i] = {(float)data[hullIdx[i]].x, (float)data[hullIdx[i]].y};
    const int n = static_cast<int>(hull.size());
    // minAreaRect
    Pt2f center{0, 0};
    float sizeW = 0, sizeH = 0, angle = 0;
    if (n > 2) {
        Pt2f out[3];
        rotatingCalipers(hull.data(), n, out);
        center.x = out[0].x + (out[1].x + out[2].x) * 0.5f;
        center.y = out[0].y + (out[1].y + out[2].y) * 0.5f;
        sizeW = (float)std::sqrt((double)out[1].x * out[1].x +
                                 (double)out[1].y * out[1].y);
        sizeH = (float)std::sqrt((double)out[2].x * out[2].x +
                                 (double)out[2].y * out[2].y);
        angle = (float)std::atan2((double)out[1].y, (double)out[1].x);
    } else if (n == 2) {
        center.x = (hull[0].x + hull[1].x) * 0.5f;
        center.y = (hull[0].y + hull[1].y) * 0.5f;
        const double dx = hull[1].x - hull[0].x;
        const double dy = hull[1].y - hull[0].y;
        sizeW = (float)std::sqrt(dx * dx + dy * dy);
        sizeH = 0;
        angle = (float)std::atan2(dy, dx);
    } else if (n == 1) {
        center = hull[0];
    }
    // float * int stays float, then the double division (upstream
    // box.angle*180/CV_PI).
    angle = (float)((angle * 180) / kPi);
    if (minSide) *minSide = std::min(sizeW, sizeH);
    // RotatedRect::points
    const double rad = angle * kPi / 180.;
    const float b = (float)std::cos(rad) * 0.5f;
    const float a = (float)std::sin(rad) * 0.5f;
    Pt2f p[4];
    p[0].x = center.x - a * sizeH - b * sizeW;
    p[0].y = center.y + b * sizeH - a * sizeW;
    p[1].x = center.x + a * sizeH - b * sizeW;
    p[1].y = center.y - b * sizeH - a * sizeW;
    p[2].x = 2 * center.x - p[0].x;
    p[2].y = 2 * center.y - p[0].y;
    p[3].x = 2 * center.x - p[1].x;
    p[3].y = 2 * center.y - p[1].y;
    // sorted(points, key=x) (stable), then the upstream index pick.
    std::stable_sort(p, p + 4,
                     [](const Pt2f& l, const Pt2f& r) { return l.x < r.x; });
    int ia, ib, ic, id;
    if (p[1].y > p[0].y) {
        ia = 0;
        id = 1;
    } else {
        ia = 1;
        id = 0;
    }
    if (p[3].y > p[2].y) {
        ib = 2;
        ic = 3;
    } else {
        ib = 3;
        ic = 2;
    }
    const int order[4] = {ia, ib, ic, id};
    for (int i = 0; i < 4; ++i) {
        box[i][0] = p[order[i]].x;
        box[i][1] = p[order[i]].y;
    }
}

void rotateCropSize(const float box[4][2], int* width, int* height) {
    // np.linalg.norm of a float32 difference: float32 x*x + y*y (no fused
    // multiply-add), float32 sqrt.
    auto norm = [&](int i, int j) {
        const float dx = box[i][0] - box[j][0];
        const float dy = box[i][1] - box[j][1];
        const float sq = dx * dx + dy * dy;
        return std::sqrt(sq);
    };
    *width = (int)std::max(norm(0, 1), norm(2, 3));
    *height = (int)std::max(norm(0, 3), norm(1, 2));
}

bool warpPerspectiveCubicReplicate(const aicore::img::Image& src,
                                   const float box[4][2],
                                   int width,
                                   int height,
                                   std::uint8_t* dst) {
    std::memset(dst, 0, static_cast<size_t>(width) * height * 3);
    if (width <= 0 || height <= 0 || src.empty()) return false;
    const Pt2f from[4] = {{box[0][0], box[0][1]},
                          {box[1][0], box[1][1]},
                          {box[2][0], box[2][1]},
                          {box[3][0], box[3][1]}};
    const Pt2f to[4] = {{0, 0},
                        {(float)width, 0},
                        {(float)width, (float)height},
                        {0, (float)height}};
    double M[9];
    if (!inverseWarpMatrix(from, to, M)) std::fill(M, M + 9, 0.0);

    const BicubicTab& tab = bicubicTab();
    const int sw = src.width, sh = src.height, scn = src.channels;
    const int cmap[3] = {src.channel_index(0), src.channel_index(1),
                         src.channel_index(2)};
    const unsigned width1 = (unsigned)std::max(sw - 3, 0);
    const unsigned height1 = (unsigned)std::max(sh - 3, 0);
    // remapBicubic<FixedPtCast<int, uchar, 15>> for one destination pixel.
    auto sample = [&](int sxy0, int sxy1, int alpha, std::uint8_t* out) {
        const int sx = sxy0 - 1, sy = sxy1 - 1;
        const short* w = tab.w[alpha];
        int xs[4], ys[4];
        const bool inside = (unsigned)sx < width1 && (unsigned)sy < height1;
        for (int i = 0; i < 4; ++i) {
            // BORDER_REPLICATE borderInterpolate
            xs[i] = inside ? sx + i : std::clamp(sx + i, 0, sw - 1);
            ys[i] = inside ? sy + i : std::clamp(sy + i, 0, sh - 1);
        }
        for (int c = 0; c < 3; ++c) {
            int sum = 0;
            for (int i = 0; i < 4; ++i) {
                const std::uint8_t* row = src.row(ys[i]);
                for (int j = 0; j < 4; ++j)
                    sum += row[xs[j] * scn + cmap[c]] * w[i * 4 + j];
            }
            out[c] = (std::uint8_t)std::clamp(
                    (sum + (1 << (kRemapCoefBits - 1))) >> kRemapCoefBits, 0,
                    255);
        }
    };

    // WarpPerspectiveInvoker block walk. The per-row origin X0/Y0/W0 is
    // taken at each block's first column, so the floating-point result
    // depends on the block grid, which depends on the parallel_for_ stripe
    // split (nstripes = total / 65536, independent of the thread count).
    const int kBlock = 32;
    int bh0 = std::min(kBlock / 2, height);
    const int bw0 = std::min(kBlock * kBlock / bh0, width);
    bh0 = std::min(kBlock * kBlock / bw0, height);
    const double total = static_cast<double>(width) * height;
    const double nstripesD = total / (double)(1 << 16);
    const int len = height;
    int nstripes = (int)std::lrint(
            nstripesD <= 0 ? (double)len
                           : std::min(std::max(nstripesD, 1.), (double)len));
    if (len <= 1) nstripes = 1;
    for (int s = 0; s < nstripes; ++s) {
        int r0 = 0, r1 = len;
        if (nstripes > 1) {
            r0 = (int)(((uint64_t)s * len + nstripes / 2) / nstripes);
            r1 = s + 1 >= nstripes
                         ? len
                         : (int)(((uint64_t)(s + 1) * len + nstripes / 2) /
                                 nstripes);
        }
        for (int y = r0; y < r1; y += bh0) {
            for (int x = 0; x < width; x += bw0) {
                const int bw = std::min(bw0, width - x);
                const int bh = std::min(bh0, r1 - y);
                for (int y1 = 0; y1 < bh; y1++) {
                    const double X0 = M[0] * x + M[1] * (y + y1) + M[2];
                    const double Y0 = M[3] * x + M[4] * (y + y1) + M[5];
                    const double W0 = M[6] * x + M[7] * (y + y1) + M[8];
                    std::uint8_t* drow =
                            dst + (static_cast<size_t>(y + y1) * width + x) * 3;
                    for (int x1 = 0; x1 < bw; x1++) {
                        double W = W0 + M[6] * x1;
                        W = W ? kInterTabSize / W : 0;
                        const double fX =
                                std::max((double)INT_MIN,
                                         std::min((double)INT_MAX,
                                                  (X0 + M[0] * x1) * W));
                        const double fY =
                                std::max((double)INT_MIN,
                                         std::min((double)INT_MAX,
                                                  (Y0 + M[3] * x1) * W));
                        const int X = roundToInt(fX);
                        const int Y = roundToInt(fY);
                        const int alpha =
                                (Y & (kInterTabSize - 1)) * kInterTabSize +
                                (X & (kInterTabSize - 1));
                        sample(saturateShortInt(X >> kInterBits),
                               saturateShortInt(Y >> kInterBits), alpha,
                               drow + x1 * 3);
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace ocr
}  // namespace aicore
