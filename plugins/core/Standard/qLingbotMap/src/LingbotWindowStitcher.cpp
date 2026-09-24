// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LingbotWindowStitcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lingbot_stitch {

namespace {

constexpr float kMinScale = 1e-3f;
constexpr float kMaxScale = 1e3f;
constexpr float kFinfoEps = std::numeric_limits<float>::epsilon();

//! Median of a scratch buffer (nth_element on a copy; overlap is a few
//! frames of pixels — sorting cost is trivial next to inference).
float medianOf(std::vector<float>& values) {
    const size_t n = values.size();
    if (n == 0) return 1.0f;
    const size_t mid = n / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    float median = values[mid];
    if (n % 2 == 0) {
        const float lower =
                *std::max_element(values.begin(), values.begin() + mid);
        median = 0.5f * (median + lower);
    }
    return median;
}

//! m1 @ m2 for row-major 3x3 matrices.
void mat3Mul(const float a[9], const float b[9], float out[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float acc = 0.f;
            for (int k = 0; k < 3; ++k) {
                acc += a[r * 3 + k] * b[k * 3 + c];
            }
            out[r * 3 + c] = acc;
        }
    }
}

//! out = m @ v
void mat3Vec(const float m[9], const float v[3], float out[3]) {
    for (int r = 0; r < 3; ++r) {
        out[r] =
                m[r * 3 + 0] * v[0] + m[r * 3 + 1] * v[1] + m[r * 3 + 2] * v[2];
    }
}

}  // namespace

void quatToMat(const float q[4], float outR[9]) {
    const float i = q[0], j = q[1], k = q[2], r = q[3];
    const float norm = i * i + j * j + k * k + r * r;
    const float twoS = 2.0f / std::max(norm, 1e-30f);
    outR[0] = 1.f - twoS * (j * j + k * k);
    outR[1] = twoS * (i * j - k * r);
    outR[2] = twoS * (i * k + j * r);
    outR[3] = twoS * (i * j + k * r);
    outR[4] = 1.f - twoS * (i * i + k * k);
    outR[5] = twoS * (j * k - i * r);
    outR[6] = twoS * (i * k - j * r);
    outR[7] = twoS * (j * k + i * r);
    outR[8] = 1.f - twoS * (i * i + j * j);
}

void matToQuat(const float m[9], float outQ[4]) {
    const float m00 = m[0], m01 = m[1], m02 = m[2];
    const float m10 = m[3], m11 = m[4], m12 = m[5];
    const float m20 = m[6], m21 = m[7], m22 = m[8];
    // q_abs[i] from the largest diagonal-sum candidate (best conditioned).
    float qAbs[4];
    qAbs[0] = std::sqrt(std::max(1.f + m00 + m11 + m22, 0.f));
    qAbs[1] = std::sqrt(std::max(1.f + m00 - m11 - m22, 0.f));
    qAbs[2] = std::sqrt(std::max(1.f - m00 + m11 - m22, 0.f));
    qAbs[3] = std::sqrt(std::max(1.f - m00 - m11 + m22, 0.f));
    // Candidate rows in (r, i, j, k)-quaternion order, per rotation.py.
    const float rows[4][4] = {
            {qAbs[0] * qAbs[0], m21 - m12, m02 - m20, m10 - m01},
            {m21 - m12, qAbs[1] * qAbs[1], m10 + m01, m02 + m20},
            {m02 - m20, m10 + m01, qAbs[2] * qAbs[2], m12 + m21},
            {m10 - m01, m20 + m02, m21 + m12, qAbs[3] * qAbs[3]},
    };
    int best = 0;
    for (int c = 1; c < 4; ++c) {
        if (qAbs[c] > qAbs[best]) best = c;
    }
    const float denom = 2.0f * std::max(qAbs[best], 0.1f);
    // Candidate rows are (r, i, j, k) = (w, x, y, z) ordered; the official
    // "rijk -> ijkr" reorder is out[c] = row[(c + 1) % 4].
    float out[4];
    for (int c = 0; c < 4; ++c) {
        out[c] = rows[best][((c + 1) % 4)] / denom;
    }
    // Standardize: w >= 0.
    if (out[3] < 0.f) {
        out[0] = -out[0];
        out[1] = -out[1];
        out[2] = -out[2];
        out[3] = -out[3];
    }
    outQ[0] = out[0];
    outQ[1] = out[1];
    outQ[2] = out[2];
    outQ[3] = out[3];
}

std::vector<std::pair<int, int>> splitWindows(int total,
                                              int effWindow,
                                              int effOverlap) {
    if (total <= 0) return {};
    if (effWindow >= total) return {{0, total}};
    std::vector<std::pair<int, int>> windows;
    const int step = std::max(effWindow - effOverlap, 1);
    for (int start = 0; start < total; start += step) {
        const int end = std::min(start + effWindow, total);
        if (end - start >= effOverlap || end == total) {
            windows.emplace_back(start, end);
        }
        if (end == total) break;
    }
    return windows;
}

WindowPlan planWindows(int total,
                       int windowSize,
                       int overlapSize,
                       int scaleFrames) {
    WindowPlan plan;
    if (total <= 0) return plan;
    const int ws = std::max(1, std::min(scaleFrames, total));
    const int kfInt = 1;  // the native engine streams every frame
    int effOverlap = std::max(overlapSize, 0);
    effOverlap = total > 1 ? std::min(effOverlap, total - 1) : 0;
    const int effWindow =
            std::min(ws + std::max(windowSize - ws, 0) * kfInt, total);
    plan.effWindow = effWindow;
    plan.effOverlap = effOverlap;
    plan.windows = splitWindows(total, effWindow, effOverlap);
    return plan;
}

Similarity pairwiseAlignment(const WindowData& prev,
                             const WindowData& curr,
                             int overlap) {
    Similarity sim;  // identity
    if (overlap <= 0) return sim;
    const int start = std::max(prev.frames - overlap, 0);
    const int eff =
            std::min(overlap, std::min(prev.frames - start, curr.frames));
    if (eff <= 0) return sim;

    const size_t plane = static_cast<size_t>(prev.width) * prev.height;
    // Depth-ratio scale: median over every finite overlap pixel with a
    // non-denormal target depth (all frames are keyframes at kf_interval=1).
    std::vector<float> ratios;
    for (int f = 0; f < eff; ++f) {
        const float* da =
                prev.depth.data() + static_cast<size_t>(start + f) * plane;
        const float* db = curr.depth.data() + static_cast<size_t>(f) * plane;
        for (size_t p = 0; p < plane; ++p) {
            const float a = da[p];
            const float b = db[p];
            if (std::isfinite(a) && std::isfinite(b) &&
                std::fabs(b) > kFinfoEps) {
                ratios.push_back(a / b);
            }
        }
    }
    if (!ratios.empty()) {
        sim.s = std::clamp(medianOf(ratios), kMinScale, kMaxScale);
    }

    // Anchor: the last overlap frame (paired keyframes at kf_interval=1).
    const int idxA = start + eff - 1;
    const int idxB = eff - 1;
    float Ra[9];
    float Rb[9];
    quatToMat(
            prev.poseEnc.data() + static_cast<size_t>(idxA) * kPoseEncSize + 3,
            Ra);
    quatToMat(
            curr.poseEnc.data() + static_cast<size_t>(idxB) * kPoseEncSize + 3,
            Rb);
    float RbT[9];
    // Transpose of Rb.
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            RbT[r * 3 + c] = Rb[c * 3 + r];
        }
    }
    mat3Mul(Ra, RbT, sim.R);  // Ra = R_ab @ Rb
    const float* ca =
            prev.poseEnc.data() + static_cast<size_t>(idxA) * kPoseEncSize;
    const float* cb =
            curr.poseEnc.data() + static_cast<size_t>(idxB) * kPoseEncSize;
    float Rcb[3];
    mat3Vec(sim.R, cb, Rcb);
    for (int r = 0; r < 3; ++r) {
        // ca = s * R_ab @ cb + t_ab  =>  t_ab = ca - s * R_ab @ cb
        sim.t[r] = ca[r] - sim.s * Rcb[r];
    }
    return sim;
}

void warpWindow(WindowData& win, const Similarity& sim) {
    const float* R = sim.R;
    float RT[9];
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            RT[r * 3 + c] = R[c * 3 + r];
        }
    }
    const size_t plane = static_cast<size_t>(win.width) * win.height;
    for (int f = 0; f < win.frames; ++f) {
        float* pe = win.poseEnc.data() + static_cast<size_t>(f) * kPoseEncSize;
        float rot[9];
        quatToMat(pe + 3, rot);
        float newRot[9];
        mat3Mul(R, rot, newRot);
        float newQ[4];
        matToQuat(newRot, newQ);
        float ctr[3] = {pe[0], pe[1], pe[2]};
        float wctr[3];
        mat3Vec(R, ctr, wctr);
        pe[0] = sim.s * wctr[0] + sim.t[0];
        pe[1] = sim.s * wctr[1] + sim.t[1];
        pe[2] = sim.s * wctr[2] + sim.t[2];
        pe[3] = newQ[0];
        pe[4] = newQ[1];
        pe[5] = newQ[2];
        pe[6] = newQ[3];

        float* d = win.depth.data() + static_cast<size_t>(f) * plane;
        for (size_t p = 0; p < plane; ++p) {
            d[p] *= sim.s;
        }

        // c2w sidecar: R' = R_c2w @ R^T, t' = s * t_c2w - R_c2w @ (R^T t).
        float* c2w = win.c2w.data() + static_cast<size_t>(f) * kC2WSize;
        const float rIn[9] = {c2w[0], c2w[1], c2w[2], c2w[4], c2w[5],
                              c2w[6], c2w[8], c2w[9], c2w[10]};
        float rOut[9];
        mat3Mul(rIn, RT, rOut);
        const float tin[3] = {c2w[3], c2w[7], c2w[11]};
        float RtT[3];
        mat3Vec(RT, sim.t, RtT);
        float rotTin[3];
        mat3Vec(rIn, RtT, rotTin);
        c2w[0] = rOut[0];
        c2w[1] = rOut[1];
        c2w[2] = rOut[2];
        c2w[3] = sim.s * tin[0] - rotTin[0];
        c2w[4] = rOut[3];
        c2w[5] = rOut[4];
        c2w[6] = rOut[5];
        c2w[7] = sim.s * tin[1] - rotTin[1];
        c2w[8] = rOut[6];
        c2w[9] = rOut[7];
        c2w[10] = rOut[8];
        c2w[11] = sim.s * tin[2] - rotTin[2];
    }
}

std::vector<std::pair<int, int>> contributionTable(
        const std::vector<WindowData>& windows, int overlap) {
    std::vector<std::pair<int, int>> table;
    if (windows.empty()) return table;
    if (windows.size() == 1) {
        for (int f = 0; f < windows.front().frames; ++f) {
            table.emplace_back(0, f);
        }
        return table;
    }
    for (size_t i = 0; i < windows.size(); ++i) {
        const bool isLast = (i + 1 == windows.size());
        const int end = isLast ? windows[i].frames
                               : std::max(windows[i].frames - overlap, 0);
        for (int f = 0; f < end; ++f) {
            table.emplace_back(static_cast<int>(i), f);
        }
    }
    return table;
}

WindowData stitchWindows(const std::vector<WindowData>& windows, int overlap) {
    WindowData merged;
    if (windows.empty()) return merged;
    if (windows.size() == 1) {
        merged = windows.front();
        return merged;
    }
    const WindowData& first = windows.front();
    merged.start = first.start;
    merged.width = first.width;
    merged.height = first.height;

    const std::vector<std::pair<int, int>> table =
            contributionTable(windows, overlap);
    const size_t totalFrames = table.size();
    merged.frames = static_cast<int>(totalFrames);
    const size_t plane = static_cast<size_t>(merged.width) * merged.height;
    merged.poseEnc.resize(totalFrames * kPoseEncSize);
    merged.depth.resize(totalFrames * plane);
    merged.depthConf.resize(totalFrames * plane);
    merged.c2w.resize(totalFrames * kC2WSize);
    merged.intrinsics.resize(totalFrames * kIntrinsicsSize);

    for (size_t k = 0; k < table.size(); ++k) {
        const WindowData& w = windows[static_cast<size_t>(table[k].first)];
        const size_t src = static_cast<size_t>(table[k].second);
        std::copy(
                w.poseEnc.begin() + static_cast<ptrdiff_t>(src * kPoseEncSize),
                w.poseEnc.begin() +
                        static_cast<ptrdiff_t>((src + 1) * kPoseEncSize),
                merged.poseEnc.begin() +
                        static_cast<ptrdiff_t>(k * kPoseEncSize));
        std::copy(w.depth.begin() + static_cast<ptrdiff_t>(src * plane),
                  w.depth.begin() + static_cast<ptrdiff_t>((src + 1) * plane),
                  merged.depth.begin() + static_cast<ptrdiff_t>(k * plane));
        std::copy(
                w.depthConf.begin() + static_cast<ptrdiff_t>(src * plane),
                w.depthConf.begin() + static_cast<ptrdiff_t>((src + 1) * plane),
                merged.depthConf.begin() + static_cast<ptrdiff_t>(k * plane));
        std::copy(w.c2w.begin() + static_cast<ptrdiff_t>(src * kC2WSize),
                  w.c2w.begin() + static_cast<ptrdiff_t>((src + 1) * kC2WSize),
                  merged.c2w.begin() + static_cast<ptrdiff_t>(k * kC2WSize));
        std::copy(w.intrinsics.begin() +
                          static_cast<ptrdiff_t>(src * kIntrinsicsSize),
                  w.intrinsics.begin() +
                          static_cast<ptrdiff_t>((src + 1) * kIntrinsicsSize),
                  merged.intrinsics.begin() +
                          static_cast<ptrdiff_t>(k * kIntrinsicsSize));
    }
    return merged;
}

}  // namespace lingbot_stitch
