// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tracker.hpp"

#if defined(QYOLO_WITH_OPENCV)
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 4)
#define QYOLO_WITH_OPENCV_SIFT \
    1  // SIFT moved into the main library in OpenCV 4.4
#endif
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <unordered_map>

namespace qyolo {
namespace track {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

bool is_angled(float angle) {
    return !std::isnan(angle) && angle >= -3.1515927f && angle <= 3.1515927f;
}

// ---- Kalman filter (kalman_filter.py: KalmanFilterXYAH / KalmanFilterXYWH)
// ----

enum class KFKind { XYAH, XYWH };

struct KalmanFilter {
    KFKind kind = KFKind::XYAH;
    double std_weight_position = 1.0 / 20;
    double std_weight_velocity = 1.0 / 160;

    // mean is [8] (box, box velocity), cov is [8][8] row-major, both float64 to
    // match the numpy float64 state upstream.
    void initiate(const double meas[4], double mean[8], double cov[64]) const {
        std::memset(mean, 0, sizeof(double) * 8);
        for (int i = 0; i < 4; i++) mean[i] = meas[i];
        std::memset(cov, 0, sizeof(double) * 64);
        double std[8];
        if (kind == KFKind::XYAH) {
            const double h = meas[3];
            const double v[] = {2 * std_weight_position * h,
                                2 * std_weight_position * h,
                                1e-2,
                                2 * std_weight_position * h,
                                10 * std_weight_velocity * h,
                                10 * std_weight_velocity * h,
                                1e-5,
                                10 * std_weight_velocity * h};
            std::copy(v, v + 8, std);
        } else {
            const double w = meas[2], h = meas[3];
            const double v[] = {
                    2 * std_weight_position * w,  2 * std_weight_position * h,
                    2 * std_weight_position * w,  2 * std_weight_position * h,
                    10 * std_weight_velocity * w, 10 * std_weight_velocity * h,
                    10 * std_weight_velocity * w, 10 * std_weight_velocity * h};
            std::copy(v, v + 8, std);
        }
        for (int i = 0; i < 8; i++) cov[i * 8 + i] = std[i] * std[i];
    }

    void predict(const double mean_in[8],
                 const double cov_in[64],
                 double mean_out[8],
                 double cov_out[64]) const {
        double sqr[8];
        if (kind == KFKind::XYAH) {
            const double h = mean_in[3];
            const double pos[] = {std_weight_position * h,
                                  std_weight_position * h, 1e-2,
                                  std_weight_position * h};
            const double vel[] = {std_weight_velocity * h,
                                  std_weight_velocity * h, 1e-5,
                                  std_weight_velocity * h};
            std::copy(pos, pos + 4, sqr);
            std::copy(vel, vel + 4, sqr + 4);
        } else {
            const double w = mean_in[2], h = mean_in[3];
            const double pos[] = {
                    std_weight_position * w, std_weight_position * h,
                    std_weight_position * w, std_weight_position * h};
            const double vel[] = {
                    std_weight_velocity * w, std_weight_velocity * h,
                    std_weight_velocity * w, std_weight_velocity * h};
            std::copy(pos, pos + 4, sqr);
            std::copy(vel, vel + 4, sqr + 4);
        }
        // F = [[I, I], [0, I]] with dt = 1: mean' = mean + mean[4:]; F cov F^T
        // adds the upper-right / lower-left cross blocks to the diagonal
        // blocks.
        for (int i = 0; i < 4; i++) mean_out[i] = mean_in[i] + mean_in[4 + i];
        for (int i = 4; i < 8; i++) mean_out[i] = mean_in[i];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++) {
                double v = cov_in[r * 8 + c];
                if (c < 4)
                    v += cov_in[r * 8 + c +
                                4];  // F * cov: col c <- col c + col c+4 (c<4)
                if (r < 4) v += cov_in[(r + 4) * 8 + c];
                if (r < 4 && c < 4) v += cov_in[(r + 4) * 8 + c + 4];
                cov_out[r * 8 + c] = v;
            }
        for (int i = 0; i < 8; i++) cov_out[i * 8 + i] += sqr[i] * sqr[i];
    }

    // Project to measurement space; confidence enables the NSA-Kalman noise
    // scaling (StrongSORT) used by TrackTrack.
    void project(const double mean[8],
                 const double cov[64],
                 double proj_mean[4],
                 double proj_cov[16],
                 double confidence) const {
        double std[4];
        if (kind == KFKind::XYAH) {
            const double h = mean[3];
            const double v[] = {std_weight_position * h,
                                std_weight_position * h, 1e-1,
                                std_weight_position * h};
            std::copy(v, v + 4, std);
        } else {
            const double w = mean[2], h = mean[3];
            const double v[] = {
                    std_weight_position * w, std_weight_position * h,
                    std_weight_position * w, std_weight_position * h};
            std::copy(v, v + 4, std);
        }
        double scale =
                std::isnan(confidence) ? 1.0 : std::max(1.0 - confidence, 0.05);
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) proj_cov[r * 4 + c] = cov[r * 8 + c];
        for (int i = 0; i < 4; i++)
            proj_cov[i * 4 + i] += std[i] * std[i] * scale;
        for (int i = 0; i < 4; i++) proj_mean[i] = mean[i];
    }

    void update(const double mean[8],
                const double cov[64],
                const double meas[4],
                double confidence,
                double mean_out[8],
                double cov_out[64]) const {
        double proj_mean[4], proj_cov[16];
        project(mean, cov, proj_mean, proj_cov, confidence);
        // K = H @ inv(P') where H = cov[:, :4] (8x4); solve P' Y = H^T, K =
        // Y^T.
        double k[32];  // 8x4 row-major: K[b][a] = k[b * 4 + a]
        for (int b = 0; b < 8;
             b++) {  // K row b: solve P' x = (cov row b, first 4 entries)
            // Gauss-Jordan with partial pivoting on a copy of [P' | v_b].
            double m[4][5];
            for (int r = 0; r < 4; r++) {
                for (int c = 0; c < 4; c++) m[r][c] = proj_cov[r * 4 + c];
                m[r][4] = cov[b * 8 + r];
            }
            for (int col = 0; col < 4; col++) {
                int piv = col;
                for (int r = col + 1; r < 4; r++)
                    if (std::fabs(m[r][col]) > std::fabs(m[piv][col])) piv = r;
                if (piv != col)
                    for (int c = 0; c < 5; c++) std::swap(m[col][c], m[piv][c]);
                const double d = m[col][col];
                if (std::fabs(d) < 1e-300) continue;
                for (int c = 0; c < 5; c++) m[col][c] /= d;
                for (int r = 0; r < 4; r++) {
                    if (r == col) continue;
                    const double f = m[r][col];
                    if (f == 0.0) continue;
                    for (int c = 0; c < 5; c++) m[r][c] -= f * m[col][c];
                }
            }
            for (int r = 0; r < 4; r++)
                k[b * 4 + r] = m[r][4];  // K[b][r] = (P'^-1 H^T)^T[b][r]
        }
        double innovation[4];
        for (int i = 0; i < 4; i++) innovation[i] = meas[i] - proj_mean[i];
        for (int i = 0; i < 8; i++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += k[i * 4 + j] * innovation[j];
            mean_out[i] = mean[i] + acc;
        }
        // cov - K P' K^T
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++) {
                double acc = 0;
                for (int j = 0; j < 4; j++) {
                    double kcj = 0;
                    for (int l = 0; l < 4; l++)
                        kcj += k[r * 4 + l] * proj_cov[l * 4 + j];
                    acc += kcj * k[c * 4 + j];
                }
                cov_out[r * 8 + c] = cov[r * 8 + c] - acc;
            }
    }
};

// tlwh <-> center/xyah measurement conversions (byte_tracker.py STrack /
// bot_sort.py BOTrack).
void tlwh_to_meas(const float tlwh[4], KFKind kind, double meas[4]) {
    const double cx = tlwh[0] + tlwh[2] / 2.0, cy = tlwh[1] + tlwh[3] / 2.0;
    if (kind == KFKind::XYAH) {
        meas[0] = cx;
        meas[1] = cy;
        meas[2] = tlwh[2] / tlwh[3];
        meas[3] = tlwh[3];
    } else {
        meas[0] = cx;
        meas[1] = cy;
        meas[2] = tlwh[2];
        meas[3] = tlwh[3];
    }
}

// ---- matching (matching.py + metrics.bbox_ioa / batch_probiou) ----

struct Box {  // xyxy
    float x1, y1, x2, y2;
};

float box_iou_xyxy(const Box& a, const Box& b) {
    const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1), h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1),
                area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

// Intersection over b's area (bbox_ioa default), used by FastTracker coverage
// and TAI NMS.
float box_ioa(const Box& a, const Box& b) {
    const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return area_b > 0 ? inter / area_b : 0.0f;
}

// probiou (metrics._get_covariance_matrix + probiou) between two xywhr boxes.
float probiou(float cx1,
              float cy1,
              float w1,
              float h1,
              float r1,
              float cx2,
              float cy2,
              float w2,
              float h2,
              float r2) {
    auto cov_terms = [](float w, float h, float r, double& a, double& b,
                        double& c) {
        const double wh = w / 2.0, hh = h / 2.0;
        const double co = std::cos((double)r), si = std::sin((double)r);
        a = (wh * co) * (wh * co) + (hh * si) * (hh * si);
        b = (wh * si) * (wh * si) + (hh * co) * (hh * co);
        c = wh * hh * std::sin(2.0 * r);
    };
    double a1, b1, c1, a2, b2, c2;
    cov_terms(w1, h1, r1, a1, b1, c1);
    cov_terms(w2, h2, r2, a2, b2, c2);
    const double dx = (double)cx1 - cx2, dy = (double)cy1 - cy2;
    const double sa = a1 + a2, sb = b1 + b2, sc = c1 + c2;
    const double denom = sa * sb - sc * sc;
    const double t1 = ((sa * dy * dy + sb * dx * dx) / (denom + 1e-7)) * 0.25;
    const double t2 = ((sc * (cx2 - cx1) * (cy1 - cy2)) / (denom + 1e-7)) * 0.5;
    const double t3 =
            std::log(
                    denom / (4.0 * std::sqrt(std::max(0.0, a1 * b1 - c1 * c1) *
                                             std::max(0.0, a2 * b2 - c2 * c2)) +
                             1e-7) +
                    1e-7) *
            0.5;
    const double bd = std::clamp(t1 + t2 + t3, 1e-7, 100.0);
    const double hd = std::sqrt(1.0 - std::exp(-bd) + 1e-7);
    return (float)(1.0 - hd);
}

Box det_box_xyxy(const TrackDet& d) {
    return {d.cx - d.w / 2, d.cy - d.h / 2, d.cx + d.w / 2, d.cy + d.h / 2};
}

// iou_distance: 1 - IoU (axis-aligned) or 1 - probiou (angled), matching
// matching.iou_distance's xyxy/xywha switch.
std::vector<float> iou_distance(const std::vector<TrackDet>& a,
                                const std::vector<TrackDet>& b) {
    const size_t n = a.size(), m = b.size();
    std::vector<float> cost(n * m, 0.0f);
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < m; j++) {
            if (is_angled(a[i].angle) && is_angled(b[j].angle)) {
                cost[i * m + j] =
                        1.0f - probiou(a[i].cx, a[i].cy, a[i].w, a[i].h,
                                       a[i].angle, b[j].cx, b[j].cy, b[j].w,
                                       b[j].h, b[j].angle);
            } else {
                cost[i * m + j] = 1.0f - box_iou_xyxy(det_box_xyxy(a[i]),
                                                      det_box_xyxy(b[j]));
            }
        }
    return cost;
}

std::vector<float> fuse_score(const std::vector<float>& cost,
                              int n,
                              const std::vector<TrackDet>& dets) {
    // fuse_sim = iou_sim * det_scores; cost = 1 - fuse_sim
    // (matching.fuse_score).
    std::vector<float> out(cost);
    for (int i = 0; i < n; i++)
        for (size_t j = 0; j < dets.size(); j++)
            out[(size_t)i * dets.size() + j] =
                    1.0f -
                    (1.0f - cost[(size_t)i * dets.size() + j]) * dets[j].score;
    return out;
}

// ---- linear assignment (matching.linear_assignment / lap.lapjv semantics)
// ----
//
// Rectangular min-cost assignment via shortest augmenting paths with dual
// variables (the Jonker-Volgenant family lapjv belongs to). The shorter side is
// fully assigned; pairs above `thresh` are dropped, matching lapjv's
// extend_cost + cost_limit behavior.
struct Assignment {
    std::vector<std::pair<int, int>> matches;  // (a-index, b-index)
    std::vector<int> unmatched_a, unmatched_b;
};

Assignment linear_assignment(const std::vector<float>& cost,
                             int n,
                             int m,
                             float thresh) {
    Assignment res;
    res.unmatched_a.reserve(n);
    res.unmatched_b.reserve(m);
    if (n == 0 || m == 0) {
        for (int i = 0; i < n; i++) res.unmatched_a.push_back(i);
        for (int j = 0; j < m; j++) res.unmatched_b.push_back(j);
        return res;
    }
    // Work orientation: rows = the side to fully assign, so transpose when n >
    // m. lapjv's cost_limit forbids above-threshold entries during the solve
    // (they may only be "assigned" when no allowed column remains, which drops
    // the pair); mirroring that here preserves every below-threshold match that
    // the unconstrained optimum would otherwise sacrifice.
    const double forbidden = 1e9;
    const bool transposed = n > m;
    const int rows = transposed ? m : n, cols = transposed ? n : m;
    std::vector<double> a((size_t)rows * cols);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) {
            // cost is [n x m] row-major; the transposed view reads cost[c][r]
            // with the original row stride m.
            const double raw = transposed ? cost[(size_t)c * m + r]
                                          : cost[(size_t)r * m + c];
            a[(size_t)r * cols + c] = raw > thresh ? forbidden : raw;
        }
    // 1-based shortest-augmenting-path Hungarian (e-maxx formulation): the
    // virtual unmatched column is index 0, real columns are 1..cols.
    std::vector<double> u(rows + 1, 0), v(cols + 1, 0);
    std::vector<int> p(cols + 1, 0), way(cols + 1, 0);
    for (int i = 1; i <= rows; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(cols + 1, kInf);
        std::vector<char> used(cols + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = kInf;
            int j1 = -1;
            for (int j = 1; j <= cols; j++)
                if (!used[j]) {
                    const double cur =
                            a[(size_t)(i0 - 1) * cols + (j - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            if (j1 < 0) break;
            for (int j = 0; j <= cols; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<char> row_matched(rows, 0), col_matched(cols, 0);
    for (int j = 1; j <= cols; j++)
        if (p[j] != 0) {
            const int r = p[j] - 1, c = j - 1;
            const double cval = a[(size_t)r * cols + c];
            if (cval <= thresh) {
                res.matches.emplace_back(transposed ? c : r,
                                         transposed ? r : c);
                row_matched[r] = 1;
                col_matched[c] = 1;
            }
        }
    if (!transposed) {
        for (int r = 0; r < rows; r++)
            if (!row_matched[r]) res.unmatched_a.push_back(r);
        for (int c = 0; c < cols; c++)
            if (!col_matched[c]) res.unmatched_b.push_back(c);
    } else {
        for (int r = 0; r < rows; r++)
            if (!row_matched[r]) res.unmatched_b.push_back(r);
        for (int c = 0; c < cols; c++)
            if (!col_matched[c]) res.unmatched_a.push_back(c);
    }
    return res;
}

// ---- track state (basetrack.py + byte_tracker.STrack + subclasses) ----

enum class TrackState { New = 0, Tracked = 1, Lost = 2, Removed = 3 };

int g_track_count =
        0;  // BaseTrack._count (shared across tracker classes upstream)

struct STrack {
    // BaseTrack
    int track_id = 0;
    bool is_activated = false;
    TrackState state = TrackState::New;
    int start_frame = 0, frame_id = 0;
    int tracklet_len = 0;
    // STrack
    float tlwh_det[4] = {0, 0, 0,
                         0};  // detection _tlwh (mean is None upstream)
    float score = 0;
    int cls = 0, idx = 0;
    float angle = -10.0f;
    bool angled() const { return is_angled(angle); }
    // Appearance features (BoT-SORT family only; empty when ReID is off or
    // the stream carries no features). Official smooth_feature contract:
    // curr_feat is the normalized latest update, smooth_feat the normalized
    // EMA state; both are the matching.embedding_distance inputs.
    std::vector<float> curr_feat, smooth_feat;
    // DeepOCSortTrack knobs (only meaningful when deep_reid is set).
    bool deep_reid = false;
    float alpha_fixed_emb = 0.95f;
    // det_thresh comes from args.track_high_thresh upstream
    // (deep_oc_sort.py init_track: det_thresh=self.args.track_high_thresh).
    float det_thresh = 0.25f;

    // EMA blend factor for the next feature update. BOTrack: fixed 0.9;
    // DeepOCSortTrack: confidence-adaptive around alpha_fixed_emb with
    // det_thresh from the config (low-trust detections keep the existing
    // state); TTSTrack: score-adaptive beta = 0.95 + 0.05 * (1 - score).
    virtual float feature_alpha() const {
        if (deep_reid) {
            if (score <= det_thresh) return 1.0f;
            const float trust =
                    (score - det_thresh) / std::max(1.0f - det_thresh, 1e-9f);
            return alpha_fixed_emb + (1.0f - alpha_fixed_emb) * (1.0f - trust);
        }
        return 0.9f;
    }

    // Official smooth_feature (trackers/utils/reid.py): L2-normalize the
    // feature and blend it into the EMA state; a zero-norm feature carries
    // no appearance information and keeps the current state.
    void update_features(const std::vector<float>& feat) {
        if (feat.empty()) return;
        float norm = 0.0f;
        for (float v : feat) norm += v * v;
        norm = std::sqrt(norm);
        if (norm < 1e-12f) return;
        curr_feat.resize(feat.size());
        for (size_t i = 0; i < feat.size(); i++) curr_feat[i] = feat[i] / norm;
        if (smooth_feat.empty()) {
            smooth_feat = curr_feat;
            return;
        }
        const float a = feature_alpha();
        for (size_t i = 0; i < smooth_feat.size(); i++)
            smooth_feat[i] = a * smooth_feat[i] + (1.0f - a) * curr_feat[i];
        float sn = 0.0f;
        for (float v : smooth_feat) sn += v * v;
        sn = std::sqrt(sn);
        if (sn > 0.0f)
            for (float& v : smooth_feat) v /= sn;
    }
    // Kalman state (mean is None until activate upstream)
    bool has_mean = false;
    double mean[8] = {0}, cov[64] = {0};
    KFKind kind = KFKind::XYAH;
    const KalmanFilter* kf = nullptr;

    virtual ~STrack() = default;
    STrack(const TrackDet& d, KFKind k)
        : score(d.score), cls(d.class_id), idx(d.idx), angle(d.angle), kind(k) {
        tlwh_det[0] = d.cx - d.w / 2;
        tlwh_det[1] = d.cy - d.h / 2;
        tlwh_det[2] = d.w;
        tlwh_det[3] = d.h;
    }

    int end_frame() const { return frame_id; }
    static int next_id() { return ++g_track_count; }
    static void reset_id() { g_track_count = 0; }
    void mark_lost() { state = TrackState::Lost; }
    void mark_removed() { state = TrackState::Removed; }

    // convert_coords: tlwh -> XYAH (cx, cy, w/h, h) or XYWH (cx, cy, w, h).
    void convert_coords(const float tlwh[4], double meas[4]) const {
        tlwh_to_meas(tlwh, kind, meas);
    }

    // tlwh from the current Kalman state (byte_tracker STrack.tlwh / bot_sort
    // BOTrack.tlwh).
    void state_tlwh(float out[4]) const {
        if (!has_mean) {
            std::copy(tlwh_det, tlwh_det + 4, out);
            return;
        }
        if (kind == KFKind::XYAH) {
            const float a = (float)mean[2], h = (float)mean[3];
            const float w = a * h;
            out[0] = (float)mean[0] - w / 2;
            out[1] = (float)mean[1] - h / 2;
            out[2] = w;
            out[3] = h;
        } else {
            const float w = (float)mean[2], h = (float)mean[3];
            out[0] = (float)mean[0] - w / 2;
            out[1] = (float)mean[1] - h / 2;
            out[2] = w;
            out[3] = h;
        }
    }

    void state_xyxy(float out[4]) const {
        float t[4];
        state_tlwh(t);
        out[0] = t[0];
        out[1] = t[1];
        out[2] = t[0] + t[2];
        out[3] = t[1] + t[3];
    }

    void state_xywh(float out[4]) const {
        float t[4];
        state_tlwh(t);
        out[0] = t[0] + t[2] / 2;
        out[1] = t[1] + t[3] / 2;
        out[2] = t[2];
        out[3] = t[3];
    }

    // predict(): zero the appropriate velocity dims for non-Tracked state —
    // index 7 (vh) for XYAH, indices 6 and 7 (vw, vh) for XYWH.
    void predict() {
        if (!has_mean) return;
        double m[8];
        std::copy(mean, mean + 8, m);
        if (state != TrackState::Tracked) {
            m[7] = 0;
            if (kind == KFKind::XYWH) m[6] = 0;
        }
        kf->predict(m, cov, mean, cov);
        has_mean = true;
    }

    virtual void activate(const KalmanFilter& f, int fid) {
        kf = &f;
        track_id = next_id();
        double meas[4];
        convert_coords(tlwh_det, meas);
        f.initiate(meas, mean, cov);
        has_mean = true;
        tracklet_len = 0;
        state = TrackState::Tracked;
        if (fid == 1) is_activated = true;
        frame_id = start_frame = fid;
    }

    virtual void re_activate(const STrack& nw, int fid, bool new_id = false) {
        float t[4];
        nw.state_tlwh(t);
        double meas[4];
        convert_coords(t, meas);
        kf->update(mean, cov, meas, kNaN, mean,
                   cov);  // no confidence on the BYTETracker paths
        has_mean = true;
        tracklet_len = 0;
        state = TrackState::Tracked;
        is_activated = true;
        frame_id = fid;
        if (new_id) track_id = next_id();
        score = nw.score;
        cls = nw.cls;
        angle = nw.angle;
        idx = nw.idx;
        // BOTrack.re_activate: refresh the appearance state from the new
        // detection's current feature.
        if (!nw.curr_feat.empty()) update_features(nw.curr_feat);
    }

    virtual void update(const STrack& nw, int fid) {
        frame_id = fid;
        tracklet_len++;
        float t[4];
        nw.state_tlwh(t);
        double meas[4];
        convert_coords(t, meas);
        kf->update(mean, cov, meas, kNaN, mean, cov);
        has_mean = true;
        state = TrackState::Tracked;
        is_activated = true;
        score = nw.score;
        cls = nw.cls;
        angle = nw.angle;
        idx = nw.idx;
        // BOTrack.update: refresh the appearance state from the new
        // detection's current feature.
        if (!nw.curr_feat.empty()) update_features(nw.curr_feat);
    }
};

// matching.embedding_distance: cosine distance 1 - dot on the normalized
// feature rows (track side smooth_feat, detection side curr_feat;
// a freshly created detection has smooth_feat == curr_feat). Pairs with a
// missing side yield 1.0 — upstream stacks a zero placeholder and forces the
// pair to 2.0; under the /2 and the appearance/proximity gates both constants
// can never lower the motion cost, so the fusion is the exact no-op the
// upstream "missing feature" behavior demands.
std::vector<float> embedding_distance(const std::vector<STrack*>& tracks,
                                      const std::vector<STrack*>& dets) {
    std::vector<float> out(tracks.size() * dets.size(), 1.0f);
    const int m = (int)dets.size();
    for (size_t i = 0; i < tracks.size(); i++) {
        const std::vector<float>& tf = tracks[i]->smooth_feat;
        if (tf.empty()) continue;
        for (int j = 0; j < m; j++) {
            const std::vector<float>& df = dets[j]->curr_feat;
            if (df.empty() || df.size() != tf.size()) continue;
            float dot = 0.0f;
            for (size_t k = 0; k < tf.size(); k++) dot += tf[k] * df[k];
            out[i * m + j] = 1.0f - dot;
        }
    }
    return out;
}

// OCSortTrack (oc_sort.py): observation-centric state for ORU/OCM/OCR.
struct OCSortTrack : STrack {
    float last_observation[4] = {-1, -1, -1, -1};  // xyxy
    std::unordered_map<int, std::array<float, 4>> observations;
    bool has_velocity = false;
    float velocity[2] = {0, 0};
    int delta_t = 3;
    bool has_saved = false;
    double saved_mean[8] = {0}, saved_cov[64] = {0};

    OCSortTrack(const TrackDet& d, int dt)
        : STrack(d, KFKind::XYAH), delta_t(dt) {}

    static void xyxy_center(const float b[4], float c[2]) {
        c[0] = (b[0] + b[2]) / 2;
        c[1] = (b[1] + b[3]) / 2;
    }

    void record_observation(const float xyxy[4], int fid) {
        std::copy(xyxy, xyxy + 4, last_observation);
        observations[fid] = {xyxy[0], xyxy[1], xyxy[2], xyxy[3]};
        const size_t max_keep = (size_t)delta_t + 2;
        if (observations.size() > max_keep) {
            std::vector<int> frames;
            frames.reserve(observations.size());
            for (const auto& [f, _] : observations) frames.push_back(f);
            std::sort(frames.begin(), frames.end());
            const size_t drop = observations.size() - max_keep;
            for (size_t i = 0; i < drop; i++) observations.erase(frames[i]);
        }
    }

    void compute_velocity() {
        if (observations.size() < 2) {
            has_velocity = false;
            return;
        }
        int current_frame = std::numeric_limits<int>::min();
        for (const auto& [f, _] : observations)
            current_frame = std::max(current_frame, f);
        float cur[2];
        xyxy_center(observations[current_frame].data(), cur);
        // Most recent observation at least delta_t frames before current.
        std::vector<int> frames;
        frames.reserve(observations.size());
        for (const auto& [f, _] : observations) frames.push_back(f);
        std::sort(frames.begin(), frames.end(), std::greater<int>());
        const float* prev_obs = nullptr;
        for (int f : frames)
            if (f < current_frame - delta_t + 1) {
                prev_obs = observations[f].data();
                break;
            }
        // Fallback: the earliest observation if nothing is delta_t frames back.
        if (!prev_obs) {
            int earliest = std::numeric_limits<int>::max();
            for (int f : frames) earliest = std::min(earliest, f);
            if (earliest == current_frame) {
                has_velocity = false;
                return;
            }
            prev_obs = observations[earliest].data();
        }
        float pc[2];
        xyxy_center(prev_obs, pc);
        const float dx = cur[0] - pc[0], dy = cur[1] - pc[1];
        const float norm = std::sqrt(dx * dx + dy * dy);
        if (norm < 1e-6f) {
            velocity[0] = velocity[1] = 0;
        } else {
            velocity[0] = dx / norm;
            velocity[1] = dy / norm;
        }
        has_velocity = true;
    }

    void save_state() {
        std::copy(mean, mean + 8, saved_mean);
        std::copy(cov, cov + 64, saved_cov);
        has_saved = true;
    }

    void activate(const KalmanFilter& f, int fid) override {
        STrack::activate(f, fid);
        float b[4];
        state_xyxy(b);  // detection-space precision at activation
        record_observation(b, fid);
        save_state();
    }

    void update(const STrack& nw, int fid) override {
        float b[4];
        nw.state_xyxy(b);
        record_observation(b, fid);
        STrack::update(nw, fid);
        save_state();
        compute_velocity();
    }

    void re_activate(const STrack& nw, int fid, bool new_id = false) override {
        float b[4];
        nw.state_xyxy(b);
        record_observation(b, fid);
        STrack::re_activate(nw, fid, new_id);
        save_state();
        compute_velocity();
    }

    // ORU: replay predict-updates on virtual observations across the gap.
    void apply_oru(const float new_xyxy[4], int current_fid) {
        if (!has_saved || observations.empty()) return;
        int last_frame = -1;
        for (const auto& [f, _] : observations)
            last_frame = std::max(last_frame, f);
        const int gap = current_fid - last_frame;
        if (gap <= 1) return;
        std::copy(saved_mean, saved_mean + 8, mean);
        std::copy(saved_cov, saved_cov + 64, cov);
        const float* last_obs = observations[last_frame].data();
        for (int t = 1; t < gap; t++) {
            const float alpha = (float)t / gap;
            const float virt[4] = {
                    (1 - alpha) * last_obs[0] + alpha * new_xyxy[0],
                    (1 - alpha) * last_obs[1] + alpha * new_xyxy[1],
                    (1 - alpha) * last_obs[2] + alpha * new_xyxy[2],
                    (1 - alpha) * last_obs[3] + alpha * new_xyxy[3]};
            const float ltwh[4] = {virt[0], virt[1], virt[2] - virt[0],
                                   virt[3] - virt[1]};
            double meas[4], m[8], c[64];
            tlwh_to_meas(ltwh, KFKind::XYAH, meas);
            kf->predict(mean, cov, m, c);
            std::copy(m, m + 8, mean);
            std::copy(c, c + 64, cov);
            kf->update(mean, cov, meas, kNaN, mean, cov);
        }
        double m[8], c[64];
        kf->predict(mean, cov, m, c);
        std::copy(m, m + 8, mean);
        std::copy(c, c + 64, cov);
    }
};

// FastSTrack (fast_tracker.py): bounded Kalman history + occlusion bookkeeping.
struct FastSTrack : STrack {
    std::deque<std::pair<std::array<double, 8>, std::array<double, 64>>>
            mean_history;
    size_t history_cap = 16;
    int not_matched = 0;
    bool is_occluded = false;
    int occluded_len = 0;
    int last_occluded_frame = -1;
    bool was_recently_occluded = false;

    FastSTrack(const TrackDet& d, size_t history_len)
        : STrack(d, KFKind::XYAH), history_cap(history_len) {}

    void push_history() {
        if (!has_mean) return;
        std::pair<std::array<double, 8>, std::array<double, 64>> snap;
        std::copy(mean, mean + 8, snap.first.begin());
        std::copy(cov, cov + 64, snap.second.begin());
        if (mean_history.size() >= history_cap) mean_history.pop_front();
        mean_history.push_back(std::move(snap));
    }

    void activate(const KalmanFilter& f, int fid) override {
        STrack::activate(f, fid);
        push_history();
    }

    void re_activate(const STrack& nw, int fid, bool new_id = false) override {
        STrack::re_activate(nw, fid, new_id);
        is_occluded = false;
        occluded_len = 0;
        not_matched = 0;
        was_recently_occluded = false;
        last_occluded_frame = -1;
        push_history();
    }

    void update(const STrack& nw, int fid) override {
        STrack::update(nw, fid);
        push_history();
    }
};

// TTSTrack (track_tracker.py): XYWH state, NSA-Kalman updates, corner velocity.
struct TTSTrack : STrack {
    static constexpr int kCornerDx[4] = {0, 0, 2, 2};
    static constexpr int kCornerDy[4] = {1, 3, 1, 3};
    float prev_score = 0;
    float velocity[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    std::deque<std::pair<int, std::array<float, 4>>>
            history;  // (frame_id, xyxy), maxlen delta_t + 1
    int delta_t = 3;
    int min_track_len = 3;

    TTSTrack(const TrackDet& d, int dt, int mtl)
        : STrack(d, KFKind::XYWH), delta_t(dt), min_track_len(mtl) {}

    // TTSTrack.update_features: score-adaptive beta = _alpha + (1 - _alpha)
    // * (1 - score) with _alpha = 0.95 (track_tracker.py TTSTrack._alpha).
    float feature_alpha() const override {
        return 0.95f + 0.05f * (1.0f - score);
    }

    void push_history(int fid) {
        float b[4];
        state_xyxy(b);
        if (history.size() >= (size_t)delta_t + 1) history.pop_front();
        history.push_back({fid, {b[0], b[1], b[2], b[3]}});
    }

    void history_box(int fid, int dt, float out[4]) const {
        const int target = fid - dt;
        for (const auto& [f, b] : history)
            if (f == target) {
                std::copy(b.begin(), b.end(), out);
                return;
            }
        if (!history.empty()) {
            std::copy(history.back().second.begin(),
                      history.back().second.end(), out);
            return;
        }
        state_xyxy(out);
    }

    void activate(const KalmanFilter& f, int fid) override {
        kf = &f;
        track_id = next_id();
        double meas[4];
        convert_coords(tlwh_det, meas);
        f.initiate(meas, mean, cov);
        has_mean = true;
        push_history(fid);
        tracklet_len = 0;
        // track_tracker.py TTSTrack.activate: min_track_len <= 1 confirms
        // immediately, otherwise the track stays New until promoted in
        // update().
        state = min_track_len <= 1 ? TrackState::Tracked : TrackState::New;
        is_activated = fid == 1 || state == TrackState::Tracked;
        frame_id = start_frame = fid;
    }

    void re_activate(const STrack& nw, int fid, bool new_id = false) override {
        prev_score = score;
        float t[4];
        nw.state_tlwh(t);
        double meas[4];
        convert_coords(t, meas);
        kf->update(mean, cov, meas, nw.score, mean, cov);  // NSA-Kalman
        has_mean = true;
        push_history(fid);
        score = nw.score;  // set before update_features so the EMA weight
                           // uses the current confidence (upstream order)
        if (!nw.curr_feat.empty()) update_features(nw.curr_feat);
        tracklet_len = 0;
        state = TrackState::Tracked;
        is_activated = true;
        frame_id = fid;
        if (new_id) track_id = next_id();
        cls = nw.cls;
        angle = nw.angle;
        idx = nw.idx;
    }

    void update(const STrack& nw, int fid) override {
        frame_id = fid;
        tracklet_len++;
        prev_score = score;
        float t[4];
        nw.state_tlwh(t);
        double meas[4];
        convert_coords(t, meas);
        kf->update(mean, cov, meas, nw.score, mean, cov);  // NSA-Kalman
        has_mean = true;
        float nb[4];
        nw.state_xyxy(nb);
        if (history.size() >= (size_t)delta_t + 1) history.pop_front();
        history.push_back({fid, {nb[0], nb[1], nb[2], nb[3]}});
        float velocity_acc[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        for (int dt = 1; dt <= delta_t; dt++) {
            float hb[4];
            history_box(fid, dt, hb);
            for (int k = 0; k < 4; k++) {
                const float dx = nb[kCornerDx[k]] - hb[kCornerDx[k]];
                const float dy = nb[kCornerDy[k]] - hb[kCornerDy[k]];
                const float norm = std::sqrt(dx * dx + dy * dy) + 1e-5f;
                velocity_acc[k][0] += (dx / norm) / dt;
                velocity_acc[k][1] += (dy / norm) / dt;
            }
        }
        for (int k = 0; k < 4; k++) {
            velocity[k][0] = velocity_acc[k][0] / delta_t;
            velocity[k][1] = velocity_acc[k][1] / delta_t;
        }
        score = nw.score;  // set before update_features so the EMA weight
                           // uses the current confidence (upstream order)
        if (!nw.curr_feat.empty()) update_features(nw.curr_feat);
        // track_tracker.py: confirm on the (tracklet_len + 1)-th history
        // entry reaching min_track_len — i.e. one update earlier than
        // tracklet_len >= min_track_len.
        if (state == TrackState::Tracked || tracklet_len + 1 >= min_track_len) {
            state = TrackState::Tracked;
            is_activated = true;
        }
        cls = nw.cls;
        angle = nw.angle;
        idx = nw.idx;
    }
};

// ---- pool helpers (utils/stracks.py) ----

void joint_stracks(std::vector<STrack*>& out, const std::vector<STrack*>& add) {
    for (STrack* t : add)
        if (std::find(out.begin(), out.end(), t) == out.end()) out.push_back(t);
}

// joint_stracks by track_id: entries in `a` win on id collisions (upstream keys
// on id).
void joint_by_id(std::vector<STrack*>& a, const std::vector<STrack*>& b) {
    for (STrack* t : b) {
        bool dup = false;
        for (STrack* s : a)
            if (s->track_id == t->track_id) {
                dup = true;
                break;
            }
        if (!dup) a.push_back(t);
    }
}

void sub_stracks(std::vector<STrack*>& a, const std::vector<STrack*>& b) {
    std::vector<STrack*> keep;
    keep.reserve(a.size());
    for (STrack* t : a) {
        bool found = false;
        for (STrack* s : b)
            if (s->track_id == t->track_id) {
                found = true;
                break;
            }
        if (!found) keep.push_back(t);
    }
    a.swap(keep);
}

void remove_duplicate_stracks(std::vector<STrack*>& a,
                              std::vector<STrack*>& b) {
    const float dup_thresh = 0.15f;
    std::vector<char> dupa(a.size(), 0), dupb(b.size(), 0);
    for (size_t p = 0; p < a.size(); p++) {
        float pa[4];
        a[p]->state_xyxy(pa);
        for (size_t q = 0; q < b.size(); q++) {
            float pb[4];
            b[q]->state_xyxy(pb);
            if (1.0f - box_iou_xyxy({pa[0], pa[1], pa[2], pa[3]},
                                    {pb[0], pb[1], pb[2], pb[3]}) >=
                dup_thresh)
                continue;
            const int timep = a[p]->frame_id - a[p]->start_frame;
            const int timeq = b[q]->frame_id - b[q]->start_frame;
            if (timep > timeq)
                dupb[q] = 1;
            else
                dupa[p] = 1;
        }
    }
    auto filter = [](std::vector<STrack*>& v, const std::vector<char>& dup) {
        std::vector<STrack*> keep;
        keep.reserve(v.size());
        for (size_t i = 0; i < v.size(); i++)
            if (!dup[i]) keep.push_back(v[i]);
        v.swap(keep);
    };
    filter(a, dupa);
    filter(b, dupb);
}

void merge_track_pools(std::vector<STrack*>& tracked,
                       std::vector<STrack*>& lost,
                       std::vector<STrack*>& removed,
                       const std::vector<STrack*>& activated,
                       const std::vector<STrack*>& refind,
                       const std::vector<STrack*>& lost_in,
                       const std::vector<STrack*>& removed_in,
                       int removed_buffer = 1000) {
    tracked.erase(std::remove_if(tracked.begin(), tracked.end(),
                                 [](const STrack* t) {
                                     return t->state != TrackState::Tracked;
                                 }),
                  tracked.end());
    joint_stracks(tracked, activated);
    joint_stracks(tracked, refind);
    sub_stracks(lost, tracked);
    for (STrack* t : lost_in) lost.push_back(t);
    sub_stracks(lost, removed);
    remove_duplicate_stracks(tracked, lost);
    for (STrack* t : removed_in) removed.push_back(t);
    if ((int)removed.size() > removed_buffer)
        removed.erase(removed.begin(), removed.end() - removed_buffer);
}

// Standard multi_gmc: rotate all four (dim, velocity) pairs block-diagonally
// (XYWH spatial state) and translate position (utils/stracks.multi_gmc).
void multi_gmc(const std::vector<STrack*>& stracks, const float H[6]) {
    const double R[2][2] = {{H[0], H[1]}, {H[3], H[4]}}, t[2] = {H[2], H[5]};
    for (STrack* st : stracks) {
        double m[8], c[64];
        for (int pair = 0; pair < 4; pair++) {
            const double x = st->mean[pair * 2], y = st->mean[pair * 2 + 1];
            m[pair * 2] = R[0][0] * x + R[0][1] * y;
            m[pair * 2 + 1] = R[1][0] * x + R[1][1] * y;
        }
        m[0] += t[0];
        m[1] += t[1];
        // cov' = R8 cov R8^T: each 2x2 pair block transforms as R C R^T.
        for (int pr = 0; pr < 4; pr++)
            for (int pc = 0; pc < 4; pc++)
                for (int ir = 0; ir < 2; ir++)
                    for (int ic = 0; ic < 2; ic++) {
                        double acc = 0;
                        for (int l = 0; l < 2; l++)
                            for (int k = 0; k < 2; k++)
                                acc += R[ir][l] *
                                       st->cov[((pr * 2 + l) * 8) +
                                               (pc * 2 + k)] *
                                       R[ic][k];
                        c[((pr * 2 + ir) * 8) + (pc * 2 + ic)] = acc;
                    }
        std::copy(m, m + 8, st->mean);
        std::copy(c, c + 64, st->cov);
    }
}

// ---- GMC (utils/gmc.py) ----
// With OpenCV (QYOLO_WITH_OPENCV) every method is the same cv2 call as the
// Python implementation; without it a self-contained sparseOptFlow fallback
// (Shi-Tomasi + pyramidal LK + RANSAC similarity) is used and orb/sift/ecc
// are rejected at create_tracker.

#if !defined(QYOLO_WITH_OPENCV)

struct GrayImage {
    int w = 0, h = 0;
    std::vector<uint8_t> px;
};

GrayImage to_gray_downscaled(const RgbFrame& frame, int downscale) {
    // cv2.cvtColor BGR2GRAY weights; our frame is RGB8 — the luma result is
    // identical.
    const int dw = frame.w / downscale, dh = frame.h / downscale;
    GrayImage g{dw, dh, std::vector<uint8_t>((size_t)dw * dh)};
    for (int y = 0; y < dh; y++)
        for (int x = 0; x < dw; x++) {
            // Box-average the downscale block (approximates cv2.resize
            // INTER_AREA).
            int x0 = x * downscale, y0 = y * downscale;
            int x1 = std::min(frame.w, x0 + downscale),
                y1 = std::min(frame.h, y0 + downscale);
            double acc = 0;
            int cnt = 0;
            for (int yy = y0; yy < y1; yy++)
                for (int xx = x0; xx < x1; xx++) {
                    const uint8_t* p =
                            &frame.rgb[(size_t)(yy * frame.w + xx) * 3];
                    acc += 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
                    cnt++;
                }
            g.px[(size_t)y * dw + x] = (uint8_t)(cnt ? acc / cnt + 0.5 : 0);
        }
    return g;
}

// Shi-Tomasi corner detection (cv2.goodFeaturesToTrack: quality 0.01,
// minDistance 1, blockSize 3, maxCorners 1000, no Harris).
std::vector<std::pair<float, float>> shi_tomasi(const GrayImage& g) {
    const int w = g.w, h = g.h;
    std::vector<float> gx((size_t)w * h, 0), gy((size_t)w * h, 0);
    for (int y = 1; y < h - 1; y++)
        for (int x = 1; x < w - 1; x++) {
            gx[(size_t)y * w + x] = 0.25f * (g.px[(size_t)y * w + x + 1] -
                                             g.px[(size_t)y * w + x - 1]);
            gy[(size_t)y * w + x] = 0.25f * (g.px[(size_t)(y + 1) * w + x] -
                                             g.px[(size_t)(y - 1) * w + x]);
        }
    // Structure tensor summed over the 3x3 blockSize window; lambda_min per
    // pixel.
    std::vector<float> score((size_t)w * h, 0.0f);
    float max_score = 0;
    for (int y = 2; y < h - 2; y++)
        for (int x = 2; x < w - 2; x++) {
            double a = 0, b = 0, c = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    const size_t k = (size_t)(y + dy) * w + (x + dx);
                    a += (double)gx[k] * gx[k];
                    b += (double)gx[k] * gy[k];
                    c += (double)gy[k] * gy[k];
                }
            const double tr = a + c;
            const double det = a * c - b * b;
            const double disc = std::sqrt(std::max(0.0, tr * tr * 0.25 - det));
            const double s = tr * 0.5 - disc;
            score[(size_t)y * w + x] = (float)std::max(0.0, s);
            max_score = std::max(max_score, score[(size_t)y * w + x]);
        }
    const float thresh = 0.01f * max_score;
    // Non-max suppression with minDistance = 1 (3x3 window), then keep top
    // 1000.
    std::vector<std::pair<float, std::pair<float, float>>> corners;
    for (int y = 2; y < h - 2; y++)
        for (int x = 2; x < w - 2; x++) {
            const float s = score[(size_t)y * w + x];
            if (s < thresh || s == 0) continue;
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (score[(size_t)(y + dy) * w + (x + dx)] > s) {
                        is_max = false;
                        break;
                    }
                }
            if (is_max) corners.push_back({s, {(float)x, (float)y}});
        }
    std::sort(corners.begin(), corners.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (corners.size() > 1000) corners.resize(1000);
    std::vector<std::pair<float, float>> pts;
    pts.reserve(corners.size());
    for (auto& c : corners) pts.push_back(c.second);
    return pts;
}

// Build gaussian pyramids (3 extra levels) for pyramidal Lucas-Kanade.
std::vector<GrayImage> build_pyramid(const GrayImage& g, int levels) {
    std::vector<GrayImage> pyr{g};
    for (int l = 0; l < levels; l++) {
        const GrayImage& prev = pyr.back();
        GrayImage next{
                (std::max)(1, prev.w / 2), (std::max)(1, prev.h / 2), {}};
        next.px.resize((size_t)next.w * next.h);
        for (int y = 0; y < next.h; y++)
            for (int x = 0; x < next.w; x++) {
                // 2x2 box average (cv2.pyrDown's kernel is a gaussian; a box
                // average preserves the motion signal LK needs).
                const int x0 = std::min(prev.w - 1, x * 2),
                          x1 = std::min(prev.w, x * 2 + 1);
                const int y0 = std::min(prev.h - 1, y * 2),
                          y1 = std::min(prev.h, y * 2 + 1);
                int acc = 0, cnt = 0;
                for (int yy = y0; yy < y1; yy++)
                    for (int xx = x0; xx < x1; xx++) {
                        acc += prev.px[(size_t)yy * prev.w + xx];
                        cnt++;
                    }
                next.px[(size_t)y * next.w + x] =
                        (uint8_t)(cnt ? (acc + cnt / 2) / cnt : 0);
            }
        pyr.push_back(std::move(next));
    }
    return pyr;
}

// Lucas-Kanade optical flow for one point over a pyramid (21x21 window, up to
// 30 iterations, eps 0.01 — cv2.calcOpticalFlowPyrLK defaults).
bool lk_track_point(const std::vector<GrayImage>& prev_pyr,
                    const std::vector<GrayImage>& next_pyr,
                    float px,
                    float py,
                    float& out_dx,
                    float& out_dy) {
    constexpr int W = 10;  // window half-size (21x21)
    constexpr int MAX_ITER = 30;
    constexpr float EPS = 0.01f;
    float gx = px / (float)(1 << (int)(prev_pyr.size() - 1)),
          gy = py / (float)(1 << (int)(prev_pyr.size() - 1));
    float dx = 0, dy = 0;
    for (int level = (int)prev_pyr.size() - 1; level >= 0; level--) {
        gx *= 2;
        gy *= 2;
        dx *= 2;
        dy *= 2;
        const GrayImage& I = prev_pyr[level];
        const GrayImage& J = next_pyr[level];
        if (I.w < 2 * W + 2 || I.h < 2 * W + 2 || J.w < 2 * W + 2 ||
            J.h < 2 * W + 2)
            continue;
        // Spatial gradient sums over the window in I.
        double A[2][2] = {{0, 0}, {0, 0}};
        for (int wy = -W; wy <= W; wy++)
            for (int wx = -W; wx <= W; wx++) {
                const int ix = (int)std::lround(gx) + wx,
                          iy = (int)std::lround(gy) + wy;
                if (ix < 1 || ix >= I.w - 1 || iy < 1 || iy >= I.h - 1)
                    continue;
                const float ix2 = 0.5f * (I.px[(size_t)iy * I.w + ix + 1] -
                                          I.px[(size_t)iy * I.w + ix - 1]);
                const float iy2 = 0.5f * (I.px[(size_t)(iy + 1) * I.w + ix] -
                                          I.px[(size_t)(iy - 1) * I.w + ix]);
                A[0][0] += (double)ix2 * ix2;
                A[0][1] += (double)ix2 * iy2;
                A[1][0] += (double)ix2 * iy2;
                A[1][1] += (double)iy2 * iy2;
            }
        const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        if (std::fabs(det) < 1e-6) return false;
        for (int iter = 0; iter < MAX_ITER; iter++) {
            double b[2] = {0, 0};
            for (int wy = -W; wy <= W; wy++)
                for (int wx = -W; wx <= W; wx++) {
                    const int ix = (int)std::lround(gx) + wx,
                              iy = (int)std::lround(gy) + wy;
                    const float jx = gx + dx + wx, jy = gy + dy + wy;
                    if (ix < 1 || ix >= I.w - 1 || iy < 1 || iy >= I.h - 1)
                        continue;
                    if (jx < 0 || jx > J.w - 1 || jy < 0 || jy > J.h - 1)
                        continue;
                    // Bilinear sample J at (jx, jy).
                    const int x0 = (int)jx, y0 = (int)jy;
                    const int x1 = std::min(J.w - 1, x0 + 1),
                              y1 = std::min(J.h - 1, y0 + 1);
                    const float fx = jx - x0, fy = jy - y0;
                    const float jv =
                            (1 - fx) * (1 - fy) * J.px[(size_t)y0 * J.w + x0] +
                            fx * (1 - fy) * J.px[(size_t)y0 * J.w + x1] +
                            (1 - fx) * fy * J.px[(size_t)y1 * J.w + x0] +
                            fx * fy * J.px[(size_t)y1 * J.w + x1];
                    const float iv = (float)I.px[(size_t)iy * I.w + ix];
                    const float diff = iv - jv;
                    const float ix2 = 0.5f * (I.px[(size_t)iy * I.w + ix + 1] -
                                              I.px[(size_t)iy * I.w + ix - 1]);
                    const float iy2 =
                            0.5f * (I.px[(size_t)(iy + 1) * I.w + ix] -
                                    I.px[(size_t)(iy - 1) * I.w + ix]);
                    b[0] += (double)diff * ix2;
                    b[1] += (double)diff * iy2;
                }
            const double eta_x = (A[1][1] * b[0] - A[0][1] * b[1]) / det;
            const double eta_y = (A[0][0] * b[1] - A[1][0] * b[0]) / det;
            dx += (float)eta_x;
            dy += (float)eta_y;
            if (eta_x * eta_x + eta_y * eta_y < EPS * EPS) break;
        }
    }
    out_dx = dx;
    out_dy = dy;
    return std::isfinite(dx) && std::isfinite(dy);
}

// Least-squares similarity transform (Umeyama 2D, no reflection) over point
// pairs.
void fit_similarity(const std::vector<std::pair<float, float>>& prev,
                    const std::vector<std::pair<float, float>>& curr,
                    float H[6]) {
    const size_t n = prev.size();
    double mcx = 0, mcy = 0, ncx = 0, ncy = 0;
    for (size_t i = 0; i < n; i++) {
        mcx += prev[i].first;
        mcy += prev[i].second;
        ncx += curr[i].first;
        ncy += curr[i].second;
    }
    mcx /= n;
    mcy /= n;
    ncx /= n;
    ncy /= n;
    double sx = 0, sy = 0, denom = 0;
    for (size_t i = 0; i < n; i++) {
        const double ax = prev[i].first - mcx, ay = prev[i].second - mcy;
        const double bx = curr[i].first - ncx, by = curr[i].second - ncy;
        sx += ax * bx + ay * by;
        sy += ax * by - ay * bx;
        denom += ax * ax + ay * ay;
    }
    // M = s·R = [[sx, -sy], [sy, sx]] / Σ|a|²; identity when the fit
    // degenerates.
    double a = 1, b = 0;
    if (denom > 1e-9) {
        a = sx / denom;
        b = sy / denom;
    }
    // curr ≈ [[a, -b], [b, a]] (prev - mean_prev) + mean_curr
    H[0] = (float)a;
    H[1] = (float)-b;
    H[2] = (float)(ncx - (a * mcx - b * mcy));
    H[3] = (float)b;
    H[4] = (float)a;
    H[5] = (float)(ncy - (b * mcx + a * mcy));
}

// estimateAffinePartial2D(prev, curr, RANSAC, reproj 3.0): similarity model.
void ransac_similarity(const std::vector<std::pair<float, float>>& prev,
                       const std::vector<std::pair<float, float>>& curr,
                       float H[6]) {
    const size_t n = prev.size();
    fit_similarity(prev, curr, H);  // fallback / final refit base
    if (n < 2) return;
    uint64_t seed = 0x9E3779B97F4A7C15ull ^ (n * 0xBF58476D1CE4E5B9ull);
    auto rnd = [&seed]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    size_t best_inliers = 0;
    float best_H[6];
    std::copy(H, H + 6, best_H);
    for (int it = 0; it < 1000; it++) {
        const size_t i = rnd() % n, j = rnd() % n;
        if (i == j) continue;
        const float v[2] = {curr[j].first - curr[i].first,
                            curr[j].second - curr[i].second};
        const float u[2] = {prev[j].first - prev[i].first,
                            prev[j].second - prev[i].second};
        const float un2 = u[0] * u[0] + u[1] * u[1];
        if (un2 < 1e-6f) continue;
        const float s = (v[0] * u[0] + v[1] * u[1]) / un2;
        const float th = std::atan2(v[1], v[0]) - std::atan2(u[1], u[0]);
        const float c = std::cos(th), si = std::sin(th);
        const float R[2][2] = {{s * c, -s * si}, {s * si, s * c}};
        const float tx = curr[i].first -
                         (R[0][0] * prev[i].first + R[0][1] * prev[i].second);
        const float ty = curr[i].second -
                         (R[1][0] * prev[i].first + R[1][1] * prev[i].second);
        size_t inliers = 0;
        std::vector<std::pair<float, float>> pin, cin;
        for (size_t k = 0; k < n; k++) {
            const float ex = R[0][0] * prev[k].first +
                             R[0][1] * prev[k].second + tx - curr[k].first;
            const float ey = R[1][0] * prev[k].first +
                             R[1][1] * prev[k].second + ty - curr[k].second;
            if (ex * ex + ey * ey <= 9.0f) {
                inliers++;
                pin.push_back(prev[k]);
                cin.push_back(curr[k]);
            }
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            if (pin.size() >= 2) {
                float refit[6];
                fit_similarity(pin, cin, refit);
                std::copy(refit, refit + 6, best_H);
            } else {
                best_H[0] = R[0][0];
                best_H[1] = R[0][1];
                best_H[2] = tx;
                best_H[3] = R[1][0];
                best_H[4] = R[1][1];
                best_H[5] = ty;
            }
        }
    }
    std::copy(best_H, best_H + 6, H);
}

struct GMC {
    std::string method;  // "sparseOptFlow" or "none"/"" (upstream normalizes to
                         // None)
    int downscale = 2;
    bool has_prev = false;
    GrayImage prev_frame;
    std::vector<std::pair<float, float>> prev_keypoints;

    explicit GMC(std::string m) : method(std::move(m)) {
        if (method == "none" || method == "None") method.clear();
    }
    bool enabled() const { return !method.empty(); }

    // Returns the 2x3 affine warp (row-major) mapping the previous frame to
    // this one. detections: this frame's high-score detections (mask out boxes
    // like the upstream apply_features mask); ignored by the sparseOptFlow
    // fallback.
    void apply(const RgbFrame& frame,
               const std::vector<TrackDet>&,
               float H[6]) {
        H[0] = H[4] = 1;
        H[1] = H[2] = H[3] = H[5] = 0;
        if (!enabled()) return;
        const GrayImage cur = to_gray_downscaled(frame, downscale);
        const auto kps = shi_tomasi(cur);
        if (!has_prev) {
            prev_frame = cur;
            prev_keypoints = kps;
            has_prev = true;
            return;
        }
        if (prev_keypoints.empty()) {
            prev_frame = cur;
            prev_keypoints = kps;
            return;
        }
        auto prev_pyr = build_pyramid(prev_frame, 3);
        auto next_pyr = build_pyramid(cur, 3);
        std::vector<std::pair<float, float>> prev_pts, curr_pts;
        for (const auto& [px, py] : prev_keypoints) {
            if (px < 0 || py < 0 || px >= prev_frame.w || py >= prev_frame.h)
                continue;
            float dx = 0, dy = 0;
            if (lk_track_point(prev_pyr, next_pyr, px, py, dx, dy))
                curr_pts.emplace_back(px + dx, py + dy),
                        prev_pts.emplace_back(px, py);
        }
        prev_frame = cur;
        prev_keypoints = kps;
        if (prev_pts.size() <= 4)
            return;  // "not enough matching points": keep identity
        ransac_similarity(prev_pts, curr_pts, H);
        if (downscale > 1) {
            H[2] *= downscale;
            H[5] *= downscale;
        }
    }

    void reset_params() {
        has_prev = false;
        prev_frame = {};
        prev_keypoints.clear();
    }
};

#else  // QYOLO_WITH_OPENCV: same cv2 calls as utils/gmc.py

struct GMC {
    std::string method;
    int downscale = 2;
    bool initialized_first_frame = false;
    cv::Mat prev_frame;
    std::vector<cv::Point2f> prev_keypoints;
    cv::Mat prev_descriptors;  // orb / sift
    cv::Ptr<cv::FastFeatureDetector> fast;
    cv::Ptr<cv::ORB> orb;
#if defined(QYOLO_WITH_OPENCV_SIFT)
    cv::Ptr<cv::SIFT> sift;
#endif
    cv::Ptr<cv::BFMatcher> matcher;

    explicit GMC(std::string m) : method(std::move(m)) {
        if (method == "none" || method == "None") method.clear();
        if (method == "orb") {
            fast = cv::FastFeatureDetector::create(20);
            orb = cv::ORB::create();
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
#if defined(QYOLO_WITH_OPENCV_SIFT)
        } else if (method == "sift") {
            sift = cv::SIFT::create(0, 3, 0.02, 20);
            matcher = cv::BFMatcher::create(cv::NORM_L2);
#endif
        }
    }
    bool enabled() const { return !method.empty(); }

    static void store_warp(const cv::Mat& warp, float H[6]) {
        H[0] = warp.at<float>(0, 0);
        H[1] = warp.at<float>(0, 1);
        H[2] = warp.at<float>(0, 2);
        H[3] = warp.at<float>(1, 0);
        H[4] = warp.at<float>(1, 1);
        H[5] = warp.at<float>(1, 2);
    }

    // cv::estimateAffinePartial2D returns CV_64F; normalize to CV_32F so the
    // downstream at<float> accesses are valid (Python's float64 warp is
    // bit-compatible with the same OpenCV solve here).
    static cv::Mat estimate_affine_partial_2d_f32(
            const std::vector<cv::Point2f>& prev,
            const std::vector<cv::Point2f>& curr) {
        cv::Mat warp64 = cv::estimateAffinePartial2D(prev, curr, cv::noArray(),
                                                     cv::RANSAC);
        if (warp64.empty()) return {};
        cv::Mat warp;
        warp64.convertTo(warp, CV_32F);
        return warp;
    }

    static cv::Mat to_gray(const RgbFrame& f) {
        cv::Mat rgb(f.h, f.w, CV_8UC3, const_cast<uint8_t*>(f.rgb),
                    (size_t)f.w * 3);
        cv::Mat gray;
        cv::cvtColor(rgb, gray,
                     cv::COLOR_RGB2GRAY);  // same luma weights as BGR2GRAY
        return gray;
    }

    // Returns the 2x3 affine warp (row-major) mapping the previous frame to
    // this one. detections: this frame's high-score detections, masked out of
    // keypoint detection like apply_features does upstream (boxes in original
    // pixels).
    void apply(const RgbFrame& raw,
               const std::vector<TrackDet>& dets,
               float H[6]) {
        H[0] = H[4] = 1;
        H[1] = H[2] = H[3] = H[5] = 0;
        if (!enabled()) return;
        if (method == "ecc")
            apply_ecc(raw, H);
        else if (method == "sparseOptFlow")
            apply_sparseoptflow(raw, H);
        else
            apply_features(raw, dets, H);  // orb | sift
    }

    // gmc.py apply_ecc
    void apply_ecc(const RgbFrame& raw, float H[6]) {
        const float width = (float)raw.w, height = (float)raw.h;
        cv::Mat frame = to_gray(raw);
        cv::Mat warp = (cv::Mat_<float>(2, 3) << 1, 0, 0, 0, 1, 0);
        if (downscale > 1) {
            cv::GaussianBlur(frame, frame, cv::Size(3, 3), 1.5);
            cv::resize(frame, frame,
                       cv::Size(raw.w / downscale, raw.h / downscale));
        }
        if (!initialized_first_frame) {
            prev_frame = frame.clone();
            initialized_first_frame = true;
            return;
        }
        try {
            const cv::TermCriteria crit(
                    cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 5000,
                    1e-6);
            cv::findTransformECC(prev_frame, frame, warp, cv::MOTION_EUCLIDEAN,
                                 crit, cv::noArray(), 1);
            warp.at<float>(0, 2) *= width / (float)frame.cols;
            warp.at<float>(1, 2) *= height / (float)frame.rows;
        } catch (const cv::Exception&) {  // "findTransformECC failed; using
                                          // identity warp"
        }
        prev_frame = frame.clone();
        store_warp(warp, H);
    }

    // gmc.py apply_sparseoptflow
    void apply_sparseoptflow(const RgbFrame& raw, float H[6]) {
        cv::Mat frame = to_gray(raw);
        if (downscale > 1)
            cv::resize(frame, frame,
                       cv::Size(raw.w / downscale, raw.h / downscale));
        std::vector<cv::Point2f> keypoints;
        cv::goodFeaturesToTrack(frame, keypoints, 1000, 0.01, 1, cv::noArray(),
                                3, false, 0.04);
        if (!initialized_first_frame) {
            prev_frame = frame.clone();
            prev_keypoints = keypoints;
            initialized_first_frame = true;
            return;
        }
        if (!prev_keypoints.empty()) {
            std::vector<cv::Point2f> matched;
            std::vector<uchar> status;
            cv::calcOpticalFlowPyrLK(prev_frame, frame, prev_keypoints, matched,
                                     status, cv::noArray());
            std::vector<cv::Point2f> prev_pts, curr_pts;
            for (size_t i = 0; i < status.size() && i < matched.size(); i++)
                if (status[i])
                    prev_pts.push_back(prev_keypoints[i]),
                            curr_pts.push_back(matched[i]);
            if (prev_pts.size() > 4) {
                cv::Mat warp =
                        estimate_affine_partial_2d_f32(prev_pts, curr_pts);
                if (!warp.empty()) {
                    if (downscale > 1) {
                        warp.at<float>(0, 2) *= downscale;
                        warp.at<float>(1, 2) *= downscale;
                    }
                    store_warp(warp, H);
                }
            }
        }
        prev_frame = frame.clone();
        prev_keypoints = keypoints;
    }

    // gmc.py apply_features (orb | sift)
    void apply_features(const RgbFrame& raw,
                        const std::vector<TrackDet>& dets,
                        float H[6]) {
        float width = (float)raw.w, height = (float)raw.h;
        cv::Mat frame = to_gray(raw);
        if (downscale > 1) {
            cv::resize(frame, frame,
                       cv::Size(raw.w / downscale, raw.h / downscale));
            width = (float)(raw.w / downscale);
            height = (float)(raw.h / downscale);
        }
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
        cv::rectangle(mask,
                      cv::Point((int)(0.02 * width), (int)(0.02 * height)),
                      cv::Point((int)(0.98 * width), (int)(0.98 * height)),
                      cv::Scalar(255), -1);
        for (const TrackDet& d : dets) {
            const cv::Point tl((int)((d.cx - d.w / 2) / downscale),
                               (int)((d.cy - d.h / 2) / downscale));
            const cv::Point br((int)((d.cx + d.w / 2) / downscale),
                               (int)((d.cy + d.h / 2) / downscale));
            cv::rectangle(mask, tl, br, cv::Scalar(0), -1);
        }
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        if (method == "orb") {
            fast->detect(frame, keypoints, mask);
            orb->compute(frame, keypoints, descriptors);
        } else {
#if defined(QYOLO_WITH_OPENCV_SIFT)
            sift->detect(frame, keypoints, mask);
            sift->compute(frame, keypoints, descriptors);
#endif
        }
        if (!initialized_first_frame) {
            prev_frame = frame.clone();
            prev_keypoints.clear();
            for (const cv::KeyPoint& k : keypoints)
                prev_keypoints.push_back(k.pt);
            prev_descriptors = descriptors.clone();
            initialized_first_frame = true;
            return;
        }
        std::vector<std::vector<cv::DMatch>> knn;
        if (!prev_descriptors.empty() && !descriptors.empty())
            matcher->knnMatch(prev_descriptors, descriptors, knn, 2);
        // Lowe's ratio test + spatial distance filter (0.25 * [width, height]).
        const float max_dx = 0.25f * width, max_dy = 0.25f * height;
        std::vector<cv::Point2f> prev_pts, curr_pts;
        std::vector<std::pair<float, float>> spatial;
        for (const std::vector<cv::DMatch>& ms : knn) {
            if (ms.size() < 2) continue;
            const cv::DMatch &m = ms[0], &n = ms[1];
            if (m.distance >= 0.9f * n.distance) continue;
            const cv::Point2f& p = prev_keypoints[m.queryIdx];
            const cv::Point2f& c = keypoints[m.trainIdx].pt;
            const float dx = p.x - c.x, dy = p.y - c.y;
            if (std::fabs(dx) < max_dx && std::fabs(dy) < max_dy) {
                spatial.emplace_back(dx, dy);
                prev_pts.push_back(p);
                curr_pts.push_back(c);
            }
        }
        if (!spatial.empty()) {
            // 2.5-sigma outlier filter over the match displacements (ddof=0,
            // exact-boundary and zero-variance matches kept, as upstream).
            double mx = 0, my = 0;
            for (const auto& s : spatial) mx += s.first, my += s.second;
            mx /= (double)spatial.size();
            my /= (double)spatial.size();
            double vx = 0, vy = 0;
            for (const auto& s : spatial)
                vx += (s.first - mx) * (s.first - mx),
                        vy += (s.second - my) * (s.second - my);
            vx /= (double)spatial.size();
            vy /= (double)spatial.size();
            const double sx = std::sqrt(vx), sy = std::sqrt(vy);
            std::vector<cv::Point2f> good_prev, good_curr;
            for (size_t i = 0; i < spatial.size(); i++)
                if (std::fabs(spatial[i].first - mx) <= 2.5 * sx &&
                    std::fabs(spatial[i].second - my) <= 2.5 * sy)
                    good_prev.push_back(prev_pts[i]),
                            good_curr.push_back(curr_pts[i]);
            if (good_prev.size() > 4) {
                cv::Mat warp =
                        estimate_affine_partial_2d_f32(good_prev, good_curr);
                if (!warp.empty()) {
                    if (downscale > 1) {
                        warp.at<float>(0, 2) *= downscale;
                        warp.at<float>(1, 2) *= downscale;
                    }
                    store_warp(warp, H);
                }
            }
        }
        prev_frame = frame.clone();
        prev_keypoints.clear();
        for (const cv::KeyPoint& k : keypoints) prev_keypoints.push_back(k.pt);
        prev_descriptors = descriptors.clone();
    }

    void reset_params() {
        initialized_first_frame = false;
        prev_frame.release();
        prev_keypoints.clear();
        prev_descriptors.release();
    }
};

#endif  // QYOLO_WITH_OPENCV

}  // namespace

// One tracked output row: the track state box in center format, mirroring the
// Python result row [coords..., id, score, cls, idx].
TrackedBox track_to_box(const STrack* t) {
    TrackDet d;
    float xywh[4];
    t->state_xywh(xywh);
    d.cx = xywh[0];
    d.cy = xywh[1];
    d.w = xywh[2];
    d.h = xywh[3];
    d.angle = t->angled() ? t->angle : -10.0f;
    d.score = t->score;
    d.class_id = t->cls;
    d.idx = t->idx;
    return {d, t->track_id};
}

// ---- BYTETracker (byte_tracker.py): two-stage high/low-score association ----

class BYTETracker : public Tracker {
public:
    explicit BYTETracker(const TrackConfig& c) : cfg(c) {
        max_frames_lost = c.track_buffer;
        kf.kind = KFKind::XYAH;
        STrack::reset_id();
    }

    std::vector<TrackedBox> update(const FrameInput& in) override {
        frame_id++;
        std::vector<STrack*> activated, refind, lost, removed;

        // _split_detections: high band, low band, degenerate boxes dropped.
        // Features ride along, index-aligned (official _input_for slices
        // the per-detection feature rows with the same band masks).
        std::vector<TrackDet> high, low;
        std::vector<std::vector<float>> high_feats, low_feats;
        for (size_t i = 0; i < in.dets.size(); i++) {
            const TrackDet& d = in.dets[i];
            if (d.w <= 0 || d.h <= 0) continue;
            static const std::vector<float> kNoFeat;
            const std::vector<float>& f =
                    i < in.feats.size() ? in.feats[i] : kNoFeat;
            if (d.score >= cfg.track_high_thresh) {
                high.push_back(d);
                high_feats.push_back(f);
            } else if (d.score > cfg.track_low_thresh &&
                       d.score < cfg.track_high_thresh) {
                low.push_back(d);
                low_feats.push_back(f);
            }
        }
        std::vector<STrack*> detections = init_track(high, high_feats);
        std::vector<STrack*> detections_second = init_track(low, low_feats);

        // _split_tracked + strack_pool
        std::vector<STrack*> unconfirmed, tracked;
        for (STrack* t : tracked_stracks)
            (t->is_activated ? tracked : unconfirmed).push_back(t);
        std::vector<STrack*> strack_pool = tracked;
        joint_by_id(strack_pool, lost_stracks);
        for (STrack* t : strack_pool) t->predict();
        pre_first_associate(strack_pool, unconfirmed, in, high);

        // _first_association
        std::vector<int> u_track, u_detection;
        {
            const CostPack pack = dists_and_sizes(strack_pool, detections);
            const Assignment res = linear_assignment(pack.cost, pack.na,
                                                     pack.nb, cfg.match_thresh);
            apply_matches(res.matches, strack_pool, detections, activated,
                          refind);
            u_track = res.unmatched_a;
            u_detection = res.unmatched_b;
        }
        // _post_first_association (OCR hook in OC-SORT; no-op base). Both
        // unmatched lists flow through: OC-SORT remaps u_detection so
        // detections consumed by the OCR pass cannot be re-consumed by the
        // unconfirmed stage or the new-track spawn (byte_tracker.py update).
        const auto post =
                post_first_association(strack_pool, detections, u_track,
                                       u_detection, activated, refind);
        u_track = post.first;
        u_detection = post.second;

        // _second_association
        second_association(strack_pool, u_track, detections_second, activated,
                           refind, lost);

        // _unconfirmed_association
        {
            std::vector<STrack*> leftover;
            for (int i : u_detection) leftover.push_back(detections[i]);
            if (!unconfirmed.empty()) {
                const CostPack pack = dists_and_sizes(unconfirmed, leftover);
                const Assignment res =
                        linear_assignment(pack.cost, pack.na, pack.nb, 0.7f);
                for (const auto& [itracked, idet] : res.matches) {
                    unconfirmed[itracked]->update(*leftover[idet], frame_id);
                    activated.push_back(unconfirmed[itracked]);
                }
                for (int it : res.unmatched_a) {
                    unconfirmed[it]->mark_removed();
                    removed.push_back(unconfirmed[it]);
                }
                u_detection = res.unmatched_b;
                detections = leftover;
            }
        }

        // _init_new_tracks
        init_new_tracks(u_detection, detections, activated, refind);
        // _remove_stale_lost
        remove_stale_lost(removed);

        merge_track_pools(tracked_stracks, lost_stracks, removed_stracks,
                          activated, refind, lost, removed);
        prune_arena();
        return format_output();
    }

    void reset() override {
        tracked_stracks.clear();
        lost_stracks.clear();
        removed_stracks.clear();
        arena.clear();
        frame_id = 0;
        kf.kind = KFKind::XYAH;
        STrack::reset_id();
    }

protected:
    const TrackConfig& cfg;
    int frame_id = 0;
    int max_frames_lost = 30;
    KalmanFilter kf;
    std::vector<std::unique_ptr<STrack>>
            arena;  // owns every STrack ever created
    std::vector<STrack*> tracked_stracks, lost_stracks, removed_stracks;
    std::optional<GMC>
            gmc;  // set by BOTSORT (mirrors hasattr(self, "gmc") upstream)

    virtual STrack* make_track(const TrackDet& d) {
        return new STrack(d, KFKind::XYAH);
    }

    std::vector<STrack*> init_track(
            const std::vector<TrackDet>& dets,
            const std::vector<std::vector<float>>& feats = {}) {
        std::vector<STrack*> out;
        out.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); i++) {
            arena.emplace_back(make_track(dets[i]));
            STrack* st = arena.back().get();
            // init_track(feat): the EMA blend factor comes from the track's
            // virtual feature_alpha (BoT-SORT family semantics).
            if (i < feats.size() && !feats[i].empty())
                st->update_features(feats[i]);
            out.push_back(st);
        }
        return out;
    }

    // Track-state views for matching.iou_distance(tracks, detections).
    static std::vector<TrackDet> track_views(
            const std::vector<STrack*>& tracks) {
        std::vector<TrackDet> out;
        out.reserve(tracks.size());
        for (const STrack* t : tracks) {
            TrackDet d;
            if (t->angled()) {
                float xywh[4];
                t->state_xywh(xywh);
                d.cx = xywh[0];
                d.cy = xywh[1];
                d.w = xywh[2];
                d.h = xywh[3];
                d.angle = t->angle;
            } else {
                float xyxy[4];
                t->state_xyxy(xyxy);
                d.cx = (xyxy[0] + xyxy[2]) / 2;
                d.cy = (xyxy[1] + xyxy[3]) / 2;
                d.w = xyxy[2] - xyxy[0];
                d.h = xyxy[3] - xyxy[1];
                d.angle = -10.0f;
            }
            d.score = t->score;
            d.class_id = t->cls;
            d.idx = t->idx;
            out.push_back(d);
        }
        return out;
    }

    // get_dists + the shapes linear_assignment needs.
    struct CostPack {
        std::vector<float> cost;
        int na, nb;
    };
    virtual CostPack dists_and_sizes(const std::vector<STrack*>& tracks,
                                     const std::vector<STrack*>& dets) {
        const std::vector<TrackDet> tv = track_views(tracks),
                                    dv = track_views(dets);
        CostPack pack{iou_distance(tv, dv), (int)tracks.size(),
                      (int)dets.size()};
        if (cfg.fuse_score) pack.cost = fuse_score(pack.cost, pack.na, dv);
        return pack;
    }

    virtual void pre_first_associate(const std::vector<STrack*>& pool,
                                     const std::vector<STrack*>& unconfirmed,
                                     const FrameInput& in,
                                     const std::vector<TrackDet>& high) {
        if (gmc && gmc->enabled() && in.frame) {
            float H[6];
            gmc->apply(*in.frame, high, H);
            multi_gmc(pool, H);
            multi_gmc(unconfirmed, H);
        }
    }

    // Returns the (possibly remapped) unmatched-track and unmatched-detection
    // index lists (default: unchanged).
    virtual std::pair<std::vector<int>, std::vector<int>>
    post_first_association(const std::vector<STrack*>&,
                           const std::vector<STrack*>&,
                           std::vector<int> u_track,
                           std::vector<int> u_detection,
                           std::vector<STrack*>&,
                           std::vector<STrack*>&) {
        return {std::move(u_track), std::move(u_detection)};
    }

    virtual void apply_match(STrack* track,
                             STrack* det,
                             std::vector<STrack*>& activated,
                             std::vector<STrack*>& refind) {
        if (track->state == TrackState::Tracked) {
            track->update(*det, frame_id);
            activated.push_back(track);
        } else {
            track->re_activate(*det, frame_id, false);
            refind.push_back(track);
        }
    }

    void apply_matches(const std::vector<std::pair<int, int>>& matches,
                       const std::vector<STrack*>& pool,
                       const std::vector<STrack*>& dets,
                       std::vector<STrack*>& activated,
                       std::vector<STrack*>& refind) {
        for (const auto& [itracked, idet] : matches)
            apply_match(pool[itracked], dets[idet], activated, refind);
    }

    virtual void second_association(const std::vector<STrack*>& strack_pool,
                                    std::vector<int>& u_track,
                                    std::vector<STrack*>& detections_second,
                                    std::vector<STrack*>& activated,
                                    std::vector<STrack*>& refind,
                                    std::vector<STrack*>& lost) {
        // IoU-only by design (ByteTrack paper sec. 3.2); fixed 0.5 threshold.
        std::vector<STrack*> r_tracked;
        for (int i : u_track)
            if (strack_pool[i]->state == TrackState::Tracked)
                r_tracked.push_back(strack_pool[i]);
        if (!r_tracked.empty() && !detections_second.empty()) {
            const std::vector<float> cost = iou_distance(
                    track_views(r_tracked), track_views(detections_second));
            const Assignment res =
                    linear_assignment(cost, (int)r_tracked.size(),
                                      (int)detections_second.size(), 0.5f);
            apply_matches(res.matches, r_tracked, detections_second, activated,
                          refind);
            u_track = res.unmatched_a;
        } else {
            u_track.clear();
            for (size_t i = 0; i < r_tracked.size(); i++)
                u_track.push_back((int)i);
        }
        for (int it : u_track) {
            STrack* track = r_tracked[it];
            if (track->state != TrackState::Lost) {
                track->mark_lost();
                lost.push_back(track);
            }
        }
    }

    // refind feeds the FASTTRACKER suppression stack; the base ignores it
    // (byte_tracker.py _init_new_tracks takes it for signature parity too).
    virtual void init_new_tracks(const std::vector<int>& u_detection,
                                 const std::vector<STrack*>& detections,
                                 std::vector<STrack*>& activated,
                                 const std::vector<STrack*>& /*refind*/) {
        for (int inew : u_detection) {
            STrack* track = detections[inew];
            if (track->score < cfg.new_track_thresh) continue;
            track->activate(kf, frame_id);
            activated.push_back(track);
        }
    }

    virtual void remove_stale_lost(std::vector<STrack*>& removed) {
        for (STrack* track : lost_stracks)
            if (frame_id - track->end_frame() > max_frames_lost) {
                track->mark_removed();
                removed.push_back(track);
            }
    }

    virtual std::vector<TrackedBox> format_output() {
        std::vector<TrackedBox> out;
        for (const STrack* t : tracked_stracks)
            if (t->is_activated) out.push_back(track_to_box(t));
        return out;
    }

    // Free detection STracks that never entered a pool (upstream lets GC do
    // this).
    void prune_arena() {
        std::vector<STrack*> keep;
        keep.reserve(tracked_stracks.size() + lost_stracks.size() +
                     removed_stracks.size());
        keep.insert(keep.end(), tracked_stracks.begin(), tracked_stracks.end());
        keep.insert(keep.end(), lost_stracks.begin(), lost_stracks.end());
        keep.insert(keep.end(), removed_stracks.begin(), removed_stracks.end());
        std::sort(keep.begin(), keep.end());
        arena.erase(std::remove_if(arena.begin(), arena.end(),
                                   [&](const std::unique_ptr<STrack>& p) {
                                       return !std::binary_search(keep.begin(),
                                                                  keep.end(),
                                                                  p.get());
                                   }),
                    arena.end());
    }
};

// ---- BOTSORT (bot_sort.py): BYTETracker + XYWH state + GMC (ReID path
// omitted) ----

class BOTSORT : public BYTETracker {
public:
    explicit BOTSORT(const TrackConfig& c) : BYTETracker(c) {
        kf.kind = KFKind::XYWH;  // get_kalmanfilter -> KalmanFilterXYWH
        gmc.emplace(c.gmc_method);
    }

    void reset() override {
        BYTETracker::reset();
        kf.kind = KFKind::XYWH;
        gmc->reset_params();
    }

protected:
    STrack* make_track(const TrackDet& d) override {
        return new STrack(d, KFKind::XYWH);
    }

    // Official get_dists (bot_sort.py BOTSORT): IoU distance + fuse_score,
    // then the ReID cosine term — halved, gated by the appearance
    // threshold and the pre-fusion proximity mask, and min-fused into the
    // motion cost (appearance can only lower it). Without features the
    // embedding rows are all-missing and the fusion is a no-op, which is
    // exactly the upstream "feats missing" behavior.
    CostPack dists_and_sizes(const std::vector<STrack*>& tracks,
                             const std::vector<STrack*>& dets) override {
        const std::vector<TrackDet> tv = track_views(tracks),
                                    dv = track_views(dets);
        std::vector<float> dists = iou_distance(tv, dv);
        std::vector<char> proximity_mask(dists.size(), 0);
        for (size_t k = 0; k < dists.size(); k++)
            proximity_mask[k] = dists[k] > (1.0f - cfg.proximity_thresh);
        if (cfg.fuse_score) dists = fuse_score(dists, (int)tracks.size(), dv);
        if (cfg.with_reid && !dists.empty()) {
            std::vector<float> emb = embedding_distance(tracks, dets);
            for (size_t k = 0; k < dists.size(); k++) {
                float e = emb[k] / 2.0f;
                if (e > 1.0f - cfg.appearance_thresh) e = 1.0f;
                if (proximity_mask[k]) e = 1.0f;
                dists[k] = std::min(dists[k], e);
            }
        }
        return {std::move(dists), (int)tracks.size(), (int)dets.size()};
    }
    // The pre-fusion proximity mask gates the ReID term exactly like the
    // upstream dists_mask (bot_sort.py BOTSORT.get_dists).
};

// ---- OC-SORT (oc_sort.py): OCM / OCR / ORU on the BYTETracker frame ----

class OCSORT : public BYTETracker {
public:
    OCSORT(const TrackConfig& c)
        : BYTETracker(c),
          delta_t(c.delta_t),
          inertia(c.inertia),
          use_byte(c.use_byte) {}

protected:
    int delta_t;
    float inertia;
    bool use_byte;

    STrack* make_track(const TrackDet& d) override {
        return new OCSortTrack(d, delta_t);
    }

    // Hook combining motion cost with appearance cost; pass-through (no ReID).
    virtual std::vector<float> fuse_appearance(std::vector<float> dists,
                                               const std::vector<STrack*>&,
                                               const std::vector<STrack*>&,
                                               const std::vector<float>&) {
        return dists;
    }

    CostPack dists_and_sizes(const std::vector<STrack*>& tracks,
                             const std::vector<STrack*>& dets) override {
        const std::vector<TrackDet> tv = track_views(tracks),
                                    dv = track_views(dets);
        const std::vector<float> iou_dists = iou_distance(tv, dv);
        CostPack pack{cfg.fuse_score
                              ? fuse_score(iou_dists, (int)tracks.size(), dv)
                              : iou_dists,
                      (int)tracks.size(), (int)dets.size()};
        // get_dists: base + inertia * OCM velocity-consistency cost.
        const std::vector<float> ocm = velocity_direction_cost(tracks, dv);
        for (size_t k = 0; k < pack.cost.size(); k++)
            pack.cost[k] += inertia * ocm[k];
        pack.cost = fuse_appearance(pack.cost, tracks, dets, iou_dists);
        return pack;
    }

    // OCM: angular difference between the track's historical motion direction
    // and the direction to each candidate detection, in [0, 1].
    std::vector<float> velocity_direction_cost(
            const std::vector<STrack*>& tracks,
            const std::vector<TrackDet>& dv) const {
        std::vector<float> out(tracks.size() * dv.size(), 0.0f);
        const int m = (int)dv.size();
        for (size_t i = 0; i < tracks.size(); i++) {
            const OCSortTrack* t = static_cast<const OCSortTrack*>(tracks[i]);
            if (!t->has_velocity || t->last_observation[0] < 0) continue;
            float tc[2];
            OCSortTrack::xyxy_center(t->last_observation, tc);
            for (int j = 0; j < m; j++) {
                const float dx = dv[j].cx - tc[0], dy = dv[j].cy - tc[1];
                const float norm = std::sqrt(dx * dx + dy * dy);
                if (norm <= 1e-6f) continue;
                const float dot =
                        std::clamp((dx / norm) * t->velocity[0] +
                                           (dy / norm) * t->velocity[1],
                                   -1.0f, 1.0f);
                out[i * m + j] = std::acos(dot) / 3.14159265f;
            }
        }
        return out;
    }

    std::pair<std::vector<int>, std::vector<int>> post_first_association(
            const std::vector<STrack*>& strack_pool,
            const std::vector<STrack*>& detections,
            std::vector<int> u_track,
            std::vector<int> u_detection_in,
            std::vector<STrack*>& activated,
            std::vector<STrack*>& refind) override {
        // OCR passes after the first stage: active tracks get first pick.
        std::vector<STrack*> ocr_dets;
        for (int i : u_detection_in) ocr_dets.push_back(detections[i]);
        if (ocr_dets.empty())
            return {std::move(u_track), std::move(u_detection_in)};
        std::vector<int> tracked_idx, other_idx;
        for (int i : u_track)
            (strack_pool[i]->state == TrackState::Tracked ? tracked_idx
                                                          : other_idx)
                    .push_back(i);
        const auto [u_t1, u_d1] = ocr_associate(
                indexed(strack_pool, tracked_idx), ocr_dets, activated, refind);
        std::vector<STrack*> remaining;
        for (int j : u_d1) remaining.push_back(ocr_dets[j]);
        const auto [u_t2, u_d2] = ocr_associate(indexed(strack_pool, other_idx),
                                                remaining, activated, refind);
        std::vector<int> u_track_out;
        for (int i : u_t1) u_track_out.push_back(tracked_idx[i]);
        for (int i : u_t2) u_track_out.push_back(other_idx[i]);
        // Remap the unmatched detections through both OCR passes so detections
        // consumed by OCR never reach the unconfirmed stage or the new-track
        // spawn (oc_sort.py _post_first_association return value).
        std::vector<int> u_det_out;
        for (int j : u_d2) u_det_out.push_back(u_detection_in[u_d1[j]]);
        return {std::move(u_track_out), std::move(u_det_out)};
    }

    void second_association(const std::vector<STrack*>& strack_pool,
                            std::vector<int>& u_track,
                            std::vector<STrack*>& detections_second,
                            std::vector<STrack*>& activated,
                            std::vector<STrack*>& refind,
                            std::vector<STrack*>& lost) override {
        if (!use_byte) {
            for (int i : u_track) {
                STrack* track = strack_pool[i];
                if (track->state == TrackState::Tracked) {
                    track->mark_lost();
                    lost.push_back(track);
                }
            }
            return;
        }
        BYTETracker::second_association(strack_pool, u_track, detections_second,
                                        activated, refind, lost);
    }

private:
    static std::vector<STrack*> indexed(const std::vector<STrack*>& v,
                                        const std::vector<int>& idx) {
        std::vector<STrack*> out;
        out.reserve(idx.size());
        for (int i : idx) out.push_back(v[i]);
        return out;
    }

    // IoU distance on last observations (OCR); OBB falls back to predicted
    // boxes.
    std::vector<float> ocr_distance(const std::vector<STrack*>& tracks,
                                    const std::vector<STrack*>& dets) {
        std::vector<TrackDet> dv = track_views(dets);
        std::vector<TrackDet> tv;
        tv.reserve(tracks.size());
        const bool angled = !tracks.empty() && tracks[0]->angled();
        for (const STrack* t : tracks) {
            const OCSortTrack* ot = static_cast<const OCSortTrack*>(t);
            TrackDet d;
            if (angled) {
                float xywh[4];
                t->state_xywh(xywh);
                d.cx = xywh[0];
                d.cy = xywh[1];
                d.w = xywh[2];
                d.h = xywh[3];
                d.angle = t->angle;
            } else if (ot->last_observation[0] >= 0) {
                const float* b = ot->last_observation;
                d.cx = (b[0] + b[2]) / 2;
                d.cy = (b[1] + b[3]) / 2;
                d.w = b[2] - b[0];
                d.h = b[3] - b[1];
                d.angle = -10.0f;
            } else {
                float xyxy[4];
                t->state_xyxy(xyxy);
                d.cx = (xyxy[0] + xyxy[2]) / 2;
                d.cy = (xyxy[1] + xyxy[3]) / 2;
                d.w = xyxy[2] - xyxy[0];
                d.h = xyxy[3] - xyxy[1];
                d.angle = -10.0f;
            }
            d.score = t->score;
            d.class_id = t->cls;
            d.idx = t->idx;
            tv.push_back(d);
        }
        return iou_distance(tv, dv);
    }

    std::pair<std::vector<int>, std::vector<int>> ocr_associate(
            const std::vector<STrack*>& tracks,
            const std::vector<STrack*>& dets,
            std::vector<STrack*>& activated,
            std::vector<STrack*>& refind) {
        if (tracks.empty() || dets.empty()) {
            std::vector<int> ta(tracks.size()), da(dets.size());
            for (size_t i = 0; i < tracks.size(); i++) ta[i] = (int)i;
            for (size_t i = 0; i < dets.size(); i++) da[i] = (int)i;
            return {ta, da};
        }
        const std::vector<float> raw_ocr = ocr_distance(tracks, dets);
        std::vector<float> dists = raw_ocr;
        if (cfg.fuse_score)
            dists = fuse_score(dists, (int)tracks.size(), track_views(dets));
        // _fuse_appearance receives the RAW OCR IoU distances so the ReID
        // proximity gate stays tied to the pre-fusion motion cost (oc_sort.py
        // _ocr_associate: iou_dists=iou_dists).
        dists = fuse_appearance(dists, tracks, dets, raw_ocr);
        const Assignment res = linear_assignment(
                dists, (int)tracks.size(), (int)dets.size(), cfg.match_thresh);
        for (const auto& [itracked, idet] : res.matches) {
            STrack* track = tracks[itracked];
            STrack* det = dets[idet];
            if (track->state == TrackState::Tracked) {
                track->update(*det, frame_id);
                activated.push_back(track);
            } else {
                float b[4];
                det->state_xyxy(b);
                static_cast<OCSortTrack*>(track)->apply_oru(b, frame_id);
                track->re_activate(*det, frame_id, false);
                refind.push_back(track);
            }
        }
        return {res.unmatched_a, res.unmatched_b};
    }
};

// Deep OC-SORT (deep_oc_sort.py): OC-SORT + GMC with the XYAH-safe warp that
// also transforms stored last observations. The ReID/EMA additions are inert
// without an encoder, which the C++ runtime does not ship.
void multi_gmc_xyah(const std::vector<STrack*>& stracks, const float H[6]) {
    const double R[2][2] = {{H[0], H[1]}, {H[3], H[4]}}, t[2] = {H[2], H[5]};
    // T = I8 with the position (0:2) and velocity (4:6) blocks rotated; the
    // aspect/height dims stay identity
    // (deep_oc_sort.DeepOCSortTrack.multi_gmc).
    double T[8][8] = {};
    for (int i = 0; i < 8; i++) T[i][i] = 1;
    T[0][0] = R[0][0];
    T[0][1] = R[0][1];
    T[1][0] = R[1][0];
    T[1][1] = R[1][1];
    T[4][4] = R[0][0];
    T[4][5] = R[0][1];
    T[5][4] = R[1][0];
    T[5][5] = R[1][1];
    for (STrack* st : stracks) {
        double m[8], c[64];
        for (int r = 0; r < 8; r++) {
            m[r] = 0;
            for (int k = 0; k < 8; k++) m[r] += T[r][k] * st->mean[k];
        }
        for (int r = 0; r < 8; r++)
            for (int c2 = 0; c2 < 8; c2++) {
                double acc = 0;
                for (int l = 0; l < 8; l++)
                    for (int k = 0; k < 8; k++)
                        acc += T[r][l] * st->cov[l * 8 + k] * T[c2][k];
                c[r * 8 + c2] = acc;
            }
        m[0] += t[0];
        m[1] += t[1];
        std::copy(m, m + 8, st->mean);
        std::copy(c, c + 64, st->cov);
        // Warp the stored last observation so OCR/ORU stay consistent.
        OCSortTrack* ot = static_cast<OCSortTrack*>(st);
        if (ot->last_observation[0] >= 0) {
            float b[4];
            std::copy(ot->last_observation, ot->last_observation + 4, b);
            const float w = b[2] - b[0], h = b[3] - b[1];
            const float cx = (float)(R[0][0] * (b[0] + w / 2) +
                                     R[0][1] * (b[1] + h / 2) + t[0]);
            const float cy = (float)(R[1][0] * (b[0] + w / 2) +
                                     R[1][1] * (b[1] + h / 2) + t[1]);
            ot->last_observation[0] = cx - w / 2;
            ot->last_observation[1] = cy - h / 2;
            ot->last_observation[2] = cx + w / 2;
            ot->last_observation[3] = cy + h / 2;
        }
    }
}

class DEEPOCSORT : public OCSORT {
public:
    explicit DEEPOCSORT(const TrackConfig& c) : OCSORT(c), gmc_(c.gmc_method) {}

    void reset() override {
        OCSORT::reset();
        gmc_.reset_params();
    }

protected:
    GMC gmc_;

    STrack* make_track(const TrackDet& d) override {
        // DeepOCSortTrack: the confidence-adaptive ReID EMA around
        // alpha_fixed_emb lives on the track row; det_thresh follows
        // args.track_high_thresh (deep_oc_sort.py init_track).
        OCSortTrack* t = new OCSortTrack(d, delta_t);
        t->deep_reid = true;
        t->alpha_fixed_emb = cfg.alpha_fixed_emb;
        t->det_thresh = cfg.track_high_thresh;
        return t;
    }

    // Official _fuse_appearance (deep_oc_sort.py): min-fuse the halved
    // cosine distance, gated by appearance_thresh and the RAW IoU
    // proximity mask, into the motion cost (BoT-SORT-style).
    std::vector<float> fuse_appearance(
            std::vector<float> dists,
            const std::vector<STrack*>& tracks,
            const std::vector<STrack*>& dets,
            const std::vector<float>& iou_dists) override {
        if (!cfg.with_reid || tracks.empty() || dets.empty()) return dists;
        std::vector<float> emb = embedding_distance(tracks, dets);
        const int m = (int)dets.size();
        for (size_t i = 0; i < tracks.size(); i++) {
            for (int j = 0; j < m; j++) {
                float e = emb[(size_t)i * m + j] / 2.0f;
                if (e > 1.0f - cfg.appearance_thresh) e = 1.0f;
                if (!iou_dists.empty() && iou_dists[(size_t)i * m + j] >
                                                  (1.0f - cfg.proximity_thresh))
                    e = 1.0f;
                dists[(size_t)i * m + j] =
                        std::min(dists[(size_t)i * m + j], e);
            }
        }
        return dists;
    }

    void pre_first_associate(const std::vector<STrack*>& pool,
                             const std::vector<STrack*>& unconfirmed,
                             const FrameInput& in,
                             const std::vector<TrackDet>& high) override {
        if (!in.frame || !gmc_.enabled()) return;
        float H[6];
        gmc_.apply(*in.frame, high, H);
        multi_gmc_xyah(pool, H);
        multi_gmc_xyah(unconfirmed, H);
    }
};

// ---- FASTTracker (fast_tracker.py): occlusion-aware ByteTrack variant ----

class FASTTRACKER : public BYTETracker {
public:
    explicit FASTTRACKER(const TrackConfig& c)
        : BYTETracker(c),
          reset_velocity_offset_occ(c.reset_velocity_offset_occ),
          reset_pos_offset_occ(c.reset_pos_offset_occ),
          enlarge_bbox_occ(c.enlarge_bbox_occ),
          dampen_motion_occ(c.dampen_motion_occ),
          active_occ_to_lost_thresh(c.active_occ_to_lost_thresh),
          init_iou_suppress(c.init_iou_suppress),
          occ_cover_thresh(c.occ_cover_thresh),
          occ_reappear_window(c.occ_reappear_window),
          history_len((size_t)std::max(c.reset_velocity_offset_occ,
                                       c.reset_pos_offset_occ) +
                      4) {}

protected:
    int reset_velocity_offset_occ, reset_pos_offset_occ;
    float enlarge_bbox_occ, dampen_motion_occ;
    int active_occ_to_lost_thresh;
    float init_iou_suppress, occ_cover_thresh;
    int occ_reappear_window;
    size_t history_len;

    STrack* make_track(const TrackDet& d) override {
        return new FastSTrack(d, history_len);
    }

    void apply_match(STrack* track,
                     STrack* det,
                     std::vector<STrack*>& activated,
                     std::vector<STrack*>& refind) override {
        BYTETracker::apply_match(track, det, activated, refind);
        FastSTrack* f = static_cast<FastSTrack*>(track);
        f->is_occluded = false;
        f->not_matched = 0;
        f->occluded_len = 0;
    }

    // Second-stage association + occlusion handling (replaces the base
    // mark-lost loop).
    void second_association(const std::vector<STrack*>& strack_pool,
                            std::vector<int>& u_track,
                            std::vector<STrack*>& detections_second,
                            std::vector<STrack*>& activated,
                            std::vector<STrack*>& refind,
                            std::vector<STrack*>& lost) override {
        std::vector<STrack*> r_tracked;
        for (int i : u_track)
            if (strack_pool[i]->state == TrackState::Tracked)
                r_tracked.push_back(strack_pool[i]);
        if (!r_tracked.empty() && !detections_second.empty()) {
            const std::vector<float> cost = iou_distance(
                    track_views(r_tracked), track_views(detections_second));
            const Assignment res =
                    linear_assignment(cost, (int)r_tracked.size(),
                                      (int)detections_second.size(), 0.5f);
            apply_matches(res.matches, r_tracked, detections_second, activated,
                          refind);
            u_track = res.unmatched_a;
        } else {
            u_track.clear();
            for (size_t i = 0; i < r_tracked.size(); i++)
                u_track.push_back((int)i);
        }
        handle_occlusions(r_tracked, u_track, activated, lost);
    }

    // Suppress new tracks that heavily overlap already-active ones. The
    // suppression stack mirrors fast_tracker.py _init_new_tracks: this
    // frame's activated tracks, the re-found tracks, and the still-Tracked
    // in-register pool entries.
    void init_new_tracks(const std::vector<int>& u_detection,
                         const std::vector<STrack*>& detections,
                         std::vector<STrack*>& activated,
                         const std::vector<STrack*>& refind) override {
        std::vector<TrackDet> active_stack;
        for (const STrack* t : activated)
            if (t->is_activated) active_stack.push_back(state_view(t));
        for (const STrack* t : refind)
            if (t->is_activated) active_stack.push_back(state_view(t));
        for (const STrack* t : tracked_stracks)
            if (t->state == TrackState::Tracked)
                active_stack.push_back(state_view(t));
        const bool suppress_on = init_iou_suppress < 1.0f;
        for (int inew : u_detection) {
            STrack* track = detections[inew];
            if (track->score < cfg.new_track_thresh) continue;
            if (suppress_on && !active_stack.empty()) {
                const TrackDet dv = state_view(track);
                float worst = 0;
                for (const TrackDet& a : active_stack)
                    worst = std::max(worst, box_iou_xyxy(det_box_xyxy(dv),
                                                         det_box_xyxy(a)));
                if (worst >= init_iou_suppress) continue;
            }
            track->activate(kf, frame_id);
            activated.push_back(track);
            active_stack.push_back(state_view(track));
        }
    }

    // Lost tracks get a grace window when recently occluded.
    void remove_stale_lost(std::vector<STrack*>& removed) override {
        for (STrack* track : lost_stracks) {
            const FastSTrack* f = static_cast<const FastSTrack*>(track);
            const bool recently_occluded =
                    f->was_recently_occluded &&
                    (frame_id - f->last_occluded_frame <= occ_reappear_window);
            if (!recently_occluded &&
                frame_id - track->end_frame() > max_frames_lost) {
                track->mark_removed();
                removed.push_back(track);
            }
        }
    }

    // Only emit tracks updated this frame (avoids stale idx values).
    std::vector<TrackedBox> format_output() override {
        std::vector<TrackedBox> out;
        for (const STrack* t : tracked_stracks)
            if (t->is_activated && t->frame_id == frame_id)
                out.push_back(track_to_box(t));
        return out;
    }

private:
    static TrackDet state_view(const STrack* t) { return track_to_box(t).det; }

    // Flag unmatched tracked tracks as occluded when covered by an active
    // neighbor.
    void handle_occlusions(const std::vector<STrack*>& r_tracked,
                           const std::vector<int>& u_track,
                           const std::vector<STrack*>& activated,
                           std::vector<STrack*>& lost) {
        if (u_track.empty()) return;
        std::vector<const FastSTrack*> active;
        for (const STrack* t : activated) {
            const FastSTrack* f = static_cast<const FastSTrack*>(t);
            if (f->is_activated && !f->is_occluded) active.push_back(f);
        }
        std::vector<const FastSTrack*> unmatched;
        for (int i : u_track)
            unmatched.push_back(static_cast<const FastSTrack*>(r_tracked[i]));

        std::vector<float> max_cov(unmatched.size(), 0.0f);
        if (!active.empty()) {
            for (size_t u = 0; u < unmatched.size(); u++) {
                float ub[4];
                unmatched[u]->state_xyxy(ub);
                for (const FastSTrack* a : active) {
                    if (a->track_id == unmatched[u]->track_id)
                        continue;  // no self-match
                    float ab[4];
                    a->state_xyxy(ab);
                    max_cov[u] = std::max(
                            max_cov[u], box_ioa({ab[0], ab[1], ab[2], ab[3]},
                                                {ub[0], ub[1], ub[2], ub[3]}));
                }
            }
        }

        for (size_t i = 0; i < unmatched.size(); i++) {
            FastSTrack* track = const_cast<FastSTrack*>(unmatched[i]);
            track->not_matched++;
            if (max_cov[i] > occ_cover_thresh && !track->is_occluded &&
                track->state == TrackState::Tracked) {
                track->is_occluded = true;
                track->occluded_len = 1;
                track->last_occluded_frame = frame_id;
                track->was_recently_occluded = true;
                const auto& hist = track->mean_history;
                if (track->has_mean && !hist.empty()) {
                    if (hist.size() >= (size_t)reset_velocity_offset_occ)
                        for (int k = 0; k < 4; k++)
                            track->mean[4 + k] =
                                    hist[hist.size() -
                                         (size_t)reset_velocity_offset_occ]
                                            .first[(size_t)4 + k];
                    if (hist.size() >= (size_t)reset_pos_offset_occ) {
                        const auto& snap = hist[hist.size() -
                                                (size_t)reset_pos_offset_occ];
                        for (int k = 0; k < 4; k++)
                            track->mean[k] = snap.first[(size_t)k];
                        std::copy(snap.second.begin(), snap.second.end(),
                                  track->cov);
                    }
                    // Enlarge height once (XYAH: w scales via w = a * h) and
                    // dampen motion.
                    track->mean[3] *= enlarge_bbox_occ;
                    for (int k = 4; k < 8; k++)
                        track->mean[k] *= dampen_motion_occ;
                }
            } else if (track->is_occluded) {
                track->occluded_len++;
            }
            if (track->was_recently_occluded &&
                frame_id - track->last_occluded_frame > occ_reappear_window)
                track->was_recently_occluded = false;
            // Grace period before marking lost.
            if (track->state != TrackState::Lost && track->not_matched > 2 &&
                (!track->is_occluded ||
                 track->occluded_len > active_occ_to_lost_thresh)) {
                track->mark_lost();
                lost.push_back(track);
            }
        }
    }
};

// ---- TRACKTRACK (track_tracker.py): multi-cue iterative association + TAI
// ----

// HMIoU distance: (iou_sim, 1 - HIoU * IoU) with HIoU = vertical overlap /
// union.
std::pair<std::vector<float>, std::vector<float>> hmiou_distance(
        const std::vector<TrackDet>& a, const std::vector<TrackDet>& b) {
    const size_t n = a.size(), m = b.size();
    std::vector<float> iou_sim(n * m, 0.0f), hmiou(n * m, 1.0f);
    if (n == 0 || m == 0) return {iou_sim, hmiou};
    for (size_t i = 0; i < n; i++) {
        const Box ba = det_box_xyxy(a[i]);
        for (size_t j = 0; j < m; j++) {
            const Box bb = det_box_xyxy(b[j]);
            const float iou = box_iou_xyxy(ba, bb);
            iou_sim[i * m + j] = iou;
            const float h_over =
                    std::min(ba.y2, bb.y2) - std::max(ba.y1, bb.y1);
            const float h_union =
                    std::max(ba.y2, bb.y2) - std::min(ba.y1, bb.y1);
            const float h_iou =
                    std::clamp(h_over / (h_union + 1e-9f), 0.0f, 1.0f);
            hmiou[i * m + j] = 1.0f - h_iou * iou;
        }
    }
    return {iou_sim, hmiou};
}

// Absolute difference between each track's projected score and each detection's
// confidence.
std::vector<float> confidence_distance(const std::vector<TTSTrack*>& tracks,
                                       const std::vector<TrackDet>& dets) {
    const size_t n = tracks.size(), m = dets.size();
    std::vector<float> out(n * m, 1.0f);
    if (n == 0 || m == 0) return out;
    for (size_t i = 0; i < n; i++) {
        const float proj =
                tracks[i]->score +
                (tracks[i]->score -
                 tracks[i]->prev_score);  // first-order extrapolation
        for (size_t j = 0; j < m; j++)
            out[i * m + j] = std::fabs(proj - dets[j].score);
    }
    return out;
}

// Angle distance over the supported pairs (delta_t = 3 fixed, the Python
// default). Greedy mutually-nearest matching with a threshold that shrinks each
// iteration.
struct IterResult {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks, unmatched_dets;
};

IterResult iterative_associate(const std::vector<float>& cost_in,
                               int n,
                               int m,
                               float match_thr,
                               float reduce_step) {
    IterResult res;
    std::vector<float> cost(cost_in);
    constexpr float kBig = std::numeric_limits<float>::infinity();
    while (n > 0 && m > 0) {
        std::vector<int> nearest_det(n, -1), nearest_track(m, -1);
        for (int i = 0; i < n; i++) {
            float best = kBig;
            for (int j = 0; j < m; j++)
                if (cost[(size_t)i * m + j] < best) {
                    best = cost[(size_t)i * m + j];
                    nearest_det[i] = j;
                }
        }
        for (int j = 0; j < m; j++) {
            float best = kBig;
            for (int i = 0; i < n; i++)
                if (cost[(size_t)i * m + j] < best) {
                    best = cost[(size_t)i * m + j];
                    nearest_track[j] = i;
                }
        }
        std::vector<std::pair<int, int>> new_matches;
        for (int i = 0; i < n; i++) {
            const int j = nearest_det[i];
            if (j >= 0 && nearest_track[j] == i &&
                cost[(size_t)i * m + j] < match_thr)
                new_matches.push_back({i, j});
        }
        if (new_matches.empty()) break;
        for (const auto& [i, j] : new_matches) {
            for (int jj = 0; jj < m; jj++) cost[(size_t)i * m + jj] = kBig;
            for (int ii = 0; ii < n; ii++) cost[(size_t)ii * m + j] = kBig;
        }
        res.matches.insert(res.matches.end(), new_matches.begin(),
                           new_matches.end());
        match_thr -= reduce_step;
    }
    std::vector<char> tdone(n, 0), ddone(m, 0);
    for (const auto& [i, j] : res.matches) tdone[i] = ddone[j] = 1;
    for (int i = 0; i < n; i++)
        if (!tdone[i]) res.unmatched_tracks.push_back(i);
    for (int j = 0; j < m; j++)
        if (!ddone[j]) res.unmatched_dets.push_back(j);
    return res;
}

// TAI NMS: suppress detections that heavily overlap an existing track or a
// stronger detection.
std::vector<char> track_aware_nms(const std::vector<TrackDet>& tracks,
                                  const std::vector<TrackDet>& dets,
                                  float tai_thr,
                                  float new_track_thresh) {
    const size_t n_tracks = tracks.size(), n_dets = dets.size();
    std::vector<char> allow(n_dets, 0);
    for (size_t j = 0; j < n_dets; j++)
        allow[j] = dets[j].score > new_track_thresh;
    if (n_tracks + n_dets < 2 || n_dets == 0) return allow;
    const size_t total = n_tracks + n_dets;
    std::vector<Box> boxes;
    boxes.reserve(total);
    for (const TrackDet& t : tracks) boxes.push_back(det_box_xyxy(t));
    for (const TrackDet& d : dets) boxes.push_back(det_box_xyxy(d));
    std::vector<float> iou((size_t)total * total, 0.0f);
    for (size_t i = 0; i < total; i++)
        for (size_t j = 0; j < total; j++)
            if (i != j) iou[i * total + j] = box_iou_xyxy(boxes[i], boxes[j]);
    if (n_tracks) {
        for (size_t j = 0; j < n_dets; j++) {
            float worst = 0;
            for (size_t t = 0; t < n_tracks; t++)
                worst = std::max(worst, iou[(n_tracks + j) * total + t]);
            if (worst > tai_thr) allow[j] = 0;
        }
    }
    // Score-descending suppression among detections.
    std::vector<size_t> order(n_dets);
    for (size_t j = 0; j < n_dets; j++) order[j] = j;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return dets[a].score > dets[b].score;
    });
    for (size_t oi : order) {
        if (!allow[oi]) continue;
        for (size_t j = 0; j < n_dets; j++) {
            if (j == oi) continue;
            if (iou[(n_tracks + oi) * total + (n_tracks + j)] > tai_thr)
                allow[j] = 0;
        }
    }
    return allow;
}

class TRACKTRACK : public Tracker {
public:
    explicit TRACKTRACK(const TrackConfig& c)
        : cfg(c),
          match_thr(c.match_thresh),
          lost_match_thr(c.lost_match_thr),
          penalty_p(c.penalty_p),
          penalty_q(c.penalty_q),
          reduce_step(c.reduce_step),
          iou_weight(c.iou_weight),
          reid_weight(c.reid_weight),
          conf_weight(c.conf_weight),
          angle_weight(c.angle_weight),
          tai_thr(c.tai_thr),
          new_track_thresh(c.new_track_thresh),
          min_track_len(c.min_track_len),
          max_time_lost(c.track_buffer),
          gmc_(c.gmc_method) {
        kf.kind = KFKind::XYWH;
    }

    std::vector<TrackedBox> update(const FrameInput& in) override {
        frame_id++;
        std::vector<STrack*> activated, refind, lost, removed;

        std::vector<TrackDet> high, low;
        std::vector<std::vector<float>> high_feats, low_feats;
        for (size_t i = 0; i < in.dets.size(); i++) {
            const TrackDet& d = in.dets[i];
            static const std::vector<float> kNoFeat;
            const std::vector<float>& f =
                    i < in.feats.size() ? in.feats[i] : kNoFeat;
            if (d.score >= cfg.track_high_thresh) {
                high.push_back(d);
                high_feats.push_back(f);
            } else if (d.score > cfg.track_low_thresh &&
                       d.score < cfg.track_high_thresh) {
                low.push_back(d);
                low_feats.push_back(f);
            }
        }
        std::vector<STrack*> dets_high = init_track(high, high_feats);
        std::vector<STrack*> dets_low = init_track(low, low_feats);
        std::vector<STrack*> dets_recovered;
        if (!in.dets_del.empty()) {
            std::vector<TrackDet> recovered;
            for (const TrackDet& d : in.dets_del)
                if (d.score > cfg.track_high_thresh) {
                    TrackDet r = d;
                    r.idx = -1;  // recovered detections carry a -1 index
                                 // upstream
                    recovered.push_back(r);
                }
            dets_recovered = init_track(recovered);
        }

        std::vector<STrack*> unconfirmed, tracked;
        // Route by state: frame-1 tracks are visible (is_activated) but still
        // New, so they must stay unconfirmed (track_tracker.py update).
        for (STrack* t : tracked_stracks)
            (t->state == TrackState::Tracked ? tracked : unconfirmed)
                    .push_back(t);
        std::vector<STrack*> pool = tracked;
        joint_by_id(pool, lost_stracks);

        if (in.frame && gmc_.enabled()) {
            float H[6];
            gmc_.apply(*in.frame, high, H);
            multi_gmc(pool, H);
            multi_gmc(unconfirmed, H);
        }
        for (STrack* t : pool) t->predict();
        // multi_predict(unconfirmed): the unconfirmed pool enters the second
        // association through Kalman-updated states, exactly like the main
        // pool (track_tracker.py update).
        for (STrack* t : unconfirmed) t->predict();

        // Main association: pool vs (high + low + recovered), per-bucket cost
        // penalties.
        std::vector<STrack*> all_dets = dets_high;
        all_dets.insert(all_dets.end(), dets_low.begin(), dets_low.end());
        all_dets.insert(all_dets.end(), dets_recovered.begin(),
                        dets_recovered.end());
        const int n_high = (int)dets_high.size(), n_low = (int)dets_low.size();
        std::vector<float> cost = cost_matrix(pool, all_dets);
        const int m = (int)all_dets.size();
        for (int i = 0; i < (int)pool.size(); i++) {
            for (int j = n_high; j < n_high + n_low; j++)
                cost[(size_t)i * m + j] += penalty_p;
            for (int j = n_high + n_low; j < m; j++)
                cost[(size_t)i * m + j] += penalty_q;
        }
        for (float& v : cost) v = std::clamp(v, 0.0f, 1.0f);

        const IterResult res = iterative_associate(cost, (int)pool.size(), m,
                                                   match_thr, reduce_step);
        for (const auto& [ti, di] : res.matches) {
            STrack* track = pool[ti];
            STrack* det = all_dets[di];
            if (track->state == TrackState::Tracked) {
                track->update(*det, frame_id);
                activated.push_back(track);
            } else {
                track->re_activate(*det, frame_id, false);
                refind.push_back(track);
            }
        }
        for (int ti : res.unmatched_tracks) {
            STrack* track = pool[ti];
            if (track->state != TrackState::Lost) {
                track->mark_lost();
                lost.push_back(track);
            }
        }

        // Second association: unconfirmed tracks vs leftover high-confidence
        // detections.
        std::vector<STrack*> leftover;
        for (int di : res.unmatched_dets)
            if (di < n_high) leftover.push_back(all_dets[di]);
        if (!unconfirmed.empty() && !leftover.empty()) {
            const std::vector<float> uc_cost =
                    cost_matrix(unconfirmed, leftover);
            const IterResult uc = iterative_associate(
                    uc_cost, (int)unconfirmed.size(), (int)leftover.size(),
                    match_thr, reduce_step);
            for (const auto& [ti, di] : uc.matches) {
                unconfirmed[ti]->update(*leftover[di], frame_id);
                activated.push_back(unconfirmed[ti]);
            }
            for (int ti : uc.unmatched_tracks) {
                unconfirmed[ti]->mark_removed();
                removed.push_back(unconfirmed[ti]);
            }
            std::vector<STrack*> remaining;
            for (int di : uc.unmatched_dets) remaining.push_back(leftover[di]);
            leftover = remaining;
        } else {
            for (STrack* t : unconfirmed) {
                t->mark_removed();
                removed.push_back(t);
            }
        }

        // Optional relaxed rebind for still-Lost tracks (disabled when
        // lost_match_thr <= 0).
        if (lost_match_thr > 0 && !leftover.empty()) {
            std::vector<STrack*> unmatched_lost;
            for (STrack* t : pool)
                if (t->state == TrackState::Lost &&
                    std::find(lost.begin(), lost.end(), t) == lost.end())
                    unmatched_lost.push_back(t);
            if (!unmatched_lost.empty()) {
                const std::vector<float> lost_cost =
                        cost_matrix(unmatched_lost, leftover);
                const IterResult lr = iterative_associate(
                        lost_cost, (int)unmatched_lost.size(),
                        (int)leftover.size(), lost_match_thr, reduce_step);
                for (const auto& [ti, di] : lr.matches) {
                    unmatched_lost[ti]->re_activate(*leftover[di], frame_id,
                                                    false);
                    refind.push_back(unmatched_lost[ti]);
                }
                std::vector<STrack*> remaining;
                for (int di : lr.unmatched_dets)
                    remaining.push_back(leftover[di]);
                leftover = remaining;
            }
        }

        // TAI: spawn new tracks from leftover detections that survive NMS
        // against active tracks.
        std::vector<STrack*> active = tracked;
        for (STrack* t : tracked_stracks)
            if (t->state == TrackState::Tracked &&
                std::find(active.begin(), active.end(), t) == active.end())
                active.push_back(t);
        active.insert(active.end(), activated.begin(), activated.end());
        const std::vector<TrackDet> active_views = views(active),
                                    leftover_views = views(leftover);
        const std::vector<char> allow = track_aware_nms(
                active_views, leftover_views, tai_thr, new_track_thresh);
        for (size_t i = 0; i < leftover.size(); i++)
            if (allow[i]) {
                leftover[i]->activate(kf, frame_id);
                activated.push_back(leftover[i]);
            }

        for (STrack* track : lost_stracks)
            if (frame_id - track->end_frame() > max_time_lost) {
                track->mark_removed();
                removed.push_back(track);
            }

        merge_track_pools(tracked_stracks, lost_stracks, removed_stracks,
                          activated, refind, lost, removed);
        prune_arena();
        std::vector<TrackedBox> out;
        for (const STrack* t : tracked_stracks)
            if (t->is_activated && t->frame_id == frame_id)
                out.push_back(track_to_box(t));
        return out;
    }

    void reset() override {
        tracked_stracks.clear();
        lost_stracks.clear();
        removed_stracks.clear();
        arena.clear();
        frame_id = 0;
        kf.kind = KFKind::XYWH;
        STrack::reset_id();  // TTSTrack.reset_id upstream
        gmc_.reset_params();
    }

private:
    const TrackConfig& cfg;
    int frame_id = 0;
    float match_thr, lost_match_thr, penalty_p, penalty_q, reduce_step;
    float iou_weight, reid_weight, conf_weight, angle_weight, tai_thr,
            new_track_thresh;
    int min_track_len, max_time_lost;
    KalmanFilter kf;
    GMC gmc_;
    std::vector<std::unique_ptr<STrack>> arena;
    std::vector<STrack*> tracked_stracks, lost_stracks, removed_stracks;

    STrack* make_track(const TrackDet& d) {
        return new TTSTrack(d, 3, min_track_len);
    }

    std::vector<STrack*> init_track(
            const std::vector<TrackDet>& dets,
            const std::vector<std::vector<float>>& feats = {}) {
        std::vector<STrack*> out;
        out.reserve(dets.size());
        for (size_t i = 0; i < dets.size(); i++) {
            arena.emplace_back(make_track(dets[i]));
            STrack* st = arena.back().get();
            // TTSTrack.update_features: score-adaptive beta blending.
            if (i < feats.size() && !feats[i].empty())
                st->update_features(feats[i]);
            out.push_back(st);
        }
        return out;
    }

    static std::vector<TrackDet> views(const std::vector<STrack*>& tracks) {
        std::vector<TrackDet> out;
        out.reserve(tracks.size());
        for (const STrack* t : tracks) {
            TrackDet d;
            float xywh[4];
            t->state_xywh(xywh);
            d.cx = xywh[0];
            d.cy = xywh[1];
            d.w = xywh[2];
            d.h = xywh[3];
            d.angle = t->angled() ? t->angle : -10.0f;
            d.score = t->score;
            d.class_id = t->cls;
            d.idx = t->idx;
            out.push_back(d);
        }
        return out;
    }

    // Multi-cue cost: HMIoU (+ ReID cosine + confidence + angle), gated by
    // IoU support. The ReID term follows _cost_matrix: cosine between the
    // track's smooth (or current) feature and the detection's current
    // feature, fused as iou_weight * hmiou + reid_weight * cos; pairs with
    // a missing side fall back to pure HMIoU (the embedding neither helps
    // nor penalizes them), which also covers the no-feature runtime.
    std::vector<float> cost_matrix(const std::vector<STrack*>& tracks,
                                   const std::vector<STrack*>& dets) {
        const std::vector<TrackDet> tv = views(tracks), dv = views(dets);
        const auto [iou_sim, hmiou] = hmiou_distance(tv, dv);
        std::vector<TTSTrack*> tt;
        tt.reserve(tracks.size());
        for (STrack* t : tracks) tt.push_back(static_cast<TTSTrack*>(t));
        std::vector<float> cost = hmiou;
        if (cfg.with_reid && !tracks.empty() && !dets.empty()) {
            const int m = (int)dets.size();
            for (int i = 0; i < (int)tracks.size(); i++) {
                // Python: tfeat = smooth_feat if not None else curr_feat.
                const std::vector<float>& tf = tt[i]->smooth_feat.empty()
                                                       ? tt[i]->curr_feat
                                                       : tt[i]->smooth_feat;
                for (int j = 0; j < m; j++) {
                    const std::vector<float>& df = dets[j]->curr_feat;
                    if (tf.empty() || df.empty() || tf.size() != df.size())
                        continue;  // NaN cos -> pure HMIoU for this pair
                    float dot = 0.0f;
                    for (size_t k = 0; k < tf.size(); k++) dot += tf[k] * df[k];
                    const size_t idx = (size_t)i * m + j;
                    cost[idx] =
                            iou_weight * cost[idx] + reid_weight * (1.0f - dot);
                }
            }
        }
        const std::vector<float> cd = confidence_distance(tt, dv);
        for (size_t k = 0; k < cost.size(); k++) cost[k] += conf_weight * cd[k];
        const int m = (int)dv.size();
        std::vector<std::pair<int, int>> pairs;
        for (int i = 0; i < (int)tracks.size(); i++)
            for (int j = 0; j < m; j++)
                if (iou_sim[(size_t)i * m + j] > 0.10f) pairs.push_back({i, j});
        if (!pairs.empty()) {
            const std::vector<float> ad =
                    angle_distance_impl(tt, dv, frame_id, pairs);
            for (size_t k = 0; k < pairs.size(); k++) {
                const auto& [i, j] = pairs[k];
                cost[(size_t)i * m + j] += angle_weight * ad[k];
            }
            std::vector<char> supported((size_t)tracks.size() * m, 0);
            for (const auto& [i, j] : pairs) supported[(size_t)i * m + j] = 1;
            for (int i = 0; i < (int)tracks.size(); i++)
                for (int j = 0; j < m; j++)
                    if (!supported[(size_t)i * m + j])
                        cost[(size_t)i * m + j] = 1.0f;
        } else {
            for (float& v : cost) v = 1.0f;
        }
        for (float& v : cost) v = std::clamp(v, 0.0f, 1.0f);
        (void)iou_weight;
        return cost;
    }

    // Corner-angle distance over the supported pairs (Python _angle_distance).
    std::vector<float> angle_distance_impl(
            const std::vector<TTSTrack*>& tracks,
            const std::vector<TrackDet>& dets,
            int fid,
            const std::vector<std::pair<int, int>>& pairs) {
        static constexpr int kCornerDx[4] = {0, 0, 2, 2};
        static constexpr int kCornerDy[4] = {1, 3, 1, 3};
        std::vector<float> out;
        out.reserve(pairs.size());
        for (const auto& [ti, dj] : pairs) {
            float tb[4];
            tracks[ti]->history_box(
                    fid, 3, tb);  // delta_t = 3, the Python function default
            const float db[4] = {
                    dets[dj].cx - dets[dj].w / 2, dets[dj].cy - dets[dj].h / 2,
                    dets[dj].cx + dets[dj].w / 2, dets[dj].cy + dets[dj].h / 2};
            float acc = 0;
            for (int k = 0; k < 4; k++) {
                float dx = db[kCornerDx[k]] - tb[kCornerDx[k]];
                float dy = db[kCornerDy[k]] - tb[kCornerDy[k]];
                const float norm = std::sqrt(dx * dx + dy * dy) + 1e-5f;
                dx /= norm;
                dy /= norm;
                const float dot =
                        std::clamp(tracks[ti]->velocity[k][0] * dx +
                                           tracks[ti]->velocity[k][1] * dy,
                                   -1.0f, 1.0f);
                acc += std::fabs(std::acos(dot));
            }
            out.push_back(acc / 4.0f / 3.14159265f * dets[dj].score);
        }
        return out;
    }

    void prune_arena() {
        std::vector<STrack*> keep;
        keep.reserve(tracked_stracks.size() + lost_stracks.size() +
                     removed_stracks.size());
        keep.insert(keep.end(), tracked_stracks.begin(), tracked_stracks.end());
        keep.insert(keep.end(), lost_stracks.begin(), lost_stracks.end());
        keep.insert(keep.end(), removed_stracks.begin(), removed_stracks.end());
        std::sort(keep.begin(), keep.end());
        arena.erase(std::remove_if(arena.begin(), arena.end(),
                                   [&](const std::unique_ptr<STrack>& p) {
                                       return !std::binary_search(keep.begin(),
                                                                  keep.end(),
                                                                  p.get());
                                   }),
                    arena.end());
    }
};

// ---- configuration ---------------------------------------------------------

bool default_tracker_config(const std::string& type, TrackConfig& out) {
    out = TrackConfig();
    out.tracker_type = type;
    if (type == "bytetrack") {
        // mirrors ultralytics/cfg/trackers/bytetrack.yaml
    } else if (type == "botsort") {
        // mirrors ultralytics/cfg/trackers/botsort.yaml
        out.gmc_method = "sparseOptFlow";
        out.proximity_thresh = 0.5f;
        out.appearance_thresh = 0.8f;
        out.with_reid = false;
        out.model = "auto";
    } else if (type == "ocsort") {
        // mirrors ultralytics/cfg/trackers/ocsort.yaml
        out.delta_t = 3;
        out.inertia = 0.2f;
        out.use_byte = false;
    } else if (type == "deepocsort") {
        // mirrors ultralytics/cfg/trackers/deepocsort.yaml
        out.track_high_thresh = 0.3f;
        out.new_track_thresh = 0.3f;
        out.delta_t = 3;
        out.inertia = 0.2f;
        out.use_byte = false;
        out.gmc_method = "none";
        out.proximity_thresh = 0.5f;
        out.appearance_thresh = 0.9f;
        out.alpha_fixed_emb = 0.95f;
    } else if (type == "fasttrack") {
        // mirrors ultralytics/cfg/trackers/fasttrack.yaml
        out.reset_velocity_offset_occ = 5;
        out.reset_pos_offset_occ = 3;
        out.enlarge_bbox_occ = 1.1f;
        out.dampen_motion_occ = 0.5f;
        out.active_occ_to_lost_thresh = 10;
        out.occ_cover_thresh = 0.7f;
        out.occ_reappear_window = 40;
        out.init_iou_suppress = 0.7f;
    } else if (type == "tracktrack") {
        // mirrors ultralytics/cfg/trackers/tracktrack.yaml
        out.track_high_thresh = 0.6f;
        out.track_low_thresh = 0.25f;
        out.new_track_thresh = 0.7f;
        out.match_thresh = 0.7f;
        out.lost_match_thr = 0.0f;
        out.penalty_p = 0.2f;
        out.penalty_q = 0.4f;
        out.reduce_step = 0.05f;
        out.tai_thr = 0.55f;
        out.min_track_len = 3;
        out.gmc_method = "sparseOptFlow";
        out.with_reid = false;
    } else {
        return false;
    }
    return true;
}

bool load_tracker_config_yaml(const std::string& path, TrackConfig& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "failed to open tracker config %s\n", path.c_str());
        return false;
    }
    std::string content;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) content.append(buf, n);
    fclose(f);

    auto trim = [](std::string s) {
        const auto b = s.find_first_not_of(" \t\r");
        if (b == std::string::npos) return std::string();
        const auto e = s.find_last_not_of(" \t\r");
        return s.substr(b, e - b + 1);
    };
    size_t pos = 0;
    while (pos < content.size()) {
        size_t eol = content.find('\n', pos);
        if (eol == std::string::npos) eol = content.size();
        std::string line = content.substr(pos, eol - pos);
        pos = eol + 1;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        if (key.empty() || value.empty()) continue;
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);
        auto as_f = [&](float def) {
            return value.empty() ? def : strtof(value.c_str(), nullptr);
        };
        auto as_i = [&](int def) {
            return value.empty() ? def : atoi(value.c_str());
        };
        auto as_b = [&](bool def) {
            if (value == "True" || value == "true") return true;
            if (value == "False" || value == "false") return false;
            return def;
        };
        if (key == "tracker_type")
            out.tracker_type = value;
        else if (key == "track_high_thresh")
            out.track_high_thresh = as_f(out.track_high_thresh);
        else if (key == "track_low_thresh")
            out.track_low_thresh = as_f(out.track_low_thresh);
        else if (key == "new_track_thresh")
            out.new_track_thresh = as_f(out.new_track_thresh);
        else if (key == "track_buffer")
            out.track_buffer = as_i(out.track_buffer);
        else if (key == "match_thresh")
            out.match_thresh = as_f(out.match_thresh);
        else if (key == "fuse_score")
            out.fuse_score = as_b(out.fuse_score);
        else if (key == "delta_t")
            out.delta_t = as_i(out.delta_t);
        else if (key == "inertia")
            out.inertia = as_f(out.inertia);
        else if (key == "use_byte")
            out.use_byte = as_b(out.use_byte);
        else if (key == "gmc_method")
            out.gmc_method = value;
        else if (key == "proximity_thresh")
            out.proximity_thresh = as_f(out.proximity_thresh);
        else if (key == "appearance_thresh")
            out.appearance_thresh = as_f(out.appearance_thresh);
        else if (key == "with_reid")
            out.with_reid = as_b(out.with_reid);
        else if (key == "model")
            out.model = value;
        else if (key == "alpha_fixed_emb")
            out.alpha_fixed_emb = as_f(out.alpha_fixed_emb);
        else if (key == "reset_velocity_offset_occ")
            out.reset_velocity_offset_occ = as_i(out.reset_velocity_offset_occ);
        else if (key == "reset_pos_offset_occ")
            out.reset_pos_offset_occ = as_i(out.reset_pos_offset_occ);
        else if (key == "enlarge_bbox_occ")
            out.enlarge_bbox_occ = as_f(out.enlarge_bbox_occ);
        else if (key == "dampen_motion_occ")
            out.dampen_motion_occ = as_f(out.dampen_motion_occ);
        else if (key == "active_occ_to_lost_thresh")
            out.active_occ_to_lost_thresh = as_i(out.active_occ_to_lost_thresh);
        else if (key == "occ_cover_thresh")
            out.occ_cover_thresh = as_f(out.occ_cover_thresh);
        else if (key == "occ_reappear_window")
            out.occ_reappear_window = as_i(out.occ_reappear_window);
        else if (key == "init_iou_suppress")
            out.init_iou_suppress = as_f(out.init_iou_suppress);
        else if (key == "lost_match_thr")
            out.lost_match_thr = as_f(out.lost_match_thr);
        else if (key == "penalty_p")
            out.penalty_p = as_f(out.penalty_p);
        else if (key == "penalty_q")
            out.penalty_q = as_f(out.penalty_q);
        else if (key == "reduce_step")
            out.reduce_step = as_f(out.reduce_step);
        else if (key == "iou_weight")
            out.iou_weight = as_f(out.iou_weight);
        else if (key == "reid_weight")
            out.reid_weight = as_f(out.reid_weight);
        else if (key == "conf_weight")
            out.conf_weight = as_f(out.conf_weight);
        else if (key == "angle_weight")
            out.angle_weight = as_f(out.angle_weight);
        else if (key == "tai_thr")
            out.tai_thr = as_f(out.tai_thr);
        else if (key == "min_track_len")
            out.min_track_len = as_i(out.min_track_len);
        // Unknown keys are ignored: the Python trackers read each knob with
        // getattr(..., default), so extra YAML keys have no effect there
        // either.
    }
    return true;
}

bool resolve_tracker_config(const std::string& spec, TrackConfig& out) {
    if (spec.size() > 4 && (spec.rfind(".yaml") == spec.size() - 5 ||
                            spec.rfind(".yml") == spec.size() - 4))
        return load_tracker_config_yaml(spec, out);
    if (!default_tracker_config(spec, out)) {
        fprintf(stderr,
                "unknown tracker '%s' (expected "
                "bytetrack|botsort|ocsort|deepocsort|fasttrack|tracktrack "
                "or a tracker YAML path)\n",
                spec.c_str());
        return false;
    }
    return true;
}

std::unique_ptr<Tracker> create_tracker(const TrackConfig& cfg) {
    // with_reid=true engages the ReID cosine term through FrameInput.feats
    // (the official model="auto" detector-feature path); trackers that
    // never consume features (bytetrack/ocsort/fasttrack) simply ignore it,
    // and a featureless stream degrades to motion-only association.
    const bool needs_gmc = cfg.tracker_type == "botsort" ||
                           cfg.tracker_type == "deepocsort" ||
                           cfg.tracker_type == "tracktrack";
    (void)needs_gmc;  // only read on builds with restricted GMC support
#if !defined(QYOLO_WITH_OPENCV)
    if (needs_gmc && cfg.gmc_method != "sparseOptFlow" &&
        cfg.gmc_method != "none" && cfg.gmc_method != "None") {
        fprintf(stderr,
                "gmc_method '%s' requires an OpenCV build (QYOLO_WITH_OPENCV); "
                "this runtime was built "
                "without OpenCV and supports sparseOptFlow|none only\n",
                cfg.gmc_method.c_str());
        return nullptr;
    }
#endif
#if !defined(QYOLO_WITH_OPENCV_SIFT)
    if (needs_gmc && cfg.gmc_method == "sift") {
        fprintf(stderr,
                "gmc_method 'sift' requires OpenCV >= 4.4 (SIFT moved into the "
                "main library; "
                "this build has OpenCV without it). Use "
                "sparseOptFlow|orb|ecc\n");
        return nullptr;
    }
#endif
    if (cfg.tracker_type == "bytetrack")
        return std::make_unique<BYTETracker>(cfg);
    if (cfg.tracker_type == "botsort") return std::make_unique<BOTSORT>(cfg);
    if (cfg.tracker_type == "ocsort") return std::make_unique<OCSORT>(cfg);
    if (cfg.tracker_type == "deepocsort")
        return std::make_unique<DEEPOCSORT>(cfg);
    if (cfg.tracker_type == "fasttrack")
        return std::make_unique<FASTTRACKER>(cfg);
    if (cfg.tracker_type == "tracktrack")
        return std::make_unique<TRACKTRACK>(cfg);
    fprintf(stderr, "unknown tracker_type '%s'\n", cfg.tracker_type.c_str());
    return nullptr;
}

}  // namespace track
}  // namespace qyolo
