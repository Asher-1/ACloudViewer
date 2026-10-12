// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Implementation of the pure MapAnything geometry helpers. Every formula is
// a direct port of the official Python utilities shipped in
// map-anything-ggml (vggt_omega/utils/{pose_enc,rotation,geometry}.py) so
// the reconstructed cloud lands in the same world frame as the upstream
// demo visualizations.

#include "MapAnythingGeometry.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

namespace MapAnythingGeometry {

// Official quat_to_mat (PyTorch3D-derived, scalar-LAST xyzw):
//   i, j, k, r = q; two_s = 2 / (q . q)
void quat_xyzw_to_mat3(const float* q, float* R) {
    const float i = q[0], j = q[1], k = q[2], r = q[3];
    const float two_s = 2.0f / (i * i + j * j + k * k + r * r);
    R[0] = 1 - two_s * (j * j + k * k);
    R[1] = two_s * (i * j - k * r);
    R[2] = two_s * (i * k + j * r);
    R[3] = two_s * (i * j + k * r);
    R[4] = 1 - two_s * (i * i + k * k);
    R[5] = two_s * (j * k - i * r);
    R[6] = two_s * (i * k - j * r);
    R[7] = two_s * (j * k + i * r);
    R[8] = 1 - two_s * (i * i + j * j);
}

void pose_enc_to_w2c(const float* pose9, float* R, float* t) {
    // The upstream VGGT-Omega pose_enc.py defines [R|T] as camera-from-world.
    // Preserve the encoded matrix here; unproject_depth_pixel performs its
    // inverse, world = R^T * (cam - T).
    quat_xyzw_to_mat3(pose9 + 3, R);
    t[0] = pose9[0];
    t[1] = pose9[1];
    t[2] = pose9[2];
}

void fov_to_intrinsics(
        float fov_h, float fov_w, int width, int height, float* out_k) {
    // Official encoding_to_camera: fy = (H/2)/tan(fov_h/2),
    // fx = (W/2)/tan(fov_w/2), cx = W/2, cy = H/2.
    const float fy = (height / 2.0f) / std::tan(std::max(fov_h, 1e-4f) / 2.0f);
    const float fx = (width / 2.0f) / std::tan(std::max(fov_w, 1e-4f) / 2.0f);
    out_k[0] = fx;
    out_k[1] = fy;
    out_k[2] = width / 2.0f;
    out_k[3] = height / 2.0f;
}

void unproject_depth_pixel(const float* R,
                           const float* t,
                           const float* k,
                           int u,
                           int v,
                           float depth,
                           float* out_world) {
    // Pinhole unproject to the OpenCV camera frame, then invert the
    // upstream camera-from-world pose: world = R^T * (cam - T).
    const float fx = k[0], fy = k[1], cx = k[2], cy = k[3];
    const float x = (u - cx) * depth / fx - t[0];
    const float y = (v - cy) * depth / fy - t[1];
    const float z = depth - t[2];
    out_world[0] = R[0] * x + R[3] * y + R[6] * z;
    out_world[1] = R[1] * x + R[4] * y + R[7] * z;
    out_world[2] = R[2] * x + R[5] * y + R[8] * z;
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

void transform_by_c2w(const float* pose16, const float* cam, float* out_world) {
    // Row-major 4x4 camera-TO-world: world = M[:3,:3] * cam + M[:3,3]
    // (translation lives in the flat 3/7/11 slots).
    for (int c = 0; c < 3; ++c) {
        out_world[c] = pose16[c * 4 + 0] * cam[0] + pose16[c * 4 + 1] * cam[1] +
                       pose16[c * 4 + 2] * cam[2] + pose16[c * 4 + 3];
    }
}

// ---- official end-to-end filter stack (port of upstream
// cpp_ggml/scripts/e2e_full_scene.py; formulas kept 1:1 with the Python
// reference: depth_edge / official percentile floor / conf-priority voxel
// merge / robust Horn Sim3) ---------------------------------------------

void depth_edge_mask(
        const float* depth, int h, int w, float rtol, uint8_t* out_mask) {
    // 3x3 window min/max through edge padding, exactly like np.pad(edge).
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float inf = std::numeric_limits<float>::infinity();
            float dmax = -inf, dmin = inf;
            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = std::min(std::max(y + dy, 0), h - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = std::min(std::max(x + dx, 0), w - 1);
                    const float d = depth[(size_t)yy * w + xx];
                    dmax = std::max(dmax, d);
                    dmin = std::min(dmin, d);
                }
            }
            const float d = depth[(size_t)y * w + x];
            const float rel = (dmax - dmin) / std::max(std::abs(d), 1e-6f);
            out_mask[(size_t)y * w + x] = rel > rtol ? 1 : 0;
        }
    }
}

float percentile_value(const float* v, size_t n, float percentile) {
    std::vector<float> vals;
    vals.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(v[i])) vals.push_back(v[i]);
    }
    if (vals.empty()) return std::numeric_limits<float>::quiet_NaN();
    // Linear interpolation between the two closest ranks: the default of
    // both np.percentile (vggt visual_util) and torch.quantile (MapAnything
    // inference.py). O(n) average: two selections, no full sort.
    const double pos = double(std::max(0.0f, std::min(100.0f, percentile))) /
                       100.0 * double(vals.size() - 1);
    const size_t lo = std::min(vals.size() - 1, size_t(pos));
    std::nth_element(vals.begin(), vals.begin() + lo, vals.end());
    const float vlo = vals[lo];
    if (lo + 1 >= vals.size()) return vlo;
    // after nth_element every element right of lo is >= vals[lo]; the next
    // rank is their minimum.
    const float vhi = *std::min_element(vals.begin() + lo + 1, vals.end());
    return float(double(vlo) +
                 (double(vhi) - double(vlo)) * (pos - double(lo)));
}

float official_conf_floor(const float* conf, size_t n, float percentile) {
    const float p = percentile_value(conf, n, percentile);
    // official visual_util.predictions_to_glb: floor = max(2.0, p)
    return std::isfinite(p) ? std::max(2.0f, p) : 2.0f;
}

std::vector<size_t> conf_priority_voxel_indices(const float* pts,
                                                const float* confs,
                                                size_t n,
                                                float voxel_size) {
    std::vector<size_t> keep;
    if (n == 0 || voxel_size <= 0.0f) {
        keep.resize(n);
        for (size_t i = 0; i < n; ++i) keep[i] = i;
        return keep;
    }
    // descending-conf order: the first hit per voxel is the max-conf one.
    // O(n log n) sort: the ownership rule needs a total confidence order.
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return confs[a] > confs[b]; });
    // linear-probe hash over the quantized voxel key
    struct Key {
        long long x, y, z;
        bool operator==(const Key& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    auto keyOf = [&](size_t i) {
        return Key{(long long)std::floor(pts[3 * i + 0] / voxel_size),
                   (long long)std::floor(pts[3 * i + 1] / voxel_size),
                   (long long)std::floor(pts[3 * i + 2] / voxel_size)};
    };
    auto keyHash = [](const Key& k) {
        unsigned long long h = 1469598103934665603ull;
        const long long v[3] = {k.x, k.y, k.z};
        for (int c = 0; c < 3; ++c) {
            h ^= (unsigned long long)v[c];
            h *= 1099511628211ull;
        }
        return h;
    };
    std::vector<Key> keys(n);
    for (size_t i = 0; i < n; ++i) keys[i] = keyOf(i);
    const size_t tableSize = n * 2 + 1;
    std::vector<long long> slots(tableSize, -1);  // -> point index
    auto slotOf = [&](const Key& k) -> long long {
        size_t h = keyHash(k) % tableSize;
        while (true) {
            if (slots[h] < 0) return (long long)h;
            if (keys[slots[h]] == k) return -2 - slots[h];  // occupied by key
            h = (h + 1) % tableSize;
        }
    };
    keep.reserve(n / 2);
    for (size_t oi = 0; oi < n; ++oi) {
        const size_t i = order[oi];
        const long long h = slotOf(keys[i]);
        if (h >= 0) {
            // Store the original point index: slotOf() compares this entry's
            // quantized key on later probes. Storing oi (the rank in the
            // confidence-sorted order) compares an unrelated point and makes
            // voxel ownership nondeterministically wrong.
            slots[h] = static_cast<long long>(i);
            keep.push_back(i);
        }
    }
    std::sort(keep.begin(), keep.end());
    return keep;
}

size_t conf_priority_voxel(float* pts,
                           uint8_t* cols,
                           float* confs,
                           size_t n,
                           float voxel_size,
                           std::vector<size_t>* survivor_indices) {
    const std::vector<size_t> keep =
            conf_priority_voxel_indices(pts, confs, n, voxel_size);
    if (survivor_indices != nullptr) *survivor_indices = keep;
    for (size_t out = 0; out < keep.size(); ++out) {
        const size_t i = keep[out];
        if (out != i) {
            pts[3 * out + 0] = pts[3 * i + 0];
            pts[3 * out + 1] = pts[3 * i + 1];
            pts[3 * out + 2] = pts[3 * i + 2];
            cols[3 * out + 0] = cols[3 * i + 0];
            cols[3 * out + 1] = cols[3 * i + 1];
            cols[3 * out + 2] = cols[3 * i + 2];
            confs[out] = confs[i];
        }
    }
    return keep.size();
}

int horn_sim3(const float* src,
              const float* dst,
              size_t n,
              float* out_s,
              float* out_R,
              float* out_t) {
    if (n < 3) return -1;
    Eigen::Matrix<double, Eigen::Dynamic, 3> S(n, 3), D(n, 3);
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            S(i, c) = double(src[3 * i + c]);
            D(i, c) = double(dst[3 * i + c]);
        }
    }
    const Eigen::RowVector3d muS = S.colwise().mean();
    const Eigen::RowVector3d muD = D.colwise().mean();
    const Eigen::MatrixXd Sc = S.rowwise() - muS;
    const Eigen::MatrixXd cov =
            (D.rowwise() - muD).transpose() * Sc / double(n);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    Eigen::Matrix3d Sv = Eigen::Matrix3d::Identity();
    Sv(2, 2) = (U * V).determinant() >= 0 ? 1.0 : -1.0;
    const Eigen::Matrix3d R = U * Sv * V.transpose();
    const double var = Sc.squaredNorm() / double(n);
    const double s =
            var > 0.0 ? (svd.singularValues().asDiagonal() * Sv).trace() / var
                      : 1.0;
    const Eigen::RowVector3d t = muD - s * (muS * R.transpose());
    out_s[0] = float(s);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out_R[3 * r + c] = float(R(r, c));
        out_t[r] = float(t(r));
    }
    return 0;
}

float median(std::vector<float> v) {
    if (v.empty()) return std::numeric_limits<float>::quiet_NaN();
    // O(n) selection; numpy.median semantics: the mean of the two middle
    // order statistics when the count is even.
    const size_t h = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const float hi = v[h];
    if (v.size() % 2 != 0) return hi;
    const float lo = *std::max_element(v.begin(), v.begin() + h);
    return 0.5f * (lo + hi);
}

int robust_sim3(const float* src,
                const float* dst,
                size_t n,
                float* out_s,
                float* out_R,
                float* out_t) {
    if (n < 3) return -1;
    std::vector<uint8_t> keep(n, 1);
    std::vector<float> res(n, 0.0f);
    for (int it = 0; it < 2; ++it) {
        std::vector<float> sSub, dSub;
        for (size_t i = 0; i < n; ++i) {
            if (keep[i]) {
                sSub.insert(sSub.end(), src + 3 * i, src + 3 * i + 3);
                dSub.insert(dSub.end(), dst + 3 * i, dst + 3 * i + 3);
            }
        }
        if (sSub.size() < 9) break;
        if (horn_sim3(sSub.data(), dSub.data(), sSub.size() / 3, out_s, out_R,
                      out_t) != 0)
            return -1;
        // residuals over ALL points (official semantics)
        for (size_t i = 0; i < n; ++i) {
            float p[3];
            apply_sim3(out_R, out_t, out_s[0], src + 3 * i, 1, p);
            res[i] = std::sqrt(std::pow(dst[3 * i + 0] - p[0], 2) +
                               std::pow(dst[3 * i + 1] - p[1], 2) +
                               std::pow(dst[3 * i + 2] - p[2], 2));
        }
        std::vector<float> kept;
        for (size_t i = 0; i < n; ++i) {
            if (keep[i]) kept.push_back(res[i]);
        }
        const float med = median(kept);
        std::vector<float> dev;
        for (float r : kept) dev.push_back(std::abs(r - med));
        const float mad = median(std::move(dev)) + 1e-12f;
        for (size_t i = 0; i < n; ++i)
            keep[i] = res[i] <= med + 3.0f * 1.4826f * mad ? 1 : 0;
    }
    return 0;
}

void apply_sim3(const float* R,
                const float* t,
                float s,
                const float* pts,
                size_t n,
                float* out) {
    for (size_t i = 0; i < n; ++i) {
        const float p[3] = {pts[3 * i + 0], pts[3 * i + 1], pts[3 * i + 2]};
        for (int r = 0; r < 3; ++r) {
            out[3 * i + r] = s * (R[3 * r + 0] * p[0] + R[3 * r + 1] * p[1] +
                                  R[3 * r + 2] * p[2]) +
                             t[r];
        }
    }
}

Sim3 sim3_compose(const Sim3& a, const Sim3& b) {
    Sim3 out;
    out.s = a.s * b.s;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += a.R[3 * r + k] * b.R[3 * k + c];
            out.R[3 * r + c] = acc;
        }
        float rt = 0.0f;
        for (int k = 0; k < 3; ++k) rt += a.R[3 * r + k] * b.t[k];
        out.t[r] = a.s * rt + a.t[r];
    }
    return out;
}

Sim3 sim3_inverse(const Sim3& a) {
    Sim3 out;
    out.s = 1.0f / a.s;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out.R[3 * r + c] = a.R[3 * c + r];
    }
    for (int r = 0; r < 3; ++r) {
        float rt = 0.0f;
        for (int k = 0; k < 3; ++k) rt += out.R[3 * r + k] * a.t[k];
        out.t[r] = -rt / a.s;
    }
    return out;
}

void sim3_apply_c2w(const Sim3& sim, const float* c2w_in, float* c2w_out) {
    float in[16];
    std::memcpy(in, c2w_in, sizeof(in));
    for (int r = 0; r < 3; ++r) {
        float tr = 0.0f;
        for (int c = 0; c < 3; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < 3; ++k) acc += sim.R[3 * r + k] * in[4 * k + c];
            c2w_out[4 * r + c] = acc;
            tr += sim.R[3 * r + c] * in[4 * c + 3];
        }
        c2w_out[4 * r + 3] = sim.s * tr + sim.t[r];
    }
    c2w_out[12] = 0.0f;
    c2w_out[13] = 0.0f;
    c2w_out[14] = 0.0f;
    c2w_out[15] = 1.0f;
}

void pose_enc_to_c2w(const float* pose9, float* c2w16) {
    float R[9], t[3];
    pose_enc_to_w2c(pose9, R, t);
    // camera-from-world [R|T]  ->  camera-to-world [R^T | -R^T T]
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) c2w16[4 * r + c] = R[3 * c + r];
        c2w16[4 * r + 3] = -(R[r] * t[0] + R[3 + r] * t[1] + R[6 + r] * t[2]);
    }
    c2w16[12] = 0.0f;
    c2w16[13] = 0.0f;
    c2w16[14] = 0.0f;
    c2w16[15] = 1.0f;
}

std::vector<int> plan_chunk_starts(int n, int K, int overlap) {
    if (n <= 0) return {};
    K = std::max(2, K);
    if (K >= n) return {0};
    overlap = std::min(std::max(1, overlap), K - 1);
    // upstream e2e_full_scene.py: range(0, n, K - overlap), each start
    // clamped to n - K, sorted + unique.
    std::vector<int> starts;
    for (int st = 0; st < n; st += K - overlap) {
        starts.push_back(std::min(st, n - K));
    }
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    return starts;
}

}  // namespace MapAnythingGeometry
