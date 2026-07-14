#include "ray_pose.hpp"
#include "linalg.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace da {

void build_ray_cloud(const float* ray, const float* ray_conf, int Hy, int Wx,
                     double z_threshold,
                     std::vector<double>& src, std::vector<double>& dst,
                     std::vector<double>& w, std::vector<double>& origin_half,
                     std::vector<double>& conf_raw) {
    const int N = Hy * Wx;
    src.resize(2 * N);
    dst.resize(2 * N);
    w.resize(N);
    origin_half.resize(3 * N);
    conf_raw.resize(N);

    // Identity-cam-plane grid (unproject_depth, ixt_normalized): for patch index j,
    // x_grid = linspace(dx, 2-dx, Wx)[j]; origin x = x_grid - 1 (I_K^{-1} subtracts cx=1).
    // (ixt_normalized branch builds the grid in float32 in torch; we use double here and
    //  absorb the ~1e-7 difference in the downstream tolerance.)
    const double dx = 1.0 / Wx, dy = 1.0 / Hy;
    std::vector<double> gx(Wx), gy(Hy);
    // linspace(start, end, n): start + k*(end-start)/(n-1)
    for (int j = 0; j < Wx; ++j)
        gx[j] = (Wx == 1) ? dx : (dx + (double)j * ((2.0 - dx) - dx) / (Wx - 1));
    for (int i = 0; i < Hy; ++i)
        gy[i] = (Hy == 1) ? dy : (dy + (double)i * ((2.0 - dy) - dy) / (Hy - 1));

    for (int i = 0; i < Hy; ++i) {
        for (int j = 0; j < Wx; ++j) {
            int n = i * Wx + j;
            const float* r = ray + (size_t)n * 6;
            double dirx = r[0], diry = r[1], dirz = r[2];   // direction half
            double ox = r[3], oy = r[4], oz = r[5];          // origin half (for T)
            double ox_plane = gx[j] - 1.0;                   // identity-plane origin x
            double oy_plane = gy[i] - 1.0;                   // identity-plane origin y
            // z_mask: |dir_z|>thr AND |origin_plane_z=1|>thr (always true).
            bool zmask = std::fabs(dirz) > z_threshold;  // origin z is 1.0
            double tx = dirx, ty = diry;
            if (zmask) { tx = dirx / dirz; ty = diry / dirz; }
            // origin (plane) z=1, so its x,y unchanged by z-norm.
            src[2 * n + 0] = ox_plane;
            src[2 * n + 1] = oy_plane;
            dst[2 * n + 0] = tx;
            dst[2 * n + 1] = ty;
            double c = ray_conf[n];
            conf_raw[n] = c;
            w[n] = zmask ? c : 0.0;
            origin_half[3 * n + 0] = ox;
            origin_half[3 * n + 1] = oy;
            origin_half[3 * n + 2] = oz;
        }
    }
}

// Fit homography on a set of absolute point indices.
static bool fit_homography_idx(const std::vector<double>& src, const std::vector<double>& dst,
                               const std::vector<double>& w, const int* idx, int M,
                               double H[9]) {
    std::vector<double> s(2 * M), d(2 * M), ww(M);
    for (int k = 0; k < M; ++k) {
        int n = idx[k];
        s[2 * k + 0] = src[2 * n + 0]; s[2 * k + 1] = src[2 * n + 1];
        d[2 * k + 0] = dst[2 * n + 0]; d[2 * k + 1] = dst[2 * n + 1];
        ww[k] = w[n];
    }
    return linalg::homography_weighted(s.data(), d.data(), ww.data(), M, H);
}

// Weighted inlier score of homography H over the full cloud + fill inlier mask.
static double score_homography(const std::vector<double>& src, const std::vector<double>& dst,
                               const std::vector<double>& w, int N, const double H[9],
                               double reproj_thr, std::vector<uint8_t>* inlier) {
    double total = 0.0;
    double thr2 = reproj_thr * reproj_thr;
    for (int n = 0; n < N; ++n) {
        double x = src[2 * n + 0], y = src[2 * n + 1];
        double px = H[0] * x + H[1] * y + H[2];
        double py = H[3] * x + H[4] * y + H[5];
        double pz = H[6] * x + H[7] * y + H[8];
        double ex = px / pz - dst[2 * n + 0];
        double ey = py / pz - dst[2 * n + 1];
        bool in = (ex * ex + ey * ey) < thr2;   // error < reproj_thr (squared compare)
        if (inlier) (*inlier)[n] = in ? 1 : 0;
        if (in) total += w[n];
    }
    return total;
}

bool solve_ray_pose(const float* ray, const float* ray_conf, int Hy, int Wx,
                    int img_H, int img_W,
                    const int* rand_idx, const int* sorted_idx_in,
                    const int* refit_idx, int n_refit,
                    const RayPoseParams& p, RayPoseOut& out) {
    const int N = Hy * Wx;
    std::vector<double> src, dst, w, origin_half, conf_raw;
    build_ray_cloud(ray, ray_conf, Hy, Wx, p.z_threshold, src, dst, w, origin_half, conf_raw);

    // n_sample is the candidate pool size; clamp to N so a tiny ray field
    // (N < num_sample) can never index sorted_idx out of range.
    const int n_sample = std::min(N, std::max(p.num_sample, (int)((double)N * p.sample_ratio)));

    // candidate ordering: argsort(w, descending). Use the fed order if provided
    // (matches torch exactly on the gated path); else compute a deterministic one.
    std::vector<int> sorted_idx;
    if (sorted_idx_in) {
        sorted_idx.assign(sorted_idx_in, sorted_idx_in + N);
    } else {
        sorted_idx.resize(N);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                         [&](int a, int b) { return w[a] > w[b]; });
    }

    // sampling indices into [0, n_sample): fed (gated) or generated (production).
    std::vector<int> gen_rand;
    const int* ridx = rand_idx;
    if (!ridx) {
        gen_rand.resize((size_t)p.n_iter * p.num_sample);
        std::mt19937 rng(p.seed);
        std::vector<int> perm(n_sample);
        std::iota(perm.begin(), perm.end(), 0);
        for (int it = 0; it < p.n_iter; ++it) {
            // partial Fisher-Yates: first num_sample of a fresh permutation
            for (int k = 0; k < p.num_sample; ++k) {
                std::uniform_int_distribution<int> u(k, n_sample - 1);
                int j = u(rng);
                std::swap(perm[k], perm[j]);
                gen_rand[(size_t)it * p.num_sample + k] = perm[k];
            }
        }
        ridx = gen_rand.data();
    }

    // RANSAC: fit per iter, score, keep best inlier mask.
    std::vector<uint8_t> best_mask(N, 0), tmp_mask(N, 0);
    double best_score = -1.0;
    int best_iter = -1;
    std::vector<int> sample(p.num_sample);
    for (int it = 0; it < p.n_iter; ++it) {
        for (int k = 0; k < p.num_sample; ++k)
            sample[k] = sorted_idx[ridx[(size_t)it * p.num_sample + k]];
        double H[9];
        if (!fit_homography_idx(src, dst, w, sample.data(), p.num_sample, H)) continue;
        double sc = score_homography(src, dst, w, N, H, p.reproj_threshold, &tmp_mask);
        if (sc > best_score) {
            best_score = sc;
            best_iter = it;
            best_mask = tmp_mask;
        }
    }
    (void)best_iter;

    // Collect best-hypothesis inliers.
    std::vector<int> inliers;
    inliers.reserve(N);
    for (int n = 0; n < N; ++n) if (best_mask[n]) inliers.push_back(n);
    out.n_inlier_best = (int)inliers.size();

    // Final consensus point set for the refit.
    std::vector<int> refit;
    if (refit_idx) {
        // RNG-free gated path: use the fed consensus exactly.
        refit.assign(refit_idx, refit_idx + n_refit);
    } else {
        // Production: sort inliers by weight desc; subsample if > max_inlier_num.
        std::stable_sort(inliers.begin(), inliers.end(),
                         [&](int a, int b) { return w[a] > w[b]; });
        if ((int)inliers.size() > p.max_inlier_num) {
            int keep_len = std::max((int)(inliers.size() * 0.95), p.max_inlier_num);
            std::vector<int> head(inliers.begin(), inliers.begin() + keep_len);
            std::mt19937 rng(p.seed ^ 0x9e3779b9u);
            std::shuffle(head.begin(), head.end(), rng);
            head.resize(p.max_inlier_num);
            refit = head;
        } else {
            refit = inliers;
        }
    }
    if ((int)refit.size() < 4) return false;

    // Refit homography on the consensus set, normalize det>0.
    double A[9];
    if (!fit_homography_idx(src, dst, w, refit.data(), (int)refit.size(), A)) return false;
    if (linalg::det3(A) < 0) for (int i = 0; i < 9; ++i) A[i] = -A[i];
    for (int i = 0; i < 9; ++i) out.A[i] = A[i];

    // ql decomposition -> R (=Q), L.
    double R[9], L[9];
    linalg::ql_decomposition(A, R, L);
    // L = L / L[2][2]. A rank-deficient consensus set gives l22~0 (and/or
    // focal f0/f1~0 below) -> Inf/NaN pose; bail rather than report garbage.
    double l22 = L[8];
    if (std::fabs(l22) < 1e-12) return false;
    for (int i = 0; i < 9; ++i) L[i] /= l22;
    double f0 = L[0], f1 = L[4];     // (L00, L11)
    double pp0 = L[6], pp1 = L[7];   // (L20, L21)
    if (std::fabs(f0) < 1e-12 || std::fabs(f1) < 1e-12) return false;

    // Translation T = sum(origin_half * conf) / sum(conf).
    double Tn[3] = { 0, 0, 0 }, csum = 0;
    for (int n = 0; n < N; ++n) {
        double c = conf_raw[n];
        Tn[0] += origin_half[3 * n + 0] * c;
        Tn[1] += origin_half[3 * n + 1] * c;
        Tn[2] += origin_half[3 * n + 2] * c;
        csum += c;
    }
    if (std::fabs(csum) < 1e-12) return false;   // all-zero confidence -> T undefined
    Tn[0] /= csum; Tn[1] /= csum; Tn[2] /= csum;

    for (int i = 0; i < 9; ++i) out.R[i] = R[i];
    for (int i = 0; i < 3; ++i) out.T[i] = Tn[i];
    // returned focal = 1/f, pp = pp_raw + 1
    out.focal[0] = 1.0 / f0;
    out.focal[1] = 1.0 / f1;
    out.pp[0] = pp0 + 1.0;
    out.pp[1] = pp1 + 1.0;

    // Assemble w2c = [[R, T],[0,0,0,1]] then affine_inverse -> c2w = [[R^T, -R^T T],[..]].
    double Rt[9];
    linalg::mat3_transpose(R, Rt);
    double c2wT[3];
    for (int i = 0; i < 3; ++i)
        c2wT[i] = -(Rt[i * 3 + 0] * Tn[0] + Rt[i * 3 + 1] * Tn[1] + Rt[i * 3 + 2] * Tn[2]);
    // extrinsics 3x4 row-major
    for (int i = 0; i < 3; ++i) {
        out.extrinsics[i * 4 + 0] = (float)Rt[i * 3 + 0];
        out.extrinsics[i * 4 + 1] = (float)Rt[i * 3 + 1];
        out.extrinsics[i * 4 + 2] = (float)Rt[i * 3 + 2];
        out.extrinsics[i * 4 + 3] = (float)c2wT[i];
    }
    // intrinsics: K00=fr0/2*W, K11=fr1/2*H, K02=pp0*W*0.5, K12=pp1*H*0.5
    for (int i = 0; i < 9; ++i) out.intrinsics[i] = 0.0f;
    out.intrinsics[0] = (float)(out.focal[0] / 2.0 * img_W);
    out.intrinsics[4] = (float)(out.focal[1] / 2.0 * img_H);
    out.intrinsics[2] = (float)(out.pp[0] * img_W * 0.5);
    out.intrinsics[5] = (float)(out.pp[1] * img_H * 0.5);
    out.intrinsics[8] = 1.0f;

    // Final safety net: never report success with a non-finite pose.
    for (int i = 0; i < 12; ++i) if (!std::isfinite(out.extrinsics[i])) return false;
    for (int i = 0; i < 9; ++i)  if (!std::isfinite(out.intrinsics[i])) return false;
    return true;
}

} // namespace da
