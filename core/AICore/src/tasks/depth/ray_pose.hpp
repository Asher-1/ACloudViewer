#pragma once
// Ray->pose SOLVER (use_ray_pose part B). Pure HOST double-precision linear
// algebra (no ggml). Turns the aux ray head's per-pixel ray field into camera
// extrinsics (3x4 c2w) + intrinsics (3x3), mirroring ray_utils.get_extrinsic_from_camray
// + da3._process_ray_pose_estimation.
#include <array>
#include <cstdint>
#include <vector>

namespace aicore {
namespace depth {

struct RayPoseParams {
    int n_iter = 100;            // RANSAC iterations
    int num_sample = 8;          // points sampled per iteration (num_sample_for_ransac)
    double sample_ratio = 0.3;   // n_sample = max(num_sample, int(N*sample_ratio))
    double reproj_threshold = 0.2;
    int max_inlier_num = 8000;
    double z_threshold = 1e-4;
    uint32_t seed = 1234;        // production-path deterministic sampling seed
};

struct RayPoseOut {
    std::array<float, 12> extrinsics; // 3x4 c2w, row-major
    std::array<float, 9> intrinsics;  // 3x3, row-major
    // diagnostics (double):
    double R[9];        // rotation (w2c) from ql
    double T[3];        // translation (w2c)
    double focal[2];    // returned focal = 1/f
    double pp[2];       // returned principal point = pp_raw + 1
    double A[9];        // homography post det-normalization
    int n_inlier_best;  // best-hypothesis inlier count (pre-subsample), C++-computed
};

// Build the z-normalized 2D point cloud the homography is fit on, exactly as
// compute_optimal_rotation_intrinsics_batch does:
//   src[n] = identity-cam-plane unprojected origin (x-1, y-1)   [origin half, z-normalized=1]
//   dst[n] = ray direction half, z-normalized: (dx/dz, dy/dz)
//   w[n]   = ray_conf, zeroed where |dz|<=z_threshold
//   origin_half[n] = ray channels [3:6] (used for the translation T)
//   conf_raw[n]    = ray_conf (unmodified; used for T's weighted mean)
// src/dst are length 2*N (x,y interleaved); w/conf_raw length N; origin_half length 3*N.
void build_ray_cloud(const float* ray, const float* ray_conf, int Hy, int Wx,
                     double z_threshold,
                     std::vector<double>& src, std::vector<double>& dst,
                     std::vector<double>& w, std::vector<double>& origin_half,
                     std::vector<double>& conf_raw);

// Full solver.
//   ray:      Hy*Wx*6 row-major (h,w,c)         ray_conf: Hy*Wx
//   rand_idx: n_iter*num_sample ints in [0,n_sample); if null, generated internally
//   sorted_idx: N ints (argsort of weights desc); if null, computed internally
//   refit_idx:  if non-null (n_refit ints in [0,N)), the final consensus refit uses
//               exactly these points (RNG-FREE gated path). Otherwise the solver
//               selects its own best-hypothesis inliers + deterministic subsample.
// Returns false on degeneracy.
bool solve_ray_pose(const float* ray, const float* ray_conf, int Hy, int Wx,
                    int img_H, int img_W,
                    const int* rand_idx, const int* sorted_idx,
                    const int* refit_idx, int n_refit,
                    const RayPoseParams& p, RayPoseOut& out);

} // namespace depth
} // namespace aicore
