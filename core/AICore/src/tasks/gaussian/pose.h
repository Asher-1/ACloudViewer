// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// pose.h — downstream camera-pose recovery + cross-run alignment (C++ port).
//
// This is the C++ port of the (now removed; see git history) proven `pose/`
// research prototype: given the engine's per-pixel gaussians it recovers each
// view's camera (PnP), and aligns successive runs into one accumulating world
// (Sim(3)). It is dependency-free — only the self-contained linalg.h — so it
// ships from the CLI / C API with no Python and no Eigen/OpenCV (see
// CLAUDE.md).
//
// Faithful to FreeSplatter's scene recipe (estimate_poses -> DUSt3R fast_pnp):
// the prototype's numpy reference solver was verified ~1e-7 against cv2 and
// bit-exact to upstream estimate_poses on real engine output.
//
// Conventions: f64 throughout; a similarity acts as x -> s*(R@x)+t; gaussian
// channel layout (scene, 23ch) is xyz[0:3] SH[3:15] opacity[15] scale[16:19]
// rotation[19:23], opacity already activated (sigmoid) in [0,1].
#ifndef FREE_SPLATTER_POSE_H
#define FREE_SPLATTER_POSE_H

#include <string>
#include <vector>

#include "linalg.h"

namespace aicore {
namespace gaussian {
namespace pose {

using fsla::Mat3;
using fsla::Mat4;
using fsla::Vec3;

// ---- focal (Weiszfeld shared-focal, mirrors pose/focal.py) -----------------
// Robust (L1) focal from camera-frame points and their pixels, principal point
// pp. N points, pts3d is N*3 (x,y,z), pixels is N*2 (col,row), all row-major.
double estimate_focal(const double *pts3d,
                      const double *pixels,
                      int N,
                      double ppx,
                      double ppy,
                      int weiszfeld_iters = 10);

// ---- similarity alignment (mirrors pose/align.py) --------------------------
struct Sim3 {
    double s;
    Mat3 R;
    Vec3 t;
};  // x -> s*(R@x)+t

Sim3 sim_identity();
Vec3 sim_apply(const Sim3 &T, const Vec3 &x);
Sim3 sim_compose(const Sim3 &T2, const Sim3 &T1);  // apply T1 then T2
Sim3 sim_invert(const Sim3 &T);
Mat4 sim_matrix(const Sim3 &T);  // 4x4 [[sR,t],[0,1]]

// Umeyama closed-form (s,R,t) minimizing ||sRX+t - Y||^2. with_scale=false ->
// rigid. X, Y are N*3 row-major.
Sim3 fit_similarity(const double *X,
                    const double *Y,
                    int N,
                    bool with_scale = true);

// RANSAC similarity robust to gross outliers; fills `inliers` (size N, 0/1).
Sim3 fit_similarity_ransac(const double *X,
                           const double *Y,
                           int N,
                           double thresh,
                           int iters,
                           std::vector<char> &inliers,
                           bool with_scale = true,
                           uint64_t seed = 0);

// Residual ladder (rigid -> similarity -> affine) + depth correlation, and the
// verdict that classifies the A<->B mismatch. Mirrors align.diagnose.
struct Ladder {
    double scene, rigid, similarity, affine, scale, depth_corr, aff_gain;
    bool structured;
    std::string verdict;  // rigid_ok | needs_scale | needs_affine | nonlinear |
                          // similarity_plus_noise
};
Ladder diagnose(const double *X,
                const double *Y,
                int N,
                double tol = 1e-3,
                double corr_tol = 0.3);

// Loop-closure metrics for a closed chain of similarities (deviation from I).
struct LoopError {
    double scale_err, rot_deg, trans;
};
LoopError loop_closure_error(const std::vector<Sim3> &links);

// Fractional power M^f of a Sim(3) 4x4 (even loop-closure relaxation). Real
// part of the eigendecomposition; valid while rotation < 180deg. Mirrors
// sim_frac_power.
Mat4 sim_frac_power(const Mat4 &M, double f);

// ---- PnP -------------------------------------------------------------------
// RANSAC DLT PnP with known intrinsics K (the prototype's numpy backend).
// Xw is N*3 world points, pixels is N*2 (col,row). Returns cam2world (4x4) and
// fills `inliers` (size N, 0/1). Kept as the asset-free golden-test reference;
// on real scenes the DLT inherits the planar/mirror degeneracy — use solve_pnp.
Mat4 solve_pnp_numpy(const double *Xw,
                     const double *pixels,
                     int N,
                     const Mat3 &K,
                     std::vector<char> &inliers,
                     double thresh_px = 5.0,
                     int iters = 100,
                     uint64_t seed = 0);

// Robust PnP: EPnP (planar-robust, deterministic — uses all points, no random
// minimal samples) for the initial pose, then a Gauss-Newton / Huber
// reprojection refine. Reaches cv2/SQPNP-grade poses on real scenes with no
// OpenCV dependency. Fills `inliers` (reprojection < thresh_px). This is what
// estimate_poses uses.
Mat4 solve_pnp(const double *Xw,
               const double *pixels,
               int N,
               const Mat3 &K,
               std::vector<char> &inliers,
               double thresh_px = 5.0,
               int gn_iters = 10);

// EPnP-only (no refine) — exposed for testing the planar-robust initialization.
Mat4 solve_pnp_epnp(const double *Xw,
                    const double *pixels,
                    int N,
                    const Mat3 &K);

// Integer pixel grid (col,row), row-major over H*W — matches upstream xy_grid
// (no half-pixel offset). Fills `out` with 2*H*W doubles.
void pixel_grid(int H, int W, std::vector<double> &out);

// ---- estimate_poses orchestration (scene recipe, mirrors pnp.estimate_poses)
// -
struct PoseResult {
    std::vector<Mat4> cam2world;
    double focal;
    double scale;
};
// points: N views, each H*W*3 (gaussian centers, view-0 frame). opacities: N
// views, each H*W (activated; pass nullptr entries for no mask). focal<=0 ->
// estimate from view 0 over all pixels (use_first_focal). normalize -> the
// runner's 1/baseline camera rescale.
PoseResult estimate_poses(const std::vector<const float *> &points,
                          const std::vector<const float *> &opacities,
                          int H,
                          int W,
                          double opacity_threshold = 0.05,
                          double focal = -1.0,
                          int pnp_iter = 100,
                          bool normalize = false,
                          uint64_t seed = 0);

// ---- after-inference parallax (depth conditioning of a recovered pair)
// ------- Quantifies how well a 2-view pair constrains depth, from the model's
// OWN recovered geometry. All angle fields are scale-invariant; B/Z is
// consistent because cameras and points share the (un-normalized) view-0 frame.
struct Parallax {
    double tri_angle_deg;  // median triangulation angle over confident points
    double lateral_angle_deg;    // baseline angle off view-0 optical axis
                                 // (0=dolly, 90=strafe)
    double baseline_over_depth;  // ||C1-C0|| / median point depth from camera 0
    double baseline;
    double median_depth;
    double focal;  // estimated (or supplied) focal, px
    int n_points;  // confident points used for the medians
};

// Pure geometry core (no PnP) — directly testable against hand-computed angles.
// pts: 3*N view-0-frame points; mask: per-point weight (>thr kept), or nullptr.
// axis0 need not be unit; it is normalized internally.
Parallax parallax_stats(const Vec3 &C0,
                        const Vec3 &C1,
                        const Vec3 &axis0,
                        const float *pts,
                        const float *mask,
                        int N,
                        double mask_thr);

// Full path: recover the pair's cameras (normalize=false), then parallax_stats.
// points/opacities: 2 views (view-0-frame xyz + activated opacity), as
// estimate_poses.
Parallax pair_parallax(const std::vector<const float *> &points,
                       const std::vector<const float *> &opacities,
                       int H,
                       int W,
                       double opacity_threshold = 0.05,
                       double focal = -1.0);

// ---- accumulation: chain successive runs into one world (mirrors
// accumulate.py) One accumulated 3D gaussian in the global frame, tagged with
// the source frame it came from (consensus fusion needs the frame id). Carries
// the full anisotropic shape (scale + rotation quaternion w,x,y,z), transformed
// through the cross-run Sim(3), so the cloud renders as oriented splats, not
// isotropic points. rgb [0,1].
struct AccumPoint {
    float x, y, z;
    float r, g, b, opacity;  // color [0,1] and activated opacity [0,1] (the
                             // splat alpha)
    float sx, sy, sz;        // anisotropic gaussian scale (global frame)
    float qw, qx, qy,
            qz;  // gaussian rotation quaternion (w,x,y,z), global frame
    int32_t frame;
};

// Diagnostics for chaining one run into the global frame.
struct ChainLink {
    Sim3 sim;      // run k -> run k-1 (local similarity); identity for run 0
    Sim3 global;   // run k -> global frame (T_k = T_{k-1} . sim)
    double scale;  // sim.s — per-link scale drift
    double inlier_frac;  // RANSAC inliers / shared-frame valid points
    double valid_frac;   // pixels valid (opacity) in BOTH shared views
    double resid_frac;   // inlier RANSAC residual / target scene extent
};

// Sliding-window accumulator: feed it each consecutive pair's engine output
// ([2,H,W,gc], where view 0 shares a frame with the previous pair's view 1) and
// it recovers each pair's cameras (estimate_poses), fits the cross-run Sim(3)
// on the shared frame, composes a global chain, and drops every new frame's
// gaussians into one world. This is the live accumulating-reconstruction loop,
// in C++.
class Accumulator {
public:
    Accumulator(int H,
                int W,
                double opacity_threshold = 0.05,
                double ransac_thresh_frac = 0.02,
                int ransac_iters = 300,
                uint64_t seed = 0);

    // gaussians: 2*H*W*gc floats (two views, engine channel layout). Recovers
    // the pair's cameras, chains the Sim(3), accumulates the new frame(s).
    // focal<=0 -> estimate per pair (use_first_focal). Returns the chain link
    // just added.
    ChainLink add_pair(const float *gaussians,
                       int gaussian_channels,
                       double focal = -1.0);

    const std::vector<AccumPoint> &cloud() const { return cloud_; }
    const std::vector<ChainLink> &links() const { return links_; }
    // count+1 global cam2world (similarity) matrices: frame f_k = T_k .
    // c2w_k[view0], plus the final frame f_n = T_{n-1} . c2w_{n-1}[view1].
    // Mirrors accumulate.py's `rec`.
    std::vector<Mat4> camera_path() const;
    int frame_count() const { return next_frame_; }
    int pair_count() const { return (int)links_.size(); }
    const Sim3 &global_transform() const { return T_; }
    // Geometric de-ghost the accumulated cloud in place (consensus_refine).
    double refine(double voxel_frac = 0.03, int iters = 8, double alpha = 0.5);

private:
    void add_view(const float *pts,
                  const float *op,
                  const float *rgb,
                  const float *scl,
                  const float *rot,  // P*3 scale, P*4 quat (w,x,y,z)
                  const Sim3 &T,
                  int frame);

    int H_, W_;
    double thr_, rthr_;
    int riters_;
    uint64_t seed_;
    Sim3 T_;
    bool have_prev_ = false;
    std::vector<float>
            prev_pts1_;  // 3*P previous view-1 points (the shared frame)
    std::vector<char> prev_mask1_;  // P   previous view-1 opacity mask
    std::vector<AccumPoint> cloud_;
    std::vector<Mat4> cams_;  // per-pair view-0 global cam2world
    Mat4 final_cam_;          // latest view-1 global cam2world (appended last)
    std::vector<ChainLink> links_;
    int next_frame_ = 0;
};

// ---- hierarchical tree accumulation (balanced binary merge) ----------------
// Linear chaining composes N Sim(3)s in series, so registration drift compounds
// over N hops and piles onto the last frame (the far end smears). This instead
// builds short submaps and merges them pairwise up a balanced binary tree — so
// drift compounds over only ~log2(N) hops and is spread evenly rather than
// dumped on the end.
struct TreeMerge {  // diagnostics per merge (defs as ChainLink)
    int level, lo_frame, hi_frame, shared_frame;
    double scale, inlier_frac, valid_frac, resid_frac;
};

// Multi-frame-overlap tree. pairs[k]: one engine output [2*H*W*gc] (pair k =
// frames k,k+1; pair k's view-1 and pair k+1's view-0 are the same physical
// frame). Adjacent submaps of `block` frames overlap by `overlap` frames and
// each merge fits its Sim(3) over ALL shared frames at once — the fit is
// over-determined, so the per-frame depth-inconsistency noise (the reason a
// single-frame alignment is only ~13% rigid-consistent) averages out and the
// compounded drift drops. Base submaps are short windows of `block` frames
// (block > overlap), chained internally by single-frame links, then merged in a
// balanced tree on their overlap bands. The degenerate block=2,overlap=1 is the
// plain overlap-by-one (single shared boundary) tree. Returns the merged root
// cloud in the first submap's frame; `merges` (optional) gets one entry per
// merge.
//
// For the staged side-by-side demo: max_levels caps the merge rounds (-1 =
// full): 0 returns the independent base submaps, 1 the first merge round, ... .
// The remaining (>1) nodes are shifted apart along X by their frame-range
// midpoint when layout_spacing != 0 (0 = none, <0 = auto 4x median submap
// extent, >0 = that spacing), each capped to its top per_node_cap points by
// opacity*radius (0 = uncapped) so every laid-out scene renders at its own
// detail. n_nodes_out (optional) reports how many nodes remain (1 once fully
// merged).
std::vector<AccumPoint> tree_accumulate_overlap(
        const std::vector<const float *> &pairs,
        int H,
        int W,
        int gaussian_channels,
        double opacity_threshold = 0.05,
        int block = 4,
        int overlap = 2,
        double ransac_thresh_frac = 0.02,
        int ransac_iters = 300,
        uint64_t seed = 0,
        std::vector<TreeMerge> *merges = nullptr,
        int max_levels = -1,
        double layout_spacing = 0.0,
        int *n_nodes_out = nullptr,
        int per_node_cap = 0);

// ---- consensus fusion (mirrors fuse.py) -----------------------------------
struct FuseStats {
    int64_t raw_points, voxels, kept_voxels, points_kept, points_dropped;
};
// Voxelize the cloud at voxel_frac * cloud-extent and keep only voxels
// corroborated by >= k DISTINCT source frames (single-frame floaters dropped).
// `mode` selects how a consensus voxel is emitted:
//   FUSE_AVERAGED (0): one averaged gaussian per voxel — denoised but decimated
//                      (sparse unless many frames overlap).
//   FUSE_KEPT (1):     every raw gaussian in the voxel — dense, nothing
//   averaged. FUSE_BEST (2):     only the single most-confident frame's
//   gaussians per voxel —
//                      dense AND de-ghosted (no overlapping copies stacked).
enum FuseMode { FUSE_AVERAGED = 0, FUSE_KEPT = 1, FUSE_BEST = 2 };
FuseStats consensus_fuse(const std::vector<AccumPoint> &cloud,
                         double voxel_frac,
                         int k,
                         std::vector<AccumPoint> &fused,
                         int mode = FUSE_AVERAGED);

// ---- gaussian-level geometric de-ghost (multi-view consensus refinement) ----
// Pairwise Sim(3) chaining leaves per-OBJECT (non-rigid) misregistration that a
// per-frame pose fit can't remove, so overlapping frames show doubled objects.
// This refines the cloud IN PLACE at the gaussian level: each iteration, for
// every point, moves it a fraction `alpha` toward the opacity-weighted
// consensus of the OTHER frames' points in its (coarse-to-fine) voxel
// neighbourhood. Non-rigid and local, so spatially-varying ghosting collapses
// to one surface; single-frame regions (no other frame nearby) are left
// untouched. Returns the RMS point-to- consensus distance after refinement, as
// a fraction of cloud extent.
double consensus_refine(std::vector<AccumPoint> &cloud,
                        double voxel_frac = 0.03,
                        int iters = 8,
                        double alpha = 0.5);

// ---- loop closure (mirrors loop_closure.py) -------------------------------
// Inverse of a similarity 4x4 [[sR,t],[0,1]] (sR = s*rotation):
// [[(1/s)Rᵀ,...],[0,1]].
Mat4 sim4_invert(const Mat4 &M);
// Distribute an accumulated Sim(3) drift D over a chain of n+1 nodes by
// applying D^(k/n) at node k (even relaxation, sim_frac_power). Returns the
// corrected camera centers. global_poses are the open-loop global cam2world
// (similarity) matrices.
std::vector<Vec3> distribute_drift(const Mat4 &D,
                                   const std::vector<Mat4> &global_poses);

}  // namespace pose
}  // namespace gaussian
}  // namespace aicore

#endif  // FREE_SPLATTER_POSE_H
