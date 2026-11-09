// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/querysplat/querysplat_rasterizer.hpp"

#include <Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace qs {

namespace {

constexpr int kTile = 16;  // gsplat tile size
// gsplat-semantics constants (gsplat/cuda/csrc, ALPHA_THRESHOLD = 1/255):
// - per-gaussian alpha floor: contributions below it are skipped. The 1e-4
//   floor we used before accumulates thousands of sub-1/255 near-field tails
//   on case1-type scenes and washes the render to fog (11.9 dB on the
//   OFFICIAL gaussians; gsplat renders the same set at 23.9 dB).
constexpr float kAlphaFloor = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.999f;   // gsplat alpha clamp (was 0.99)
constexpr float kTEarlyStop = 1e-4f;  // transmittance early-stop (gsplat same)
// gsplat rendering.py: render_colors[..., -1:] / render_alphas.clamp(1e-10)
constexpr float kDepthDenorm = 1e-10f;
// gsplat's default bbox extend (in sigmas) before the opacity-aware clamp
constexpr float kBBoxExtendMax = 3.33f;
constexpr float kEps = 1e-6f;

// (w, x, y, z) scalar-first rotation matrix (gsplat convention, matching the
// PLY rot_0..3 field order).
void quat_to_rotmat(const float* q, float* R) {
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1.f - 2.f * (y * y + z * z);
    R[1] = 2.f * (x * y - w * z);
    R[2] = 2.f * (x * z + w * y);
    R[3] = 2.f * (x * y + w * z);
    R[4] = 1.f - 2.f * (x * x + z * z);
    R[5] = 2.f * (y * z - w * x);
    R[6] = 2.f * (x * z - w * y);
    R[7] = 2.f * (y * z + w * x);
    R[8] = 1.f - 2.f * (x * x + y * y);
}

// Per-view projected gaussian: screen mean, depth, and the 2x2 screen-space
// covariance (conic form). Returns false when culled.
struct ProjGaussian {
    float mx, my;    // screen mean
    float depth;     // camera z
    float conic[3];  // a, b, c of the inverse 2x2 covariance
    float radius;    // opacity-aware screen radius
    bool culled;
};

// One tile-bin entry, self-contained so the blend loop streams memory.
struct Splat {
    float mx, my, radius;
    float conic[3];
    float opacity, sigma_max;  // sigma_max: beyond it alpha < kAlphaFloor
    float depth;
    float color[3];
};

ProjGaussian project_one(const float* mean,
                         const float* quat,
                         const float* scale,
                         const float* cov_cam_pre,
                         const float* w2c,
                         float fx,
                         float fy,
                         float cx,
                         float cy,
                         float z_near,
                         float z_far,
                         float src_op,
                         float lim_x_pos,
                         float lim_x_neg,
                         float lim_y_pos,
                         float lim_y_neg) {
    ProjGaussian pg;
    pg.culled = true;
    // gsplat culls sub-threshold opacities at projection time (they can
    // never pass the per-pixel alpha floor)
    if (src_op < kAlphaFloor) return pg;
    // p_cam = W * p_world + t
    const float px = mean[0], py = mean[1], pz = mean[2];
    const float cx_ = w2c[0] * px + w2c[1] * py + w2c[2] * pz + w2c[3];
    const float cy_ = w2c[4] * px + w2c[5] * py + w2c[6] * pz + w2c[7];
    const float cz = w2c[8] * px + w2c[9] * py + w2c[10] * pz + w2c[11];
    if (cz < z_near || cz > z_far) return pg;
    if (cz <= kEps) return pg;

    // cov2d = J * W * cov_world * W^T * J^T with the 0.3 low-pass dilate.
    // gsplat/Inria persp_proj clamps the ray slopes (x/z, y/z) to the frustum
    // margins (+0.3*tan half-fov) before building the Jacobian's z
    // cross-terms; mean2d keeps the unclamped slopes. Without the clamp a
    // near-plane gaussian whose center projects far outside the frustum gets
    // an unbounded cov2d (j02 ~ fx*x/z^2 with x/z unbounded) that covers the
    // whole screen at ~flat alpha - the case1 fog (depth stuck at the near
    // plane, alpha = 1 everywhere).
    const float invz = 1.0f / cz;
    float tx = cx_ * invz;
    tx = std::min(lim_x_pos, std::max(-lim_x_neg, tx));
    float ty = cy_ * invz;
    ty = std::min(lim_y_pos, std::max(-lim_y_neg, ty));
    const float j00 = fx * invz, j02 = -fx * tx * invz;
    const float j11 = fy * invz, j12 = -fy * ty * invz;
    // M = W (3x3 rotation rows of w2c); C = symmetric cov_world as 6 floats
    // [xx, xy, xz, yy, yz, zz]. T = M * C:
    const float* M = w2c;
    const float* C = cov_cam_pre;
    static const int CIDX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    float T[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            T[r * 3 + c] = M[r * 4 + 0] * C[CIDX[0][c]] +
                           M[r * 4 + 1] * C[CIDX[1][c]] +
                           M[r * 4 + 2] * C[CIDX[2][c]];
    // out = T * M^T: out[r][c] = sum_k T[r][k] * M[c][k]
    float outm[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            outm[r * 3 + c] = T[r * 3 + 0] * M[c * 4 + 0] +
                              T[r * 3 + 1] * M[c * 4 + 1] +
                              T[r * 3 + 2] * M[c * 4 + 2];
    const float cxx = outm[0], cxy = outm[1], cxz = outm[2];
    const float cyy = outm[4], cyz = outm[5], czz = outm[8];
    // cov2d = J * C_cam * J^T with J = [[j00, 0, j02], [0, j11, j12]]:
    //   a_ = j00²·cxx + 2·j00·j02·cxz + j02²·czz   (xz cross-term!)
    //   b_ = j00·j11·cxy + j00·j12·cxz + j02·j11·cyz + j02·j12·czz
    //   c_ = j11²·cyy + 2·j11·j12·cyz + j12²·czz
    const float a_ = j00 * j00 * cxx + 2.f * j00 * j02 * cxz + j02 * j02 * czz;
    const float b_ = j00 * j11 * cxy + j00 * j12 * cxz + j02 * j11 * cyz +
                     j02 * j12 * czz;
    const float c_ = j11 * j11 * cyy + 2.f * j11 * j12 * cyz + j12 * j12 * czz;
    // low-pass filter (gsplat antialiasing=false path)
    const float a = a_ + 0.3f;
    const float cc = c_ + 0.3f;
    const float det = a * cc - b_ * b_;
    if (det <= 0.0f) return pg;
    const float inv_det = 1.0f / det;
    pg.conic[0] = cc * inv_det;
    pg.conic[1] = -b_ * inv_det;
    pg.conic[2] = a * inv_det;
    // screen radius: opacity-aware bbox extent (gsplat ProjectionEWA3DGSFused,
    // https://arxiv.org/pdf/2402.00525 Section B.2): alpha reaches the floor
    // at d^T Sigma^-1 d = 2*ln(op/floor), i.e. extend sigmas along each axis;
    // clamp to gsplat's 3.33-sigma default. Using the max eigenvalue keeps
    // the bound circular (a superset of gsplat's per-axis box); pixels in the
    // slack are skipped by the per-pixel alpha floor, so parity holds.
    const float mid = 0.5f * (a + cc);
    const float disc = std::sqrt(std::max(0.01f, mid * mid - det));
    const float eig = mid + disc;
    float extend = kBBoxExtendMax;
    extend = std::min(extend, std::sqrt(2.0f * std::log(src_op / kAlphaFloor)));
    pg.radius = extend * std::sqrt(std::max(eig, 0.0f));
    if (pg.radius <= 0.0f) return pg;
    // screen point (pixel centers at +0.5)
    pg.mx = cx_ * fx * invz + cx;
    pg.my = cy_ * fy * invz + cy;
    pg.depth = cz;
    pg.culled = false;
    return pg;
}

// ---- shared per-(splat, pixel) alpha test: the exact rejection sequence
// of the forward blend, reused verbatim by the backward walk (t1: one
// function so fwd/bwd can never drift) ----
struct PixelAlpha {
    float dx, dy;  // pixel - mean (our blend convention)
    float sigma;   // conic-weighted distance, 2x gsplat's sigma
    float vis;     // exp(-sigma/2)
    float a_raw;   // opacity * vis (pre-clamp)
};

// Per-view pipeline consumed by both the forward blend and the backward
// walk (same projection -> tile bins -> depth sort -> Splat stream, so
// fwd/bwd see an identical splat order by construction).
struct ViewPipeline {
    std::vector<ProjGaussian> proj;
    std::vector<int> tile_count, tile_start;
    std::vector<std::pair<float, int>>
            keys;  // (depth, gaussian) per tile entry
    std::vector<Splat> splats;
};

inline bool eval_splat(const Splat& sp, float px, float py, PixelAlpha* pa) {
    const float dx = px - sp.mx, dy = py - sp.my;
    if (dx > sp.radius || dx < -sp.radius) return false;
    if (dy > sp.radius || dy < -sp.radius) return false;
    const float sigma = sp.conic[0] * dx * dx + 2.f * sp.conic[1] * dx * dy +
                        sp.conic[2] * dy * dy;
    if (sigma < 0.0f) return false;             // gsplat: power > 0 skip
    if (sigma > sp.sigma_max) return false;     // alpha below the floor
    const float vis = std::exp(-0.5f * sigma);  // gsplat __expf(-sigma)
    const float a_raw = sp.opacity * vis;
    if (a_raw < kAlphaFloor) return false;
    pa->dx = dx;
    pa->dy = dy;
    pa->sigma = sigma;
    pa->vis = vis;
    pa->a_raw = a_raw;
    return true;
}

}  // namespace

RenderedViews render_gaussians(const GaussianOutput& g,
                               const float* w2c,
                               const float* intrinsics,
                               int S,
                               const RenderConfig& cfg) {
    const int W = cfg.width, H = cfg.height;
    const int N = (int)g.N;
    const int stride = 11 + g.color_dim;
    const int tiles_x = (W + kTile - 1) / kTile;
    const int tiles_y = (H + kTile - 1) / kTile;
    RenderedViews out;
    out.S = S;
    out.width = W;
    out.height = H;
    out.images.assign((size_t)S * 3 * H * W, 0.0f);
    out.alphas.assign((size_t)S * H * W, 0.0f);
    out.depths.assign((size_t)S * H * W, 0.0f);
    out.last_k.assign((size_t)S * H * W, -1);

    // world-space covariance per gaussian: R S S R^T (symmetric, 6 floats)
    std::vector<float> cov_world((size_t)N * 6);
    for (int i = 0; i < N; i++) {
        const float* src = &g.data[(size_t)i * stride];
        float R[9];
        quat_to_rotmat(src + 7, R);
        const float s[3] = {src[4], src[5], src[6]};
        // M = R * diag(s)
        float M[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) M[r * 3 + c] = R[r * 3 + c] * s[c];
        float* C = &cov_world[(size_t)i * 6];
        C[0] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];  // xx
        C[1] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];  // xy
        C[2] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];  // xz
        C[3] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];  // yy
        C[4] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];  // yz
        C[5] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];  // zz
    }

    const float sh_c0 = 0.28209479177387814f;
    const float sh_c1 = 0.4886025119029199f;
    const int n_tiles = tiles_x * tiles_y;

    ViewPipeline vp;
    vp.tile_count.assign((size_t)n_tiles + 1, 0);
    vp.tile_start.assign((size_t)n_tiles + 1, 0);

    for (int v = 0; v < S; v++) {
        const float* w2c_v = w2c + (size_t)v * 16;
        const float fx = intrinsics[(size_t)v * 4 + 0];
        const float fy = intrinsics[(size_t)v * 4 + 1];
        const float cx = intrinsics[(size_t)v * 4 + 2];
        const float cy = intrinsics[(size_t)v * 4 + 3];
        const float tan_fovx = 0.5f * W / fx;
        const float tan_fovy = 0.5f * H / fy;
        const float lim_x_pos = (W - cx) / fx + 0.3f * tan_fovx;
        const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
        const float lim_y_pos = (H - cy) / fy + 0.3f * tan_fovy;
        const float lim_y_neg = cy / fy + 0.3f * tan_fovy;
        // camera world position = -R^T * T (R = w2c[:3,:3], T = w2c[:3,3])
        const float tx = w2c_v[3], ty = w2c_v[7], tz = w2c_v[11];
        const float cam_x = -(w2c_v[0] * tx + w2c_v[4] * ty + w2c_v[8] * tz);
        const float cam_y = -(w2c_v[1] * tx + w2c_v[5] * ty + w2c_v[9] * tz);
        const float cam_z = -(w2c_v[2] * tx + w2c_v[6] * ty + w2c_v[10] * tz);

        // ---- project all gaussians (independent per gaussian) ----
        vp.proj.resize((size_t)N);
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(static))
#endif
        for (int i = 0; i < N; i++) {
            const float* src = &g.data[(size_t)i * stride];
            vp.proj[i] = project_one(
                    src, src + 7, src + 4, &cov_world[(size_t)i * 6], w2c_v, fx,
                    fy, cx, cy, cfg.near_plane, cfg.far_plane, src[3],
                    lim_x_pos, lim_x_neg, lim_y_pos, lim_y_neg);
        }

        // ---- tile bins (gaussians whose screen disk touches the tile) ----
        // Counting sort into one flat array (count, prefix-sum, fill) instead
        // of a vector per tile: O(N + entries), no per-tile reallocations.
        auto tile_range = [&](const ProjGaussian& pg, int* x0, int* x1, int* y0,
                              int* y1) {
            *x0 = std::max(0, (int)std::floor((pg.mx - pg.radius) / kTile));
            *x1 = std::min(tiles_x - 1,
                           (int)std::floor((pg.mx + pg.radius) / kTile));
            *y0 = std::max(0, (int)std::floor((pg.my - pg.radius) / kTile));
            *y1 = std::min(tiles_y - 1,
                           (int)std::floor((pg.my + pg.radius) / kTile));
        };
        std::fill(vp.tile_count.begin(), vp.tile_count.end(), 0);
        for (int i = 0; i < N; i++) {
            if (vp.proj[i].culled) continue;
            int x0, x1, y0, y1;
            tile_range(vp.proj[i], &x0, &x1, &y0, &y1);
            for (int ty = y0; ty <= y1; ty++)
                for (int tx = x0; tx <= x1; tx++)
                    vp.tile_count[(size_t)ty * tiles_x + tx]++;
        }
        vp.tile_start[0] = 0;
        for (int t = 0; t < n_tiles; t++)
            vp.tile_start[(size_t)t + 1] =
                    vp.tile_start[(size_t)t] + vp.tile_count[(size_t)t];
        vp.keys.resize((size_t)vp.tile_start[(size_t)n_tiles]);
        std::copy(vp.tile_start.begin(), vp.tile_start.end() - 1,
                  vp.tile_count.begin());  // fill cursors
        for (int i = 0; i < N;
             i++) {  // ascending i: each bin starts in index order
            if (vp.proj[i].culled) continue;
            int x0, x1, y0, y1;
            tile_range(vp.proj[i], &x0, &x1, &y0, &y1);
            for (int ty = y0; ty <= y1; ty++)
                for (int tx = x0; tx <= x1; tx++)
                    vp.keys[(size_t)vp.tile_count[(size_t)ty * tiles_x +
                                                  tx]++] = {vp.proj[i].depth,
                                                            i};
        }
        // Front-to-back per tile, ties by gaussian index (gsplat's radix sort
        // on (tile|depth) keys is stable over index order). The per-tile
        // entries are then flattened into self-contained Splats so the blend
        // streams one contiguous array instead of gathering from proj[] and
        // g.data. O(E log E_tile) for E bin entries.
        vp.splats.resize(vp.keys.size());
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(dynamic, 4))
#endif
        for (int t = 0; t < n_tiles; t++) {
            const size_t b = (size_t)vp.tile_start[(size_t)t],
                         e = (size_t)vp.tile_start[(size_t)t + 1];
            std::sort(vp.keys.begin() + (std::ptrdiff_t)b,
                      vp.keys.begin() + (std::ptrdiff_t)e);
            for (size_t k = b; k < e; k++) {
                const int i = vp.keys[k].second;
                const ProjGaussian& pg = vp.proj[i];
                const float* src = &g.data[(size_t)i * stride];
                Splat& sp = vp.splats[k];
                sp.mx = pg.mx;
                sp.my = pg.my;
                sp.radius = pg.radius;
                sp.conic[0] = pg.conic[0];
                sp.conic[1] = pg.conic[1];
                sp.conic[2] = pg.conic[2];
                sp.opacity = src[3];
                // alpha = op*exp(-sigma/2) < floor  <=>  sigma > 2
                // ln(op/floor); the 1e-3 margin dwarfs the f32 rounding of
                // expf/logf, so every sigma above it would also have failed the
                // alpha-floor test
                sp.sigma_max = 2.0f * std::log(src[3] / kAlphaFloor) + 1e-3f;
                sp.depth = pg.depth;
                // color: direct (sh0) or per-view SH eval (sh1); per gaussian
                // and view, as gsplat evaluates it before rasterization
                if (cfg.sh_degree == 0) {
                    sp.color[0] = src[11];
                    sp.color[1] = src[12];
                    sp.color[2] = src[13];
                } else {
                    // view direction world <- camera position
                    float dir[3] = {src[0] - cam_x, src[1] - cam_y,
                                    src[2] - cam_z};
                    const float n =
                            1.0f / std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                                             dir[2] * dir[2] + kEps);
                    dir[0] *= n;
                    dir[1] *= n;
                    dir[2] *= n;
                    const float* sh = src + 11;
                    for (int c = 0; c < 3; c++) {
                        // 3DGS/gsplat band layout: result = C0*sh0
                        //   - C1*y*sh1 + C1*z*sh2 - C1*x*sh3,
                        // then +0.5 / clamp_min(0) — gsplat rendering.py does
                        // exactly this after the SH eval ("apple-to-apple with
                        // Inria's backend"), and QuerySplat's dc =
                        // (rgb-0.5)/0.282 is defined against it.
                        float col = sh_c0 * sh[c] - sh_c1 * dir[1] * sh[3 + c] +
                                    sh_c1 * dir[2] * sh[6 + c] -
                                    sh_c1 * dir[0] * sh[9 + c] + 0.5f;
                        sp.color[c] = col < 0.0f ? 0.0f : col;
                    }
                }
            }
        }

        // ---- blend ----
        // Gaussian-outer, pixel-inner per tile: each splat only visits the
        // pixels of its bounding box inside the tile, while every pixel still
        // composites its gaussians front to back in the same order with the
        // same arithmetic as a per-pixel loop (bit-identical output).
        // gsplat RasterizeToPixels3DGSFwd semantics: a gaussian whose next_T
        // would fall to <= 1e-4 is EXCLUDED (not blended) and the pixel is
        // done; the splat index of the last blended gaussian is recorded for
        // the backward pass (per-pixel last_ids).
        float* img = &out.images[(size_t)v * 3 * H * W];
        float* alp = &out.alphas[(size_t)v * H * W];
        float* dep = &out.depths[(size_t)v * H * W];
        int32_t* lst = &out.last_k[(size_t)v * H * W];
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(dynamic, 4))
#endif
        for (int t = 0; t < n_tiles; t++) {
            const int px0 = (t % tiles_x) * kTile, py0 = (t / tiles_x) * kTile;
            const int tw = std::min(W - px0, kTile),
                      th = std::min(H - py0, kTile);
            float Tp[kTile * kTile], acc[kTile * kTile][4];
            unsigned char done[kTile * kTile];
            int32_t lastp[kTile * kTile];
            for (int p = 0; p < kTile * kTile; p++) {
                Tp[p] = 1.0f;
                acc[p][0] = acc[p][1] = acc[p][2] = acc[p][3] =
                        0.f;  // rgb + depth
                done[p] = 0;
                lastp[p] = -1;
            }
            int live = tw * th;  // pixels not yet done
            const size_t e = (size_t)vp.tile_start[(size_t)t + 1];
            for (size_t k = (size_t)vp.tile_start[(size_t)t]; k < e && live > 0;
                 k++) {
                const Splat& sp = vp.splats[k];
                // superset of the |dx|,|dy| <= radius pixels; the exact test
                // below decides
                const int xa = std::max(
                        0, (int)std::floor(sp.mx - sp.radius - 0.5f) - px0);
                const int xb = std::min(
                        tw - 1, (int)std::ceil(sp.mx + sp.radius - 0.5f) - px0);
                const int ya = std::max(
                        0, (int)std::floor(sp.my - sp.radius - 0.5f) - py0);
                const int yb = std::min(
                        th - 1, (int)std::ceil(sp.my + sp.radius - 0.5f) - py0);
                for (int ly = ya; ly <= yb; ly++) {
                    for (int lx = xa; lx <= xb; lx++) {
                        const int p = ly * kTile + lx;
                        if (done[p]) continue;
                        float& T = Tp[p];
                        PixelAlpha pa;
                        if (!eval_splat(sp, (px0 + lx) + 0.5f,
                                        (py0 + ly) + 0.5f, &pa))
                            continue;
                        const float a = std::min(pa.a_raw, kMaxAlpha);
                        const float next_T = T * (1.0f - a);
                        if (next_T <= kTEarlyStop) {
                            // gsplat: exclusive - the crossing gaussian is NOT
                            // blended and the pixel is finished
                            done[p] = 1;
                            live--;
                            continue;
                        }
                        acc[p][0] += T * a * sp.color[0];
                        acc[p][1] += T * a * sp.color[1];
                        acc[p][2] += T * a * sp.color[2];
                        acc[p][3] += T * a * sp.depth;
                        lastp[p] = (int32_t)k;
                        T = next_T;
                    }
                }
            }
            for (int ly = 0; ly < th; ly++) {
                for (int lx = 0; lx < tw; lx++) {
                    const int p = ly * kTile + lx;
                    const float T = Tp[p];
                    const float alpha_acc = 1.0f - T;
                    const size_t o = (size_t)(py0 + ly) * W + (px0 + lx);
                    img[o] = acc[p][0] + T * cfg.bg_color[0];
                    img[(size_t)H * W + o] = acc[p][1] + T * cfg.bg_color[1];
                    img[(size_t)2 * H * W + o] =
                            acc[p][2] + T * cfg.bg_color[2];
                    alp[o] = alpha_acc;
                    dep[o] = acc[p][3] / std::max(alpha_acc, kDepthDenorm);
                    lst[o] = lastp[p];
                }
            }
        }
    }
    return out;
}

// ---- backward ----

namespace {

// glm col-major -> row-major translated gsplat helpers (Utils.cuh).

// vjp of the 2x2 inverse: v_M = -Minv * v_Minv * Minv, with the off-diagonal
// gradient halved on entry (gsplat inverse_vjp + the symmetric unpack).
void inverse_vjp2(const float conic[3],
                  const float v_conic[3],
                  float v_cov[3]) {
    const float A = conic[0], B = conic[1], C = conic[2];
    // vConicMat = [[vA, vB/2], [vB/2, vC]]
    const float g00 = v_conic[0], g01 = 0.5f * v_conic[1], g11 = v_conic[2];
    // -Minv * G * Minv for symmetric 2x2 (row-major math)
    const float m00 = -(A * (g00 * A + g01 * B) + B * (g01 * A + g11 * B));
    const float m01 = -(A * (g00 * B + g01 * C) + B * (g01 * B + g11 * C));
    const float m11 = -(B * (g00 * B + g01 * C) + C * (g01 * B + g11 * C));
    v_cov[0] += m00;
    v_cov[1] += m01;  // symmetric: one off-diagonal accumulation
    v_cov[2] += m11;
}

// gsplat persp_proj_vjp (Utils.cuh), row-major math. Inputs: camera-space
// mean/covar, intrinsics, and the vjp seeds dL/dcov2d (3) + dL/dmean2d (2).
// Outputs accumulate into v_mean_c (3) and v_covar_c (6, xx/xy/xz/yy/yz/zz).
void persp_proj_vjp(const float mean_c[3],
                    const float cov_c[6],
                    float fx,
                    float fy,
                    float cx,
                    float cy,
                    int width,
                    int height,
                    const float v_cov2d[3],
                    const float v_mean2d[2],
                    float v_mean_c[3],
                    float v_covar_c[6]) {
    const float x = mean_c[0], y = mean_c[1], z = mean_c[2];
    const float tan_fovx = 0.5f * width / fx;
    const float tan_fovy = 0.5f * height / fy;
    const float lim_x_pos = (width - cx) / fx + 0.3f * tan_fovx;
    const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
    const float lim_y_pos = (height - cy) / fy + 0.3f * tan_fovy;
    const float lim_y_neg = cy / fy + 0.3f * tan_fovy;
    const float rz = 1.f / z, rz2 = rz * rz, rz3 = rz2 * rz;
    const float tx = z * std::min(lim_x_pos, std::max(-lim_x_neg, x * rz));
    const float ty = z * std::min(lim_y_pos, std::max(-lim_y_neg, y * rz));
    // J (2x3 row-major math): rows = pixel x/y, cols = camera x/y/z
    const float J[2][3] = {{fx * rz, 0.f, -fx * tx * rz2},
                           {0.f, fy * rz, -fy * ty * rz2}};
    // v_cov3d += J^T * v_cov2d * J  (v_cov2d symmetric [[g00,g01],[g01,g11]])
    const float g00 = v_cov2d[0], g01 = v_cov2d[1], g11 = v_cov2d[2];
    // temp = v_cov2d * J (2x3)
    const float t[2][3] = {
            {g00 * J[0][0] + g01 * J[1][0], g00 * J[0][1] + g01 * J[1][1],
             g00 * J[0][2] + g01 * J[1][2]},
            {g01 * J[0][0] + g11 * J[1][0], g01 * J[0][1] + g11 * J[1][1],
             g01 * J[0][2] + g11 * J[1][2]}};
    // v_cov3d[p][q] += sum_{r,s} J[r][p] * G[r][s] * J[s][q] = sum_r J[r][p] *
    // t[r][q] (cov_c is [xx, xy, xz, yy, yz, zz]; accumulate both symmetric
    // halves once)
    static const int CIDX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    for (int p = 0; p < 3; p++)
        for (int q = p; q < 3; q++) {
            float acc = 0.f;
            for (int r = 0; r < 2; r++) acc += J[r][p] * t[r][q];
            v_covar_c[CIDX[p][q]] += acc;
        }
    // mean2d path
    v_mean_c[0] += fx * rz * v_mean2d[0];
    v_mean_c[1] += fy * rz * v_mean2d[1];
    v_mean_c[2] += -(fx * x * v_mean2d[0] + fy * y * v_mean2d[1]) * rz2;
    // J path (v_J = G * J * cov3d^T + G^T * J * cov3d; with G = v_cov2d and
    // cov3d both symmetric this collapses to 2 * G * J * cov3d)
    float vJ[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
    const float* Cv = cov_c;
    for (int r = 0; r < 2; r++) {
        // G[r][0], G[r][1] of the symmetric seed matrix
        const float gr[2] = {r == 0 ? g00 : g01, r == 0 ? g01 : g11};
        for (int c = 0; c < 3; c++)
            for (int s = 0; s < 2; s++) {
                float jc = 0.f;
                for (int p = 0; p < 3; p++) jc += J[s][p] * Cv[CIDX[c][p]];
                vJ[r][c] += 2.f * gr[s] * jc;
            }
    }
    // fov clipping (gsplat persp_proj_vjp)
    if (x * rz <= lim_x_pos && x * rz >= -lim_x_neg) {
        v_mean_c[0] += -fx * rz2 * vJ[0][2];
    } else {
        v_mean_c[2] += -fx * rz3 * vJ[0][2] * tx;
    }
    if (y * rz <= lim_y_pos && y * rz >= -lim_y_neg) {
        v_mean_c[1] += -fy * rz2 * vJ[1][2];
    } else {
        v_mean_c[2] += -fy * rz3 * vJ[1][2] * ty;
    }
    v_mean_c[2] += -fx * rz2 * vJ[0][0] - fy * rz2 * vJ[1][1] +
                   2.f * fx * tx * rz3 * vJ[0][2] +
                   2.f * fy * ty * rz3 * vJ[1][2];
}

// gsplat quat_scale_to_covar_vjp + quat_to_rotmat_vjp, row-major math.
// quat = (w,x,y,z) scalar-first; v_covar is the symmetric 6-float world
// covariance gradient.
void quat_scale_to_covar_vjp(const float quat[4],
                             const float scale[3],
                             const float v_covar[6],
                             float v_quat[4],
                             float v_scale[3]) {
    float R[9];
    quat_to_rotmat(quat, R);
    const float sx = scale[0], sy = scale[1], sz = scale[2];
    // M = R * diag(s) (row-major)
    const float M[3][3] = {{R[0] * sx, R[1] * sy, R[2] * sz},
                           {R[3] * sx, R[4] * sy, R[5] * sz},
                           {R[6] * sx, R[7] * sy, R[8] * sz}};
    // vC as a full symmetric matrix
    static const int CIDX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    float C[3][3];
    for (int p = 0; p < 3; p++)
        for (int q = 0; q < 3; q++) C[p][q] = v_covar[CIDX[p][q]];
    // v_M = (vC + vC^T) * M
    float vM[3][3];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) {
            float acc = 0.f;
            for (int k = 0; k < 3; k++) acc += (C[r][k] + C[k][r]) * M[k][c];
            vM[r][c] = acc;
        }
    // v_scale[j] = sum_r R[r][j] * v_M[r][j]
    for (int j = 0; j < 3; j++) {
        float acc = 0.f;
        for (int r = 0; r < 3; r++) acc += R[r * 3 + j] * vM[r][j];
        v_scale[j] += acc;
    }
    // quat_to_rotmat_vjp: v_Rgrad[r][c] = v_M[r][c] * s[c]
    const float vR[3][3] = {{vM[0][0] * sx, vM[0][1] * sy, vM[0][2] * sz},
                            {vM[1][0] * sx, vM[1][1] * sy, vM[1][2] * sz},
                            {vM[2][0] * sx, vM[2][1] * sy, vM[2][2] * sz}};
    // gsplat quat_to_rotmat_vjp with glm col-major v_R[col][row] == row-major
    // vR[row][col] translated; the quaternion is normalized first and the
    // gradient is projected onto the unit sphere.
    const float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    const float inv_norm = 1.0f / std::sqrt(x * x + y * y + z * z + w * w);
    const float qn[4] = {w * inv_norm, x * inv_norm, y * inv_norm,
                         z * inv_norm};
    const float r00 = vR[0][0], r01 = vR[0][1], r02 = vR[0][2];
    const float r10 = vR[1][0], r11 = vR[1][1], r12 = vR[1][2];
    const float r20 = vR[2][0], r21 = vR[2][1], r22 = vR[2][2];
    const float vqn[4] = {
            2.f * (x * (r21 - r12) + y * (r02 - r20) + z * (r10 - r01)),
            2.f * (-2.f * x * (r11 + r22) + y * (r10 + r01) + z * (r20 + r02) +
                   w * (r21 - r12)),
            2.f * (x * (r10 + r01) - 2.f * y * (r00 + r22) + z * (r21 + r12) +
                   w * (r02 - r20)),
            2.f * (x * (r20 + r02) + y * (r21 + r12) - 2.f * z * (r00 + r11) +
                   w * (r10 - r01)),
    };
    const float dot =
            vqn[0] * qn[0] + vqn[1] * qn[1] + vqn[2] * qn[2] + vqn[3] * qn[3];
    v_quat[0] += (vqn[0] - dot * qn[0]) * inv_norm;
    v_quat[1] += (vqn[1] - dot * qn[1]) * inv_norm;
    v_quat[2] += (vqn[2] - dot * qn[2]) * inv_norm;
    v_quat[3] += (vqn[3] - dot * qn[3]) * inv_norm;
}

}  // namespace

bool render_backward(const GaussianOutput& g,
                     const float* w2c,
                     const float* intrinsics,
                     int S,
                     const RenderConfig& cfg,
                     const RenderedViews& fwd,
                     const float* d_images,
                     GaussianGrads* out_grads) {
    if (cfg.sh_degree != 0) {
        // the TTO contract (t4/t5): sh0 only; the sh1 splat color is a
        // per-view SH evaluation whose backward is not ported
        return false;
    }
    const int W = cfg.width, H = cfg.height;
    const int N = (int)g.N;
    const int stride = 11 + g.color_dim;
    const int tiles_x = (W + kTile - 1) / kTile;
    const int tiles_y = (H + kTile - 1) / kTile;
    const int n_tiles = tiles_x * tiles_y;
    const int color_dim = g.color_dim;

    out_grads->N = N;
    out_grads->color_dim = color_dim;
    out_grads->pos.assign((size_t)N * 3, 0.0f);
    out_grads->opacity.assign((size_t)N, 0.0f);
    out_grads->scale.assign((size_t)N * 3, 0.0f);
    out_grads->rot.assign((size_t)N * 4, 0.0f);
    out_grads->color.assign((size_t)N * color_dim, 0.0f);
    if (fwd.last_k.size() != (size_t)S * H * W ||
        fwd.alphas.size() != (size_t)S * H * W) {
        // render_backward requires the same-step render_gaussians output
        return false;
    }

    // world-space covariance per gaussian (same as forward)
    std::vector<float> cov_world((size_t)N * 6);
    for (int i = 0; i < N; i++) {
        const float* src = &g.data[(size_t)i * stride];
        float R[9];
        quat_to_rotmat(src + 7, R);
        const float s[3] = {src[4], src[5], src[6]};
        float M[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) M[r * 3 + c] = R[r * 3 + c] * s[c];
        float* C = &cov_world[(size_t)i * 6];
        C[0] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];
        C[1] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];
        C[2] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];
        C[3] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];
        C[4] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];
        C[5] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];
    }

    // per-thread scatter buffers (pixel pass -> per-gaussian seeds), reduced
    // per view. Capped: N-sized buffers per thread, 12 x ~25 MB worst case.
    constexpr int kMaxBwdThreads = 12;
    int nthreads = 1;
#ifdef QS_USE_OPENMP
    nthreads = cloudViewer::utility::GetMaxThreads(kMaxBwdThreads);
#endif
    struct Seeds {
        std::vector<float> conic;  // N*3
        std::vector<float> xy;     // N*2
        std::vector<float> col;    // N*color_dim
        std::vector<float> op;     // N
    };
    std::vector<Seeds> seeds((size_t)nthreads);
    for (auto& s : seeds) {
        s.conic.assign((size_t)N * 3, 0.0f);
        s.xy.assign((size_t)N * 2, 0.0f);
        s.col.assign((size_t)N * color_dim, 0.0f);
        s.op.assign((size_t)N, 0.0f);
    }

    ViewPipeline vp;
    vp.tile_count.assign((size_t)n_tiles + 1, 0);
    vp.tile_start.assign((size_t)n_tiles + 1, 0);

    for (int v = 0; v < S; v++) {
        // the scatter buffers are per-view: without this reset, view v+1
        // accumulates on top of view v's residues and every earlier view is
        // counted again (observed as a 1.5x gradient inflation at S=2)
        for (auto& s : seeds) {
            std::fill(s.conic.begin(), s.conic.end(), 0.0f);
            std::fill(s.xy.begin(), s.xy.end(), 0.0f);
            std::fill(s.col.begin(), s.col.end(), 0.0f);
            std::fill(s.op.begin(), s.op.end(), 0.0f);
        }
        const float* w2c_v = w2c + (size_t)v * 16;
        const float fx = intrinsics[(size_t)v * 4 + 0];
        const float fy = intrinsics[(size_t)v * 4 + 1];
        const float cx = intrinsics[(size_t)v * 4 + 2];
        const float cy = intrinsics[(size_t)v * 4 + 3];
        const float tan_fovx = 0.5f * W / fx;
        const float tan_fovy = 0.5f * H / fy;
        const float lim_x_pos = (W - cx) / fx + 0.3f * tan_fovx;
        const float lim_x_neg = cx / fx + 0.3f * tan_fovx;
        const float lim_y_pos = (H - cy) / fy + 0.3f * tan_fovy;
        const float lim_y_neg = cy / fy + 0.3f * tan_fovy;

        // identical pipeline as forward (projection + bins + sort + splats)
        vp.proj.resize((size_t)N);
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(static))
#endif
        for (int i = 0; i < N; i++) {
            const float* src = &g.data[(size_t)i * stride];
            vp.proj[i] = project_one(
                    src, src + 7, src + 4, &cov_world[(size_t)i * 6], w2c_v, fx,
                    fy, cx, cy, cfg.near_plane, cfg.far_plane, src[3],
                    lim_x_pos, lim_x_neg, lim_y_pos, lim_y_neg);
        }
        auto tile_range = [&](const ProjGaussian& pg, int* x0, int* x1, int* y0,
                              int* y1) {
            *x0 = std::max(0, (int)std::floor((pg.mx - pg.radius) / kTile));
            *x1 = std::min(tiles_x - 1,
                           (int)std::floor((pg.mx + pg.radius) / kTile));
            *y0 = std::max(0, (int)std::floor((pg.my - pg.radius) / kTile));
            *y1 = std::min(tiles_y - 1,
                           (int)std::floor((pg.my + pg.radius) / kTile));
        };
        std::fill(vp.tile_count.begin(), vp.tile_count.end(), 0);
        for (int i = 0; i < N; i++) {
            if (vp.proj[i].culled) continue;
            int x0, x1, y0, y1;
            tile_range(vp.proj[i], &x0, &x1, &y0, &y1);
            for (int ty = y0; ty <= y1; ty++)
                for (int tx = x0; tx <= x1; tx++)
                    vp.tile_count[(size_t)ty * tiles_x + tx]++;
        }
        vp.tile_start[0] = 0;
        for (int t = 0; t < n_tiles; t++)
            vp.tile_start[(size_t)t + 1] =
                    vp.tile_start[(size_t)t] + vp.tile_count[(size_t)t];
        vp.keys.resize((size_t)vp.tile_start[(size_t)n_tiles]);
        std::copy(vp.tile_start.begin(), vp.tile_start.end() - 1,
                  vp.tile_count.begin());
        for (int i = 0; i < N; i++) {
            if (vp.proj[i].culled) continue;
            int x0, x1, y0, y1;
            tile_range(vp.proj[i], &x0, &x1, &y0, &y1);
            for (int ty = y0; ty <= y1; ty++)
                for (int tx = x0; tx <= x1; tx++)
                    vp.keys[(size_t)vp.tile_count[(size_t)ty * tiles_x +
                                                  tx]++] = {vp.proj[i].depth,
                                                            i};
        }
        vp.splats.resize(vp.keys.size());
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(dynamic, 4))
#endif
        for (int t = 0; t < n_tiles; t++) {
            const size_t b = (size_t)vp.tile_start[(size_t)t],
                         e = (size_t)vp.tile_start[(size_t)t + 1];
            std::sort(vp.keys.begin() + (std::ptrdiff_t)b,
                      vp.keys.begin() + (std::ptrdiff_t)e);
            for (size_t k = b; k < e; k++) {
                const int i = vp.keys[k].second;
                const ProjGaussian& pg = vp.proj[i];
                const float* src = &g.data[(size_t)i * stride];
                Splat& sp = vp.splats[k];
                sp.mx = pg.mx;
                sp.my = pg.my;
                sp.radius = pg.radius;
                sp.conic[0] = pg.conic[0];
                sp.conic[1] = pg.conic[1];
                sp.conic[2] = pg.conic[2];
                sp.opacity = src[3];
                sp.sigma_max = 2.0f * std::log(src[3] / kAlphaFloor) + 1e-3f;
                sp.depth = pg.depth;
                sp.color[0] = src[11];
                sp.color[1] = src[12];
                sp.color[2] = src[13];
            }
        }

        // ---- reverse pixel walk (gsplat RasterizeToPixels3DGSBwd) ----
        const float* alpha_out = &fwd.alphas[(size_t)v * H * W];  // 1 - T_final
        const float* dimg = d_images + (size_t)v * 3 * H * W;
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(dynamic, 4) num_threads(nthreads))
#endif
        for (int t = 0; t < n_tiles; t++) {
#ifdef QS_USE_OPENMP
            Seeds& sd =
                    seeds[(size_t)cloudViewer::utility::GetOpenMPThreadNum()];
#else
            Seeds& sd = seeds[0];
#endif
            const int px0 = (t % tiles_x) * kTile, py0 = (t / tiles_x) * kTile;
            const int tw = std::min(W - px0, kTile),
                      th = std::min(H - py0, kTile);
            const int ks = vp.tile_start[(size_t)t];
            for (int ly = 0; ly < th; ly++) {
                for (int lx = 0; lx < tw; lx++) {
                    const size_t o = (size_t)(py0 + ly) * W + (px0 + lx);
                    const int32_t last = fwd.last_k[(size_t)v * H * W + o];
                    if (last < ks) continue;  // nothing blended in this pixel
                    const float G[3] = {dimg[o], dimg[(size_t)H * W + o],
                                        dimg[(size_t)2 * H * W + o]};
                    if (G[0] == 0.f && G[1] == 0.f && G[2] == 0.f) continue;
                    const float T_final = 1.0f - alpha_out[o];
                    float T = T_final;
                    float buffer[3] = {0.f, 0.f, 0.f};
                    const float px = (px0 + lx) + 0.5f, py = (py0 + ly) + 0.5f;
                    for (int k = last; k >= ks; k--) {
                        const Splat& sp = vp.splats[(size_t)k];
                        PixelAlpha pa;
                        if (!eval_splat(sp, px, py, &pa)) continue;
                        const float a = std::min(pa.a_raw, kMaxAlpha);
                        const float ra = 1.0f / (1.0f - a);
                        T *= ra;  // T: after -> before gaussian k
                        const float fac = a * T;
                        const int gi = vp.keys[(size_t)k].second;
                        float* vc = &sd.conic[(size_t)gi * 3];
                        float* vx = &sd.xy[(size_t)gi * 2];
                        float* vk = &sd.col[(size_t)gi * color_dim];
                        float v_alpha = 0.f;
                        for (int c = 0; c < 3; c++) {
                            vk[c] += fac * G[c];
                            // (c_k*T - buffer_k*ra) * G_k, minus the
                            // background path T_final*ra*(bg*G)
                            v_alpha +=
                                    (sp.color[c] * T - buffer[c] * ra) * G[c];
                            v_alpha -= T_final * ra * cfg.bg_color[c] * G[c];
                        }
                        if (pa.a_raw <= kMaxAlpha) {  // not clamped: grad flows
                            const float v_sigma = -pa.a_raw * v_alpha;
                            // delta = mean - pixel (gsplat convention)
                            const float ddx = -pa.dx, ddy = -pa.dy;
                            vc[0] += 0.5f * v_sigma * ddx * ddx;
                            vc[1] += v_sigma * ddx * ddy;
                            vc[2] += 0.5f * v_sigma * ddy * ddy;
                            vx[0] += v_sigma *
                                     (sp.conic[0] * ddx + sp.conic[1] * ddy);
                            vx[1] += v_sigma *
                                     (sp.conic[1] * ddx + sp.conic[2] * ddy);
                            sd.op[(size_t)gi] += pa.vis * v_alpha;
                        }
                        for (int c = 0; c < 3; c++)
                            buffer[c] += sp.color[c] * fac;
                    }
                }
            }
        }

        // ---- reduce threads + projection chain (per gaussian) ----
#ifdef QS_USE_OPENMP
        CV_OMP_PARALLEL_FOR(schedule(static))
#endif
        for (int i = 0; i < N; i++) {
            float vconic[3] = {0.f, 0.f, 0.f}, vxy[2] = {0.f, 0.f};
            float vcol[3] = {0.f, 0.f, 0.f}, vop = 0.f;
            for (const Seeds& sd : seeds) {
                for (int c = 0; c < 3; c++) {
                    vconic[c] += sd.conic[(size_t)i * 3 + c];
                    vcol[c] += sd.col[(size_t)i * color_dim + c];
                }
                vxy[0] += sd.xy[(size_t)i * 2];
                vxy[1] += sd.xy[(size_t)i * 2 + 1];
                vop += sd.op[(size_t)i];
            }
            const bool zero = vconic[0] == 0.f && vconic[1] == 0.f &&
                              vconic[2] == 0.f && vxy[0] == 0.f &&
                              vxy[1] == 0.f && vop == 0.f;
            for (int c = 0; c < color_dim; c++)
                out_grads->color[(size_t)i * color_dim + c] += vcol[c];
            out_grads->opacity[(size_t)i] += vop;
            if (zero) continue;
            // conic/mean2d seeds -> camera-space grads
            const float* src = &g.data[(size_t)i * stride];
            float v_cov2d[3] = {0.f, 0.f, 0.f};
            inverse_vjp2(vp.proj[i].conic, vconic, v_cov2d);
            // camera-space mean/covar of this gaussian
            const float px = src[0], py = src[1], pz = src[2];
            const float mc[3] = {
                    w2c_v[0] * px + w2c_v[1] * py + w2c_v[2] * pz + w2c_v[3],
                    w2c_v[4] * px + w2c_v[5] * py + w2c_v[6] * pz + w2c_v[7],
                    w2c_v[8] * px + w2c_v[9] * py + w2c_v[10] * pz + w2c_v[11]};
            const float* Cc = &cov_world[(size_t)i * 6];
            float vmc[3] = {0.f, 0.f, 0.f},
                  vcc[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            persp_proj_vjp(mc, Cc, fx, fy, cx, cy, W, H, v_cov2d, vxy, vmc,
                           vcc);
            // camera -> world
            // v_pW = R^T v_pC (R = w2c rotation rows)
            float* dpos = &out_grads->pos[(size_t)i * 3];
            dpos[0] +=
                    w2c_v[0] * vmc[0] + w2c_v[4] * vmc[1] + w2c_v[8] * vmc[2];
            dpos[1] +=
                    w2c_v[1] * vmc[0] + w2c_v[5] * vmc[1] + w2c_v[9] * vmc[2];
            dpos[2] +=
                    w2c_v[2] * vmc[0] + w2c_v[6] * vmc[1] + w2c_v[10] * vmc[2];
            // v_covarW = R^T v_covarC R (symmetric accumulate)
            static const int CIDX[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
            float Cc3[3][3];
            for (int p = 0; p < 3; p++)
                for (int q = 0; q < 3; q++) Cc3[p][q] = vcc[CIDX[p][q]];
            float vW[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            for (int p = 0; p < 3; p++)
                for (int q = p; q < 3; q++) {
                    float acc = 0.f;
                    for (int r = 0; r < 3; r++)
                        for (int s = 0; s < 3; s++)
                            acc += w2c_v[r * 4 + p] * Cc3[r][s] *
                                   w2c_v[s * 4 + q];
                    vW[CIDX[p][q]] += acc;
                }
            float vq[4] = {0.f, 0.f, 0.f, 0.f}, vs[3] = {0.f, 0.f, 0.f};
            quat_scale_to_covar_vjp(src + 7, src + 4, vW, vq, vs);
            float* drot = &out_grads->rot[(size_t)i * 4];
            float* dscl = &out_grads->scale[(size_t)i * 3];
            for (int c = 0; c < 4; c++) drot[c] += vq[c];
            for (int c = 0; c < 3; c++) dscl[c] += vs[c];
        }
    }
    return true;
}

void render_to_rgb8(const float* rgb,
                    int width,
                    int height,
                    unsigned char* out_rgb8) {
    // rgb is the render-output plane layout (3, H, W): read per channel and
    // interleave. Reading it as interleaved (rgb[i]) produced the
    // notorious "3x3 tiled grayscale" images — the f32 bins were always
    // right (upstream pitfall #7; always audit bins, not PNGs).
    const size_t plane = (size_t)width * height;
    for (int c = 0; c < 3; c++) {
        const float* ch = rgb + (size_t)c * plane;
        for (size_t i = 0; i < plane; i++) {
            float v = std::min(std::max(ch[i], 0.0f), 1.0f);
            // infer.py: (clamp(0, 1).numpy() * 255).astype(np.uint8) —
            // truncation, not rounding
            out_rgb8[i * 3 + c] = (unsigned char)(v * 255.0f);
        }
    }
}

}  // namespace qs
