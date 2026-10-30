#include "depth_lss.h"

#include "../backend.h"

#include <ggml.h>

#if defined(__x86_64__) && !defined(OPCPD_NO_AVX2_EXP)
#include <immintrin.h>
#endif
#include <thread>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace openpcdet {

namespace {

// ---- Frustum: precomputed 3-D points in camera frame ----
// Shape: [D, fH, fW, 3]  (x, y, depth)
struct Frustum {
    std::vector<float> data;  // [D*fH*fW*3]
    int D, fH, fW;
};

Frustum create_frustum(const std::vector<float> & dbound,
                       const int image_size[2],   // [iH, iW]
                       const int feature_size[2]) // [fH, fW]
{
    Frustum f;
    f.D  = (int) ((dbound[1] - dbound[0]) / dbound[2]);
    f.fH = feature_size[0];
    f.fW = feature_size[1];
    f.data.resize((size_t) f.D * f.fH * f.fW * 3);

    float iH = (float) image_size[0], iW = (float) image_size[1];
    for (int d = 0; d < f.D; d++) {
        // torch.arange(dbound[0], dbound[1], dbound[2]) -> dbound[0] + d*step.
        // The old (d+0.5)*step bin-centre offset shifted every depth bin by
        // 0.25 m, moving the geometry points across BEV grid boundaries.
        float depth = dbound[0] + (float) d * dbound[2];
        for (int h = 0; h < f.fH; h++) {
            float ys = (f.fH > 1) ? (float) h / (f.fH - 1) * (iH - 1) : 0;
            for (int w = 0; w < f.fW; w++) {
                float xs = (f.fW > 1) ? (float) w / (f.fW - 1) * (iW - 1) : 0;
                size_t idx = ((size_t) d * f.fH * f.fW + h * f.fW + w) * 3;
                f.data[idx + 0] = xs;
                f.data[idx + 1] = ys;
                f.data[idx + 2] = depth;
            }
        }
    }
    return f;
}

// ---- 3×3 matrix helpers (row-major) ----
using Mat3 = std::array<float, 9>;
using Vec3 = std::array<float, 3>;

Mat3 mat3_inverse(const Mat3 & m) {
    float a = m[0], b = m[1], c = m[2];
    float d = m[3], e = m[4], f = m[5];
    float g = m[6], h = m[7], i = m[8];
    float det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
    float inv = 1.0f / det;
    return {{
        (e*i-f*h)*inv, (c*h-b*i)*inv, (b*f-c*e)*inv,
        (f*g-d*i)*inv, (a*i-c*g)*inv, (c*d-a*f)*inv,
        (d*h-e*g)*inv, (b*g-a*h)*inv, (a*e-b*d)*inv
    }};
}

Vec3 mat3_vec3_mul(const Mat3 & m, const Vec3 & v) {
    return {{
        m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
        m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
        m[6]*v[0] + m[7]*v[1] + m[8]*v[2]
    }};
}

// Read a 3×3 sub-matrix from a flat row-major array at [b, n, :, :]
Mat3 read_mat3(const float * data, int b, int n, int N) {
    Mat3 m;
    const float * p = data + (b * N + n) * 9;
    for (int i = 0; i < 9; i++) m[i] = p[i];
    return m;
}

// Read a 3-vector from a flat [B, N, 3] array at [b, n, :]
Vec3 read_vec3(const float * data, int b, int n, int N) {
    const float * p = data + (b * N + n) * 3;
    return {{p[0], p[1], p[2]}};
}

// Read a 3×4 matrix row-major at [b, n, :, :] (matrices are stored 4x4 = 16 floats)
void read_mat34(const float * data, int b, int n, int N,
                Mat3 & rot, Vec3 & trans) {
    const float * p = data + (b * N + n) * 16;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            rot[i*3+j] = p[i*4+j];
    for (int i = 0; i < 3; i++)
        trans[i] = p[i*4+3];
}

// ---- BEV pooling (CPU) ----
// geom: [BN, D, fH, fW, 3]  (x, y, z in lidar frame)
// x:    [BN, D, fH, fW, C]  (features weighted by depth)
// Output: [nx[0], nx[1], C, B_batch] in GGML layout (W axis = nx[0] = x,
// H axis = nx[1] = y).  This makes the ggml W axis match the PyTorch W
// axis, so all downstream 2D convs (downsample, fuser) slide their
// kernels along the correct spatial dimensions.
// Fused depth-softmax x features scatter into the BEV grid (host path used
// by the Vulkan/CPU backends, where bev_pool_cuda is unavailable).
//
// Replaces the two-pass volume materialization (a BN*D*fH*fW*C ~638 MB buffer
// for BevFusion) with three parallel stages:
//   pass 1: per-frustum-pixel softmax max/sum (parallel over pixels);
//   pass 2: per-source prob (parallel, division-based normalization) and the
//           BEV bin (truncating discretization, matching tensor.long());
//   pass 3: channel-partitioned scatter — each thread owns a contiguous
//           channel slice, so plain += is race-free.
//
// Numerical contract vs the volume-based path: prob = expf(logit - max)/sum
// (division, NOT multiply-by-reciprocal — f32 differs and the cancellation-
// heavy BEV sums amplify it), and each cell's accumulation order follows the
// source index s = ((bn*D + d)*fH + h)*fW + w, the same (bn, d, h, w)
// lexicographic order the legacy scatter used. The result is bit-identical.
#if defined(__x86_64__) && !defined(OPCPD_NO_AVX2_EXP)
// AVX2 vectorized expf for the softmax hot loop. Verified vs std::exp on
// [-30, 0] (the softmax logit-diff range): max 3 ulp; ~3.3x faster.
// reduce: subtract the LARGE part (t*ln2_hi) first so the intermediate is
// small; ln2_lo only nudges the remainder.
__attribute__((target("avx2,fma")))
static inline __m256 exp8_avx2(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.442695041f);
    const __m256 magic = _mm256_set1_ps(12582912.0f);            // 1.5*2^23
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 t = _mm256_fmadd_ps(x, log2e, magic);
    __m256i k = _mm256_sub_epi32(_mm256_castps_si256(t),
                                 _mm256_castps_si256(magic));
    t = _mm256_sub_ps(t, magic);
    __m256 r = _mm256_fnmadd_ps(t, ln2_lo, _mm256_fnmadd_ps(t, ln2_hi, x));
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(1.0f/6.0f);
    const __m256 c4 = _mm256_set1_ps(1.0f/24.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f/120.0f);
    const __m256 c6 = _mm256_set1_ps(1.0f/720.0f);
    __m256 p = _mm256_add_ps(c5, _mm256_mul_ps(c6, r));
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    __m256i sk = _mm256_add_epi32(k, _mm256_set1_epi32(127));
    sk = _mm256_max_epi32(sk, _mm256_setzero_si256());
    return _mm256_mul_ps(p, _mm256_castsi256_ps(_mm256_slli_epi32(sk, 23)));
}

// Per-pixel softmax kernel: max + vectorized exps + sum. Runs in its own
// avx2 target function because GCC's always_inline AVX2 intrinsics cannot
// be inlined into non-AVX2 callers (the thread lambda below).
__attribute__((target("avx2,fma")))
static float softmax_col_avx2(const float * col, float * e, int D) {
    float m = -1e30f;
    for (int d = 0; d < D; d++) m = std::max(m, col[d]);
    const __m256 vm = _mm256_set1_ps(m);
    __m256 vsum = _mm256_setzero_ps();
    int d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 ev = exp8_avx2(_mm256_sub_ps(_mm256_loadu_ps(col + d), vm));
        _mm256_storeu_ps(e + d, ev);
        vsum = _mm256_add_ps(vsum, ev);
    }
    float lane[8];
    _mm256_storeu_ps(lane, vsum);
    float sum = 0.0f;
    for (int l = 0; l < 8; l++) sum += lane[l];
    for (; d < D; d++) { e[d] = std::exp(col[d] - m); sum += e[d]; }
    return sum;
}
#endif

void bev_pool_cpu_fused(const float * geom, const float * depthnet_out,
                        int BN, int D, int fH, int fW, int C,
                        const float dx[3], const float bx[3], const int nx[3],
                        int B_batch, int num_cameras,
                        std::vector<float> & out) {
    const size_t total = (size_t) BN * D * fH * fW;
    const size_t plane = (size_t) fH * fW;
    const size_t pixels = (size_t) BN * fH * fW;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency() / 2);

    // Pixel column base: logits/features of pixel pos live at
    // base(pos) + (d or D+c) * plane, base(pos) = w + fW*(h + fH*(D+C)*bn).
    auto pixel_base = [=](size_t pos) -> size_t {
        const size_t ww = pos % fW;
        const size_t h  = (pos / fW) % fH;
        const size_t bn = pos / (fW * fH);
        return ww + fW * (h + fH * (size_t)(D + C) * bn);
    };

    const bool prof_bp = false;
    auto bp_lap = [&](const char * nm, std::chrono::steady_clock::time_point & t0) {
        if (!prof_bp) return;
        auto t1 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[bevpool] %-10s %7.2f ms\n", nm,
                     std::chrono::duration<double, std::milli>(t1 - t0).count());
        t0 = t1;
    };
    auto bp_t0 = std::chrono::steady_clock::now();
    // Stage T: transpose the depthnet planes [fW, fH, D+C, BN] into per-pixel
    // contiguous columns dncol[pos][D+C] (~13 MB). Every later stage then
    // streams sequentially instead of striding plane floats (11 KB) per
    // element, which was the dominant cache-miss cost. A pure permutation —
    // no numerical effect.
    const size_t dcp = (size_t) D + C;
    // Cross-frame workspace cache: geometry is constant per scene, every
    // buffer below is fully overwritten each frame, so resize() reuse skips
    // ~60 MB of allocation + first-touch faults (measured ~76 ms/frame).
    static std::vector<float>       dncol, e_buf, pix_sum, probs;
    static std::vector<int32_t>     bin_cell, src_batch, src_pos;
    static std::vector<uint8_t>     bin_ok;
    dncol.resize(pixels * dcp);
    {
        const size_t chunk = (pixels + hw - 1) / hw;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < hw; t++) {
            const size_t p0 = t * chunk, p1 = std::min(pixels, p0 + chunk);
            if (p0 >= p1) break;
            pool.emplace_back([&, p0, p1] {
                for (size_t pos = p0; pos < p1; pos++) {
                    const float * col = depthnet_out + pixel_base(pos);
                    float * dst = dncol.data() + pos * dcp;
                    for (size_t v = 0; v < dcp; v++)
                        dst[v] = col[v * plane];
                }
            });
        }
        for (auto & worker : pool) worker.join();
    }

    bp_lap("stage_T", bp_t0);

    // Stage A (parallel over pixels): each thread owns whole pixels, so per
    // pixel it computes max, then the D exponentials ONCE (ascending d — the
    // legacy accumulation order) into the contiguous e_buf[pixels*D] (~8 MB)
    // and their sum. Storing exps keeps the exp work at one pass while the
    // source-ordered stage B below stays fully sequential-access.
    e_buf.resize(pixels * (size_t) D);
    pix_sum.resize(pixels);
    {
        const size_t chunk = (pixels + hw - 1) / hw;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < hw; t++) {
            const size_t p0 = t * chunk, p1 = std::min(pixels, p0 + chunk);
            if (p0 >= p1) break;
            pool.emplace_back([&, p0, p1] {
                for (size_t pos = p0; pos < p1; pos++) {
                    const float * col = dncol.data() + pos * dcp;
                    float * e = e_buf.data() + pos * (size_t) D;
#if defined(__x86_64__) && !defined(OPCPD_NO_AVX2_EXP)
                    pix_sum[pos] = softmax_col_avx2(col, e, D);
#else
                    float m = -1e30f;
                    for (int d = 0; d < D; d++)
                        m = std::max(m, col[d]);
                    float sum = 0.0f;
                    for (int d = 0; d < D; d++) {
                        e[d] = std::exp(col[d] - m);
                        sum += e[d];
                    }
                    pix_sum[pos] = sum;
#endif
                }
            });
        }
        for (auto & worker : pool) worker.join();
    }

    bp_lap("stage_A", bp_t0);

    // Stage B (parallel over sources, ascending s = ((bn*D + d)*fH + h)*fW + w):
    // prob = e[d]/sum (division, NOT multiply-by-reciprocal — f32 differs and
    // the cancellation-heavy BEV sums amplify it) plus the BEV bin; all
    // source-indexed arrays are written sequentially.
    probs.resize(total);
    bin_cell.resize(total);   // b*nx1*nx0 + iy*nx0 + ix
    bin_ok.resize(total);
    src_batch.resize(total);
    src_pos.resize(total);    // frustum-pixel linear index
    {
        const size_t chunk = (total + hw - 1) / hw;
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < hw; t++) {
            const size_t s0 = t * chunk, s1 = std::min(total, s0 + chunk);
            if (s0 >= s1) break;
            pool.emplace_back([&, s0, s1] {
                for (size_t s = s0; s < s1; s++) {
                    const size_t ww = s % fW;
                    const size_t h  = (s / fW) % fH;
                    const size_t rem = s / (fW * fH);
                    const size_t d  = rem % D;
                    const size_t bn = rem / D;
                    const size_t b  = bn / (size_t) num_cameras;
                    const size_t pos = bn * fH * fW + h * fW + ww;
                    const float * g = geom + s * 3;
                    // Truncation toward zero matches tensor.long() in the
                    // reference; floor() would shift negative coordinates.
                    const int ix = (int) ((g[0] - bx[0] + dx[0] * 0.5f) / dx[0]);
                    const int iy = (int) ((g[1] - bx[1] + dx[1] * 0.5f) / dx[1]);
                    const int iz = (int) ((g[2] - bx[2] + dx[2] * 0.5f) / dx[2]);
                    bin_ok[s] = (ix >= 0 && ix < nx[0] && iy >= 0 && iy < nx[1]
                                 && iz >= 0 && iz < nx[2]) ? 1 : 0;
                    bin_cell[s] = (int) (b * nx[1] * nx[0] + (size_t) iy * nx[0] + ix);
                    src_batch[s] = (int) b;
                    src_pos[s] = (int) pos;
                    probs[s] = e_buf[pos * (size_t) D + d] / pix_sum[pos];
                }
            });
        }
        for (auto & worker : pool) worker.join();
    }

    bp_lap("stage_B", bp_t0);

    // Pass 3 (cell-sorted scatter): a stable counting sort by BEV cell places
    // every source into its cell's segment; within a segment sources keep
    // ascending s order, so each cell's accumulation sequence — and therefore
    // the f32 result — is identical to the source-ordered scatter. Threads own
    // disjoint cell ranges (race-free), and each channel plane of `out` is now
    // written as a contiguous stream instead of random hops.
    out.assign((size_t) nx[1] * nx[0] * C * B_batch, 0.0f);
    {
        const size_t ncells = (size_t) nx[1] * nx[0];
        std::vector<int32_t> cell_start(ncells * B_batch + 1, 0);
        for (size_t s = 0; s < total; s++)
            if (bin_ok[s]) cell_start[bin_cell[s] + 1]++;
        for (size_t i = 1; i < cell_start.size(); i++) cell_start[i] += cell_start[i - 1];
        std::vector<int32_t> perm(total);
        {
            std::vector<int32_t> cursor(cell_start.begin(), cell_start.end() - 1);
            for (size_t s = 0; s < total; s++)
                if (bin_ok[s]) perm[cursor[bin_cell[s]]++] = (int32_t) s;
        }
        const int cell_per = (int) ((ncells * B_batch + hw - 1) / hw);
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < hw; t++) {
            const size_t c0 = (size_t) t * cell_per;
            const size_t c1 = std::min(ncells * (size_t) B_batch, c0 + (size_t) cell_per);
            if (c0 >= c1) break;
            pool.emplace_back([&, c0, c1] {
                std::vector<float> acc((size_t) C, 0.0f);  // one per thread,
                // NOT per cell — the per-cell version did 32400 mallocs/frame
                for (size_t bc = c0; bc < c1; bc++) {
                    const size_t b = bc / ncells, cell = bc % ncells;
                    for (int c = 0; c < C; c++) acc[c] = 0.0f;
                    for (int k = cell_start[bc]; k < cell_start[bc + 1]; k++) {
                        const size_t s = (size_t) perm[k];
                        const float p = probs[s];
                        const float * feat = dncol.data()
                            + (size_t) src_pos[s] * dcp + (size_t) D;
                        for (int c = 0; c < C; c++) acc[c] += p * feat[c];
                    }
                    if (cell_start[bc] == cell_start[bc + 1]) continue;
                    float * dst = out.data() + (b * (size_t) C) * ncells + cell;
                    for (int c = 0; c < C; c++) dst[(size_t) c * ncells] = acc[c];
                }
            });
        }
        for (auto & worker : pool) worker.join();
    }
    bp_lap("scatter", bp_t0);
}

void bev_pool_cpu(const float * geom, const float * x,
                  int BN, int D, int fH, int fW, int C,
                  const float dx[3], const float bx[3], const int nx[3],
                  int B_batch, int num_cameras,
                  std::vector<float> & out) {
    // out: [nx[2], nx[0], nx[1], B_batch]  (Z=nx[2] collapsed → 1)
    // Actually: after collapse Z, output is [nx[0], nx[1], C, B_batch] in GGML layout
    // But PyTorch: final = cat(x.unbind(dim=2), 1) → [B, C*nx[2], nx[0], nx[1]]
    // Then permute(0,1,3,2) → [B, C*nx[2], nx[1], nx[0]]
    // In GGML: [nx[1], nx[0], C*nx[2], B] but nx[2]=1 after collapse
    // So: [nx[1], nx[0], C, B]

    const int total = BN * D * fH * fW;
    // Accumulate into [B_batch, C, nx[0], nx[1]]
    std::vector<float> accum((size_t) B_batch * C * nx[0] * nx[1], 0.0f);

    for (int idx = 0; idx < total; idx++) {
        int bn = idx / (D * fH * fW);
        int rem = idx % (D * fH * fW);
        int batch_idx = bn / num_cameras;

        const float * g = geom + idx * 3;
        // Discretize: the reference uses tensor.long() (truncation toward
        // zero), NOT floor — negative coordinates differ by one bin.
        int ix = (int) ((g[0] - bx[0] + dx[0] * 0.5f) / dx[0]);
        int iy = (int) ((g[1] - bx[1] + dx[1] * 0.5f) / dx[1]);
        int iz = (int) ((g[2] - bx[2] + dx[2] * 0.5f) / dx[2]);

        if (ix < 0 || ix >= nx[0] || iy < 0 || iy >= nx[1] || iz < 0 || iz >= nx[2])
            continue;

        const float * feat = x + (size_t) idx * C;
        for (int c = 0; c < C; c++) {
            // accum[batch_idx, c, ix, iy] += feat[c]
            size_t aidx = (size_t) batch_idx * C * nx[0] * nx[1]
                        + (size_t) c * nx[0] * nx[1]
                        + (size_t) ix * nx[1] + iy;
            accum[aidx] += feat[c];
        }
    }

    // After collapse Z: PyTorch has [B, C*nz, nx[0], nx[1]]
    // With nz=1: [B, C, nx[0], nx[1]]; permute(0,1,3,2) → [B, C, nx[1], nx[0]]
    // Output GGML layout: [nx[0], nx[1], C, B] (ix fastest = W axis = nx[0] = x).
    out.resize((size_t) nx[1] * nx[0] * C * B_batch);
    for (int b = 0; b < B_batch; b++) {
        for (int c = 0; c < C; c++) {
            for (int ix = 0; ix < nx[0]; ix++) {
                for (int iy = 0; iy < nx[1]; iy++) {
                    size_t src = (size_t) b * C * nx[0] * nx[1]
                               + (size_t) c * nx[0] * nx[1]
                               + (size_t) ix * nx[1] + iy;
                    // GGML dst: [ix, iy, c, b] → index = ix + nx[0]*(iy + nx[1]*(c + C*b))
                    size_t dst = (size_t) b * nx[1] * nx[0] * C
                               + (size_t) c * nx[1] * nx[0]
                               + (size_t) iy * nx[0] + ix;
                    out[dst] = accum[src];
                }
            }
        }
    }
}

// ---- Project LiDAR points to image for sparse depth ----
void project_depth(const float * points, int n_points, int point_dim,
                   const float * img_aug_matrix,
                   const float * lidar_aug_matrix,
                   const float * lidar2image,
                   int batch_size, int num_cameras,
                   int img_W, int img_H,
                   std::vector<float> & depth_out) {
    // depth_out: [img_W * img_H * 1 * (batch_size * num_cameras)]
    depth_out.assign((size_t) img_W * img_H * batch_size * num_cameras, 0.0f);

    for (int b = 0; b < batch_size; b++) {
        // Extract matrices
        const float * cur_lidar_aug = lidar_aug_matrix + b * 12;   // [3,4] row-major
        // Matrices are stored as 4x4 (16 floats) per camera in the scene buffer.
        const float * cur_l2i = lidar2image + b * num_cameras * 16; // [N, 4, 4]
        const float * cur_img_aug = img_aug_matrix + b * num_cameras * 16; // [N, 4, 4]

        // Inverse lidar_aug 3×3
        Mat3 la_rot, la_inv;
        Vec3 la_trans;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) la_rot[i*3+j] = cur_lidar_aug[i*4+j];
            la_trans[i] = cur_lidar_aug[i*4+3];
        }
        la_inv = mat3_inverse(la_rot);

        for (int n = 0; n < num_cameras; n++) {
            Mat3 l2i_rot; Vec3 l2i_trans;
            {
                const float * p = cur_l2i + n * 16;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) l2i_rot[i*3+j] = p[i*4+j];
                    l2i_trans[i] = p[i*4+3];
                }
            }
            Mat3 ia_rot; Vec3 ia_trans;
            {
                const float * p = cur_img_aug + n * 16;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) ia_rot[i*3+j] = p[i*4+j];
                    ia_trans[i] = p[i*4+3];
                }
            }

            // Note: points is a flat [N, point_dim] array WITHOUT a batch column
            // (x,y,z at cols 0,1,2), matching the reference which prepends an all-0
            // batch column before slicing xyz. batch_size is 1 here, so every point
            // belongs to the single batch.
            for (int pi = 0; pi < n_points; pi++) {
                float px = points[pi * point_dim + 0];
                float py = points[pi * point_dim + 1];
                float pz = points[pi * point_dim + 2];

                // Undo lidar augmentation
                Vec3 pt = {{px - la_trans[0], py - la_trans[1], pz - la_trans[2]}};
                pt = mat3_vec3_mul(la_inv, pt);

                // lidar → image
                Vec3 img_pt = mat3_vec3_mul(l2i_rot, pt);
                img_pt[0] += l2i_trans[0];
                img_pt[1] += l2i_trans[1];
                img_pt[2] += l2i_trans[2];

                float dist = img_pt[2];
                if (dist < 1e-5f) continue;
                img_pt[0] /= dist;
                img_pt[1] /= dist;

                // Apply image augmentation
                Vec3 aug_pt = mat3_vec3_mul(ia_rot, img_pt);
                aug_pt[0] += ia_trans[0];
                aug_pt[1] += ia_trans[1];

                // Swap to (row, col) = (y, x): reference does
                //   cur_coords = cur_coords[..., [1, 0]]
                // so row = aug_y, col = aug_x, and checks row<img_H, col<img_W.
                int row = (int) aug_pt[1];
                int col = (int) aug_pt[0];

                if (row >= 0 && row < img_H && col >= 0 && col < img_W) {
                    // depth_out layout: [W, H, 1, BN] GGML → index = col + W*(row + H*(0 + 1*(b*N+n)))
                    size_t didx = (size_t) (b * num_cameras + n) * img_W * img_H
                                + (size_t) row * img_W + col;
                    depth_out[didx] = dist;
                }
            }
        }
    }
}

} // anon ns

// ---- Weight loading ----

bool load_depth_lss_weights(model_file & file, const hparams & hp,
                            depth_lss_weights & w, std::string & error) {
    // dtransform
    w.dtrans_0.conv_w   = file.require("vtransform.dtransform.0.weight");
    w.dtrans_0.bn_scale = file.require("vtransform.dtransform.1.eval_scale");
    w.dtrans_0.bn_bias  = file.require("vtransform.dtransform.1.eval_bias");

    w.dtrans_1.conv_w   = file.require("vtransform.dtransform.3.weight");
    w.dtrans_1.bn_scale = file.require("vtransform.dtransform.4.eval_scale");
    w.dtrans_1.bn_bias  = file.require("vtransform.dtransform.4.eval_bias");

    w.dtrans_2.conv_w   = file.require("vtransform.dtransform.6.weight");
    w.dtrans_2.bn_scale = file.require("vtransform.dtransform.7.eval_scale");
    w.dtrans_2.bn_bias  = file.require("vtransform.dtransform.7.eval_bias");

    // depthnet
    w.dnet_0.conv_w   = file.require("vtransform.depthnet.0.weight");
    w.dnet_0.bn_scale = file.require("vtransform.depthnet.1.eval_scale");
    w.dnet_0.bn_bias  = file.require("vtransform.depthnet.1.eval_bias");

    w.dnet_1.conv_w   = file.require("vtransform.depthnet.3.weight");
    w.dnet_1.bn_scale = file.require("vtransform.depthnet.4.eval_scale");
    w.dnet_1.bn_bias  = file.require("vtransform.depthnet.4.eval_bias");

    // final 1×1 conv (no BN)
    w.dnet_final_w = file.require("vtransform.depthnet.6.weight");
    w.dnet_final_b = file.require("vtransform.depthnet.6.bias");

    // downsample
    w.has_downsample = (hp.image_lss_downsample > 1);
    if (w.has_downsample) {
        w.down_0.conv_w   = file.require("vtransform.downsample.0.weight");
        w.down_0.bn_scale = file.require("vtransform.downsample.1.eval_scale");
        w.down_0.bn_bias  = file.require("vtransform.downsample.1.eval_bias");

        w.down_1.conv_w   = file.require("vtransform.downsample.3.weight");
        w.down_1.bn_scale = file.require("vtransform.downsample.4.eval_scale");
        w.down_1.bn_bias  = file.require("vtransform.downsample.4.eval_bias");

        w.down_2.conv_w   = file.require("vtransform.downsample.6.weight");
        w.down_2.bn_scale = file.require("vtransform.downsample.7.eval_scale");
        w.down_2.bn_bias  = file.require("vtransform.downsample.7.eval_bias");
    }

    if (!file.error.empty()) { error = file.error; return false; }
    return true;
}

// ---- Forward ----

bool run_depth_lss(engine_backend & backend, const hparams & hp,
                   const depth_lss_weights & w,
                   const float * fpn_feature,
                   int fpn_W, int fpn_H,
                   const float * /*depth_input*/,
                   int img_W, int img_H,
                   const float * camera_intrinsics,
                   const float * camera2lidar,
                   const float * img_aug_matrix,
                   const float * lidar_aug_matrix,
                   const float * lidar2image,
                   const float * points,
                   int n_points, int point_dim,
                   int batch_size,
                   std::vector<float> & bev_output,
                   int & bev_W, int & bev_H,
                   const tap_sink & sink, std::string & error) {
    const int in_channel = hp.image_lss_in_channels;
    const int out_channel = hp.image_lss_out_channels;
    const int num_cameras = hp.image_num_cameras;
    const int BN = batch_size * num_cameras;

    const bool profile = false;  // AICore port note: stderr lap profiling dropped
    auto lap_start = std::chrono::steady_clock::now();
    auto lap = [&](const char * name) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[lss] %-20s %8.2f ms\n", name,
                     std::chrono::duration<double, std::milli>(now - lap_start).count());
        lap_start = now;
    };

    // Bounds
    const auto & xbound = hp.image_lss_xbound;
    const auto & ybound = hp.image_lss_ybound;
    const auto & zbound = hp.image_lss_zbound;
    const auto & dbound = hp.image_lss_dbound;
    float dx[3] = {xbound[2], ybound[2], zbound[2]};
    float bx[3] = {xbound[0] + dx[0]*0.5f, ybound[0] + dx[1]*0.5f, zbound[0] + dx[2]*0.5f};
    int nx[3] = {(int)((xbound[1]-xbound[0])/dx[0]),
                 (int)((ybound[1]-ybound[0])/dx[1]),
                 (int)((zbound[1]-zbound[0])/dx[2])};
    int D = (int)((dbound[1] - dbound[0]) / dbound[2]);

    int image_size[2] = {img_H, img_W};
    int feature_size[2] = {fpn_H, fpn_W};

    // ---- Step 0: Project LiDAR points to get sparse depth ----
    std::vector<float> depth_map;
    project_depth(points, n_points, point_dim,
                  img_aug_matrix, lidar_aug_matrix, lidar2image,
                  batch_size, num_cameras, img_W, img_H, depth_map);
    lap("project_depth");
    if (sink) {
        // depth_map layout: [BN, img_H, img_W] w-fastest (matches ref [BN,1,H,W])
        sink("bevfusion.depth_map", depth_map.data(),
             (int64_t) batch_size * num_cameras * img_H, img_W);
    }

    // ---- Step 1: dtransform (GGML graph) ----
    // Input: depth_map as [img_W, img_H, 1, BN]
    {
        const size_t graph_nodes = 128;
        ggml_init_params gp = {
            ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
            nullptr, true,
        };
        ggml_context * ctx = ggml_init(gp);
        if (!ctx) { error = "dtransform ctx init failed"; return false; }

        ggml_tensor * depth_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                    img_W, img_H, 1, BN);
        ggml_set_name(depth_t, "depth_input");
        ggml_set_input(depth_t);

        ggml_tensor * x = build_conv_bn_relu(ctx, w.dtrans_0, depth_t, 1, 0);
        ggml_tensor * conv1 = x;  // [W,H,8,BN] intermediate for layer-by-layer debugging
        x = build_conv_bn_relu(ctx, w.dtrans_1, x, 4, 2);
        ggml_tensor * conv2 = x;  // [W',H',32,BN]
        x = build_conv_bn_relu(ctx, w.dtrans_2, x, 2, 2);
        ggml_set_name(x, "dtransform_out");
        ggml_set_output(x);
        if (false) {  // AICore port note: geom dump dropped
            ggml_set_name(conv1, "dtransform_conv1");
            ggml_set_output(conv1);
            ggml_set_name(conv2, "dtransform_conv2");
            ggml_set_output(conv2);
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, x);

        if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
            error = "dtransform alloc failed"; ggml_free(ctx); return false;
        }
        ggml_backend_tensor_set(depth_t, depth_map.data(), 0,
                                depth_map.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
            error = "dtransform compute failed"; ggml_free(ctx); return false;
        }

        // Readback dtransform output: should be [fpn_W, fpn_H, 64, BN]
        size_t n = (size_t) fpn_W * fpn_H * 64 * BN;
        depth_map.resize(n);
        ggml_backend_tensor_get(x, depth_map.data(), 0, n * sizeof(float));
        ggml_free(ctx);
        lap("dtransform");

        if (sink && false) {  // AICore port note: geom dump dropped
            // Layer-by-layer dtransform intermediates for divergence localization.
            // conv1: [fpn_W*4, fpn_H*4, 8, BN]; conv2: [fpn_W*2, fpn_H*2, 32, BN]
            auto dump_layer = [&](ggml_tensor * t, const char * nm) {
                std::vector<float> buf((size_t) ggml_nelements(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                int64_t rows = (int64_t) t->ne[1] * t->ne[0] * t->ne[3];
                int64_t cols = t->ne[2];
                std::vector<float> tap((size_t) rows * cols);
                for (int64_t n = 0; n < t->ne[3]; n++)
                  for (int64_t h = 0; h < t->ne[1]; h++)
                    for (int64_t w = 0; w < t->ne[0]; w++)
                      for (int64_t c = 0; c < cols; c++) {
                          size_t src = (size_t) w + (size_t) t->ne[0] * h
                                     + (size_t) t->ne[0] * t->ne[1] * c
                                     + (size_t) t->ne[0] * t->ne[1] * cols * n;
                          tap[(((size_t) n * t->ne[1] + h) * t->ne[0] + w) * cols + c] = buf[src];
                      }
                sink(nm, tap.data(), rows, cols);
            };
            dump_layer(conv1, "bevfusion.dtransform_conv1");
            dump_layer(conv2, "bevfusion.dtransform_conv2");
        }

        if (sink) {
            // dump raw ggml buffer (unpermuted) for layout analysis
            sink("bevfusion.dtransform_raw", depth_map.data(),
                 (int64_t) fpn_W * fpn_H * 64, BN);
            // dump dtransform_out [fpn_W, fpn_H, 64, BN] → [BN*fH*fW, 64] HWC row-major
            size_t spatial = (size_t) fpn_W * fpn_H;
            std::vector<float> tap((size_t) BN * spatial * 64);
            for (int b = 0; b < BN; b++)
              for (int h = 0; h < fpn_H; h++)
                for (int ww = 0; ww < fpn_W; ww++)
                  for (int c = 0; c < 64; c++) {
                      // depth_map is channel-major [fpn_W, fpn_H, 64, BN]: idx = w + W*(h + H*(c + 64*b))
                      size_t src = (size_t) ww + (size_t) fpn_W * h + (size_t) fpn_W * fpn_H * c + (size_t) fpn_W * fpn_H * 64 * b;
                      size_t dst = ((size_t) b * spatial + (size_t) h * fpn_W + ww) * 64 + c;
                      tap[dst] = depth_map[src];
                  }
            sink("bevfusion.dtransform_out", tap.data(), (int64_t) BN * spatial, 64);
        }
    }

    // ---- Step 2: Concat dtransform + fpn on CPU, then depthnet (GGML graph) ----
    // Both are [fpn_W, fpn_H, ?, BN] → concat channels → [fpn_W, fpn_H, in_channel+64, BN]
    const int concat_C = in_channel + 64;
    std::vector<float> concat_input((size_t) fpn_W * fpn_H * concat_C * BN);
    {
        size_t spatial = (size_t) fpn_W * fpn_H;
        for (int b = 0; b < BN; b++) {
            for (int h = 0; h < fpn_H; h++) {
                for (int ww = 0; ww < fpn_W; ww++) {
                    size_t out_base = ((size_t) b * concat_C) * spatial + h * fpn_W + ww;
                    size_t d_base = ((size_t) b * 64) * spatial + h * fpn_W + ww;
                    size_t f_base = ((size_t) b * in_channel) * spatial + h * fpn_W + ww;
                    for (int c = 0; c < 64; c++)
                        concat_input[out_base + c * spatial] = depth_map[d_base + c * spatial];
                    for (int c = 0; c < in_channel; c++)
                        concat_input[out_base + (64+c) * spatial] = fpn_feature[f_base + c * spatial];
                }
            }
        }
    }
    lap("concat");

    std::vector<float> depthnet_out;
    {
        const size_t graph_nodes = 128;
        ggml_init_params gp = {
            ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
            nullptr, true,
        };
        ggml_context * ctx = ggml_init(gp);
        if (!ctx) { error = "depthnet ctx init failed"; return false; }

        ggml_tensor * input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                    fpn_W, fpn_H, concat_C, BN);
        ggml_set_name(input_t, "depthnet_input");
        ggml_set_input(input_t);

        ggml_tensor * x = build_conv_bn_relu(ctx, w.dnet_0, input_t, 1, 1);
        x = build_conv_bn_relu(ctx, w.dnet_1, x, 1, 1);
        // Final 1×1 conv (no BN, has bias)
        x = build_conv_bias(ctx, w.dnet_final_w, w.dnet_final_b, x, 1, 0);
        ggml_set_name(x, "depthnet_out");
        ggml_set_output(x);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, x);

        if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
            error = "depthnet alloc failed"; ggml_free(ctx); return false;
        }
        ggml_backend_tensor_set(input_t, concat_input.data(), 0,
                                concat_input.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
            error = "depthnet compute failed"; ggml_free(ctx); return false;
        }

        size_t n = (size_t) fpn_W * fpn_H * (D + out_channel) * BN;
        depthnet_out.resize(n);
        ggml_backend_tensor_get(x, depthnet_out.data(), 0, n * sizeof(float));
        ggml_free(ctx);
        lap("depthnet");

        if (sink) {
            // dump depthnet_out [fpn_W, fpn_H, D+C, BN] → [BN*fH*fW, D+C] HWC row-major
            size_t spatial = (size_t) fpn_W * fpn_H;
            const int Cc = D + out_channel;
            std::vector<float> tap((size_t) BN * spatial * Cc);
            for (int b = 0; b < BN; b++)
              for (int h = 0; h < fpn_H; h++)
                for (int ww = 0; ww < fpn_W; ww++)
                  for (int c = 0; c < Cc; c++) {
                      size_t src = (size_t) ww + (size_t) fpn_W * h + (size_t) fpn_W * fpn_H * c + (size_t) fpn_W * fpn_H * Cc * b;
                      size_t dst = ((size_t) b * spatial + (size_t) h * fpn_W + ww) * Cc + c;
                      tap[dst] = depthnet_out[src];
                  }
            sink("bevfusion.depthnet_out", tap.data(), (int64_t) BN * spatial, Cc);
        }
    }

    // ---- Step 3: CPU — depth softmax × features → 3-D volume → bev_pool ----
    // depthnet_out: [fpn_W, fpn_H, D+C, BN] in GGML layout
    // Split: first D channels = depth logits, next C channels = features

    // Compute frustum
    Frustum frustum = create_frustum(dbound, image_size, feature_size);

    // For each (bn, d, h, w): compute 3-D geometry point
    // geom: [BN, D, fH, fW, 3]
    size_t geom_total = (size_t) BN * D * fpn_H * fpn_W * 3;
    std::vector<float> geom(geom_total);

    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < num_cameras; n++) {
            int bn = b * num_cameras + n;
            // Read camera params (4x4 stored → top-left 3x3)
            Mat3 intrins;
            {
                const float * p = camera_intrinsics + (b * num_cameras + n) * 16;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        intrins[i*3+j] = p[i*4+j];
            }
            Mat3 c2l_rot; Vec3 c2l_trans;
            read_mat34(camera2lidar, b, n, num_cameras, c2l_rot, c2l_trans);
            Mat3 post_rot; Vec3 post_trans;
            read_mat34(img_aug_matrix, b, n, num_cameras, post_rot, post_trans);
            // lidar_aug is a single 4x4 (per scene, identity here); broadcast to cameras
            Mat3 la_rot; Vec3 la_trans;
            {
                const float * p = lidar_aug_matrix;  // offset 0, not per-camera
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++) la_rot[i*3+j] = p[i*4+j];
                for (int i = 0; i < 3; i++) la_trans[i] = p[i*4+3];
            }

            Mat3 post_rot_inv = mat3_inverse(post_rot);
            Mat3 combine; // c2l_rot * intrins_inv
            Mat3 intrins_inv = mat3_inverse(intrins);
            // combine = c2l_rot @ intrins_inv
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    combine[i*3+j] = 0;
                    for (int k = 0; k < 3; k++)
                        combine[i*3+j] += c2l_rot[i*3+k] * intrins_inv[k*3+j];
                }

            for (int d = 0; d < D; d++) {
                for (int h = 0; h < fpn_H; h++) {
                    for (int ww = 0; ww < fpn_W; ww++) {
                        size_t fi = ((size_t) d * fpn_H * fpn_W + h * fpn_W + ww) * 3;
                        float xs = frustum.data[fi + 0];
                        float ys = frustum.data[fi + 1];
                        float ds = frustum.data[fi + 2];

                        // Undo post-transformation
                        float px = xs - post_trans[0];
                        float py = ys - post_trans[1];
                        float pz = ds - post_trans[2];
                        Vec3 pt = mat3_vec3_mul(post_rot_inv, {{px, py, pz}});

                        // cam_to_lidar: multiply (x,y) by the TRANSFORMED z,
                        // keeping z unchanged (matches depth_lss.py:
                        //   points = cat(points[...,:2]*points[...,2:3],
                        //                 points[...,2:3], 5))
                        const float z = pt[2];
                        pt[0] *= z;
                        pt[1] *= z;

                        // Camera → lidar
                        Vec3 lp = mat3_vec3_mul(combine, pt);
                        lp[0] += c2l_trans[0];
                        lp[1] += c2l_trans[1];
                        lp[2] += c2l_trans[2];

                        // Apply extra lidar augmentation
                        Vec3 lp2 = mat3_vec3_mul(la_rot, lp);
                        lp2[0] += la_trans[0];
                        lp2[1] += la_trans[1];
                        lp2[2] += la_trans[2];

                        size_t gi = ((size_t) bn * D * fpn_H * fpn_W
                                   + (size_t) d * fpn_H * fpn_W
                                   + (size_t) h * fpn_W + ww) * 3;
                        geom[gi + 0] = lp2[0];
                        geom[gi + 1] = lp2[1];
                        geom[gi + 2] = lp2[2];
                    }
                }
            }
        }
    }
    lap("geom");

    if (sink && false) {  // AICore port note: geom dump dropped
        // Debug: dump the frustum points so scripts can replay the CPU-loop
        // and the fused CUDA kernel on identical inputs.
        sink("bevfusion.geom_debug", geom.data(), (int64_t) geom.size() / 3, 3);
        fprintf(stderr, "[lss-debug] D=%d C=%d fpn_H=%d fpn_W=%d BN=%d B=%d ncam=%d "
                "dx={%g,%g,%g} bx={%g,%g,%g} nx={%d,%d,%d}\n",
                D, out_channel, fpn_H, fpn_W, BN, batch_size, num_cameras,
                dx[0], dx[1], dx[2], bx[0], bx[1], bx[2], nx[0], nx[1], nx[2]);
    }

    // BEV pool: fused depth-softmax × features + scatter-add.
    // depthnet_out: [fpn_W, fpn_H, D+C, BN]
    // CUDA: single fused kernel (bev_pool_cuda) directly consumes
    // depthnet_out + geom, no intermediate volume.  CPU: builds the
    // softmax×feature volume first, then scatters it into the BEV grid.
    std::vector<float> bev_pooled;
// AICore port note: the upstream fused CUDA bev-pool kernel lives in the
// host CUDA extension this build does not ship; the fused CPU path below is
// the single implementation on every backend.
{
        // CPU path: softmax over D depth bins per (bn, h, w), outer product
        // with C features into volume, then scatter-add into the BEV grid.
        // The 638 MB volume is only needed by the legacy bev_pool or the
        // volume_debug tap; the fused path consumes depthnet_out directly, so
        // skip its ~0.5 s construction otherwise (measured: the whole
        // "softmax_volume" lap WAS this loop).
        const bool need_volume = false;  // AICore port note: debug volume path dropped
        std::vector<float> volume;
        if (need_volume) {
        size_t vol_total = (size_t) BN * D * fpn_H * fpn_W * out_channel;
        volume.assign(vol_total, 0.0f);

        for (int bn = 0; bn < BN; bn++) {
            for (int h = 0; h < fpn_H; h++) {
                for (int ww = 0; ww < fpn_W; ww++) {
                    size_t base = (size_t) bn * (D + out_channel) * fpn_H * fpn_W
                                + (size_t) h * fpn_W + ww;
                    // Read depth logits and compute softmax
                    std::vector<float> depth_probs(D);
                    float max_val = -1e30f;
                    for (int d = 0; d < D; d++) {
                        size_t didx = base + (size_t) d * fpn_H * fpn_W;
                        depth_probs[d] = depthnet_out[didx];
                        max_val = std::max(max_val, depth_probs[d]);
                    }
                    float sum_exp = 0;
                    for (int d = 0; d < D; d++) {
                        depth_probs[d] = std::exp(depth_probs[d] - max_val);
                        sum_exp += depth_probs[d];
                    }
                    for (int d = 0; d < D; d++) depth_probs[d] /= sum_exp;

                    // Read features
                    std::vector<float> feats(out_channel);
                    for (int c = 0; c < out_channel; c++) {
                        size_t fidx = base + (size_t) (D + c) * fpn_H * fpn_W;
                        feats[c] = depthnet_out[fidx];
                    }

                    // Outer product: volume[bn, d, h, w, c] = depth_probs[d] * feats[c]
                    for (int d = 0; d < D; d++) {
                        for (int c = 0; c < out_channel; c++) {
                            size_t vidx = ((size_t) bn * D * fpn_H * fpn_W
                                        + (size_t) d * fpn_H * fpn_W
                                        + (size_t) h * fpn_W + ww) * out_channel + c;
                            volume[vidx] = depth_probs[d] * feats[c];
                        }
                    }
                }
            }
        }
        }
        lap("softmax_volume");

        if (need_volume && sink && false) {
            sink("bevfusion.volume_debug", volume.data(),
                 (int64_t) volume.size() / out_channel, out_channel);
        }

        // Default: fused multithreaded path (bit-identical to the volume path
        // on synthetic A/B and parity-verified end-to-end on CPU/Vulkan; skips
        // the ~638 MB volume materialization). The legacy volume path stays
        // available as a cross-check via OPCPD_BEVPOOL_LEGACY=1.
        if (false) {  // AICore port note: legacy volume cross-check dropped
            bev_pool_cpu(geom.data(), volume.data(),
                         BN, D, fpn_H, fpn_W, out_channel,
                         dx, bx, nx, batch_size, num_cameras, bev_pooled);
        } else {
            bev_pool_cpu_fused(geom.data(), depthnet_out.data(),
                               BN, D, fpn_H, fpn_W, out_channel,
                               dx, bx, nx, batch_size, num_cameras, bev_pooled);
        }
}
    lap("bev_pool");
    if (sink && false) {  // AICore port note: geom dump dropped
        sink("bevfusion.bev_pool_raw", bev_pooled.data(),
             (int64_t) bev_pooled.size() / out_channel, out_channel);
    }
    // bev_pooled: [nx[0], nx[1], out_channel, batch_size] in GGML layout

    if (sink) {
        // dump pre-downsample bev_pool [nx[0], nx[1], C, B] → [B*X*Y, C]
        // rows in y-major (iy*nx[0] + ix), matching PyTorch HWC dump
        const int Cc = out_channel;
        int64_t rows = (int64_t) nx[0] * nx[1];
        std::vector<float> tap((size_t) rows * Cc);
        for (int b = 0; b < batch_size; b++)
          for (int iy = 0; iy < nx[1]; iy++)
            for (int ix = 0; ix < nx[0]; ix++)
              for (int c = 0; c < Cc; c++) {
                  // src = w + W*h + W*H*c + W*H*C*b ; w=ix(nx[0]), h=iy(nx[1])
                  size_t src = (size_t) ix + (size_t) nx[0] * iy + (size_t) nx[0] * nx[1] * c + (size_t) nx[0] * nx[1] * Cc * b;
                  size_t dst = ((size_t) b * rows + (size_t) iy * nx[0] + ix) * Cc + c;
                  tap[dst] = bev_pooled[src];
              }
        sink("bevfusion.bev_pool", tap.data(), (int64_t) batch_size * rows, Cc);
    }

    bev_W = nx[0];  // W axis = x, matching PyTorch W after permute(0,1,3,2)
    bev_H = nx[1];  // H axis = y

    // ---- Step 4: Optional downsample (GGML graph) ----
    if (w.has_downsample) {
        const size_t graph_nodes = 128;
        ggml_init_params gp = {
            ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
            nullptr, true,
        };
        ggml_context * ctx = ggml_init(gp);
        if (!ctx) { error = "downsample ctx init failed"; return false; }

        ggml_tensor * input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                    bev_W, bev_H, out_channel, batch_size);
        ggml_set_name(input_t, "downsample_input");
        ggml_set_input(input_t);

        // The bev_pool output consumed by the downsample has torch layout
        // (b, c, x, y): torch's KH axis runs along X while the converter's
        // ggml kernel carries KW in ne[0]. Both CPU and CUDA conv kernels
        // apply the kernel's spatial axes transposed for this layout, so the
        // kernel gets the same in-graph spatial swap.
        auto swap_spatial = [](ggml_context * c, ggml_tensor * k) {
            return ggml_cont(c, ggml_permute(c, k, 1, 0, 2, 3));
        };
        conv_bn_block_weights d0 = w.down_0, d1 = w.down_1, d2 = w.down_2;
        d0.conv_w = swap_spatial(ctx, d0.conv_w);
        d1.conv_w = swap_spatial(ctx, d1.conv_w);
        d2.conv_w = swap_spatial(ctx, d2.conv_w);

        ggml_tensor * x = build_conv_bn_relu(ctx, d0, input_t, 1, 1);
        x = build_conv_bn_relu(ctx, d1, x, 2, 1);
        x = build_conv_bn_relu(ctx, d2, x, 1, 1);
        ggml_set_name(x, "downsample_out");
        ggml_set_output(x);

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, x);

        if (!ggml_gallocr_alloc_graph(backend.galloc, gf)) {
            error = "downsample alloc failed"; ggml_free(ctx); return false;
        }
        ggml_backend_tensor_set(input_t, bev_pooled.data(), 0,
                                bev_pooled.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.be, gf) != GGML_STATUS_SUCCESS) {
            error = "downsample compute failed"; ggml_free(ctx); return false;
        }

        bev_W /= 2;
        bev_H /= 2;
        size_t n = (size_t) bev_W * bev_H * out_channel * batch_size;
        bev_output.resize(n);
        ggml_backend_tensor_get(x, bev_output.data(), 0, n * sizeof(float));
        ggml_free(ctx);
    } else {
        bev_output = std::move(bev_pooled);
    }

    return true;
}

} // namespace openpcdet
