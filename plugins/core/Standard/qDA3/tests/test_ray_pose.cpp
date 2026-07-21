// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// T3 gate: C++ ray->pose solver vs the seeded reference dump.
// PRIMARY (RNG-free): feed the dumped ray field + rand_sample_iters_idx +
// sorted_idx
// + the captured consensus refit_idx; assert C++ extrinsics (rotation)
// max|d|<1e-3 and intrinsics rel-err<1e-3. The fed refit_idx removes the
// reference's tail randperm subsample so only SVD/QR algorithm (f32-torch vs
// f64-host) differences remain. Also proves the gate DISCRIMINATES: perturbing
// the ray field moves the pose past tol.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "parity.hpp"
#include "ray_pose.hpp"

using aicore::depth::RayPoseOut;
using aicore::depth::RayPoseParams;

static const int ROT_IDX[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};

int main() {
    const char* dump = std::getenv("DA_TEST_BASELINE_RAY_POSE");
    if (!dump) return 77;
    std::string path = dump;
    // The dump is a gitignored dev artifact (scripts/dump_ray_pose.py); SKIP if
    // absent.
    if (!std::ifstream(path).good()) {
        std::fprintf(stderr,
                     "ray_pose dump not found at %s -> SKIP (run "
                     "scripts/dump_ray_pose.py)\n",
                     path.c_str());
        return 77;
    }

    std::vector<float> ray, conf, src_ref, dst_ref, w_ref, ext_ref, intr_ref;
    std::vector<int64_t> sh;
    std::vector<int32_t> rand_idx, sorted_idx, refit_idx;
    bool ok = true;
    ok &= da_parity::load_baseline(path, "ray", ray, sh);
    ok &= da_parity::load_baseline(path, "ray_conf", conf, sh);
    ok &= da_parity::load_baseline(path, "src_full", src_ref, sh);
    ok &= da_parity::load_baseline(path, "dst_full", dst_ref, sh);
    ok &= da_parity::load_baseline(path, "w_full", w_ref, sh);
    ok &= da_parity::load_baseline(path, "extrinsics", ext_ref, sh);
    ok &= da_parity::load_baseline(path, "intrinsics", intr_ref, sh);
    ok &= da_parity::load_baseline_i32(path, "rand_sample_iters_idx", rand_idx);
    ok &= da_parity::load_baseline_i32(path, "sorted_idx", sorted_idx);
    ok &= da_parity::load_baseline_i32(path, "refit_idx", refit_idx);
    if (!ok) {
        std::fprintf(stderr, "failed to load dump tensors\n");
        return 1;
    }

    int Hy = (int)da_parity::load_kv_u32(path, "ray_pose.Hy");
    int Wx = (int)da_parity::load_kv_u32(path, "ray_pose.Wx");
    int img_H = (int)da_parity::load_kv_u32(path, "ray_pose.img_h");
    int img_W = (int)da_parity::load_kv_u32(path, "ray_pose.img_w");
    RayPoseParams p;
    p.n_iter = (int)da_parity::load_kv_u32(path, "ray_pose.n_iter");
    p.num_sample = (int)da_parity::load_kv_u32(path, "ray_pose.num_sample");
    const int N = Hy * Wx;
    std::fprintf(
            stderr,
            "[ray_pose] Hy=%d Wx=%d N=%d img=%dx%d n_iter=%d num_sample=%d "
            "n_refit=%zu\n",
            Hy, Wx, N, img_H, img_W, p.n_iter, p.num_sample, refit_idx.size());

    // ---- (1) cloud construction parity ----
    std::vector<double> src, dst, w, oh, cr;
    aicore::depth::build_ray_cloud(ray.data(), conf.data(), Hy, Wx,
                                   p.z_threshold, src, dst, w, oh, cr);
    {
        double m_src = 0, m_dst = 0, m_w = 0;
        for (int i = 0; i < 2 * N; ++i) {
            m_src = std::max(m_src, std::fabs(src[i] - (double)src_ref[i]));
            m_dst = std::max(m_dst, std::fabs(dst[i] - (double)dst_ref[i]));
        }
        for (int i = 0; i < N; ++i)
            m_w = std::max(m_w, std::fabs(w[i] - (double)w_ref[i]));
        std::fprintf(stderr, "[cloud] max|d| src=%.3e dst=%.3e w=%.3e\n", m_src,
                     m_dst, m_w);
        if (m_src > 1e-5 || m_dst > 1e-4 || m_w > 1e-5) {
            std::fprintf(stderr, "[cloud] FAIL (grid/z-norm parity)\n");
            return 1;
        }
    }

    // ---- (2) PRIMARY gated pose parity ----
    RayPoseOut o;
    if (!aicore::depth::solve_ray_pose(ray.data(), conf.data(), Hy, Wx, img_H,
                                       img_W, rand_idx.data(),
                                       sorted_idx.data(), refit_idx.data(),
                                       (int)refit_idx.size(), p, o)) {
        std::fprintf(stderr, "solve_ray_pose (gated) failed\n");
        return 1;
    }
    std::fprintf(stderr,
                 "[solve] C++ best-hypothesis inliers=%d (ref consensus=%zu)\n",
                 o.n_inlier_best, refit_idx.size());

    double rot_max = 0.0;
    for (int k = 0; k < 9; ++k)
        rot_max = std::max(rot_max, std::fabs((double)o.extrinsics[ROT_IDX[k]] -
                                              (double)ext_ref[ROT_IDX[k]]));
    double tr_max = 0.0;
    for (int k : {3, 7, 11})
        tr_max = std::max(tr_max, std::fabs((double)o.extrinsics[k] -
                                            (double)ext_ref[k]));
    // intrinsics rel-err on the 4 active entries (fx,fy,cx,cy)
    double intr_rel = 0.0;
    for (int k : {0, 4, 2, 5}) {
        double ref = std::fabs((double)intr_ref[k]);
        double d = std::fabs((double)o.intrinsics[k] - (double)intr_ref[k]);
        intr_rel = std::max(intr_rel, ref > 1e-9 ? d / ref : d);
    }
    std::fprintf(stderr,
                 "[pose] rotation max|d|=%.3e  translation max|d|=%.3e  "
                 "intrinsics rel=%.3e\n",
                 rot_max, tr_max, intr_rel);
    std::fprintf(
            stderr,
            "       C++ K=[%.4f %.4f %.4f %.4f]  ref K=[%.4f %.4f %.4f %.4f]\n",
            o.intrinsics[0], o.intrinsics[4], o.intrinsics[2], o.intrinsics[5],
            intr_ref[0], intr_ref[4], intr_ref[2], intr_ref[5]);

    bool gate = (rot_max < 1e-3) && (tr_max < 1e-3) && (intr_rel < 1e-3);
    std::fprintf(stderr, "[gate] PRIMARY pose parity -> %s\n",
                 gate ? "PASS" : "FAIL");
    if (!gate) return 1;

    // ---- (3) discrimination: perturb the ray field, pose must move past tol
    // ----
    {
        std::vector<float> ray2 = ray;
        // perturb the direction channels in a 16x16 block (well within
        // consensus)
        for (int i = 20; i < 36; ++i)
            for (int j = 20; j < 36; ++j) {
                int n = i * Wx + j;
                ray2[(size_t)n * 6 + 0] += 0.15f;
                ray2[(size_t)n * 6 + 1] -= 0.15f;
            }
        RayPoseOut o2;
        if (!aicore::depth::solve_ray_pose(ray2.data(), conf.data(), Hy, Wx,
                                           img_H, img_W, rand_idx.data(),
                                           sorted_idx.data(), refit_idx.data(),
                                           (int)refit_idx.size(), p, o2)) {
            std::fprintf(stderr, "perturbed solve failed\n");
            return 1;
        }
        double drot = 0.0, dintr = 0.0;
        for (int k = 0; k < 9; ++k)
            drot = std::max(drot, std::fabs((double)o2.extrinsics[ROT_IDX[k]] -
                                            (double)o.extrinsics[ROT_IDX[k]]));
        for (int k : {0, 4, 2, 5})
            dintr = std::max(dintr, std::fabs((double)o2.intrinsics[k] -
                                              (double)o.intrinsics[k]));
        std::fprintf(stderr,
                     "[discriminate] perturbed vs clean: rotation max|d|=%.3e "
                     "intrinsics max|d|=%.3e\n",
                     drot, dintr);
        if (!(drot > 1e-3 || dintr > 1e-1)) {
            std::fprintf(stderr,
                         "[discriminate] FAIL: gate does not respond to input "
                         "perturbation\n");
            return 1;
        }
        std::fprintf(stderr, "[discriminate] PASS\n");
    }

    std::fprintf(stderr, "test_ray_pose: all OK\n");
    return 0;
}
