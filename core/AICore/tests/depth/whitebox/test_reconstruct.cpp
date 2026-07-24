// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "reconstruct.hpp"

static bool approx(double a, double b, double eps) {
    return std::fabs(a - b) <= eps;
}

int main() {
    bool ok = true;

    // ---- Test 1: identity back-projection ----
    {
        const int H = 2, W = 2, N = 1;
        std::vector<float> depth(N * H * W, 2.0f);
        std::vector<float> conf(N * H * W, 2.0f);
        std::array<float, 9> K = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // identity
        std::array<float, 16> ext = {1, 0, 0, 0, 0, 1, 0, 0,
                                     0, 0, 1, 0, 0, 0, 0, 1};  // identity
        std::vector<uint8_t> img(H * W * 3);
        for (int i = 0; i < H * W; ++i) {
            img[3 * i + 0] = 10;
            img[3 * i + 1] = 20;
            img[3 * i + 2] = 30;
        }
        std::vector<const uint8_t*> images = {img.data()};
        std::vector<std::array<float, 9>> Ks = {K};
        std::vector<std::array<float, 16>> exts = {ext};

        aicore::depth::WorldPoints wp = aicore::depth::back_project(
                depth, conf, Ks, exts, images, H, W, N, 1.0f);

        if (wp.xyz.size() != 4u * 3) {
            printf("T1 FAIL: expected 4 points, got %zu\n", wp.xyz.size() / 3);
            ok = false;
        }
        // pixel order: v outer, u inner. For pixel (u,v): world == (2u, 2v, 2)
        int idx = 0;
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                double x = wp.xyz[3 * idx + 0], y = wp.xyz[3 * idx + 1],
                       z = wp.xyz[3 * idx + 2];
                if (!approx(x, 2.0 * u, 1e-5) || !approx(y, 2.0 * v, 1e-5) ||
                    !approx(z, 2.0, 1e-5)) {
                    printf("T1 FAIL: pixel (u=%d,v=%d) world=(%f,%f,%f)\n", u,
                           v, x, y, z);
                    ok = false;
                }
                if (wp.rgb[3 * idx + 0] != 10 || wp.rgb[3 * idx + 1] != 20 ||
                    wp.rgb[3 * idx + 2] != 30) {
                    printf("T1 FAIL: rgb mismatch at idx %d\n", idx);
                    ok = false;
                }
                if (wp.frame[idx] != 0 || wp.u[idx] != u || wp.v[idx] != v) {
                    printf("T1 FAIL: provenance idx %d frame=%d u=%d v=%d\n",
                           idx, wp.frame[idx], wp.u[idx], wp.v[idx]);
                    ok = false;
                }
                ++idx;
            }
        }
    }

    // ---- Test 2: percentile_linear ----
    {
        std::vector<float> v = {1, 2, 3, 4};
        double p50 = aicore::depth::percentile_linear(v, 50.0);
        double p40 = aicore::depth::percentile_linear(v, 40.0);
        if (!approx(p50, 2.5, 1e-9)) {
            printf("T2 FAIL: p50=%f expected 2.5\n", p50);
            ok = false;
        }
        if (!approx(p40, 2.2, 1e-9)) {
            printf("T2 FAIL: p40=%f expected 2.2\n", p40);
            ok = false;
        }
        // ensure input not mutated
        if (!(v[0] == 1 && v[1] == 2 && v[2] == 3 && v[3] == 4)) {
            printf("T2 FAIL: input mutated\n");
            ok = false;
        }
    }

    // ---- Test 3: rotmat2qvec round-trip ----
    {
        // 90 deg about Z: R = [[0,-1,0],[1,0,0],[0,0,1]]  (row-major)
        std::array<float, 9> R = {0, -1, 0, 1, 0, 0, 0, 0, 1};
        std::array<float, 4> q = aicore::depth::rotmat2qvec(R);
        if (q[0] < 0) {
            printf("T3 FAIL: qw<0 (%f)\n", q[0]);
            ok = false;
        }
        // reconstruct R from quaternion (qw,qx,qy,qz), per qvec2rotmat
        double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
        std::array<double, 9> Rr = {
                1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz,
                2 * qz * qx + 2 * qw * qy,     2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx,
                2 * qz * qx - 2 * qw * qy,     2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy};
        for (int i = 0; i < 9; ++i) {
            if (!approx(Rr[i], R[i], 1e-6)) {
                printf("T3 FAIL: R[%d] got %f expected %f\n", i, Rr[i],
                       (double)R[i]);
                ok = false;
            }
        }
    }

    printf("%s\n", ok ? "ALL PASS" : "FAILURES");
    return ok ? 0 : 1;
}
