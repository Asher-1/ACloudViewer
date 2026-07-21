// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Asset-free golden tests for the C++ pose component (src/pose.{h,cpp}) — the
// direct mirror of pose/test_pose.py. No model, no fixtures, no cv2: synthetic
// geometry with KNOWN ground truth, the way CLAUDE.md wants new ops verified
// ("pinned to hand-computed references... a wrong op should fail with zero
// fixtures"). Runs in the fast tier (ctest -LE model).
//
// The RNG differs from numpy's (the checks are tolerance-based on recovered
// quantities, not bitstream-matched), so this validates the C++ port to the
// same correctness bar the Python prototype meets, independently of numpy.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "pose.h"
#include "splat.h"

using namespace aicore::gaussian::pose;
using fsla::Mat3;
using fsla::Mat4;
using fsla::Vec3;

static int failures = 0;
static std::mt19937_64 RNG(12345);

static void check(const char* name, bool cond, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", name,
                detail[0] ? "  " : "", detail);
    if (!cond) failures++;
}

// --- helpers ---------------------------------------------------------------

static Mat3 rand_rotation() {
    std::normal_distribution<double> g(0, 1);
    Vec3 ax = {g(RNG), g(RNG), g(RNG)};
    double n = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
    ax = {ax[0] / n, ax[1] / n, ax[2] / n};
    std::uniform_real_distribution<double> a(0.2, M_PI - 0.2);
    const double ang = a(RNG), c = std::cos(ang), s = std::sin(ang), v = 1 - c;
    const double x = ax[0], y = ax[1], z = ax[2];
    return Mat3{{c + x * x * v, x * y * v - z * s, x * z * v + y * s,
                 y * x * v + z * s, c + y * y * v, y * z * v - x * s,
                 z * x * v - y * s, z * y * v + x * s, c + z * z * v}};
}

static double rot_angle(const Mat3& R) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    return std::acos(std::max(-1.0, std::min(1.0, (tr - 1) / 2))) * 180.0 /
           M_PI;
}

static Mat3 mul(const Mat3& A, const Mat3& B) { return fsla::mat3_mul(A, B); }
static Mat3 T(const Mat3& A) { return fsla::mat3_transpose(A); }

// apply (s,R,t) to a row-major N*3 array -> out
static void apply_sim_arr(double s,
                          const Mat3& R,
                          const Vec3& t,
                          const double* X,
                          int N,
                          double* out) {
    for (int i = 0; i < N; i++) {
        Vec3 x = {X[3 * i], X[3 * i + 1], X[3 * i + 2]};
        Vec3 Rx = fsla::mat3_apply(R, x);
        out[3 * i] = s * Rx[0] + t[0];
        out[3 * i + 1] = s * Rx[1] + t[1];
        out[3 * i + 2] = s * Rx[2] + t[2];
    }
}

static double rms3(const double* a, const double* b, int N) {
    double s = 0;
    for (int i = 0; i < 3 * N; i++) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s / N);
}

static Vec3 apply_mat4(const Mat4& M, const Vec3& p) {
    return {M(0, 0) * p[0] + M(0, 1) * p[1] + M(0, 2) * p[2] + M(0, 3),
            M(1, 0) * p[0] + M(1, 1) * p[1] + M(1, 2) * p[2] + M(1, 3),
            M(2, 0) * p[0] + M(2, 1) * p[1] + M(2, 2) * p[2] + M(2, 3)};
}

// ===========================================================================
// Coordinate-system normalization (the scale question)
// ===========================================================================

static void test_similarity_roundtrip() {
    std::printf("test_similarity_roundtrip\n");
    const int N = 200;
    std::normal_distribution<double> g(0, 1);
    std::vector<double> X(3 * N), Y(3 * N);
    for (int i = 0; i < N; i++) {
        X[3 * i] = g(RNG) * 3;
        X[3 * i + 1] = g(RNG) * 2;
        X[3 * i + 2] = g(RNG) * 5;
    }
    const double s_t = 1.7;
    Mat3 R_t = rand_rotation();
    Vec3 t_t = {4.0, -1.0, 2.0};
    apply_sim_arr(s_t, R_t, t_t, X.data(), N, Y.data());

    Sim3 T_ = fit_similarity(X.data(), Y.data(), N);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.12f vs %g", T_.s, s_t);
    check("scale", std::fabs(T_.s - s_t) < 1e-9, buf);
    check("rotation", rot_angle(mul(T_.R, T(R_t))) < 1e-7);
    check("translation",
          std::sqrt((T_.t[0] - t_t[0]) * (T_.t[0] - t_t[0]) +
                    (T_.t[1] - t_t[1]) * (T_.t[1] - t_t[1]) +
                    (T_.t[2] - t_t[2]) * (T_.t[2] - t_t[2])) < 1e-7);
    std::vector<double> Yp(3 * N);
    apply_sim_arr(T_.s, T_.R, T_.t, X.data(), N, Yp.data());
    check("residual~0", rms3(Yp.data(), Y.data(), N) < 1e-9);
}

static void test_scale_detection() {
    std::printf("test_scale_detection\n");
    const int N = 300;
    std::normal_distribution<double> g(0, 1);
    std::vector<double> X(3 * N), Y(3 * N);
    for (int i = 0; i < 3 * N; i++) X[i] = g(RNG) * 2;
    const double s_t = 2.5;
    apply_sim_arr(s_t, rand_rotation(), {1.0, 2.0, -3.0}, X.data(), N,
                  Y.data());
    Ladder L = diagnose(X.data(), Y.data(), N);
    check("recovered scale", std::fabs(L.scale - s_t) < 1e-8);
    check("rigid residual large", L.rigid / L.scene > 0.1);
    check("similarity residual ~0", L.similarity / L.scene < 1e-9);
    check("verdict needs_scale", L.verdict == "needs_scale", L.verdict.c_str());
}

static void test_nonlinear_detection() {
    std::printf("test_nonlinear_detection\n");
    const int N = 400;
    std::normal_distribution<double> g(0, 1);
    std::vector<double> X(3 * N);
    for (int i = 0; i < N; i++) {
        X[3 * i] = g(RNG) * 2;
        X[3 * i + 1] = g(RNG) * 2;
        X[3 * i + 2] = g(RNG) * 2 + 6.0;
    }

    // (a) anisotropic (affine) warp
    Mat3 A = mul(Mat3{{1.0, 0, 0, 0, 1.6, 0, 0, 0, 0.7}}, rand_rotation());
    std::vector<double> Yaff(3 * N);
    for (int i = 0; i < N; i++) {
        Vec3 x = {X[3 * i], X[3 * i + 1], X[3 * i + 2]};
        Vec3 Ax = fsla::mat3_apply(A, x);
        Yaff[3 * i] = Ax[0] + 1.0;
        Yaff[3 * i + 1] = Ax[1];
        Yaff[3 * i + 2] = Ax[2] + 2.0;
    }
    Ladder La = diagnose(X.data(), Yaff.data(), N);
    check("affine: similarity fails", La.similarity / La.scene > 1e-3);
    check("affine: affine fits", La.affine / La.scene < 1e-9);
    check("affine: verdict needs_affine", La.verdict == "needs_affine",
          La.verdict.c_str());

    // (b) depth-quadratic (focal-error fingerprint)
    double zmean = 0;
    for (int i = 0; i < N; i++) zmean += X[3 * i + 2];
    zmean /= N;
    double zvar = 0;
    for (int i = 0; i < N; i++)
        zvar += (X[3 * i + 2] - zmean) * (X[3 * i + 2] - zmean);
    zvar /= N;
    std::vector<double> Ynl(X);
    for (int i = 0; i < N; i++) {
        const double Z = X[3 * i + 2];
        Ynl[3 * i] *= (1.0 + 0.15 * (Z - zmean) * (Z - zmean) / zvar);
    }
    Ladder Ln = diagnose(X.data(), Ynl.data(), N);
    check("nonlinear: affine fails", Ln.affine / Ln.scene > 1e-3);
    check("nonlinear: verdict nonlinear", Ln.verdict == "nonlinear",
          Ln.verdict.c_str());
    check("nonlinear: depth-correlated residual", Ln.structured);
}

static void test_outlier_robustness() {
    std::printf("test_outlier_robustness\n");
    const int N = 400;
    std::normal_distribution<double> g(0, 1);
    std::vector<double> X(3 * N), Y(3 * N);
    for (int i = 0; i < 3 * N; i++) X[i] = g(RNG) * 2;
    const double s_t = 1.3;
    Mat3 R_t = rand_rotation();
    Vec3 t_t = {0.5, -2.0, 1.0};
    apply_sim_arr(s_t, R_t, t_t, X.data(), N, Y.data());
    const int n_out = (int)(0.3 * N);
    std::vector<int> oidx(N);
    for (int i = 0; i < N; i++) oidx[i] = i;
    std::shuffle(oidx.begin(), oidx.end(), RNG);
    for (int k = 0; k < n_out; k++) {
        int i = oidx[k];
        Y[3 * i] += g(RNG) * 10;
        Y[3 * i + 1] += g(RNG) * 10;
        Y[3 * i + 2] += g(RNG) * 10;
    }
    std::vector<char> inl;
    Sim3 T_ = fit_similarity_ransac(X.data(), Y.data(), N, 0.05, 300, inl);
    check("scale recovered", std::fabs(T_.s - s_t) < 1e-3);
    check("rotation recovered", rot_angle(mul(T_.R, T(R_t))) < 0.1);
    check("translation recovered",
          std::sqrt((T_.t[0] - t_t[0]) * (T_.t[0] - t_t[0]) +
                    (T_.t[1] - t_t[1]) * (T_.t[1] - t_t[1]) +
                    (T_.t[2] - t_t[2]) * (T_.t[2] - t_t[2])) < 1e-2);
    int rejected = 0;
    for (int k = 0; k < n_out; k++)
        if (!inl[oidx[k]]) rejected++;
    check("outliers excluded", rejected > 0.9 * n_out);
}

static void test_loop_correction() {
    std::printf("test_loop_correction\n");
    Mat3 R = rand_rotation();
    Mat4 M = sim_matrix({1.07, R, {0.3, -0.2, 0.1}});
    Mat4 m0 = sim_frac_power(M, 0.0), m1 = sim_frac_power(M, 1.0),
         mh = sim_frac_power(M, 0.5);
    bool id_ok = true, m1_ok = true;
    for (int i = 0; i < 16; i++) {
        id_ok &= std::fabs(m0.a[i] - fsla::mat4_identity().a[i]) < 1e-9;
        m1_ok &= std::fabs(m1.a[i] - M.a[i]) < 1e-9;
    }
    Mat4 mh2 = fsla::mat4_mul(mh, mh);
    bool half_ok = true;
    for (int i = 0; i < 16; i++) half_ok &= std::fabs(mh2.a[i] - M.a[i]) < 1e-9;
    check("M^0 == I", id_ok);
    check("M^1 == M", m1_ok);
    check("(M^.5)^2 == M", half_ok);

    const int n = 12;
    Mat4 D = sim_matrix({1.15, R, {0.4, -0.2, 0.1}});
    std::vector<Vec3> clean(n + 1), drift(n + 1), corr(n + 1);
    for (int k = 0; k <= n; k++) {
        const double tt = 2 * M_PI * k / n;
        clean[k] = {std::cos(tt), std::sin(tt), 0.1 * tt};
        drift[k] = apply_mat4(sim_frac_power(D, -(double)k / n),
                              clean[k]);  // D^{-k/n}
        corr[k] = apply_mat4(sim_frac_power(D, (double)k / n),
                             drift[k]);  // D^{ k/n}
    }
    double pre = 0, post = 0;
    for (int k = 0; k <= n; k++) {
        for (int c = 0; c < 3; c++) {
            pre += (drift[k][c] - clean[k][c]) * (drift[k][c] - clean[k][c]);
            post += (corr[k][c] - clean[k][c]) * (corr[k][c] - clean[k][c]);
        }
    }
    check("drift present before", std::sqrt(pre / (n + 1)) > 0.1);
    char buf[64];
    std::snprintf(buf, sizeof buf, "ATE %.1e", std::sqrt(post / (n + 1)));
    check("recovers clean loop", std::sqrt(post / (n + 1)) < 1e-9, buf);
}

static void test_loop_closure() {
    std::printf("test_loop_closure\n");
    std::uniform_real_distribution<double> us(0.8, 1.2);
    std::normal_distribution<double> g(0, 1);
    std::vector<Sim3> links;
    for (int i = 0; i < 4; i++)
        links.push_back({us(RNG), rand_rotation(), {g(RNG), g(RNG), g(RNG)}});
    Sim3 acc = sim_identity();
    for (int i = 0; i < 3; i++) acc = sim_compose(links[i], acc);
    std::vector<Sim3> closed = {links[0], links[1], links[2], sim_invert(acc)};
    LoopError e = loop_closure_error(closed);
    // rot_deg has a ~sqrt(eps) (~1e-6 deg) floor: it is acos(~1) of RᵀR, whose
    // deviation from I is at the fp roundoff level — scale_err and trans are
    // exactly 0 here, so 1e-5 deg cleanly separates "closed" from any real
    // drift.
    check("closed loop ~ identity",
          e.scale_err < 1e-9 && e.rot_deg < 1e-5 && e.trans < 1e-9);
    std::vector<Sim3> bad = closed;
    bad[0].s *= 1.12;
    LoopError e2 = loop_closure_error(bad);
    check("scale drift detected",
          std::fabs(e2.scale_err - std::log(1.12)) < 1e-9);
}

// ===========================================================================
// PnP (verified against analytic ground truth)
// ===========================================================================

static Mat3 make_K(double f, double W, double H) {
    Mat3 K = fsla::mat3_identity();
    K(0, 0) = f;
    K(1, 1) = f;
    K(0, 2) = W / 2;
    K(1, 2) = H / 2;
    return K;
}

// project world points by (R,t) world2cam through K; fill px (N*2) and z (N)
static void project(const double* Pw,
                    int N,
                    const Mat3& R,
                    const Vec3& t,
                    const Mat3& K,
                    double* px,
                    double* z) {
    for (int i = 0; i < N; i++) {
        Vec3 p = {Pw[3 * i], Pw[3 * i + 1], Pw[3 * i + 2]};
        Vec3 Xc = fsla::mat3_apply(R, p);
        Xc[0] += t[0];
        Xc[1] += t[1];
        Xc[2] += t[2];
        px[2 * i] = K(0, 0) * Xc[0] / Xc[2] + K(0, 2);
        px[2 * i + 1] = K(1, 1) * Xc[1] / Xc[2] + K(1, 2);
        z[i] = Xc[2];
    }
}

static void test_focal_recovery() {
    std::printf("test_focal_recovery\n");
    const int N = 500;
    const double f = 437.0, W = 512, H = 512;
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2), uz(4, 9);
    std::vector<double> Xc(3 * N), px(2 * N);
    for (int i = 0; i < N; i++) {
        Xc[3 * i] = ux(RNG);
        Xc[3 * i + 1] = uy(RNG);
        Xc[3 * i + 2] = uz(RNG);
        px[2 * i] = f * Xc[3 * i] / Xc[3 * i + 2] + W / 2;
        px[2 * i + 1] = f * Xc[3 * i + 1] / Xc[3 * i + 2] + H / 2;
    }
    double fe = estimate_focal(Xc.data(), px.data(), N, W / 2, H / 2);
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9f vs %g", fe, f);
    check("exact focal", std::fabs(fe - f) < 1e-6, buf);
    std::normal_distribution<double> g(0, 0.5);
    std::vector<double> pn(px);
    for (int i = 0; i < 2 * N; i++) pn[i] += g(RNG);
    double fn = estimate_focal(Xc.data(), pn.data(), N, W / 2, H / 2);
    check("focal under noise", std::fabs(fn - f) / f < 0.02);
}

static void test_pnp_recovery() {
    std::printf("test_pnp_recovery\n");
    const int N = 500;
    Mat3 K = make_K(400, 512, 512);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2), uz(4, 8);
    std::vector<double> Pw(3 * N);
    for (int i = 0; i < N; i++) {
        Pw[3 * i] = ux(RNG);
        Pw[3 * i + 1] = uy(RNG);
        Pw[3 * i + 2] = uz(RNG);
    }
    std::normal_distribution<double> g(0, 1);
    std::uniform_real_distribution<double> ut(-0.6, 0.6), ua(3, 20);
    double worst_R = 0, worst_t = 0;
    for (int trial = 0; trial < 8; trial++) {
        Vec3 ax = {g(RNG), g(RNG), g(RNG)};
        double n = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
        ax = {ax[0] / n, ax[1] / n, ax[2] / n};
        const double ang = ua(RNG) * M_PI / 180.0, c = std::cos(ang),
                     s = std::sin(ang), v = 1 - c;
        Mat3 R{{c + ax[0] * ax[0] * v, ax[0] * ax[1] * v - ax[2] * s,
                ax[0] * ax[2] * v + ax[1] * s, ax[1] * ax[0] * v + ax[2] * s,
                c + ax[1] * ax[1] * v, ax[1] * ax[2] * v - ax[0] * s,
                ax[2] * ax[0] * v - ax[1] * s, ax[2] * ax[1] * v + ax[0] * s,
                c + ax[2] * ax[2] * v}};
        Vec3 t = {ut(RNG), ut(RNG), ut(RNG)};
        std::vector<double> px(2 * N), z(N);
        project(Pw.data(), N, R, t, K, px.data(), z.data());
        bool behind = false;
        for (int i = 0; i < N; i++)
            if (z[i] <= 0) behind = true;
        if (behind) continue;
        std::vector<char> inl;
        Mat4 c2w = solve_pnp_numpy(Pw.data(), px.data(), N, K, inl);
        Mat4 w2c = fsla::inv_rigid4(c2w);
        Mat3 Rr{};
        Vec3 tr{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) Rr(i, j) = w2c(i, j);
            tr[i] = w2c(i, 3);
        }
        worst_R = std::max(worst_R, (double)rot_angle(mul(Rr, T(R))));
        worst_t = std::max(worst_t, std::sqrt((tr[0] - t[0]) * (tr[0] - t[0]) +
                                              (tr[1] - t[1]) * (tr[1] - t[1]) +
                                              (tr[2] - t[2]) * (tr[2] - t[2])));
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "worst %.2e deg", worst_R);
    check("rotation recovered", worst_R < 1e-3, buf);
    check("translation recovered", worst_t < 1e-4);
}

static void test_pnp_outliers() {
    std::printf("test_pnp_outliers\n");
    const int N = 600;
    Mat3 K = make_K(400, 512, 512);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2), uz(4, 8);
    std::vector<double> Pw(3 * N);
    for (int i = 0; i < N; i++) {
        Pw[3 * i] = ux(RNG);
        Pw[3 * i + 1] = uy(RNG);
        Pw[3 * i + 2] = uz(RNG);
    }
    Mat3 R = rand_rotation();
    // shrink towards identity so all points stay in front
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R(i, j) = (i == j ? 1.0 : 0.0) * 0.85 + R(i, j) * 0.15;
    {
        fsla::Mat3 U, V;
        Vec3 s;
        fsla::svd3(R, U, s, V);
        R = mul(U, T(V));
        if (fsla::det3(R) < 0)
            for (int i = 0; i < 3; i++) R(i, 2) = -R(i, 2);
    }
    Vec3 t = {0.3, -0.2, 0.4};
    std::vector<double> px(2 * N), z(N);
    project(Pw.data(), N, R, t, K, px.data(), z.data());
    // keep only points in front
    std::vector<double> Pk, pk;
    for (int i = 0; i < N; i++)
        if (z[i] > 0) {
            for (int c = 0; c < 3; c++) Pk.push_back(Pw[3 * i + c]);
            pk.push_back(px[2 * i]);
            pk.push_back(px[2 * i + 1]);
        }
    const int M = (int)pk.size() / 2;
    const int n_out = (int)(0.25 * M);
    std::vector<int> oidx(M);
    for (int i = 0; i < M; i++) oidx[i] = i;
    std::shuffle(oidx.begin(), oidx.end(), RNG);
    std::uniform_real_distribution<double> corrupt(-150, 150);
    for (int k = 0; k < n_out; k++) {
        int i = oidx[k];
        pk[2 * i] += corrupt(RNG);
        pk[2 * i + 1] += corrupt(RNG);
    }
    std::vector<char> inl;
    Mat4 c2w = solve_pnp_numpy(Pk.data(), pk.data(), M, K, inl, 2.0, 300);
    Mat4 w2c = fsla::inv_rigid4(c2w);
    Mat3 Rr{};
    Vec3 tr{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) Rr(i, j) = w2c(i, j);
        tr[i] = w2c(i, 3);
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3f deg",
                  (double)rot_angle(mul(Rr, T(R))));
    check("rotation under outliers", rot_angle(mul(Rr, T(R))) < 0.2, buf);
    check("translation under outliers",
          std::sqrt((tr[0] - t[0]) * (tr[0] - t[0]) +
                    (tr[1] - t[1]) * (tr[1] - t[1]) +
                    (tr[2] - t[2]) * (tr[2] - t[2])) < 0.02);
    int rejected = 0;
    for (int k = 0; k < n_out; k++)
        if (!inl[oidx[k]]) rejected++;
    check("outliers rejected", rejected > 0.85 * n_out);
}

// ===========================================================================
// Robust PnP: EPnP + Gauss-Newton (the shipped real-data solver)
// ===========================================================================

static void test_pnp_robust_recovery() {
    std::printf("test_pnp_robust_recovery\n");
    const int N = 500;
    Mat3 K = make_K(400, 512, 512);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2), uz(4, 8);
    std::vector<double> Pw(3 * N);
    for (int i = 0; i < N; i++) {
        Pw[3 * i] = ux(RNG);
        Pw[3 * i + 1] = uy(RNG);
        Pw[3 * i + 2] = uz(RNG);
    }
    std::normal_distribution<double> g(0, 1);
    std::uniform_real_distribution<double> ut(-0.6, 0.6);
    double worst_R = 0, worst_t = 0;
    for (int trial = 0; trial < 8; trial++) {
        Mat3 R = rand_rotation();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R(i, j) = (i == j ? 1.0 : 0.0) * 0.8 + R(i, j) * 0.2;
        {
            fsla::Mat3 U, V;
            Vec3 s;
            fsla::svd3(R, U, s, V);
            R = mul(U, T(V));
            if (fsla::det3(R) < 0)
                for (int i = 0; i < 3; i++) R(i, 2) = -R(i, 2);
        }
        Vec3 t = {ut(RNG), ut(RNG), ut(RNG)};
        std::vector<double> px(2 * N), z(N);
        project(Pw.data(), N, R, t, K, px.data(), z.data());
        bool behind = false;
        for (int i = 0; i < N; i++)
            if (z[i] <= 0) behind = true;
        if (behind) continue;
        std::vector<char> inl;
        Mat4 c2w = solve_pnp(Pw.data(), px.data(), N, K, inl);
        Mat4 w2c = fsla::inv_rigid4(c2w);
        Mat3 Rr{};
        Vec3 tr{};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) Rr(i, j) = w2c(i, j);
            tr[i] = w2c(i, 3);
        }
        worst_R = std::max(worst_R, (double)rot_angle(mul(Rr, T(R))));
        worst_t = std::max(worst_t, std::sqrt((tr[0] - t[0]) * (tr[0] - t[0]) +
                                              (tr[1] - t[1]) * (tr[1] - t[1]) +
                                              (tr[2] - t[2]) * (tr[2] - t[2])));
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "worst %.2e deg", worst_R);
    check("rotation recovered", worst_R < 1e-3, buf);
    check("translation recovered", worst_t < 1e-5);
}

static void test_pnp_robust_planar() {
    // Near-planar scene (a thin slab) — the config where the DLT's coplanar
    // minimal samples flip. EPnP's all-point control basis stays well-posed.
    std::printf("test_pnp_robust_planar\n");
    const int N = 500;
    Mat3 K = make_K(400, 512, 512);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2);
    std::normal_distribution<double> thin(0,
                                          0.02);  // ~planar, slight thickness
    std::vector<double> Pw(3 * N);
    for (int i = 0; i < N; i++) {
        Pw[3 * i] = ux(RNG);
        Pw[3 * i + 1] = uy(RNG);
        Pw[3 * i + 2] = 6.0 + thin(RNG);
    }
    Mat3 R = rand_rotation();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R(i, j) = (i == j ? 1.0 : 0.0) * 0.85 + R(i, j) * 0.15;
    {
        fsla::Mat3 U, V;
        Vec3 s;
        fsla::svd3(R, U, s, V);
        R = mul(U, T(V));
        if (fsla::det3(R) < 0)
            for (int i = 0; i < 3; i++) R(i, 2) = -R(i, 2);
    }
    Vec3 t = {0.2, -0.1, 0.3};
    std::vector<double> px(2 * N), z(N);
    project(Pw.data(), N, R, t, K, px.data(), z.data());
    std::vector<char> inl;
    Mat4 w2c = fsla::inv_rigid4(solve_pnp(Pw.data(), px.data(), N, K, inl));
    Mat3 Rr{};
    Vec3 tr{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) Rr(i, j) = w2c(i, j);
        tr[i] = w2c(i, 3);
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.4f deg",
                  (double)rot_angle(mul(Rr, T(R))));
    check("planar rotation recovered", rot_angle(mul(Rr, T(R))) < 0.05, buf);
    check("planar translation recovered",
          std::sqrt((tr[0] - t[0]) * (tr[0] - t[0]) +
                    (tr[1] - t[1]) * (tr[1] - t[1]) +
                    (tr[2] - t[2]) * (tr[2] - t[2])) < 0.01);
}

static void test_pnp_robust_outliers() {
    std::printf("test_pnp_robust_outliers\n");
    const int N = 600;
    Mat3 K = make_K(400, 512, 512);
    std::uniform_real_distribution<double> ux(-2, 2), uy(-2, 2), uz(4, 8);
    std::vector<double> Pw(3 * N);
    for (int i = 0; i < N; i++) {
        Pw[3 * i] = ux(RNG);
        Pw[3 * i + 1] = uy(RNG);
        Pw[3 * i + 2] = uz(RNG);
    }
    Mat3 R = rand_rotation();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R(i, j) = (i == j ? 1.0 : 0.0) * 0.85 + R(i, j) * 0.15;
    {
        fsla::Mat3 U, V;
        Vec3 s;
        fsla::svd3(R, U, s, V);
        R = mul(U, T(V));
        if (fsla::det3(R) < 0)
            for (int i = 0; i < 3; i++) R(i, 2) = -R(i, 2);
    }
    Vec3 t = {0.3, -0.2, 0.4};
    std::vector<double> px(2 * N), z(N);
    project(Pw.data(), N, R, t, K, px.data(), z.data());
    std::vector<double> Pk, pk;
    for (int i = 0; i < N; i++)
        if (z[i] > 0) {
            for (int c = 0; c < 3; c++) Pk.push_back(Pw[3 * i + c]);
            pk.push_back(px[2 * i]);
            pk.push_back(px[2 * i + 1]);
        }
    const int M = (int)pk.size() / 2;
    const int n_out = (int)(0.15 * M);  // 15% gross pixel corruption
    std::vector<int> oidx(M);
    for (int i = 0; i < M; i++) oidx[i] = i;
    std::shuffle(oidx.begin(), oidx.end(), RNG);
    std::uniform_real_distribution<double> corrupt(-120, 120);
    for (int k = 0; k < n_out; k++) {
        int i = oidx[k];
        pk[2 * i] += corrupt(RNG);
        pk[2 * i + 1] += corrupt(RNG);
    }
    std::vector<char> inl;
    Mat4 w2c =
            fsla::inv_rigid4(solve_pnp(Pk.data(), pk.data(), M, K, inl, 3.0));
    Mat3 Rr{};
    Vec3 tr{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) Rr(i, j) = w2c(i, j);
        tr[i] = w2c(i, 3);
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.3f deg",
                  (double)rot_angle(mul(Rr, T(R))));
    check("rotation under outliers", rot_angle(mul(Rr, T(R))) < 0.5, buf);
    int rejected = 0;
    for (int k = 0; k < n_out; k++)
        if (!inl[oidx[k]]) rejected++;
    check("outliers rejected", rejected > 0.85 * n_out);
}

// ===========================================================================
// Accumulation / fusion / loop closure (the orchestration on top of the
// primitives, mirrors accumulate.py / fuse.py / loop_closure.py)
// ===========================================================================

// Synthetic multi-frame clip with KNOWN geometry: a pinhole camera on a smooth
// trajectory; each pair's gaussians are exact back-projections (engine channel
// layout), and consecutive runs are rescaled by a distinct per-run factor (the
// 1/baseline normalization that makes cross-run alignment a *similarity*). The
// accumulator must recover the camera trajectory (up to a global Sim(3)) and
// the per-link scale ratios.
static Mat4 cam2world_frame(int i) {
    const double ang = i * 4.0 * M_PI / 180.0;  // small yaw per frame
    const double c = std::cos(ang), s = std::sin(ang);
    Mat4 M = fsla::mat4_identity();
    M(0, 0) = c;
    M(0, 2) = s;
    M(2, 0) = -s;
    M(2, 2) = c;
    M(0, 3) = i * 0.4;  // translate along x
    return M;
}
static double depth_at(int i, int r, int c) {
    return 5.0 + 0.3 * std::sin(0.7 * i) + 0.05 * r - 0.03 * c +
           0.4 * (((r * 7 + c * 13 + i * 5) % 11) /
                  11.0);  // non-coplanar spread
}

static void test_accumulate_chain() {
    std::printf("test_accumulate_chain\n");
    const int H = 16, W = 16, gc = 23, P = H * W, F = 5;  // 5 frames -> 4 pairs
    const double f = 18.0;
    Mat3 K = make_K(f, W, H), Kinv = fsla::inv3(K);
    auto backproj = [&](int i, int r, int c) -> Vec3 {
        const double d = depth_at(i, r, c);
        Vec3 uv = {(double)c, (double)r, 1.0};
        Vec3 n = fsla::mat3_apply(Kinv, uv);
        return {n[0] * d, n[1] * d, n[2] * d};
    };

    Accumulator acc(H, W, /*opacity_threshold=*/0.05);
    std::vector<double> sigma(F);
    for (int k = 0; k < F - 1; k++) {     // pair (k, k+1) -> run k
        const double sk = 1.0 + 0.1 * k;  // distinct per-run scale
        sigma[k] = sk;
        Mat4 w2c_k = fsla::inv_rigid4(cam2world_frame(k));
        Mat4 rel = fsla::mat4_mul(
                w2c_k, cam2world_frame(k + 1));  // cam_k <- cam_{k+1}
        std::vector<float> buf((size_t)2 * P * gc, 0.0f);
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++) {
                const int i = r * W + c;
                Vec3 X0 = backproj(k, r, c);      // camera-k frame
                Vec3 Xn = backproj(k + 1, r, c);  // camera-(k+1) frame
                Vec3 X1 = apply_mat4(rel, Xn);    // -> camera-k frame
                float* v0 = &buf[(size_t)i * gc];
                float* v1 = &buf[(size_t)(P + i) * gc];
                for (int t = 0; t < 3; t++) {
                    v0[t] = (float)(sk * X0[t]);
                    v1[t] = (float)(sk * X1[t]);
                }
                v0[15] = v1[15] = 0.9f;  // opacity (activated)
                for (int t = 0; t < 3; t++) {
                    v0[3 + t] = 0.1f * t;
                    v1[3 + t] = 0.1f * t;
                }  // color
            }
        acc.add_pair(buf.data(), gc);
    }

    check("pair count", acc.pair_count() == F - 1);
    check("frame count", acc.frame_count() == F);
    check("cloud non-empty",
          acc.cloud().size() == (size_t)F * P);  // all opacities valid

    // per-link scale recovers sigma_{k-1}/sigma_k
    bool scales_ok = true;
    double worst_scale = 0;
    for (int k = 1; k < F - 1; k++) {
        const double expect = sigma[k - 1] / sigma[k];
        worst_scale =
                std::max(worst_scale, std::fabs(acc.links()[k].scale - expect));
        scales_ok &= std::fabs(acc.links()[k].scale - expect) < 1e-6;
    }
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "worst |Δs| %.2e", worst_scale);
        check("per-link scale recovered", scales_ok, buf);
    }
    double worst_resid = 0;
    for (int k = 1; k < F - 1; k++)
        worst_resid = std::max(worst_resid, acc.links()[k].resid_frac);
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.2e", worst_resid);
        check("per-link residual ~0", worst_resid < 1e-6, buf);
    }

    // recovered trajectory vs GT, after a global Sim(3) alignment (ATE)
    std::vector<Mat4> path = acc.camera_path();
    std::vector<double> rec(3 * F), gt(3 * F);
    for (int i = 0; i < F; i++) {
        rec[3 * i] = path[i](0, 3);
        rec[3 * i + 1] = path[i](1, 3);
        rec[3 * i + 2] = path[i](2, 3);
        Mat4 g = cam2world_frame(i);
        gt[3 * i] = g(0, 3);
        gt[3 * i + 1] = g(1, 3);
        gt[3 * i + 2] = g(2, 3);
    }
    Sim3 A = fit_similarity(rec.data(), gt.data(), F);
    std::vector<double> rec_al(3 * F);
    apply_sim_arr(A.s, A.R, A.t, rec.data(), F, rec_al.data());
    double mg[3] = {0, 0, 0};
    for (int i = 0; i < F; i++)
        for (int c = 0; c < 3; c++) mg[c] += gt[3 * i + c];
    for (int c = 0; c < 3; c++) mg[c] /= F;
    double ext = 0;
    for (int i = 0; i < F; i++)
        for (int c = 0; c < 3; c++) {
            double d = gt[3 * i + c] - mg[c];
            ext += d * d;
        }
    ext = std::sqrt(ext / F);
    const double ate = rms3(rec_al.data(), gt.data(), F) / ext;
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "ATE %.2e of extent", ate);
        check("trajectory recovered", ate < 1e-3, buf);
    }
}

static void test_consensus_fuse() {
    std::printf("test_consensus_fuse\n");
    std::vector<AccumPoint> cloud;
    // 5 surface points, each corroborated by frames {0,1,2} (tight cluster ->
    // one voxel, support 3); 20 isolated single-frame floaters far apart
    // (support 1). AccumPoint: {xyz, rgb, opacity, scale[3], quat(w,x,y,z),
    // frame}; opacity/ scale/quat are unused by fusion's counts here, set to a
    // unit opaque gaussian.
    for (int loc = 0; loc < 5; loc++)
        for (int fr = 0; fr < 3; fr++)
            cloud.push_back({(float)(2.0 * loc) + 0.001f * fr, 0.0f, 0.0f, 0.5f,
                             0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, fr});
    for (int j = 0; j < 20; j++)
        cloud.push_back({40.0f + 2.0f * j, 0.0f, 0.0f, 0.2f, 0.2f, 0.2f, 1.0f,
                         1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, j % 4});

    std::vector<AccumPoint> fused;
    FuseStats st = consensus_fuse(cloud, /*voxel_frac=*/0.02, /*k=*/2, fused,
                                  FUSE_AVERAGED);
    check("raw points", st.raw_points == 35);
    check("kept voxels (the surface)", st.kept_voxels == 5);
    check("fused size (averaged: one per voxel)", fused.size() == 5);
    check("points kept", st.points_kept == 15);
    check("floaters dropped", st.points_dropped == 20);
    // FUSE_KEPT: dense — every raw point in a consensus voxel, floaters dropped
    std::vector<AccumPoint> kept;
    consensus_fuse(cloud, /*voxel_frac=*/0.02, /*k=*/2, kept, FUSE_KEPT);
    check("kept mode: dense (all 15 consensus raw points)", kept.size() == 15);
    // FUSE_BEST: per voxel only the most-confident frame. Each surface voxel
    // has 3 frames with equal opacity (0.5) so frame 0 (first seen) wins the
    // tie -> 5 points.
    std::vector<AccumPoint> best;
    consensus_fuse(cloud, /*voxel_frac=*/0.02, /*k=*/2, best, FUSE_BEST);
    check("best mode: one frame per voxel (5 points)", best.size() == 5);
    // a fused point sits at its cluster centroid (~2*loc), color preserved
    bool centroid_ok = true;
    for (const auto& p : fused) {
        const double loc = std::round(p.x / 2.0);
        centroid_ok &= std::fabs(p.x - 2.0 * loc) < 0.01 &&
                       std::fabs(p.r - 0.5) < 1e-6;
    }
    check("fused centroid + color", centroid_ok);
}

// Gaussian-level de-ghost: a base scene seen by 3 frames through small
// per-object misalignments ghosts; consensus_refine (non-rigid, per-point) must
// pull the overlapping copies together.
static void test_consensus_refine() {
    std::printf("test_consensus_refine\n");
    // well-separated base points (a grid, spacing 1) so each object's 3 copies
    // share a voxel but distinct objects don't — the real (2D surface) regime.
    const int side = 4, M = side * side * side, Fr = 3;
    std::normal_distribution<double> g(0, 1);
    std::vector<double> base(3 * M);
    {
        int idx = 0;
        for (int a = 0; a < side; a++)
            for (int b = 0; b < side; b++)
                for (int c = 0; c < side; c++) {
                    base[3 * idx] = a;
                    base[3 * idx + 1] = b;
                    base[3 * idx + 2] = c;
                    idx++;
                }
    }
    auto small_rot = [&](double deg) {
        Vec3 ax = {g(RNG), g(RNG), g(RNG)};
        double n = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]);
        ax = {ax[0] / n, ax[1] / n, ax[2] / n};
        const double a = deg * M_PI / 180, c = std::cos(a), s = std::sin(a),
                     v = 1 - c, x = ax[0], y = ax[1], z = ax[2];
        return Mat3{{c + x * x * v, x * y * v - z * s, x * z * v + y * s,
                     y * x * v + z * s, c + y * y * v, y * z * v - x * s,
                     z * x * v - y * s, z * y * v + x * s, c + z * z * v}};
    };
    Sim3 P[3] = {sim_identity(),
                 {1.0, small_rot(1.0), {0.04, -0.03, 0.035}},
                 {1.0, small_rot(1.0), {-0.035, 0.04, -0.03}}};
    std::vector<AccumPoint> cloud(Fr * M);
    for (int f = 0; f < Fr; f++)
        for (int i = 0; i < M; i++) {
            Vec3 b = {base[3 * i], base[3 * i + 1], base[3 * i + 2]};
            Vec3 w = sim_apply(P[f], b);
            AccumPoint p{};
            p.x = (float)w[0];
            p.y = (float)w[1];
            p.z = (float)w[2];
            p.opacity = 1.0f;
            p.sx = p.sy = p.sz = 0.01f;
            p.qw = 1.0f;
            p.frame = f;
            cloud[(size_t)f * M + i] = p;
        }
    double mean[3] = {0, 0, 0};
    for (auto& p : cloud) {
        mean[0] += p.x;
        mean[1] += p.y;
        mean[2] += p.z;
    }
    for (int c = 0; c < 3; c++) mean[c] /= cloud.size();
    double ss = 0;
    for (auto& p : cloud) {
        double dx = p.x - mean[0], dy = p.y - mean[1], dz = p.z - mean[2];
        ss += dx * dx + dy * dy + dz * dz;
    }
    const double ext = std::sqrt(ss / cloud.size());
    auto spread = [&](const std::vector<AccumPoint>& c) {
        double s = 0;
        for (int i = 0; i < M; i++) {
            double cx = 0, cy = 0, cz = 0;
            for (int f = 0; f < Fr; f++) {
                cx += c[(size_t)f * M + i].x;
                cy += c[(size_t)f * M + i].y;
                cz += c[(size_t)f * M + i].z;
            }
            cx /= Fr;
            cy /= Fr;
            cz /= Fr;
            for (int f = 0; f < Fr; f++) {
                const auto& p = c[(size_t)f * M + i];
                s += (p.x - cx) * (p.x - cx) + (p.y - cy) * (p.y - cy) +
                     (p.z - cz) * (p.z - cz);
            }
        }
        return std::sqrt(s / (M * Fr)) / ext;
    };
    const double pre = spread(cloud);
    consensus_refine(cloud, /*voxel_frac=*/0.1, /*iters=*/12, /*alpha=*/0.6);
    const double post = spread(cloud);
    char buf[80];
    std::snprintf(buf, sizeof buf, "ghosting %.3f%% -> %.3f%% of extent",
                  100 * pre, 100 * post);
    check("ghosting present before", pre > 0.01);
    check("consensus_refine de-ghosts", post < 0.4 * pre, buf);
}

static void test_loop_distribute() {
    std::printf("test_loop_distribute\n");
    Mat3 R = rand_rotation();
    Mat4 D = sim_matrix({1.15, R, {0.4, -0.2, 0.1}});
    // sim4_invert is a true inverse of a similarity 4x4
    Mat4 S = sim_matrix({1.3, rand_rotation(), {1.0, -2.0, 0.5}});
    Mat4 SI = fsla::mat4_mul(sim4_invert(S), S);
    bool inv_ok = true;
    for (int i = 0; i < 16; i++)
        inv_ok &= std::fabs(SI.a[i] - fsla::mat4_identity().a[i]) < 1e-9;
    check("sim4_invert is an inverse", inv_ok);

    // a clean loop, drifted by D^{-k/n}; distribute_drift(D) must cancel it.
    const int n = 8;
    std::vector<Mat4> poses(n + 1);
    std::vector<Vec3> clean(n + 1);
    for (int k = 0; k <= n; k++) {
        const double tt = 2 * M_PI * k / n;
        clean[k] = {std::cos(tt), std::sin(tt), 0.1 * tt};
        Vec3 drifted = apply_mat4(sim_frac_power(D, -(double)k / n), clean[k]);
        Mat4 Pk = fsla::mat4_identity();
        Pk(0, 3) = drifted[0];
        Pk(1, 3) = drifted[1];
        Pk(2, 3) = drifted[2];
        poses[k] = Pk;
    }
    std::vector<Vec3> corr = distribute_drift(D, poses);
    double post = 0;
    for (int k = 0; k <= n; k++)
        for (int c = 0; c < 3; c++)
            post += (corr[k][c] - clean[k][c]) * (corr[k][c] - clean[k][c]);
    char buf[64];
    std::snprintf(buf, sizeof buf, "ATE %.1e", std::sqrt(post / (n + 1)));
    check("distribute_drift recovers clean loop",
          std::sqrt(post / (n + 1)) < 1e-9, buf);
}

// ===========================================================================
// Regression guards for the gaussian -> .splat seam (the class of bug that
// dropped rotation, then opacity, from the accumulated cloud writer)
// ===========================================================================

// #1 guard: pin the shared encoder's byte output. Both writers (single-run and
// accumulated cloud) call this; if it ever drops a channel or changes the
// convention the bytes change here. The opacity->alpha byte is the exact field
// the regression got wrong (it had been forced to 255).
static void test_splat_record() {
    std::printf("test_splat_record\n");
    const float pos[3] = {1.0f, 2.0f, 3.0f};
    const float scale[3] = {0.01f, 0.02f, 0.03f};
    const float quat[4] = {0.70710678f, 0.70710678f, 0.0f,
                           0.0f};  // (w,x,y,z), unit
    const float rgb[3] = {0.2f, 0.4f, 0.6f};
    unsigned char rec[32];
    aicore::gaussian::encode_splat_record(rec, pos, scale, quat, rgb, 0.5f);

    float p[3], sc[3];
    std::memcpy(p, rec, 12);
    std::memcpy(sc, rec + 12, 12);
    check("pos: OpenCV->GL flips y,z",
          p[0] == 1.0f && p[1] == -2.0f && p[2] == -3.0f);
    check("scale: passthrough",
          sc[0] == 0.01f && sc[1] == 0.02f && sc[2] == 0.03f);
    check("rgb bytes", rec[24] == (unsigned char)(0.2f * 255.0f) &&
                               rec[25] == (unsigned char)(0.4f * 255.0f) &&
                               rec[26] == (unsigned char)(0.6f * 255.0f));
    // THE regression field: opacity must become the alpha, not a forced 255.
    char buf[48];
    std::snprintf(buf, sizeof buf, "alpha=%d (must be 127, not 255)", rec[27]);
    check("opacity -> alpha (not forced opaque)", rec[27] == 127, buf);
    // rotation must be encoded with the (w,x,y,z)->(-x,w,-z,y) remap, not
    // dropped. q=(0.7071,0.7071,0,0) -> (-0.7071,0.7071,0,0) -> bytes
    // (37,218,128,128).
    check("rotation: remap encoded (not identity)",
          rec[28] == 37 && rec[29] == 218 && rec[30] == 128 && rec[31] == 128);
    // boundary opacities
    unsigned char r1[32], r0[32];
    aicore::gaussian::encode_splat_record(r1, pos, scale, quat, rgb, 1.0f);
    aicore::gaussian::encode_splat_record(r0, pos, scale, quat, rgb, 0.0f);
    check("alpha boundaries (1->255, 0->0)", r1[27] == 255 && r0[27] == 0);
}

// #2 guard: run-0 accumulation must preserve EVERY gaussian channel. With one
// pair the global transform is identity, so each AccumPoint should equal the
// engine's gaussian for that pixel (xyz, SH->rgb, opacity, scale, rotation,
// frame). A dropped channel — the exact bug — fails here, asset-free.
static void test_accumulate_channels() {
    std::printf("test_accumulate_channels\n");
    const int H = 4, W = 4, gc = 23, P = H * W;
    std::vector<float> g((size_t)2 * P * gc, 0.0f);
    for (int v = 0; v < 2; v++)
        for (int i = 0; i < P; i++) {
            float* a = &g[(size_t)(v * P + i) * gc];
            const float b = (float)(v * 100 + i);
            a[0] = b + 0.11f;
            a[1] = b + 0.22f;
            a[2] = b + 0.33f;  // xyz (distinct)
            a[3] = 0.5f;
            a[4] = -0.5f;
            a[5] = 1.0f;  // SH-DC -> in-range rgb
            a[15] = 0.2f +
                    0.5f * ((i % 3) / 2.0f);  // opacity in {0.2,0.45,0.7} > thr
            a[16] = 0.01f + 0.001f * i;
            a[17] = 0.02f;
            a[18] = 0.03f;  // scale (distinct)
            a[19] = 0.8f;
            a[20] = 0.6f;
            a[21] = 0.0f;
            a[22] = 0.0f;  // unit quat (w,x,y,z)
        }
    Accumulator acc(H, W, /*opacity_threshold=*/0.05);
    acc.add_pair(g.data(), gc);
    const auto& cloud = acc.cloud();
    check("run-0 cloud = all pixels (both views kept)",
          cloud.size() == (size_t)2 * P);

    const double C0 = 0.28209479177387814;
    bool ok = cloud.size() == (size_t)2 * P;
    auto cl = [](float x, float y) { return std::fabs(x - y) < 1e-5f; };
    for (int v = 0; v < 2 && ok; v++)
        for (int i = 0; i < P && ok; i++) {
            const float* a = &g[(size_t)(v * P + i) * gc];
            const AccumPoint& p = cloud[(size_t)v * P + i];
            ok &= cl(p.x, a[0]) && cl(p.y, a[1]) &&
                  cl(p.z, a[2]);  // xyz (identity sim)
            float er = (float)(a[3] * C0 + 0.5), eg = (float)(a[4] * C0 + 0.5),
                  eb = (float)(a[5] * C0 + 0.5);
            ok &= cl(p.r, er) && cl(p.g, eg) && cl(p.b, eb);  // SH -> rgb
            ok &= cl(p.opacity, a[15]);  // opacity (the dropped one)
            ok &= cl(p.sx, a[16]) && cl(p.sy, a[17]) &&
                  cl(p.sz, a[18]);  // scale (T.s=1)
            ok &= cl(p.qw, a[19]) && cl(p.qx, a[20]) && cl(p.qy, a[21]) &&
                  cl(p.qz, a[22]);  // rotation
            ok &= (p.frame == v);   // source frame
        }
    check("every gaussian channel survives run-0 accumulation", ok);
}

// Golden: parallax_stats on synthetic cameras with hand-computed angles. A
// lateral (strafe) baseline yields a large triangulation/lateral angle; a pure
// forward (dolly) baseline of the SAME length yields ~0 — the whole point of
// the metric (it measures depth-resolving motion, not raw camera displacement).
static void test_parallax_geometry() {
    std::printf("test_parallax_geometry\n");
    const double DEG = 180.0 / 3.14159265358979323846;
    const Vec3 axis0 = {0, 0, 1};  // view-0 looks down +z
    // points at depth 1 (a few, all same depth -> clean medians)
    const int N = 5;
    std::vector<float> pts(3 * N);
    for (int i = 0; i < N; i++) {
        pts[3 * i] = 0.0f;
        pts[3 * i + 1] = 0.0f;
        pts[3 * i + 2] = 1.0f;
    }

    // (1) pure strafe: baseline 0.1 along x, perpendicular to the optical axis
    Parallax s = parallax_stats({0, 0, 0}, {0.1, 0, 0}, axis0, pts.data(),
                                nullptr, N, 0.0);
    const double tri_expect = std::atan(0.1) * DEG;  // 5.7106 deg
    char b1[96];
    std::snprintf(b1, sizeof b1,
                  "strafe lateral=%.3f tri=%.3f (expect 90, %.3f)",
                  s.lateral_angle_deg, s.tri_angle_deg, tri_expect);
    check("strafe -> lateral angle ~90 deg",
          std::fabs(s.lateral_angle_deg - 90.0) < 1e-3, b1);
    check("strafe -> tri angle = atan(B/Z)",
          std::fabs(s.tri_angle_deg - tri_expect) < 1e-3, b1);
    check("strafe -> B/Z = 0.1", std::fabs(s.baseline_over_depth - 0.1) < 1e-4);
    check("strafe -> median depth = 1", std::fabs(s.median_depth - 1.0) < 1e-4);

    // (2) pure dolly: SAME baseline length 0.1 but along the optical axis
    Parallax d = parallax_stats({0, 0, 0}, {0, 0, 0.1}, axis0, pts.data(),
                                nullptr, N, 0.0);
    char b2[96];
    std::snprintf(b2, sizeof b2, "dolly lateral=%.3f tri=%.3f (expect ~0)",
                  d.lateral_angle_deg, d.tri_angle_deg);
    check("dolly -> lateral angle ~0 deg",
          std::fabs(d.lateral_angle_deg) < 1e-3, b2);
    check("dolly -> tri angle ~0 deg (no parallax)",
          std::fabs(d.tri_angle_deg) < 1e-3, b2);
    check("dolly baseline equals strafe baseline",
          std::fabs(d.baseline - s.baseline) < 1e-6);
}

int main() {
    test_similarity_roundtrip();
    test_parallax_geometry();
    test_scale_detection();
    test_nonlinear_detection();
    test_outlier_robustness();
    test_loop_correction();
    test_loop_closure();
    test_focal_recovery();
    test_pnp_recovery();
    test_pnp_outliers();
    test_pnp_robust_recovery();
    test_pnp_robust_planar();
    test_pnp_robust_outliers();
    test_accumulate_chain();
    test_consensus_fuse();
    test_consensus_refine();
    test_loop_distribute();
    test_splat_record();
    test_accumulate_channels();
    std::printf(failures ? "\ntest_pose: %d FAILURES\n" : "\ntest_pose: ok\n",
                failures);
    return failures ? 1 : 0;
}
