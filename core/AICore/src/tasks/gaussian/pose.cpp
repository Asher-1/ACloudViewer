// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// pose.cpp — C++ port of the (now removed) pose/ research prototype: focal +
// Umeyama/Sim(3) align + robust PnP + sliding-window accumulation / loop
// closure / consensus fusion. See git history for the Python prototype it was
// validated against, layer by layer.
//
// Faithful to FreeSplatter's scene recipe; the dependency-free solver is what
// ships here (the prototype's numpy reference was verified ~1e-7 vs cv2 and
// bit-exact vs upstream estimate_poses on real engine output). All linear
// algebra goes through the self-contained Jacobi eigensolver in linalg.h — no
// Eigen, no OpenCV.
#include "pose.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace aicore {
namespace gaussian {
namespace pose {

using fsla::det3;
using fsla::inv3;
using fsla::inv_rigid4;
using fsla::mat3_apply;
using fsla::mat3_identity;
using fsla::mat3_mul;
using fsla::mat3_transpose;
using fsla::mat4_identity;
using fsla::smallest_eigenvector;
using fsla::svd3;

namespace {

// Deterministic SplitMix64 — so RANSAC is reproducible across platforms (we
// don't need numpy's PCG64 bitstream, only a fixed inlier-converging sampler).
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    int bounded(int n) { return (int)(next() % (uint64_t)n); }
};

void sample_distinct(Rng &rng, int N, int k, int *out) {
    for (int i = 0; i < k; i++) {
        int v;
        bool dup;
        do {
            v = rng.bounded(N);
            dup = false;
            for (int j = 0; j < i; j++)
                if (out[j] == v) {
                    dup = true;
                    break;
                }
        } while (dup);
        out[i] = v;
    }
}

double rms3(const double *r, int N) {  // sqrt(mean over rows of ||row||^2)
    double s = 0;
    for (int i = 0; i < N; i++)
        s += r[3 * i] * r[3 * i] + r[3 * i + 1] * r[3 * i + 1] +
             r[3 * i + 2] * r[3 * i + 2];
    return std::sqrt(s / (double)N);
}

void mean3(const double *X, int N, double *m) {
    m[0] = m[1] = m[2] = 0;
    for (int i = 0; i < N; i++) {
        m[0] += X[3 * i];
        m[1] += X[3 * i + 1];
        m[2] += X[3 * i + 2];
    }
    m[0] /= N;
    m[1] /= N;
    m[2] /= N;
}

// Solve A X = B in place (A n x n row-major, B n x m row-major) by Gaussian
// elimination with partial pivoting. Result overwrites B. Used for the affine
// normal equations in the residual ladder.
void solve_lin(double *A, int n, double *B, int m) {
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col]))
                piv = r;
        if (piv != col) {
            for (int c = 0; c < n; c++)
                std::swap(A[col * n + c], A[piv * n + c]);
            for (int c = 0; c < m; c++)
                std::swap(B[col * m + c], B[piv * m + c]);
        }
        const double d = A[col * n + col];
        if (std::fabs(d) < 1e-300) continue;
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            const double f = A[r * n + col] / d;
            if (f == 0) continue;
            for (int c = col; c < n; c++) A[r * n + c] -= f * A[col * n + c];
            for (int c = 0; c < m; c++) B[r * m + c] -= f * B[col * m + c];
        }
    }
    for (int col = 0; col < n; col++) {
        const double d = A[col * n + col];
        if (std::fabs(d) > 1e-300)
            for (int c = 0; c < m; c++) B[col * m + c] /= d;
    }
}

Mat3 rodrigues(const Vec3 &axis,
               double phi) {  // rotation about unit axis by phi
    const double cphi = std::cos(phi), sphi = std::sin(phi), v = 1.0 - cphi;
    const double x = axis[0], y = axis[1], z = axis[2];
    Mat3 R{};
    R(0, 0) = cphi + x * x * v;
    R(0, 1) = x * y * v - z * sphi;
    R(0, 2) = x * z * v + y * sphi;
    R(1, 0) = y * x * v + z * sphi;
    R(1, 1) = cphi + y * y * v;
    R(1, 2) = y * z * v - x * sphi;
    R(2, 0) = z * x * v - y * sphi;
    R(2, 1) = z * y * v + x * sphi;
    R(2, 2) = cphi + z * z * v;
    return R;
}

}  // namespace

// ---- focal ----------------------------------------------------------------

double estimate_focal(const double *pts3d,
                      const double *pixels,
                      int N,
                      double ppx,
                      double ppy,
                      int weiszfeld_iters) {
    std::vector<double> P(2 * N), U(2 * N), pu(N), uu(N);
    for (int i = 0; i < N; i++) {
        const double X = pts3d[3 * i], Y = pts3d[3 * i + 1],
                     Z = pts3d[3 * i + 2];
        double ux = (Z != 0.0) ? X / Z : 0.0;
        double uy = (Z != 0.0) ? Y / Z : 0.0;
        if (!std::isfinite(ux))
            ux = 0.0;  // mirror np.nan_to_num(posinf/neginf->0)
        if (!std::isfinite(uy)) uy = 0.0;
        const double Px = pixels[2 * i] - ppx;
        const double Py = pixels[2 * i + 1] - ppy;
        P[2 * i] = Px;
        P[2 * i + 1] = Py;
        U[2 * i] = ux;
        U[2 * i + 1] = uy;
        pu[i] = Px * ux + Py * uy;
        uu[i] = ux * ux + uy * uy;
    }
    double spu = 0, suu = 0;
    for (int i = 0; i < N; i++) {
        spu += pu[i];
        suu += uu[i];
    }
    double f = (suu != 0.0) ? spu / suu : 0.0;
    for (int it = 0; it < weiszfeld_iters; it++) {
        double num = 0, den = 0;
        for (int i = 0; i < N; i++) {
            const double dx = P[2 * i] - f * U[2 * i],
                         dy = P[2 * i + 1] - f * U[2 * i + 1];
            const double r = std::sqrt(dx * dx + dy * dy);
            const double w = 1.0 / std::max(r, 1e-8);
            num += w * pu[i];
            den += w * uu[i];
        }
        f = (den != 0.0) ? num / den : f;
    }
    return f;
}

// ---- similarity -----------------------------------------------------------

Sim3 sim_identity() { return {1.0, mat3_identity(), {0, 0, 0}}; }

Vec3 sim_apply(const Sim3 &T, const Vec3 &x) {
    Vec3 Rx = mat3_apply(T.R, x);
    return {T.s * Rx[0] + T.t[0], T.s * Rx[1] + T.t[1], T.s * Rx[2] + T.t[2]};
}

Sim3 sim_compose(const Sim3 &T2, const Sim3 &T1) {  // apply T1 then T2
    Sim3 r;
    r.s = T2.s * T1.s;
    r.R = mat3_mul(T2.R, T1.R);
    Vec3 R2t1 = mat3_apply(T2.R, T1.t);
    r.t = {T2.s * R2t1[0] + T2.t[0], T2.s * R2t1[1] + T2.t[1],
           T2.s * R2t1[2] + T2.t[2]};
    return r;
}

Sim3 sim_invert(const Sim3 &T) {
    Sim3 r;
    r.s = 1.0 / T.s;
    r.R = mat3_transpose(T.R);
    Vec3 Rit = mat3_apply(r.R, T.t);
    r.t = {-r.s * Rit[0], -r.s * Rit[1], -r.s * Rit[2]};
    return r;
}

Mat4 sim_matrix(const Sim3 &T) {
    Mat4 M = mat4_identity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) M(i, j) = T.s * T.R(i, j);
    for (int i = 0; i < 3; i++) M(i, 3) = T.t[i];
    return M;
}

Sim3 fit_similarity(const double *X, const double *Y, int N, bool with_scale) {
    double mx[3], my[3];
    mean3(X, N, mx);
    mean3(Y, N, my);
    // Sigma = (Ycᵀ Xc)/n, cross-covariance mapping X -> Y; var_x = mean
    // ||Xc||^2.
    Mat3 Sigma{};
    double var_x = 0;
    for (int i = 0; i < N; i++) {
        const double xc[3] = {X[3 * i] - mx[0], X[3 * i + 1] - mx[1],
                              X[3 * i + 2] - mx[2]};
        const double yc[3] = {Y[3 * i] - my[0], Y[3 * i + 1] - my[1],
                              Y[3 * i + 2] - my[2]};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) Sigma(r, c) += yc[r] * xc[c];
        var_x += xc[0] * xc[0] + xc[1] * xc[1] + xc[2] * xc[2];
    }
    for (int i = 0; i < 9; i++) Sigma.a[i] /= N;
    var_x /= N;

    Mat3 U, V;
    Vec3 D;
    svd3(Sigma, U, D, V);  // Sigma = U diag(D) Vᵀ
    double sgn = (det3(U) * det3(V) < 0) ? -1.0 : 1.0;
    // R = U diag(1,1,sgn) Vᵀ
    Mat3 US = U;
    for (int r = 0; r < 3; r++) US(r, 2) *= sgn;
    Mat3 R = mat3_mul(US, mat3_transpose(V));

    double s = 1.0;
    if (with_scale && var_x > 0) s = (D[0] + D[1] + sgn * D[2]) / var_x;

    Vec3 mxv = {mx[0], mx[1], mx[2]};
    Vec3 Rmx = mat3_apply(R, mxv);
    Sim3 T;
    T.s = s;
    T.R = R;
    T.t = {my[0] - s * Rmx[0], my[1] - s * Rmx[1], my[2] - s * Rmx[2]};
    return T;
}

Sim3 fit_similarity_ransac(const double *X,
                           const double *Y,
                           int N,
                           double thresh,
                           int iters,
                           std::vector<char> &inliers,
                           bool with_scale,
                           uint64_t seed) {
    // RANSAC's minimal sample is 3 pairs; with fewer there is nothing to sample
    // (and the sampler would divide by N). Degenerate but valid at the public
    // boundary (e.g. an image pair with no overlapping valid pixels): take all
    // points as inliers, and a plain fit when there is at least one.
    if (N < 3) {
        inliers.assign(N, 1);
        return (N >= 1) ? fit_similarity(X, Y, N, with_scale) : sim_identity();
    }
    Rng rng(seed);
    std::vector<char> best(N, 0);
    int best_cnt = 0;
    std::vector<double> xs(9), ys(9);  // 3-point minimal samples
    for (int it = 0; it < iters; it++) {
        int idx[3];
        sample_distinct(rng, N, 3, idx);
        for (int j = 0; j < 3; j++)
            for (int c = 0; c < 3; c++) {
                xs[3 * j + c] = X[3 * idx[j] + c];
                ys[3 * j + c] = Y[3 * idx[j] + c];
            }
        Sim3 T = fit_similarity(xs.data(), ys.data(), 3, with_scale);
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            Vec3 xi = {X[3 * i], X[3 * i + 1], X[3 * i + 2]};
            Vec3 p = sim_apply(T, xi);
            const double dx = p[0] - Y[3 * i], dy = p[1] - Y[3 * i + 1],
                         dz = p[2] - Y[3 * i + 2];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) < thresh) cnt++;
        }
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best.assign(N, 0);
            for (int i = 0; i < N; i++) {
                Vec3 xi = {X[3 * i], X[3 * i + 1], X[3 * i + 2]};
                Vec3 p = sim_apply(T, xi);
                const double dx = p[0] - Y[3 * i], dy = p[1] - Y[3 * i + 1],
                             dz = p[2] - Y[3 * i + 2];
                best[i] = (std::sqrt(dx * dx + dy * dy + dz * dz) < thresh) ? 1
                                                                            : 0;
            }
        }
    }
    if (best_cnt < 3) std::fill(best.begin(), best.end(), 1);
    // refit on the inlier set
    std::vector<double> xi, yi;
    for (int i = 0; i < N; i++)
        if (best[i]) {
            for (int c = 0; c < 3; c++) {
                xi.push_back(X[3 * i + c]);
                yi.push_back(Y[3 * i + c]);
            }
        }
    inliers = best;
    return fit_similarity(xi.data(), yi.data(), (int)(xi.size() / 3),
                          with_scale);
}

// affine Y ~= A X + b (12-DoF), returns prediction rms only via caller; here we
// fill `pred` (N*3). Normal equations on Xh=[X,1].
static double affine_residual(const double *X, const double *Y, int N) {
    double G[16] = {0};  // XhᵀXh (4x4)
    double B[12] = {0};  // XhᵀY  (4x3)
    for (int i = 0; i < N; i++) {
        const double xh[4] = {X[3 * i], X[3 * i + 1], X[3 * i + 2], 1.0};
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) G[r * 4 + c] += xh[r] * xh[c];
            for (int c = 0; c < 3; c++) B[r * 3 + c] += xh[r] * Y[3 * i + c];
        }
    }
    solve_lin(G, 4, B, 3);  // B now holds M (4x3): rows are coeffs
    double s = 0;
    for (int i = 0; i < N; i++) {
        const double xh[4] = {X[3 * i], X[3 * i + 1], X[3 * i + 2], 1.0};
        for (int c = 0; c < 3; c++) {
            double p = 0;
            for (int r = 0; r < 4; r++) p += xh[r] * B[r * 3 + c];
            const double d = p - Y[3 * i + c];
            s += d * d;
        }
    }
    return std::sqrt(s / (double)N);
}

Ladder diagnose(
        const double *X, const double *Y, int N, double tol, double corr_tol) {
    Ladder L{};
    double my[3];
    mean3(Y, N, my);
    std::vector<double> yc(3 * N);
    for (int i = 0; i < N; i++) {
        yc[3 * i] = Y[3 * i] - my[0];
        yc[3 * i + 1] = Y[3 * i + 1] - my[1];
        yc[3 * i + 2] = Y[3 * i + 2] - my[2];
    }
    L.scene = rms3(yc.data(), N);

    Sim3 Tr = fit_similarity(X, Y, N, /*with_scale=*/false);
    Sim3 Ts = fit_similarity(X, Y, N, /*with_scale=*/true);
    std::vector<double> res_r(3 * N), res_s(3 * N);
    for (int i = 0; i < N; i++) {
        Vec3 xi = {X[3 * i], X[3 * i + 1], X[3 * i + 2]};
        Vec3 pr = sim_apply(Tr, xi), ps = sim_apply(Ts, xi);
        for (int c = 0; c < 3; c++) {
            res_r[3 * i + c] = pr[c] - Y[3 * i + c];
            res_s[3 * i + c] = ps[c] - Y[3 * i + c];
        }
    }
    L.rigid = rms3(res_r.data(), N);
    L.similarity = rms3(res_s.data(), N);
    L.affine = affine_residual(X, Y, N);
    L.scale = Ts.s;

    // depth_corr: Pearson(||X-mean||, ||similarity residual||)
    double mx[3];
    mean3(X, N, mx);
    std::vector<double> depth(N), rmag(N);
    for (int i = 0; i < N; i++) {
        const double dx = X[3 * i] - mx[0], dy = X[3 * i + 1] - mx[1],
                     dz = X[3 * i + 2] - mx[2];
        depth[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        rmag[i] = std::sqrt(res_s[3 * i] * res_s[3 * i] +
                            res_s[3 * i + 1] * res_s[3 * i + 1] +
                            res_s[3 * i + 2] * res_s[3 * i + 2]);
    }
    double md = 0, mr = 0;
    for (int i = 0; i < N; i++) {
        md += depth[i];
        mr += rmag[i];
    }
    md /= N;
    mr /= N;
    double cov = 0, vd = 0, vr = 0;
    for (int i = 0; i < N; i++) {
        cov += (depth[i] - md) * (rmag[i] - mr);
        vd += (depth[i] - md) * (depth[i] - md);
        vr += (rmag[i] - mr) * (rmag[i] - mr);
    }
    L.depth_corr = (vr > 1e-24) ? cov / std::sqrt(vd * vr) : 0.0;

    const double sc = (L.scene > 0) ? L.scene : 1.0;
    const double rr = L.rigid / sc, rs = L.similarity / sc, ra = L.affine / sc;
    L.aff_gain =
            (L.similarity > 0) ? (L.similarity - L.affine) / L.similarity : 0.0;
    L.structured = std::fabs(L.depth_corr) > corr_tol || L.aff_gain > 0.25;
    if (rr < tol)
        L.verdict = "rigid_ok";
    else if (rs < tol)
        L.verdict = "needs_scale";
    else if (ra < tol)
        L.verdict = "needs_affine";
    else if (L.structured)
        L.verdict = "nonlinear";
    else
        L.verdict = "similarity_plus_noise";
    return L;
}

LoopError loop_closure_error(const std::vector<Sim3> &links) {
    Sim3 T = sim_identity();
    for (const Sim3 &Ti : links) T = sim_compose(Ti, T);
    LoopError e;
    e.scale_err = std::fabs(std::log(T.s));
    const double tr = T.R(0, 0) + T.R(1, 1) + T.R(2, 2);
    e.rot_deg = std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0))) *
                180.0 / M_PI;
    e.trans = std::sqrt(T.t[0] * T.t[0] + T.t[1] * T.t[1] + T.t[2] * T.t[2]);
    return e;
}

// M^f for a Sim(3) 4x4 [[sR,t],[0,1]], closed form via the group's
// one-parameter subgroup: A^f = s^f R^f (R^f = same axis, f*angle), translation
// by (A^f - I)(A - I)^{-1} t (the analytic continuation of the integer-power
// geometric series). Equals numpy's eig-based principal value while angle <
// 180.
Mat4 sim_frac_power(const Mat4 &M, double f) {
    Mat3 A{};
    Vec3 t{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) A(i, j) = M(i, j);
        t[i] = M(i, 3);
    }
    const double s = std::cbrt(det3(A));
    Mat3 R = A;
    for (int i = 0; i < 9; i++) R.a[i] /= s;
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    const double ang =
            std::acos(std::max(-1.0, std::min(1.0, (tr - 1.0) / 2.0)));

    Mat3 Rf;
    if (ang < 1e-12) {
        Rf = mat3_identity();
    } else {
        const double sa = std::sin(ang);
        Vec3 axis = {(R(2, 1) - R(1, 2)) / (2 * sa),
                     (R(0, 2) - R(2, 0)) / (2 * sa),
                     (R(1, 0) - R(0, 1)) / (2 * sa)};
        Rf = rodrigues(axis, f * ang);
    }
    const double sf = std::pow(s, f);
    Mat3 Af = Rf;
    for (int i = 0; i < 9; i++) Af.a[i] *= sf;

    Mat4 out = mat4_identity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) out(i, j) = Af(i, j);

    // translation: (Af - I)(A - I)^{-1} t, with the A->I limit handled as f*t.
    Mat3 AmI = A;
    for (int i = 0; i < 3; i++) AmI(i, i) -= 1.0;
    if (std::fabs(det3(AmI)) < 1e-12) {
        for (int i = 0; i < 3; i++) out(i, 3) = f * t[i];
    } else {
        Mat3 AfmI = Af;
        for (int i = 0; i < 3; i++) AfmI(i, i) -= 1.0;
        Mat3 K = mat3_mul(AfmI, inv3(AmI));
        Vec3 tf = mat3_apply(K, t);
        for (int i = 0; i < 3; i++) out(i, 3) = tf[i];
    }
    return out;
}

// ---- PnP ------------------------------------------------------------------

namespace {

// DLT: smallest right singular vector of the 2N-row system, via the smallest
// eigenvector of AᵀA (12x12). idx selects the k points used. Returns the 3x4
// matrix p (row-major) up to sign/scale.
std::array<double, 12> dlt(const double *Xw,
                           const double *xn,
                           const int *idx,
                           int k) {
    double AtA[144] = {0};
    for (int j = 0; j < k; j++) {
        const int i = idx[j];
        const double X = Xw[3 * i], Y = Xw[3 * i + 1], Z = Xw[3 * i + 2];
        const double u = xn[2 * i], v = xn[2 * i + 1];
        const double ra[12] = {X, Y, Z,      1,      0,      0,
                               0, 0, -u * X, -u * Y, -u * Z, -u};
        const double rb[12] = {0, 0, 0,      0,      X,      Y,
                               Z, 1, -v * X, -v * Y, -v * Z, -v};
        for (int r = 0; r < 12; r++)
            for (int c = 0; c < 12; c++)
                AtA[r * 12 + c] += ra[r] * ra[c] + rb[r] * rb[c];
    }
    std::vector<double> v = smallest_eigenvector(AtA, 12);
    std::array<double, 12> p;
    for (int i = 0; i < 12; i++) p[i] = v[i];
    return p;
}

struct RT {
    Mat3 R;
    Vec3 t;
    double err;
    bool ok;
};

// Resolve the DLT sign and recover a proper (R,t) world2cam, picking the sign
// with lower reprojection error among cheirality-valid candidates (points in
// front).
RT decode(const std::array<double, 12> &p,
          const double *Xw,
          const double *xn,
          const int *idx,
          int k) {
    RT best;
    best.ok = false;
    best.err = 1e300;
    for (double sgn : {1.0, -1.0}) {
        Mat3 M{};
        Vec3 tau{};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) M(r, c) = sgn * p[r * 4 + c];
            tau[r] = sgn * p[r * 4 + 3];
        }
        Mat3 U, V;
        Vec3 Sv;
        svd3(M, U, Sv, V);
        const double d = (det3(U) * det3(V) < 0) ? -1.0 : 1.0;
        Mat3 US = U;
        for (int r = 0; r < 3; r++) US(r, 2) *= d;
        Mat3 R = mat3_mul(US, mat3_transpose(V));  // nearest proper rotation
        const double lam = (Sv[0] + Sv[1] + Sv[2]) / 3.0;
        if (lam < 1e-12) continue;
        Vec3 t = {tau[0] / lam, tau[1] / lam, tau[2] / lam};

        double err = 0;
        bool behind = false;
        for (int j = 0; j < k; j++) {
            const int i = idx[j];
            Vec3 xi = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
            Vec3 Xc = mat3_apply(R, xi);
            Xc[0] += t[0];
            Xc[1] += t[1];
            Xc[2] += t[2];
            if (Xc[2] <= 0) {
                behind = true;
                break;
            }
            const double px = Xc[0] / Xc[2], py = Xc[1] / Xc[2];
            const double dx = px - xn[2 * i], dy = py - xn[2 * i + 1];
            err += std::sqrt(dx * dx + dy * dy);
        }
        if (behind) continue;
        err /= k;
        if (err < best.err) {
            best.err = err;
            best.R = R;
            best.t = t;
            best.ok = true;
        }
    }
    return best;
}

}  // namespace

void pixel_grid(int H, int W, std::vector<double> &out) {
    out.resize((size_t)2 * H * W);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            const size_t k = (size_t)r * W + c;
            out[2 * k] = c;
            out[2 * k + 1] = r;
        }
}

Mat4 solve_pnp_numpy(const double *Xw,
                     const double *pixels,
                     int N,
                     const Mat3 &K,
                     std::vector<char> &inliers,
                     double thresh_px,
                     int iters,
                     uint64_t seed) {
    const Mat3 Kinv = inv3(K);
    std::vector<double> xn((size_t)2 * N);
    for (int i = 0; i < N; i++) {
        Vec3 uv = {pixels[2 * i], pixels[2 * i + 1], 1.0};
        Vec3 n = mat3_apply(Kinv, uv);
        xn[2 * i] = n[0];
        xn[2 * i + 1] = n[1];
    }
    const double thresh = thresh_px / K(0, 0);

    Rng rng(seed);
    std::vector<char> best(N, 0);
    int best_cnt = 0;
    for (int it = 0; it < iters; it++) {
        int idx[6];
        sample_distinct(rng, N, 6, idx);
        RT rt = decode(dlt(Xw, xn.data(), idx, 6), Xw, xn.data(), idx, 6);
        if (!rt.ok) continue;
        int cnt = 0;
        for (int i = 0; i < N; i++) {
            Vec3 xi = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
            Vec3 Xc = mat3_apply(rt.R, xi);
            Xc[2] += rt.t[2];
            if (Xc[2] <= 0) continue;
            Xc[0] += rt.t[0];
            Xc[1] += rt.t[1];
            const double dx = Xc[0] / Xc[2] - xn[2 * i],
                         dy = Xc[1] / Xc[2] - xn[2 * i + 1];
            if (std::sqrt(dx * dx + dy * dy) < thresh) cnt++;
        }
        if (cnt > best_cnt) {
            best_cnt = cnt;
            best.assign(N, 0);
            for (int i = 0; i < N; i++) {
                Vec3 xi = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
                Vec3 Xc = mat3_apply(rt.R, xi);
                Xc[0] += rt.t[0];
                Xc[1] += rt.t[1];
                Xc[2] += rt.t[2];
                if (Xc[2] <= 0) continue;
                const double dx = Xc[0] / Xc[2] - xn[2 * i],
                             dy = Xc[1] / Xc[2] - xn[2 * i + 1];
                best[i] = (std::sqrt(dx * dx + dy * dy) < thresh) ? 1 : 0;
            }
        }
    }
    std::vector<int> sel;
    for (int i = 0; i < N; i++)
        if (best[i]) sel.push_back(i);
    if ((int)sel.size() < 6) {
        sel.clear();
        for (int i = 0; i < N; i++) sel.push_back(i);
    }
    RT rt = decode(dlt(Xw, xn.data(), sel.data(), (int)sel.size()), Xw,
                   xn.data(), sel.data(), (int)sel.size());

    Mat4 world2cam = mat4_identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) world2cam(i, j) = rt.R(i, j);
        world2cam(i, 3) = rt.t[i];
    }
    inliers = best;
    return inv_rigid4(world2cam);
}

// ---- robust PnP: EPnP + Gauss-Newton --------------------------------------
//
// EPnP (Lepetit/Moreno-Noguer/Fua 2009): express each world point as a
// barycentric combination of 4 control points; the camera-frame control points
// live in the null space of a 2n x 12 system, recovered from the eigenvectors
// of the 12x12 MᵀM (the same Jacobi eigensolver as everywhere else). It is
// non-iterative, uses ALL points (no random minimal samples), and handles
// near-planar configs — so it does not suffer the DLT's seed-dependent mirror
// flips. We try N=1,2,3 null vectors, pick the lowest reprojection error, then
// polish (R,t) with a Huber-robust Gauss-Newton on the reprojection residual.

namespace {

Mat3 skew_neg(const Vec3 &x) {  // -[x]_×  = ∂Xc/∂ω for left perturbation
    return Mat3{{0, x[2], -x[1], -x[2], 0, x[0], x[1], -x[0], 0}};
}

Mat3 so3_exp(const Vec3 &w) {  // rotation from an axis-angle (rotation vector)
    const double th = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (th < 1e-12) {  // I + [w]_× to first order
        Mat3 R = mat3_identity();
        R(0, 1) = -w[2];
        R(0, 2) = w[1];
        R(1, 0) = w[2];
        R(1, 2) = -w[0];
        R(2, 0) = -w[1];
        R(2, 1) = w[0];
        return R;
    }
    return rodrigues({w[0] / th, w[1] / th, w[2] / th}, th);
}

double reproj_rms(const double *Xw,
                  const double *px,
                  int N,
                  const Mat3 &K,
                  const Mat3 &R,
                  const Vec3 &t) {
    const double fu = K(0, 0), fv = K(1, 1), cu = K(0, 2), cv = K(1, 2);
    double s = 0;
    int m = 0;
    for (int i = 0; i < N; i++) {
        Vec3 p = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
        Vec3 Xc = mat3_apply(R, p);
        Xc[0] += t[0];
        Xc[1] += t[1];
        Xc[2] += t[2];
        if (Xc[2] <= 1e-6) {
            s += 1e6;
            m++;
            continue;
        }
        const double dx = fu * Xc[0] / Xc[2] + cu - px[2 * i];
        const double dy = fv * Xc[1] / Xc[2] + cv - px[2 * i + 1];
        s += dx * dx + dy * dy;
        m++;
    }
    return std::sqrt(s / std::max(m, 1));
}

// Recover camera-frame control points from the null-space vector x (12), fix
// the global sign by cheirality (scene in front), and get world2cam (R,t) from
// the rigid fit between the 4 world and 4 camera control points.
void recover_Rt(const double *x12,
                const double cw[4][3],
                const double *alpha,
                int N,
                Mat3 &R,
                Vec3 &t) {
    double cc[12];
    for (int i = 0; i < 12; i++) cc[i] = x12[i];
    // mean camera-frame depth over the points (pc = sum_j alpha_ij cc_j)
    double meanz = 0;
    for (int i = 0; i < N; i++) {
        double z = 0;
        for (int j = 0; j < 4; j++) z += alpha[4 * i + j] * cc[3 * j + 2];
        meanz += z;
    }
    if (meanz < 0)
        for (int i = 0; i < 12; i++) cc[i] = -cc[i];
    double cwf[12];
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 3; k++) cwf[3 * j + k] = cw[j][k];
    Sim3 T = fit_similarity(cwf, cc, 4,
                            /*with_scale=*/false);  // rigid world->camera
    R = T.R;
    t = T.t;
}

// EPnP core: returns the best (R,t) world2cam over N=1,2,3 by reprojection
// error.
void epnp(const double *Xw,
          const double *px,
          int N,
          const Mat3 &K,
          Mat3 &Rout,
          Vec3 &tout) {
    // 1. control points: centroid + principal axes (sqrt-eigenvalue scaled).
    double c0[3] = {0, 0, 0};
    for (int i = 0; i < N; i++) {
        c0[0] += Xw[3 * i];
        c0[1] += Xw[3 * i + 1];
        c0[2] += Xw[3 * i + 2];
    }
    c0[0] /= N;
    c0[1] /= N;
    c0[2] /= N;
    double Cov[9] = {0};
    for (int i = 0; i < N; i++) {
        const double d[3] = {Xw[3 * i] - c0[0], Xw[3 * i + 1] - c0[1],
                             Xw[3 * i + 2] - c0[2]};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) Cov[r * 3 + c] += d[r] * d[c];
    }
    for (int i = 0; i < 9; i++) Cov[i] /= N;
    double ev[3], evec[9];
    fsla::jacobi_eigh(Cov, 3, ev, evec);
    double lam_max = std::max({ev[0], ev[1], ev[2], 1e-12});
    double cw[4][3];
    for (int k = 0; k < 3; k++) cw[0][k] = c0[k];
    for (int j = 0; j < 3; j++) {
        // floor the axis length so a near-planar scene keeps 4
        // affinely-independent control points (an invertible barycentric
        // matrix).
        const double s = std::sqrt(std::max(ev[j], 1e-6 * lam_max));
        for (int k = 0; k < 3; k++) cw[j + 1][k] = c0[k] + s * evec[k * 3 + j];
    }

    // 2. barycentric coordinates: alpha_i = C^{-1} [p_i; 1], C =
    // [[cw^T];[1...]].
    double C[16];
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 3; k++) C[k * 4 + j] = cw[j][k];
        C[3 * 4 + j] = 1.0;
    }
    double Cinv[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    solve_lin(C, 4, Cinv, 4);  // Cinv now holds C^{-1}
    std::vector<double> alpha((size_t)4 * N);
    for (int i = 0; i < N; i++) {
        const double p[4] = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2], 1.0};
        for (int j = 0; j < 4; j++) {
            double a = 0;
            for (int k = 0; k < 4; k++) a += Cinv[j * 4 + k] * p[k];
            alpha[4 * i + j] = a;
        }
    }

    // 3. M (2n x 12) -> MtM (12 x 12).
    const double fu = K(0, 0), fv = K(1, 1), cu = K(0, 2), cv = K(1, 2);
    double MtM[144] = {0};
    for (int i = 0; i < N; i++) {
        double r1[12] = {0}, r2[12] = {0};
        const double uu = px[2 * i], vv = px[2 * i + 1];
        for (int j = 0; j < 4; j++) {
            const double a = alpha[4 * i + j];
            r1[3 * j + 0] = a * fu;
            r1[3 * j + 2] = a * (cu - uu);
            r2[3 * j + 1] = a * fv;
            r2[3 * j + 2] = a * (cv - vv);
        }
        for (int r = 0; r < 12; r++)
            for (int c = 0; c < 12; c++)
                MtM[r * 12 + c] += r1[r] * r1[c] + r2[r] * r2[c];
    }
    double mev[12], mevec[144];
    fsla::jacobi_eigh(MtM, 12, mev, mevec);
    int ord[12];
    for (int i = 0; i < 12; i++) ord[i] = i;
    std::sort(ord, ord + 12, [&](int a, int b) { return mev[a] < mev[b]; });
    double V[3][12];  // the 3 smallest-eigenvalue vectors
    for (int n = 0; n < 3; n++)
        for (int i = 0; i < 12; i++) V[n][i] = mevec[i * 12 + ord[n]];

    // control-point pairs and their world distances
    const int pr[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
    double dw2[6];
    for (int p = 0; p < 6; p++) {
        const int i = pr[p][0], j = pr[p][1];
        double d = 0;
        for (int k = 0; k < 3; k++) {
            const double e = cw[i][k] - cw[j][k];
            d += e * e;
        }
        dw2[p] = d;
    }
    auto blkdiff = [&](int n, int p, double out[3]) {
        const int i = pr[p][0], j = pr[p][1];
        for (int k = 0; k < 3; k++) out[k] = V[n][3 * i + k] - V[n][3 * j + k];
    };
    auto dot3 = [](const double a[3], const double b[3]) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };

    Mat3 bestR{};
    Vec3 bestT{};
    double best_err = 1e300;
    bool have = false;
    auto consider = [&](const double x12[12]) {
        Mat3 R;
        Vec3 t;
        recover_Rt(x12, cw, alpha.data(), N, R, t);
        const double e = reproj_rms(Xw, px, N, K, R, t);
        if (e < best_err) {
            best_err = e;
            bestR = R;
            bestT = t;
            have = true;
        }
    };

    // N=1: x = beta * v0, beta from the 6 distance constraints (least squares).
    {
        double num = 0, den = 0;
        for (int p = 0; p < 6; p++) {
            double d0[3];
            blkdiff(0, p, d0);
            const double nd = std::sqrt(dot3(d0, d0));
            num += nd * std::sqrt(dw2[p]);
            den += nd * nd;
        }
        const double beta = (den > 0) ? num / den : 0.0;
        double x[12];
        for (int i = 0; i < 12; i++) x[i] = beta * V[0][i];
        consider(x);
    }
    // N=2: x = b1 v0 + b2 v1; distance constraints linear in (b1^2,b1b2,b2^2).
    {
        double L[18], rho[6];
        for (int p = 0; p < 6; p++) {
            double d0[3], d1[3];
            blkdiff(0, p, d0);
            blkdiff(1, p, d1);
            L[p * 3 + 0] = dot3(d0, d0);
            L[p * 3 + 1] = 2 * dot3(d0, d1);
            L[p * 3 + 2] = dot3(d1, d1);
            rho[p] = dw2[p];
        }
        double G[9] = {0}, g[3] = {0};  // normal equations (3x3)
        for (int p = 0; p < 6; p++) {
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++)
                    G[r * 3 + c] += L[p * 3 + r] * L[p * 3 + c];
                g[r] += L[p * 3 + r] * rho[p];
            }
        }
        solve_lin(G, 3, g, 1);  // g = [b11,b12,b22]
        const double b1 = std::sqrt(std::fabs(g[0]));
        double b2 = std::sqrt(std::fabs(g[2]));
        if (g[1] < 0) b2 = -b2;
        double x[12];
        for (int i = 0; i < 12; i++) x[i] = b1 * V[0][i] + b2 * V[1][i];
        consider(x);
    }
    // N=3: x = b1 v0 + b2 v1 + b3 v2; 6 constraints in 6 quadratic unknowns.
    {
        double L[36], rho[6];
        for (int p = 0; p < 6; p++) {
            double d0[3], d1[3], d2[3];
            blkdiff(0, p, d0);
            blkdiff(1, p, d1);
            blkdiff(2, p, d2);
            L[p * 6 + 0] = dot3(d0, d0);
            L[p * 6 + 1] = 2 * dot3(d0, d1);
            L[p * 6 + 2] = 2 * dot3(d0, d2);
            L[p * 6 + 3] = dot3(d1, d1);
            L[p * 6 + 4] = 2 * dot3(d1, d2);
            L[p * 6 + 5] = dot3(d2, d2);
            rho[p] = dw2[p];
        }
        double Lc[36], rc[6];
        for (int i = 0; i < 36; i++) Lc[i] = L[i];
        for (int i = 0; i < 6; i++) rc[i] = rho[i];
        solve_lin(Lc, 6, rc, 1);  // rc = [b11,b12,b13,b22,b23,b33]
        const double b1 = std::sqrt(std::fabs(rc[0]));
        const double b2 = (b1 > 1e-12) ? rc[1] / b1 : 0.0;
        const double b3 = (b1 > 1e-12) ? rc[2] / b1 : 0.0;
        double x[12];
        for (int i = 0; i < 12; i++)
            x[i] = b1 * V[0][i] + b2 * V[1][i] + b3 * V[2][i];
        consider(x);
    }

    if (!have) {
        Rout = mat3_identity();
        tout = {0, 0, 0};
        return;
    }
    Rout = bestR;
    tout = bestT;
}

// Huber-robust Gauss-Newton on the reprojection residual; 6-DoF left
// perturbation (R <- exp(dw) R, t += dt). Refines an EPnP init to the
// reprojection minimum and downweights gross outliers, so the deterministic
// EPnP+GN matches cv2's RANSAC+SQPNP+refine without random sampling.
void gauss_newton_pnp(const double *Xw,
                      const double *px,
                      int N,
                      const Mat3 &K,
                      Mat3 &R,
                      Vec3 &t,
                      int iters,
                      double huber_px) {
    const double fu = K(0, 0), fv = K(1, 1), cu = K(0, 2), cv = K(1, 2);
    for (int it = 0; it < iters; it++) {
        double H[36] = {0}, b[6] = {0};
        for (int i = 0; i < N; i++) {
            Vec3 p = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
            Vec3 Xc = mat3_apply(R, p);
            Xc[0] += t[0];
            Xc[1] += t[1];
            Xc[2] += t[2];
            if (Xc[2] <= 1e-6) continue;
            const double iz = 1.0 / Xc[2];
            const double rx = fu * Xc[0] * iz + cu - px[2 * i];
            const double ry = fv * Xc[1] * iz + cv - px[2 * i + 1];
            const double rn = std::sqrt(rx * rx + ry * ry);
            const double w =
                    (rn <= huber_px || rn < 1e-12) ? 1.0 : huber_px / rn;
            // dproj/dXc (2x3)
            const double Jp[2][3] = {{fu * iz, 0, -fu * Xc[0] * iz * iz},
                                     {0, fv * iz, -fv * Xc[1] * iz * iz}};
            // dXc/d(delta) (3x6) = [ -[Xc]_x | I ]
            const Mat3 S = skew_neg(Xc);
            double Jd[3][6];
            for (int r = 0; r < 3; r++) {
                Jd[r][0] = S(r, 0);
                Jd[r][1] = S(r, 1);
                Jd[r][2] = S(r, 2);
                Jd[r][3] = (r == 0);
                Jd[r][4] = (r == 1);
                Jd[r][5] = (r == 2);
            }
            double J[2][6];
            for (int a = 0; a < 2; a++)
                for (int c = 0; c < 6; c++) {
                    double s = 0;
                    for (int k = 0; k < 3; k++) s += Jp[a][k] * Jd[k][c];
                    J[a][c] = s;
                }
            const double r2[2] = {rx, ry};
            for (int a = 0; a < 2; a++) {
                for (int c = 0; c < 6; c++) {
                    b[c] += w * J[a][c] * r2[a];
                    for (int d = 0; d < 6; d++)
                        H[c * 6 + d] += w * J[a][c] * J[a][d];
                }
            }
        }
        double rhs[6];
        for (int i = 0; i < 6; i++) rhs[i] = -b[i];
        solve_lin(H, 6, rhs, 1);  // solve H delta = -b
        Vec3 dw = {rhs[0], rhs[1], rhs[2]};
        R = mat3_mul(so3_exp(dw), R);
        t[0] += rhs[3];
        t[1] += rhs[4];
        t[2] += rhs[5];
        if (std::fabs(rhs[0]) + std::fabs(rhs[1]) + std::fabs(rhs[2]) +
                    std::fabs(rhs[3]) + std::fabs(rhs[4]) + std::fabs(rhs[5]) <
            1e-12)
            break;
    }
}

}  // namespace

Mat4 solve_pnp_epnp(const double *Xw,
                    const double *pixels,
                    int N,
                    const Mat3 &K) {
    Mat3 R;
    Vec3 t;
    epnp(Xw, pixels, N, K, R, t);
    Mat4 w2c = mat4_identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) w2c(i, j) = R(i, j);
        w2c(i, 3) = t[i];
    }
    return inv_rigid4(w2c);
}

Mat4 solve_pnp(const double *Xw,
               const double *pixels,
               int N,
               const Mat3 &K,
               std::vector<char> &inliers,
               double thresh_px,
               int gn_iters) {
    Mat3 R;
    Vec3 t;
    epnp(Xw, pixels, N, K, R, t);
    gauss_newton_pnp(Xw, pixels, N, K, R, t, gn_iters, thresh_px);

    inliers.assign(N, 0);
    const double fu = K(0, 0), fv = K(1, 1), cu = K(0, 2), cv = K(1, 2);
    for (int i = 0; i < N; i++) {
        Vec3 p = {Xw[3 * i], Xw[3 * i + 1], Xw[3 * i + 2]};
        Vec3 Xc = mat3_apply(R, p);
        Xc[0] += t[0];
        Xc[1] += t[1];
        Xc[2] += t[2];
        if (Xc[2] <= 1e-6) continue;
        const double dx = fu * Xc[0] / Xc[2] + cu - pixels[2 * i];
        const double dy = fv * Xc[1] / Xc[2] + cv - pixels[2 * i + 1];
        if (std::sqrt(dx * dx + dy * dy) < thresh_px) inliers[i] = 1;
    }
    Mat4 w2c = mat4_identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) w2c(i, j) = R(i, j);
        w2c(i, 3) = t[i];
    }
    return inv_rigid4(w2c);
}

// ---- estimate_poses orchestration -----------------------------------------

PoseResult estimate_poses(const std::vector<const float *> &points,
                          const std::vector<const float *> &opacities,
                          int H,
                          int W,
                          double opacity_threshold,
                          double focal,
                          int pnp_iter,
                          bool normalize,
                          uint64_t seed) {
    (void)pnp_iter;
    (void)seed;  // robust PnP is deterministic (EPnP+GN)
    const int Nv = (int)points.size();
    const int P = H * W;
    const double ppx = W / 2.0, ppy = H / 2.0;
    std::vector<double> grid;
    pixel_grid(H, W, grid);

    // per-view opacity masks
    std::vector<std::vector<char>> masks(Nv);
    for (int v = 0; v < Nv; v++) {
        masks[v].assign(P, 1);
        if (v < (int)opacities.size() && opacities[v]) {
            for (int i = 0; i < P; i++)
                masks[v][i] = (opacities[v][i] > opacity_threshold) ? 1 : 0;
        }
    }

    // focal: view 0 only, ALL pixels (use_first_focal, scene recipe)
    if (focal <= 0.0) {
        std::vector<double> pts(3 * P), pix(2 * P);
        for (int i = 0; i < P; i++) {
            pts[3 * i] = points[0][3 * i];
            pts[3 * i + 1] = points[0][3 * i + 1];
            pts[3 * i + 2] = points[0][3 * i + 2];
            pix[2 * i] = grid[2 * i];
            pix[2 * i + 1] = grid[2 * i + 1];
        }
        focal = estimate_focal(pts.data(), pix.data(), P, ppx, ppy);
    }
    Mat3 K = mat3_identity();
    K(0, 0) = focal;
    K(1, 1) = focal;
    K(0, 2) = ppx;
    K(1, 2) = ppy;

    PoseResult res;
    res.focal = focal;
    res.scale = 1.0;
    for (int v = 0; v < Nv; v++) {
        std::vector<double> Xw, px;
        for (int i = 0; i < P; i++)
            if (masks[v][i]) {
                Xw.push_back(points[v][3 * i]);
                Xw.push_back(points[v][3 * i + 1]);
                Xw.push_back(points[v][3 * i + 2]);
                px.push_back(grid[2 * i]);
                px.push_back(grid[2 * i + 1]);
            }
        std::vector<char> inl;
        Mat4 c2w = solve_pnp(Xw.data(), px.data(), (int)(Xw.size() / 3), K, inl,
                             5.0, 10);
        res.cam2world.push_back(c2w);
    }

    if (normalize && Nv >= 2) {
        Vec3 t0 = {res.cam2world.front()(0, 3), res.cam2world.front()(1, 3),
                   res.cam2world.front()(2, 3)};
        Vec3 tl = {res.cam2world.back()(0, 3), res.cam2world.back()(1, 3),
                   res.cam2world.back()(2, 3)};
        const double baseline = std::sqrt((tl[0] - t0[0]) * (tl[0] - t0[0]) +
                                          (tl[1] - t0[1]) * (tl[1] - t0[1]) +
                                          (tl[2] - t0[2]) * (tl[2] - t0[2])) +
                                1e-2;
        res.scale = 1.0 / baseline;
        for (auto &c : res.cam2world)
            for (int i = 0; i < 3; i++) c(i, 3) *= res.scale;
    }
    return res;
}

// ---- after-inference parallax ---------------------------------------------

Parallax parallax_stats(const Vec3 &C0,
                        const Vec3 &C1,
                        const Vec3 &axis0,
                        const float *pts,
                        const float *mask,
                        int N,
                        double mask_thr) {
    constexpr double PI = 3.14159265358979323846;
    auto sub = [](const Vec3 &a, const Vec3 &b) -> Vec3 {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto dot = [](const Vec3 &a, const Vec3 &b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    auto nrm = [&](const Vec3 &a) { return std::sqrt(dot(a, a)); };

    Parallax px{};
    // unit optical axis of view 0
    Vec3 ax = axis0;
    const double an = nrm(ax);
    if (an > 1e-12) {
        ax[0] /= an;
        ax[1] /= an;
        ax[2] /= an;
    }

    // baseline split: forward (dolly, no parallax) vs lateral (strafe,
    // parallax)
    const Vec3 b = sub(C1, C0);
    px.baseline = nrm(b);
    const double fwd = dot(b, ax);
    const Vec3 blat = {b[0] - fwd * ax[0], b[1] - fwd * ax[1],
                       b[2] - fwd * ax[2]};
    px.lateral_angle_deg = std::atan2(nrm(blat), std::fabs(fwd)) * 180.0 / PI;

    // per confident point: depth from cam0 + triangulation angle (rays from
    // C0,C1)
    std::vector<double> depths, angles;
    depths.reserve(N);
    angles.reserve(N);
    for (int i = 0; i < N; i++) {
        if (mask && !(mask[i] > mask_thr)) continue;
        const Vec3 X = {(double)pts[3 * i], (double)pts[3 * i + 1],
                        (double)pts[3 * i + 2]};
        if (!std::isfinite(X[0]) || !std::isfinite(X[1]) ||
            !std::isfinite(X[2]))
            continue;
        const Vec3 r0 = sub(X, C0), r1 = sub(X, C1);
        const double n0 = nrm(r0), n1 = nrm(r1);
        if (n0 < 1e-9 || n1 < 1e-9) continue;
        const double depth = dot(r0, ax);
        if (depth > 0) depths.push_back(depth);
        double c = dot(r0, r1) / (n0 * n1);
        c = c < -1.0 ? -1.0 : (c > 1.0 ? 1.0 : c);
        angles.push_back(std::acos(c) * 180.0 / PI);
    }
    auto median = [](std::vector<double> &v) -> double {
        if (v.empty()) return 0.0;
        const size_t m = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + m, v.end());
        return v[m];
    };
    px.median_depth = median(depths);
    px.tri_angle_deg = median(angles);
    px.baseline_over_depth =
            (px.median_depth > 1e-9) ? px.baseline / px.median_depth : 0.0;
    px.n_points = (int)angles.size();
    return px;
}

Parallax pair_parallax(const std::vector<const float *> &points,
                       const std::vector<const float *> &opacities,
                       int H,
                       int W,
                       double opacity_threshold,
                       double focal) {
    Parallax px{};
    if (points.size() < 2) return px;
    PoseResult pr = estimate_poses(points, opacities, H, W, opacity_threshold,
                                   focal, 100, /*normalize=*/false, /*seed=*/0);
    if (pr.cam2world.size() < 2) return px;
    const Mat4 &c0 = pr.cam2world[0];
    const Mat4 &c1 = pr.cam2world[1];
    const Vec3 C0 = {c0(0, 3), c0(1, 3), c0(2, 3)};
    const Vec3 C1 = {c1(0, 3), c1(1, 3), c1(2, 3)};
    const Vec3 axis0 = {c0(0, 2), c0(1, 2),
                        c0(2, 2)};  // camera +z (optical axis) in world
    px = parallax_stats(C0, C1, axis0, points[0],
                        opacities.empty() ? nullptr : opacities[0], H * W,
                        opacity_threshold);
    px.focal = pr.focal;
    return px;
}

// ---- accumulation ---------------------------------------------------------

namespace {
inline float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}
const double SH_C0 = 0.28209479177387814;  // SH degree-0 basis (DC -> rgb)

// quaternion (w,x,y,z) from a rotation matrix (Shepperd's method)
std::array<double, 4> mat3_to_quat(const Mat3 &R) {
    const double tr = R(0, 0) + R(1, 1) + R(2, 2);
    double w, x, y, z;
    if (tr > 0) {
        double s = std::sqrt(tr + 1.0) * 2;
        w = 0.25 * s;
        x = (R(2, 1) - R(1, 2)) / s;
        y = (R(0, 2) - R(2, 0)) / s;
        z = (R(1, 0) - R(0, 1)) / s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2;
        w = (R(2, 1) - R(1, 2)) / s;
        x = 0.25 * s;
        y = (R(0, 1) + R(1, 0)) / s;
        z = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2;
        w = (R(0, 2) - R(2, 0)) / s;
        x = (R(0, 1) + R(1, 0)) / s;
        y = 0.25 * s;
        z = (R(1, 2) + R(2, 1)) / s;
    } else {
        double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2;
        w = (R(1, 0) - R(0, 1)) / s;
        x = (R(0, 2) + R(2, 0)) / s;
        y = (R(1, 2) + R(2, 1)) / s;
        z = 0.25 * s;
    }
    return {w, x, y, z};
}
// Hamilton product (w,x,y,z): compose rotation a then... q = a ⊗ b applies b
// first.
std::array<double, 4> quat_mul(const std::array<double, 4> &a,
                               const std::array<double, 4> &b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
std::array<double, 4> quat_normalize(std::array<double, 4> q) {
    const double n =
            std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 1e-12)
        for (auto &c : q) c /= n;
    else
        q = {1, 0, 0, 0};
    return q;
}
}  // namespace

Accumulator::Accumulator(int H,
                         int W,
                         double opacity_threshold,
                         double ransac_thresh_frac,
                         int ransac_iters,
                         uint64_t seed)
    : H_(H),
      W_(W),
      thr_(opacity_threshold),
      rthr_(ransac_thresh_frac),
      riters_(ransac_iters),
      seed_(seed),
      T_(sim_identity()),
      final_cam_(mat4_identity()) {}

void Accumulator::add_view(const float *pts,
                           const float *op,
                           const float *rgb,
                           const float *scl,
                           const float *rot,
                           const Sim3 &T,
                           int frame) {
    const int P = H_ * W_;
    // A similarity x -> s*(R@x)+t scales each gaussian's covariance by s^2 and
    // rotates it by R: so the world-frame axis lengths are s*scale and the
    // world rotation is qT ⊗ q_local. Compute qT (from T.R) once for the whole
    // view.
    const std::array<double, 4> qT = mat3_to_quat(T.R);
    for (int i = 0; i < P; i++) {
        if (op[i] <= thr_) continue;
        Vec3 x = {pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]};
        Vec3 w = sim_apply(T, x);
        AccumPoint p;
        p.x = (float)w[0];
        p.y = (float)w[1];
        p.z = (float)w[2];
        p.r = rgb[3 * i];
        p.g = rgb[3 * i + 1];
        p.b = rgb[3 * i + 2];
        p.opacity = op[i];
        p.sx = (float)(T.s * scl[3 * i]);
        p.sy = (float)(T.s * scl[3 * i + 1]);
        p.sz = (float)(T.s * scl[3 * i + 2]);
        std::array<double, 4> ql = quat_normalize(
                {rot[4 * i], rot[4 * i + 1], rot[4 * i + 2], rot[4 * i + 3]});
        std::array<double, 4> qw = quat_normalize(quat_mul(qT, ql));
        p.qw = (float)qw[0];
        p.qx = (float)qw[1];
        p.qy = (float)qw[2];
        p.qz = (float)qw[3];
        p.frame = frame;
        cloud_.push_back(p);
    }
}

ChainLink Accumulator::add_pair(const float *g, int gc, double focal) {
    const int P = H_ * W_;
    // de-interleave the two views: points (ch 0:3), opacity (ch 15, already
    // sigmoid-activated), rgb = clip(SH_DC * C0 + 0.5, 0, 1) (ch 3:6).
    std::vector<float> pts0(3 * P), pts1(3 * P), op0(P), op1(P), rgb0(3 * P),
            rgb1(3 * P);
    std::vector<float> scl0(3 * P), scl1(3 * P), rot0(4 * P), rot1(4 * P);
    for (int i = 0; i < P; i++) {
        const float *a0 = g + (size_t)i * gc;
        const float *a1 = g + (size_t)(P + i) * gc;
        for (int c = 0; c < 3; c++) {
            pts0[3 * i + c] = a0[c];
            pts1[3 * i + c] = a1[c];
        }
        op0[i] = a0[15];
        op1[i] = a1[15];
        for (int c = 0; c < 3; c++) {
            rgb0[3 * i + c] = clamp01((float)(a0[3 + c] * SH_C0 + 0.5));
            rgb1[3 * i + c] = clamp01((float)(a1[3 + c] * SH_C0 + 0.5));
        }
        // gaussian shape: scale ch16:19, rotation quaternion (w,x,y,z) ch19:23
        for (int c = 0; c < 3; c++) {
            scl0[3 * i + c] = a0[16 + c];
            scl1[3 * i + c] = a1[16 + c];
        }
        for (int c = 0; c < 4; c++) {
            rot0[4 * i + c] = a0[19 + c];
            rot1[4 * i + c] = a1[19 + c];
        }
    }

    // recover this pair's cameras (view-0 frame, use_first_focal)
    std::vector<const float *> pts = {pts0.data(), pts1.data()};
    std::vector<const float *> ops = {op0.data(), op1.data()};
    PoseResult pr =
            estimate_poses(pts, ops, H_, W_, thr_, focal, 100, false, seed_);
    const Mat4 c2w0 = pr.cam2world[0], c2w1 = pr.cam2world[1];

    ChainLink link{};
    if (!have_prev_) {
        T_ = sim_identity();
        link.sim = sim_identity();
        link.global = T_;
        link.scale = 1.0;
        link.inlier_frac = 1.0;
        link.valid_frac = 1.0;
        link.resid_frac = 0.0;
        add_view(pts0.data(), op0.data(), rgb0.data(), scl0.data(), rot0.data(),
                 T_, next_frame_++);  // f_0
        add_view(pts1.data(), op1.data(), rgb1.data(), scl1.data(), rot1.data(),
                 T_, next_frame_++);  // f_1
    } else {
        // fit Sim(3) from this run's view 0 (the shared frame, run-k coords) to
        // the previous run's view 1 (same frame, run-(k-1) coords). Mask =
        // valid in both.
        std::vector<double> XA, XB;
        for (int i = 0; i < P; i++)
            if (op0[i] > thr_ && prev_mask1_[i]) {
                XA.push_back(pts0[3 * i]);
                XA.push_back(pts0[3 * i + 1]);
                XA.push_back(pts0[3 * i + 2]);
                XB.push_back(prev_pts1_[3 * i]);
                XB.push_back(prev_pts1_[3 * i + 1]);
                XB.push_back(prev_pts1_[3 * i + 2]);
            }
        const int n = (int)(XA.size() / 3);
        double myB[3];
        mean3(XB.data(), n, myB);
        std::vector<double> ycB(3 * n);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 3; c++) ycB[3 * i + c] = XB[3 * i + c] - myB[c];
        const double scene = rms3(ycB.data(), n);

        std::vector<char> inl;
        Sim3 S = fit_similarity_ransac(XA.data(), XB.data(), n, rthr_ * scene,
                                       riters_, inl, true, seed_);
        T_ = sim_compose(T_, S);  // run k -> global

        int n_inl = 0;
        std::vector<double> rA, rB;
        for (int i = 0; i < n; i++)
            if (inl[i]) {
                n_inl++;
                Vec3 a = {XA[3 * i], XA[3 * i + 1], XA[3 * i + 2]};
                Vec3 sa = sim_apply(S, a);
                for (int c = 0; c < 3; c++) {
                    rA.push_back(sa[c] - XB[3 * i + c]);
                }
                (void)rB;
            }
        link.sim = S;
        link.global = T_;
        link.scale = S.s;
        link.inlier_frac = (n > 0) ? (double)n_inl / n : 0.0;
        link.valid_frac = (double)n / P;
        link.resid_frac =
                (scene > 0 && n_inl > 0) ? rms3(rA.data(), n_inl) / scene : 0.0;

        add_view(pts1.data(), op1.data(), rgb1.data(), scl1.data(), rot1.data(),
                 T_, next_frame_++);  // f_{k+1}
    }

    cams_.push_back(
            mat4_mul(sim_matrix(T_), c2w0));      // frame f_k camera (view 0)
    final_cam_ = mat4_mul(sim_matrix(T_), c2w1);  // latest view-1 camera

    // this run's view 1 becomes the next run's shared frame
    prev_pts1_.assign(pts1.begin(), pts1.end());
    prev_mask1_.assign(P, 0);
    for (int i = 0; i < P; i++) prev_mask1_[i] = (op1[i] > thr_) ? 1 : 0;
    have_prev_ = true;
    links_.push_back(link);
    return link;
}

double Accumulator::refine(double voxel_frac, int iters, double alpha) {
    return consensus_refine(cloud_, voxel_frac, iters, alpha);
}

std::vector<Mat4> Accumulator::camera_path() const {
    std::vector<Mat4> path = cams_;
    if (!cams_.empty()) path.push_back(final_cam_);
    return path;
}

// ---- hierarchical tree accumulation ---------------------------------------

// One gaussian (gc-channel engine layout) -> AccumPoint at identity (leaf
// frame).
static AccumPoint tree_leaf_point(const float *a, int frame) {
    AccumPoint p;
    p.x = a[0];
    p.y = a[1];
    p.z = a[2];
    p.r = clamp01((float)(a[3] * SH_C0 + 0.5));
    p.g = clamp01((float)(a[4] * SH_C0 + 0.5));
    p.b = clamp01((float)(a[5] * SH_C0 + 0.5));
    p.opacity = a[15];
    p.sx = a[16];
    p.sy = a[17];
    p.sz = a[18];
    std::array<double, 4> q = quat_normalize({a[19], a[20], a[21], a[22]});
    p.qw = (float)q[0];
    p.qx = (float)q[1];
    p.qy = (float)q[2];
    p.qz = (float)q[3];
    p.frame = frame;
    return p;
}

// Apply a Sim(3) to a whole cloud (position s*(R@x)+t, scale *s, orientation
// qS⊗q).
static void tree_apply_sim_cloud(std::vector<AccumPoint> &c, const Sim3 &S) {
    const std::array<double, 4> qS = mat3_to_quat(S.R);
    for (auto &p : c) {
        Vec3 w = sim_apply(S, {p.x, p.y, p.z});
        p.x = (float)w[0];
        p.y = (float)w[1];
        p.z = (float)w[2];
        p.sx = (float)(S.s * p.sx);
        p.sy = (float)(S.s * p.sy);
        p.sz = (float)(S.s * p.sz);
        std::array<double, 4> q =
                quat_normalize(quat_mul(qS, {p.qw, p.qx, p.qy, p.qz}));
        p.qw = (float)q[0];
        p.qx = (float)q[1];
        p.qy = (float)q[2];
        p.qz = (float)q[3];
    }
}
static void tree_apply_sim_pts(std::vector<float> &pts, const Sim3 &S) {
    for (size_t i = 0; i + 3 <= pts.size(); i += 3) {
        Vec3 w = sim_apply(S, {pts[i], pts[i + 1], pts[i + 2]});
        pts[i] = (float)w[0];
        pts[i + 1] = (float)w[1];
        pts[i + 2] = (float)w[2];
    }
}

// Robustly register one band of shared-frame correspondences (XA -> XB, both
// flat 3*n): RANSAC similarity at a scene-relative threshold, plus the inlier
// residual. The single registration primitive for both tree merges (caller
// derives the inlier/valid fractions, which it normalises differently per
// tree).
struct BandFit {
    Sim3 S;
    int n_inl;
    double scene;
    double resid;
};
static BandFit fit_band_sim(const std::vector<double> &XA,
                            const std::vector<double> &XB,
                            double rthr,
                            int riters,
                            uint64_t seed) {
    const int n = (int)(XA.size() / 3);
    BandFit f{sim_identity(), 0, 0.0, 0.0};
    if (n < 3) return f;
    double mB[3];
    mean3(XB.data(), n, mB);
    std::vector<double> yc(3 * n);
    for (int k = 0; k < n; k++)
        for (int c = 0; c < 3; c++) yc[3 * k + c] = XB[3 * k + c] - mB[c];
    f.scene = rms3(yc.data(), n);
    std::vector<char> inl;
    f.S = fit_similarity_ransac(XA.data(), XB.data(), n, rthr * f.scene, riters,
                                inl, true, seed);
    std::vector<double> rA;
    for (int k = 0; k < n; k++)
        if (inl[k]) {
            f.n_inl++;
            Vec3 sa = sim_apply(f.S, {XA[3 * k], XA[3 * k + 1], XA[3 * k + 2]});
            for (int c = 0; c < 3; c++) rA.push_back(sa[c] - XB[3 * k + c]);
        }
    f.resid = (f.scene > 0 && f.n_inl > 0) ? rms3(rA.data(), f.n_inl) / f.scene
                                           : 0.0;
    return f;
}

// ---- multi-frame-overlap tree ---------------------------------------------

// A submap covering a contiguous global frame range, carrying the per-pixel
// point arrays of every frame it spans (in the submap's local frame) so a
// neighbour can align on the whole band of frames they share, not just one.
struct ONode {
    std::vector<AccumPoint> cloud;
    int lo, hi;                            // inclusive global frame range
    std::vector<std::vector<float>> fpts;  // [f-lo] -> P*3 per-pixel points
    std::vector<std::vector<char>> fmsk;   // [f-lo] -> P opacity mask
};

// One pair (frames a, a+1) as a 2-frame submap (identity local frame).
static ONode opair(const float *g, int a, int P, int gc, double thr) {
    ONode nd;
    nd.lo = a;
    nd.hi = a + 1;
    nd.fpts.assign(2, {});
    nd.fmsk.assign(2, {});
    nd.fpts[0].resize(3 * P);
    nd.fpts[1].resize(3 * P);
    nd.fmsk[0].assign(P, 0);
    nd.fmsk[1].assign(P, 0);
    nd.cloud.reserve(2 * P);
    for (int i = 0; i < P; i++) {
        const float *a0 = g + (size_t)i * gc;
        const float *a1 = g + (size_t)(P + i) * gc;
        for (int c = 0; c < 3; c++) {
            nd.fpts[0][3 * i + c] = a0[c];
            nd.fpts[1][3 * i + c] = a1[c];
        }
        nd.fmsk[0][i] = (a0[15] > thr) ? 1 : 0;
        nd.fmsk[1][i] = (a1[15] > thr) ? 1 : 0;
        if (a0[15] > thr) nd.cloud.push_back(tree_leaf_point(a0, a));
        if (a1[15] > thr) nd.cloud.push_back(tree_leaf_point(a1, a + 1));
    }
    return nd;
}

// Merge R into L's frame, fitting one Sim(3) over EVERY frame they share.
// Returns the combined submap; optionally records the merge diagnostics.
static ONode omerge(ONode &L,
                    ONode &R,
                    int P,
                    double rthr,
                    int riters,
                    uint64_t seed,
                    std::vector<TreeMerge> *merges,
                    int level) {
    const int so = std::max(L.lo, R.lo),
              sh = std::min(L.hi, R.hi);  // shared frame range
    std::vector<double> XA, XB;
    for (int f = so; f <= sh; f++) {
        const std::vector<float> &ra = R.fpts[f - R.lo];
        const std::vector<char> &rm = R.fmsk[f - R.lo];
        const std::vector<float> &la = L.fpts[f - L.lo];
        const std::vector<char> &lm = L.fmsk[f - L.lo];
        for (int i = 0; i < P; i++)
            if (rm[i] && lm[i]) {
                XA.push_back(ra[3 * i]);
                XA.push_back(ra[3 * i + 1]);
                XA.push_back(ra[3 * i + 2]);
                XB.push_back(la[3 * i]);
                XB.push_back(la[3 * i + 1]);
                XB.push_back(la[3 * i + 2]);
            }
    }
    const int n = (int)(XA.size() / 3);
    const int shared = sh - so + 1;
    BandFit fit = fit_band_sim(XA, XB, rthr, riters, seed);
    const Sim3 &S = fit.S;
    if (merges)
        merges->push_back(
                TreeMerge{level, std::min(L.lo, R.lo), std::max(L.hi, R.hi),
                          shared, S.s, (n > 0) ? (double)fit.n_inl / n : 0.0,
                          (double)n / ((double)shared * P), fit.resid});

    tree_apply_sim_cloud(R.cloud, S);
    for (auto &a : R.fpts) tree_apply_sim_pts(a, S);
    ONode m;
    m.lo = std::min(L.lo, R.lo);
    m.hi = std::max(L.hi, R.hi);
    const int F = m.hi - m.lo + 1;
    m.fpts.assign(F, {});
    m.fmsk.assign(F, {});
    for (int f = L.lo; f <= L.hi; f++) {
        m.fpts[f - m.lo] = std::move(L.fpts[f - L.lo]);
        m.fmsk[f - m.lo] = std::move(L.fmsk[f - L.lo]);
    }
    for (int f = R.lo; f <= R.hi; f++)
        if (m.fpts[f - m.lo].empty()) {
            m.fpts[f - m.lo] = std::move(R.fpts[f - R.lo]);
            m.fmsk[f - m.lo] = std::move(R.fmsk[f - R.lo]);
        }
    m.cloud = std::move(L.cloud);
    m.cloud.insert(m.cloud.end(), R.cloud.begin(), R.cloud.end());
    return m;
}

std::vector<AccumPoint> tree_accumulate_overlap(
        const std::vector<const float *> &pairs,
        int H,
        int W,
        int gc,
        double thr,
        int block,
        int overlap,
        double rthr,
        int riters,
        uint64_t seed,
        std::vector<TreeMerge> *merges,
        int max_levels,
        double layout_spacing,
        int *n_nodes_out,
        int per_node_cap) {
    const int P = H * W;
    const int M = (int)pairs.size();
    if (n_nodes_out) *n_nodes_out = 0;
    if (M < 1) return {};
    const int nframes = M + 1;
    overlap = std::max(1, overlap);
    block = std::max(block, overlap + 1);
    block = std::min(block, nframes);
    overlap = std::min(overlap, block - 1);
    const int step = std::max(1, block - overlap);

    // overlapping base submaps: each chains `block-1` consecutive pairs
    // (single-frame internal links), adjacent submaps share `overlap` frames.
    std::vector<ONode> nodes;
    for (int s = 0; s < M; s += step) {
        const int w = std::min(block, nframes - s);  // frames in this submap
        if (w < 2) break;
        ONode nd = opair(pairs[s], s, P, gc, thr);  // frames s, s+1
        for (int p = s + 1; p <= s + w - 2;
             p++) {  // fold the rest of the block
            ONode nx = opair(pairs[p], p, P, gc, thr);
            nd = omerge(nd, nx, P, rthr, riters, seed, nullptr, -1);
        }
        nodes.push_back(std::move(nd));
        if (s + block >= nframes) break;  // last submap reached the end
    }

    // median base-submap extent — the auto layout pitch for side-by-side
    // stages. Only the auto-layout (layout_spacing<0, staged demo) path reads
    // it, so the full-merge callers skip these two passes over every submap.
    double leaf_extent = 1.0;
    if (layout_spacing < 0.0) {
        std::vector<double> ex;
        for (auto &nd : nodes) {
            double m[3] = {0, 0, 0};
            int64_t nf = 0;
            for (auto &p : nd.cloud)
                if (std::isfinite(p.x) && std::isfinite(p.y) &&
                    std::isfinite(p.z)) {
                    m[0] += p.x;
                    m[1] += p.y;
                    m[2] += p.z;
                    nf++;
                }
            if (nf == 0) continue;
            for (int c = 0; c < 3; c++) m[c] /= nf;
            double ss = 0;
            for (auto &p : nd.cloud)
                if (std::isfinite(p.x) && std::isfinite(p.y) &&
                    std::isfinite(p.z)) {
                    double dx = p.x - m[0], dy = p.y - m[1], dz = p.z - m[2];
                    ss += dx * dx + dy * dy + dz * dz;
                }
            ex.push_back(std::sqrt(ss / nf));
        }
        if (!ex.empty()) {
            std::sort(ex.begin(), ex.end());
            leaf_extent = ex[ex.size() / 2];
        }
    }

    // balanced tree-merge of submaps; each merge fits over the overlap band.
    // Stop after max_levels rounds (-1 = fully to the root) for the staged
    // demo.
    int level = 0;
    while (nodes.size() > 1 && (max_levels < 0 || level < max_levels)) {
        std::vector<ONode> nxt;
        nxt.reserve(nodes.size() / 2 + 1);
        for (size_t i = 0; i + 1 < nodes.size(); i += 2)
            nxt.push_back(omerge(nodes[i], nodes[i + 1], P, rthr, riters, seed,
                                 merges, level));
        if (nodes.size() % 2 == 1) nxt.push_back(std::move(nodes.back()));
        nodes = std::move(nxt);
        level++;
    }
    if (n_nodes_out) *n_nodes_out = (int)nodes.size();
    if (nodes.empty()) return {};

    // concatenate remaining nodes, laid out side by side (layout_spacing != 0)
    // and capped per node (opacity*radius) — same as tree_accumulate.
    const double sigma =
            (layout_spacing < 0.0) ? 4.0 * leaf_extent : layout_spacing;
    const double gcentre = 0.5 * M;
    auto importance = [](const AccumPoint &p) -> double {
        const double vol = (double)std::max(p.sx, 1e-9f) *
                           std::max(p.sy, 1e-9f) * std::max(p.sz, 1e-9f);
        return (double)std::max(p.opacity, 0.0f) * std::cbrt(vol);
    };
    std::vector<AccumPoint> out;
    size_t total = 0;
    for (auto &nd : nodes)
        total += (per_node_cap > 0)
                         ? std::min((size_t)per_node_cap, nd.cloud.size())
                         : nd.cloud.size();
    out.reserve(total);
    for (auto &nd : nodes) {
        const double dx = (sigma != 0.0)
                                  ? (0.5 * (nd.lo + nd.hi) - gcentre) * sigma
                                  : 0.0;
        if (per_node_cap > 0 &&
            (int)nd.cloud.size() >
                    per_node_cap) {  // keep each scene's own detail
            std::vector<size_t> idx(nd.cloud.size());
            for (size_t i = 0; i < idx.size(); i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + per_node_cap,
                              idx.end(), [&](size_t a, size_t b) {
                                  return importance(nd.cloud[a]) >
                                         importance(nd.cloud[b]);
                              });
            for (int i = 0; i < per_node_cap; i++) {
                AccumPoint p = nd.cloud[idx[i]];
                p.x = (float)(p.x + dx);
                out.push_back(p);
            }
        } else {
            for (const auto &src : nd.cloud) {
                AccumPoint p = src;
                p.x = (float)(p.x + dx);
                out.push_back(p);
            }
        }
    }
    return out;
}

// ---- consensus fusion -----------------------------------------------------

FuseStats consensus_fuse(const std::vector<AccumPoint> &cloud,
                         double voxel_frac,
                         int k,
                         std::vector<AccumPoint> &fused,
                         int mode) {
    fused.clear();
    FuseStats st{};
    st.raw_points = (int64_t)cloud.size();
    if (cloud.empty()) return st;

    // A non-finite point (NaN/Inf — possible since the cloud is engine output
    // and this is a public boundary) would poison the extent and, worse, make
    // the float->int voxel-coord cast UB. Skip them throughout; fuse only
    // finite points.
    auto finite = [](const AccumPoint &p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };
    // extent = rms distance to centroid; voxel = voxel_frac * extent; coords
    // from min.
    double mean[3] = {0, 0, 0}, lo[3] = {1e300, 1e300, 1e300};
    int64_t nf = 0;
    for (const auto &p : cloud) {
        if (!finite(p)) continue;
        mean[0] += p.x;
        mean[1] += p.y;
        mean[2] += p.z;
        lo[0] = std::min(lo[0], (double)p.x);
        lo[1] = std::min(lo[1], (double)p.y);
        lo[2] = std::min(lo[2], (double)p.z);
        nf++;
    }
    if (nf == 0) return st;
    for (int c = 0; c < 3; c++) mean[c] /= nf;
    double ss = 0;
    for (const auto &p : cloud) {
        if (!finite(p)) continue;
        const double dx = p.x - mean[0], dy = p.y - mean[1], dz = p.z - mean[2];
        ss += dx * dx + dy * dy + dz * dz;
    }
    const double ext = std::sqrt(ss / nf);
    const double v = voxel_frac * ext;
    if (!(v > 0)) return st;

    struct Vox {
        double sx = 0, sy = 0, sz = 0, sr = 0, sg = 0, sb = 0, sop = 0;
        double ssx = 0, ssy = 0, ssz = 0;
        int64_t cnt = 0;
        float q[4] = {1, 0, 0, 0};
        bool has_q = false;
        std::vector<std::pair<int32_t, double>>
                frames;  // (frame, opacity weight)
        int best_frame() const {
            int32_t b = -1;
            double bw = -1;
            for (auto &fr : frames)
                if (fr.second > bw) {
                    bw = fr.second;
                    b = fr.first;
                }
            return b;
        }
    };
    struct Key {
        int32_t i, j, k;
        bool operator==(const Key &o) const {
            return i == o.i && j == o.j && k == o.k;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &q) const {
            uint64_t h = (uint64_t)(uint32_t)q.i * 0x9E3779B97F4A7C15ULL;
            h ^= (uint64_t)(uint32_t)q.j + 0x9E3779B9U + (h << 6) + (h >> 2);
            h ^= (uint64_t)(uint32_t)q.k + 0x85EBCA6BU + (h << 6) + (h >> 2);
            return (size_t)h;
        }
    };
    std::unordered_map<Key, Vox, KeyHash> grid;
    grid.reserve(cloud.size() / 2 + 1);

    auto vcoord = [&](double x, double lov) -> int32_t {
        const double c = std::floor((x - lov) / v);  // finite (x finite, v>0)
        // clamp to int32 so an extreme-but-finite coordinate can't overflow the
        // cast
        if (c < -2147483640.0) return -2147483640;
        if (c > 2147483640.0) return 2147483640;
        return (int32_t)c;
    };
    for (const auto &p : cloud) {
        if (!finite(p)) continue;
        Key key{vcoord(p.x, lo[0]), vcoord(p.y, lo[1]), vcoord(p.z, lo[2])};
        Vox &vx = grid[key];
        vx.sx += p.x;
        vx.sy += p.y;
        vx.sz += p.z;
        vx.sr += p.r;
        vx.sg += p.g;
        vx.sb += p.b;
        vx.sop += p.opacity;
        vx.cnt++;
        vx.ssx += p.sx;
        vx.ssy += p.sy;
        vx.ssz += p.sz;
        if (!vx.has_q) {
            vx.q[0] = p.qw;
            vx.q[1] = p.qx;
            vx.q[2] = p.qy;
            vx.q[3] = p.qz;
            vx.has_q = true;
        }  // representative orientation
        const double w = std::max((double)p.opacity, 1e-3);
        bool seen = false;
        for (auto &fr : vx.frames)
            if (fr.first == p.frame) {
                fr.second += w;
                seen = true;
                break;
            }
        if (!seen) vx.frames.push_back({p.frame, w});
    }

    st.voxels = (int64_t)grid.size();
    for (const auto &kv : grid) {
        if ((int)kv.second.frames.size() < k) continue;
        st.kept_voxels++;
        st.points_kept += kv.second.cnt;
    }
    st.points_dropped = st.raw_points - st.points_kept;

    if (mode == FUSE_KEPT) {
        // DENSE: keep every raw gaussian whose voxel is corroborated by >= k
        // frames (floaters dropped, nothing averaged/decimated away).
        fused.reserve((size_t)st.points_kept);
        for (const auto &p : cloud) {
            if (!finite(p)) continue;
            Key key{vcoord(p.x, lo[0]), vcoord(p.y, lo[1]), vcoord(p.z, lo[2])};
            auto it = grid.find(key);
            if (it != grid.end() && (int)it->second.frames.size() >= k)
                fused.push_back(p);
        }
    } else if (mode == FUSE_BEST) {
        // DENSE + DE-GHOSTED: in each consensus voxel keep only the single
        // most- confident frame's gaussians, so overlapping (disagreeing)
        // copies aren't stacked — one frame represents each region.
        for (const auto &p : cloud) {
            if (!finite(p)) continue;
            Key key{vcoord(p.x, lo[0]), vcoord(p.y, lo[1]), vcoord(p.z, lo[2])};
            auto it = grid.find(key);
            if (it != grid.end() && (int)it->second.frames.size() >= k &&
                p.frame == it->second.best_frame())
                fused.push_back(p);
        }
    } else {
        // DENOISED: one averaged gaussian per consensus voxel.
        for (const auto &kv : grid) {
            const Vox &vx = kv.second;
            if ((int)vx.frames.size() < k) continue;
            AccumPoint p;
            p.x = (float)(vx.sx / vx.cnt);
            p.y = (float)(vx.sy / vx.cnt);
            p.z = (float)(vx.sz / vx.cnt);
            p.r = (float)(vx.sr / vx.cnt);
            p.g = (float)(vx.sg / vx.cnt);
            p.b = (float)(vx.sb / vx.cnt);
            p.opacity = (float)(vx.sop / vx.cnt);
            p.sx = (float)(vx.ssx / vx.cnt);
            p.sy = (float)(vx.ssy / vx.cnt);
            p.sz = (float)(vx.ssz / vx.cnt);
            p.qw = vx.q[0];
            p.qx = vx.q[1];
            p.qy = vx.q[2];
            p.qz = vx.q[3];
            p.frame =
                    (int32_t)vx.frames.size();  // support count (informational)
            fused.push_back(p);
        }
    }
    return st;
}

// ---- gaussian-level geometric de-ghost -------------------------------------

double consensus_refine(std::vector<AccumPoint> &cloud,
                        double voxel_frac,
                        int iters,
                        double alpha) {
    if (cloud.size() < 2 || iters < 1) return 0.0;
    auto finite = [](const AccumPoint &p) {
        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };
    struct Key {
        int32_t i, j, k;
        bool operator==(const Key &o) const {
            return i == o.i && j == o.j && k == o.k;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &q) const {
            uint64_t h = (uint64_t)(uint32_t)q.i * 0x9E3779B97F4A7C15ULL;
            h ^= (uint64_t)(uint32_t)q.j + 0x9E3779B9U + (h << 6) + (h >> 2);
            h ^= (uint64_t)(uint32_t)q.k + 0x85EBCA6BU + (h << 6) + (h >> 2);
            return (size_t)h;
        }
    };
    struct FC {
        int32_t f;
        double sx, sy, sz, w;
    };  // per-frame opacity-weighted centroid

    double disagree = 0.0;
    for (int it = 0; it < iters; it++) {
        double mean[3] = {0, 0, 0}, lo[3] = {1e300, 1e300, 1e300};
        int64_t nf = 0;
        for (const auto &p : cloud) {
            if (!finite(p)) continue;
            mean[0] += p.x;
            mean[1] += p.y;
            mean[2] += p.z;
            lo[0] = std::min(lo[0], (double)p.x);
            lo[1] = std::min(lo[1], (double)p.y);
            lo[2] = std::min(lo[2], (double)p.z);
            nf++;
        }
        if (nf == 0) return disagree;
        for (int c = 0; c < 3; c++) mean[c] /= nf;
        double ss = 0;
        for (const auto &p : cloud) {
            if (!finite(p)) continue;
            const double dx = p.x - mean[0], dy = p.y - mean[1],
                         dz = p.z - mean[2];
            ss += dx * dx + dy * dy + dz * dz;
        }
        const double ext = std::sqrt(ss / nf);
        const double frac =
                (iters > 1)
                        ? voxel_frac *
                                  std::pow(
                                          2.0,
                                          1.0 - (double)it /
                                                          (iters -
                                                           1))  // 2*vf -> vf
                                                                // (coarse->fine)
                        : voxel_frac;
        const double v = frac * ext;
        if (!(v > 0)) return disagree;
        auto vc = [&](double x, double l) -> int32_t {
            const double c = std::floor((x - l) / v);
            if (c < -2.1e9) return -2100000000;
            if (c > 2.1e9) return 2100000000;
            return (int32_t)c;
        };

        // pass 1: per-voxel, per-frame opacity-weighted centroid (a snapshot)
        std::unordered_map<Key, std::vector<FC>, KeyHash> grid;
        grid.reserve(cloud.size() / 2 + 1);
        for (const auto &p : cloud) {
            if (!finite(p)) continue;
            Key key{vc(p.x, lo[0]), vc(p.y, lo[1]), vc(p.z, lo[2])};
            const double w = std::max((double)p.opacity, 1e-3);
            auto &vec = grid[key];
            bool hit = false;
            for (auto &fc : vec)
                if (fc.f == p.frame) {
                    fc.sx += w * p.x;
                    fc.sy += w * p.y;
                    fc.sz += w * p.z;
                    fc.w += w;
                    hit = true;
                    break;
                }
            if (!hit)
                vec.push_back({p.frame, w * (double)p.x, w * (double)p.y,
                               w * (double)p.z, w});
        }
        // pass 2: move each point toward the OTHER frames' consensus in its
        // voxel
        double dsum = 0;
        int64_t dcnt = 0;
        for (auto &p : cloud) {
            if (!finite(p)) continue;
            Key key{vc(p.x, lo[0]), vc(p.y, lo[1]), vc(p.z, lo[2])};
            auto git = grid.find(key);
            if (git == grid.end() || git->second.size() < 2) continue;
            double tx = 0, ty = 0, tz = 0, tw = 0;
            for (const auto &fc : git->second) {
                if (fc.f == p.frame) continue;
                tx += fc.sx;
                ty += fc.sy;
                tz += fc.sz;
                tw += fc.w;
            }
            if (tw <= 0) continue;
            tx /= tw;
            ty /= tw;
            tz /= tw;
            const double dx = tx - p.x, dy = ty - p.y, dz = tz - p.z;
            dsum += dx * dx + dy * dy + dz * dz;
            dcnt++;
            p.x += (float)(alpha * dx);
            p.y += (float)(alpha * dy);
            p.z += (float)(alpha * dz);
        }
        disagree = (dcnt && ext > 0) ? std::sqrt(dsum / dcnt) / ext : 0.0;
    }
    return disagree;
}

// ---- loop closure ---------------------------------------------------------

Mat4 sim4_invert(const Mat4 &M) {
    Mat3 A{};
    Vec3 t{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) A(i, j) = M(i, j);
        t[i] = M(i, 3);
    }
    const double s = std::cbrt(det3(A));
    Mat3 R = A;
    for (int i = 0; i < 9; i++) R.a[i] /= s;
    Sim3 inv = sim_invert(Sim3{s, R, t});
    return sim_matrix(inv);
}

std::vector<Vec3> distribute_drift(const Mat4 &D,
                                   const std::vector<Mat4> &global_poses) {
    const int m = (int)global_poses.size();
    std::vector<Vec3> out(m);
    const int n = std::max(m - 1, 1);
    for (int k = 0; k < m; k++) {
        Mat4 Dk = sim_frac_power(D, (double)k / n);
        Vec3 c = {global_poses[k](0, 3), global_poses[k](1, 3),
                  global_poses[k](2, 3)};
        Vec3 r = mat3_apply(
                Mat3{{Dk(0, 0), Dk(0, 1), Dk(0, 2), Dk(1, 0), Dk(1, 1),
                      Dk(1, 2), Dk(2, 0), Dk(2, 1), Dk(2, 2)}},
                c);
        out[k] = {r[0] + Dk(0, 3), r[1] + Dk(1, 3), r[2] + Dk(2, 3)};
    }
    return out;
}

}  // namespace pose
}  // namespace gaussian
}  // namespace aicore
