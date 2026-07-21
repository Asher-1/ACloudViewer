// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "linalg.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aicore {
namespace depth {
namespace linalg {

static inline double dsign(double x) { return (x > 0.0) - (x < 0.0); }

double det3(const double A[9]) {
    return A[0] * (A[4] * A[8] - A[5] * A[7]) -
           A[1] * (A[3] * A[8] - A[5] * A[6]) +
           A[2] * (A[3] * A[7] - A[4] * A[6]);
}

void mat3_mul(const double A[9], const double B[9], double out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * B[k * 3 + j];
            out[i * 3 + j] = s;
        }
}

void mat3_transpose(const double A[9], double out[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i * 3 + j] = A[j * 3 + i];
}

bool mat3_inverse(const double A[9], double out[9]) {
    double d = det3(A);
    if (std::fabs(d) < 1e-300) return false;
    double inv = 1.0 / d;
    out[0] = (A[4] * A[8] - A[5] * A[7]) * inv;
    out[1] = (A[2] * A[7] - A[1] * A[8]) * inv;
    out[2] = (A[1] * A[5] - A[2] * A[4]) * inv;
    out[3] = (A[5] * A[6] - A[3] * A[8]) * inv;
    out[4] = (A[0] * A[8] - A[2] * A[6]) * inv;
    out[5] = (A[2] * A[3] - A[0] * A[5]) * inv;
    out[6] = (A[3] * A[7] - A[4] * A[6]) * inv;
    out[7] = (A[1] * A[6] - A[0] * A[7]) * inv;
    out[8] = (A[0] * A[4] - A[1] * A[3]) * inv;
    return true;
}

void householder_qr(const double* A, int n, double* Q, double* R) {
    for (int i = 0; i < n * n; ++i) R[i] = A[i];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) Q[i * n + j] = (i == j) ? 1.0 : 0.0;
    std::vector<double> v(n);
    // n-1 reflectors suffice for a square matrix. LAPACK/torch leave the final
    // 1x1 reflector as identity (tau=0), so the last diagonal keeps its sign --
    // matching that exactly here (looping to n would flip the last diag sign,
    // still a valid QR but not bit-matching torch.linalg.qr).
    for (int k = 0; k < n - 1; ++k) {
        double normx = 0;
        for (int i = k; i < n; ++i) normx += R[i * n + k] * R[i * n + k];
        normx = std::sqrt(normx);
        if (normx < 1e-300) continue;
        double alpha = (R[k * n + k] >= 0.0) ? -normx : normx;
        for (int i = 0; i < n; ++i) v[i] = 0;
        for (int i = k; i < n; ++i) v[i] = R[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0;
        for (int i = k; i < n; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-300) continue;
        // R <- (I - 2 v v^T / vnorm2) R : per column j
        for (int j = 0; j < n; ++j) {
            double dot = 0;
            for (int i = k; i < n; ++i) dot += v[i] * R[i * n + j];
            double f = 2.0 * dot / vnorm2;
            for (int i = k; i < n; ++i) R[i * n + j] -= f * v[i];
        }
        // Q <- Q (I - 2 v v^T / vnorm2) : per row r
        for (int r = 0; r < n; ++r) {
            double dot = 0;
            for (int i = k; i < n; ++i) dot += Q[r * n + i] * v[i];
            double f = 2.0 * dot / vnorm2;
            for (int i = k; i < n; ++i) Q[r * n + i] -= f * v[i];
        }
    }
    // Clean tiny sub-diagonal noise.
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j) R[i * n + j] = 0.0;
}

void qr3x3(const double A[9], double Q[9], double R[9]) {
    householder_qr(A, 3, Q, R);
}

void ql_decomposition(const double A[9], double Q[9], double L[9]) {
    // P = anti-identity (reverses index 0<->2).
    // A_tilde = A @ P  (reverse columns of A)
    double At[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) At[i * 3 + j] = A[i * 3 + (2 - j)];
    double Qt[9], Rt[9];
    qr3x3(At, Qt, Rt);
    // Q = Qt @ P  (reverse columns of Qt)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) Q[i * 3 + j] = Qt[i * 3 + (2 - j)];
    // L = P @ Rt @ P  (reverse rows and columns of Rt)
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) L[i * 3 + j] = Rt[(2 - i) * 3 + (2 - j)];
    // Sign-normalize by sign(diag(L)).
    double s0 = dsign(L[0]), s1 = dsign(L[4]), s2 = dsign(L[8]);
    double sc[3] = {s0, s1, s2};
    for (int i = 0; i < 3; ++i) {
        Q[i * 3 + 0] *= s0;
        Q[i * 3 + 1] *= s1;
        Q[i * 3 + 2] *= s2;
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) L[r * 3 + c] *= sc[r];
}

void jacobi_eigen_sym(const double* M,
                      int n,
                      double* eigvals,
                      double* eigvecs) {
    std::vector<double> A(M, M + n * n);
    // V = I
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) eigvecs[i * n + j] = (i == j) ? 1.0 : 0.0;
    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0;
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
        if (off < 1e-30) break;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = A[p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                double app = A[p * n + p];
                double aqq = A[q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = dsign(theta) /
                           (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                if (theta == 0.0) t = 1.0;
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                // Rotate rows/cols p,q of A.
                for (int i = 0; i < n; ++i) {
                    double aip = A[i * n + p];
                    double aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[i * n + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < n; ++i) {
                    double api = A[p * n + i];
                    double aqi = A[q * n + i];
                    A[p * n + i] = c * api - s * aqi;
                    A[q * n + i] = s * api + c * aqi;
                }
                // Accumulate eigenvectors (columns).
                for (int i = 0; i < n; ++i) {
                    double vip = eigvecs[i * n + p];
                    double viq = eigvecs[i * n + q];
                    eigvecs[i * n + p] = c * vip - s * viq;
                    eigvecs[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) eigvals[i] = A[i * n + i];
}

void smallest_eigvec(const double* M, int n, double* v) {
    std::vector<double> eval(n), evec(n * n);
    jacobi_eigen_sym(M, n, eval.data(), evec.data());
    int imin = 0;
    for (int i = 1; i < n; ++i)
        if (eval[i] < eval[imin]) imin = i;
    for (int i = 0; i < n; ++i) v[i] = evec[i * n + imin];
}

bool homography_weighted(const double* src,
                         const double* dst,
                         const double* w,
                         int M,
                         double H[9]) {
    double AtA[81];
    std::memset(AtA, 0, sizeof(AtA));
    for (int m = 0; m < M; ++m) {
        double x = src[2 * m + 0], y = src[2 * m + 1];
        double u = dst[2 * m + 0], v = dst[2 * m + 1];
        double sw = std::sqrt(w[m] < 0 ? 0.0 : w[m]);
        // Row 1: [-x*sw, -y*sw, -sw, 0,0,0, x*u*sw, y*u*sw, u*sw]
        double r1[9] = {-x * sw, -y * sw,    -sw,        0,     0,
                        0,       x * u * sw, y * u * sw, u * sw};
        // Row 2: [0,0,0, -x*sw, -y*sw, -sw, x*v*sw, y*v*sw, v*sw]
        double r2[9] = {0,   0,          0,          -x * sw, -y * sw,
                        -sw, x * v * sw, y * v * sw, v * sw};
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                AtA[i * 9 + j] += r1[i] * r1[j] + r2[i] * r2[j];
    }
    double h[9];
    smallest_eigvec(AtA, 9, h);
    if (std::fabs(h[8]) < 1e-300) return false;
    double inv = 1.0 / h[8];
    for (int i = 0; i < 9; ++i) H[i] = h[i] * inv;
    return true;
}

}  // namespace linalg
}  // namespace depth
}  // namespace aicore
