// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// linalg.h — self-contained small dense linear algebra for the pose component.
//
// The downstream pose math (Umeyama similarity, DLT PnP, pose decode) needs
// only a handful of small dense operations. Rather than pull in Eigen/OpenCV —
// neither of which we want as a runtime dependency (see CLAUDE.md: everything
// ships in C++ with no extra runtime deps beyond ggml) — every routine here
// reduces to one primitive: a symmetric cyclic-Jacobi eigensolver. SVD of a 3x3
// is built as the eigendecomposition of MᵀM; the DLT nullspace is the smallest
// eigenvector of AᵀA. This mirrored the dependency-free numpy reference solver
// of the (now removed; see git history) pose/ prototype, itself verified
// bit-for-bit (~1e-7) against cv2.
//
// Everything is f64, row-major, and header-only. Sizes are tiny (<=12) so the
// O(n^3) Jacobi sweeps are negligible next to the engine forward pass.
#ifndef FREE_SPLATTER_LINALG_H
#define FREE_SPLATTER_LINALG_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fsla {

using Vec3 = std::array<double, 3>;

// Row-major fixed matrices. m[r][c] lives at a[r*N + c].
struct Mat3 {
    double a[9];
    double operator()(int r, int c) const { return a[r * 3 + c]; }
    double& operator()(int r, int c) { return a[r * 3 + c]; }
};
struct Mat4 {
    double a[16];
    double operator()(int r, int c) const { return a[r * 4 + c]; }
    double& operator()(int r, int c) { return a[r * 4 + c]; }
};

inline Mat3 mat3_identity() { return Mat3{{1, 0, 0, 0, 1, 0, 0, 0, 1}}; }
inline Mat4 mat4_identity() {
    return Mat4{{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
}

inline Mat3 mat3_mul(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

inline Mat3 mat3_transpose(const Mat3& A) {
    Mat3 T{};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) T(i, j) = A(j, i);
    return T;
}

inline Vec3 mat3_apply(const Mat3& A, const Vec3& x) {
    return {A(0, 0) * x[0] + A(0, 1) * x[1] + A(0, 2) * x[2],
            A(1, 0) * x[0] + A(1, 1) * x[1] + A(1, 2) * x[2],
            A(2, 0) * x[0] + A(2, 1) * x[1] + A(2, 2) * x[2]};
}

inline double det3(const Mat3& M) {
    return M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) -
           M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
           M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));
}

inline Mat3 inv3(const Mat3& M) {
    const double d = det3(M);
    const double id = (std::fabs(d) > 0) ? 1.0 / d : 0.0;
    Mat3 R{};
    R(0, 0) = (M(1, 1) * M(2, 2) - M(1, 2) * M(2, 1)) * id;
    R(0, 1) = -(M(0, 1) * M(2, 2) - M(0, 2) * M(2, 1)) * id;
    R(0, 2) = (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) * id;
    R(1, 0) = -(M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) * id;
    R(1, 1) = (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)) * id;
    R(1, 2) = -(M(0, 0) * M(1, 2) - M(0, 2) * M(1, 0)) * id;
    R(2, 0) = (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0)) * id;
    R(2, 1) = -(M(0, 0) * M(2, 1) - M(0, 1) * M(2, 0)) * id;
    R(2, 2) = (M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0)) * id;
    return R;
}

// Inverse of a rigid 4x4 [[R,t],[0,1]] (R orthonormal): [[Rᵀ, -Rᵀt],[0,1]].
// Exact and cheap; used for cam2world = inv(world2cam) where world2cam is
// rigid.
inline Mat4 inv_rigid4(const Mat4& M) {
    Mat4 R = mat4_identity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) R(i, j) = M(j, i);  // Rᵀ
    for (int i = 0; i < 3; i++)
        R(i, 3) = -(R(i, 0) * M(0, 3) + R(i, 1) * M(1, 3) + R(i, 2) * M(2, 3));
    return R;
}

inline Mat4 mat4_mul(const Mat4& A, const Mat4& B) {
    Mat4 C{};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += A(i, k) * B(k, j);
            C(i, j) = s;
        }
    return C;
}

// --- symmetric eigensolver -------------------------------------------------
//
// Cyclic Jacobi for a symmetric n x n matrix (Golub & Van Loan, Alg. 8.4.x).
// Ain is row-major and assumed symmetric. Writes n eigenvalues to `eval` and
// the corresponding eigenvectors as the COLUMNS of `evec` (evec[i*n+j] =
// component i of eigenvector j). Output is NOT sorted. Converges in a handful
// of sweeps for n <= 12; we cap at 100 as a safety net.
inline void jacobi_eigh(const double* Ain, int n, double* eval, double* evec) {
    std::vector<double> A(Ain, Ain + (size_t)n * n);
    for (int i = 0; i < n * n; i++) evec[i] = 0.0;
    for (int i = 0; i < n; i++) evec[i * n + i] = 1.0;

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
        if (off <= 1e-300) break;

        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                const double apq = A[p * n + q];
                if (std::fabs(apq) <= 1e-300) continue;
                const double phi =
                        0.5 * (A[q * n + q] - A[p * n + p]) / apq;  // = tau
                const double t = (phi >= 0 ? 1.0 : -1.0) /
                                 (std::fabs(phi) + std::sqrt(phi * phi + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                // A <- Jᵀ A J : rotate columns p,q then rows p,q (keeps
                // symmetry).
                for (int i = 0; i < n; i++) {
                    const double aip = A[i * n + p], aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[i * n + q] = s * aip + c * aiq;
                }
                for (int i = 0; i < n; i++) {
                    const double api = A[p * n + i], aqi = A[q * n + i];
                    A[p * n + i] = c * api - s * aqi;
                    A[q * n + i] = s * api + c * aqi;
                }
                for (int i = 0; i < n;
                     i++) {  // accumulate eigenvectors V <- V J
                    const double vip = evec[i * n + p], viq = evec[i * n + q];
                    evec[i * n + p] = c * vip - s * viq;
                    evec[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) eval[i] = A[i * n + i];
}

// Smallest-eigenvalue eigenvector of a symmetric n x n matrix (the homogeneous
// least-squares / DLT nullspace). Returns the n-vector.
inline std::vector<double> smallest_eigenvector(const double* A, int n) {
    std::vector<double> eval(n), evec((size_t)n * n);
    jacobi_eigh(A, n, eval.data(), evec.data());
    int k = 0;
    for (int i = 1; i < n; i++)
        if (eval[i] < eval[k]) k = i;
    std::vector<double> v(n);
    for (int i = 0; i < n; i++) v[i] = evec[i * n + k];
    return v;
}

// --- 3x3 SVD ---------------------------------------------------------------
//
// M = U diag(s) Vᵀ with s sorted DESCENDING (numpy/LAPACK convention). Built
// from the eigendecomposition of the symmetric S = MᵀM: its eigenvectors are V,
// the singular values are sqrt(eigenvalues), and U_i = M v_i / s_i. A
// (near-)zero singular direction is completed by Gram-Schmidt so U stays
// orthonormal. This is all the SVD the similarity fit and the PnP decode need.
inline void svd3(const Mat3& M, Mat3& U, Vec3& s, Mat3& V) {
    Mat3 S = mat3_mul(mat3_transpose(M), M);  // 3x3 symmetric PSD
    double eval[3], evec[9];
    jacobi_eigh(S.a, 3, eval, evec);

    int ord[3] = {0, 1, 2};  // sort eigenpairs descending
    std::sort(ord, ord + 3, [&](int x, int y) { return eval[x] > eval[y]; });

    for (int j = 0; j < 3; j++) {
        const int e = ord[j];
        s[j] = std::sqrt(std::max(eval[e], 0.0));
        for (int i = 0; i < 3; i++) V(i, j) = evec[i * 3 + e];
    }
    // Force a right-handed V (proper basis); flip the last column if reflected.
    if (det3(V) < 0)
        for (int i = 0; i < 3; i++) V(i, 2) = -V(i, 2);

    for (int j = 0; j < 3; j++) {
        Vec3 vj = {V(0, j), V(1, j), V(2, j)};
        Vec3 uj = mat3_apply(M, vj);
        const double nrm =
                std::sqrt(uj[0] * uj[0] + uj[1] * uj[1] + uj[2] * uj[2]);
        if (nrm > 1e-12)
            for (int i = 0; i < 3; i++) U(i, j) = uj[i] / nrm;
        else
            for (int i = 0; i < 3; i++) U(i, j) = 0.0;  // fixed up below
    }
    // Complete any degenerate U columns via Gram-Schmidt against the filled
    // ones.
    for (int j = 0; j < 3; j++) {
        double nrm = std::sqrt(U(0, j) * U(0, j) + U(1, j) * U(1, j) +
                               U(2, j) * U(2, j));
        if (nrm > 1e-9) continue;
        Vec3 cand[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (auto& e : cand) {
            for (int k = 0; k < 3; k++) {
                if (k == j) continue;
                const double d =
                        e[0] * U(0, k) + e[1] * U(1, k) + e[2] * U(2, k);
                e[0] -= d * U(0, k);
                e[1] -= d * U(1, k);
                e[2] -= d * U(2, k);
            }
            const double en =
                    std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
            if (en > 1e-6) {
                U(0, j) = e[0] / en;
                U(1, j) = e[1] / en;
                U(2, j) = e[2] / en;
                break;
            }
        }
    }
}

}  // namespace fsla

#endif  // FREE_SPLATTER_LINALG_H
