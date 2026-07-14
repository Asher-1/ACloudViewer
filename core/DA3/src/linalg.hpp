#pragma once
// Small dense host linear algebra for the ray->pose solver (NO ggml).
// All matrices are row-major double. These mirror the exact operations the
// reference ray_utils.py performs (torch.linalg.qr / svd) but on tiny dense
// matrices, in double precision, returning float at the boundaries.
namespace da {
namespace linalg {

// Determinant of a row-major 3x3.
double det3(const double A[9]);
// out = A * B (3x3 row-major).
void mat3_mul(const double A[9], const double B[9], double out[9]);
// out = A^T (3x3 row-major).
void mat3_transpose(const double A[9], double out[9]);
// out = A^{-1}; returns false if singular.
bool mat3_inverse(const double A[9], double out[9]);

// Reduced Householder QR of a square n x n row-major matrix: A = Q * R with Q
// orthonormal and R upper-triangular. Sign convention matches LAPACK/torch
// (R[k,k] = -sign(x_k)*||x||). Q,R are n*n row-major outputs.
void householder_qr(const double* A, int n, double* Q, double* R);
// 3x3 convenience wrapper.
void qr3x3(const double A[9], double Q[9], double R[9]);

// ql_decomposition exactly as ray_utils.ql_decomposition: returns Q (orthonormal)
// and L (lower-triangular, diag>=0) with A = Q * L. Sign-normalized so the result
// is invariant to the QR sign ambiguity.
void ql_decomposition(const double A[9], double Q[9], double L[9]);

// Cyclic Jacobi eigendecomposition of a symmetric n x n row-major matrix M.
// eigvals[n] (not sorted), eigvecs[n*n] row-major with eigenvector j in COLUMN j
// (eigvecs[i*n+j]). M is not modified.
void jacobi_eigen_sym(const double* M, int n, double* eigvals, double* eigvecs);
// Eigenvector of the SMALLEST eigenvalue of symmetric n x n M -> v[n].
void smallest_eigvec(const double* M, int n, double* v);

// Weighted least-squares homography from M point correspondences, matching
// find_homography_least_squares_weighted_torch: builds the 2M x 9 system,
// solves for the smallest right-singular vector via Jacobi eigen of A^T A,
// reshapes to 3x3 and normalizes by H[2,2]. src/dst are length 2*M (x,y
// interleaved), w is length M (the confidence, NOT sqrt'd by caller).
// Returns false if degenerate (H[2,2]~0). H is 3x3 row-major.
bool homography_weighted(const double* src, const double* dst, const double* w,
                         int M, double H[9]);

} // namespace linalg
} // namespace da
