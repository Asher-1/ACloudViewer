// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace colmap {

// Sparse Cholesky solver based on Eigen's simplicial LDLT (more numerically
// tolerant than supernodal solvers: it does not abort at the first
// non-positive pivot on ill-conditioned but mathematically PD systems).
// Upstream additionally accelerates with CHOLMOD when SuiteSparse is
// available; the fork uses the portable Eigen implementation until a
// conditional SuiteSparse dependency is introduced.
class SparseCholeskyWithFallbackSolver {
public:
    // One-shot factorization (analyze + factorize).
    bool Compute(const Eigen::SparseMatrix<double>& A);

    // For iterative reuse with matrices of identical sparsity but changing
    // values, call AnalyzePattern once then Factorize per iteration.
    void AnalyzePattern(const Eigen::SparseMatrix<double>& A);
    bool Factorize(const Eigen::SparseMatrix<double>& A);

    bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) const;

private:
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_;
    bool analyzed_ = false;
};

}  // namespace colmap
