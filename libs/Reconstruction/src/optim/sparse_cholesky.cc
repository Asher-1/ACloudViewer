// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "optim/sparse_cholesky.h"

#include "util/logging.h"

namespace colmap {

void SparseCholeskyWithFallbackSolver::AnalyzePattern(
    const Eigen::SparseMatrix<double>& A) {
  ldlt_.analyzePattern(A);
  analyzed_ = true;
}

bool SparseCholeskyWithFallbackSolver::Factorize(
    const Eigen::SparseMatrix<double>& A) {
  ldlt_.factorize(A);
  return ldlt_.info() == Eigen::Success;
}

bool SparseCholeskyWithFallbackSolver::Compute(
    const Eigen::SparseMatrix<double>& A) {
  AnalyzePattern(A);
  return Factorize(A);
}

bool SparseCholeskyWithFallbackSolver::Solve(const Eigen::VectorXd& b,
                                             Eigen::VectorXd* x) const {
  THROW_CHECK_NOTNULL(x);
  THROW_CHECK(analyzed_) << "Factorize/Compute must be called before Solve";
  x->noalias() = ldlt_.solve(b);
  return ldlt_.info() == Eigen::Success;
}


}  // namespace colmap
