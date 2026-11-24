// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// T1 gate: host linear-algebra primitives for the ray->pose solver.
// Pure host math, no GGUF needed -> always runs.
#include <gtest/gtest.h>

#include "tasks/depth/linalg.hpp"

using namespace aicore::depth::linalg;

namespace {
// Fixed test matrix matching the torch reference generated in the harness.
constexpr double A[9] = {2.0, -1.0, 0.3, 0.5, 3.0, 1.0, -0.7, 0.2, 2.5};
}  // namespace

TEST(DepthLinalg, DeterminantMatchesReference) {
    EXPECT_NEAR(det3(A), 17.21, 1e-9);
}

TEST(DepthLinalg, QrMatchesTorchAndReconstructsOrthonormalFactors) {
    const double QR_Q[9] = {-0.9186304243, 0.2313704865,  0.3202902456,
                            -0.2296576061, -0.9722910038, 0.0436759426,
                            0.3215206485,  -0.0334350414, 0.9463120894};
    const double QR_R[9] = {-2.1771541057, 0.2939617358, 0.2985548879,
                            0.0,           -3.154930506, -0.9864674613,
                            0.0,           0.0,          2.5055432398};
    double Q[9], R[9];
    qr3x3(A, Q, R);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(Q[i], QR_Q[i], 1e-8) << "Q element " << i;
        EXPECT_NEAR(R[i], QR_R[i], 1e-8) << "R element " << i;
    }
    double QR[9];
    mat3_mul(Q, R, QR);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(QR[i], A[i], 1e-12) << "Q*R element " << i;
    }
    double Qt[9], QtQ[9];
    mat3_transpose(Q, Qt);
    mat3_mul(Qt, Q, QtQ);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(QtQ[i * 3 + j], (i == j) ? 1.0 : 0.0, 1e-12)
                    << "Q^T*Q at " << i << "," << j;
        }
    }
}

TEST(DepthLinalg, QlMatchesTorchAndHasPositiveLowerTriangularFactor) {
    const double QL_Q[9] = {0.9164203548,  -0.3845934904, 0.1107320206,
                            0.3213748094,  0.8720541554,  0.3691067353,
                            -0.2385203663, -0.3026704433, 0.9227668382};
    const double QL_L[9] = {2.1604923708,  0.0,          0.0,
                            -0.1212905928, 2.9402218678, 0.0,
                            -0.2399193779, 1.1811415528, 2.7092434368};
    double L[9], Q[9];
    ql_decomposition(A, Q, L);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(Q[i], QL_Q[i], 1e-8) << "Q element " << i;
        EXPECT_NEAR(L[i], QL_L[i], 1e-8) << "L element " << i;
    }
    double QL[9];
    mat3_mul(Q, L, QL);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(QL[i], A[i], 1e-12) << "Q*L element " << i;
    }
    EXPECT_NEAR(L[1], 0, 1e-12);
    EXPECT_NEAR(L[2], 0, 1e-12);
    EXPECT_NEAR(L[5], 0, 1e-12);
    EXPECT_GT(L[0], 0);
    EXPECT_GT(L[4], 0);
    EXPECT_GT(L[8], 0);
}

TEST(DepthLinalg, JacobiSmallestEigenpairHasSmallResidual) {
    const int n = 4;
    double M[16] = {4, 1, 0, 2, 1, 3, 1, 0, 0, 1, 2, 1, 2, 0, 1, 5};
    double v[4];
    smallest_eigvec(M, n, v);
    double Mv[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) Mv[i] += M[i * n + j] * v[j];
    }
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        num += v[i] * Mv[i];
        den += v[i] * v[i];
    }
    ASSERT_GT(den, 0);
    const double lambda = num / den;
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(Mv[i], lambda * v[i], 1e-9) << "element " << i;
    }
    EXPECT_LT(lambda, 2.0);
    EXPECT_GT(lambda, 0.5);
}

TEST(DepthLinalg, WeightedHomographyRecoversKnownTransform) {
    const double Htrue[9] = {1.2, 0.1, 0.3, -0.05, 0.9, -0.2, 0.02, -0.01, 1.0};
    double src[2 * 12], dst[2 * 12], w[12];
    const double pts[12][2] = {{0.1, 0.2},   {-0.3, 0.5},  {0.7, -0.4},
                               {-0.6, -0.8}, {0.4, 0.9},   {-0.2, 0.1},
                               {0.8, 0.3},   {-0.7, 0.6},  {0.3, -0.5},
                               {-0.4, -0.2}, {0.55, 0.15}, {-0.15, -0.65}};
    for (int m = 0; m < 12; ++m) {
        const double x = pts[m][0], y = pts[m][1];
        const double X = Htrue[0] * x + Htrue[1] * y + Htrue[2];
        const double Y = Htrue[3] * x + Htrue[4] * y + Htrue[5];
        const double Z = Htrue[6] * x + Htrue[7] * y + Htrue[8];
        src[2 * m + 0] = x;
        src[2 * m + 1] = y;
        dst[2 * m + 0] = X / Z;
        dst[2 * m + 1] = Y / Z;
        w[m] = 1.0;
    }
    double Hrec[9];
    ASSERT_TRUE(homography_weighted(src, dst, w, 12, Hrec));
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(Hrec[i], Htrue[i], 1e-6) << "element " << i;
    }
}
