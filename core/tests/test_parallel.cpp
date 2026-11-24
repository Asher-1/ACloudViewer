// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Parallel.h>
#include <gtest/gtest.h>

#include <array>

TEST(Parallel, ThreadCapsRespectExplicitAndRuntimeLimits) {
    using cloudViewer::utility::GetMaxThreads;
    EXPECT_EQ(GetMaxThreads(0), 1);
    EXPECT_EQ(GetMaxThreads(-3), 1);
    EXPECT_EQ(GetMaxThreads(1), 1);
    EXPECT_GE(GetMaxThreads(8), 1);
    EXPECT_LE(GetMaxThreads(8), 8);
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(3);
    EXPECT_EQ(GetMaxThreads(), 3);
    EXPECT_EQ(GetMaxThreads(2), 2);
    omp_set_num_threads(previous);
#else
    EXPECT_EQ(GetMaxThreads(), 1);
#endif
}

TEST(Parallel, TwoDimensionalLoopPreservesSerialInnerReduction) {
    using cloudViewer::utility::GetMaxThreads;
    constexpr int rows = 7, columns = 9;
    std::array<int, rows * columns> cells{};
    CV_OMP_PARALLEL_FOR_2D(schedule(static) num_threads(GetMaxThreads(3)))
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            int value = 0;
            for (int k = 0; k < 11; ++k) value += row + column + k;
            cells[row * columns + column] = value;
        }
    }
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column)
            EXPECT_EQ(cells[row * columns + column], 11 * (row + column) + 55);
}

TEST(Parallel, DynamicLoopCoversEachCellAndUsesValidThreadIds) {
    using cloudViewer::utility::GetMaxThreads;
    int sum = 0;
    int invalid_thread_ids = 0;
    CV_OMP_PARALLEL_FOR(schedule(dynamic, 2) num_threads(GetMaxThreads(3))
                        reduction(+ : sum, invalid_thread_ids))
    for (int i = 0; i < 63; ++i) {
        const int thread = cloudViewer::utility::GetOpenMPThreadNum();
        if (thread < 0 || thread >= 3) ++invalid_thread_ids;
        sum += i;
    }
    EXPECT_EQ(sum, 1953);
    EXPECT_EQ(invalid_thread_ids, 0);
}

TEST(Parallel, SimdReductionPreservesSum) {
    int square_sum = 0;
    CV_OMP_SIMD(reduction(+ : square_sum))
    for (int i = 0; i < 9; ++i) square_sum += i * i;
    EXPECT_EQ(square_sum, 204);
}
