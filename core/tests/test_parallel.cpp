// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Parallel.h>

#include <array>
#include <cstdio>

int main() {
    using cloudViewer::utility::GetMaxThreads;
    if (GetMaxThreads(0) != 1 || GetMaxThreads(-3) != 1 ||
        GetMaxThreads(1) != 1 || GetMaxThreads(8) < 1 || GetMaxThreads(8) > 8) {
        std::fprintf(stderr, "Invalid thread cap\n");
        return 1;
    }
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(3);
    if (GetMaxThreads() != 3 || GetMaxThreads(2) != 2) {
        std::fprintf(stderr, "OpenMP runtime limit was not respected\n");
        return 1;
    }
    omp_set_num_threads(previous);
#else
    if (GetMaxThreads() != 1) return 1;
#endif

    constexpr int rows = 7, columns = 9;
    std::array<int, rows * columns> cells{};
    CV_OMP_PARALLEL_FOR_2D(schedule(static) num_threads(GetMaxThreads(3)))
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            // A serial inner reduction must keep its order in either loop mode.
            int value = 0;
            for (int k = 0; k < 11; ++k) value += row + column + k;
            cells[row * columns + column] = value;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            if (cells[row * columns + column] != 11 * (row + column) + 55) {
                std::fprintf(stderr, "Parallel loop changed a cell\n");
                return 1;
            }
        }
    }

    int sum = 0;
    int invalid_thread_ids = 0;
    CV_OMP_PARALLEL_FOR(schedule(dynamic, 2) num_threads(GetMaxThreads(3))
                        reduction(+ : sum, invalid_thread_ids))
    for (int i = 0; i < 63; ++i) {
        const int thread = cloudViewer::utility::GetOpenMPThreadNum();
        if (thread < 0 || thread >= 3) ++invalid_thread_ids;
        sum += i;
    }
    if (sum != 1953 || invalid_thread_ids != 0) return 1;

    int square_sum = 0;
    CV_OMP_SIMD(reduction(+ : square_sum))
    for (int i = 0; i < 9; ++i) square_sum += i * i;
    if (square_sum != 204) return 1;
    std::puts("Parallel thread limits, loop coverage and reductions passed");
    return 0;
}
