// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "CVCoreLib.h"

// Keep compiler syntax and OpenMP feature checks at the parallel boundary.
#ifdef _MSC_VER
#define CV_OMP_PRAGMA_IMPL(directive) __pragma(directive)
#else
#define CV_OMP_PRAGMA_IMPL(directive) _Pragma(#directive)
#endif
#define CV_OMP_PRAGMA(directive) CV_OMP_PRAGMA_IMPL(directive)

#ifdef _OPENMP
#define CV_OMP_PARALLEL_FOR(...) CV_OMP_PRAGMA(omp parallel for __VA_ARGS__)
#if _OPENMP >= 200805
#define CV_OMP_PARALLEL_FOR_2D(...) CV_OMP_PARALLEL_FOR(collapse(2) __VA_ARGS__)
#else
// OpenMP 2.0 parallelizes the outer axis without an unsupported collapse.
#define CV_OMP_PARALLEL_FOR_2D(...) CV_OMP_PARALLEL_FOR(__VA_ARGS__)
#endif
#else
#define CV_OMP_PARALLEL_FOR(...)
#define CV_OMP_PARALLEL_FOR_2D(...)
#endif

// MSVC's SIMD extension is enabled explicitly beside /openmp:experimental.
#if defined(_OPENMP) && (_OPENMP >= 201307 || defined(CV_OPENMP_SIMD_ENABLED))
#define CV_OMP_SIMD(...) CV_OMP_PRAGMA(omp simd __VA_ARGS__)
#else
#define CV_OMP_SIMD(...)
#endif

namespace cloudViewer {
namespace utility {

/// Return this translation unit's OpenMP team limit, capped by ceiling (min 1).
/// Unlike EstimateMaxThreads(), this does not select a physical-core policy.
/// Internal linkage keeps serial and OpenMP callers independent in mixed
/// builds.
static inline int GetMaxThreads(
        int ceiling = (std::numeric_limits<int>::max)()) {
#ifdef _OPENMP
    return (std::max)(1, (std::min)(ceiling, omp_get_max_threads()));
#else
    (void)ceiling;
    return 1;
#endif
}

/// Return the current caller's OpenMP thread ID, or zero for a serial caller.
/// Use this with header directives when CVCoreLib has different OpenMP flags.
static inline int GetOpenMPThreadNum() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/// Estimate the maximum number of threads to be used in a parallel region.
int CV_CORE_LIB_API EstimateMaxThreads();

/// Returns the thread ID in the current parallel region.
int CV_CORE_LIB_API GetThreadNum();

/// Returns true if in an parallel section.
bool CV_CORE_LIB_API InParallel();

}  // namespace utility
}  // namespace cloudViewer
