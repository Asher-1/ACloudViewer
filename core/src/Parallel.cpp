// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <Parallel.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <CPUInfo.h>
#include <Logging.h>

#include <cstdlib>
#include <string>

namespace cloudViewer {
namespace utility {

static std::string GetEnvVar(const std::string& name) {
    if (const char* value = std::getenv(name.c_str())) {
        return std::string(value);
    } else {
        return "";
    }
}

int EstimateMaxThreads() {
#ifdef _OPENMP
    if (!GetEnvVar("OMP_NUM_THREADS").empty() ||
        !GetEnvVar("OMP_DYNAMIC").empty()) {
        // See the full list of OpenMP environment variables at:
        // https://www.openmp.org/spec-html/5.0/openmpch6.html
        return omp_get_max_threads();
    } else {
        // Returns the number of physical cores.
        return utility::CPUInfo::GetInstance().NumCores();
    }
#else
    (void)GetEnvVar;  // Avoids compiler warning.
    return 1;
#endif
}

int GetThreadNum() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;  // No parallelism available, so thread ID is always 0.
#endif
}

bool InParallel() {
    // TODO: when we add TBB/Parallel STL support to ParallelFor, update this.
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

}  // namespace utility
}  // namespace cloudViewer
