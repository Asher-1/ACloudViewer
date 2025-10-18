// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "CVCoreLib.h"

namespace cloudViewer {
namespace system {

/**
 * @brief Returns the CPU clock, as reported by the OS.
 *
 * Taken from https://github.com/anrieff/libcpuid.
 * Duplicated to avoid the dependency for one function.
 */
CV_CORE_LIB_API int cpu_clock_by_os();

/**
 * @brief Returns the total number of CPUs.
 */
CV_CORE_LIB_API int get_total_cpus();

}  // namespace system
}  // namespace cloudViewer
