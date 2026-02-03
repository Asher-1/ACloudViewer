// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CVCoreLib.h"

namespace cloudViewer {
namespace utility {

/// Estimate the maximum number of threads to be used in a parallel region.
int CV_CORE_LIB_API EstimateMaxThreads();

/// Returns the thread ID in the current parallel region.
int CV_CORE_LIB_API GetThreadNum();

/// Returns true if in an parallel section.
bool CV_CORE_LIB_API InParallel();

}  // namespace utility
}  // namespace cloudViewer
