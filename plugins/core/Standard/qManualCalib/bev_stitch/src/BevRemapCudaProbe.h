// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

namespace mcalib {
namespace bev_cuda {

// Probes CUDA runtime availability without crashing when it is missing.
// Implemented in BevRemapCudaProbe.cpp: a host C++ file so that MSVC's
// __try/__except (SEH) can be used on Windows (nvcc cannot parse it in .cu).
bool probeAvailable();

}  // namespace bev_cuda
}  // namespace mcalib
