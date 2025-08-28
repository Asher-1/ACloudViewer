// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace cloudViewer {
namespace gpu {

/**
 * @brief Check if the system support CUDA with the given parameters
 * @param[in] minComputeCapabilityMajor The minimum compute capability major
 * @param[in] minComputeCapabilityMinor The minimum compute capability minor
 * @param[in] minTotalDeviceMemory The minimum device total memory in MB
 * @return True if system support CUDA with the given parameters
 */
bool gpuSupportCUDA(int minComputeCapabilityMajor,
                    int minComputeCapabilityMinor,
                    int minTotalDeviceMemory = 0);


/**
 * @brief gpuInformationCUDA
 * @return string with all CUDA device(s) information
 */
std::string gpuInformationCUDA();

} // namespace gpu
} // namespace cloudViewer
