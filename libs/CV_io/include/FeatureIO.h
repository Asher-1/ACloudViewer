// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ecvFeature.h>

#include "CV_io.h"

/**
 * @file FeatureIO.h
 * @brief Feature descriptor file I/O utilities
 *
 * Provides functions for reading and writing feature descriptors,
 * which are typically used for point cloud registration and matching.
 * Features can be FPFH, SHOT, or other descriptor types.
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Read feature descriptors from file (general entrance)
 *
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param feature Output feature descriptor object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadFeature(const std::string &filename,
                               utility::Feature &feature);

/**
 * @brief Write feature descriptors to file (general entrance)
 *
 * Automatically selects appropriate writer based on file extension.
 * @param filename Output file path
 * @param feature Feature descriptor to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteFeature(const std::string &filename,
                                const utility::Feature &feature);

/**
 * @brief Read feature descriptors from binary file
 * @param filename Input binary file path
 * @param feature Output feature descriptor object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API ReadFeatureFromBIN(const std::string &filename,
                                      utility::Feature &feature);

/**
 * @brief Write feature descriptors to binary file
 * @param filename Output binary file path
 * @param feature Feature descriptor to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WriteFeatureToBIN(const std::string &filename,
                                     const utility::Feature &feature);

}  // namespace io
}  // namespace cloudViewer
