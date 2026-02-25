// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "CV_io.h"
#include "camera/PinholeCameraTrajectory.h"

/**
 * @file PinholeCameraTrajectoryIO.h
 * @brief Camera trajectory file I/O utilities
 *
 * Provides functions for reading and writing pinhole camera trajectories,
 * which are sequences of camera poses (position and orientation) used in
 * SLAM, structure-from-motion, and RGB-D reconstruction applications.
 *
 * Supported formats:
 * - LOG: CloudViewer native format
 * - TUM: TUM RGB-D benchmark format
 */

namespace cloudViewer {
namespace io {

/**
 * @brief Factory function to create camera trajectory from file
 *
 * Automatically detects file format from extension.
 * @param filename Input file path
 * @return Shared pointer to loaded trajectory (empty if loading fails)
 */
std::shared_ptr<camera::PinholeCameraTrajectory> CV_IO_LIB_API
CreatePinholeCameraTrajectoryFromFile(const std::string &filename);

/**
 * @brief Read camera trajectory from file (general entrance)
 *
 * Automatically selects appropriate reader based on file extension.
 * @param filename Input file path
 * @param trajectory Output camera trajectory object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
ReadPinholeCameraTrajectory(const std::string &filename,
                            camera::PinholeCameraTrajectory &trajectory);

/**
 * @brief Write camera trajectory to file (general entrance)
 *
 * Automatically selects appropriate writer based on file extension.
 * @param filename Output file path
 * @param trajectory Camera trajectory to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
WritePinholeCameraTrajectory(const std::string &filename,
                             const camera::PinholeCameraTrajectory &trajectory);

/**
 * @brief Read camera trajectory from LOG format file
 *
 * LOG is CloudViewer's native camera trajectory format.
 * @param filename Input LOG file path
 * @param trajectory Output camera trajectory object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
ReadPinholeCameraTrajectoryFromLOG(const std::string &filename,
                                   camera::PinholeCameraTrajectory &trajectory);

/**
 * @brief Write camera trajectory to LOG format file
 * @param filename Output LOG file path
 * @param trajectory Camera trajectory to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

/**
 * @brief Read camera trajectory from TUM format file
 *
 * TUM format is used by the TUM RGB-D benchmark dataset.
 * Format: timestamp tx ty tz qx qy qz qw
 * @param filename Input TUM file path
 * @param trajectory Output camera trajectory object
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API
ReadPinholeCameraTrajectoryFromTUM(const std::string &filename,
                                   camera::PinholeCameraTrajectory &trajectory);

/**
 * @brief Write camera trajectory to TUM format file
 * @param filename Output TUM file path
 * @param trajectory Camera trajectory to write
 * @return true if successful, false otherwise
 */
bool CV_IO_LIB_API WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

}  // namespace io
}  // namespace cloudViewer
