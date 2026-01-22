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

namespace cloudViewer {
namespace io {

/// Factory function to create a PinholeCameraTrajectory from a file
/// (PinholeCameraTrajectoryFactory.cpp)
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<camera::PinholeCameraTrajectory> CV_IO_LIB_API
CreatePinholeCameraTrajectoryFromFile(const std::string &filename);

/// The general entrance for reading a PinholeCameraTrajectory from a file
/// The function calls read functions based on the extension name of filename.
/// \return If the read function is successful.
bool CV_IO_LIB_API
ReadPinholeCameraTrajectory(const std::string &filename,
                            camera::PinholeCameraTrajectory &trajectory);

/// The general entrance for writing a PinholeCameraTrajectory to a file
/// The function calls write functions based on the extension name of filename.
/// \return If the write function is successful.
bool CV_IO_LIB_API
WritePinholeCameraTrajectory(const std::string &filename,
                             const camera::PinholeCameraTrajectory &trajectory);

bool CV_IO_LIB_API
ReadPinholeCameraTrajectoryFromLOG(const std::string &filename,
                                   camera::PinholeCameraTrajectory &trajectory);

bool CV_IO_LIB_API WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

bool CV_IO_LIB_API
ReadPinholeCameraTrajectoryFromTUM(const std::string &filename,
                                   camera::PinholeCameraTrajectory &trajectory);

bool CV_IO_LIB_API WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

}  // namespace io
}  // namespace cloudViewer
