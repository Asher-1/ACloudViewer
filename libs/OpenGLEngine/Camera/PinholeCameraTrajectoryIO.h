// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include "qGL.h"
#include <string>

#include "Camera/PinholeCameraTrajectory.h"

namespace cloudViewer {
namespace io {

/// Factory function to create a PinholeCameraTrajectory from a file
/// (PinholeCameraTrajectoryFactory.cpp)
/// Return an empty PinholeCameraTrajectory if fail to read the file.
std::shared_ptr<camera::PinholeCameraTrajectory>
OPENGL_ENGINE_LIB_API
CreatePinholeCameraTrajectoryFromFile(const std::string &filename);

/// The general entrance for reading a PinholeCameraTrajectory from a file
/// The function calls read functions based on the extension name of filename.
/// \return If the read function is successful.
bool OPENGL_ENGINE_LIB_API ReadPinholeCameraTrajectory(const std::string &filename,
                                 camera::PinholeCameraTrajectory &trajectory);

/// The general entrance for writing a PinholeCameraTrajectory to a file
/// The function calls write functions based on the extension name of filename.
/// \return If the write function is successful.
bool OPENGL_ENGINE_LIB_API WritePinholeCameraTrajectory(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

bool OPENGL_ENGINE_LIB_API ReadPinholeCameraTrajectoryFromLOG(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory);

bool OPENGL_ENGINE_LIB_API WritePinholeCameraTrajectoryToLOG(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

bool OPENGL_ENGINE_LIB_API ReadPinholeCameraTrajectoryFromTUM(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory);

bool OPENGL_ENGINE_LIB_API WritePinholeCameraTrajectoryToTUM(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory);

}  // namespace io
}  // namespace cloudViewer
