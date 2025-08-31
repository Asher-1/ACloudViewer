// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "eCV_io.h"
#include <string>

#include <Image.h>

namespace cloudViewer {
namespace io {

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<geometry::Image> ECV_IO_LIB_API CreateImageFromFile(const std::string &filename);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ECV_IO_LIB_API ReadImage(const std::string &filename, geometry::Image &image);

constexpr int kCloudViewerImageIODefaultQuality = -1;

/// The general entrance for writing an Image to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports quality, the parameter will be used.
/// Otherwise it will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool ECV_IO_LIB_API WriteImage(const std::string &filename,
                               const geometry::Image &image,
                               int quality = kCloudViewerImageIODefaultQuality);

bool ECV_IO_LIB_API ReadImageFromPNG(const std::string &filename, geometry::Image &image);

bool ECV_IO_LIB_API WriteImageToPNG(const std::string &filename,
                                    const geometry::Image &image,
                                    int quality = kCloudViewerImageIODefaultQuality);

bool ECV_IO_LIB_API ReadImageFromJPG(const std::string &filename, geometry::Image &image);

bool ECV_IO_LIB_API WriteImageToJPG(const std::string &filename,
                                    const geometry::Image &image,
                                    int quality = kCloudViewerImageIODefaultQuality);

}  // namespace io
}  // namespace cloudViewer
