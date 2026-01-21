// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Image.h>

#include <string>

#include "CV_io.h"

namespace cloudViewer {
namespace io {

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<geometry::Image> CV_IO_LIB_API
CreateImageFromFile(const std::string &filename);

/// Factory function to create an image from memory.
std::shared_ptr<geometry::Image> CV_IO_LIB_API
CreateImageFromMemory(const std::string &image_format,
                      const unsigned char *image_data_ptr,
                      size_t image_data_size);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API ReadImage(const std::string &filename,
                             geometry::Image &image);

/// The general entrance for reading an Image from memory
/// The function calls read functions based on format of image.
/// \param image_format the format of image, "png" or "jpg".
/// \param image_data_ptr the pointer to image data in memory.
/// \param image_data_size the size of image data in memory.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API ReadImageFromMemory(const std::string &image_format,
                                       const unsigned char *image_data_ptr,
                                       size_t image_data_size,
                                       geometry::Image &image);

constexpr int kCloudViewerImageIODefaultQuality = -1;

/// The general entrance for writing an Image to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports quality, the parameter will be used.
/// Otherwise it will be ignored.
/// \param quality: PNG: [0-9] <=2 fast write for storing intermediate data
///                            >=3 (default) normal write for balanced speed and
///                            file size
///                 JPEG: [0-100] Typically in [70,95]. 90 is default (good
///                 quality).
/// \return return true if the write function is successful, false otherwise.
bool CV_IO_LIB_API WriteImage(const std::string &filename,
                              const geometry::Image &image,
                              int quality = kCloudViewerImageIODefaultQuality);

bool CV_IO_LIB_API ReadImageFromPNG(const std::string &filename,
                                    geometry::Image &image);

/// Read a PNG image from memory.
/// \param image_data_ptr the pointer to image data in memory.
/// \param image_data_size the size of image data in memory.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API ReadPNGFromMemory(const unsigned char *image_data_ptr,
                                     size_t image_data_size,
                                     geometry::Image &image);

bool CV_IO_LIB_API
WriteImageToPNG(const std::string &filename,
                const geometry::Image &image,
                int quality = kCloudViewerImageIODefaultQuality);

bool CV_IO_LIB_API ReadImageFromJPG(const std::string &filename,
                                    geometry::Image &image);

/// Read a JPG image from memory.
/// \param image_data_ptr the pointer to image data in memory.
/// \param image_data_size the size of image data in memory.
/// \return return true if the read function is successful, false otherwise.
bool CV_IO_LIB_API ReadJPGFromMemory(const unsigned char *image_data_ptr,
                                     size_t image_data_size,
                                     geometry::Image &image);

bool CV_IO_LIB_API
WriteImageToJPG(const std::string &filename,
                const geometry::Image &image,
                int quality = kCloudViewerImageIODefaultQuality);

}  // namespace io
}  // namespace cloudViewer
