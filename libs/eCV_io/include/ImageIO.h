// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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
