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

#ifndef ECV_FEATURE_IO_HEADER
#define ECV_FEATURE_IO_HEADER

#include "eCV_io.h"

#include <ecvFeature.h>

namespace cloudViewer {
namespace io {

/// The general entrance for reading a Feature from a file
/// \return If the read function is successful.
bool ECV_IO_LIB_API ReadFeature(const std::string &filename, utility::Feature &feature);

/// The general entrance for writing a Feature to a file
/// \return If the write function is successful.
bool ECV_IO_LIB_API WriteFeature(const std::string &filename,
                                 const utility::Feature &feature);

bool ECV_IO_LIB_API ReadFeatureFromBIN(const std::string &filename,
                                       utility::Feature &feature);

bool ECV_IO_LIB_API WriteFeatureToBIN(const std::string &filename,
                                      const utility::Feature &feature);

}  // namespace io
}  // namespace cloudViewer

#endif // ECV_FEATURE_IO_HEADER
