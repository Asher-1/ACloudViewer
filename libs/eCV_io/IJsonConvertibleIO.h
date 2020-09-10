// ----------------------------------------------------------------------------
// -                                    ECV_DB                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.CVLib.org
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

#ifndef ECV_IJSONCONVERTIBLE_IO_HEADER
#define ECV_IJSONCONVERTIBLE_IO_HEADER

// LOCAL
#include "eCV_io.h"

// CV_CORE_LIB
#include <IJsonConvertible.h>

// SYSTEM
#include <string>

namespace cloudViewer {
namespace io {

/// The general entrance for reading an IJsonConvertible from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ECV_IO_LIB_API ReadIJsonConvertible(const std::string &filename,
                          CVLib::utility::IJsonConvertible &object);

/// The general entrance for writing an IJsonConvertible to a file
/// The function calls write functions based on the extension name of filename.
/// \return return true if the write function is successful, false otherwise.
bool ECV_IO_LIB_API WriteIJsonConvertible(const std::string &filename,
                           const CVLib::utility::IJsonConvertible &object);

bool ECV_IO_LIB_API ReadIJsonConvertibleFromJSON(const std::string &filename,
                                  CVLib::utility::IJsonConvertible &object);

bool ECV_IO_LIB_API WriteIJsonConvertibleToJSON(const std::string &filename,
                                 const CVLib::utility::IJsonConvertible &object);

bool ECV_IO_LIB_API ReadIJsonConvertibleFromJSONString(const std::string &json_string,
                                        CVLib::utility::IJsonConvertible &object);

bool ECV_IO_LIB_API WriteIJsonConvertibleToJSONString(std::string &json_string,
                                       const CVLib::utility::IJsonConvertible &object);

}  // namespace io
}  // namespace cloudViewer

#endif // ECV_IJSONCONVERTIBLE_IO_HEADER