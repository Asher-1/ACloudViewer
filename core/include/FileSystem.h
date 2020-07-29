// ----------------------------------------------------------------------------
// -                        CVLib: www.CVLib.org                            -
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

#ifndef CV_FILESYSTEM_HEADER
#define CV_FILESYSTEM_HEADER

#include "CVCoreLib.h"
#include <string>
#include <vector>

namespace CVLib {
namespace utility {
namespace filesystem {

std::string CV_CORE_LIB_API GetFileExtensionInLowerCase(const std::string &filename);

std::string CV_CORE_LIB_API GetFileNameWithoutExtension(const std::string &filename);

std::string CV_CORE_LIB_API GetFileNameWithoutDirectory(const std::string &filename);

std::string CV_CORE_LIB_API GetFileParentDirectory(const std::string &filename);

std::string CV_CORE_LIB_API GetRegularizedDirectoryName(const std::string &directory);

std::string CV_CORE_LIB_API GetWorkingDirectory();

bool CV_CORE_LIB_API ChangeWorkingDirectory(const std::string &directory);

bool CV_CORE_LIB_API DirectoryExists(const std::string &directory);

bool CV_CORE_LIB_API MakeDirectory(const std::string &directory);

bool CV_CORE_LIB_API MakeDirectoryHierarchy(const std::string &directory);

bool CV_CORE_LIB_API DeleteDirectory(const std::string &directory);

bool CV_CORE_LIB_API FileExists(const std::string &filename);

bool CV_CORE_LIB_API RemoveFile(const std::string &filename);

bool CV_CORE_LIB_API ListFilesInDirectory(const std::string &directory,
                          std::vector<std::string> &filenames);

bool CV_CORE_LIB_API ListFilesInDirectoryWithExtension(const std::string &directory,
                                       const std::string &extname,
                                       std::vector<std::string> &filenames);

// wrapper for fopen that enables unicode paths on Windows
CV_CORE_LIB_API FILE* FOpen(const std::string &filename, const std::string &mode);

}  // namespace filesystem
}  // namespace utility
}  // namespace CVLib

#endif // CV_FILESYSTEM_HEADER
