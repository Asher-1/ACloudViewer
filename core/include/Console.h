// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
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

#include <Eigen/Core>
#include <string>
#include <vector>

#include "CVCoreLib.h"

namespace cloudViewer {
namespace utility {

std::string CV_CORE_LIB_API
GetProgramOptionAsString(int argc,
                         char **argv,
                         const std::string &option,
                         const std::string &default_value = "");

int CV_CORE_LIB_API GetProgramOptionAsInt(int argc,
                                          char **argv,
                                          const std::string &option,
                                          const int default_value = 0);

double CV_CORE_LIB_API
GetProgramOptionAsDouble(int argc,
                         char **argv,
                         const std::string &option,
                         const double default_value = 0.0);

Eigen::VectorXd CV_CORE_LIB_API GetProgramOptionAsEigenVectorXd(
        int argc,
        char **argv,
        const std::string &option,
        const Eigen::VectorXd default_value = Eigen::VectorXd::Zero(0));

bool CV_CORE_LIB_API ProgramOptionExists(int argc,
                                         char **argv,
                                         const std::string &option);

bool CV_CORE_LIB_API ProgramOptionExistsAny(
        int argc, char **argv, const std::vector<std::string> &options);

}  // namespace utility
}  // namespace cloudViewer
