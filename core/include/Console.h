// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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
