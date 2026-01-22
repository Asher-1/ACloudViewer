// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// CV_CORE_LIB
#include <ClassMap.h>

// LOCAL
#include "recognition/DeepSemanticSegmentation.h"
#include "utility/PythonModules.h"

namespace PythonInterface {
bool CV_PYTHON_LIB_API SetPythonHome(const wchar_t* pyHome);
bool CV_PYTHON_LIB_API SetPythonHome(const char* pyHome);
}  // namespace PythonInterface
