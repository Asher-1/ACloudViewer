// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/utility/CompilerInfo.h"

#include <Logging.h>

#include <memory>
#include <string>

namespace cloudViewer {
namespace utility {

CompilerInfo::CompilerInfo() {}

CompilerInfo& CompilerInfo::GetInstance() {
    static CompilerInfo instance;
    return instance;
}

std::string CompilerInfo::CXXStandard() const {
    return std::string(CLOUDVIEWER_CXX_STANDARD);
}

std::string CompilerInfo::CXXCompilerId() const {
    return std::string(CLOUDVIEWER_CXX_COMPILER_ID);
}

std::string CompilerInfo::CXXCompilerVersion() const {
    return std::string(CLOUDVIEWER_CXX_COMPILER_VERSION);
}

std::string CompilerInfo::CUDACompilerId() const {
    return std::string(CLOUDVIEWER_CUDA_COMPILER_ID);
}

std::string CompilerInfo::CUDACompilerVersion() const {
    return std::string(CLOUDVIEWER_CUDA_COMPILER_VERSION);
}

void CompilerInfo::Print() const {
#ifdef BUILD_CUDA_MODULE
    utility::LogInfo("CompilerInfo: C++ {}, {} {}, {} {}, SYCL disabled.",
                     CXXStandard(), CXXCompilerId(), CXXCompilerVersion(),
                     CUDACompilerId(), CUDACompilerVersion());
#else
#ifdef BUILD_SYCL_MODULE
    utility::LogInfo(
            "CompilerInfo: C++ {}, {} {}, CUDA disabled, SYCL enabled.",
            CXXStandard(), CXXCompilerId(), CXXCompilerVersion());
#else
    utility::LogInfo(
            "CompilerInfo: C++ {}, {} {}, CUDA disabled, SYCL disabled",
            CXXStandard(), CXXCompilerId(), CXXCompilerVersion());
#endif
#endif
}

}  // namespace utility
}  // namespace cloudViewer
