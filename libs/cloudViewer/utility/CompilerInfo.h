// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>

namespace cloudViewer {
namespace utility {

/// \brief Compiler information.
class CompilerInfo {
    // This does not need to be a class. It is a class just for the sake of
    // consistency with CPUInfo.
public:
    static CompilerInfo& GetInstance();

    ~CompilerInfo() = default;
    CompilerInfo(const CompilerInfo&) = delete;
    void operator=(const CompilerInfo&) = delete;

    std::string CXXStandard() const;

    std::string CXXCompilerId() const;
    std::string CXXCompilerVersion() const;

    std::string CUDACompilerId() const;
    std::string CUDACompilerVersion() const;

    void Print() const;

private:
    CompilerInfo();
};

}  // namespace utility
}  // namespace cloudViewer
