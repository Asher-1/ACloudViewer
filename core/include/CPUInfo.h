// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "CVCoreLib.h"

namespace cloudViewer {
namespace utility {

/// \brief CPU information.
class CV_CORE_LIB_API CPUInfo {
public:
    struct CV_CORE_LIB_API Impl {
        int num_cores_;
        int num_threads_;
    };

public:
    static CPUInfo& GetInstance();

    ~CPUInfo() = default;
    CPUInfo(const CPUInfo&) = delete;
    void operator=(const CPUInfo&) = delete;

    /// Returns the number of physical CPU cores.
    /// This is similar to boost::thread::physical_concurrency().
    int NumCores() const;

    /// Returns the number of logical CPU cores.
    /// This returns the same result as std::thread::hardware_concurrency() or
    /// boost::thread::hardware_concurrency().
    int NumThreads() const;

    /// Prints CPUInfo in the console.
    void Print() const;

private:
    CPUInfo();
    std::unique_ptr<Impl> impl_;
};

}  // namespace utility
}  // namespace cloudViewer
