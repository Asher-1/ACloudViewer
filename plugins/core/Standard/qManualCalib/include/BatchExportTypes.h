// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <functional>
#include <string>

namespace mcalib {

struct BatchExportProgress {
    using ReportFn = std::function<bool(
            int completed, int total, const std::string& label)>;
    ReportFn report;
    std::atomic<bool>* cancel_flag = nullptr;
};

struct BatchExportResult {
    int exported = 0;
    int total = 0;
    bool cancelled = false;

    bool ok() const { return exported > 0; }
};

}  // namespace mcalib
