// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include "CVCoreLib.h"

#include "Logging.h"  //for ConsoleProgressBar

namespace cloudViewer {
namespace utility {

/// Progress reporting through update_progress(double percent) function.
/// If you have a set number of items to process (or bytes to load),
/// CountingProgressReporter will convert that to percentages (you still have to
/// specify how many items you have, of course)
class CV_CORE_LIB_API CountingProgressReporter {
public:
    CountingProgressReporter(std::function<bool(double)> f) {
        update_progress_ = f;
    }
    void SetTotal(int64_t total) { total_ = total; }
    bool Update(int64_t count) {
        if (!update_progress_) return true;
        last_count_ = count;
        double percent = 0;
        if (total_ > 0) {
            if (count < total_) {
                percent = count * 100.0 / total_;
            } else {
                percent = 100.0;
            }
        }
        return CallUpdate(percent);
    }
    void Finish() { CallUpdate(100); }
    // for compatibility with ConsoleProgressBar
    void operator++() { Update(last_count_ + 1); }

private:
    bool CallUpdate(double percent) {
        if (update_progress_) {
            return update_progress_(percent);
        }
        return true;
    }
    std::function<bool(double)> update_progress_;
    int64_t total_ = -1;
    int64_t last_count_ = -1;
};

using utility::ConsoleProgressBar;
/// update_progress(double percent) functor for ConsoleProgressBar
struct CV_CORE_LIB_API ConsoleProgressUpdater {
    ConsoleProgressUpdater(const std::string &progress_info,
                           bool active = false)
        : progress_bar_(100, progress_info, active) {}
    bool operator()(double pct) {
        while (last_pct_ < pct) {
            ++last_pct_;
            ++progress_bar_;
        }
        return true;
    }

private:
    ConsoleProgressBar progress_bar_;
    int last_pct_ = 0;
};

}  // namespace utility
}  // namespace cloudViewer
