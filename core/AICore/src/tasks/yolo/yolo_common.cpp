// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/yolo/yolo_common.hpp"

#include <cstdarg>
#include <cstdio>

namespace yolo {

void set_log_level(int level) { aicore_set_log_level(level); }
int get_log_level() { return aicore_get_log_level(); }

void logf(int level, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // Forward through the shared AICore log gate (CVLog when built into
    // ACloudViewer, stderr otherwise) with runtime level filtering.
    aicore_log_at(level, "[YOLO] ", "%s", buf);
}

}  // namespace yolo
