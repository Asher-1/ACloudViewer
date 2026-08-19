// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "yolo_common.hpp"

#include <cstdarg>
#include <cstdio>

namespace yolo {

int g_log_level = (int)LogLevel::INFO;

void logf(LogLevel level, const char* fmt, ...) {
    if ((int)level < g_log_level) return;
    static const char* kPrefix[] = {"[YOLO][debug]", "[YOLO][info]",
                                    "[YOLO][warn]", "[YOLO][error]"};
    std::fprintf(stderr, "%s ", kPrefix[(int)level]);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

}  // namespace yolo
