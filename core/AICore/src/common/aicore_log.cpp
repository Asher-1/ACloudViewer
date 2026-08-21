// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "common/aicore_log.hpp"

#include <cstdarg>
#include <cstdio>

#ifdef AICore_HAS_CVLOG
#include <QString>
#endif

namespace {

// Thread-local minimum severity; messages below it are dropped. Default INFO
// matches the historical yolo::tls_log_level behavior.
thread_local int tls_log_level = AICORE_LOG_LEVEL_INFO;

}  // namespace

extern "C" {

void aicore_set_log_level(int level) { tls_log_level = level; }

int aicore_get_log_level(void) { return tls_log_level; }

void aicore_log_at(int level, const char* tag, const char* fmt, ...) {
    if (level < tls_log_level) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    // Severity maps onto the CVLog levels via the shared gate (CVLog when
    // built into ACloudViewer, stderr otherwise). The AICORE_LOG_* macros
    // cannot be used here because they paste `tag` into a string literal.
#ifdef AICore_HAS_CVLOG
    switch (level) {
        case AICORE_LOG_LEVEL_DEBUG:
            CVLog::PrintDebug(QString::fromUtf8("%1%2").arg(tag, buf));
            break;
        case AICORE_LOG_LEVEL_WARN:
            CVLog::Warning(QString::fromUtf8("%1%2").arg(tag, buf));
            break;
        case AICORE_LOG_LEVEL_ERROR:
            CVLog::Error(QString::fromUtf8("%1%2").arg(tag, buf));
            break;
        case AICORE_LOG_LEVEL_INFO:
        default:
            CVLog::Print(QString::fromUtf8("%1%2").arg(tag, buf));
            break;
    }
#else
    std::fprintf(stderr, "%s%s\n", tag, buf);
#endif
}

}  // extern "C"
