#pragma once
#include <cstdio>
#include <cstdlib>
#include "common/aicore_log.hpp"


// Routed through the shared AICore log gate (CVLog when built into
// ACloudViewer, stderr otherwise).
#define FD_LOG(...) AICORE_LOG_PRINT("[facedetect] ", __VA_ARGS__)

// Hard runtime precondition: log the failed expression with file:line and abort.
// Used for invariants whose violation would silently corrupt results (e.g. a
// zero det_scale mapping every decoded box to +inf) rather than fail loudly.
#define FD_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            AICORE_LOG_ERROR("[facedetect] assertion failed: %s (%s:%d)", \
                             #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)
