#pragma once

// ----------------------------------------------------------------------------
// ALIKED task logging gate.
//
// Error / warning paths route through aicore_log.hpp, which forwards to CVLog
// when the AICore target is built with AICore_HAS_CVLOG (ACloudViewer
// Console) and falls back to stderr otherwise. Low-level stage/profile
// diagnostics stay on stderr on purpose: they are machine-parseable bench
// output consumed by core/AICore/tests/yolo/bench_compare.py and the GPU
// stage benchmarks.
// ----------------------------------------------------------------------------

#include "common/aicore_log.hpp"


#define ALIKED_LOG(...) AICORE_LOG_PRINT("[aliked] ", __VA_ARGS__)
#define ALIKED_LOG_ERR(...) AICORE_LOG_ERROR("[aliked] ", __VA_ARGS__)
#define ALIKED_LOG_WARN(...) AICORE_LOG_WARN("[aliked] ", __VA_ARGS__)
