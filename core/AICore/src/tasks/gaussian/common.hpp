#pragma once

#include "aicore_log.hpp"

#define FS_LOG(...) AICORE_LOG_PRINT("[FS] ", __VA_ARGS__)
#define FS_DEBUG_LOG(...) AICORE_LOG_DEBUG("[FS] ", __VA_ARGS__)
#define FS_WARN(...) AICORE_LOG_WARN("[FS] ", __VA_ARGS__)
#define FS_ERR(...) AICORE_LOG_ERROR("[FS] ", __VA_ARGS__)
