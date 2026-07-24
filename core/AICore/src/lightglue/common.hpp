#pragma once

#include "aicore_log.hpp"

#define LG_LOG(...) AICORE_LOG_PRINT("[LG] ", __VA_ARGS__)
#define LG_DEBUG_LOG(...) AICORE_LOG_DEBUG("[LG] ", __VA_ARGS__)
#define LG_WARN(...) AICORE_LOG_WARN("[LG] ", __VA_ARGS__)
#define LG_ERR(...) AICORE_LOG_ERROR("[LG] ", __VA_ARGS__)
