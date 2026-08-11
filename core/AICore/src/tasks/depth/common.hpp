#pragma once

#include "aicore_log.hpp"

#define DA_LOG(...) AICORE_LOG_PRINT("[DA3] ", __VA_ARGS__)
#define DA_DEBUG_LOG(...) AICORE_LOG_DEBUG("[DA3] ", __VA_ARGS__)
#define DA_WARN(...) AICORE_LOG_WARN("[DA3] ", __VA_ARGS__)
#define DA_ERR(...) AICORE_LOG_ERROR("[DA3] ", __VA_ARGS__)
