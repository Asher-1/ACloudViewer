#pragma once

#include <cstdio>

#ifdef AICore_HAS_CVLOG
#include <CVLog.h>
#define AICORE_LOG_PRINT(tag, ...) CVLog::Print(tag __VA_ARGS__)
#define AICORE_LOG_DEBUG(tag, ...) CVLog::PrintDebug(tag __VA_ARGS__)
#define AICORE_LOG_WARN(tag, ...) CVLog::Warning(tag __VA_ARGS__)
#define AICORE_LOG_ERROR(tag, ...) CVLog::Error(tag __VA_ARGS__)
#else
#define AICORE_LOG_PRINT(tag, ...)                                           \
    do {                                                                     \
        std::fprintf(stderr, tag __VA_ARGS__);                               \
        std::fprintf(stderr, "\n");                                          \
    } while (0)
#define AICORE_LOG_DEBUG(tag, ...) AICORE_LOG_PRINT(tag, __VA_ARGS__)
#define AICORE_LOG_WARN(tag, ...) AICORE_LOG_PRINT(tag, __VA_ARGS__)
#define AICORE_LOG_ERROR(tag, ...) AICORE_LOG_PRINT(tag, __VA_ARGS__)
#endif
