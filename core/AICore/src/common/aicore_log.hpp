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

// Unified runtime log-level filtering (thread-local). The AICORE_LOG_* macros
// above are unconditional; callers that need a runtime threshold (e.g. a
// session log_level option) must go through aicore_log_at. This is the single
// place where severity maps onto the CVLog levels, so tasks do not each
// re-implement a private level enum + dispatch.
#define AICORE_LOG_LEVEL_DEBUG 0
#define AICORE_LOG_LEVEL_INFO 1
#define AICORE_LOG_LEVEL_WARN 2
#define AICORE_LOG_LEVEL_ERROR 3

#ifdef __cplusplus
extern "C" {
#endif
/** Set the thread-local minimum level (messages below it are dropped). */
void aicore_set_log_level(int level);
/** Get the thread-local minimum level (default INFO = 1). */
int aicore_get_log_level(void);
/** Format and forward at an explicit level (see AICORE_LOG_LEVEL_*). */
void aicore_log_at(int level, const char* tag, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
        __attribute__((format(printf, 3, 4)))
#endif
        ;
#ifdef __cplusplus
}
#endif
