#pragma once
#include <cstdio>
#include <cstdlib>
#define FD_LOG(...)  do { std::fprintf(stderr, "[facedetect] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)

// Hard runtime precondition: log the failed expression with file:line and abort.
// Used for invariants whose violation would silently corrupt results (e.g. a
// zero det_scale mapping every decoded box to +inf) rather than fail loudly.
#define FD_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "[facedetect] assertion failed: %s (%s:%d)\n", \
                         #cond, __FILE__, __LINE__); \
            std::abort(); \
        } \
    } while (0)
