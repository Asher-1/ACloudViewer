// Task-identity log helpers for the querysplat task.
//
// The querysplat decoder runs on the shared map-anything-ggml runtime and
// consumes its helpers directly, but the runtime's fixed log tag would make
// every querysplat message masquerade as [map_anything]. These wrappers keep
// the shared level gating while stamping the emitting task's own tag.
//
// Header-only by design: the task directory is GLOBbed for *.cpp, so a
// translation unit here would need a CMake edit for no functional gain; the
// va_list entry points of the shared logger make the wrappers trivial.
#pragma once

#include "map_anything_common.hpp"

#include <cstdarg>

namespace qs {

constexpr const char* kLogTag = "[querysplat] ";
constexpr const char* kLogTagWarn = "[querysplat:WARN] ";
constexpr const char* kLogTagError = "[querysplat:ERROR] ";

inline void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    map_anything::vlog_info_tag(kLogTag, fmt, args);
    va_end(args);
}

inline void log_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    map_anything::vlog_warn_tag(kLogTagWarn, fmt, args);
    va_end(args);
}

inline void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    map_anything::vlog_error_tag(kLogTagError, fmt, args);
    va_end(args);
}

}  // namespace qs
