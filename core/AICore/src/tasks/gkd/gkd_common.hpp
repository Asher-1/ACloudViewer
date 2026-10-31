// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKDT ggml runtime - common utilities (in-tree port of the upstream
// General-Keypoint-Detection-GGML cpp_ggml/src/common.*; logging kept
// self-contained like every other AICore task).
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gkd {

enum class LogLevel { Debug = 0, Info, Warn, Error };

void set_log_level(LogLevel level);
void logf(LogLevel level, const char* fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

#define GKD_LOG_DEBUG(...) ::gkd::logf(::gkd::LogLevel::Debug, __VA_ARGS__)
#define GKD_LOG_INFO(...) ::gkd::logf(::gkd::LogLevel::Info, __VA_ARGS__)
#define GKD_LOG_WARN(...) ::gkd::logf(::gkd::LogLevel::Warn, __VA_ARGS__)
#define GKD_LOG_ERROR(...) ::gkd::logf(::gkd::LogLevel::Error, __VA_ARGS__)

// Monotonic wall clock in milliseconds.
double now_ms();

// Parity-dump helper: writes `[int64 ndims, int64 dim...] + row-major f32`.
// Only used when a debug dump dir is explicitly set through the options.
void dump_f32(const std::string& path, const std::vector<int64_t>& shape,
              const float* data);

}  // namespace gkd
