// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aicore/pipeline_timing.h"

namespace wwv {

enum class LogLevel : int {
    kDebug = 0,
    kInfo = 1,
    kWarn = 2,
    kError = 3,
};

// Sets the process-wide log level (explicit API, never an environment read).
void set_log_level(LogLevel level);
LogLevel log_level();

// printf-style logging through the level gate; prefixes [wwv:debug] etc.
void logf(LogLevel level, const char* fmt, ...)
        __attribute__((format(printf, 2, 3)));

// Monotonic wall clock in milliseconds.
double now_ms();

// Returns a fully-populated timings struct with only the given fields set.
aicore_pipeline_timings make_timings(double preprocess_ms,
                                     double inference_ms,
                                     double postprocess_ms,
                                     double serialization_ms,
                                     double e2e_ms);

// Reads a whole binary file; false with *err set on failure.
bool read_file_bytes(const std::string& path,
                     std::vector<uint8_t>* out,
                     std::string* err);

// Minimal JSON string escaping (quotes, backslash, control chars).
std::string json_escape(const std::string& text);

// Writes raw bytes atomically enough for debug dumps (tmp + rename).
bool write_file_bytes(const std::string& path, const void* data, size_t size);

// Dumps an f32 buffer as <dir>/<name>.bin for the parity tooling.
bool dump_f32(const std::string& dir,
              const std::string& name,
              const float* data,
              size_t count);

}  // namespace wwv
