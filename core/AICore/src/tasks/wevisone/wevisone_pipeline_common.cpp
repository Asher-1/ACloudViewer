// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/wevisone/wevisone_pipeline_common.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>

#include "common/aicore_log.hpp"

namespace wwv {

// The level gate and the CVLog/console dispatch are owned by the shared
// AICore log runtime (thread-local threshold); the task keeps only its tag.
void set_log_level(LogLevel level) {
    aicore_set_log_level(static_cast<int>(level));
}

LogLevel log_level() { return static_cast<LogLevel>(aicore_get_log_level()); }

void logf(LogLevel level, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char buf[1024];
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // aicore_log_at concatenates the literal tag and formats fmt separately;
    // passing the tagged string as fmt would print a literal "%s".
    aicore_log_at(static_cast<int>(level), "[AICore wevisone] ", "%s\n", buf);
}

double now_ms() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

aicore_pipeline_timings make_timings(double preprocess_ms,
                                     double inference_ms,
                                     double postprocess_ms,
                                     double serialization_ms,
                                     double e2e_ms) {
    aicore_pipeline_timings t{};
    t.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    t.valid_fields = AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                     AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    t.preprocess_ms = preprocess_ms;
    t.inference_ms = inference_ms;
    t.postprocess_ms = postprocess_ms;
    if (serialization_ms > 0.0) {
        t.valid_fields |= AICORE_TIMING_SERIALIZATION;
        t.serialization_ms = serialization_ms;
    }
    t.e2e_ms = e2e_ms;
    return t;
}

bool read_file_bytes(const std::string& path,
                     std::vector<uint8_t>* out,
                     std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        if (err != nullptr) *err = "cannot open " + path;
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
        std::fclose(f);
        if (err != nullptr) *err = "cannot stat " + path;
        return false;
    }
    out->resize(static_cast<size_t>(size));
    const size_t got =
            size > 0 ? std::fread(out->data(), 1, out->size(), f) : 0;
    std::fclose(f);
    if (got != out->size()) {
        if (err != nullptr) *err = "short read on " + path;
        return false;
    }
    return true;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

bool write_file_bytes(const std::string& path, const void* data, size_t size) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) return false;
    const size_t put = size > 0 ? std::fwrite(data, 1, size, f) : 0;
    std::fclose(f);
    if (put != size) {
        std::remove(tmp.c_str());
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

bool dump_f32(const std::string& dir,
              const std::string& name,
              const float* data,
              size_t count) {
    if (dir.empty()) return true;
    return write_file_bytes(dir + "/" + name + ".bin", data,
                            count * sizeof(float));
}

}  // namespace wwv
