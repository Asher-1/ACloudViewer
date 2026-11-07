// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKDT ggml runtime - common utilities (implementation).

#include "tasks/gkd/gkd_common.hpp"

#include <algorithm>
#include <cstdarg>
#include <filesystem>
#include <vector>

namespace gkd {

static LogLevel g_level = LogLevel::Info;

void set_log_level(LogLevel level) { g_level = level; }

void logf(LogLevel level, const char* fmt, ...) {
    if (level < g_level) return;
    const char* tag = "[GKD]";
    switch (level) {
        case LogLevel::Debug:
            tag = "[GKD:dbg]";
            break;
        case LogLevel::Info:
            tag = "[GKD]";
            break;
        case LogLevel::Warn:
            tag = "[GKD:warn]";
            break;
        case LogLevel::Error:
            tag = "[GKD:err]";
            break;
    }
    std::fprintf(stderr, "%s ", tag);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch())
            .count();
}

void dump_f32(const std::string& path,
              const std::vector<int64_t>& shape,
              const float* data) {
    // create parent directories so a fresh dump path works
    size_t slash = path.rfind('/');
    if (slash != std::string::npos) {
        std::error_code ec;
        std::filesystem::create_directories(path.substr(0, slash), ec);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        GKD_LOG_ERROR("cannot open %s for writing", path.c_str());
        return;
    }
    int64_t ndims = (int64_t)shape.size();
    std::fwrite(&ndims, sizeof(int64_t), 1, f);
    std::fwrite(shape.data(), sizeof(int64_t), shape.size(), f);
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
}

}  // namespace gkd
