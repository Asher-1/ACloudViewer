// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "map_anything_common.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace map_anything {

namespace {
std::atomic<int> g_log_level{(int)LogLevel::Info};
}  // namespace

void set_log_level(LogLevel level) {
    g_log_level.store((int)level, std::memory_order_relaxed);
}

void log_info(const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Info)
        return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[map_anything] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void log_warn(const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Warn)
        return;
    va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "[map_anything:WARN] ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

void log_error(const char* fmt, ...) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Error)
        return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[map_anything:ERROR] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        log_error("cannot open file: %s", path.c_str());
        return {};
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)size);
    if (size > 0 && fread(buf.data(), 1, (size_t)size, f) != (size_t)size) {
        log_error("short read on %s", path.c_str());
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

bool dump_tensor(const ggml_tensor* t, const std::string& path) {
    if (!t || !t->data) {
        log_error("dump_tensor: null tensor or data for %s", path.c_str());
        return false;
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        log_error("dump_tensor: cannot open %s", path.c_str());
        return false;
    }
    const size_t n = ggml_nbytes(t);
    const size_t written = fwrite(t->data, 1, n, f);
    fclose(f);
    if (written != n) {
        log_error("dump_tensor: short write on %s", path.c_str());
        return false;
    }
    return true;
}

}  // namespace map_anything
