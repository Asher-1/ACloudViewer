// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "tasks/map_anything/map_anything_common.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <string>

#include "FileSystem.h"

namespace map_anything {

namespace {
std::atomic<int> g_log_level{(int)LogLevel::Info};
// Thread-local task identity for the shared-runtime log helpers (see
// set_log_tag in the header). Owned strings keep the pointer valid.
thread_local std::string g_log_tag_override;
const char* effective_log_tag() {
    return g_log_tag_override.empty() ? "[map_anything] "
                                      : g_log_tag_override.c_str();
}
}  // namespace

void set_log_level(LogLevel level) {
    g_log_level.store((int)level, std::memory_order_relaxed);
}

void set_log_tag(const char* tag) { g_log_tag_override = tag ? tag : ""; }

void vlog_info_tag(const char* tag, const char* fmt, va_list args) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Info)
        return;
    fprintf(stderr, "%s", tag);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

void vlog_warn_tag(const char* tag, const char* fmt, va_list args) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Warn)
        return;
    fprintf(stderr, "%s", tag);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

void vlog_error_tag(const char* tag, const char* fmt, va_list args) {
    if (g_log_level.load(std::memory_order_relaxed) > (int)LogLevel::Error)
        return;
    fprintf(stderr, "%s", tag);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

void log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog_info_tag(effective_log_tag(), fmt, args);
    va_end(args);
}

void log_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog_warn_tag(effective_log_tag(), fmt, args);
    va_end(args);
}

void log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vlog_error_tag(effective_log_tag(), fmt, args);
    va_end(args);
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    FILE* f = cloudViewer::utility::filesystem::FOpen(path.c_str(), "rb");
    if (!f) {
        log_error("cannot open file: %s", path.c_str());
        return {};
    }
    size_t size = 0;
    if (!cloudViewer::utility::filesystem::FSize(f, size)) {
        log_error("cannot determine size of %s", path.c_str());
        fclose(f);
        return {};
    }
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
    FILE* f = cloudViewer::utility::filesystem::FOpen(path.c_str(), "wb");
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
