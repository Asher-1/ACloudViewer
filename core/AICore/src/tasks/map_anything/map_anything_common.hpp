// Shared helpers for map_anything (VGGT-Omega ggml inference).
#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdarg>
#include <map>
#include <string>
#include <vector>

namespace map_anything {

enum class LogLevel { Debug = 0, Info, Warn, Error };

// Gate for the log_* helpers below (default Info). Thread-safe enough for
// the AICore contract: the level is written once at context-load time.
void set_log_level(LogLevel level);

// Thread-local log tag override: a task that embeds this runtime (the
// querysplat decoder runs the shared VGGT-Omega encoder) sets it at its
// public entry points so the shared loader/runtime log lines carry the
// CALLING task's identity instead of [map_anything]. NULL or empty
// restores the default tag. The vlog_*_tag entry points below ignore it
// (they carry an explicit tag).
void set_log_tag(const char* tag);

// A weight tensor staged on the host, straight out of the GGUF file.
struct HostTensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};
    std::vector<uint8_t> data;  // raw bytes in ggml layout
};

inline size_t nbytes_of(enum ggml_type type, const int64_t ne[4]) {
    const size_t tsz = ggml_type_size(type);
    const int64_t nelements = ne[0] * ne[1] * ne[2] * ne[3];
    // Block types pack GGML_BLCK_SIZE elements per block along ne0.
    const int bs = ggml_blck_size(type);
    return (size_t)((nelements / bs) * tsz);
}

// Simple stderr logging with a map_anything prefix.
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

// Tagged variants for the sibling tasks that share this runtime (the
// querysplat decoder runs on the same staged weights and helpers): the
// tag identifies the emitting task, so a shared runtime must not
// masquerade every message as [map_anything]. The va_list entry points
// let a task header wrap them into its own fixed-tag log_info/log_warn/
// log_error without a per-task translation unit.
void vlog_info_tag(const char* tag, const char* fmt, va_list args);
void vlog_warn_tag(const char* tag, const char* fmt, va_list args);
void vlog_error_tag(const char* tag, const char* fmt, va_list args);

// Load a whole binary file (raw f32 image buffers for the CLI).
std::vector<uint8_t> read_file_bytes(const std::string& path);

// Dump a ggml tensor's contents to a raw file (parity debugging).
bool dump_tensor(const struct ggml_tensor* t, const std::string& path);

}  // namespace map_anything
