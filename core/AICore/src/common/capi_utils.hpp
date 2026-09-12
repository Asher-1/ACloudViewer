// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Shared helpers for the per-task C API layers (src/tasks/*/capi.cpp).
//
// Every task exports the same mechanical plumbing under its own
// aicore_<task>_ symbol namespace (C ABI): malloc'd string duplication,
// JSON escaping, QImage -> packed-RGB conversion and the common
// device/threads option fields. This header is the single implementation
// those exports delegate to. The per-task C ABI function NAMES stay
// task-scoped (ABI compatibility); only the bodies converge.
//
// Exception fencing: most capi.cpp catch blocks carry task- and stage-specific
// error context ("YOLO out of memory in depth post-processing") that a generic
// fence would erase, so those keep their hand-written try/catch. For fences
// with no per-stage context, catch_to_error() below is the single pattern:
// it runs F and funnels std::exception/what() (and unknown throws) into the
// task's last_error string, returning f's result.

#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "aicore/image_view.h"
#include "aicore/pipeline_timing.h"

class QImage;

namespace aicore {
namespace capi {

using PipelineClock = std::chrono::steady_clock;

inline void record_pipeline_e2e(aicore_pipeline_timings& timings,
                                PipelineClock::time_point start) {
    timings = {};
    timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    timings.valid_fields = AICORE_TIMING_E2E;
    timings.e2e_ms =
            std::chrono::duration<double, std::milli>(PipelineClock::now() -
                                                       start)
                    .count();
}

inline int copy_pipeline_timings(const aicore_pipeline_timings& timings,
                                 aicore_pipeline_timings* out) {
    if (out == nullptr || timings.abi_version == 0 ||
        timings.valid_fields == 0) {
        return -1;
    }
    *out = timings;
    return 0;
}

/** Run f() under a try/catch that records std::exception what() (or a
 *  generic "unknown exception") into last_error and returns f()'s result
 *  unchanged on success. Use only for fences WITHOUT per-stage error
 *  context; callers with resource cleanup or staged messages keep their
 *  hand-written catch. */
template <class F>
auto catch_to_error(std::string& last_error, F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown exception";
    }
    using Result = decltype(f());
    return Result{};
}

/** malloc'd C-string copy (ownership transfers to the caller; release with
 *  the task's aicore_<task>_free_string, which is std::free). Returns
 *  nullptr on allocation failure. */
char* dup_cstr(const std::string& s);

/** Escape a string for interpolation into a JSON string literal: quotes,
 *  backslash and control characters (`\n` `\r` `\t`, others as `\u00xx`). */
std::string json_escape(const std::string& s);

/** Tightly-packed RGB888 (HWC, 3 bytes/pixel) extracted from a QImage,
 *  malloc'd with std::malloc (release with the task's free_vec /
 *  free_buffer helper, which is std::free). Handles rows whose stride is
 *  wider than width*3. `data` is nullptr on an invalid image or OOM. */
struct PackedRgb {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
};
PackedRgb qimage_to_packed_rgb(const QImage& image);

/** Convert a borrowed public image view to packed RGB in one owned buffer.
 *  This is the common boundary for stride-aware Qt/OpenCV callers. */
bool image_view_to_packed_rgb(const aicore_image_view& view,
                              std::vector<uint8_t>& out);

/** The device/threads fields every aicore_<task>_options struct carries.
 *  Defaults: device "auto", threads 0 (backend default). Compose (do not
 *  inherit) into the per-task options struct and delegate the setters. */
struct CommonOptions {
    std::string device = "auto";
    int32_t threads = 0;
};

inline void set_device(CommonOptions& opts, const char* device) {
    if (device != nullptr) opts.device = device;
}

inline void set_threads(CommonOptions& opts, int n_threads) {
    opts.threads = n_threads;
}

}  // namespace capi
}  // namespace aicore
