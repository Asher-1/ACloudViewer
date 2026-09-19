// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Unit tests for the shared GGUF open helper (src/common/gguf_file_io.hpp):
// success in both no_alloc modes, ownership round-trip, and the shared
// failure-message shape. Runs headless (no GPU, no model assets).

#include <chrono>
#include <cstdio>
#include <string>

#include "gguf.h"
#include "src/common/gguf_file_io.hpp"

namespace {

int failures = 0;

void CHECK(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "gguf_file_io: FAIL %s\n", what);
        ++failures;
    }
}

/** Write a minimal valid GGUF file with two scalar KV pairs. */
std::string write_synthetic_gguf(const std::string& path) {
    gguf_context* g = gguf_init_empty();
    if (!g) {
        std::fprintf(stderr, "gguf_file_io: FAIL gguf_init_empty\n");
        ++failures;
        return {};
    }
    gguf_set_val_str(g, "general.architecture", "test-arch");
    gguf_set_val_u32(g, "test.answer", 42);
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    return path;
}

int64_t find_key(const gguf_context* g, const char* key) {
    return gguf_find_key(g, key);
}

}  // namespace

int main() {
    // Unique per-process temp path (the test binary runs under ctest next to
    // other tests; a fixed name could race parallel runs).
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const std::string path =
            "aicore_test_gguf_file_io_" +
            std::to_string(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                            .count()) +
            ".gguf";
    std::remove(path.c_str());

    CHECK(write_synthetic_gguf(path) == path, "synthetic gguf written");

    // --- success, no_alloc=true: metadata context only -----------------------
    {
        ggml_context* meta = nullptr;
        std::string error;
        gguf_context* g = ggml_common::open_gguf_file(path, /*no_alloc=*/true,
                                                      &meta, "test", &error);
        CHECK(g != nullptr, "open no_alloc=true returns context");
        CHECK(meta != nullptr, "open no_alloc=true returns meta context");
        CHECK(error.empty(), "open success leaves error empty");
        if (g) {
            const int64_t arch = find_key(g, "general.architecture");
            CHECK(arch >= 0 &&
                          std::string(gguf_get_val_str(g, arch)) == "test-arch",
                  "KV round-trip through the helper");
            gguf_free(g);
        }
        if (meta) ggml_free(meta);
    }

    // --- success, no_alloc=false ---------------------------------------------
    {
        ggml_context* meta = nullptr;
        gguf_context* g = ggml_common::open_gguf_file(path, /*no_alloc=*/false,
                                                      &meta, "test", nullptr);
        CHECK(g != nullptr && meta != nullptr,
              "open no_alloc=false succeeds with error=nullptr");
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
    }

    // --- ownership: two independent opens round-trip and free cleanly --------
    {
        for (int round = 0; round < 2; ++round) {
            ggml_context* meta = nullptr;
            gguf_context* g = ggml_common::open_gguf_file(
                    path, /*no_alloc=*/true, &meta, "test", nullptr);
            CHECK(g != nullptr, "repeat open succeeds");
            if (g) gguf_free(g);
            if (meta) ggml_free(meta);
        }
    }

    // --- failure: missing file carries the shared message shape --------------
    {
        const std::string missing = path + ".does-not-exist";
        ggml_context* meta = nullptr;
        std::string error;
        gguf_context* g = ggml_common::open_gguf_file(
                missing, /*no_alloc=*/true, &meta, "some_task", &error);
        CHECK(g == nullptr, "missing file returns nullptr");
        CHECK(error == "some_task: failed to open GGUF '" + missing + "'",
              "failure message shape: <context>: failed to open GGUF '<path>'");
    }

    // --- failure: null context tag falls back to the generic prefix ----------
    {
        std::string error;
        (void)ggml_common::open_gguf_file(path + ".missing2",
                                          /*no_alloc=*/true, nullptr, nullptr,
                                          &error);
        const bool shape = error.rfind("gguf: failed to open GGUF '", 0) == 0;
        CHECK(shape, "null context tag defaults to the 'gguf' prefix");
    }

    std::remove(path.c_str());

    std::fprintf(stderr, "gguf_file_io: %s (%d failures)\n",
                 failures == 0 ? "ok" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}
