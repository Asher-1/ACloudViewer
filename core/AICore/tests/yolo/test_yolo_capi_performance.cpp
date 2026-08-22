// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// YOLO C API performance benchmark — integrated-side counterpart of the
// upstream ultralytics-ggml `yolo-cli bench` matrix.
//
// Timing contract (must stay 1:1 with upstream cpp_ggml/src/cli.cpp bench):
//   preprocess_ms  = letterbox (+ canvas no-op check)
//   graph_ms       = session_run + output readback
//   post_ms        = postprocess (+seg proto readback/mask compose, +depth
//                    restore)
//   e2e_ms         = all of the above; JSON serialization is EXCLUDED
//                    (reported separately as json_ms) because upstream has no
//                    JSON stage and the plugin hot path uses typed results.
//
// Assets (location-only env vars; unset => skip with 77):
//   AICORE_TEST_YOLO_MODELS_DIR  directory with *.gguf (all tasks/dtypes)
//   AICORE_TEST_YOLO_IMAGE       benchmark image (upstream uses bus.jpg)
//   AICORE_TEST_YOLO_DEVICE      device request (default env_or "auto")
//   AICORE_TEST_YOLO_THREADS     CPU thread count (default 0 = 1 thread; set
//                                to match the upstream matrix for CPU rows)
//   AICORE_TEST_YOLO_WARMUP      warmup iterations (default 20)
//   AICORE_TEST_YOLO_ITERS       timed iterations (default 50)

#include <dirent.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/yolo_capi.h"

namespace {

const char* env_or(const char* primary, const char* fallback) {
    const char* value = std::getenv(primary);
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv(fallback);
    }
    return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

struct Stats {
    std::vector<double> ms;
    double mean = 0, p50 = 0, p90 = 0;
    void push(double v) { ms.push_back(v); }
    void finish() {
        if (ms.empty()) return;
        std::sort(ms.begin(), ms.end());
        const size_t n = ms.size();
        double sum = 0;
        for (double v : ms) sum += v;
        mean = sum / (double)n;
        p50 = ms[n / 2];
        p90 = ms[std::min(n - 1, n * 9 / 10)];
    }
};

std::vector<std::string> list_ggufs(const std::string& dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return out;
    while (dirent* e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// One benchmark row, printed as JSONL on stdout (upstream-compatible field
// names so diff tooling can join on name/dtype/backend).
void emit_row(const char* build,
              const std::string& file,
              const char* task,
              const char* dtype,
              const char* device,
              int threads,
              int warmup,
              int iters,
              const Stats& preprocess,
              const Stats& graph,
              const Stats& post,
              const Stats& e2e,
              long long sanity) {
    std::printf(
            "{\"suite\":\"aicore\",\"build\":\"%s\",\"file\":\"%s\","
            "\"task\":\"%s\",\"dtype\":\"%s\",\"device\":\"%s\","
            "\"threads\":%d,\"warmup\":%d,\"iters\":%d,"
            "\"preprocess_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},"
            "\"graph_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},"
            "\"post_ms\":{\"mean\":%.3f,\"p50\":%.3f},"
            "\"e2e_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},"
            "\"sanity\":%lld}\n",
            build, file.c_str(), task, dtype, device, threads, warmup, iters,
            preprocess.mean, preprocess.p50, preprocess.p90, graph.mean,
            graph.p50, graph.p90, post.mean, post.p50, e2e.mean, e2e.p50,
            e2e.p90, sanity);
    std::fflush(stdout);
}

int bench_model(const std::string& gguf,
                const uint8_t* rgb,
                int w,
                int h,
                const char* device,
                int threads,
                int warmup,
                int iters) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (opts == nullptr) return 1;
    aicore_yolo_options_set_device(opts, device);
    if (threads > 0) aicore_yolo_options_set_threads(opts, threads);
    if (std::getenv("AICORE_TEST_YOLO_PROFILE") != nullptr) {
        aicore_yolo_options_set_profile_ops(opts, 1);
    }
    aicore_yolo_ctx* ctx = aicore_yolo_load_opts(gguf.c_str(), opts);
    aicore_yolo_options_free(opts);
    if (ctx == nullptr || !aicore_yolo_is_ready(ctx)) {
        std::printf("[yolo-perf] skip %s: %s\n", gguf.c_str(),
                    ctx != nullptr && aicore_yolo_last_error(ctx)
                            ? aicore_yolo_last_error(ctx)
                            : "load failed");
        aicore_yolo_free(ctx);
        return 0;  // backend/model unavailable is a skip, not a failure
    }

    const char* task = aicore_yolo_context_task(ctx);
    const char* dtype = "f32";
    {
        // dtype from filename suffix, same convention as the catalog
        const size_t pos = gguf.rfind('/');
        const std::string base =
                pos == std::string::npos ? gguf : gguf.substr(pos + 1);
        if (base.find("-f16.") != std::string::npos) dtype = "f16";
        if (base.find("-q8_0.") != std::string::npos) dtype = "q8_0";
    }

    // Warmup (also covers the first-call canvas rebuild).
    for (int i = 0; i < warmup; ++i) {
        if (std::strcmp(task, "detect") == 0) {
            char* j = aicore_yolo_detect_rgb_json(ctx, rgb, w, h);
            if (j == nullptr) {
                std::fprintf(stderr, "[yolo-perf] detect failed: %s\n",
                             aicore_yolo_last_error(ctx)
                                     ? aicore_yolo_last_error(ctx)
                                     : "?");
                aicore_yolo_free(ctx);
                return 1;
            }
            aicore_yolo_free_buffer(j);
        } else if (std::strcmp(task, "depth") == 0) {
            int32_t dw = 0, dh = 0;
            float* m = aicore_yolo_depth_rgb(ctx, rgb, w, h, &dw, &dh);
            if (m == nullptr) {
                std::fprintf(stderr, "[yolo-perf] depth failed: %s\n",
                             aicore_yolo_last_error(ctx)
                                     ? aicore_yolo_last_error(ctx)
                                     : "?");
                aicore_yolo_free(ctx);
                return 1;
            }
            aicore_yolo_free_buffer(m);
        } else if (std::strcmp(task, "segment") == 0) {
            aicore_yolo_segment_result* r = aicore_yolo_seg_rgb(ctx, rgb, w, h);
            if (r == nullptr) {
                std::fprintf(stderr, "[yolo-perf] segment failed: %s\n",
                             aicore_yolo_last_error(ctx)
                                     ? aicore_yolo_last_error(ctx)
                                     : "?");
                aicore_yolo_free(ctx);
                return 1;
            }
            aicore_yolo_seg_result_free(r);
        } else {
            std::printf("[yolo-perf] skip %s: unsupported task=%s\n",
                        gguf.c_str(), task);
            aicore_yolo_free(ctx);
            return 0;
        }
    }

    Stats preprocess, graph, post, e2e;
    long long sanity = -1;
    for (int i = 0; i < iters; ++i) {
        aicore_yolo_timings t{};
        if (std::strcmp(task, "detect") == 0) {
            char* j = aicore_yolo_detect_rgb_json(ctx, rgb, w, h);
            if (j == nullptr || aicore_yolo_last_timings(ctx, &t) != 0) {
                aicore_yolo_free_buffer(j);
                aicore_yolo_free(ctx);
                return 1;
            }
            // count detections from the JSON (cheap sanity signal)
            if (j != nullptr) {
                long long n = 0;
                for (const char* p = j;
                     (p = std::strstr(p, "\"box\":")) != nullptr; p += 6) {
                    ++n;
                }
                sanity = n;
            }
            aicore_yolo_free_buffer(j);
        } else if (std::strcmp(task, "depth") == 0) {
            int32_t dw = 0, dh = 0;
            float* m = aicore_yolo_depth_rgb(ctx, rgb, w, h, &dw, &dh);
            if (m == nullptr || aicore_yolo_last_timings(ctx, &t) != 0) {
                aicore_yolo_free_buffer(m);
                aicore_yolo_free(ctx);
                return 1;
            }
            sanity = (long long)dw * dh;
            aicore_yolo_free_buffer(m);
        } else {  // segment
            aicore_yolo_segment_result* r = aicore_yolo_seg_rgb(ctx, rgb, w, h);
            if (r == nullptr || aicore_yolo_last_timings(ctx, &t) != 0) {
                aicore_yolo_seg_result_free(r);
                aicore_yolo_free(ctx);
                return 1;
            }
            sanity = aicore_yolo_seg_det_count(r);
            aicore_yolo_seg_result_free(r);
        }
        preprocess.push(t.preprocess_ms);
        graph.push(t.inference_ms);
        post.push(t.postprocess_ms);
        e2e.push(t.e2e_ms);
    }
    preprocess.finish();
    graph.finish();
    post.finish();
    e2e.finish();

    const size_t pos = gguf.rfind('/');
    const std::string base =
            pos == std::string::npos ? gguf : gguf.substr(pos + 1);
    emit_row("aicore", base, task, dtype, aicore_yolo_context_device(ctx),
             aicore_yolo_context_threads(ctx), warmup, iters, preprocess, graph,
             post, e2e, sanity);
    aicore_yolo_free(ctx);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const char* models_dir =
            env_or("AICORE_TEST_YOLO_MODELS_DIR", "AICORE_TEST_YOLO_DIR");
    const char* image = env_or("AICORE_TEST_YOLO_IMAGE", "AICORE_TEST_IMAGE");
    const char* device =
            env_or("AICORE_TEST_YOLO_DEVICE", "AICORE_TEST_DEVICE");
    if (device == nullptr) device = "auto";
    const char* warmup_env = std::getenv("AICORE_TEST_YOLO_WARMUP");
    const int warmup = std::max(1, warmup_env ? std::atoi(warmup_env) : 20);
    const char* iters_env = std::getenv("AICORE_TEST_YOLO_ITERS");
    const int iters = std::max(1, iters_env ? std::atoi(iters_env) : 50);
    const char* threads_env = std::getenv("AICORE_TEST_YOLO_THREADS");
    const int threads = threads_env ? std::atoi(threads_env) : 0;

    // A single model can be given directly (argv[1] or AICORE_TEST_YOLO_GGUF).
    std::vector<std::string> models;
    if (argc > 1) {
        models.push_back(argv[1]);
    } else if (const char* single = std::getenv("AICORE_TEST_YOLO_GGUF")) {
        if (single[0]) models.push_back(single);
    } else if (models_dir != nullptr) {
        models = list_ggufs(models_dir);
    }
    if (models.empty() || image == nullptr) {
        std::printf(
                "[yolo-perf] skipped: AICORE_TEST_YOLO_MODELS_DIR/GGUF and "
                "AICORE_TEST_YOLO_IMAGE are required\n");
        return 77;
    }

    uint8_t* rgb = nullptr;
    int32_t w = 0, h = 0;
    if (aicore_yolo_load_path_rgb(image, &rgb, &w, &h) != 0 || rgb == nullptr) {
        std::fprintf(stderr, "[yolo-perf] failed to load image %s\n", image);
        return 1;
    }

    int rc = 0;
    for (const std::string& m : models) {
        rc |= bench_model(m, rgb, w, h, device, threads, warmup, iters);
    }
    aicore_yolo_free_buffer(reinterpret_cast<float*>(rgb));
    return rc;
}
