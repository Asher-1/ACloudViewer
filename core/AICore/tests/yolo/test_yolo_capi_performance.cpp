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
//   AICORE_TEST_YOLO_CLASSES     comma-separated open-vocabulary class list
//                                (world/yoloe models; e.g. "person,bus,car")
//   AICORE_TEST_YOLO_TEXT_MODEL  text-encoder GGUF path for --classes
//                                (clip-ViT-B-32 / mobileclip2_b)

// dirent.h is POSIX-only; the vendored 3rdparty/dirent (tronkko dirent)
// provides the Windows-compatible implementation (same pattern as
// core/src/FileSystem.cpp).
#ifdef _WIN32
#include <dirent/dirent.h>
#else
#include <dirent.h>
#endif

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

// One inference dispatch covering every task family. Returns false on
// failure (error already logged); `sanity` receives a cheap task-specific
// output signal (detection count / mask pixels / map size / top prob).
bool run_once(aicore_yolo_ctx* ctx,
              const char* task,
              const uint8_t* rgb,
              int w,
              int h,
              long long* sanity) {
    if (std::strcmp(task, "detect") == 0) {
        char* j = aicore_yolo_detect_rgb_json(ctx, rgb, w, h);
        if (j == nullptr) return false;
        long long n = 0;
        for (const char* p = j; (p = std::strstr(p, "\"box\":")) != nullptr;
             p += 6) {
            ++n;
        }
        *sanity = n;
        aicore_yolo_free_buffer(j);
        return true;
    }
    if (std::strcmp(task, "depth") == 0) {
        int32_t dw = 0, dh = 0;
        float* m = aicore_yolo_depth_rgb(ctx, rgb, w, h, &dw, &dh);
        if (m == nullptr) return false;
        *sanity = (long long)dw * dh;
        aicore_yolo_free_buffer(m);
        return true;
    }
    if (std::strcmp(task, "segment") == 0) {
        aicore_yolo_segment_result* r = aicore_yolo_seg_rgb(ctx, rgb, w, h);
        if (r == nullptr) return false;
        long long bits = 0;
        for (int i = 0; i < aicore_yolo_seg_det_count(r); ++i) {
            const aicore_yolo_plane_view v = aicore_yolo_seg_mask_at(r, i);
            const auto* bytes = static_cast<const uint8_t*>(v.data);
            for (size_t b = 0;
                 bytes != nullptr && b < v.row_stride_bytes * v.height; ++b)
                bits += bytes[b] != 0;
        }
        *sanity = aicore_yolo_seg_det_count(r) * 1000000 + bits % 1000000;
        aicore_yolo_seg_result_free(r);
        return true;
    }
    if (std::strcmp(task, "pose") == 0) {
        aicore_yolo_pose_result* r = aicore_yolo_pose_rgb(ctx, rgb, w, h);
        if (r == nullptr) return false;
        *sanity = (long long)aicore_yolo_pose_det_count(r) *
                  aicore_yolo_pose_kpt_count(r);
        aicore_yolo_pose_result_free(r);
        return true;
    }
    if (std::strcmp(task, "obb") == 0) {
        aicore_yolo_obb_result* r = aicore_yolo_obb_rgb(ctx, rgb, w, h);
        if (r == nullptr) return false;
        *sanity = aicore_yolo_obb_count(r);
        aicore_yolo_obb_result_free(r);
        return true;
    }
    if (std::strcmp(task, "semantic") == 0) {
        aicore_yolo_semantic_result* r =
                aicore_yolo_semantic_rgb(ctx, rgb, w, h);
        if (r == nullptr) return false;
        const aicore_yolo_plane_view v = aicore_yolo_semantic_class_map(r);
        *sanity = (long long)v.width * v.height *
                  aicore_yolo_semantic_num_classes(r);
        aicore_yolo_semantic_result_free(r);
        return true;
    }
    if (std::strcmp(task, "classify") == 0) {
        aicore_yolo_classify_result* r =
                aicore_yolo_classify_rgb(ctx, rgb, w, h);
        if (r == nullptr) return false;
        // Top probability x1000 as the sanity signal.
        float top = 0.0f;
        const int n = aicore_yolo_classify_count(r);
        for (int i = 0; i < n; ++i) {
            const float p = aicore_yolo_classify_prob_at(r, i);
            if (p > top) top = p;
        }
        *sanity = (long long)(top * 1000.0f);
        aicore_yolo_classify_result_free(r);
        return true;
    }
    return false;
}

int bench_model(const std::string& gguf,
                const uint8_t* rgb,
                int w,
                int h,
                const char* device,
                int threads,
                int warmup,
                int iters,
                const char* classes_env,
                const char* text_model_env) {
    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (opts == nullptr) return 1;
    aicore_yolo_options_set_device(opts, device);
    if (threads > 0) aicore_yolo_options_set_threads(opts, threads);
    if (std::getenv("AICORE_TEST_YOLO_PROFILE") != nullptr) {
        aicore_yolo_options_set_profile_ops(opts, 1);
    }
    // Open-vocabulary models: classes + text encoder ride into the load
    // (matching the plugin's world/yoloe tabs).
    std::vector<std::string> class_storage;
    std::vector<const char*> class_ptrs;
    if (classes_env != nullptr && classes_env[0] != '\0') {
        std::string item;
        for (const char* p = classes_env;; ++p) {
            if (*p == ',' || *p == '\0') {
                class_storage.push_back(item);
                class_ptrs.push_back(class_storage.back().c_str());
                item.clear();
                if (*p == '\0') break;
            } else {
                item.push_back(*p);
            }
        }
        aicore_yolo_options_set_classes(opts, class_ptrs.data(),
                                        (int32_t)class_ptrs.size());
        if (text_model_env != nullptr && text_model_env[0] != '\0') {
            aicore_yolo_options_set_text_model(opts, text_model_env);
        }
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
    long long sanity0 = -1;
    for (int i = 0; i < warmup; ++i) {
        if (!run_once(ctx, task, rgb, w, h, &sanity0)) {
            std::fprintf(stderr, "[yolo-perf] %s failed: %s\n", task,
                         aicore_yolo_last_error(ctx)
                                 ? aicore_yolo_last_error(ctx)
                                 : "?");
            aicore_yolo_free(ctx);
            return 1;
        }
    }

    Stats preprocess, graph, post, e2e;
    long long sanity = -1;
    for (int i = 0; i < iters; ++i) {
        aicore_yolo_timings t{};
        sanity = -1;
        if (!run_once(ctx, task, rgb, w, h, &sanity) ||
            aicore_yolo_last_timings(ctx, &t) != 0) {
            aicore_yolo_free(ctx);
            return 1;
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
    const char* classes_env = std::getenv("AICORE_TEST_YOLO_CLASSES");
    const char* text_model_env = std::getenv("AICORE_TEST_YOLO_TEXT_MODEL");

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
        rc |= bench_model(m, rgb, w, h, device, threads, warmup, iters,
                          classes_env, text_model_env);
    }
    aicore_yolo_free_buffer(reinterpret_cast<float*>(rgb));
    return rc;
}
