// Real-model SAM3 CPU/Vulkan acceptance probe for a ggml upgrade worktree.
//
// Usage:
//   bench_sam3_backend_acceptance <sam3-visual-f16.gguf> <image> [runs=10]
//                                 [backend=all|cpu|vulkan]
//
// The probe uses a fixed box prompt and requires every run to return a finite,
// non-empty PVS mask. It executes load -> encode -> PVS -> tracker setup /
// propagation per run, records stage latency, and compares Vulkan to the CPU
// reference: mask IoU >= 0.98, max box error <= 1 px, score error <= 5e-3.

#include <strings.h>

#include <QImage>
#include <QImageReader>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "aicore/sam3_capi.h"

namespace {

constexpr int kWarmupRuns = 2;
// CPU vs Vulkan mask IoU gate. Measured 0.985 on sam3-visual-f16 @1008px
// (RTX 3060): this is the FIRST valid parity number for windowed attention —
// before ggml 0.21 + the runtime win_part dispatch, ggml-vulkan silently
// skipped WIN_PART/WIN_UNPART nodes, so no valid GPU reference existed.
// fp16 Vulkan flash-attention vs f32 CPU accounts for the ~1.5% gap.
constexpr double kMinMaskIou = 0.98;
constexpr float kMaxBoxErrorPx = 1.0f;
// Presence score = sigmoid(logit). fp16 Vulkan vs f32 CPU shifts the logit by
// ~0.1 around 21, i.e. an absolute score delta of ~1.5e-3 (measured). 5e-3
// keeps a 3x margin over backend numerics while still catching divergence.
constexpr float kMaxScoreError = 5e-3f;

struct Image {
    std::vector<uint8_t> rgb;
    int width = 0;
    int height = 0;
};

struct MaskResult {
    std::vector<uint8_t> mask;
    aicore_sam3_box box{};
    float score = 0.0f;
    int nonzero = 0;
    bool valid = false;
};

struct Timings {
    std::vector<double> load;
    std::vector<double> encode;
    std::vector<double> pvs;
    std::vector<double> track;
    std::vector<double> total;
    long long peak_vram_mib = -1;
    std::string backend;
};

bool load_image(const char *path, Image *out) {
    QImageReader reader(QString::fromUtf8(path));
    reader.setAutoTransform(true);
    const QImage decoded = reader.read();
    if (decoded.isNull()) return false;
    const QImage rgb = decoded.convertToFormat(QImage::Format_RGB888);
    out->width = rgb.width();
    out->height = rgb.height();
    out->rgb.resize(static_cast<size_t>(out->width) * out->height * 3);
    for (int y = 0; y < out->height; ++y) {
        std::memcpy(out->rgb.data() + static_cast<size_t>(y) * out->width * 3,
                    rgb.constScanLine(y), static_cast<size_t>(out->width) * 3);
    }
    return true;
}

long long gpu_vram_mib() {
#if defined(__linux__)
    FILE *pipe =
            popen("nvidia-smi --query-gpu=memory.used "
                  "--format=csv,noheader,nounits 2>/dev/null",
                  "r");
    if (!pipe) return -1;
    long long mib = -1;
    (void)std::fscanf(pipe, "%lld", &mib);
    (void)pclose(pipe);
    return mib;
#else
    return -1;
#endif
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(q * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

void print_timing(const char *name, const std::vector<double> &samples) {
    std::printf("\"%s\":{\"median_ms\":%.3f,\"p95_ms\":%.3f}", name,
                percentile(samples, 0.5), percentile(samples, 0.95));
}

double mask_iou(const MaskResult &a, const MaskResult &b) {
    if (!a.valid || !b.valid || a.mask.size() != b.mask.size()) return 0.0;
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a.mask.size(); ++i) {
        const bool av = a.mask[i] != 0;
        const bool bv = b.mask[i] != 0;
        inter += av && bv;
        uni += av || bv;
    }
    return uni == 0 ? 0.0 : static_cast<double>(inter) / uni;
}

float max_box_error(const aicore_sam3_box &a, const aicore_sam3_box &b) {
    return std::max({std::fabs(a.x0 - b.x0), std::fabs(a.y0 - b.y0),
                     std::fabs(a.x1 - b.x1), std::fabs(a.y1 - b.y1)});
}

bool copy_result(aicore_sam3_seg_result *result, MaskResult *out) {
    if (!result || aicore_sam3_seg_det_count(result) <= 0) return false;
    const aicore_sam3_plane_view view = aicore_sam3_seg_mask_at(result, 0);
    if (!view.data || view.width <= 0 || view.height <= 0 ||
        view.row_stride_bytes < static_cast<size_t>(view.width))
        return false;
    out->mask.resize(static_cast<size_t>(view.width) * view.height);
    out->nonzero = 0;
    const auto *data = static_cast<const uint8_t *>(view.data);
    for (int y = 0; y < view.height; ++y) {
        const auto *row = data + static_cast<size_t>(y) * view.row_stride_bytes;
        auto *dst = out->mask.data() + static_cast<size_t>(y) * view.width;
        std::memcpy(dst, row, static_cast<size_t>(view.width));
        out->nonzero += static_cast<int>(std::count_if(
                dst, dst + view.width, [](uint8_t v) { return v != 0; }));
    }
    out->box = aicore_sam3_seg_det_box_at(result, 0);
    out->score = aicore_sam3_seg_det_score_at(result, 0);
    out->valid = std::isfinite(out->score) && std::isfinite(out->box.x0) &&
                 std::isfinite(out->box.y0) && std::isfinite(out->box.x1) &&
                 std::isfinite(out->box.y1) && out->nonzero > 0;
    return out->valid;
}

bool run_backend(const char *device,
                 const char *model_path,
                 const Image &image,
                 int total_runs,
                 MaskResult *representative,
                 Timings *timings) {
    // The qSAM3 integration log records this successful cand1.jpg box. Keep
    // it proportional so the probe remains valid if the decoded image scale
    // changes while avoiding an arbitrary prompt that could yield no object.
    aicore_sam3_pvs_prompt prompt{};
    prompt.box = {image.width * (947.0f / 1920.0f),
                  image.height * (515.0f / 1280.0f),
                  image.width * (1084.0f / 1920.0f),
                  image.height * (623.0f / 1280.0f)};
    prompt.use_box = 1;
    const size_t stride = static_cast<size_t>(image.width) * 3;

    for (int run = -kWarmupRuns; run < total_runs; ++run) {
        const auto begin = std::chrono::steady_clock::now();
        aicore_sam3_options *options = aicore_sam3_options_new();
        aicore_sam3_options_set_device(options, device);
        aicore_sam3_options_set_threads(options, 4);
        aicore_sam3_ctx *ctx = aicore_sam3_load_opts(model_path, options);
        aicore_sam3_options_free(options);
        const auto loaded = std::chrono::steady_clock::now();
        if (!ctx || !aicore_sam3_is_ready(ctx)) {
            std::fprintf(stderr, "[%s] load failed: %s\n", device,
                         ctx ? aicore_sam3_last_error(ctx)
                             : aicore_sam3_last_load_error());
            aicore_sam3_free(ctx);
            return false;
        }
        timings->backend = aicore_sam3_context_backend_name(ctx);
        if (std::strcmp(device, "vulkan") == 0 &&
            strncasecmp(timings->backend.c_str(), "vulkan", 6) != 0) {
            std::fprintf(stderr, "[%s] requested backend resolved to %s\n",
                         device, timings->backend.c_str());
            aicore_sam3_free(ctx);
            return false;
        }
        timings->peak_vram_mib =
                std::max(timings->peak_vram_mib, gpu_vram_mib());

        if (aicore_sam3_encode_rgb(ctx, image.rgb.data(), image.width,
                                   image.height, stride, 1) != 0) {
            std::fprintf(stderr, "[%s] encode failed: %s\n", device,
                         aicore_sam3_last_error(ctx));
            aicore_sam3_free(ctx);
            return false;
        }
        const auto encoded = std::chrono::steady_clock::now();
        aicore_sam3_seg_result *pvs =
                aicore_sam3_segment_pvs_rgb(ctx, &prompt, image.rgb.data(),
                                            image.width, image.height, stride);
        const auto segmented = std::chrono::steady_clock::now();
        MaskResult result;
        const bool pvs_ok = copy_result(pvs, &result);
        aicore_sam3_seg_result_free(pvs);
        if (!pvs_ok) {
            std::fprintf(stderr, "[%s] PVS failed/non-finite/empty: %s\n",
                         device, aicore_sam3_last_error(ctx));
            aicore_sam3_free(ctx);
            return false;
        }

        aicore_sam3_tracker_ctx *tracker = aicore_sam3_tracker_create(ctx);
        const auto tracker_begin = std::chrono::steady_clock::now();
        const bool visual_only = aicore_sam3_context_visual_only(ctx) != 0;
        aicore_sam3_seg_result *frame =
                tracker ? (visual_only
                                   ? aicore_sam3_propagate_frame(
                                             tracker, image.rgb.data(),
                                             image.width, image.height, stride)
                                   : aicore_sam3_track_frame(
                                             tracker, image.rgb.data(),
                                             image.width, image.height, stride))
                        : nullptr;
        const bool frame_ok = frame != nullptr;
        aicore_sam3_seg_result_free(frame);
        if (!tracker || !frame_ok ||
            aicore_sam3_tracker_add_instance(tracker, &prompt) < 0) {
            std::fprintf(stderr, "[%s] tracker setup failed: %s\n", device,
                         tracker ? aicore_sam3_tracker_last_error(tracker)
                                 : "tracker allocation failed");
            aicore_sam3_tracker_free(tracker);
            aicore_sam3_free(ctx);
            return false;
        }
        aicore_sam3_seg_result *seeded =
                aicore_sam3_tracker_segment_pvs(tracker, &prompt);
        MaskResult seeded_result;
        const bool seeded_ok = copy_result(seeded, &seeded_result);
        aicore_sam3_seg_result_free(seeded);
        if (!seeded_ok) {
            std::fprintf(stderr,
                         "[%s] tracker PVS failed/non-finite/empty: %s\n",
                         device, aicore_sam3_tracker_last_error(tracker));
            aicore_sam3_tracker_free(tracker);
            aicore_sam3_free(ctx);
            return false;
        }
        aicore_sam3_seg_result *tracked =
                visual_only ? aicore_sam3_propagate_frame(
                                      tracker, image.rgb.data(), image.width,
                                      image.height, stride)
                            : aicore_sam3_track_frame(tracker, image.rgb.data(),
                                                      image.width, image.height,
                                                      stride);
        const auto tracker_end = std::chrono::steady_clock::now();
        const bool tracked_ok = tracked != nullptr;
        aicore_sam3_seg_result_free(tracked);
        if (!tracked_ok) {
            std::fprintf(stderr, "[%s] tracker propagation failed: %s\n",
                         device, aicore_sam3_tracker_last_error(tracker));
            aicore_sam3_tracker_free(tracker);
            aicore_sam3_free(ctx);
            return false;
        }
        aicore_sam3_tracker_free(tracker);
        aicore_sam3_free(ctx);
        timings->peak_vram_mib =
                std::max(timings->peak_vram_mib, gpu_vram_mib());

        if (run >= 0) {
            timings->load.push_back(
                    std::chrono::duration<double, std::milli>(loaded - begin)
                            .count());
            timings->encode.push_back(
                    std::chrono::duration<double, std::milli>(encoded - loaded)
                            .count());
            timings->pvs.push_back(std::chrono::duration<double, std::milli>(
                                           segmented - encoded)
                                           .count());
            timings->track.push_back(std::chrono::duration<double, std::milli>(
                                             tracker_end - tracker_begin)
                                             .count());
            timings->total.push_back(std::chrono::duration<double, std::milli>(
                                             tracker_end - begin)
                                             .count());
            if (run == 0) *representative = std::move(result);
        }
    }
    return true;
}

void print_report(const char *device, const Timings &t) {
    std::printf("\"%s\":{\"backend\":\"%s\",\"peak_vram_mib\":%lld,", device,
                t.backend.c_str(), t.peak_vram_mib);
    print_timing("load", t.load);
    std::printf(",");
    print_timing("encode", t.encode);
    std::printf(",");
    print_timing("pvs", t.pvs);
    std::printf(",");
    print_timing("track", t.track);
    std::printf(",");
    print_timing("total", t.total);
    std::printf("}");
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <sam3-visual-f16.gguf> <image> [runs=10] "
                     "[backend=all|cpu|vulkan]\n",
                     argv[0]);
        return 2;
    }
    const int runs = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 10;
    const std::string selected = argc >= 5 ? argv[4] : "all";
    if (selected != "all" && selected != "cpu" && selected != "vulkan") {
        std::fprintf(stderr, "unknown backend selector: %s\n",
                     selected.c_str());
        return 2;
    }
    Image image;
    if (!load_image(argv[2], &image)) {
        std::fprintf(stderr, "unable to decode image: %s\n", argv[2]);
        return 2;
    }
    MaskResult cpu, vulkan;
    Timings cpu_t, vulkan_t;
    const bool cpu_requested = selected == "all" || selected == "cpu";
    const bool vk_requested = selected == "all" || selected == "vulkan";
    const bool cpu_ok = !cpu_requested ||
                        run_backend("cpu", argv[1], image, runs, &cpu, &cpu_t);
    const bool vk_ok = !vk_requested || run_backend("vulkan", argv[1], image,
                                                    runs, &vulkan, &vulkan_t);
    const double iou = mask_iou(cpu, vulkan);
    const float box_error = max_box_error(cpu.box, vulkan.box);
    const float score_error = std::fabs(cpu.score - vulkan.score);
    const bool parity_checked = cpu_requested && vk_requested;
    const bool gates_ok = parity_checked
                                  ? cpu_ok && vk_ok && iou >= kMinMaskIou &&
                                            box_error <= kMaxBoxErrorPx &&
                                            score_error <= kMaxScoreError
                                  : cpu_ok && vk_ok;
    std::printf(
            "{\"runs\":%d,\"backend_selector\":\"%s\","
            "\"cpu_nonzero\":%d,\"vulkan_nonzero\":%d,"
            "\"mask_iou\":%.8f,\"box_error_px\":%.6f,\"score_error\":%.8f,"
            "\"parity_checked\":%s,\"gates_passed\":%s,",
            runs, selected.c_str(), cpu.nonzero, vulkan.nonzero, iou, box_error,
            score_error, parity_checked ? "true" : "false",
            gates_ok ? "true" : "false");
    print_report("cpu", cpu_t);
    std::printf(",");
    print_report("vulkan", vulkan_t);
    std::printf("}\n");
    return gates_ok ? 0 : 1;
}
