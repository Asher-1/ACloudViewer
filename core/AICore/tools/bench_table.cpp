// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Headless table reconstruction driver: image -> aicore_ocr_run (det + rec
// [+ layout]) -> aicore_table_run, one JSON line per image with the HTML,
// grid shape and stage timings, then a summary line with p50/p95.
// core/AICore/scripts/run_table_gt_matrix.py feeds it the GT image list and
// scores the HTML (TEDS / TEDS-S / structure exact match); the official
// counterpart is scripts/run_paddle_table_baseline.py.
//
// usage: bench_table --table <slanet.gguf> [--det <det.gguf> --rec <rec.gguf>
//                    [--layout <layout.gguf> --layout-regions]]
//                    [--device auto|cpu|cuda|vulkan] [--threads n]
//                    [--det-limit-type 0|1] [--det-max-side n]
//                    [--warmups n] [--list <file>] [images...]
//
// Without --det/--rec the tables carry structure only (TEDS-S). The image
// is decoded through the public aicore_map_anything_load_path_rgb helper.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/ocr_capi.h"
#include "aicore/table_capi.h"

namespace {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * (v.size() - 1))];
}

std::string jsonString(const char* s) {
    std::string out = "\"";
    for (const char* p = s ? s : ""; *p; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
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
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

}  // namespace

int main(int argc, char** argv) {
    const char* tablePath = nullptr;
    const char* detPath = nullptr;
    const char* recPath = nullptr;
    const char* layoutPath = nullptr;
    const char* device = "auto";
    int threads = 0, warmups = 1, detLimitType = 0, detMaxSide = 0;
    bool layoutRegions = false;
    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "bench_table: %s needs a value\n",
                             a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--table")
            tablePath = next();
        else if (a == "--det")
            detPath = next();
        else if (a == "--rec")
            recPath = next();
        else if (a == "--layout")
            layoutPath = next();
        else if (a == "--layout-regions")
            layoutRegions = true;
        else if (a == "--device")
            device = next();
        else if (a == "--threads")
            threads = std::atoi(next());
        else if (a == "--warmups")
            warmups = std::atoi(next());
        else if (a == "--det-limit-type")
            detLimitType = std::atoi(next());
        else if (a == "--det-max-side")
            detMaxSide = std::atoi(next());
        else if (a == "--list") {
            std::ifstream list(next());
            for (std::string line; std::getline(list, line);)
                if (!line.empty()) images.push_back(line);
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "bench_table: unknown option %s\n", a.c_str());
            return 2;
        } else {
            images.push_back(a);
        }
    }
    if (!tablePath || images.empty() || (!detPath) != (!recPath) ||
        (layoutRegions && !layoutPath)) {
        std::fprintf(stderr,
                     "usage: %s --table <gguf> [--det d --rec r [--layout l "
                     "--layout-regions]] [--device dev] [--list file] "
                     "[images...]\n",
                     argv[0]);
        return 2;
    }

    aicore_table_options topt{};
    topt.struct_size = sizeof(topt);
    topt.abi_version = AICORE_TABLE_ABI_VERSION;
    topt.model_path = tablePath;
    topt.device = device;
    topt.n_threads = threads;
    aicore_table_ctx* table = aicore_table_context_create(&topt);
    if (!table) {
        std::fprintf(stderr, "bench_table: %s\n", aicore_table_last_error());
        return 1;
    }
    aicore_ocr_ctx* ocr = nullptr;
    if (detPath) {
        aicore_ocr_options oopt{};
        oopt.struct_size = sizeof(oopt);
        oopt.abi_version = AICORE_OCR_ABI_VERSION;
        oopt.det_model_path = detPath;
        oopt.rec_model_path = recPath;
        oopt.layout_model_path = layoutPath;
        oopt.device = device;
        oopt.n_threads = threads;
        oopt.det_limit_type = detLimitType;
        oopt.det_max_side = detMaxSide;
        ocr = aicore_ocr_context_create(&oopt);
        if (!ocr) {
            std::fprintf(stderr, "bench_table: %s\n", aicore_ocr_last_error());
            return 1;
        }
    }
    std::fprintf(stderr, "[device] table=%s ocr=%s\n",
                 aicore_table_context_device(table),
                 ocr ? aicore_ocr_context_device(ocr) : "-");

    aicore_table_run_options ropt{};
    ropt.struct_size = sizeof(ropt);
    ropt.abi_version = AICORE_TABLE_ABI_VERSION;
    ropt.use_layout_regions = layoutRegions ? 1 : 0;

    std::vector<double> tableMs, ocrMs, totalMs;
    int failures = 0;
    for (size_t idx = 0; idx < images.size(); ++idx) {
        const std::string& path = images[idx];
        uint8_t* rgb = nullptr;
        int32_t w = 0, h = 0;
        if (aicore_map_anything_load_path_rgb(path.c_str(), &rgb, &w, &h) !=
                    0 ||
            !rgb) {
            std::fprintf(stderr, "bench_table: failed to decode %s\n",
                         path.c_str());
            ++failures;
            continue;
        }
        aicore_image_view view{};
        view.data = rgb;
        view.width = w;
        view.height = h;
        view.format = AICORE_IMAGE_RGB8;
        view.row_stride_bytes = static_cast<size_t>(w) * 3;
        const int passes = idx == 0 ? warmups + 1 : 1;
        for (int pass = 0; pass < passes; ++pass) {
            const bool timed = pass == passes - 1;
            const auto t0 = std::chrono::steady_clock::now();
            aicore_ocr_result* ocrResult =
                    ocr ? aicore_ocr_run(ocr, &view) : nullptr;
            if (ocr && !ocrResult) {
                std::fprintf(stderr, "bench_table: ocr failed on %s: %s\n",
                             path.c_str(), aicore_ocr_last_error());
                ++failures;
                break;
            }
            const auto t1 = std::chrono::steady_clock::now();
            aicore_table_result* result =
                    aicore_table_run(table, &view, ocr, ocrResult, &ropt);
            const auto t2 = std::chrono::steady_clock::now();
            if (!result) {
                std::fprintf(stderr, "bench_table: table failed on %s: %s\n",
                             path.c_str(), aicore_table_last_error());
                aicore_ocr_result_release(ocrResult);
                ++failures;
                break;
            }
            if (timed) {
                const double o =
                        std::chrono::duration<double, std::milli>(t1 - t0)
                                .count();
                const double t =
                        std::chrono::duration<double, std::milli>(t2 - t1)
                                .count();
                ocrMs.push_back(o);
                tableMs.push_back(t);
                totalMs.push_back(o + t);
                const aicore_pipeline_timings tm = aicore_table_timings(result);
                std::printf(
                        "{\"image\":%s,\"ocr_boxes\":%d,\"ocr_ms\":%.2f,"
                        "\"table_ms\":%.2f,\"table_inference_ms\":%.2f,"
                        "\"tables\":[",
                        jsonString(path.c_str()).c_str(),
                        aicore_ocr_box_count(ocrResult), o, t, tm.inference_ms);
                for (int k = 0; k < aicore_table_count(result); ++k) {
                    const aicore_table* tb = aicore_table_at(result, k);
                    std::printf(
                            "%s{\"box\":[%.1f,%.1f,%.1f,%.1f],\"rows\":%d,"
                            "\"cols\":%d,\"cells\":%d,\"tokens\":%d,"
                            "\"score\":%.4f,\"layout_region\":%d,"
                            "\"html\":%s,\"csv\":%s}",
                            k ? "," : "", tb->box[0], tb->box[1], tb->box[2],
                            tb->box[3], tb->rows, tb->cols, tb->cell_count,
                            tb->token_count, tb->structure_score,
                            tb->layout_region, jsonString(tb->html).c_str(),
                            jsonString(tb->csv).c_str());
                }
                std::printf("]}\n");
                std::fflush(stdout);
            }
            aicore_table_result_release(result);
            aicore_ocr_result_release(ocrResult);
        }
        aicore_map_anything_free_buffer(rgb);
    }
    std::printf(
            "{\"summary\":true,\"device\":%s,\"images\":%zu,"
            "\"failures\":%d,\"table_p50_ms\":%.2f,\"table_p95_ms\":%.2f,"
            "\"ocr_p50_ms\":%.2f,\"total_p50_ms\":%.2f,"
            "\"total_p95_ms\":%.2f}\n",
            jsonString(aicore_table_context_device(table)).c_str(),
            tableMs.size(), failures, percentile(tableMs, 0.5),
            percentile(tableMs, 0.95), percentile(ocrMs, 0.5),
            percentile(totalMs, 0.5), percentile(totalMs, 0.95));
    if (ocr) aicore_ocr_context_destroy(ocr);
    aicore_table_context_destroy(table);
    aicore_table_runtime_shutdown();
    return failures ? 1 : 0;
}
