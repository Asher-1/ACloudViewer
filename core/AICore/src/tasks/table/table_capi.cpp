// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C API of the `table` task (core/AICore/include/aicore/table_capi.h):
//
//   image_view -> table regions (layout "table" class or whole image)
//     -> per region: int-truncated crop, long-side resize to max_len
//        (cv2 INTER_LINEAR, Python round), per-channel BGR normalize,
//        zero pad to pad_size, NCHW  (PaddleX TableStructureRecognition)
//     -> SLANet encoder + structure decoder (table_runtime)
//     -> tokens/cell polygons, OCR assignment, HTML, CSV (table_postprocess)
//
// The OCR result is borrowed through the public OCR accessors only; the
// caller runs OCR once per image and both tasks share it.

#include "aicore/table_capi.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/image_utils.hpp"
#include "table_postprocess.hpp"
#include "table_runtime.hpp"

namespace {

thread_local std::string g_lastError;

void setLastError(const std::string& message) { g_lastError = message; }

// PaddleX TableStructureRecognition Normalize (ImageNet statistics, BGR
// plane order as decoded by cv2).
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

double msBetween(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Python round() (half to even), as in ResizeByLong.
int roundHalfEven(double v) {
    const double f = std::floor(v);
    const double d = v - f;
    if (d > 0.5) return static_cast<int>(f) + 1;
    if (d < 0.5) return static_cast<int>(f);
    return std::fmod(f, 2.0) != 0.0 ? static_cast<int>(f) + 1
                                    : static_cast<int>(f);
}

bool iequals(const char* a, const char* b) {
    if (!a || !b) return false;
    for (; *a && *b; ++a, ++b)
        if (std::tolower(static_cast<unsigned char>(*a)) !=
            std::tolower(static_cast<unsigned char>(*b)))
            return false;
    return *a == *b;
}

struct Region {
    float box[4];        // float table box (offset of the cell coordinates)
    int x0, y0, x1, y1;  // int-truncated crop, clamped to the image
    int layoutIndex;
};

}  // namespace

struct aicore_table_ctx {
    aicore::table::TableStructureModel model;
    aicore::runtime::BackendLease primaryLease;
    aicore::runtime::BackendLease cpuLease;
    std::string resolvedDevice;
    // Per-crop scratch reused across runs (one owning thread).
    std::vector<uint8_t> rgbScratch;
    std::vector<float> nchwScratch;
    bool ready = false;
};

struct aicore_table_result {
    std::vector<aicore_table> tables;
    std::vector<aicore_table_cell> cells;
    std::vector<std::vector<std::string>> tokens;
    // Backing store of every handed-out const char*; a deque never moves
    // its elements on push_back.
    std::deque<std::string> strings;
    aicore_pipeline_timings timings{};

    const char* keep(std::string s) {
        strings.push_back(std::move(s));
        return strings.back().c_str();
    }
};

namespace {

// Long-side resize + normalize + pad of one crop into `nchw`
// (3 x pad x pad). Memory: pad^2 x 3 floats (2.9 MB at 488).
void packCrop(const aicore::img::Image& crop,
              int maxLen,
              int pad,
              std::vector<uint8_t>& rgb,
              std::vector<float>& nchw) {
    const double scale =
            static_cast<double>(maxLen) / std::max(crop.width, crop.height);
    const int rw = std::clamp(roundHalfEven(crop.width * scale), 1, pad);
    const int rh = std::clamp(roundHalfEven(crop.height * scale), 1, pad);
    rgb.resize(static_cast<size_t>(rw) * rh * 3);
    aicore::img::cv_resize_linear_u8(crop, rgb.data(), rw, rh, 3);
    const size_t plane = static_cast<size_t>(pad) * pad;
    nchw.assign(plane * 3, 0.0f);
    for (int c = 0; c < 3; ++c) {
        // Plane c is BGR channel c = RGB channel 2 - c; alpha/beta are
        // float32 like the reference's float32 array arithmetic.
        const float alpha = static_cast<float>((1.0 / 255.0) / kStd[c]);
        const float beta = static_cast<float>(-kMean[c] / kStd[c]);
        float* dst = nchw.data() + plane * static_cast<size_t>(c);
        const int src = 2 - c;
        for (int y = 0; y < rh; ++y) {
            const uint8_t* row = rgb.data() + static_cast<size_t>(y) * rw * 3;
            float* out = dst + static_cast<size_t>(y) * pad;
            for (int x = 0; x < rw; ++x) {
                const float v = static_cast<float>(row[x * 3 + src]) * alpha;
                out[x] = v + beta;
            }
        }
    }
}

}  // namespace

extern "C" {

aicore_table_ctx* aicore_table_context_create(
        const aicore_table_options* options) {
    if (!options || options->struct_size < sizeof(aicore_table_options)) {
        setLastError("table: invalid options (struct_size)");
        return nullptr;
    }
    if (options->abi_version != AICORE_TABLE_ABI_VERSION) {
        setLastError("table: abi version mismatch");
        return nullptr;
    }
    if (!options->model_path || !options->model_path[0]) {
        setLastError("table: model_path is required");
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_table_ctx();
    if (!ctx) {
        setLastError("table: context allocation failed");
        return nullptr;
    }
    std::string error;
    if (!ctx->model.load(options->model_path, &error)) {
        setLastError("table: model load failed: " + error);
        delete ctx;
        return nullptr;
    }
    ggml_common::load_backends_once();
    const std::string deviceRequest =
            (options->device && options->device[0]) ? options->device : "auto";
    // Same small-graph thread cap as the OCR task (the encoder is the OCR
    // interpreter; the decoder step graphs are tiny).
    int nThreads = options->n_threads;
    if (nThreads <= 0) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        nThreads = static_cast<int>(std::min(hw, 8u));
    }
    std::string leaseError;
    ctx->primaryLease = aicore::runtime::acquire_backend_lease(
            deviceRequest, nThreads, &leaseError);
    if (!ctx->primaryLease) {
        setLastError("table: requested device unavailable (" + deviceRequest +
                     "): " + leaseError);
        delete ctx;
        return nullptr;
    }
    ctx->resolvedDevice = ctx->primaryLease.device();
    if (!ctx->primaryLease.is_cpu()) {
        // Vulkan keeps the single-backend encoder route (the OCR task's
        // measured gate); CUDA keeps the scheduler offload.
        const bool isVulkan =
                std::string(ggml_backend_name(ctx->primaryLease.handle()))
                        .find("Vulkan") != std::string::npos;
        ctx->model.setSchedGpuTagging(!isVulkan);
        ctx->cpuLease = aicore::runtime::acquire_parallel_backend_lease(
                "cpu", nThreads, &leaseError);
        if (!ctx->cpuLease) {
            setLastError("table: CPU secondary backend unavailable: " +
                         leaseError);
            delete ctx;
            return nullptr;
        }
    }
    ctx->ready = true;
    return ctx;
}

void aicore_table_context_destroy(aicore_table_ctx* ctx) {
    if (!ctx) return;
    // Graphs and device buffers go before the leases that own the handles.
    ctx->model.release();
    ctx->cpuLease.reset();
    ctx->primaryLease.reset();
    delete ctx;
}

int aicore_table_context_ready(const aicore_table_ctx* ctx) {
    return ctx && ctx->ready ? 1 : 0;
}

const char* aicore_table_context_device(const aicore_table_ctx* ctx) {
    return ctx ? ctx->resolvedDevice.c_str() : "";
}

aicore_table_result* aicore_table_run(aicore_table_ctx* ctx,
                                      const aicore_image_view* image,
                                      const aicore_ocr_ctx* ocr_ctx,
                                      const aicore_ocr_result* ocr_result,
                                      const aicore_table_run_options* options) {
    if (!ctx || !ctx->ready || !image) {
        setLastError("table: run called with an invalid context or image");
        return nullptr;
    }
    if (options && (options->struct_size < offsetof(aicore_table_run_options,
                                                    fallback_region_mask) ||
                    options->abi_version != AICORE_TABLE_ABI_VERSION)) {
        setLastError("table: invalid run options (struct_size/abi_version)");
        return nullptr;
    }
    // Callers compiled against the pre-fallback header pass the old struct
    // size; the trailing field reads as 0 (upstream behavior).
    const uint32_t fallbackMask =
            options && options->struct_size >= sizeof(aicore_table_run_options)
                    ? options->fallback_region_mask
                    : 0u;
    aicore::img::Image src;
    if (!aicore::img::image_from_view(*image, src)) {
        setLastError("table: unsupported or empty image view");
        return nullptr;
    }
    const bool useLayout = options && options->use_layout_regions != 0;
    const float minScore = options ? options->min_region_score : 0.0f;
    if (useLayout && (!ocr_result || aicore_ocr_label_count(ocr_ctx) == 0)) {
        setLastError(
                "table: layout regions requested without an OCR result "
                "produced by a layout model");
        return nullptr;
    }
    const auto start = std::chrono::steady_clock::now();
    const float imgW = static_cast<float>(src.width);
    const float imgH = static_cast<float>(src.height);

    // --- table regions.
    std::vector<Region> regions;
    const auto addRegion = [&](float x0, float y0, float x1, float y1,
                               int layoutIndex) {
        Region r{{x0, y0, x1, y1}, 0, 0, 0, 0, layoutIndex};
        // CropByBoxes: int() truncation, numpy slice clamping.
        r.x0 = std::clamp(static_cast<int>(x0), 0, src.width);
        r.y0 = std::clamp(static_cast<int>(y0), 0, src.height);
        r.x1 = std::clamp(static_cast<int>(x1), 0, src.width);
        r.y1 = std::clamp(static_cast<int>(y1), 0, src.height);
        if (r.x1 > r.x0 && r.y1 > r.y0) regions.push_back(r);
    };
    if (useLayout) {
        const int n = aicore_ocr_region_count(ocr_result);
        for (int i = 0; i < n; ++i) {
            const aicore_ocr_region* reg = aicore_ocr_region_at(ocr_result, i);
            if (!reg ||
                !iequals(aicore_ocr_label_at(ocr_ctx, reg->class_id), "table"))
                continue;
            if (minScore > 0.0f && reg->score < minScore) continue;
            float x0 = reg->quad[0][0], y0 = reg->quad[0][1];
            float x1 = x0, y1 = y0;
            for (int k = 1; k < 4; ++k) {
                x0 = std::min(x0, reg->quad[k][0]);
                y0 = std::min(y0, reg->quad[k][1]);
                x1 = std::max(x1, reg->quad[k][0]);
                y1 = std::max(y1, reg->quad[k][1]);
            }
            addRegion(x0, y0, x1, y1, i);
        }
        if (regions.empty() && fallbackMask != 0) {
            // Class fallback: CDLA labels some synthetic-border tables
            // "figure"/"text"; accept the single best-scoring region whose
            // class is in the caller's mask, only when no "table" region
            // exists so pages with real tables keep the upstream behavior.
            const aicore_ocr_region* best = nullptr;
            int bestIndex = -1;
            for (int i = 0; i < n; ++i) {
                const aicore_ocr_region* reg =
                        aicore_ocr_region_at(ocr_result, i);
                if (!reg || reg->class_id > 31 ||
                    (fallbackMask & (1u << reg->class_id)) == 0)
                    continue;
                if (minScore > 0.0f && reg->score < minScore) continue;
                if (!best || reg->score > best->score) {
                    best = reg;
                    bestIndex = i;
                }
            }
            if (best) {
                float x0 = best->quad[0][0], y0 = best->quad[0][1];
                float x1 = x0, y1 = y0;
                for (int k = 1; k < 4; ++k) {
                    x0 = std::min(x0, best->quad[k][0]);
                    y0 = std::min(y0, best->quad[k][1]);
                    x1 = std::max(x1, best->quad[k][0]);
                    y1 = std::max(y1, best->quad[k][1]);
                }
                addRegion(x0, y0, x1, y1, bestIndex);
            }
        }
    } else {
        // Whole-image table: the reference box is [0, 0, w - 1, h - 1] and
        // the crop is the full image.
        Region r{{0.0f, 0.0f, imgW - 1.0f, imgH - 1.0f},
                 0,
                 0,
                 src.width,
                 src.height,
                 -1};
        regions.push_back(r);
    }

    // --- OCR boxes in the reference reading order (sorted before any
    // region filter, as the OCR pipeline sorts before the table stage).
    std::vector<std::array<float, 4>> ocrBoxes;
    std::vector<std::string> ocrTexts;
    if (ocr_result) {
        const int n = aicore_ocr_box_count(ocr_result);
        std::vector<std::array<float, 2>> topLeft;
        std::vector<const aicore_ocr_box*> raw;
        for (int i = 0; i < n; ++i) {
            const aicore_ocr_box* b = aicore_ocr_box_at(ocr_result, i);
            if (!b) continue;
            raw.push_back(b);
            topLeft.push_back({b->quad[0][0], b->quad[0][1]});
        }
        for (int idx : aicore::table::paddleReadingOrder(topLeft)) {
            const aicore_ocr_box* b = raw[static_cast<size_t>(idx)];
            std::array<float, 4> box{b->quad[0][0], b->quad[0][1],
                                     b->quad[0][0], b->quad[0][1]};
            for (int k = 1; k < 4; ++k) {
                box[0] = std::min(box[0], b->quad[k][0]);
                box[1] = std::min(box[1], b->quad[k][1]);
                box[2] = std::max(box[2], b->quad[k][0]);
                box[3] = std::max(box[3], b->quad[k][1]);
            }
            ocrBoxes.push_back(box);
            ocrTexts.emplace_back(b->text ? b->text : "");
        }
    }

    auto* result = new (std::nothrow) aicore_table_result();
    if (!result) {
        setLastError("table: result allocation failed");
        return nullptr;
    }
    ggml_backend_t gpu =
            ctx->primaryLease.is_cpu() ? nullptr : ctx->primaryLease.handle();
    ggml_backend_t cpu = ctx->primaryLease.is_cpu() ? ctx->primaryLease.handle()
                                                    : ctx->cpuLease.handle();
    double preprocessMs = 0.0, inferenceMs = 0.0, postprocessMs = 0.0;
    std::string error;
    for (const Region& region : regions) {
        // --- preprocess: borrowed crop view -> padded NCHW.
        const auto t0 = std::chrono::steady_clock::now();
        aicore::img::Image crop = src;
        crop.borrowed_data = src.data() +
                             static_cast<size_t>(region.y0) * src.stride() +
                             static_cast<size_t>(region.x0) * src.channels;
        crop.row_stride_bytes = src.stride();
        crop.width = region.x1 - region.x0;
        crop.height = region.y1 - region.y0;
        packCrop(crop, ctx->model.maxLen(), ctx->model.padSize(),
                 ctx->rgbScratch, ctx->nchwScratch);
        const auto t1 = std::chrono::steady_clock::now();

        // --- inference under the backend execution lock.
        aicore::table::TableStructureOutput out;
        bool ok;
        {
            std::vector<aicore::runtime::BackendLease> held{ctx->primaryLease};
            if (ctx->cpuLease) held.push_back(ctx->cpuLease);
            const auto lock = aicore::runtime::lock_backend_leases(held);
            ok = ctx->model.infer(gpu, cpu, ctx->nchwScratch.data(), out,
                                  error);
        }
        if (!ok) {
            delete result;
            setLastError("table: inference failed: " + error);
            return nullptr;
        }
        const auto t2 = std::chrono::steady_clock::now();

        // --- postprocess.
        const aicore::table::DecodedTable decoded =
                aicore::table::decodeStructure(
                        out, ctx->model.vocabulary(), ctx->model.locReg(),
                        ctx->model.padSize(), crop.width, crop.height);
        const aicore::table::TableGrid grid =
                aicore::table::buildGrid(decoded.tokens);
        std::vector<std::array<float, 4>> cellBoxes;
        cellBoxes.reserve(decoded.cells.size());
        for (const auto& p : decoded.cells) {
            std::array<float, 4> b{p[0], p[1], p[0], p[1]};
            for (int k = 1; k < 4; ++k) {
                b[0] = std::min(b[0], p[static_cast<size_t>(2 * k)]);
                b[1] = std::min(b[1], p[static_cast<size_t>(2 * k + 1)]);
                b[2] = std::max(b[2], p[static_cast<size_t>(2 * k)]);
                b[3] = std::max(b[3], p[static_cast<size_t>(2 * k + 1)]);
            }
            b[0] = std::clamp(b[0] + region.box[0], 0.0f, imgW);
            b[1] = std::clamp(b[1] + region.box[1], 0.0f, imgH);
            b[2] = std::clamp(b[2] + region.box[0], 0.0f, imgW);
            b[3] = std::clamp(b[3] + region.box[1], 0.0f, imgH);
            cellBoxes.push_back(b);
        }
        // get_sub_regions_ocr_res: boxes overlapping the table box by more
        // than 3 px on both axes.
        std::vector<std::array<float, 4>> tableOcrBoxes;
        std::vector<std::string> tableOcrTexts;
        for (size_t i = 0; i < ocrBoxes.size(); ++i) {
            const auto& b = ocrBoxes[i];
            const float pubW = std::min(region.box[2], b[2]) -
                               std::max(region.box[0], b[0]);
            const float pubH = std::min(region.box[3], b[3]) -
                               std::max(region.box[1], b[1]);
            if (pubW > 3.0f && pubH > 3.0f) {
                tableOcrBoxes.push_back(b);
                tableOcrTexts.push_back(ocrTexts[i]);
            }
        }
        const auto matched =
                aicore::table::matchOcrToCells(cellBoxes, tableOcrBoxes);
        std::vector<std::string> cellTexts;
        const std::string html = aicore::table::buildHtml(
                decoded.tokens, matched, tableOcrTexts, &cellTexts);
        const std::string csv = aicore::table::buildCsv(grid, cellTexts);

        aicore_table table{};
        std::memcpy(table.box, region.box, sizeof(table.box));
        table.structure_score = decoded.score;
        table.layout_region = region.layoutIndex;
        table.rows = grid.rows;
        table.cols = grid.cols;
        table.cell_offset = static_cast<int32_t>(result->cells.size());
        table.cell_count = static_cast<int32_t>(decoded.cells.size());
        table.token_count = static_cast<int32_t>(decoded.tokens.size());
        table.html = result->keep(html);
        table.csv = result->keep(csv);
        for (size_t i = 0; i < decoded.cells.size(); ++i) {
            aicore_table_cell cell{};
            const auto& p = decoded.cells[i];
            for (int k = 0; k < 4; ++k) {
                cell.quad[k][0] = p[static_cast<size_t>(2 * k)] + region.box[0];
                cell.quad[k][1] =
                        p[static_cast<size_t>(2 * k + 1)] + region.box[1];
            }
            std::memcpy(cell.box, cellBoxes[i].data(), sizeof(cell.box));
            const aicore::table::GridCell& g = grid.cells[i];
            cell.row = g.row;
            cell.col = g.col;
            cell.row_span = g.rowSpan;
            cell.col_span = g.colSpan;
            cell.text = result->keep(i < cellTexts.size() ? cellTexts[i] : "");
            cell.ocr_box_count = static_cast<int32_t>(matched[i].size());
            result->cells.push_back(cell);
        }
        result->tables.push_back(table);
        result->tokens.push_back(decoded.tokens);
        const auto t3 = std::chrono::steady_clock::now();
        preprocessMs += msBetween(t0, t1);
        inferenceMs += msBetween(t1, t2);
        postprocessMs += msBetween(t2, t3);
    }
    result->timings.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    result->timings.valid_fields =
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
            AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    result->timings.preprocess_ms = preprocessMs;
    result->timings.inference_ms = inferenceMs;
    result->timings.postprocess_ms = postprocessMs;
    result->timings.e2e_ms = msBetween(start, std::chrono::steady_clock::now());
    return result;
}

int aicore_table_count(const aicore_table_result* result) {
    return result ? static_cast<int>(result->tables.size()) : 0;
}

const aicore_table* aicore_table_at(const aicore_table_result* result,
                                    int index) {
    if (!result || index < 0 ||
        index >= static_cast<int>(result->tables.size()))
        return nullptr;
    return &result->tables[static_cast<size_t>(index)];
}

int aicore_table_cell_count(const aicore_table_result* result) {
    return result ? static_cast<int>(result->cells.size()) : 0;
}

const aicore_table_cell* aicore_table_cell_at(const aicore_table_result* result,
                                              int index) {
    if (!result || index < 0 || index >= static_cast<int>(result->cells.size()))
        return nullptr;
    return &result->cells[static_cast<size_t>(index)];
}

const char* aicore_table_token_at(const aicore_table_result* result,
                                  int table_index,
                                  int index) {
    if (!result || table_index < 0 ||
        table_index >= static_cast<int>(result->tokens.size()))
        return nullptr;
    const auto& tokens = result->tokens[static_cast<size_t>(table_index)];
    if (index < 0 || index >= static_cast<int>(tokens.size())) return nullptr;
    return tokens[static_cast<size_t>(index)].c_str();
}

void aicore_table_result_release(aicore_table_result* result) { delete result; }

aicore_pipeline_timings aicore_table_timings(
        const aicore_table_result* result) {
    aicore_pipeline_timings timings{};
    if (result) timings = result->timings;
    return timings;
}

const char* aicore_table_last_error(void) { return g_lastError.c_str(); }

void aicore_table_runtime_shutdown(void) { aicore_runtime_shutdown(); }

}  // extern "C"
