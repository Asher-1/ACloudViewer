// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Per-stage CPU/Vulkan divergence locator for the SAM3 ViT image encoder.
//
// Usage:
//   cmp_sam3_vit_stages <sam3-visual-f16.gguf> [max_blocks=32] [threads=4]
//
// Loads the same GGUF twice (CPU backend and Vulkan backend), then replays
// the ViT encoder as isolated sub-graphs using the same test hooks the
// aicore_sam3_profile_encoder profiler uses:
//
//   prefix:  PATCH_EMBED -> POS_ADD -> LN_PRE_NORM -> LN_PRE
//   blocks:  NORM1 [-> WINDOW_PART] -> QKV_PROJ -> ATTN_CORE (RoPE + flash
//            attention) -> ATTN_PROJ [-> WINDOW_UNPART] -> +residual ->
//            NORM2 -> MLP -> +residual
//
// For every stage the tool feeds the *CPU reference* output of the previous
// stage into both backends and reports the isolated divergence (max abs
// error, RMSE, relative L2, cosine).  A second, independent per-backend chain
// also tracks the cumulative drift at each block boundary.
//
// Output is one JSON object per line; the last line is the summary.  Exit
// code is always 0 unless loading fails — this is a diagnostic locator.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "tasks/sam3/sam3.h"

namespace {

struct StageMetrics {
    double max_abs = 0.0;
    double rmse = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
    size_t n = 0;
    bool valid = false;
};

StageMetrics compare(const std::vector<float>& ref,
                     const std::vector<float>& test) {
    StageMetrics m;
    const size_t n = std::min(ref.size(), test.size());
    if (n == 0) return m;
    m.n = n;
    double sum_sq = 0.0, ref_sq = 0.0, dot = 0.0, test_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)test[i] - (double)ref[i];
        const double r = (double)ref[i];
        const double t = (double)test[i];
        m.max_abs = std::max(m.max_abs, std::fabs(d));
        sum_sq += d * d;
        ref_sq += r * r;
        test_sq += t * t;
        dot += r * t;
    }
    m.rmse = std::sqrt(sum_sq / (double)n);
    m.rel_l2 = ref_sq > 0.0 ? std::sqrt(sum_sq / ref_sq) : 0.0;
    m.cosine = (ref_sq > 0.0 && test_sq > 0.0)
                       ? dot / std::sqrt(ref_sq * test_sq)
                       : 0.0;
    m.valid = true;
    return m;
}

void print_metrics(const char* kind,
                   int block,
                   int stage,
                   const char* name,
                   const StageMetrics& m) {
    std::printf(
            "{\"kind\":\"%s\",\"block\":%d,\"stage\":%d,\"name\":\"%s\","
            "\"n\":%zu,\"max_abs\":%.9g,\"rmse\":%.9g,\"rel_l2\":%.9g,"
            "\"cosine\":%.9f}\n",
            kind, block, stage, name, m.n, m.max_abs, m.rmse, m.rel_l2,
            m.cosine);
    std::fflush(stdout);
}

// Deterministic pseudo-random input in [-2, 2] (fixed LCG).
void fill_deterministic(std::vector<float>& v) {
    uint64_t state = 0x2545F4914F6CDD1Dull;
    for (auto& x : v) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const double u = (double)(state >> 11) / (double)(1ull << 53);
        x = (float)(u * 4.0 - 2.0);
    }
}

void add_host(const std::vector<float>& a,
              const std::vector<float>& b,
              std::vector<float>* out) {
    const size_t n = std::min(a.size(), b.size());
    out->resize(n);
    for (size_t i = 0; i < n; ++i) (*out)[i] = a[i] + b[i];
}

struct Geometry {
    int patch = 0;
    int embed = 0;
    int grid = 0;
    int img_size = 0;
    int depth = 0;
};

bool infer_geometry(const sam3_model& model, Geometry* g) {
    sam3_tensor_info ti;
    if (!sam3_get_model_tensor_info(model, "vit.patch_embed.proj.weight", ti))
        return false;
    g->patch = (int)ti.ne[0];
    g->embed = (int)ti.ne[3];
    int64_t max_n = 0;
    for (int b = 0; b < 64; ++b) {
        const std::string name =
                "vit.blocks." + std::to_string(b) + ".attn.freqs_cis";
        if (!sam3_get_model_tensor_info(model, name, ti)) break;
        max_n = std::max(max_n, ti.ne[2]);
        g->depth = b + 1;
    }
    if (max_n > 0) {
        const int64_t s = (int64_t)std::sqrt((double)max_n);
        if (s * s == max_n) g->grid = (int)s;
    }
    if (g->grid <= 0 &&
        sam3_get_model_tensor_info(model, "vit.pos_embed", ti)) {
        g->grid = (int)ti.ne[1] * 3;
    }
    if (g->embed <= 0 || g->grid <= 0 || g->depth <= 0) return false;
    g->img_size = g->grid * g->patch;
    return true;
}

bool block_is_global(const sam3_model& model, int b, int grid) {
    sam3_tensor_info ti;
    const std::string name =
            "vit.blocks." + std::to_string(b) + ".attn.freqs_cis";
    return sam3_get_model_tensor_info(model, name, ti) &&
           ti.ne[2] == (int64_t)grid * grid;
}

bool run_prefix(const sam3_model& model,
                sam3_vit_prefix_stage stage,
                const std::vector<float>& in,
                const int64_t in_ne[4],
                std::vector<float>* out,
                int64_t out_ne[4],
                int n_threads) {
    return sam3_test_run_vit_prefix_stage(model, stage, in.data(), in_ne, *out,
                                          out_ne, n_threads);
}

bool run_block_stage(const sam3_model& model,
                     int block,
                     sam3_vit_block_stage stage,
                     const std::vector<float>& in,
                     const int64_t in_ne[4],
                     std::vector<float>* out,
                     int64_t out_ne[4],
                     int n_threads) {
    return sam3_test_run_vit_block_stage(model, block, stage, in.data(), in_ne,
                                         *out, out_ne, n_threads);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <sam3-visual-f16.gguf> [max_blocks=32] "
                     "[threads=4]\n",
                     argv[0]);
        return 2;
    }
    const int max_blocks = argc >= 3 ? std::max(1, std::atoi(argv[2])) : 32;
    const int n_threads = argc >= 4 ? std::max(1, std::atoi(argv[3])) : 4;

    sam3_params cpu_params;
    cpu_params.model_path = argv[1];
    cpu_params.n_threads = n_threads;
    cpu_params.use_gpu = false;
    cpu_params.device = SAM3_DEVICE_CPU;
    auto cpu_model = sam3_load_model(cpu_params);
    if (!cpu_model) {
        std::fprintf(stderr, "cpu load failed\n");
        return 1;
    }

    sam3_params vk_params = cpu_params;
    vk_params.use_gpu = true;
    vk_params.device = SAM3_DEVICE_VULKAN;
    auto vk_model = sam3_load_model(vk_params);
    if (!vk_model) {
        std::fprintf(stderr, "vulkan load failed\n");
        return 1;
    }
    std::printf("{\"backend_cpu\":\"%s\",\"backend_vulkan\":\"%s\"}\n",
                sam3_backend_name(*cpu_model), sam3_backend_name(*vk_model));

    Geometry geo;
    if (!infer_geometry(*cpu_model, &geo)) {
        std::fprintf(stderr, "geometry inference failed\n");
        return 1;
    }
    std::printf(
            "{\"patch\":%d,\"embed\":%d,\"grid\":%d,\"img\":%d,"
            "\"depth\":%d,\"max_blocks\":%d}\n",
            geo.patch, geo.embed, geo.grid, geo.img_size, geo.depth,
            max_blocks);

    // ── Prefix chain ─────────────────────────────────────────────────────
    std::vector<float> input((size_t)geo.img_size * geo.img_size * 3);
    fill_deterministic(input);
    int64_t in_ne[4] = {geo.img_size, geo.img_size, 3, 1};

    const struct {
        sam3_vit_prefix_stage stage;
        const char* name;
    } prefix_stages[] = {
            {SAM3_VIT_PREFIX_STAGE_PATCH_EMBED, "patch_embed"},
            {SAM3_VIT_PREFIX_STAGE_POS_ADD, "pos_add"},
            {SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM, "ln_pre_norm"},
            {SAM3_VIT_PREFIX_STAGE_LN_PRE, "ln_pre"},
    };

    std::vector<float> chain_cpu = input;
    std::vector<float> chain_vk = input;
    int64_t chain_cpu_ne[4], chain_vk_ne[4];
    std::copy(std::begin(in_ne), std::end(in_ne), chain_cpu_ne);
    std::copy(std::begin(in_ne), std::end(in_ne), chain_vk_ne);

    for (const auto& ps : prefix_stages) {
        std::vector<float> out_cpu, out_vk;
        int64_t ne_cpu[4] = {}, ne_vk[4] = {};
        const bool ok_cpu =
                run_prefix(*cpu_model, ps.stage, chain_cpu, chain_cpu_ne,
                           &out_cpu, ne_cpu, n_threads);
        const bool ok_vk = run_prefix(*vk_model, ps.stage, chain_cpu,
                                      chain_cpu_ne, &out_vk, ne_vk, n_threads);
        if (!ok_cpu) {
            std::printf("{\"stage\":\"%s\",\"error\":\"cpu stage failed\"}\n",
                        ps.name);
            return 1;
        }
        if (!ok_vk) {
            std::printf(
                    "{\"stage\":\"%s\",\"error\":\"vulkan stage failed\"}\n",
                    ps.name);
            return 1;
        }
        print_metrics("prefix", -1, (int)ps.stage, ps.name,
                      compare(out_cpu, out_vk));
        chain_cpu = std::move(out_cpu);
        std::copy(std::begin(ne_cpu), std::end(ne_cpu), chain_cpu_ne);
        // Cumulative chain on Vulkan fed by its own previous output.
        std::vector<float> vk_next;
        int64_t ne_vk2[4] = {};
        if (run_prefix(*vk_model, ps.stage, chain_vk, chain_vk_ne, &vk_next,
                       ne_vk2, n_threads)) {
            chain_vk = std::move(vk_next);
            std::copy(std::begin(ne_vk2), std::end(ne_vk2), chain_vk_ne);
            print_metrics("prefix_cumulative", -1, (int)ps.stage, ps.name,
                          compare(chain_cpu, chain_vk));
        }
    }

    // ── Block chain ──────────────────────────────────────────────────────
    std::vector<float> blk_cpu = chain_cpu;  // isolated-input chain (CPU ref)
    std::vector<float> blk_vk = chain_vk;    // cumulative per-backend chains
    int64_t blk_ne[4], vk_ne[4];
    std::copy(std::begin(chain_cpu_ne), std::end(chain_cpu_ne), blk_ne);
    std::copy(std::begin(chain_vk_ne), std::end(chain_vk_ne), vk_ne);

    const int depth = std::min(geo.depth, max_blocks);
    for (int b = 0; b < depth; ++b) {
        const bool is_global = block_is_global(*cpu_model, b, geo.grid);

        struct StageDef {
            sam3_vit_block_stage stage;
            const char* name;
            bool windowed_only;
        };
        const StageDef seq[] = {
                {SAM3_VIT_BLOCK_STAGE_NORM1, "norm1", false},
                {SAM3_VIT_BLOCK_STAGE_WINDOW_PART, "window_part", true},
                {SAM3_VIT_BLOCK_STAGE_QKV_PROJ, "qkv_proj", false},
                {SAM3_VIT_BLOCK_STAGE_ATTN_CORE, "attn_core", false},
                {SAM3_VIT_BLOCK_STAGE_ATTN_PROJ, "attn_proj", false},
                {SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART, "window_unpart", true},
        };

        // Isolated comparison: feed the CPU-reference running state into both
        // backends stage by stage.
        std::vector<float> x = blk_cpu;
        int64_t x_ne[4];
        std::copy(std::begin(blk_ne), std::end(blk_ne), x_ne);
        std::vector<float> shortcut = blk_cpu;

        for (const auto& sd : seq) {
            if (sd.windowed_only && is_global) continue;
            std::vector<float> out_cpu, out_vk;
            int64_t ne_c[4] = {}, ne_v[4] = {};
            const bool ok_cpu =
                    run_block_stage(*cpu_model, b, sd.stage, x, x_ne, &out_cpu,
                                    ne_c, n_threads);
            const bool ok_vk = run_block_stage(*vk_model, b, sd.stage, x, x_ne,
                                               &out_vk, ne_v, n_threads);
            if (!ok_cpu || !ok_vk) {
                std::printf(
                        "{\"block\":%d,\"stage\":\"%s\",\"error\":"
                        "\"stage failed cpu=%d vk=%d\"}\n",
                        b, sd.name, ok_cpu, ok_vk);
                break;
            }
            print_metrics("block", b, (int)sd.stage, sd.name,
                          compare(out_cpu, out_vk));
            x = std::move(out_cpu);
            std::copy(std::begin(ne_c), std::end(ne_c), x_ne);
        }

        // Cumulative chains (each backend fed by its own previous outputs);
        // residual additions mirror sam3_vit_block_forward on the host.
        auto advance_chain = [&](const sam3_model& model, std::vector<float>& c,
                                 int64_t c_ne[4]) -> bool {
            std::vector<float> cur = c;
            int64_t cur_ne[4];
            std::copy(c_ne, c_ne + 4, cur_ne);
            std::vector<float> resid = c;
            std::vector<float> tmp;
            int64_t tmp_ne[4] = {};
            auto run = [&](sam3_vit_block_stage st) -> bool {
                std::vector<float> out;
                int64_t one[4] = {};
                if (!run_block_stage(model, b, st, cur, cur_ne, &out, one,
                                     n_threads))
                    return false;
                cur = std::move(out);
                std::copy(std::begin(one), std::end(one), cur_ne);
                return true;
            };
            if (!run(SAM3_VIT_BLOCK_STAGE_NORM1)) return false;
            if (!is_global && !run(SAM3_VIT_BLOCK_STAGE_WINDOW_PART))
                return false;
            if (!run(SAM3_VIT_BLOCK_STAGE_QKV_PROJ)) return false;
            if (!run(SAM3_VIT_BLOCK_STAGE_ATTN_CORE)) return false;
            if (!run(SAM3_VIT_BLOCK_STAGE_ATTN_PROJ)) return false;
            if (!is_global && !run(SAM3_VIT_BLOCK_STAGE_WINDOW_UNPART))
                return false;
            add_host(resid, cur, &tmp);  // resid1
            resid = tmp;
            cur = resid;
            std::copy(c_ne, c_ne + 4, cur_ne);
            if (!run(SAM3_VIT_BLOCK_STAGE_NORM2)) return false;
            if (!run(SAM3_VIT_BLOCK_STAGE_MLP)) return false;
            add_host(resid, cur, &tmp);  // block output
            c = std::move(tmp);
            std::copy(c_ne, c_ne + 4, c_ne);  // shape unchanged per block
            return true;
        };

        const bool ok_cpu_chain = advance_chain(*cpu_model, blk_cpu, blk_ne);
        const bool ok_vk_chain = advance_chain(*vk_model, blk_vk, vk_ne);
        if (!ok_cpu_chain || !ok_vk_chain) {
            std::printf(
                    "{\"block\":%d,\"error\":\"cumulative chain failed "
                    "cpu=%d vk=%d\"}\n",
                    b, ok_cpu_chain, ok_vk_chain);
            break;
        }
        print_metrics("block_cumulative", b, -1, "block_out",
                      compare(blk_cpu, blk_vk));
        (void)shortcut;
    }

    std::printf("{\"done\":true}\n");
    sam3_free_model(*vk_model);
    sam3_free_model(*cpu_model);
    return 0;
}
